"""Rehydrate SPR-KKR results from project run records."""

from __future__ import annotations

import os
import logging

import pyparsing as pp

from ..common.process_output_reader import ProcessExitError
from ..common.warnings import DataValidityError
from ..common.module_registry import RegistryEntryNotFound
from kkrunner.files import ArtifactAvailabilityState, FileState
from kkrunner.runs import FailurePhase, HistoricalFileMismatchError
from kkrunner import RunFileState
from ..input_parameters.input_parameters import InputParameters
from ..input_parameters.task_definition import TaskMismatchError
from .reader_registry import reader_registry
from .output_only_result import OutputOnlyResult


logger = logging.getLogger(__name__)
# Only recognized native output errors are recoverable; arbitrary ValueError,
# TypeError, KeyError and AttributeError still indicate a possible defect.
_UNPARSABLE_OUTPUT_ERRORS = (
    EOFError, UnicodeError, pp.ParseBaseException, DataValidityError,
    ProcessExitError,
)


class ProjectResultUnavailableError(RuntimeError):
    """A stored run cannot be reconstructed as its historical TaskResult."""

    def __init__(self, record, message):
        self.record = record
        super().__init__(f"Run {record.id}: {message}")


class UnrecoverableProjectResultError(ProjectResultUnavailableError):
    """The persisted metadata is insufficient to reconstruct this result."""


def _record(manager, run):
    return manager.store.get(run) if isinstance(run, str) else manager.store.get(run.id)


def application_metadata(record):
    """Validate and read the immutable SPR-KKR namespace of a generic record."""
    metadata = record.metadata.get("ase2sprkkr")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("task"), str):
        raise UnrecoverableProjectResultError(record, "missing SPR-KKR task metadata")
    return metadata


def _task_definition_for_record(record):
    try:
        return InputParameters.task_registry[application_metadata(record)["task"]]
    except RegistryEntryNotFound:
        raise UnrecoverableProjectResultError(
            record, f"unknown persisted task {application_metadata(record)['task']!r}"
        ) from None


def _input_parameters_for_record(manager, record, task_definition):
    result_files = record.files
    input_file = None if result_files is None else result_files.input_file
    current_input = (manager.input_file_is_current(record) if manager is not None
                     else any(str(v.path) == input_file and v.is_current()
                              for v in result_files.input_versions))
    if input_file is None or not current_input:
        return None
    if application_metadata(record).get("input_definition") is None:
        return None

    input_registry = InputParameters.input_parameters_registry
    try:
        definition = input_registry[application_metadata(record).get("input_definition")]
    except RegistryEntryNotFound:
        raise UnrecoverableProjectResultError(
            record,
            f"unknown persisted input definition {application_metadata(record).get('input_definition')!r}",
        ) from None

    try:
        return input_registry.read_for_task(
            task_definition,
            input_file,
            input_definition=definition,
        )
    except TaskMismatchError as exc:
        raise UnrecoverableProjectResultError(
            record,
            "persisted task does not match historical input "
            f"({application_metadata(record)['task']!r} != {exc.task_name!r})",
        ) from exc


def _reader_for_record(manager, record):
    """Create the task-owned output reader declared by the persisted task."""
    task_definition = _task_definition_for_record(record)
    parameters = _input_parameters_for_record(manager, record, task_definition)
    reader_class = reader_registry.class_for_task(task_definition)
    return reader_class(
        parameters,
        None,
        record.execution.directory.local,
        task_name=application_metadata(record)["task"],
    )


def _result_processor_for_record(manager, record):
    """Build the generic execution→TaskResult boundary in the outputs layer."""
    reader = _reader_for_record(manager, record)

    def process_result(outcome):
        return reader.read_from_file(
            outcome.output_file,
            outcome.error_file,
            return_code=outcome.return_code,
            input_file=outcome.input_file,
            directory=record.execution.directory.local,
            run_record=_record(manager, record),
        )

    return process_result


def attach_project_run(
    manager,
    run,
    *,
    executor=None,
    callback=None,
    output_listener=None,
):
    """Reconnect an active execution whose output is not yet confirmed published."""
    record = _record(manager, run)
    if record.file_state == RunFileState.COLLECTING or (
        record.file_state == RunFileState.FAILED
        and record.file_failure_phase == FailurePhase.DOWNLOAD
    ):
        raise ProjectResultUnavailableError(
            record, "use finish_project_files() to complete file collection"
        )
    if not record.is_active:
        raise ProjectResultUnavailableError(
            record, f"run files are already finalized ({record.file_state.value})"
        )

    if record.file_state in (RunFileState.READY, RunFileState.POSTPROCESSING):
        raise ProjectResultUnavailableError(
            record, "files are ready for local processing; use retry_project_postprocessing()"
        )

    # An attach attempt does not own a running processor's lock, so reader
    # reconstruction failures must not finalize its processing state.
    process_result = _result_processor_for_record(manager, record)

    def completed(result):
        from .task_result import TaskResult
        if isinstance(result, TaskResult):
            result.run_record = manager.store.get(record.id)
        if callback is not None:
            callback(result)

    return manager.attach(
        record,
        process_result=process_result,
        client=executor,
        callback=completed,
        output_listener=output_listener,
    )


def finish_project_files(manager, run):
    """Collect published files and interpret them without reattaching a job."""
    record = _record(manager, run)
    if record.file_state == RunFileState.POSTPROCESSED:
        return load_project_result(manager, record, allow_unparsed=True)
    # Resolve the reader before reopening a failed reservation.
    processor = _result_processor_for_record(manager, record)
    record = manager.collect_files(record)
    if record.file_state == RunFileState.POSTPROCESSED:
        return load_project_result(manager, record, allow_unparsed=True)
    if record.file_state != RunFileState.READY:
        raise ProjectResultUnavailableError(
            record, record.file_message or f"file collection ended in {record.file_state.value}"
        )
    return _process_project_record(manager, record, processor)


def retry_project_postprocessing(manager, run):
    """Process prepared or preserved result files locally, without executing the task."""
    record = _record(manager, run)
    # Reader construction is outside the transition: programming errors and
    # missing definitions must leave the original failure record unchanged.
    return _process_project_record(manager, record, _result_processor_for_record(manager, record))


def _process_project_record(manager, record, processor):
    result = manager.retry_postprocessing(record, processor)
    # TaskResult provenance belongs to the outputs layer, not the generic project manager.
    # Generic processors may return immutable or scalar values.  Only task
    # results carry the optional project provenance attribute.
    from .task_result import TaskResult
    if isinstance(result, TaskResult):
        result.run_record = manager.store.get(record.id)
        from .result_loader import ResultLoader
        result = ResultLoader().associate_result(result)
    return result


def load_project_result(manager, run, *, allow_unparsed=False, restore_missing=False):
    """Reconstruct a historical TaskResult from a terminal project run.

    With allow_unparsed=True, a verified OUT remains inspectable when a
    recognized native-output error prevents parsing. Unverified or replaced
    historical files and unrelated programming errors never fall back.
    Loading is local-only unless the caller explicitly requests restoration
    of missing captured files.
    """
    record = _record(manager, run)
    if record.is_active:
        raise ProjectResultUnavailableError(
            record, f"run files are still active ({record.file_state.value})"
        )

    if restore_missing:
        # Restore every available file independently. Missing auxiliary files do
        # not invalidate a historical result whose primary output is intact.
        try:
            record = manager.retrieve_result_files(record)
        except HistoricalFileMismatchError as exc:
            # Older KKRunner versions raise the digest mismatch directly, rather
            # than returning a per-file failure. Preserve the same public error
            # and store the primary result's unavailability in either case.
            output_file = None if record.files is None else record.files.output_file
            if output_file is not None:
                record = manager.update_artifact_availability(
                    record, output_file, ArtifactAvailabilityState.FAILED, str(exc),
                )
            raise ProjectResultUnavailableError(record, str(exc)) from exc
    result_files = record.files
    if result_files is not None and result_files.captured_at is None:
        raise ProjectResultUnavailableError(
            record, "historical output is unavailable: no authoritative file snapshot"
        )
    if result_files is None or result_files.output_file is None:
        raise ProjectResultUnavailableError(
            record, "no persisted output result file"
        )

    output_file = result_files.output_file
    output_available = manager.output_file_is_current(record)
    if not output_available:
        # Distinguish a changed historical file from a missing file: both must
        # prevent parsing, but the user needs the actual reason for recovery.
        output_error = (
            result_files.stdout.availability_message
            if result_files.stdout is not None else None
        )
        version = next((
            item for item in result_files.output_versions
            if os.path.abspath(os.fspath(item.path)) == os.path.abspath(os.fspath(output_file))
        ), None)
        if version is not None and version.state() == FileState.OVERWRITTEN:
            output_error = f"historical output was overwritten: {output_file}"
        raise ProjectResultUnavailableError(
            record, output_error or "output result file is unavailable"
        )

    reader = _reader_for_record(manager, record)
    # The record's terminal state and its OS exit status are independent.
    return_code = record.return_code

    input_file = None
    if reader.input_parameters is not None:
        if not manager.input_file_is_current(record):
            raise ProjectResultUnavailableError(record, "historical input changed during parsing")
        input_file = result_files.input_file

    try:
        result = reader.read_from_file(
            output_file,
            return_code=return_code,
            input_file=input_file,
            directory=record.execution.directory.local,
            run_record=record,
        )
    except _UNPARSABLE_OUTPUT_ERRORS as exc:
        if not allow_unparsed:
            raise
        # A file may be replaced while the parser is running.
        if not os.path.isfile(output_file) or not manager.output_file_is_current(record):
            raise ProjectResultUnavailableError(
                record, "historical output disappeared or changed during parsing"
            ) from exc
        logger.warning(
            "Cannot parse historical run %s; exposing verified OUT only",
            record.id, exc_info=True,
        )
        result = OutputOnlyResult(record, exc)
    result.run_record = record
    from .result_loader import ResultLoader
    result = ResultLoader().associate_result(result)
    if not manager.output_file_is_current(record):
        raise ProjectResultUnavailableError(record, "historical output disappeared or changed during parsing")
    if input_file is not None and not manager.input_file_is_current(record):
        raise ProjectResultUnavailableError(record, "historical input changed during parsing")
    return result
