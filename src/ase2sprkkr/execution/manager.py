"""Orchestration of durable calculation run lifecycles."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from .base import FailurePhase, RunState, RunStatus, SubmissionRecovery
from .endpoint import ExecutorEndpoint
from .job_files import JobFiles
from .pool import Executors
from .postprocessing import CalculationRun
from .run_model import Execution, WorkingDirectory, FileManifest
from .run_record import RunRecord, SubmissionPhase, _UNCHANGED, _failure_phase, _now
from .run_persistence import RunPersistence


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class DuplicateSubmissionError(RuntimeError):
    """A project run would duplicate an active calculation namespace."""

    def __init__(self, record, conflicts):
        self.record = record
        self.conflicts = tuple(conflicts)
        super().__init__(
            "Run conflicts with {} active project run{}".format(
                len(self.conflicts), "" if len(self.conflicts) == 1 else "s"
            )
        )


class ExecutionManager:
    """Persist public run lifecycle without task-specific knowledge."""

    def __init__(self, store=None, *, executors=None, submitting_grace=30.0,
                 project_repository=None):
        if submitting_grace < 0:
            raise ValueError("submitting_grace must be non-negative")
        self.projects = project_repository
        if self.projects is not None:
            canonical_store = self.projects.run_store
            if store is not None and store is not canonical_store:
                raise ValueError("Project-bound execution requires the repository run store")
            store = canonical_store
        self.store = store
        self.executors = executors if executors is not None else Executors()
        self.submitting_grace = float(submitting_grace)

    def new_record(
        self, *, task, directory, dataset=None, endpoint=None,
        remote_directory=None, files=None, input_definition=None, run_id=None,
        project_id=None,
    ):
        if self.projects is not None:
            if project_id is None:
                raise ValueError("Tracked runs require an explicit project_id")
            from .projects import Project
            identity = project_id.id if isinstance(project_id, Project) else str(project_id)
            project_id = self.projects.project(identity).id
        now = _now()
        local = os.path.abspath(os.fspath(directory))
        return RunRecord(
            id=run_id or uuid4().hex, state=RunState.NEW, created_at=now,
            updated_at=now, task=str(task).upper(), dataset=dataset,
            execution=Execution(endpoint or ExecutorEndpoint(), WorkingDirectory(local, remote_directory)),
            files=files if files is not None else FileManifest(),
            input_definition=input_definition,
            project_id=project_id,
        )

    def transition(
        self,
        record,
        *,
        state=None,
        execution=_UNCHANGED,
        return_code=_UNCHANGED,
        message=_UNCHANGED,
        failure_phase=_UNCHANGED,
        files=_UNCHANGED,
        expected_updated_at=_UNCHANGED,
        expected_record=None,
        skip_if_terminal=False,
        validate_completed_files=False,
    ):
        """Atomically commit the value object's validated state transition."""
        return self.store.update(
            record,
            lambda latest: latest.with_transition(
                state=state, execution=execution, return_code=return_code,
                message=message, failure_phase=failure_phase, files=files,
                expected_updated_at=expected_updated_at,
                expected_record=expected_record,
                skip_if_terminal=skip_if_terminal,
                validate_completed_files=validate_completed_files,
            ),
        )

    def _is_stale_submitting(self, record, *, now=None):
        if record.state != RunState.SUBMITTING or record.execution.backend_id is not None:
            return False
        if record.submission_phase == SubmissionPhase.PREPARING:
            return True  # Lock ownership, not age, decides whether it is abandoned.
        if record.submission_phase != SubmissionPhase.DISPATCHING:
            return False  # Unclassified historical/foreign state: fail closed.
        now = now or datetime.now(timezone.utc)
        return (now - _timestamp(record.updated_at)).total_seconds() >= self.submitting_grace

    def recover_submitting(self, record):
        # Acquire before reading: the original submitter holds this lock from
        # reserve through durable submission, and cannot launch after we claim it.
        with self.store.submission_owner(record) as acquired:
            record = self.store.get(record.id)
            if not acquired or not self._is_stale_submitting(record):
                return record

            def mark_absent(latest):
                if (latest != record or latest.state != RunState.SUBMITTING
                        or latest.execution.backend_id is not None):
                    return latest
                return replace(
                    latest, state=RunState.FAILED, submission_phase=None,
                    updated_at=_now(), finished_at=_now(),
                    message="Abandoned submission was proven not to have launched",
                    failure_phase=FailurePhase.SUBMISSION,
                )

            executor = self.executors.for_endpoint(record.execution.endpoint)
            if record.submission_phase == SubmissionPhase.PREPARING:
                # Only executors with no independently surviving staging can
                # reclaim this reservation without risking concurrent file IO.
                if not executor.abandoned_preparation_is_safe():
                    return record
                return self.store.update(record, mark_absent)

            try:
                status = executor.recover_submission(
                    record.id, record, submitted_at=record.updated_at,
                )
            except (NotImplementedError, OSError, subprocess.CalledProcessError):
                # Unavailable backend is not an absent backend.
                return record
            if status is SubmissionRecovery.ABSENT:
                return self.store.update(record, mark_absent)
            if status is None:
                # Neither an absent ticket nor empty squeue/sacct proves no job.
                return record
            if not isinstance(status, RunStatus):
                raise TypeError("Invalid backend submission recovery result")
            backend_id = status.backend_id or (
                None if status.execution is None else status.execution.backend_id
            )
            if backend_id is None:
                raise ValueError("Recovered submission lacks backend identity")
            return self.update(record, status, expected_updated_at=record.updated_at)

    def reserve(self, record):
        """Reserve files and recover stale submissions before retrying."""
        attempted_recovery = set()
        while True:
            reserved, conflicts = self.store.reserve_files(record)
            if reserved is not None:
                return reserved
            stale = tuple(
                conflict for conflict in conflicts
                if conflict.id not in attempted_recovery
                and self._is_stale_submitting(conflict)
            )
            if not stale:
                raise DuplicateSubmissionError(record, conflicts)
            progressed = False
            for conflict in stale:
                attempted_recovery.add(conflict.id)
                recovered = self.recover_submitting(conflict)
                if (recovered.state != RunState.SUBMITTING
                        or recovered.execution.backend_id is not None):
                    progressed = True
            if not progressed:
                raise DuplicateSubmissionError(record, conflicts)

    def update(self, record, status: RunStatus, *, expected_updated_at=_UNCHANGED):
        phase = _failure_phase(status.failure_phase)
        return_code = status.return_code
        state = RunState.POSTPROCESSING if status.state == RunState.FINISHED else status.state
        execution = status.execution
        if execution is None and status.backend_id is not None:
            execution = record.execution.submitted(status.backend_id)
        updated = self.transition(
            record,
            state=state,
            execution=execution,
            files=status.files if status.files is not None else _UNCHANGED,
            return_code=return_code,
            message=status.message,
            failure_phase=phase,
            expected_updated_at=expected_updated_at,
        )
        if updated.execution.backend_id is not None:
            self._acknowledge_submission(
                self.executors.for_endpoint(updated.execution.endpoint), updated,
            )
        return updated

    def complete(
        self,
        record,
        *,
        state,
        execution=None,
        return_code=None,
        message=None,
        failure_phase=None,
        files_available=False,
        files=None,
        expected_record=None,
    ):
        # Another observer (or our own interrupted durable write) may have
        # committed completion already. In that case no file re-capture is
        # needed and the existing terminal snapshot is authoritative.
        confirmed = self.store.terminal_record(record)
        if confirmed is not None:
            if expected_record is not None and confirmed != expected_record:
                raise RuntimeError("Run changed during postprocessing retry")
            return confirmed
        state = RunState(state)
        phase = _failure_phase(failure_phase)
        result_files = record.files if files is None else files

        if result_files.failed_files:
            files_available = False
            # Whether this also fails an otherwise successful run is decided
            # atomically against the latest stored file states in transition().

        if files_available:
            try:
                result_files = JobFiles.capture(result_files)
            except OSError as exc:
                state = RunState.FAILED
                phase = FailurePhase.FILES
                message = str(exc)
                # Capture never verified the remaining pending files.
                # Preserve their recorded availability rather than invent failures.

        return self.transition(
            record,
            state=state,
            execution=execution,
            return_code=return_code,
            message=message,
            failure_phase=phase,
            files=result_files,
            expected_record=expected_record,
            skip_if_terminal=True,
            validate_completed_files=True,
        )

    def resume_download(self, run):
        """Explicitly reopen a failed post-execution transfer for retry.

        This is not an execution resubmit. The original backend handle remains
        authoritative and the attached raw runner replays/synchronizes files
        before task postprocessing can finish.
        """
        record = self.store.get(run if isinstance(run, str) else run.id)

        def reopen(latest):
            if (latest.state != RunState.FAILED
                    or latest.failure_phase != FailurePhase.DOWNLOAD
                    or latest.return_code != 0
                    or latest.execution.backend_id is None
                    or latest.files.captured_at is not None):
                raise ValueError("Only failed downloads of successful backend jobs can be retried")
            # Keep already transferred files and all bytes on disk.
            # Only failed transfers become eligible for a fresh attempt.
            files = latest.files.retry_failed_downloads()
            return replace(
                latest, state=RunState.DOWNLOADING, finished_at=None,
                failure_phase=None, message=None, files=files, updated_at=_now(),
            )

        return self.store.update(record, reopen)

    def retry_postprocessing(self, run, process_result):
        """Retry historical parsing without resubmitting or attaching a backend."""
        from .postprocessing import claim_historical_postprocessing, parse_historical_result

        record = self.store.get(run if isinstance(run, str) else run.id)
        reopened = self.store.update(record, claim_historical_postprocessing)
        try:
            result = parse_historical_result(reopened, process_result)
        except BaseException as exc:
            self.complete(
                reopened, state=RunState.FAILED,
                return_code=reopened.return_code,
                message=str(exc), failure_phase=FailurePhase.POSTPROCESSING,
                files_available=False, expected_record=reopened,
            )
            raise
        self.complete(
            reopened, state=RunState.FINISHED,
            return_code=reopened.return_code, files=reopened.files,
            files_available=False, expected_record=reopened,
        )
        return result

    def retrieve_result_files(self, run):
        record = self.store.get(run if isinstance(run, str) else run.id)
        if not record.state.is_terminal:
            raise RuntimeError(f"Run {record.id} is not finished")
        missing = JobFiles.missing_files(record.files)
        executor = (
            self.executors.for_endpoint(record.execution.endpoint)
            if missing and record.execution.backend_id is not None else None
        )
        return JobFiles.retrieve(record, self.store, executor)

    @staticmethod
    def _acknowledge_submission(executor, record):
        """Discard the temporary backend PID only after RunStore is durable."""
        if record.execution.backend_id is None:
            return
        try:
            executor.acknowledge_submission(record)
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            # The receipt remains recoverable until a subsequent refresh.
            pass

    def create_run(
        self,
        spec,
        *,
        endpoint,
        process_result,
        task,
        input_definition=None,
        dataset=None,
        callback=None,
        run_id=None,
        project_id=None,
        resources=None,
    ):
        """Create a project run from raw execution plus generic processing."""
        executor = self.executors.for_endpoint(endpoint)
        record = self.new_record(
            task=task,
            input_definition=input_definition,
            dataset=dataset,
            directory=spec.directory,
            endpoint=endpoint,
            files=FileManifest.from_spec(spec),
            run_id=run_id,
            project_id=project_id,
        )
        raw = executor.create_project_run(spec, run_id=record.id)
        # The runner owns mutable working files; RunRecord holds frozen snapshots.
        raw.job_files.replace(record.files)
        persistence = RunPersistence(self, executor, record, spec=spec)
        return CalculationRun(
            raw, process_result, callback=callback,
            resources=resources, persistence=persistence,
        )

    def refresh(self, records=None):
        records = tuple(self.store.runs(active=True) if records is None else records)
        current = {record.id: record for record in records}
        groups = {}
        for record in records:
            if record.state.is_terminal:
                continue
            if record.execution.backend_id is None:
                if record.state == RunState.SUBMITTING:
                    current[record.id] = self.recover_submitting(record)
                continue
            groups.setdefault(record.execution.endpoint, []).append(record)

        for endpoint, group in groups.items():
            statuses = self.executors.for_endpoint(endpoint).status(*group)
            if len(statuses) != len(group):
                raise RuntimeError("Executor returned wrong number of run statuses")
            for record, status in zip(group, statuses):
                current[record.id] = self.update(
                    record, status, expected_updated_at=record.updated_at,
                )
        return tuple(current[record.id] for record in records)

    def attach(
        self,
        run,
        *,
        process_result,
        callback=None,
        output_listener=None,
    ):
        """Attach raw execution, then resume client-side postprocessing."""
        record = self.store.get(run) if isinstance(run, str) else run
        if record.execution.backend_id is None:
            raise RuntimeError(f"Run {record.id} has no backend reference")
        executor = self.executors.for_endpoint(record.execution.endpoint)

        raw = executor.attach(
            record,
            output_listener=output_listener,
            start=False,
        )
        persistence = RunPersistence(self, executor, record)
        processed = CalculationRun(
            raw, process_result, callback=callback, persistence=persistence,
        )
        processed.start(attached=True)
        return processed
