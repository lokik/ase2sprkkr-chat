"""Project run history and lifecycle orchestration."""

from __future__ import annotations

import asyncio
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Optional
from uuid import uuid4


from ..common.file_locks import FileLock
from ..common.file_versions import FileState, file_digest
from ._durability import durable_mkdir, durable_write_json, sync_file
from .base import (
    Executor,
    FailurePhase,
    RunState,
    RunStatus,
    SubmissionNotStarted,
)
from .file_history import (
    capture_result_files, file_is_current, missing_result_files, sync_result_files,
)
from .endpoint import ExecutorEndpoint, endpoint_for_executor
from .pool import Executors
from .file_conflicts import submission_conflicts
from .projects import _project_id
from .postprocessing import CalculationRun
from .run_model import (
    FileAvailabilityState, Execution, WorkingDirectory, FileManifest,
)


class HistoricalFileMismatchError(OSError):
    """Restored contents differ from a run's captured SHA-256 digest."""


_RUN_RECORD_SCHEMA_VERSION = 3
_STATE_RANK = {
    RunState.NEW: 0,
    RunState.SUBMITTING: 1,
    RunState.QUEUED: 2,
    RunState.RUNNING: 3,
    RunState.DOWNLOADING: 4,
    RunState.POSTPROCESSING: 5,
    RunState.FINISHED: 6,
    RunState.FAILED: 6,
    RunState.CANCELLED: 6,
}
_UNCHANGED = object()
_NO_FILES_FAILURES = {
    FailurePhase.SUBMISSION,
    FailurePhase.LAUNCH,
    FailurePhase.FILES,
    FailurePhase.DOWNLOAD,
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _failure_phase(value):
    if value is None or isinstance(value, FailurePhase):
        return value
    return FailurePhase(value)


class CorruptRunRecord(RuntimeError):
    """A persisted run record cannot be decoded or validated."""

    def __init__(self, path, message):
        self.path = Path(path)
        super().__init__(f"Corrupt run record {self.path}: {message}")


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


@dataclass(frozen=True)
class RunRecord:
    """Single durable identity, file manifest and lifecycle of one calculation."""

    id: str
    state: RunState
    created_at: str
    updated_at: str
    task: str
    execution: Execution
    files: FileManifest
    finished_at: Optional[str] = None
    dataset: Optional[str] = None
    return_code: Optional[int] = None
    message: Optional[str] = None
    input_definition: Optional[str] = None
    failure_phase: Optional[FailurePhase] = None
    schema_version: int = _RUN_RECORD_SCHEMA_VERSION
    project_id: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "task", str(self.task).upper())
        if self.dataset is not None:
            object.__setattr__(self, "dataset", str(self.dataset))
        if self.project_id is not None:
            object.__setattr__(self, "project_id", str(self.project_id))
        if self.input_definition is not None:
            object.__setattr__(self, "input_definition", str(self.input_definition).upper())
        if isinstance(self.execution, dict):
            object.__setattr__(self, "execution", Execution(**self.execution))
        if isinstance(self.files, dict):
            object.__setattr__(self, "files", FileManifest(**self.files))
        if not isinstance(self.state, RunState):
            object.__setattr__(self, "state", RunState(self.state))
        if self.failure_phase is not None and not isinstance(self.failure_phase, FailurePhase):
            object.__setattr__(self, "failure_phase", FailurePhase(self.failure_phase))
        if self.schema_version != _RUN_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {self.schema_version}; "
                f"expected {_RUN_RECORD_SCHEMA_VERSION}"
            )


class RunStore:
    """Project run history stored as date-sharded durable JSON files."""

    def __init__(self, directory, *, projects_directory=None):
        """A run store belongs to one directory; project stores guard deletion."""
        self.directory = Path(directory).expanduser()
        self.projects_directory = (
            None if projects_directory is None else Path(projects_directory).expanduser()
        )
        self._lock = RLock()
        self._warned_corrupt = set()

    @contextmanager
    def _project_guard(self, record):
        if self.projects_directory is None:
            yield
            return
        if record.project_id is None:
            raise ValueError("Project run history requires a canonical project ID")
        project_id = _project_id(record.project_id)
        if record.project_id != project_id:
            raise ValueError("Project run history requires a canonical project ID")
        with FileLock(self.projects_directory / f".runs-{project_id}.lock"):
            if not (self.projects_directory / f"{project_id}.json").is_file():
                raise KeyError(f"Project {project_id} no longer exists")
            yield

    @staticmethod
    def _bucket(created_at):
        try:
            return datetime.fromisoformat(created_at).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid run creation timestamp: {created_at!r}") from exc

    def _path(self, record):
        return self.directory / self._bucket(record.created_at) / f"{record.id}.json"

    def _run_lock_path(self, record):
        if self.projects_directory is not None:
            return self.directory / f".{record.id}.lock"
        path = self._path(record)
        return path.with_name(f".{record.id}.lock")

    def _path_for_id(self, run_id):
        filename = f"{run_id}.json"
        for day in self._day_directories(newest=True):
            path = day / filename
            if path.is_file():
                return path
        raise FileNotFoundError(run_id)

    @staticmethod
    def _decode(path):
        try:
            with path.open() as file:
                data = json.load(file)
            version = data.get("schema_version")
            if version != _RUN_RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported schema version {version!r}; "
                    f"expected {_RUN_RECORD_SCHEMA_VERSION}"
                )
            return RunRecord(**data)
        except CorruptRunRecord:
            raise
        except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
            raise CorruptRunRecord(path, str(exc)) from exc

    def _warn_corrupt(self, exc):
        key = os.fspath(exc.path)
        if key in self._warned_corrupt:
            return
        self._warned_corrupt.add(key)
        warnings.warn(str(exc), RuntimeWarning, stacklevel=3)

    @staticmethod
    def _json_default(value):
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _write_unlocked(self, record):
        durable_mkdir(self.directory, parents=True, exist_ok=True, mode=0o700)
        path = self._path(record)
        durable_mkdir(path.parent, parents=True, exist_ok=True, mode=0o700)
        durable_write_json(
            path,
            asdict(record),
            mode=0o600,
            sort_keys=True,
            default=self._json_default,
        )
        return record

    @contextmanager
    def submission_lock(self):
        durable_mkdir(self.directory, parents=True, exist_ok=True, mode=0o700)
        with self._lock, FileLock(self.directory / ".submit.lock"):
            yield

    def reserve_files(self, record):
        """Atomically claim the file namespace of a new run.

        Return (reserved_record, conflicts). The run record is the only
        durable reservation; there is no separate reservation state.
        """
        with self.submission_lock():
            conflicts = tuple(
                active for active in self.runs(active=True)
                if active.id != record.id
                and submission_conflicts(record, active)
            )
            if conflicts:
                return None, conflicts
            sync_result_files(record.files.input_files)

            def claim(latest):
                if latest.state != RunState.NEW:
                    raise RuntimeError("Only new runs can reserve files")
                return replace(
                    latest, state=RunState.SUBMITTING,
                    files=record.files, execution=record.execution,
                    updated_at=_now(),
                )

            return self.update(record, claim), ()

    def _check_persisted_project(self, record):
        if self.projects_directory is None:
            return
        filename = f"{record.id}.json"
        for day in self._day_directories(newest=False):
            path = day / filename
            if path.is_file() and self._decode(path).project_id != record.project_id:
                raise ValueError("Run write cannot change persistent project identity")

    def save(self, record: RunRecord):
        with self._lock, FileLock(self._run_lock_path(record)):
            self._check_persisted_project(record)
            with self._project_guard(record):
                return self._write_unlocked(record)

    def update(self, record: RunRecord, updater):
        with self._lock, FileLock(self._run_lock_path(record)):
            self._check_persisted_project(record)
            with self._project_guard(record):
                path = self._path(record)
                latest = self._decode(path) if path.is_file() else record
                updated = updater(latest)
                if (updated.id != latest.id or updated.created_at != latest.created_at
                        or updated.project_id != latest.project_id):
                    raise ValueError("Run update cannot change persistent identity")
                if updated is latest:
                    # A previous atomic rename may have succeeded just before
                    # fsync failed. Reaffirm durability without rewriting the
                    # terminal record or changing its update timestamp.
                    if latest.state.is_terminal and path.is_file():
                        sync_file(path)
                    return latest
                return self._write_unlocked(updated)

    def terminal_record(self, record):
        """Return a crash-durable terminal record, if one already exists.

        Take the same per-run lock as writers: visibility after an atomic
        rename does not by itself imply that the directory is durable.
        """
        with self._lock, FileLock(self._run_lock_path(record)):
            self._check_persisted_project(record)
            with self._project_guard(record):
                path = self._path(record)
                if not path.is_file():
                    return None
                latest = self._decode(path)
                if (latest.id != record.id
                        or latest.created_at != record.created_at
                        or latest.project_id != record.project_id):
                    raise ValueError("Terminal lookup cannot change persistent identity")
                if not latest.state.is_terminal:
                    return None
                sync_file(path)
                return latest

    def get(self, run_id):
        return self._decode(self._path_for_id(run_id))

    def _day_directories(self, *, newest):
        if not self.directory.is_dir():
            return ()
        days = [path for path in self.directory.iterdir() if path.is_dir()]
        return tuple(sorted(days, key=lambda path: path.name, reverse=newest))

    def _records_in_directory(self, directory, *, newest):
        records = []
        for path in directory.glob("*.json"):
            try:
                records.append(self._decode(path))
            except CorruptRunRecord as exc:
                self._warn_corrupt(exc)
        return sorted(records, key=lambda record: record.created_at, reverse=newest)

    def runs(self, *, active=None, limit=None, newest=False):
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return ()

        with self._lock:
            records = []
            for day in self._day_directories(newest=newest):
                for record in self._records_in_directory(day, newest=newest):
                    if active is True and record.state.is_terminal:
                        continue
                    if active is False and not record.state.is_terminal:
                        continue
                    records.append(record)
                    if limit is not None and len(records) >= limit:
                        return tuple(records)
            return tuple(records)


class ExecutionManager:
    """Persist public run lifecycle without task-specific knowledge."""

    def __init__(self, store=None, *, executors=None, submitting_grace=60.0,
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

    def register_executor(self, executor: Executor, endpoint=None):
        return self.executors.register(executor, endpoint)

    def executor_for(self, record):
        return self.executors.for_record(record)

    def input_file_is_current(self, run):
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        result_files = record.files
        return file_is_current(
            result_files, result_files.input_file, result_files.input_versions,
        )

    def output_file_is_current(self, run):
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        result_files = record.files
        return file_is_current(
            result_files, result_files.output_file, result_files.output_versions,
        )

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

    @staticmethod
    def _merged_state(latest, requested):
        if requested is None:
            return latest.state
        requested = RunState(requested)
        if latest.state.is_terminal:
            return latest.state
        if requested == RunState.UNKNOWN:
            # Uncertainty cannot retract confirmed execution completion.
            if latest.state in (RunState.POSTPROCESSING, RunState.DOWNLOADING):
                return latest.state
            return requested
        if latest.state == RunState.UNKNOWN:
            return requested
        if _STATE_RANK[requested] < _STATE_RANK[latest.state]:
            return latest.state
        return requested

    @staticmethod
    def _merged_result_files(latest, requested):
        if requested is _UNCHANGED or requested is None:
            return latest.files
        if isinstance(requested, dict):
            requested = FileManifest(**requested)
        current = latest.files
        if current is None:
            return requested
        if current.captured_at is not None:
            # Immutable historical evidence: an old runner must not replace
            # the captured digests with its uncaptured in-memory manifest.
            return current
        return requested

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
    ):
        def merge(latest):
            if expected_record is not None and latest != expected_record:
                raise RuntimeError("Run changed during postprocessing retry")
            if skip_if_terminal and latest.state.is_terminal:
                return latest
            if (expected_updated_at is not _UNCHANGED
                    and latest.updated_at != expected_updated_at):
                # An independent refresh has superseded this observation.
                return latest
            final_locked = latest.state.is_terminal
            now = _now()
            merged_state = self._merged_state(latest, state)
            if (state is not None and RunState(state) == RunState.UNKNOWN
                    and merged_state != RunState.UNKNOWN):
                # Ignore the entire stale observation, including its message
                # and file metadata, not merely its proposed state.
                return latest
            changes = {"state": merged_state, "updated_at": now}
            if not latest.state.is_terminal and merged_state.is_terminal:
                changes["finished_at"] = now
            if not final_locked:
                merged_result_files = self._merged_result_files(latest, files)
                if merged_result_files is not latest.files:
                    changes["files"] = merged_result_files
            if execution is not _UNCHANGED and execution is not None:
                if execution.endpoint != latest.execution.endpoint:
                    raise ValueError("Execution endpoint cannot change")
                if execution.directory.local != latest.execution.directory.local:
                    raise ValueError("Execution local directory cannot change")
                if latest.execution.backend_id is not None:
                    if execution.backend_id != latest.execution.backend_id:
                        raise ValueError("Backend identity cannot change")
                    if execution.directory != latest.execution.directory:
                        raise ValueError("Submitted execution directory cannot change")
                elif final_locked and execution != latest.execution:
                    raise ValueError("Final execution identity cannot change")
                changes["execution"] = execution
            if return_code is not _UNCHANGED and return_code is not None:
                if not final_locked or latest.return_code is None:
                    changes["return_code"] = return_code
            if message is not _UNCHANGED:
                if message is None and latest.state == RunState.UNKNOWN and merged_state != RunState.UNKNOWN:
                    # A later scheduler observation supersedes the stale
                    # connection-error diagnostic.
                    changes["message"] = None
                elif message is not None and (not final_locked or latest.message is None):
                    changes["message"] = message
            if failure_phase is not _UNCHANGED:
                normalized = _failure_phase(failure_phase)
                if not final_locked or latest.failure_phase is None:
                    changes["failure_phase"] = normalized
            return replace(latest, **changes)

        return self.store.update(record, merge)

    def update_file_availability(self, record, path, state, message=None):
        """Persist availability of one file without changing the run outcome."""
        state = FileAvailabilityState(state)

        def update(latest):
            return replace(
                latest,
                updated_at=_now(),
                files=latest.files.with_file_availability(
                    path, state.value, message
                ),
            )

        return self.store.update(record, update)

    def _is_stale_submitting(self, record, *, now=None):
        if record.state != RunState.SUBMITTING or record.execution.backend_id is not None:
            return False
        now = now or datetime.now(timezone.utc)
        return (
            now - _timestamp(record.updated_at)
        ).total_seconds() >= self.submitting_grace

    def recover_submitting(self, record):
        record = self.store.get(record.id)
        if not self._is_stale_submitting(record):
            return record
        executor = self.executor_for(record)
        try:
            status = executor.recover_submission(
                record.id,
                record,
                submitted_at=record.updated_at,
            )
        except NotImplementedError:
            return record
        if status is not None:
            return self.update(record, status)

        def mark_absent(latest):
            if latest.state != RunState.SUBMITTING or latest.execution.backend_id is not None:
                return latest
            now = _now()
            return replace(
                latest,
                state=RunState.FAILED,
                updated_at=now,
                finished_at=now,
                message="Stale submission was not found by its execution backend",
                failure_phase=FailurePhase.SUBMISSION,
            )

        return self.store.update(record, mark_absent)

    def reserve(self, record):
        """Claim files; only the manager orchestrates stale submission recovery."""
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

    def update(self, record, status: RunStatus):
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
        )
        self._acknowledge_submission(self.executor_for(updated), updated)
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
            if state == RunState.FINISHED:
                # The successful backend does not excuse a required download
                # failure; the failed file retains the specific diagnostics.
                state = RunState.FAILED
                phase = FailurePhase.DOWNLOAD
                message = result_files.availability_message or message

        if files_available:
            try:
                result_files = capture_result_files(result_files)
            except OSError as exc:
                state = RunState.FAILED
                phase = FailurePhase.FILES
                message = str(exc)
                result_files = result_files._map_files(
                    lambda item: replace(
                        item, availability_state="failed",
                        availability_message=message,
                    ) if item.availability_state in (None, "pending") else item
                )

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
        )

    def resume_download(self, run):
        """Explicitly reopen a failed post-execution transfer for retry.

        This is not an execution resubmit. The original backend handle remains
        authoritative and the attached raw runner replays/synchronizes files
        before task postprocessing can finish.
        """
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)

        def reopen(latest):
            if (latest.state != RunState.FAILED
                    or latest.failure_phase != FailurePhase.DOWNLOAD
                    or latest.return_code != 0
                    or latest.execution.backend_id is None
                    or latest.files.captured_at is not None):
                raise ValueError("Only failed downloads of successful backend jobs can be retried")
            # Keep already transferred files and all bytes on disk.
            # Only failed transfers become eligible for a fresh attempt.
            files = latest.files._map_files(
                lambda item: replace(
                    item, availability_state="pending",
                    availability_message=None,
                ) if item.availability_state == "failed" else item
            )
            return replace(
                latest, state=RunState.DOWNLOADING, finished_at=None,
                failure_phase=None, message=None, files=files, updated_at=_now(),
            )

        return self.store.update(record, reopen)

    def retry_postprocessing(self, run, process_result):
        """Retry parsing captured files, never submitting/attaching a backend.

        The durable FAILED -> POSTPROCESSING transition is atomic, so competing
        callers cannot both claim the same retry.
        """
        from .base import ExecutionOutcome

        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)

        def eligible(latest):
            files = latest.files
            if (latest.state != RunState.FAILED
                    or latest.failure_phase != FailurePhase.POSTPROCESSING
                    or latest.return_code != 0
                    or latest.execution.backend_id is None):
                raise ValueError("Only postprocessing failures of successful backend runs can be retried")
            if files.captured_at is None or files.output_file is None:
                raise ValueError("Postprocessing retry requires captured output files")
            if files.failed_files:
                raise ValueError("Postprocessing retry requires locally available files")
            # Captured output is authoritative; input may no longer be present.
            if (not files.output_versions
                    or any(version.state() is not FileState.CURRENT
                           for version in files.output_versions)
                    or not file_is_current(files, files.output_file,
                                                 files.output_versions)):
                raise ValueError("Postprocessing retry requires unchanged historical files")
            return replace(
                latest, state=RunState.POSTPROCESSING, finished_at=None,
                failure_phase=None, message=None, updated_at=_now(),
            )

        reopened = self.store.update(record, eligible)
        files = reopened.files
        outcome = ExecutionOutcome(
            input_file=(files.input_file if file_is_current(
                files, files.input_file, files.input_versions) else None),
            output_file=files.output_file,
            error_file=None if files.stderr is None else files.stderr.local_path,
            return_code=reopened.return_code,
        )
        try:
            result = process_result(outcome)
            # A concurrent external rewrite must not turn into a successful
            # historical result, even if it happened while parsing.
            if any(version.state() is not FileState.CURRENT
                   for version in files.output_versions):
                raise RuntimeError("Historical files changed during postprocessing retry")
        except BaseException as exc:
            self.complete(
                reopened, state=RunState.FAILED,
                return_code=reopened.return_code,
                message=str(exc), failure_phase=FailurePhase.POSTPROCESSING,
                files_available=False,
                expected_record=reopened,
            )
            raise
        self.complete(
            reopened, state=RunState.FINISHED,
            return_code=reopened.return_code, files=files,
            files_available=False, expected_record=reopened,
        )
        return result

    def retrieve_result_files(self, run):
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        if not record.state.is_terminal:
            raise RuntimeError(f"Run {record.id} is not finished")

        missing = missing_result_files(record.files)
        if not missing or record.execution.backend_id is None:
            return record

        executor = self.executor_for(record)
        if not executor.supports_result_file_retrieval:
            return record

        versions = {
            os.path.abspath(os.fspath(version.path)): version
            for version in record.files.all_versions
        }
        for path in missing:
            path = os.path.abspath(os.fspath(path))
            version = versions.get(path)
            staged = (
                Path(path).with_name(f".{Path(path).name}.{uuid4().hex}.tmp")
                if version is not None and version.digest is not None else None
            )
            try:
                if staged is None:
                    executor.fetch_result_files(record, (path,))
                else:
                    executor.fetch_result_files(
                        record, (path,), destinations={path: os.fspath(staged)},
                    )
                    if not staged.is_file():
                        raise FileNotFoundError(
                            f"Historical file was not retrieved: {path}"
                        )
                    if file_digest(staged) != version.digest:
                        raise HistoricalFileMismatchError(
                            f"Historical result files have been overwritten: {path}"
                        )
                    os.replace(staged, path)
            except OSError as exc:
                self.update_file_availability(
                    record, path, FileAvailabilityState.FAILED, str(exc),
                )
                raise
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
            refreshed = self.store.get(record.id)
            if path in missing_result_files(refreshed.files):
                record = self.update_file_availability(
                    refreshed, path, FileAvailabilityState.FAILED,
                    "Historical file is still missing",
                )
            else:
                record = self.update_file_availability(
                    refreshed, path, FileAvailabilityState.AVAILABLE,
                )
        return record

    @staticmethod
    def _acknowledge_submission(executor, record):
        """Discard the temporary backend PID only after RunStore is durable."""
        if record.execution.backend_id is None:
            return
        acknowledge = getattr(executor, "acknowledge_submission", None)
        if acknowledge is not None:
            try:
                acknowledge(record)
            except (OSError, RuntimeError):
                # The receipt remains recoverable until a subsequent refresh.
                pass

    def create_run(
        self,
        executor,
        spec,
        *,
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
        endpoint = endpoint_for_executor(executor)
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
        self.register_executor(executor, endpoint)

        raw = executor.create_project_run(spec, run_id=record.id)
        # The manifest is owned by RunRecord. Local execution observes its
        # publication outcomes, without persisting an independent file list.
        raw.set_manifest(record.files)
        processed = CalculationRun(raw, process_result, resources=resources)
        persistence = PersistentRun(self, executor, record, spec=spec)
        processed.hooks = persistence
        processed.callback = callback
        return processed

    def refresh(self, records=None):
        records = tuple(self.store.runs(active=True) if records is None else records)
        current = {record.id: record for record in records}
        groups = {}
        for record in records:
            if record.execution.backend_id is None:
                continue
            executor = self.executor_for(record)
            groups.setdefault(executor, []).append(record)

        for executor, group in groups.items():
            statuses = executor.status(*group)
            if len(statuses) != len(group):
                raise RuntimeError("Executor returned wrong number of run statuses")
            for record, status in zip(group, statuses):
                current[record.id] = self.update(record, status)
        return tuple(current[record.id] for record in records)

    def attach(
        self,
        run,
        *,
        process_result,
        executor=None,
        callback=None,
        output_listener=None,
    ):
        """Attach raw execution, then resume client-side postprocessing."""
        record = self.store.get(run) if isinstance(run, str) else run
        if record.execution.backend_id is None:
            raise RuntimeError(f"Run {record.id} has no backend reference")
        if executor is None:
            executor = self.executor_for(record)
        else:
            endpoint = record.execution.endpoint
            self.register_executor(executor, endpoint)

        raw = executor.attach(
            record,
            output_listener=output_listener,
            start=False,
        )
        processed = CalculationRun(raw, process_result)
        persistence = PersistentRun(self, executor, record)
        processed.hooks = persistence
        processed.callback = callback
        processed.start(attached=True)
        return processed


class PersistentRun:
    """Durable transaction participant for one CalculationRun.

    CalculationRun sequences execution and public completion. This object
    serializes durable reservation and completion commits. Its submission
    guard and completion lock protect distinct transactional boundaries,
    not a second runner lifecycle state machine.
    """

    def __init__(self, manager, executor, record, *, spec=None):
        self._spec = spec
        self._lifecycle_lock = Lock()
        self._submission_in_progress = False
        self.manager = manager
        self.executor = executor
        self._record = record
        self._completion_lock = Lock()
        # Network ACK is best-effort and intentionally separate from the
        # durable terminal state: a new instance retries it after a crash.
        self._terminal_acknowledged = False
        self._monitor_record_version = record.updated_at

    @property
    def id(self):
        return self._record.id

    @property
    def record(self):
        return self._record

    @staticmethod
    def _execution_for(run, record):
        """Derive execution from the observation and durable path mapping."""
        proposed = run.execution
        if proposed is not None:
            return proposed
        if run.backend_id is None:
            return None
        return record.execution.submitted(run.backend_id)

    def before_submit(self, run: CalculationRun):
        """Reserve the run atomically before any backend file staging."""
        with self._lifecycle_lock:
            if self._submission_in_progress or self._record.state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._record.state.value}")
            self._submission_in_progress = True
        try:
            if self._spec is not None:
                files, remote_directory = self.executor.reservation_context(
                    self._spec, self._record.files,
                )
                execution = self._record.execution
                if remote_directory is not None:
                    execution = replace(
                        execution,
                        directory=replace(
                            execution.directory, remote=remote_directory,
                        ),
                    )
                self._record = replace(self._record, files=files, execution=execution)
            self._record = self.manager.reserve(self._record)
            run.set_manifest(self._record.files)
            self._monitor_record_version = self._record.updated_at
        except BaseException:
            with self._lifecycle_lock:
                self._submission_in_progress = False
            raise

    def prepared(self, run: CalculationRun):
        """Commit the exact staged layout before the backend is launched."""
        self._record = self.manager.transition(
            self._record,
            execution=self._execution_for(run, self._record),
            files=run.files if run.files is not None else _UNCHANGED,
        )
        self._monitor_record_version = self._record.updated_at

    def submitted(self, run: CalculationRun):
        """Persist backend identity before clearing its temporary ticket."""
        try:
            backend_id = run.backend_id
            if backend_id is None:
                raise RuntimeError("Executor submit returned without a backend ID")
            self.manager.register_executor(
                self.executor, self._record.execution.endpoint,
            )
            with self._completion_lock:
                if not self._record.state.is_terminal:
                    observed = run.state
                    if observed.is_terminal:
                        # A backend may complete synchronously inside submit().
                        # The public completion callback is deferred until its
                        # identity is durable; do not commit a terminal record
                        # without its return code, error and file snapshot.
                        observed = RunState.SUBMITTING
                    self._record = self.manager.transition(
                        self._record, state=observed, execution=self._execution_for(run, self._record),
                        failure_phase=(None if observed == RunState.SUBMITTING
                                       else _failure_phase(run.failure_phase)),
                        files=run.files if run.files is not None else _UNCHANGED,
                    )
                    self._monitor_record_version = self._record.updated_at
            self.manager._acknowledge_submission(self.executor, self._record)
        finally:
            with self._lifecycle_lock:
                self._submission_in_progress = False

    def submission_failed(
        self, run: CalculationRun, error: BaseException, *, uncertain: bool,
    ):
        """Record definite non-launches and preserve uncertain submissions."""
        try:
            if not uncertain:
                if self._record.state == RunState.NEW:
                    return  # Reservation was refused; nothing was launched.
                if isinstance(error, asyncio.CancelledError):
                    self._record = self.manager.complete(
                        self._record, state=RunState.CANCELLED,
                        files_available=False,
                    )
                else:
                    phase = _failure_phase(run.failure_phase) or FailurePhase.SUBMISSION
                    message = run.failure_message or str(error)
                    self._record = self.manager.complete(
                        self._record, state=RunState.FAILED, message=message,
                        failure_phase=phase, files_available=False,
                    )
            elif run.backend_id is not None:
                with self._completion_lock:
                    if not self._record.state.is_terminal:
                        self._record = self.manager.transition(
                            self._record, state=run.state,
                            execution=self._execution_for(run, self._record),
                            failure_phase=_failure_phase(run.failure_phase),
                        )
        finally:
            with self._lifecycle_lock:
                self._submission_in_progress = False

    def attached(self, run: CalculationRun):
        """Synchronize an attached observation without resubmitting its job."""
        with self._completion_lock:
            state = run.state
            if (not self._record.state.is_terminal and run.backend_id is not None
                    and state != RunState.NEW):
                self._record = self.manager.transition(
                    self._record, state=state,
                    execution=self._execution_for(run, self._record), failure_phase=_failure_phase(run.failure_phase),
                )
                self._monitor_record_version = self._record.updated_at

    def can_cancel(self, run: CalculationRun):
        """A durable terminal record cannot be cancelled by a stale observer."""
        return not self._record.state.is_terminal

    def _acknowledge_terminal(self, run):
        """ACK only a durable terminal result; retry after a lost ACK reply."""
        if (self._terminal_acknowledged or not self._record.state.is_terminal
                or (self._record.execution.backend_id is None
                    and run.backend_id is None)):
            return
        acknowledge = getattr(self.executor, "acknowledge", None)
        if acknowledge is not None:
            try:
                acknowledge(self._record)
            except (OSError, RuntimeError):
                # Backend receipts remain safely repeatable after this failure.
                return
        self._terminal_acknowledged = True

    def completed(self, run: CalculationRun):
        """Persist one observation, taking idempotence from the durable record."""
        with self._completion_lock:
            if self._record.state.is_terminal:
                self._acknowledge_terminal(run)
                return self._record.state
            state = run.state
            phase = _failure_phase(run.failure_phase)
            message = run.failure_message
            error = run.exception
            if message is None and error is not None:
                message = str(error)
            if state == RunState.UNKNOWN:
                # Repeated UNKNOWN is not a new backend observation. A fresh
                # reattachment has its own PersistentRun and can recover it.
                if self._record.state == RunState.UNKNOWN:
                    return self._record.state
                # A stale observer cannot overwrite a newer refresh.
                self._record = self.manager.transition(
                    self._record, state=state, execution=self._execution_for(run, self._record),
                    message=message,
                    expected_updated_at=self._monitor_record_version,
                )
                return self._record.state
            if not state.is_terminal:
                return
            if state == RunState.FAILED and phase is None:
                phase = FailurePhase.EXECUTION

            files_available = phase not in _NO_FILES_FAILURES
            self._record = self.manager.complete(
                self._record,
                state=state,
                execution=self._execution_for(run, self._record),
                return_code=run.return_code,
                message=message,
                failure_phase=phase,
                files_available=files_available,
                files=run.files,
            )
            # ACK follows the crash-durable transition. A new PersistentRun
            # retries a lost ACK, without repeating terminal persistence.
            self._acknowledge_terminal(run)
            return self._record.state

    def refresh(self):
        with self._completion_lock:
            self._record = self.manager.refresh((self._record,))[0]
            return self._record
