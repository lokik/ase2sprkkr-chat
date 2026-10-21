"""Project run history and lifecycle orchestration."""

from __future__ import annotations

import asyncio
import fnmatch
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
from ._durability import durable_mkdir, durable_write_json
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
from .pool import default_executor_pool
from .projects import _project_id
from .postprocessing import CalculationRun
from .run_model import (
    FileAvailabilityState, Execution, WorkingDirectory, FileManifest,
)
from .transport import has_file_magic


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


def _manifest_accesses(record):
    """Resolve declared read/write paths on the execution host."""
    files = record.files
    directory = record.execution.directory
    remote = directory.remote is not None

    def path(entry):
        if entry is None:
            return None
        value = entry.remote_path if remote else entry.local_path
        if value is None:
            return None
        return os.path.normpath(
            value if os.path.isabs(value) else os.path.join(directory.target, value)
        )

    reads = {path(entry) for entry in files.input_entries}
    reads.update(path(cache) for cache in files.caches)
    writes = {path(entry) for entry in (files.stdout, files.stderr, *files.outputs)}
    if remote and (files.mapping is None or files.mapping.input_sync != "never"):
        writes.update(path(entry) for entry in files.input_entries)
    writes.update(path(cache) for cache in files.caches)
    return reads - {None}, writes - {None}


def _path_overlap(a, b):
    """Conservatively compare literal paths and glob declarations."""
    a_magic, b_magic = has_file_magic(a), has_file_magic(b)
    if not a_magic:
        return fnmatch.fnmatchcase(a, b) if b_magic else a == b
    if not b_magic:
        return fnmatch.fnmatchcase(b, a)
    if a == b:
        return True

    # Glob/glob intersection is not generally decidable by fnmatch.
    # Only disjoint literal prefixes can prove that no collision is possible.
    def prefix(pattern):
        return pattern[:min((pattern.find(c) for c in "*?[" if c in pattern),
                            default=len(pattern))]

    pa, pb = prefix(a), prefix(b)
    return pa.startswith(pb) or pb.startswith(pa)


def _paths_overlap(first, second):
    return any(_path_overlap(a, b) for a in first for b in second)


def submission_conflicts(left: RunRecord, right: RunRecord) -> bool:
    """Conservatively exclude overlapping declared read/write operations."""
    a, b = left.execution, right.execution
    if a.endpoint != b.endpoint:
        return False
    same_directory = a.directory.target == b.directory.target
    # SCF can overwrite undeclared canonical files in its working directory.
    if same_directory and (left.task == "SCF" or right.task == "SCF"):
        return True
    left_reads, left_writes = _manifest_accesses(left)
    right_reads, right_writes = _manifest_accesses(right)
    if (_paths_overlap(left_writes, right_reads | right_writes)
            or _paths_overlap(right_writes, left_reads)):
        return True
    # Without declared writes, preserve namespace-based protection for legacy
    # and generic executors that cannot describe their output footprint.
    if not left_writes or not right_writes:
        return (same_directory and left.task == right.task
                and left.dataset == right.dataset)
    return False


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
                return self._write_unlocked(updated)

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

    def __init__(self, store=None, *, executor_pool=None, submitting_grace=60.0,
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
        self.executor_pool = (
            executor_pool if executor_pool is not None else default_executor_pool()
        )
        self.submitting_grace = float(submitting_grace)

    def register_executor(self, executor: Executor, endpoint=None):
        endpoint = endpoint or endpoint_for_executor(executor)
        self.executor_pool.get(endpoint, lambda: executor)
        return endpoint

    def executor_for(self, record):
        return self.executor_pool.for_endpoint(record.execution.endpoint)

    def input_file_is_current(self, run):
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        result_files = record.files
        return file_is_current(
            result_files,
            None if result_files is None else result_files.input_file,
            () if result_files is None else result_files.input_versions,
        )

    def output_file_is_current(self, run):
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        result_files = record.files
        return file_is_current(
            result_files,
            None if result_files is None else result_files.output_file,
            () if result_files is None else result_files.output_versions,
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
    ):
        def merge(latest):
            if expected_record is not None and latest != expected_record:
                raise RuntimeError("Run changed during postprocessing retry")
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
        attempted_recovery = set()
        while True:
            with self.store.submission_lock():
                conflicts = tuple(
                    active
                    for active in self.store.runs(active=True)
                    if active.id != record.id
                    and submission_conflicts(record, active)
                )
                if not conflicts:
                    if record.files is not None:
                        sync_result_files(record.files.input_files)
                    return self.transition(
                        record,
                        state=RunState.SUBMITTING,
                        files=record.files,
                        execution=record.execution,
                    )
                stale = tuple(
                    conflict
                    for conflict in conflicts
                    if conflict.id not in attempted_recovery
                    and self._is_stale_submitting(conflict)
                )

            if not stale:
                raise DuplicateSubmissionError(record, conflicts)
            progressed = False
            for conflict in stale:
                attempted_recovery.add(conflict.id)
                recovered = self.recover_submitting(conflict)
                if recovered.state != RunState.SUBMITTING or recovered.execution.backend_id is not None:
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
        state = RunState(state)
        phase = _failure_phase(failure_phase)
        result_files = record.files if files is None else files

        if result_files is not None and result_files.failed_files:
            files_available = False
            if state == RunState.FINISHED:
                # The successful backend does not excuse a required download
                # failure; the failed file retains the specific diagnostics.
                state = RunState.FAILED
                phase = FailurePhase.DOWNLOAD
                message = result_files.availability_message or message

        if files_available and result_files is not None:
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
                    or latest.files is None
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
            if files is None or files.captured_at is None or files.output_file is None:
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
            for version in (() if record.files is None else record.files.all_versions)
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

    @staticmethod
    def _complete(project_run, result, callback):
        project_run._sync_completion(result)
        if callback is not None:
            # Completion of observation, including UNKNOWN, is notification.
            # It is not proof that the backend calculation finished.
            callback(result)

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
        set_manifest = getattr(raw, "set_manifest", None)
        if set_manifest is not None:
            set_manifest(record.files)
        processed = CalculationRun(raw, process_result, resources=resources)
        project_run = ProjectRunner(self, executor, processed, record, spec=spec)
        processed.hooks = project_run
        processed.callback = callback
        return project_run

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
        project_run = ProjectRunner(self, executor, processed, record)
        processed.hooks = project_run
        processed.callback = callback
        project_run.start()
        return project_run


class ProjectRunner:
    """Caller-owned runner that persists the public lifecycle directly."""

    def __init__(self, manager, executor, runner: CalculationRun, record, *, spec=None):
        self._spec = spec
        self._lifecycle_lock = Lock()
        self._submission_in_progress = False
        self.manager = manager
        self.executor = executor
        self._runner = runner
        self._record = record
        self._completion_lock = Lock()
        self._completion_persisted = False
        self._monitor_record_version = record.updated_at

    @property
    def id(self):
        return self._record.id

    @property
    def record(self):
        return self._record

    @property
    def result(self):
        return self._runner.result

    @property
    def backend_id(self):
        return self._runner.backend_id

    @property
    def execution(self):
        proposed = self._runner.execution
        if proposed is not None:
            return proposed
        if self.backend_id is None:
            return None
        return self._record.execution.submitted(self.backend_id)

    @property
    def files(self):
        return self._runner.files

    @property
    def exception(self):
        return self._runner.exception

    @property
    def failure_phase(self):
        return _failure_phase(self._runner.failure_phase)

    @property
    def state(self):
        """Observe runner state without persisting a transition."""
        state = self._runner.state
        if state == RunState.NEW and self._record.state != RunState.NEW:
            return self._record.state
        if state == RunState.UNKNOWN and self._record.state != RunState.NEW:
            # The completed observer cannot overwrite a newer refresh.
            return self._record.state
        return self._record.state if self._record.state.is_terminal else state

    def add_output_listener(self, callback):
        return self._runner.add_output_listener(callback)

    def add_output_error_listener(self, callback):
        return self._runner.add_output_error_listener(callback)

    def submit(self):
        # Claim exactly once, but never serialize cancellation behind backend I/O.
        with self._lifecycle_lock:
            if self._runner.cancellation_requested or self._runner.state == RunState.CANCELLED:
                raise RuntimeError("Run was cancelled before submission")
            if self._submission_in_progress or self._record.state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._record.state.value}")
            self._submission_in_progress = True
        submitted = False
        try:
            try:
                # Keep canonical manifest and directory resolution from #93;
                # potentially blocking target I/O stays outside lifecycle locks.
                if self._spec is not None:
                    files, remote_directory = self.executor.reservation_context(
                        self._spec, self._record.files,
                    )
                    execution = self._record.execution
                    if remote_directory is not None:
                        execution = replace(
                            execution,
                            directory=replace(execution.directory, remote=remote_directory),
                        )
                    self._record = replace(self._record, files=files, execution=execution)
                self._record = self.manager.reserve(self._record)
                self._runner.set_manifest(self._record.files)
                self._monitor_record_version = self._record.updated_at
            except BaseException:
                # Reservation failed before backend submission. Cleanup must
                # not hide the original store error.
                try:
                    self._runner.close()
                except BaseException:
                    pass
                raise
            backend_submit_attempted = False
            try:
                if self._runner.prepare_submission():
                    # Persist the exact staged file map and target directory before
                    # calling the backend. A lost submit reply can then be recovered
                    # without depending on live runner state.
                    self._record = self.manager.transition(
                        self._record,
                        execution=self.execution,
                        files=self.files if self.files is not None else _UNCHANGED,
                    )
                    self._monitor_record_version = self._record.updated_at
                if self._runner.cancellation_requested:
                    # This is still a definite pre-submit cancellation;
                    # settle the durable record in the exception handler.
                    raise asyncio.CancelledError("Run was cancelled before submission")
                backend_submit_attempted = True
                result = self._runner.submit()
                submitted = True
            except SubmissionNotStarted as exc:
                phase = self.failure_phase or FailurePhase.SUBMISSION
                message = self._runner.failure_message or str(exc)
                self._record = self.manager.complete(
                    self._record,
                    state=RunState.FAILED,
                    message=message,
                    failure_phase=phase,
                    files_available=False,
                )
                raise
            except asyncio.CancelledError:
                if not backend_submit_attempted:
                    # Cancellation during preflight is a definite non-submit,
                    # even when the raw runner has only latched the request.
                    self._runner.cancel()
                    if self._record.state != RunState.CANCELLED:
                        self._record = self.manager.complete(
                            self._record, state=RunState.CANCELLED,
                            files_available=False,
                        )
                # After backend submit was attempted, acceptance is uncertain.
                # Keep SUBMITTING for recovery rather than allowing a retry.
                raise
            except BaseException as exc:
                if not backend_submit_attempted:
                    # The manifest could not be persisted, so no backend submit
                    # was made. This is a definite submission failure, not an
                    # unknown job requiring recover_submitting().
                    try:
                        self._runner.close()
                    except BaseException:
                        pass
                    try:
                        self._record = self.manager.complete(
                            self._record, state=RunState.FAILED,
                            message=str(exc), failure_phase=FailurePhase.SUBMISSION,
                            files_available=False,
                        )
                    except Exception:
                        # Storage itself may be unavailable. Preserve the
                        # originating failure rather than masking it.
                        pass
                    raise
                # A backend may have accepted the job despite a lost reply.
                # Preserve SUBMITTING for recover_submitting().
                backend_id = self.backend_id
                if backend_id is not None:
                    self._record = self.manager.transition(
                        self._record,
                        state=self._runner.state,
                        execution=self.execution,
                        failure_phase=self.failure_phase,
                    )
                raise
            backend_id = self.backend_id
            if backend_id is None:
                raise RuntimeError("Executor submit returned without a backend ID")
            self.manager.register_executor(self.executor, self._record.execution.endpoint)
            # A fast backend may already have finished, but only the
            # postprocessing callback may publish a terminal RunRecord.
            with self._completion_lock:
                if not self._completion_persisted:
                    observed_state = self._runner.state
                    if observed_state == RunState.FINISHED:
                        observed_state = RunState.POSTPROCESSING
                    self._record = self.manager.transition(
                        self._record, state=observed_state,
                        execution=self.execution, failure_phase=self.failure_phase,
                        files=self.files if self.files is not None else _UNCHANGED,
                    )
                    self._monitor_record_version = self._record.updated_at
            self.manager._acknowledge_submission(self.executor, self._record)
            return result
        finally:
            with self._lifecycle_lock:
                self._submission_in_progress = False
                dispatch = submitted and self._runner.cancellation_requested
            if dispatch:
                # Project cancellation was accepted while the backend was
                # submitting. The raw runner is now allowed to dispatch it.
                self._runner.cancel()

    def start(self):
        self._runner.start()
        with self._completion_lock:
            # An attached PSI/J job can initially report NEW before the
            # scheduler provides its actual state.
            state = self._runner.state
            if (not self._completion_persisted and self.backend_id is not None
                    and state != RunState.NEW):
                self._record = self.manager.transition(
                    self._record,
                    state=state,
                    execution=self.execution,
                    failure_phase=self.failure_phase,
                )
                self._monitor_record_version = self._record.updated_at

    def completed(self, run: CalculationRun):
        """Durably record an observed completion before user notifications."""
        self._sync_completion(run.result)

    def _sync_completion(self, result):
        # Callback and wait() can reach this path concurrently. Only the
        # first successful durable completion may publish and acknowledge it.
        with self._completion_lock:
            if self._completion_persisted:
                return
            state = self._runner.state
            phase = self.failure_phase
            message = self._runner.failure_message
            error = self._runner.exception
            if message is None and error is not None:
                message = str(error)
            if state == RunState.UNKNOWN:
                # The observer ended, not necessarily the calculation.
                # No final file publication or completion timestamp.
                self._record = self.manager.transition(
                    self._record, state=state, execution=self.execution,
                    message=message,
                    expected_updated_at=self._monitor_record_version,
                )
                self._completion_persisted = True
                return
            if not state.is_terminal:
                return
            if state == RunState.FAILED and phase is None:
                phase = FailurePhase.EXECUTION

            files_available = phase not in _NO_FILES_FAILURES
            self._record = self.manager.complete(
                self._record,
                state=state,
                execution=self.execution,
                return_code=self._runner.return_code,
                message=message,
                failure_phase=phase,
                files_available=files_available,
                files=self.files,
            )
            # ACK is strictly after RunStore's crash-durable terminal transition.
            # A lost ACK reply is harmless: backend receipts are idempotently deleted.
            # Keep the persisted result even if the acknowledgement cannot be sent.
            if self._record.state.is_terminal and self.backend_id is not None:
                acknowledge = getattr(self.executor, "acknowledge", None)
                if acknowledge is not None:
                    try:
                        acknowledge(self._record)
                    except (OSError, RuntimeError):
                        pass

            if self._record.state.is_terminal:
                self._completion_persisted = True

    def wait(self):
        try:
            return self._runner.wait()
        finally:
            # A failed durable completion must be retried explicitly, not
            # silently during wait() while the caller is handling the error.
            if (self._record.state != RunState.NEW
                    and not (isinstance(self._runner, CalculationRun)
                             and self._runner.persistence_error is not None)):
                self._sync_completion(self.result)

    def run(self):
        if self._runner.state == RunState.CANCELLED and self._record.state == RunState.NEW:
            raise RuntimeError("Run was cancelled before submission")
        if self._record.state == RunState.NEW:
            self.submit()
        return self.wait()

    def close(self):
        """Release an unsubmitted project's prepared resources."""
        with self._lifecycle_lock:
            if self._submission_in_progress or self._record.state != RunState.NEW:
                raise RuntimeError("Cannot close a submitted project run")
            return self._runner.close()

    async def run_async(self):
        return await asyncio.to_thread(self.run)

    def refresh(self):
        with self._completion_lock:
            self._record = self.manager.refresh((self._record,))[0]
            return self._record

    def stop_observation(self):
        """Detach this client without cancelling its still-running job.

        The completion callback persists a recoverable UNKNOWN observation.
        Call wait() to ensure the monitor has finished and been detached.
        """
        return self._runner.stop_observation()

    def cancel(self):
        # The persisted terminal state is final. In particular, never request
        # backend cancellation merely because a stale observer still runs.
        if self._record.state.is_terminal:
            return None
        # CalculationRun owns the cancellation latch. This must happen before
        # acquiring any lock so a blocking reservation cannot delay cancel.
        self._runner.request_cancel()
        with self._lifecycle_lock:
            if self._submission_in_progress:
                # Project reservation/staging owns the prepared inputs until
                # submit() settles the request or dispatches backend cancel.
                return None
        with self._completion_lock:
            if self._record.state.is_terminal:
                return None
        try:
            return self._runner.cancel()
        finally:
            with self._completion_lock:
                if self._record.state != RunState.NEW and not self._completion_persisted:
                    self._record = self.manager.transition(
                        self._record, state=self._runner.state,
                        execution=self.execution, failure_phase=self.failure_phase,
                    )
                    self._monitor_record_version = self._record.updated_at

