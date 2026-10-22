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
from ._durability import durable_mkdir, durable_write_json, sync_file
from .base import (
    Executor,
    FailurePhase,
    RunState,
    RunStatus,
)
from .job_files import JobFiles, HistoricalFileMismatchError
from .file_reservations import reservation_conflict
from .endpoint import ExecutorEndpoint
from .pool import Executors
from .projects import _project_id
from .postprocessing import CalculationRun
from .run_model import (
    Execution, WorkingDirectory, FileManifest,
)


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
    def _merged_state(self, requested):
        if requested is None:
            return self.state
        requested = RunState(requested)
        if self.state.is_terminal:
            return self.state
        if requested == RunState.UNKNOWN:
            # Uncertainty cannot retract confirmed execution completion.
            if self.state in (RunState.POSTPROCESSING, RunState.DOWNLOADING):
                return self.state
            return requested
        if self.state == RunState.UNKNOWN:
            return requested
        if _STATE_RANK[requested] < _STATE_RANK[self.state]:
            return self.state
        return requested

    def with_transition(
        self,
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
        """Return a validated immutable snapshot of a run-state observation.

        No disk I/O or backend calls happen here; RunStore owns atomicity.
        """
        if expected_record is not None and self != expected_record:
            raise RuntimeError("Run changed during postprocessing retry")
        if skip_if_terminal and self.state.is_terminal:
            return self
        if (expected_updated_at is not _UNCHANGED
                and self.updated_at != expected_updated_at):
            # An independent refresh has superseded this observation.
            return self
        final_locked = self.state.is_terminal
        now = _now()
        merged_state = self._merged_state(state)
        if (state is not None and RunState(state) == RunState.UNKNOWN
                and merged_state != RunState.UNKNOWN):
            # Ignore the entire stale observation, including its message
            # and file metadata, not merely its proposed state.
            return self
        changes = {"state": merged_state, "updated_at": now}
        if not self.state.is_terminal and merged_state.is_terminal:
            changes["finished_at"] = now
        if not final_locked and files is not _UNCHANGED and files is not None:
            merged_files = JobFiles.merge_snapshot(self.files, files)
            if merged_files is not self.files:
                changes["files"] = merged_files
        if execution is not _UNCHANGED and execution is not None:
            if execution.endpoint != self.execution.endpoint:
                raise ValueError("Execution endpoint cannot change")
            if execution.directory.local != self.execution.directory.local:
                raise ValueError("Execution local directory cannot change")
            if self.execution.backend_id is not None:
                if execution.backend_id != self.execution.backend_id:
                    raise ValueError("Backend identity cannot change")
                if execution.directory != self.execution.directory:
                    raise ValueError("Submitted execution directory cannot change")
            elif final_locked and execution != self.execution:
                raise ValueError("Final execution identity cannot change")
            changes["execution"] = execution
        if return_code is not _UNCHANGED and return_code is not None:
            if not final_locked or self.return_code is None:
                changes["return_code"] = return_code
        if message is not _UNCHANGED:
            if message is None and self.state == RunState.UNKNOWN and merged_state != RunState.UNKNOWN:
                # A later scheduler observation supersedes the stale
                # connection-error diagnostic.
                changes["message"] = None
            elif message is not None and (not final_locked or self.message is None):
                changes["message"] = message
        if failure_phase is not _UNCHANGED:
            normalized = _failure_phase(failure_phase)
            if not final_locked or self.failure_phase is None:
                changes["failure_phase"] = normalized
        if validate_completed_files and changes["state"] == RunState.FINISHED:
            # Completion must use the manifest merged under the RunStore
            # lock. A concurrent download failure may have been persisted
            # after the caller captured its candidate manifest.
            completed_files = changes.get("files", self.files)
            if completed_files.failed_files:
                changes["state"] = RunState.FAILED
                changes["failure_phase"] = FailurePhase.DOWNLOAD
                changes["message"] = completed_files.availability_message or message
                if (completed_files.captured_at is not None
                        and self.files.captured_at is None):
                    changes["files"] = JobFiles.discard_capture(completed_files)
        return replace(self, **changes)


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
        # Run IDs are unique across the entire store, not only within a day shard.
        return self.directory / f".{record.id}.lock"

    def _path_for_id(self, run_id):
        filename = f"{run_id}.json"
        found = None
        for day in self._day_directories(newest=True):
            path = day / filename
            if not path.is_file():
                continue
            if found is not None:
                raise ValueError(f"Duplicate run ID {run_id!r} across date shards")
            found = path
        if found is None:
            raise FileNotFoundError(run_id)
        return found

    @staticmethod
    def _decode(path):
        try:
            with path.open() as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError("Run record must be a JSON object")
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
        """Atomically check file conflicts and commit a new run's submission."""
        with self.submission_lock():
            conflicts = tuple(
                active for active in self.runs(active=True, strict=True)
                if active.id != record.id
                and active.state != RunState.NEW
                and reservation_conflict(record, active)
            )
            if conflicts:
                return None, conflicts

            def claim(latest):
                if latest.state != RunState.NEW:
                    raise RuntimeError("Only new runs can reserve files")
                if latest != record:
                    raise RuntimeError("Run changed during file reservation")
                # Do not sync inputs from a stale claim. Persist the claim
                # only after the current manifest's inputs are durable.
                JobFiles(record.files).sync_inputs()
                return replace(
                    latest, state=RunState.SUBMITTING,
                    updated_at=_now(),
                )
            return self.update(record, claim), ()

    def _check_persisted_identity(self, record):
        try:
            path = self._path_for_id(record.id)
        except FileNotFoundError:
            return
        existing = self._decode(path)
        if existing.project_id != record.project_id:
            raise ValueError("Run write cannot change persistent project identity")
        if existing.created_at != record.created_at:
            raise ValueError("Run write cannot change persistent creation time")

    @contextmanager
    def _locked_run(self, record):
        """Serialize durable access and validate the run's project."""
        with self._lock, FileLock(self._run_lock_path(record)):
            with self._project_guard(record):
                self._check_persisted_identity(record)
                yield

    def save(self, record: RunRecord):
        with self._locked_run(record):
            return self._write_unlocked(record)

    def update(self, record: RunRecord, updater):
        with self._locked_run(record):
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
        """Return a crash-durable terminal record, if one already exists."""
        with self._locked_run(record):
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

    def _records_in_directory(self, directory, *, newest, strict=False):
        records = []
        for path in directory.glob("*.json"):
            try:
                records.append(self._decode(path))
            except CorruptRunRecord as exc:
                if strict:
                    raise
                self._warn_corrupt(exc)
        return sorted(records, key=lambda record: record.created_at, reverse=newest)

    def runs(self, *, active=None, limit=None, newest=False, strict=False):
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0:
            return ()

        with self._lock:
            records = []
            seen_ids = set()
            for day in self._day_directories(newest=newest):
                for record in self._records_in_directory(day, newest=newest, strict=strict):
                    if record.id in seen_ids:
                        raise ValueError(f"Duplicate run ID {record.id!r} across date shards")
                    seen_ids.add(record.id)
                    if active is True and record.state.is_terminal:
                        continue
                    if active is False and not record.state.is_terminal:
                        continue
                    records.append(record)
            return tuple(records if limit is None else records[:limit])


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
        now = now or datetime.now(timezone.utc)
        return (
            now - _timestamp(record.updated_at)
        ).total_seconds() >= self.submitting_grace

    def recover_submitting(self, record):
        record = self.store.get(record.id)
        if not self._is_stale_submitting(record):
            return record
        executor = self.executors.for_endpoint(record.execution.endpoint)
        try:
            status = executor.recover_submission(
                record.id,
                record,
                submitted_at=record.updated_at,
            )
        except NotImplementedError:
            return record
        if status is not None:
            return self.update(record, status, expected_updated_at=record.updated_at)

        def mark_absent(latest):
            # Do not invalidate a newer submission observation while backend
            # recovery was in progress.
            if (latest.updated_at != record.updated_at
                    or latest.state != RunState.SUBMITTING
                    or latest.execution.backend_id is not None):
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
                result_files = JobFiles(record.files).capture(result_files)
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
        missing = JobFiles(record.files).missing_files()
        executor = (
            self.executors.for_endpoint(record.execution.endpoint)
            if missing and record.execution.backend_id is not None else None
        )
        return JobFiles(record.files).retrieve(record, self.store, executor)

    @staticmethod
    def _acknowledge_submission(executor, record):
        """Discard the temporary backend PID only after RunStore is durable."""
        if record.execution.backend_id is None:
            return
        try:
            executor.acknowledge_submission(record)
        except (OSError, RuntimeError):
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
            if record.state.is_terminal or record.execution.backend_id is None:
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


class RunPersistence:
    """Durable transaction participant for one CalculationRun.

    CalculationRun sequences execution and public completion. This object
    serializes durable commits. The completion lock protects refresh and
    finalization; submission uniqueness is owned by CalculationRun and the
    atomically reserving RunStore, not by a second lifecycle state machine.
    """

    def __init__(self, manager, executor, record, *, spec=None):
        self._spec = spec
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
        """Preserve reserved placement; only backend identity comes from the runner."""
        if run.backend_id is None:
            return None
        return record.execution.submitted(run.backend_id)

    def before_submit(self, run: CalculationRun):
        """Reserve atomically before any backend file staging."""
        if self._record.state != RunState.NEW:
            raise RuntimeError(f"Run is already {self._record.state.value}")
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
        run.job_files.replace(self._record.files)
        self._monitor_record_version = self._record.updated_at

    def prepared(self, run: CalculationRun):
        """Commit the exact staged layout before the backend is launched."""
        self._record = self.manager.transition(
            self._record,
            execution=self._execution_for(run, self._record),
            files=run.job_files.snapshot(),
        )
        self._monitor_record_version = self._record.updated_at

    def submitted(self, run: CalculationRun):
        """Persist backend identity before clearing its temporary ticket."""
        if run.backend_id is None:
            raise RuntimeError("Executor submit returned without a backend ID")
        with self._completion_lock:
            if not self._record.state.is_terminal:
                observed = run.state
                if observed.is_terminal:
                    # The backend may finish synchronously in submit().
                    # Its completion callback waits for durable identity.
                    observed = RunState.SUBMITTING
                self._record = self.manager.transition(
                    self._record, state=observed,
                    execution=self._execution_for(run, self._record),
                    failure_phase=(None if observed == RunState.SUBMITTING
                                   else _failure_phase(run.failure_phase)),
                    files=run.job_files.snapshot(),
                )
                self._monitor_record_version = self._record.updated_at
        self.manager._acknowledge_submission(self.executor, self._record)

    def submission_failed(
        self, run: CalculationRun, error: BaseException, *, uncertain: bool,
    ):
        """Record definite non-launches and preserve uncertain submissions."""
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
        try:
            self.executor.acknowledge(self._record)
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
                # reattachment has its own RunPersistence and can recover it.
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
                files=run.job_files.snapshot(),
            )
            # ACK follows the crash-durable transition. A new RunPersistence
            # retries a lost ACK, without repeating terminal persistence.
            self._acknowledge_terminal(run)
            return self._record.state

    def refresh(self):
        with self._completion_lock:
            self._record = self.manager.refresh((self._record,))[0]
            return self._record
