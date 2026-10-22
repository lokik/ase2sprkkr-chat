"""Atomic, date-sharded persistent storage of project run records."""

from __future__ import annotations

import json
from hashlib import sha256
import os
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from threading import RLock

from ..common.file_locks import FileLock
from ._durability import durable_mkdir, durable_write_json, fsync_directory, sync_file
from .base import RunState
from .file_reservations import reservation_conflict
from .file_history import sync_result_files
from .projects import _project_id
from .run_record import RunRecord, SubmissionPhase, _RUN_RECORD_SCHEMA_VERSION, _now, _run_id


class CorruptRunRecord(RuntimeError):
    """A persisted run record cannot be decoded or validated."""

    def __init__(self, path, message):
        self.path = Path(path)
        super().__init__(f"Corrupt run record {self.path}: {message}")


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
        return self.directory / self._bucket(record.created_at) / f"{_run_id(record.id)}.json"

    def _run_lock_path(self, record):
        # Run IDs are unique across the entire store, not only within a day shard.
        return self.directory / f".{_run_id(record.id)}.lock"

    def _path_for_id(self, run_id):
        filename = f"{_run_id(run_id)}.json"
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

    @contextmanager
    def submission_owner(self, record):
        """Nonblocking ownership of one submission, released on all exits."""
        digest = sha256(record.id.encode("utf8")).hexdigest()
        lock = FileLock(self.directory / f".submission-{digest}.lock")
        if not lock.acquire(blocking=False):
            yield False
            return
        try:
            yield True
        finally:
            lock.release()

    def begin_dispatch(self, record):
        """Durably cross the uncertain backend launch boundary."""
        def dispatch(latest):
            if (latest != record or latest.state != RunState.SUBMITTING
                    or latest.submission_phase != SubmissionPhase.PREPARING
                    or latest.execution.backend_id is not None):
                raise RuntimeError("Submission reservation was lost before dispatch")
            return replace(latest, submission_phase=SubmissionPhase.DISPATCHING, updated_at=_now())
        return self.update(record, dispatch)

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
                sync_result_files(record.files.input_files)
                return replace(
                    latest, state=RunState.SUBMITTING,
                    submission_phase=SubmissionPhase.PREPARING,
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

    def delete_records(self, records):
        """Remove terminal snapshots under the caller's project-deletion locks.

        The project repository must first exclude concurrent submissions and
        writes. Validate the full batch before removing any history.
        """
        records = tuple(records)
        if any(not record.state.is_terminal for record in records):
            raise ValueError("Cannot delete nonterminal run history")
        with self._lock:
            for record in records:
                path = self._path(record)
                path.unlink(missing_ok=True)
                fsync_directory(path.parent)

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
