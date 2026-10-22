"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_record import RunRecord

from ..common.file_locks import FileLock
from .run_model import FileAvailabilityState, FileManifest
from .transport import has_file_magic
from .historical_restore import (
    restore_historical_file,
)

from .file_history import (
    capture_result_files, file_is_current, missing_result_files, sync_result_files,
)


class JobFiles:
    """Mutable per-run file state over immutable FileManifest values.

    File manifests are validated value objects. No RunRecord or RunStore is
    retained by the working file owner; historical operations receive their
    RunRecord explicitly.
    """

    def __init__(self, manifest: FileManifest):
        if not isinstance(manifest, FileManifest):
            raise TypeError("JobFiles expects a FileManifest")
        self._manifest = manifest
        self._lock = RLock()

    @property
    def manifest(self) -> FileManifest:
        with self._lock:
            return self._manifest

    def snapshot(self) -> FileManifest:
        """Return a stable immutable value; subsequent changes cannot mutate it."""
        return self.manifest

    def replace(self, manifest: FileManifest) -> FileManifest:
        """Install a reserved or recovered authoritative file state."""
        if not isinstance(manifest, FileManifest):
            raise TypeError("JobFiles expects a FileManifest")
        with self._lock:
            self._manifest = manifest
            return manifest

    def merge(self, observed: FileManifest) -> FileManifest:
        """Apply an observation to the latest working state atomically."""
        with self._lock:
            self._manifest = self.merge_snapshot(self._manifest, observed)
            return self._manifest

    def mark_availability(self, path, state, message=None, *, remote_path=None):
        """Update a declared file in working memory, retaining older snapshots."""
        with self._lock:
            self._manifest = self._manifest.with_file_availability(
                path, FileAvailabilityState(state).value, message,
                remote_path=remote_path,
            )
            return self._manifest

    @staticmethod
    def sync_inputs(manifest):
        sync_result_files(manifest.input_files)

    @staticmethod
    def capture(manifest):
        return capture_result_files(manifest)

    @staticmethod
    def missing_files(manifest):
        return missing_result_files(manifest)

    @staticmethod
    def input_file_is_current(record: RunRecord):
        """Check the input file against the immutable historical snapshot."""
        files = record.files
        return file_is_current(files, files.input_file, files.input_versions)

    @staticmethod
    def output_file_is_current(record: RunRecord):
        """Check the primary output against the immutable historical snapshot."""
        files = record.files
        return file_is_current(files, files.output_file, files.output_versions)

    @staticmethod
    def _merge_availability(previous, received, preserve):
        """Reconcile matching files without altering the received membership.

        FileManifest already rejects output/cache identity collisions.
        """
        # Remote-only caches are included even though FileManifest.files
        # deliberately omits them. Merged streams share one physical identity.
        recorded = {}
        priority = {"failed": 3, "available": 2, "skipped": 1, "pending": 0, None: 0}
        for entry in (previous.stdout, previous.stderr,
                      *previous.outputs, *previous.caches):
            if entry is None:
                continue
            identity = (entry.local_path, entry.remote_path)
            prior = recorded.get(identity)
            if (prior is None or
                    priority[entry.availability_state] > priority[prior.availability_state]):
                recorded[identity] = entry

        def merged(entry):
            if entry is None:
                return None
            prior = recorded.get((entry.local_path, entry.remote_path))
            if prior is None or not preserve(prior, entry):
                return entry
            if (entry.availability_state == prior.availability_state
                    and entry.availability_message == prior.availability_message):
                return entry
            return replace(
                entry, availability_state=prior.availability_state,
                availability_message=prior.availability_message,
            )

        return received._map_files(merged)

    @staticmethod
    def discard_capture(manifest):
        """Invalidate speculative captured versions, preserving file statuses."""
        return replace(manifest, captured_at=None, input_versions=(), output_versions=())

    @staticmethod
    def merge_snapshot(current, requested):
        """Do not let stale observations erase independently persisted delivery."""
        from .run_model import FileManifest

        if isinstance(requested, dict):
            requested = FileManifest(**requested)
        if current is None:
            return requested
        if current.captured_at is not None:
            if requested != current:
                raise ValueError("Cannot replace a captured file manifest")
            return current

        def preserve(prior, entry):
            if prior.availability_state not in ("available", "failed", "skipped"):
                return False
            if entry.availability_state == "pending":
                return True
            # A concurrent failed transfer invalidates speculative capture.
            return (
                requested.captured_at is not None
                and entry.availability_state in ("available", "skipped")
                and prior.availability_state == "failed"
            )

        return JobFiles._merge_availability(current, requested, preserve)

    @staticmethod
    def merge_terminal_files(previous, received):
        """Retain only still-present local deliveries in terminal facts."""
        if previous is None:
            return received

        def preserve(prior, entry):
            if entry.availability_state in ("failed", "skipped"):
                return False
            if prior.availability_state == "skipped":
                return True
            return (
                prior.availability_state == "available"
                and entry.local_path is not None
                and os.path.isfile(entry.local_path)
            )

        return JobFiles._merge_availability(previous, received, preserve)

    @classmethod
    async def collect_remote(
        cls, previous, frozen, handler, outcome, spec, *,
        required, materialization_error=None,
    ):
        """Reconcile frozen membership, collect files and verify FOLLOW streams.

        The handler performs I/O; this operation owns the terminal manifest
        reconciliation and error precedence. A failed transfer must not erase
        independently delivered streams or previously successful files.
        """
        manifest = cls.merge_terminal_files(previous, frozen)
        try:
            manifest, transfer_error = await handler.collect(
                manifest.mapping, manifest, outcome, required=required,
            )
        except Exception as exc:
            # A preflight failure precedes any transfer; still verify FOLLOW
            # streams on successful jobs without changing output membership.
            transfer_error = exc

        if required and manifest is not None:
            try:
                manifest, stream_error = await handler.verify_stream_availability(
                    manifest, spec,
                )
            except Exception as exc:
                # Preserve all independently delivered output/cache files.
                stream_error = exc
            if transfer_error is None:
                transfer_error = stream_error

        # The producer's failed wildcard scan is a separate terminal failure
        # and has precedence over any later transfer/verification diagnostic.
        error = (
            RuntimeError(materialization_error)
            if materialization_error else transfer_error
        )
        return manifest, error

    @staticmethod
    def update_availability(record: RunRecord, store, path, state, message=None):
        """Update one file's availability using the latest durable record."""
        state = FileAvailabilityState(state)

        def update(latest):
            files = latest.files.with_file_availability(path, state.value, message)
            if files is latest.files:
                return latest
            return replace(
                latest,
                updated_at=datetime.now(timezone.utc).isoformat(),
                files=files,
            )

        return store.update(record, update)

    @classmethod
    def retrieve(cls, record: RunRecord, store, executor=None):
        """Restore missing history and reconcile already-published failed files.

        A captured file may be present and hash-correct after a failed
        directory fsync. It still needs a durability retry and an atomic
        availability update, even when no backend retrieval is available.
        """
        if not record.state.is_terminal:
            raise RuntimeError(f"Run {record.id} is not finished")

        versions = {
            os.path.abspath(os.fspath(version.path)): version
            for version in record.files.all_versions
        }
        tracked = {
            os.path.abspath(os.fspath(item.local_path))
            for item in record.files.files if item.local_path is not None
        }
        recoverable = tuple(
            os.path.abspath(os.fspath(entry.local_path))
            for entry in record.files.failed_files
            if entry.local_path is not None
            and (version := versions.get(os.path.abspath(entry.local_path))) is not None
            and version.is_current()
        )
        can_fetch = (
            executor is not None
            and executor.supports_result_file_retrieval
            and record.execution.backend_id is not None
        )
        missing = missing_result_files(record.files) if can_fetch else ()
        # A symlink can point at unchanged historical contents while having
        # replaced the recorded regular file. Inspect it explicitly because
        # FileVersion.state() follows the link and may call it CURRENT.
        linked = tuple(path for path in versions if os.path.islink(path))
        paths = tuple(dict.fromkeys((*missing, *recoverable, *linked)))
        if not paths:
            return record

        for path in paths:
            path = os.path.abspath(os.fspath(path))
            if has_file_magic(path):
                raise ValueError(
                    f"Cannot restore unresolved historical wildcard: {path}"
                )
            lock_path = Path(path).with_name(f".{Path(path).name}.restore.lock")
            with FileLock(lock_path):
                version = versions.get(path)
                # The file may disappear while waiting for the lock. Without
                # an executor, leave the recorded failed state untouched.
                if (not can_fetch and not os.path.islink(path)
                        and (version is None or not version.is_current())):
                    continue
                try:
                    restore_historical_file(record, path, version, executor)
                except OSError as exc:
                    if path in tracked:
                        record = cls.update_availability(
                            record, store, path, FileAvailabilityState.FAILED, str(exc),
                        )
                    raise
                if path in tracked:
                    record = cls.update_availability(
                        record, store, path, FileAvailabilityState.AVAILABLE,
                    )
        return store.get(record.id)


class JobFilesOwner:
    """Expose file-state operations without leaking the mutable JobFiles owner."""

    def _initialize_job_files(self, manifest: FileManifest):
        self._job_files = JobFiles(manifest)

    def file_snapshot(self) -> FileManifest:
        return self._job_files.snapshot()

    def replace_file_manifest(self, manifest: FileManifest) -> FileManifest:
        return self._job_files.replace(manifest)

    def merge_file_manifest(self, observed: FileManifest) -> FileManifest:
        return self._job_files.merge(observed)
