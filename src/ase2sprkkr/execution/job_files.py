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
from .run_model import FileAvailabilityState, FileAvailabilityUpdate, FileManifest
from .transport import FILE_DELIVERY_ERRORS, has_file_magic
from .historical_restore import (
    preflight_historical_destination, restore_historical_file,
)

from .file_history import file_is_current, missing_result_files


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

    def snapshot(self) -> FileManifest:
        """Return the immutable current value without copying it."""
        with self._lock:
            return self._manifest

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

    def apply_updates(self, updates):
        """Atomically install a batch of file availability changes.

        A failed validation leaves both the working state and every previously
        issued snapshot unchanged.
        """
        with self._lock:
            self._manifest = self._manifest.with_file_availabilities(updates)
            return self._manifest

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

    async def collect_remote(
        self, frozen, handler, outcome, spec, *,
        required, materialization_error=None,
    ):
        """Collect terminal files into this job's mutable working state.

        The handler only transfers and verifies bytes. This owner installs
        terminal membership first, then publishes each successful phase.
        Files already delivered survive failures in later phases.
        """
        with self._lock:
            self._manifest = self.merge_terminal_files(self._manifest, frozen)
            manifest = self._manifest

        transfer_error = None
        verify = True
        try:
            observed, transfer_error = await handler.collect(
                manifest.mapping, manifest, outcome, required=required,
            )
        except FILE_DELIVERY_ERRORS as exc:
            transfer_error = exc
        else:
            if observed is None:
                verify = False
            else:
                self.merge(observed)

        if required and verify:
            try:
                observed, stream_error = await handler.verify_stream_availability(
                    self.snapshot(), spec,
                )
            except FILE_DELIVERY_ERRORS as exc:
                stream_error = exc
            else:
                if observed is not None:
                    self.merge(observed)
            if transfer_error is None:
                transfer_error = stream_error

        return (
            RuntimeError(materialization_error)
            if materialization_error else transfer_error
        )

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

        # Preflight the entire historical batch before touching any file.
        # The per-file operation repeats these checks after acquiring its
        # lock, protecting against changes between validation and publication.
        for path in paths:
            if has_file_magic(path):
                raise ValueError(
                    f"Cannot restore unresolved historical wildcard: {path}"
                )
            preflight_historical_destination(path, versions.get(path))
        if can_fetch:
            executor.preflight_result_files(record, paths)

        for path in paths:
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


