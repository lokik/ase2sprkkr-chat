"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_record import RunRecord

from ..common.file_locks import FileLock
from .run_model import FileAvailabilityState, FileManifest
from .manifest_merge import merge_snapshot, merge_terminal_files
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

    @property
    def manifest(self) -> FileManifest:
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
            self._manifest = merge_snapshot(self._manifest, observed)
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
    def input_file_is_current(record: RunRecord):
        """Check the input file against the immutable historical snapshot."""
        files = record.files
        return file_is_current(files, files.input_file, files.input_versions)

    @staticmethod
    def output_file_is_current(record: RunRecord):
        """Check the primary output against the immutable historical snapshot."""
        files = record.files
        return file_is_current(files, files.output_file, files.output_versions)

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
        manifest = merge_terminal_files(previous, frozen)
        try:
            manifest, transfer_error = await handler.collect(
                manifest.mapping, manifest, outcome, required=required,
            )
        except FILE_DELIVERY_ERRORS as exc:
            # A preflight failure precedes any transfer; still verify FOLLOW
            # streams on successful jobs without changing output membership.
            transfer_error = exc

        if required and manifest is not None:
            try:
                manifest, stream_error = await handler.verify_stream_availability(
                    manifest, spec,
                )
            except FILE_DELIVERY_ERRORS as exc:
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


