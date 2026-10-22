"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
from threading import Lock
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
        self._lock = Lock()

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

    async def collect_remote(
        self, frozen, handler, outcome, spec, *,
        required, materialization_error=None,
    ):
        """Collect terminal files into this job's mutable working state.

        The frozen membership is authoritative. Transfer and stream
        verification observations are installed independently so a later
        failure cannot discard a successful earlier delivery.
        """
        with self._lock:
            self._manifest = merge_terminal_files(self._manifest, frozen)
            manifest = self._manifest

        try:
            observed, transfer_error = await handler.collect(
                manifest.mapping, manifest, outcome, required=required,
            )
        except FILE_DELIVERY_ERRORS as exc:
            # A preflight failure cannot revoke independently delivered
            # files. Still verify FOLLOW streams for successful jobs.
            transfer_error = exc
        else:
            if observed is not None:
                merged = self.merge(observed)
                # An unrelated failed attempt may race with a successfully
                # verified download. Only suppress the transfer error if
                # every reported failure was superseded by intact delivery.
                if (transfer_error is not None and observed.failed_files
                        and not merged.failed_files):
                    transfer_error = None
            else:
                manifest = None

        if required and manifest is not None:
            try:
                observed, stream_error = await handler.verify_stream_availability(
                    self.manifest, spec,
                )
            except FILE_DELIVERY_ERRORS as exc:
                stream_error = exc
            else:
                if observed is not None:
                    self.merge(observed)
            if transfer_error is None:
                transfer_error = stream_error

        # Wildcard materialization is an independent backend-side failure
        # that precedes transfer/stream diagnostics.
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


