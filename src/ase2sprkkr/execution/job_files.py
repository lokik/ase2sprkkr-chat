"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4
from dataclasses import replace
from datetime import datetime, timezone

from ..common.file_versions import file_digest
from .run_model import FileAvailabilityState

from .file_history import (
    capture_result_files, file_is_current, missing_result_files, sync_result_files,
)


class HistoricalFileMismatchError(OSError):
    """Restored file contents differ from the captured digest."""


class JobFiles:
    """Operate on one run's manifest without duplicating its persistent state."""

    def __init__(self, record):
        self.record = record

    @property
    def manifest(self):
        return self.record.files

    def input_file_is_current(self):
        files = self.manifest
        return file_is_current(files, files.input_file, files.input_versions)

    def output_file_is_current(self):
        files = self.manifest
        return file_is_current(files, files.output_file, files.output_versions)

    def sync_inputs(self):
        sync_result_files(self.manifest.input_files)

    def capture(self, manifest=None):
        return capture_result_files(self.manifest if manifest is None else manifest)

    def missing_files(self):
        return missing_result_files(self.manifest)

    @staticmethod
    def merge_snapshot(current, requested):
        """Preserve captured historical file evidence on stale updates."""
        from .run_model import FileManifest

        if isinstance(requested, dict):
            requested = FileManifest(**requested)
        if current is None:
            return requested
        if current.captured_at is not None:
            return current
        return requested

    @staticmethod
    def mark_pending_failed(manifest, message):
        return manifest._map_files(
            lambda item: replace(
                item, availability_state="failed", availability_message=message,
            ) if item.availability_state in (None, "pending") else item
        )

    @staticmethod
    def merge_terminal_files(previous, received):
        """Preserve completed local deliveries in frozen terminal facts."""
        if previous is None:
            return received
        delivered = {
            (entry.local_path, entry.remote_path): entry
            for entry in previous.files
        }

        def preserve(entry):
            if entry is None or entry.availability_state in ("failed", "skipped"):
                return entry
            prior = delivered.get((entry.local_path, entry.remote_path))
            if prior is None or prior.availability_state not in ("available", "skipped"):
                return entry
            if (prior.availability_state == "available"
                    and (entry.local_path is None
                         or not os.path.isfile(entry.local_path))):
                return entry
            return replace(
                entry, availability_state=prior.availability_state,
                availability_message=prior.availability_message,
            )

        return received._map_files(preserve)


    @classmethod
    async def collect_terminal_remote(
        cls, previous, frozen, handler, outcome, spec, *,
        required, materialization_error=None,
    ):
        """Reconcile frozen membership and collect; retain partial delivery."""
        manifest = cls.merge_terminal_files(previous, frozen)
        manifest, transfer_error = await cls.collect_remote(
            manifest, handler, outcome, spec, required=required,
        )
        error = (
            RuntimeError(materialization_error)
            if materialization_error else transfer_error
        )
        return manifest, error

    @staticmethod
    async def collect_remote(manifest, handler, outcome, spec, *, required):
        """Coordinate per-file download and stream verification for a run.

        The handler owns I/O. The returned manifest is the sole file state;
        successfully delivered files survive a later transfer failure.
        """
        manifest, error = await handler.collect(
            manifest.mapping, manifest, outcome, required=required,
        )
        if not required or manifest is None:
            return manifest, error
        # A failed output/cache transfer does not imply failed FOLLOW streams.
        # Verify them independently, retaining the first transfer error.
        manifest, stream_error = await handler.verify_stream_availability(
            manifest, spec,
        )
        return manifest, error if error is not None else stream_error

    def retry_failed_downloads(self):
        return self.manifest._map_files(
            lambda item: replace(
                item, availability_state="pending", availability_message=None,
            ) if item.availability_state == "failed" else item
        )

    def update_availability(self, store, path, state, message=None):
        """Update one file's availability using the latest durable record."""
        state = FileAvailabilityState(state)

        def update(latest):
            return replace(
                latest,
                updated_at=datetime.now(timezone.utc).isoformat(),
                files=latest.files.with_file_availability(path, state.value, message),
            )

        return store.update(self.record, update)

    def retrieve(self, store, executor=None):
        """Restore historical files without modifying the terminal run outcome."""
        record = self.record
        if not record.state.is_terminal:
            raise RuntimeError(f"Run {record.id} is not finished")
        missing = self.missing_files()
        if not missing or record.execution.backend_id is None:
            return record
        if executor is None or not executor.supports_result_file_retrieval:
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
                self.update_availability(
                    store, path, FileAvailabilityState.FAILED, str(exc),
                )
                raise
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
            refreshed = store.get(record.id)
            availability = (
                FileAvailabilityState.FAILED
                if path in missing_result_files(refreshed.files)
                else FileAvailabilityState.AVAILABLE
            )
            record = JobFiles(refreshed).update_availability(
                store, path, availability,
                "Historical file is still missing"
                if availability == FileAvailabilityState.FAILED else None,
            )
        return record
