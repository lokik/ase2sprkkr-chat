"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path
from uuid import uuid4
from dataclasses import replace
from datetime import datetime, timezone

from ..common.file_versions import file_digest
from .run_model import FileAvailabilityState
from .transport import has_file_magic

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

    def _accesses(self):
        """Resolve this job's declared filesystem read/write accesses."""
        files = self.manifest
        directory = self.record.execution.directory
        remote = directory.remote is not None

        def path(entry):
            if entry is None:
                return None
            value = entry.remote_path if remote else entry.local_path
            if value is None:
                return None
            resolved = value if os.path.isabs(value) else os.path.join(directory.target, value)
            return os.path.normpath(resolved) if remote else os.path.realpath(resolved)

        inputs = {path(entry) for entry in files.input_entries} - {None}
        caches = {path(entry) for entry in files.caches} - {None}
        reads = inputs | caches
        writes = {path(entry) for entry in (files.stdout, files.stderr, *files.outputs)} - {None}
        writes.update(caches)
        if remote and (files.mapping is None or files.mapping.input_sync != "never"):
            writes.update(inputs)
        return reads, writes

    def conflicts_with(self, other):
        """Conservatively reject conflicting filesystem operations."""
        left, right = self.record, other.record
        a, b = left.execution, right.execution
        if a.endpoint != b.endpoint:
            return False
        local = a.directory.remote is None and b.directory.remote is None
        same_directory = (
            os.path.realpath(a.directory.target) == os.path.realpath(b.directory.target)
            if local else a.directory.target == b.directory.target
        )
        if same_directory and (left.task == "SCF" or right.task == "SCF"):
            return True
        left_reads, left_writes = self._accesses()
        right_reads, right_writes = other._accesses()
        if (_paths_overlap(left_writes, right_reads | right_writes, check_hardlinks=local)
                or _paths_overlap(right_writes, left_reads, check_hardlinks=local)):
            return True
        return (
            (not left_writes or not right_writes)
            and same_directory
            and left.task == right.task
            and left.dataset == right.dataset
        )

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
    def _merge_availability(previous, received, preserve):
        """Reconcile availability for matching file identities and roles once.

        The prior manifest is evidence, never a source of new members. Output
        and cache identities remain separate even when their paths coincide.
        """
        recorded = {
            (role, entry.local_path, entry.remote_path): entry
            for role, entries in (
                ("output", (previous.stdout, previous.stderr, *previous.outputs)),
                ("cache", previous.caches),
            )
            for entry in entries if entry is not None
        }

        def merged(entry, role):
            if entry is None:
                return None
            prior = recorded.get((role, entry.local_path, entry.remote_path))
            if prior is None or not preserve(prior, entry):
                return entry
            return replace(
                entry, availability_state=prior.availability_state,
                availability_message=prior.availability_message,
            )

        return replace(
            received,
            stdout=merged(received.stdout, "output"),
            stderr=merged(received.stderr, "output"),
            outputs=tuple(merged(entry, "output") for entry in received.outputs),
            caches=tuple(merged(entry, "cache") for entry in received.caches),
        )

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
                and entry.availability_state == "available"
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
        try:
            manifest, error = await handler.collect(
                manifest.mapping, manifest, outcome, required=required,
            )
        except Exception as exc:
            # A preflight failure precedes any transfer. Keep the exact
            # manifest and still assess independently delivered FOLLOW streams.
            error = exc
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
        # Captured input versions are recoverable too, but inputs have no
        # availability field: only output/cache entries may be updated through
        # FileManifest.with_file_availability().
        tracked = {
            os.path.abspath(os.fspath(item.local_path))
            for item in record.files.files if item.local_path is not None
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
                if path in tracked:
                    self.update_availability(
                        store, path, FileAvailabilityState.FAILED, str(exc),
                    )
                raise
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
            refreshed = store.get(record.id)
            still_missing = path in missing_result_files(refreshed.files)
            if path not in tracked:
                # There is no per-input availability record to update. A
                # failed restore must be reported, not turned into a KeyError
                # from the output/cache availability API.
                if still_missing:
                    raise FileNotFoundError(
                        f"Historical input file is still missing: {path}"
                    )
                record = refreshed
                continue
            record = JobFiles(refreshed).update_availability(
                store, path,
                FileAvailabilityState.FAILED if still_missing
                else FileAvailabilityState.AVAILABLE,
                "Historical file is still missing" if still_missing else None,
            )
        return record


def _paths_overlap(first, second, *, check_hardlinks=False):
    """Conservatively detect overlapping declared paths and glob patterns."""
    def prefix(pattern):
        return pattern[:min((pattern.find(c) for c in "*?[" if c in pattern),
                            default=len(pattern))]

    for a in first:
        for b in second:
            if a == b:
                return True
            a_magic, b_magic = has_file_magic(a), has_file_magic(b)
            if not a_magic and not b_magic:
                if check_hardlinks:
                    try:
                        if os.path.samefile(a, b):
                            return True
                    except (FileNotFoundError, NotADirectoryError):
                        pass
            elif not a_magic:
                if fnmatch.fnmatchcase(a, b):
                    return True
            elif not b_magic:
                if fnmatch.fnmatchcase(b, a):
                    return True
            elif (prefix(a).startswith(prefix(b))
                  or prefix(b).startswith(prefix(a))):
                # Intersecting literal prefixes cannot rule out overlap.
                return True
    return False


