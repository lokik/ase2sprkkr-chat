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

    def conflicts_with(self, other):
        """Check file access conflicts with another run."""
        return submission_conflicts(self.record, other.record)

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


    @staticmethod
    async def collect_remote(manifest, handler, outcome, spec, *, required):
        """Coordinate per-file download and stream verification for a run.

        The handler owns I/O. The returned manifest is the sole file state;
        successfully delivered files survive a later transfer failure.
        """
        manifest, error = await handler.collect(
            manifest.mapping, manifest, outcome, required=required,
        )
        if error is not None or not required or manifest is None:
            return manifest, error
        return await handler.verify_stream_availability(manifest, spec)

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
        resolved = value if os.path.isabs(value) else os.path.join(directory.target, value)
        # Resolve aliases only on the host whose filesystem is accessible.
        return os.path.normpath(resolved) if remote else os.path.realpath(resolved)

    inputs = {path(entry) for entry in files.input_entries} - {None}
    caches = {path(entry) for entry in files.caches} - {None}
    reads = inputs | caches
    writes = {path(entry) for entry in (files.stdout, files.stderr, *files.outputs)} - {None}
    writes.update(caches)
    if remote and (files.mapping is None or files.mapping.input_sync != "never"):
        writes.update(inputs)
    return reads, writes


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


def submission_conflicts(left, right) -> bool:
    """Conservatively exclude overlapping declared read/write operations."""
    a, b = left.execution, right.execution
    if a.endpoint != b.endpoint:
        return False
    local = a.directory.remote is None and b.directory.remote is None
    same_directory = (
        os.path.realpath(a.directory.target) == os.path.realpath(b.directory.target)
        if local else a.directory.target == b.directory.target
    )
    # SCF can overwrite undeclared canonical files in its working directory.
    if same_directory and (left.task == "SCF" or right.task == "SCF"):
        return True
    left_reads, left_writes = _manifest_accesses(left)
    right_reads, right_writes = _manifest_accesses(right)
    if (_paths_overlap(left_writes, right_reads | right_writes, check_hardlinks=local)
            or _paths_overlap(right_writes, left_reads, check_hardlinks=local)):
        return True
    # Without declared writes, preserve namespace-based protection for generic
    # executors that cannot describe their output footprint.
    return (
        (not left_writes or not right_writes)
        and same_directory
        and left.task == right.task
        and left.dataset == right.dataset
    )

