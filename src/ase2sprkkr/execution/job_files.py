"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

import os
import fnmatch
from pathlib import Path
from uuid import uuid4
from dataclasses import replace
from datetime import datetime, timezone

from ..common.file_locks import FileLock
from ._durability import durable_publish_new, sync_file
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
        """Reconcile matching files without altering the received membership.

        FileManifest already rejects output/cache identity collisions.
        """
        # Remote-only cache baselines are absent from the availability summary.
        recorded = {
            (entry.local_path, entry.remote_path): entry
            for entry in (previous.stdout, previous.stderr,
                          *previous.outputs, *previous.caches)
            if entry is not None
        }

        def merged(entry):
            if entry is None:
                return None
            prior = recorded.get((entry.local_path, entry.remote_path))
            if prior is None or not preserve(prior, entry):
                return entry
            return replace(
                entry, availability_state=prior.availability_state,
                availability_message=prior.availability_message,
            )

        return received._map_files(merged)

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
        try:
            manifest, stream_error = await handler.verify_stream_availability(
                manifest, spec,
            )
        except Exception as exc:
            # Keep the output/cache manifest if stream verification itself fails.
            stream_error = exc
        return manifest, error if error is not None else stream_error

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
            # A historical restore must have a concrete, frozen file identity.
            # Expanding a wildcard now would invent members absent at capture.
            if has_file_magic(path):
                raise ValueError(
                    f"Cannot restore unresolved historical wildcard: {path}"
                )
            # Serialize restorers of the same concrete path, without holding
            # a RunStore lock during network transfer.
            lock_path = Path(path).with_name(f".{Path(path).name}.restore.lock")
            with FileLock(lock_path):
                version = versions.get(path)
                staged = Path(path).with_name(
                    f".{Path(path).name}.{uuid4().hex}.tmp"
                )
                try:
                    # A previous restorer might already have completed while
                    # this caller was waiting for the filesystem lock.
                    if version is not None and version.is_current():
                        pass
                    elif version is None and os.path.isfile(path):
                        pass
                    elif version is not None and os.path.lexists(path):
                        raise HistoricalFileMismatchError(
                            f"Historical result file was modified: {path}"
                        )
                    else:
                        executor.fetch_result_files(
                            record, (path,),
                            destinations={path: os.fspath(staged)},
                        )
                        if staged.is_symlink():
                            raise OSError(
                                f"Historical restore supplied a symlink: {staged}"
                            )
                        if not staged.is_file():
                            raise FileNotFoundError(
                                f"Historical file was not retrieved: {path}"
                            )
                        if (version is not None and version.digest is not None
                                and file_digest(staged) != version.digest):
                            raise HistoricalFileMismatchError(
                                f"Historical result files have been overwritten: {path}"
                            )
                        # Do not publish partial files or record availability
                        # before both file contents and its directory entry
                        # have crossed the durability barrier.
                        sync_file(staged)
                        try:
                            durable_publish_new(staged, path)
                        except FileExistsError:
                            # An unrelated writer published the path without
                            # participating in our advisory restore lock.
                            # Never overwrite it even when its bytes match.
                            if version is None or not version.is_current():
                                raise HistoricalFileMismatchError(
                                    f"Historical destination appeared during retrieval: {path}"
                                )
                except OSError as exc:
                    if path in tracked:
                        JobFiles(store.get(record.id)).update_availability(
                            store, path, FileAvailabilityState.FAILED, str(exc),
                        )
                    raise
                finally:
                    staged.unlink(missing_ok=True)

                # The captured version is immutable. Checking only this
                # path avoids re-reading the run and scanning all versions for
                # every downloaded file. RunStore.update reads the latest
                # record atomically when changing availability.
                error = None
                if not os.path.isfile(path):
                    error = FileNotFoundError(
                        f"Historical file is still missing after retrieval: {path}"
                    )
                elif version is not None and not version.is_current():
                    error = HistoricalFileMismatchError(
                        f"Historical file changed during retrieval: {path}"
                    )
                if path in tracked:
                    record = self.update_availability(
                        store, path,
                        FileAvailabilityState.FAILED if error
                        else FileAvailabilityState.AVAILABLE,
                        str(error) if error else None,
                    )
                if error is not None:
                    raise error
        # Inputs have no availability field; only their frozen versions are
        # checked. Return the latest run without a redundant per-input read.
        return store.get(record.id)


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


