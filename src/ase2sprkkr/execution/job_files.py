"""Job-owned file operations; FileManifest remains the single source of truth."""

from __future__ import annotations

from dataclasses import replace
import fnmatch
import os

from .transport import has_file_magic

from .file_history import (
    capture_result_files, file_is_current, missing_result_files, sync_result_files,
)


class JobFiles:
    """Operate on one run's manifest without duplicating its persistent state."""

    def __init__(self, record):
        self.record = record

    @property
    def manifest(self):
        return self.record.files

    def conflicts_with(self, other):
        """Whether the two jobs require conflicting file access."""
        return _files_conflict(self.record, other.record)

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

    def retry_failed_downloads(self):
        return self.manifest._map_files(
            lambda item: replace(
                item, availability_state="pending", availability_message=None,
            ) if item.availability_state == "failed" else item
        )


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


def _path_overlap(a, b, *, local=False):
    """Conservatively compare paths, glob patterns and local hardlinks."""
    a_magic, b_magic = has_file_magic(a), has_file_magic(b)
    if not a_magic and not b_magic:
        if a == b:
            return True
        if local:
            try:
                return os.path.samefile(a, b)
            except (FileNotFoundError, NotADirectoryError):
                pass
        return False
    if not a_magic:
        return fnmatch.fnmatchcase(a, b)
    if not b_magic:
        return fnmatch.fnmatchcase(b, a)
    if a == b:
        return True

    # Only disjoint literal prefixes prove that two glob patterns cannot overlap.
    def prefix(pattern):
        return pattern[:min((pattern.find(c) for c in "*?[" if c in pattern),
                            default=len(pattern))]

    pa, pb = prefix(a), prefix(b)
    return pa.startswith(pb) or pb.startswith(pa)


def _paths_overlap(first, second, *, local=False):
    return any(_path_overlap(a, b, local=local) for a in first for b in second)


def _files_conflict(left, right) -> bool:
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
    if (_paths_overlap(left_writes, right_reads | right_writes, local=local)
            or _paths_overlap(right_writes, left_reads, local=local)):
        return True
    # Without declared writes, preserve namespace-based protection for generic
    # executors that cannot describe their output footprint.
    return (
        (not left_writes or not right_writes)
        and same_directory
        and left.task == right.task
        and left.dataset == right.dataset
    )

