"""Determine whether two jobs require conflicting filesystem access."""

from __future__ import annotations

import fnmatch
import os

from .transport import has_file_magic


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


def submission_conflicts(left: RunRecord, right: RunRecord) -> bool:
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

