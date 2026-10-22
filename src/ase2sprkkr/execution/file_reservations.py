"""Pure file-access conflict rules for atomic submission reservations.

Only RunStore owns the reservation transaction. This module derives transient
read/write sets from the frozen manifest and compares them; it never persists
or mutates a run, touches a backend, or reserves anything by itself.
"""

from __future__ import annotations

import fnmatch
import os

from .transport import has_file_magic


def _accesses(record):
    """Resolve declared read/write targets in the execution namespace."""
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
        return os.path.normpath(resolved) if remote else os.path.realpath(resolved)

    inputs = {path(entry) for entry in files.input_entries} - {None}
    caches = {path(entry) for entry in files.caches} - {None}
    reads = inputs | caches
    writes = {path(entry) for entry in (files.stdout, files.stderr, *files.outputs)} - {None}
    writes.update(caches)
    # A staged input writes to its remote target unless input upload is disabled.
    if remote and (files.mapping is None or files.mapping.input_sync != "never"):
        writes.update(inputs)
    return reads, writes


def _paths_overlap(first, second, *, check_hardlinks=False):
    """Conservatively intersect exact targets, inode aliases and glob patterns."""
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


def reservation_conflict(first, second):
    """Check whether two runs may safely share an endpoint's filesystem."""
    a, b = first.execution, second.execution
    if a.endpoint != b.endpoint:
        return False
    local = a.directory.remote is None and b.directory.remote is None
    same_directory = (
        os.path.realpath(a.directory.target) == os.path.realpath(b.directory.target)
        if local else a.directory.target == b.directory.target
    )
    if same_directory and (first.task == "SCF" or second.task == "SCF"):
        return True

    first_reads, first_writes = _accesses(first)
    second_reads, second_writes = _accesses(second)
    if (_paths_overlap(first_writes, second_reads | second_writes, check_hardlinks=local)
            or _paths_overlap(second_writes, first_reads, check_hardlinks=local)):
        return True
    return (
        (not first_writes or not second_writes)
        and same_directory
        and first.task == second.task
        and first.dataset == second.dataset
    )
