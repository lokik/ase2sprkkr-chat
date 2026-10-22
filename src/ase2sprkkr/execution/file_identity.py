"""One local filesystem identity rule for file declarations and reservations.

Lexical/symlink aliases count as the same destination even when the file does
not yet exist. Existing hardlinks additionally share an inode. This is
deliberately distinct from remote POSIX path identity: the client cannot stat
the endpoint filesystem when constructing a file reservation.
"""

from __future__ import annotations

import os


def same_local_file(first, second):
    """Compare two local paths without requiring them to exist yet."""
    if first is None or second is None:
        return False
    a, b = os.fspath(first), os.fspath(second)
    if os.path.realpath(a) == os.path.realpath(b):
        return True
    try:
        return os.path.samefile(a, b)
    except (FileNotFoundError, NotADirectoryError):
        return False
