"""Verified historical file retrieval, independent of RunRecord persistence.

The caller holds the destination's restore lock until the observation is
recorded in RunStore. No backend I/O occurs under a RunStore lock.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from ..common.file_versions import file_digest
from ._durability import durable_publish_new, sync_file


class HistoricalFileMismatchError(OSError):
    """Restored file contents differ from the captured digest."""


def preflight_historical_destination(path, version):
    """Reject unsafe local targets before any member of a batch transfers.

    Return whether this file needs a download. The same check is repeated
    under its restore lock because another process may change the destination.
    """
    # FileVersion follows symlinks: a new link to the original contents is
    # nevertheless not the captured regular file.
    if os.path.islink(path):
        raise HistoricalFileMismatchError(
            f"Historical destination is a symlink: {path}"
        )
    if version is not None and version.is_current():
        return False
    if version is None and os.path.isfile(path):
        return False
    if os.path.lexists(path):
        raise HistoricalFileMismatchError(
            f"Historical result file was modified: {path}"
        )
    return True


def restore_historical_file(record, path, version, executor):
    """Fetch, verify and atomically publish one concrete historical file.

    The caller must hold its per-path restore lock throughout this call and
    the following availability update. A transfer is only published when its
    recorded digest matches; never replace any existing destination.
    """
    if not preflight_historical_destination(path, version):
        if version is not None:
            # Retry the durability barrier after a successful prior publish.
            sync_file(path)
            if not version.is_current():
                raise HistoricalFileMismatchError(
                    f"Historical file changed during durability verification: {path}"
                )
        return

    target = Path(path)
    staged = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        executor.fetch_result_files(
            record, (path,), destinations={path: os.fspath(staged)},
        )
        if staged.is_symlink():
            raise OSError(f"Historical restore supplied a symlink: {staged}")
        if not staged.is_file():
            raise FileNotFoundError(
                f"Historical file was not retrieved: {path}"
            )
        if (version is not None and version.digest is not None
                and file_digest(staged) != version.digest):
            raise HistoricalFileMismatchError(
                f"Historical result files have been overwritten: {path}"
            )
        # Flush staged contents before atomic no-overwrite publication.
        sync_file(staged)
        try:
            durable_publish_new(staged, path)
        except FileExistsError:
            # A writer ignoring our advisory lock may have published a file.
            # Accept only the already-captured version; never overwrite it.
            if version is None or not version.is_current():
                raise HistoricalFileMismatchError(
                    f"Historical destination appeared during retrieval: {path}"
                )
    finally:
        staged.unlink(missing_ok=True)

    # A concurrent writer can modify the destination immediately after
    # publication; the receipt must reflect the bytes currently on disk.
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Historical file is still missing after retrieval: {path}"
        )
    if version is not None and not version.is_current():
        raise HistoricalFileMismatchError(
            f"Historical file changed during retrieval: {path}"
        )
