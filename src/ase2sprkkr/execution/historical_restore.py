"""Verified historical retrieval and atomic RunRecord availability updates.

The caller holds the destination's restore lock until the observation is
recorded in RunStore. No backend I/O occurs under a RunStore lock.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from ..common.file_locks import FileLock
from ..common.file_versions import file_digest
from ._durability import durable_publish_new, sync_file
from .file_history import missing_result_files
from .run_model import FileAvailabilityState
from .transport import has_file_magic

if TYPE_CHECKING:
    from .run_record import RunRecord


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


def update_historical_availability(record: RunRecord, store, path, state, message=None):
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

def retrieve_historical_files(record: RunRecord, store, executor=None):
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
                    record = update_historical_availability(
                        record, store, path, FileAvailabilityState.FAILED, str(exc),
                    )
                raise
            if path in tracked:
                record = update_historical_availability(
                    record, store, path, FileAvailabilityState.AVAILABLE,
                )
    return store.get(record.id)
