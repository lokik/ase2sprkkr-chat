"""Immutable local file snapshots and historical file integrity.

Pure file operations live here; run lifecycle and durable RunRecord updates
remain the responsibility of ExecutionManager.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone

from ..common.file_fingerprints import file_fingerprint
from ..common.file_versions import FileState, FileVersion
from ._durability import sync_file
from .transport import expand_local_candidates, has_file_magic
from .file_availability import discover_local_files


def concrete_result_files(manifests):
    paths = []
    seen = set()
    for manifest in manifests:
        for path in expand_local_candidates(manifest):
            path = os.path.abspath(path)
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def snapshot_result_files(manifests, *, sync=False, verified_versions=()):
    """Capture stable history, reusing delivery hashes only for unchanged files."""
    verified = {
        os.fspath(version.path): version for version in verified_versions
        if version is not None and version.available
    }
    versions = []
    for path in concrete_result_files(manifests):
        if sync:
            sync_file(path)
        previous = verified.get(path)
        if (previous is not None
                and file_fingerprint(path) == previous.fingerprint):
            # The delivery already computed a stable SHA-256 for these bytes.
            versions.append(previous)
        else:
            versions.append(FileVersion.capture(path, require_digest=True))
    return tuple(sorted(versions, key=lambda value: os.fspath(value.path)))


def sync_result_files(manifests):
    for path in concrete_result_files(manifests):
        sync_file(path)


def file_is_current(result_files, path, versions):
    if result_files is None or path is None:
        return False
    path = os.path.abspath(path)
    if result_files.captured_at is None:
        return os.path.isfile(path)
    for version in versions:
        if os.path.abspath(os.fspath(version.path)) == path:
            return version.is_current()
    return False


def missing_result_files(result_files):
    if result_files is None:
        return ()
    if result_files.captured_at is not None:
        return tuple(
            os.fspath(item.path)
            for item in result_files.all_versions
            if item.state() is FileState.MISSING
        )

    missing = []
    for manifest in result_files.manifests:
        if has_file_magic(manifest):
            if not any(os.path.isfile(path) for path in expand_local_candidates(manifest)):
                missing.append(manifest)
        elif not os.path.isfile(manifest):
            missing.append(manifest)
    return tuple(dict.fromkeys(value for value in missing if value is not None))


def capture_result_files(result_files):
    if result_files is None or result_files.captured_at is not None:
        return result_files
    verified_versions = (
        entry.verified_local_version for entry in result_files.files
        if entry.availability_state == "available"
        and entry.verified_local_version is not None
    )
    captured = replace(
        result_files,
        input_versions=snapshot_result_files(result_files.input_files),
        output_versions=snapshot_result_files(
            result_files.output_files, sync=True,
            verified_versions=verified_versions,
        ),
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
    return discover_local_files(captured)
