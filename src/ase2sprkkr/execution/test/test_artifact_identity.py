"""Regression tests: artifact availability belongs to a role and source."""

from dataclasses import asdict

import pytest

from ase2sprkkr.execution.run_model import FileEntry, FileManifest


def test_output_and_cache_sharing_local_destination_have_independent_states():
    path = "/local/shared.dat"
    manifest = FileManifest(
        outputs=(FileEntry(path, "/remote/result.dat", "pending"),),
        caches=(FileEntry(path, "/remote/cache.dat", "pending"),),
    )
    assert len(manifest.artifacts) == 2

    with pytest.raises(ValueError, match="Ambiguous artifact destination"):
        manifest.with_file_availability(path, "available")

    changed = manifest.with_file_availability(
        path, "failed", "cache transfer failed", kind="cache",
    )
    assert changed.outputs[0].availability_state == "pending"
    assert changed.caches[0].availability_state == "failed"
    assert changed.failed_artifacts == (changed.caches[0],)
    assert changed.availability_state == "failed"
    assert changed.availability_message == "cache transfer failed"
    assert FileManifest(**asdict(changed)) == changed


def test_distinct_remote_sources_at_same_local_path_require_disambiguation():
    path = "/local/shared.log"
    files = FileManifest(
        stdout=FileEntry(path, "/remote/stdout", "pending"),
        stderr=FileEntry(path, "/remote/stderr", "pending"),
    )
    with pytest.raises(ValueError, match="Ambiguous artifact destination"):
        files.with_file_availability(path, "available")

    updated = files.with_file_availability(
        path, "failed", "stdout mismatch",
        kind="output", remote_path="/remote/stdout",
    )
    assert updated.stdout.availability_state == "failed"
    assert updated.stderr.availability_state == "pending"
    assert updated.failed_artifacts == (updated.stdout,)
    assert updated.availability_state == "failed"


def test_shared_stream_identity_and_duplicate_output_alias_remain_synchronized():
    path = "/local/shared.log"
    files = FileManifest(
        stdout=FileEntry(path, "/remote/shared", "pending"),
        stderr=FileEntry(path, "/remote/shared", "pending"),
        outputs=(FileEntry(path, "/remote/shared", "pending"),),
    )
    updated = files.with_file_availability(path, "available")
    assert len(updated.artifacts) == 1
    assert updated.stdout.availability_state == "available"
    assert updated.stderr.availability_state == "available"
    assert updated.outputs[0].availability_state == "available"
    assert FileManifest(**asdict(updated)) == updated


def test_cache_only_path_updates_without_explicit_kind():
    cache = FileEntry("/local/cache.dat", "/remote/cache.dat", "pending")
    files = FileManifest(caches=(cache,))
    updated = files.with_file_availability(
        cache.local_path, "failed", "download failed",
    )
    assert updated.outputs == ()
    assert updated.caches[0].availability_state == "failed"
    assert updated.caches[0].availability_message == "download failed"


def test_conflicting_duplicate_states_are_not_lost_in_summary():
    path = "/local/shared"
    files = FileManifest(
        stdout=FileEntry(path, "/remote/shared", "failed", "verification failed"),
        stderr=FileEntry(path, "/remote/shared", "available"),
    )
    assert files.availability_state == "failed"
    assert files.failed_artifacts == (files.stdout,)
    assert len(files.artifacts) == 2


def test_distinct_artifact_status_can_be_added_to_existing_other_role():
    path = "/local/shared"
    files = FileManifest(caches=(FileEntry(path, "/remote/cache", "pending"),))
    updated = files.with_file_availability(
        path, "available", remote_path="/remote/result", kind="output",
    )
    assert updated.caches == files.caches
    assert updated.outputs == (FileEntry(path, "/remote/result", "available"),)
