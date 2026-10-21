"""One file has one declaration, except explicitly merged output streams."""

from dataclasses import asdict

import pytest

from ase2sprkkr.execution.run_model import FileEntry, FileManifest


@pytest.mark.parametrize("roles", ["output-cache", "different-output-sources"])
def test_distinct_artifacts_cannot_target_one_local_file(roles):
    path = "/local/shared.dat"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        if roles == "output-cache":
            FileManifest(
                outputs=(FileEntry(path, "/remote/result.dat"),),
                caches=(FileEntry(path, "/remote/cache.dat"),),
            )
        else:
            FileManifest(outputs=(
                FileEntry(path, "/remote/first.dat"),
                FileEntry(path, "/remote/second.dat"),
            ))


def test_merged_streams_require_same_remote_destination():
    with pytest.raises(ValueError, match="Merged stdout/stderr"):
        FileManifest(
            stdout=FileEntry("/local/merged", "/remote/out"),
            stderr=FileEntry("/local/merged", "/remote/err"),
        )


def test_shared_stream_identity_remains_one_artifact():
    path = "/local/shared.log"
    files = FileManifest(
        stdout=FileEntry(path, "/remote/shared", "pending"),
        stderr=FileEntry(path, "/remote/shared", "pending"),
    )
    updated = files.with_file_availability(path, "available")
    assert len(updated.artifacts) == 1
    assert updated.stdout.availability_state == "available"
    assert updated.stderr.availability_state == "available"
    assert FileManifest(**asdict(updated)) == updated


def test_output_alias_of_merged_stream_is_rejected():
    path = "/local/shared.log"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            stdout=FileEntry(path, "/remote/shared"),
            stderr=FileEntry(path, "/remote/shared"),
            outputs=(FileEntry(path, "/remote/shared"),),
        )


def test_cache_only_path_updates_without_explicit_kind():
    cache = FileEntry("/local/cache.dat", "/remote/cache.dat", "pending")
    files = FileManifest(caches=(cache,))
    updated = files.with_file_availability(
        cache.local_path, "failed", "download failed",
    )
    assert updated.outputs == ()
    assert updated.caches[0].availability_state == "failed"
    assert updated.caches[0].availability_message == "download failed"


def test_adding_output_for_cache_destination_is_rejected():
    path = "/local/shared"
    files = FileManifest(caches=(FileEntry(path, "/remote/cache", "pending"),))
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        files.with_file_availability(
            path, "available", remote_path="/remote/result", kind="output",
        )
