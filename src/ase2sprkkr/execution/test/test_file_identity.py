"""One file has one declaration, except explicitly merged output streams."""

from dataclasses import asdict

import pytest

from ase2sprkkr.execution.run_model import FileEntry, FileManifest


@pytest.mark.parametrize("roles", ["output-cache", "different-output-sources"])
def test_distinct_files_cannot_target_one_local_file(roles):
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


def test_shared_stream_identity_remains_one_file():
    path = "/local/shared.log"
    files = FileManifest(
        stdout=FileEntry(path, "/remote/shared", "pending"),
        stderr=FileEntry(path, "/remote/shared", "pending"),
    )
    updated = files.with_file_availability(path, "available")
    assert len(updated.files) == 1
    assert updated.stdout.availability_state == "available"
    assert updated.stderr.availability_state == "available"
    assert FileManifest(**asdict(updated)) == updated


@pytest.mark.parametrize("role", ("stdout", "stderr"))
def test_stream_and_output_same_path_rejected(role):
    stream = FileEntry("/local/shared.log", "/remote/shared.log", "pending")
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(**{role: stream}, outputs=(stream,))


@pytest.mark.parametrize("role", ("stdout", "stderr"))
def test_stream_and_output_wildcard_overlap_rejected(role):
    stream = FileEntry("/local/run.log", "/remote/run.log", "pending")
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(**{role: stream}, outputs=(
            FileEntry("/local/*.log", "/remote/*.log", "pending"),
        ))


def test_stdin_and_auxiliary_input_rejected():
    path = "/local/main.inp"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(inputs=(FileEntry(path),), stdin=FileEntry(path))


def test_primary_input_may_feed_stdin():
    entry = FileEntry("/local/main.inp")
    files = FileManifest(inputs=(entry,), primary_input=entry, stdin=entry)
    assert files.input_files == (entry.local_path,)


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
        files.with_added_file(
            FileEntry(path, "/remote/result", "available"), kind="output",
        )


def test_availability_update_cannot_register_unknown_file():
    files = FileManifest(outputs=(
        FileEntry("/local/known", "/remote/known", "pending"),
    ))
    with pytest.raises(KeyError, match="not declared"):
        files.with_file_availability("/local/new", "available", kind="output")
    assert len(files.outputs) == 1


def test_explicit_registration_shares_identity_update_logic():
    files = FileManifest()
    entry = FileEntry("/local/new", "/remote/new", "available",
                      source="/local/*.dat")
    registered = files.with_added_file(entry)
    assert registered.outputs == (entry,)
    updated = registered.with_file_availability(
        entry.local_path, "failed", "download failed", remote_path=entry.remote_path,
    )
    assert updated.outputs[0].availability_state == "failed"
    assert updated.outputs[0].source == entry.source
