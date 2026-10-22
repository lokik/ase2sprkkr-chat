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


def test_stdin_and_auxiliary_input_wildcard_overlap_rejected():
    primary = FileEntry("/local/main.inp")
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            inputs=(primary, FileEntry("/local/*.inp", optional=True)),
            primary_input=primary,
            stdin=primary,
        )


def test_materialized_output_cannot_duplicate_stdout():
    stream = FileEntry("/local/run.log", "/remote/run.log", "pending")
    expanded = FileEntry(
        "/local/run.log", "/remote/run.log", "pending",
        source="/local/*.log",
    )
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(stdout=stream, outputs=(expanded,))


@pytest.mark.parametrize("role", ("stdout", "stderr"))
@pytest.mark.parametrize(
    ("stream_local", "stream_remote", "output_local", "output_remote"),
    (
        ("/local/shared", "/remote/stream", "/local/shared", "/remote/output"),
        ("/local/stream", "/remote/shared", "/local/output", "/remote/shared"),
        ("/local/shared", "/remote/shared", "/local/shared", "/remote/shared"),
    ),
)
def test_stream_and_output_cannot_share_either_local_or_remote_identity(
    role, stream_local, stream_remote, output_local, output_remote,
):
    """Stream/output conflicts are errors even when only one side collides."""
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            **{role: FileEntry(stream_local, stream_remote)},
            outputs=(FileEntry(output_local, output_remote),),
        )


@pytest.mark.parametrize("role", ("stdout", "stderr"))
@pytest.mark.parametrize(
    ("stream_remote", "output_remote"),
    (
        ("/remote/run.log", "/remote/*.log"),
        ("/remote/*.log", "/remote/run.log"),
        ("/remote/run.log", "/remote/run.*"),
    ),
)
def test_stream_output_remote_wildcard_overlap_rejected(
    role, stream_remote, output_remote,
):
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            **{role: FileEntry("/local/stream", stream_remote)},
            outputs=(FileEntry("/local/other", output_remote),),
        )


def test_manifest_rejects_remote_path_outside_shared_mapping():
    from ase2sprkkr.execution.run_model import FileMapping

    mapping = FileMapping("/local/project", "/remote/project")
    files = FileManifest(
        stdout=FileEntry("/local/project/stdout.log", "/remote/elsewhere/stdout.log"),
    )
    with pytest.raises(ValueError, match="Submission path differs from reserved path"):
        files.with_mapping(mapping)


def test_manifest_accepts_remote_path_derived_from_shared_mapping():
    from ase2sprkkr.execution.run_model import FileMapping

    mapping = FileMapping("/local/project", "/remote/project")
    files = FileManifest(
        outputs=(FileEntry("/local/project/out.dat"),),
    ).with_mapping(mapping)
    assert files.outputs[0].remote_path == mapping.map_path(files.outputs[0].local_path)


def test_recovered_stream_path_cannot_escape_shared_mapping():
    from ase2sprkkr.execution.run_model import FileMapping

    mapping = FileMapping("/local/project", "/remote/project")
    files = FileManifest(
        mapping=mapping,
        stdout=FileEntry("/local/project/stdout.log", "/remote/project/stdout.log"),
    )
    with pytest.raises(ValueError, match="does not match the file mapping"):
        files.with_remote_stream_paths(stdout="/remote/elsewhere/stdout.log")
