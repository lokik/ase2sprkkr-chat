"""Declaration, launch cleanup and reservation share local file identity."""

import os

import pytest

from ase2sprkkr.execution import (
    RunManager, FileEntry, FileManifest, RunStore,
)
from ase2sprkkr.execution.file_availability import validate_output_collisions
from ase2sprkkr.execution.file_identity import same_local_file
from ase2sprkkr.execution.file_reservations import reservation_conflict


@pytest.mark.parametrize("policy, side, deleted", (
    ("both", "local", True), ("both", "remote", True),
    ("local", "local", True), ("local", "remote", False),
    ("remote", "local", False), ("remote", "remote", True),
    ("none", "local", False), ("none", "remote", False),
))
def test_output_cleanup_selection(tmp_path, policy, side, deleted):
    from ase2sprkkr.execution import ExecutionSpec, LaunchSpec
    from ase2sprkkr.execution.file_availability import prepare_local_files

    old = tmp_path / "old.dat"
    old.write_text("old")
    inp = tmp_path / "input.dat"
    inp.write_text("input")
    spec = ExecutionSpec(
        argv=("program",), directory=str(tmp_path), input_file=str(inp),
        output_file=None, output_files=(str(old),),
        launch=LaunchSpec(cleanup_outputs=policy),
    )
    prepare_local_files(spec, side=side)
    assert old.exists() != deleted
    assert inp.read_text() == "input"


def test_cleanup_disabled_does_not_disable_collision_validation(tmp_path):
    from ase2sprkkr.execution import ExecutionSpec, LaunchSpec
    from ase2sprkkr.execution.file_availability import prepare_local_files

    inp = tmp_path / "input.dat"
    inp.write_text("input")
    spec = ExecutionSpec(
        argv=("program",), directory=str(tmp_path), input_file=str(inp),
        output_file=str(inp), launch=LaunchSpec(cleanup_outputs="none"),
    )
    with pytest.raises(ValueError, match="overlaps protected"):
        prepare_local_files(spec)
    assert inp.read_text() == "input"


def test_missing_path_lexical_alias_counts_as_same_file(tmp_path):
    a = tmp_path / "missing.dat"
    b = tmp_path / "subdir" / ".." / "missing.dat"
    assert same_local_file(a, b)
    assert not same_local_file(a, tmp_path / "other.dat")
    assert not same_local_file(None, a)


@pytest.mark.parametrize("alias_kind", ("symlink-file", "symlink-dir", "hardlink"))
def test_same_file_identity_is_shared_across_all_three_policies(tmp_path, alias_kind):
    file = tmp_path / "main.out"
    file.write_text("existing output")
    if alias_kind == "symlink-file":
        alias = tmp_path / "alias.out"
        alias.symlink_to(file)
    elif alias_kind == "symlink-dir":
        folder = tmp_path / "alias"
        folder.symlink_to(tmp_path, target_is_directory=True)
        alias = folder / "main.out"
    else:
        alias = tmp_path / "alias.out"
        os.link(file, alias)
    assert same_local_file(file, alias)

    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            stdout=FileEntry(local_path=str(file)),
            caches=(FileEntry(local_path=str(alias)),),
        )
    with pytest.raises(ValueError, match="overlaps protected"):
        validate_output_collisions((str(alias),), protected=(str(file),))

    manager = RunManager(RunStore(tmp_path / "runs"))
    first = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(outputs=(FileEntry(local_path=str(file)),)),
    )
    second = manager.new_record(
        task="BSF", directory=tmp_path,
        files=FileManifest(stdout=FileEntry(local_path=str(alias))),
    )
    assert reservation_conflict(first, second)
    assert reservation_conflict(second, first)


def test_nonexistent_symlink_directory_alias_is_protected(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    target = root / "future.out"
    linked = alias / "future.out"
    assert same_local_file(target, linked)
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(outputs=(
            FileEntry(local_path=str(target)),
            FileEntry(local_path=str(linked)),
        ))


def test_cleanup_replaces_named_stdout_without_touching_hardlink(tmp_path):
    from ase2sprkkr.execution import ExecutionSpec, LaunchSpec
    from ase2sprkkr.execution.file_availability import prepare_local_files

    stdout = tmp_path / "run.out"
    external = tmp_path / "retained"
    external.write_text("prior output")
    os.link(external, stdout)
    spec = ExecutionSpec(
        argv=("program",), directory=str(tmp_path), input_file=None,
        output_file=str(stdout), launch=LaunchSpec(cleanup_outputs="both"),
    )
    prepare_local_files(spec, preserve_open_streams=True)
    assert stdout.read_bytes() == b""
    assert external.read_text() == "prior output"


def test_cleanup_preserves_open_stdout_descriptor(tmp_path):
    from ase2sprkkr.execution import ExecutionSpec, LaunchSpec
    from ase2sprkkr.execution.file_availability import prepare_local_files

    stdout = tmp_path / "run.out"
    with stdout.open("w+b") as handle:
        handle.write(b"existing open output")
        handle.flush()
        spec = ExecutionSpec(
            argv=("program",), directory=str(tmp_path), input_file=None,
            output_file=handle, launch=LaunchSpec(cleanup_outputs="both"),
        )
        prepare_local_files(spec, preserve_open_streams=True)
        assert stdout.read_bytes() == b"existing open output"
        handle.seek(0, os.SEEK_END)
        handle.write(b" next")
        handle.flush()
        assert stdout.read_bytes() == b"existing open output next"
