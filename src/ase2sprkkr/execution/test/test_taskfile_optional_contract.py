"""TaskFile metadata at the execution manifest boundary."""

from dataclasses import asdict

import pytest

from ase2sprkkr.execution.base import ExecutionSpec
from ase2sprkkr.execution.run_model import FileEntry, FileManifest
from ase2sprkkr.input_parameters.task_definition import TaskFile


def spec(tmp_path, **changes):
    arguments = dict(
        argv=("true",), directory=str(tmp_path),
        input_file=tmp_path / "input.inp",
        output_file=tmp_path / "stdout.out",
    )
    arguments.update(changes)
    return ExecutionSpec(**arguments)


def test_optional_task_files_survive_manifest_roundtrip(tmp_path):
    files = FileManifest.from_spec(spec(
        tmp_path,
        input_files=(TaskFile(tmp_path / "maybe.inp", optional=True),),
        output_files=(
            TaskFile(tmp_path / "maybe-*.dat", optional=True),
            TaskFile(tmp_path / "required.dat"),
        ),
    ))
    assert files.inputs[1].optional
    assert [entry.optional for entry in files.outputs] == [True, False]
    assert FileManifest(**asdict(files)) == files


def test_repeated_auxiliary_input_is_not_silently_coalesced(tmp_path):
    path = tmp_path / "extra.inp"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(tmp_path, input_files=(path, path)))


def test_repeated_output_is_not_silently_coalesced(tmp_path):
    path = tmp_path / "extra.dat"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(
            tmp_path, output_files=(TaskFile(path, optional=True), path),
        ))


def test_primary_and_stdin_are_references(tmp_path):
    primary = tmp_path / "input.inp"
    files = FileManifest.from_spec(spec(
        tmp_path, input_files=(primary,), stdin=primary,
    ))
    assert len(files.inputs) == 1
    assert files.primary_input.local_path == str(primary)
    assert files.stdin.local_path == str(primary)


def test_output_cache_collision_is_rejected(tmp_path):
    path = tmp_path / "shared.dat"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(
            tmp_path, output_files=(path,), cache_files=(path,),
        ))


def test_optional_flag_must_be_bool():
    with pytest.raises(TypeError, match="optional"):
        FileEntry("/tmp/optional.dat", optional="yes")
