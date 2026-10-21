"""TaskFile metadata at the execution manifest boundary."""

from dataclasses import asdict, dataclass
import os

import pytest

from ase2sprkkr.execution.base import ExecutionSpec
from ase2sprkkr.execution.run_model import FileEntry, FileManifest
# Model the public TaskFile path/optional protocol without importing #86.
@dataclass(frozen=True)
class DeclaredPath(os.PathLike):
    pathname: str
    optional: bool = False

    def __fspath__(self):
        return self.pathname


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
        input_files=(DeclaredPath(str(tmp_path / "maybe.inp"), optional=True),),
        output_files=(
            DeclaredPath(str(tmp_path / "maybe-*.dat"), optional=True),
            DeclaredPath(str(tmp_path / "required.dat")),
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
            tmp_path, output_files=(DeclaredPath(str(path), optional=True), path),
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


def test_optional_missing_input_does_not_block_stage(tmp_path):
    import asyncio

    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileMapping
    from ase2sprkkr.execution.transport import LocalTransport

    source = tmp_path / "absent.inp"
    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"))
    spec_obj = spec(tmp_path)
    handler = RemoteFileHandler(LocalTransport())
    staged = asyncio.run(handler._stage_file(
        mapping, spec_obj, source, {}, required=False,
    ))
    assert staged == mapping.map_path(source)


def test_optional_and_required_output_transfer_are_independent(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileMapping
    from ase2sprkkr.execution.transport import LocalTransport

    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"))
    files = FileManifest(
        outputs=(
            FileEntry(str(tmp_path / "maybe.dat"),
                      str(tmp_path / "remote/maybe.dat"), optional=True,
                      availability_state="pending"),
            FileEntry(str(tmp_path / "mandatory.dat"),
                      str(tmp_path / "remote/mandatory.dat"),
                      availability_state="pending"),
        ),
        mapping=mapping,
    )
    result = SimpleNamespace(output_file=None, error_file=None)
    completed, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        mapping, files, result, required=True,
    ))
    assert isinstance(error, FileNotFoundError)
    assert [entry.availability_state for entry in completed.outputs] == [
        "skipped", "failed",
    ]
    assert completed.availability_state == "failed"


def test_optional_alias_cannot_weaken_mandatory_stdin(tmp_path):
    import asyncio

    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileMapping
    from ase2sprkkr.execution.transport import LocalTransport

    (tmp_path / "input.inp").write_text("input")
    remote = tmp_path / "remote"
    remote.mkdir()
    missing = tmp_path / "missing-stdin.inp"
    spec_obj = spec(
        tmp_path, stdin=missing,
        input_files=(DeclaredPath(str(missing), optional=True),),
    )
    mapping = FileMapping(str(tmp_path), str(remote))
    handler = RemoteFileHandler(LocalTransport())
    with pytest.raises(FileNotFoundError, match="neither locally nor remotely"):
        asyncio.run(handler.prepare(
            mapping, spec_obj, FileManifest.from_spec(spec_obj),
        ))
