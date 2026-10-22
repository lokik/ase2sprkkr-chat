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


def test_identical_auxiliary_inputs_are_rejected(tmp_path):
    path = tmp_path / "extra.inp"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(tmp_path, input_files=(path, path)))


@pytest.mark.parametrize("first_optional, second_optional", [
    (True, False), (False, True), (True, True), (False, False),
])
def test_repeated_auxiliary_input_never_merges_optionality(
    tmp_path, first_optional, second_optional,
):
    path = str(tmp_path / "extra.inp")
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(
            tmp_path,
            input_files=(
                DeclaredPath(path, optional=first_optional),
                DeclaredPath(path, optional=second_optional),
            ),
        ))


def test_repeated_output_is_not_silently_coalesced(tmp_path):
    path = tmp_path / "extra.dat"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(
            tmp_path, output_files=(DeclaredPath(str(path), optional=True), path),
        ))


def test_first_input_can_feed_stdin_but_duplicate_input_alias_is_invalid(tmp_path):
    primary = tmp_path / "input.inp"
    files = FileManifest.from_spec(spec(tmp_path, stdin=primary))
    assert len(files.inputs) == 1
    assert files.inputs[0].local_path == str(primary)
    assert files.stdin is True
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec(tmp_path, input_files=(primary,), stdin=primary))


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
    handler = RemoteFileHandler(LocalTransport())
    staged = asyncio.run(handler._stage_file(
        mapping, str(source), {}, required=False,
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
    missing = tmp_path / "missing-stdin.inp"
    spec_obj = spec(
        tmp_path, stdin=missing,
        input_files=(DeclaredPath(str(missing), optional=True),),
    )
    # Invalid declarations fail before staging or any remote transfer:
    # a mandatory stdin source cannot also be an optional input.
    with pytest.raises(ValueError, match="Standard input cannot be optional"):
        FileManifest.from_spec(spec_obj)


@pytest.mark.parametrize("present", (False, True))
def test_optional_exact_output_absence_vs_transfer_error(tmp_path, present):
    """Optional tolerates absence, but never a failed transfer of an existing file."""
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileMapping
    from ase2sprkkr.execution.transport import LocalTransport

    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    remote_file = remote / "optional.dat"
    if present:
        remote_file.write_text("present")

    class FailingTransfer(LocalTransport):
        async def get_file(self, source, destination):
            raise OSError("simulated transfer failure")

    mapping = FileMapping(str(local), str(remote))
    files = FileManifest(
        outputs=(FileEntry(str(local / "optional.dat"),
                           str(remote_file), optional=True,
                           availability_state="pending"),),
        mapping=mapping,
    )
    outcome = SimpleNamespace(output_file=None, error_file=None)
    completed, error = asyncio.run(
        RemoteFileHandler(FailingTransfer()).collect(
            mapping, files, outcome, required=True))
    entry = completed.outputs[0]
    if present:
        assert isinstance(error, OSError)
        assert "simulated transfer failure" in str(error)
        assert entry.availability_state == "failed"
    else:
        assert error is None
        assert entry.availability_state == "skipped"


@pytest.mark.parametrize("required_alias", (False, True))
def test_staged_input_wildcard_members_preserve_optionality(tmp_path, required_alias):
    """Concrete members retain optionality; a required overlap wins."""
    import asyncio
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileMapping
    from ase2sprkkr.execution.transport import LocalTransport

    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (local / "part-a.dat").write_text("a")
    (local / "part-b.dat").write_text("b")
    optional = DeclaredPath(str(local / "part-*.dat"), optional=True)
    declarations = (optional,)
    if required_alias:
        declarations += (local / "part-a.dat",)
    execution = ExecutionSpec(
        argv=("true",), directory=str(local), input_file=None,
        output_file=None, input_files=declarations,
    )
    mapping = FileMapping(str(local), str(remote))
    files = asyncio.run(RemoteFileHandler(LocalTransport()).prepare(
        mapping, execution, FileManifest.from_spec(execution),
    ))
    by_name = {os.path.basename(entry.local_path): entry for entry in files.inputs}
    assert set(by_name) == {"part-a.dat", "part-b.dat"}
    assert by_name["part-a.dat"].optional is not required_alias
    assert by_name["part-b.dat"].optional is True
