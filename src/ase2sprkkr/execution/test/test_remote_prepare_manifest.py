"""Remote preparation is driven only by the frozen file manifest."""

import asyncio
from dataclasses import replace

import pytest

from ase2sprkkr.execution import (
    ExecutionSpec, FileManifest, FileMapping, LaunchSpec, LocalTransport,
    RemoteEndpointExecutor,
)


def test_staging_does_not_flush_new_unreserved_source_handle(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    original = local / "original.in"
    original.write_text("original")
    extra = local / "extra.in"
    extra.write_text("extra")

    class UnexpectedSource:
        name = str(extra)

        def flush(self):
            raise AssertionError("Unreserved source must not be accessed")

    spec = ExecutionSpec(
        argv=("true",), directory=str(local),
        input_file=str(original), output_file=None,
        launch=LaunchSpec(local_root=local, remote_root=remote),
    )
    mapping = FileMapping(str(local), str(remote))
    reserved = FileManifest.from_spec(spec).with_mapping(mapping)
    changed = replace(spec, input_files=(UnexpectedSource(),))
    executor = RemoteEndpointExecutor(LocalTransport())
    prepared = asyncio.run(
        executor.file_handler.prepare(mapping, changed, reserved),
    )
    assert prepared.inputs == reserved.inputs
    assert (remote / "original.in").read_text() == "original"
    assert not (remote / "extra.in").exists()


def test_prepared_caches_keep_declaration_order_and_wildcard_baselines(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (local / "restart-a.dat").write_text("restart")
    spec = ExecutionSpec(
        argv=("true",), input_file=None, output_file=None,
        directory=str(local),
        cache_files=("restart-*.dat", "restart-a.dat"),
        launch=LaunchSpec(local_root=local, remote_root=remote),
    )
    mapping = FileMapping(str(local), str(remote))
    reserved = FileManifest.from_spec(spec).with_mapping(mapping)
    executor = RemoteEndpointExecutor(LocalTransport())
    prepared = asyncio.run(
        executor.file_handler.prepare(mapping, spec, reserved),
    )
    assert prepared.caches[:2][0].local_path == str(local / "restart-*.dat")
    assert prepared.caches[:2][1].local_path == str(local / "restart-a.dat")
    assert prepared.caches[2].local_path is None
    assert prepared.caches[2].remote_path == str(remote / "restart-a.dat")
    assert prepared.caches[2].pre_run_signature is not None
    assert prepared.caches[1].pre_run_signature is not None
    assert (remote / "restart-a.dat").read_text() == "restart"


def test_optional_wildcard_input_materializes_required_exact_file(tmp_path):
    from .test_taskfile_optional_contract import DeclaredPath

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    (local / "data-a.in").write_text("A")
    spec = ExecutionSpec(
        argv=("true",), input_file=None, output_file=None,
        directory=str(local),
        input_files=(
            DeclaredPath(str(local / "data-*.in"), optional=True),
            DeclaredPath(str(local / "data-a.in"), optional=False),
        ),
        launch=LaunchSpec(local_root=local, remote_root=remote),
    )
    mapping = FileMapping(str(local), str(remote))
    reserved = FileManifest.from_spec(spec).with_mapping(mapping)
    prepared = asyncio.run(
        RemoteEndpointExecutor(LocalTransport()).file_handler.prepare(
            mapping, spec, reserved,
        )
    )
    entry = next(e for e in prepared.inputs if e.local_path == str(local / "data-a.in"))
    assert entry.optional is False


@pytest.mark.parametrize("role", ("primary", "stdin", "extra"))
def test_never_checks_mandatory_input_roles_without_upload(tmp_path, role):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    source = local / "required.in"
    source.write_text("only-local")
    roles = {
        "primary": {"input_file": source},
        "stdin": {"stdin": source},
        "extra": {"input_files": (source,)},
    }
    spec = ExecutionSpec(
        argv=("true",), directory=str(local), output_file=None,
        **{"input_file": None, **roles[role]},
    )
    mapping = FileMapping(str(local), str(remote), input_sync="never")

    class NoUpload(LocalTransport):
        async def put_file(self, source, destination):
            pytest.fail("NEVER must not upload")

        async def file_digest(self, path):
            pytest.fail("NEVER must not calculate digests")

    with pytest.raises(FileNotFoundError, match="Required remote input file"):
        asyncio.run(RemoteEndpointExecutor(NoUpload()).file_handler.prepare(mapping, spec))


def test_never_accepts_remote_only_input(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = local / "remote-only.in"
    (remote / source.name).write_text("already available")

    class NoUpload(LocalTransport):
        async def put_file(self, source, destination):
            pytest.fail("NEVER must not upload")

    spec = ExecutionSpec(
        argv=("true",), directory=str(local), input_file=source, output_file=None,
    )
    mapping = FileMapping(str(local), str(remote), input_sync="never")
    prepared = asyncio.run(
        RemoteEndpointExecutor(NoUpload()).file_handler.prepare(mapping, spec)
    )
    assert prepared.inputs[0].remote_path == str(remote / source.name)
    assert not source.exists()


def test_never_does_not_check_missing_optional_exact_input_or_cache(tmp_path):
    from .test_taskfile_optional_contract import DeclaredPath

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    optional = local / "optional.in"
    cache = local / "cache.pot"
    spec = ExecutionSpec(
        argv=("true",), directory=str(local), input_file=None,
        output_file=None, input_files=(DeclaredPath(str(optional), optional=True),),
        cache_files=(str(cache),),
    )
    mapping = FileMapping(
        str(local), str(remote), input_sync="never", output_sync="never",
    )

    class NoChecks(LocalTransport):
        async def file_signature(self, path):
            pytest.fail("Optional exact inputs and caches must not require a probe")

        async def file_digest(self, path):
            pytest.fail("NEVER must not hash")

        async def put_file(self, source, destination):
            pytest.fail("NEVER must not upload")

    prepared = asyncio.run(
        RemoteEndpointExecutor(NoChecks()).file_handler.prepare(mapping, spec)
    )
    assert prepared.inputs[0].optional is True
    assert len(prepared.caches) == 1
    assert not remote.exists()


def test_never_rechecks_required_input_after_optional_alias(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    source = local / "shared.in"
    source.write_text("local")
    mapping = FileMapping(str(local), str(remote), input_sync="never")
    handler = RemoteEndpointExecutor(LocalTransport()).file_handler

    async def check():
        used = {}
        await handler._stage_file(mapping, str(source), used, required=False)
        with pytest.raises(FileNotFoundError, match="Required remote input file"):
            await handler._stage_file(mapping, str(source), used, required=True)

    asyncio.run(check())


@pytest.mark.parametrize("present", (False, True))
def test_never_required_wildcard_expands_remote_only_members(tmp_path, present):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    pattern = local / "remote-*.dat"
    member = remote / "remote-a.dat"
    if present:
        member.write_text("only-remote")
    spec = ExecutionSpec(
        argv=("true",), directory=str(local), input_file=None,
        output_file=None, input_files=(str(pattern),),
    )
    mapping = FileMapping(str(local), str(remote), input_sync="never")

    class NoUpload(LocalTransport):
        async def put_file(self, source, destination):
            pytest.fail("NEVER must not upload")

        async def put_files(self, sources, source_root, destination_root):
            pytest.fail("NEVER must not batch upload")

    handler = RemoteEndpointExecutor(NoUpload()).file_handler
    if present:
        prepared = asyncio.run(handler.prepare(mapping, spec))
        assert len(prepared.inputs) == 1
        assert prepared.inputs[0].remote_path == str(member)
        assert prepared.inputs[0].local_path == str(local / member.name)
        assert not (local / member.name).exists()
    else:
        with pytest.raises(FileNotFoundError, match="Required remote input pattern"):
            asyncio.run(handler.prepare(mapping, spec))
