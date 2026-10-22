"""Remote preparation is driven only by the frozen file manifest."""

import asyncio
from dataclasses import replace

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
