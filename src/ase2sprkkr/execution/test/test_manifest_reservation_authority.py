"""Reservation and staging share a single target identity."""

import asyncio
from dataclasses import asdict

import pytest

from ase2sprkkr.execution import (
    DuplicateSubmissionError, ExecutionSpec, FileManifest, LaunchSpec,
    LocalTransport, RemoteEndpointExecutor, RunStore,
)
from ase2sprkkr.execution.endpoint import endpoint_for_executor
from ase2sprkkr.execution.run_manager import RunManager


def test_explicit_input_roles_roundtrip(tmp_path):
    spec = ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file=None, output_file=None,
        input_files=("aux.in",), stdin="stdin.in",
    )
    manifest = FileManifest.from_spec(spec)
    assert manifest.input_file == str(tmp_path / "stdin.in")
    assert manifest.stdin is True
    assert manifest.input_files == (
        str(tmp_path / "stdin.in"), str(tmp_path / "aux.in"),
    )
    assert FileManifest(**asdict(manifest)) == manifest


@pytest.mark.parametrize("sync,conflict", [
    ("changed", True), ("always", True),
    ("missing", True), ("never", False),
])
def test_reserved_remote_input_upload_access(tmp_path, sync, conflict):
    manager = RunManager(RunStore(tmp_path / "runs"))
    executor = RemoteEndpointExecutor(LocalTransport())
    records = []
    for name in ("one", "two"):
        root = tmp_path / name
        spec = ExecutionSpec(
            argv=("true",), directory=str(root),
            input_file=None, output_file=name + ".out",
            stdin="shared.in",
            launch=LaunchSpec(
                local_root=str(root), remote_root=str(tmp_path / "remote"),
                input_sync=sync,
            ),
        )
        manifest, target = executor.reservation_context(
            spec, FileManifest.from_spec(spec),
        )
        records.append(manager.new_record(
            task=name, directory=spec.directory,
            remote_directory=target,
            endpoint=endpoint_for_executor(executor), files=manifest,
        ))
    manager.reserve(records[0])
    if conflict:
        with pytest.raises(DuplicateSubmissionError):
            manager.reserve(records[1])
    else:
        manager.reserve(records[1])


def test_reserved_tilde_root_is_used_by_staging(tmp_path, monkeypatch):
    executor = RemoteEndpointExecutor(LocalTransport())
    local = tmp_path / "local"
    local.mkdir()
    (local / "primary.in").write_text("primary")
    (local / "stdin.in").write_text("stdin")
    spec = ExecutionSpec(
        argv=("true",), directory=str(local),
        input_file="primary.in", output_file="run.out",
        stdin="stdin.in",
        launch=LaunchSpec(local_root=str(tmp_path), remote_root="~",
                          input_sync="never"),
    )
    async def home():
        return "/home/resolved"
    monkeypatch.setattr(executor.connection, "remote_home", home)
    manifest, directory = executor.reservation_context(
        spec, FileManifest.from_spec(spec),
    )
    assert directory == "/home/resolved/local"
    assert manifest.inputs[0].remote_path == directory + "/stdin.in"
    assert manifest.inputs[1].remote_path == directory + "/primary.in"
    assert manifest.stdin is True
    checked = []

    async def remote_signature(path):
        checked.append(path)
        if path in {entry.remote_path for entry in manifest.inputs}:
            return (7, 123)
        return None

    async def unexpected_upload(*args, **kwargs):
        raise AssertionError("input_sync=never must not upload files")

    monkeypatch.setattr(executor.file_handler.transport, "file_signature", remote_signature)
    monkeypatch.setattr(executor.file_handler.transport, "put_file", unexpected_upload)
    prepared = asyncio.run(
        executor.file_handler.prepare(manifest.mapping, spec, manifest),
    )
    assert set(checked) == {entry.remote_path for entry in manifest.inputs}
    assert prepared.mapping == manifest.mapping
    assert prepared.inputs == manifest.inputs
    assert prepared.stdin is True
    assert prepared.inputs[0].remote_path == manifest.inputs[0].remote_path


def test_staging_rejects_changed_reserved_target(tmp_path):
    executor = RemoteEndpointExecutor(LocalTransport())
    spec = ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file=None, output_file="out",
        launch=LaunchSpec(local_root=str(tmp_path),
                          remote_root=str(tmp_path / "remote")),
    )
    manifest, _ = executor.reservation_context(
        spec, FileManifest.from_spec(spec),
    )
    from dataclasses import replace
    mismatched = replace(manifest, stdout=replace(
        manifest.stdout, remote_path="/unexpected/path",
    ))
    with pytest.raises(ValueError, match="differs from reserved path"):
        asyncio.run(executor.file_handler.prepare(
            mismatched.mapping, spec, mismatched,
        ))


def test_staging_uses_reserved_manifest_not_spec_declarations(tmp_path):
    """Stage reserved paths; changed spec declarations cannot expand the transfer."""
    from dataclasses import replace

    executor = RemoteEndpointExecutor(LocalTransport())
    source = tmp_path / "work"
    source.mkdir()
    (source / "first.in").write_text("first")
    (source / "extra.in").write_text("extra")
    spec = ExecutionSpec(
        argv=("true",), directory=str(source),
        input_file="first.in", output_file="out",
        launch=LaunchSpec(
            local_root=str(tmp_path), remote_root=str(tmp_path / "remote"),
        ),
    )
    reserved, _ = executor.reservation_context(
        spec, FileManifest.from_spec(spec),
    )
    changed = replace(
        spec, input_files=("extra.in",),
        output_files=("extra.out",), cache_files=("extra.cache",),
    )
    prepared = asyncio.run(executor.file_handler.prepare(
        reserved.mapping, changed, reserved,
    ))
    assert prepared.inputs == reserved.inputs
    assert prepared.outputs == reserved.outputs
    assert prepared.caches == reserved.caches
    assert (tmp_path / "remote" / "work" / "first.in").read_text() == "first"
    assert not (tmp_path / "remote" / "work" / "extra.in").exists()


def test_staging_validates_all_reserved_paths_before_transfer(tmp_path):
    """A malformed output mapping cannot leave a partially uploaded input."""
    from dataclasses import replace

    executor = RemoteEndpointExecutor(LocalTransport())
    source = tmp_path / "work"
    source.mkdir()
    (source / "first.in").write_text("first")
    spec = ExecutionSpec(
        argv=("true",), directory=str(source),
        input_file="first.in", output_file="out",
        launch=LaunchSpec(
            local_root=str(tmp_path), remote_root=str(tmp_path / "remote"),
        ),
    )
    reserved, _ = executor.reservation_context(
        spec, FileManifest.from_spec(spec),
    )
    invalid = replace(reserved, stdout=replace(
        reserved.stdout, remote_path="/unexpected/path",
    ))
    with pytest.raises(ValueError, match="differs from reserved path"):
        asyncio.run(executor.file_handler.prepare(
            invalid.mapping, spec, invalid,
        ))
    assert not (tmp_path / "remote" / "work" / "first.in").exists()
