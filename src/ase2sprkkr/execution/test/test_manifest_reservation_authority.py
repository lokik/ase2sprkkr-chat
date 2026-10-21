"""Reservation and staging share a single target identity."""

import asyncio
from dataclasses import asdict

import pytest

from ase2sprkkr.execution import (
    DuplicateSubmissionError, ExecutionSpec, FileManifest, LaunchSpec,
    LocalTransport, RemoteEndpointExecutor, RunStore,
)
from ase2sprkkr.execution.endpoint import endpoint_for_executor
from ase2sprkkr.execution.manager import ExecutionManager


def test_explicit_input_roles_roundtrip(tmp_path):
    spec = ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file=None, output_file=None,
        input_files=("aux.in",), stdin="stdin.in",
    )
    manifest = FileManifest.from_spec(spec)
    assert manifest.input_file is None
    assert manifest.input_files == (
        str(tmp_path / "aux.in"), str(tmp_path / "stdin.in"),
    )
    assert FileManifest(**asdict(manifest)) == manifest


@pytest.mark.parametrize("sync,conflict", [
    ("changed", True), ("always", True),
    ("missing", True), ("never", False),
])
def test_reserved_remote_input_upload_access(tmp_path, sync, conflict):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
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
    assert manifest.primary_input.remote_path == directory + "/primary.in"
    assert manifest.stdin.remote_path == directory + "/stdin.in"
    prepared = asyncio.run(
        executor.file_handler.prepare(manifest.mapping, spec, manifest),
    )
    assert prepared.mapping == manifest.mapping
    assert prepared.primary_input == manifest.primary_input
    assert prepared.stdin == manifest.stdin
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
