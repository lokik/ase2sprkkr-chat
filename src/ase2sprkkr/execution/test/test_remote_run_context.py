import asyncio
from dataclasses import asdict

import pytest

from ase2sprkkr.execution import LaunchSpec
from ase2sprkkr.execution.remote_endpoint import RemoteEndpointExecutor


class _Transport:
    host = "cluster"



def test_run_context_is_staging_policy_not_executor(tmp_path):
    endpoint = RemoteEndpointExecutor(_Transport())
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(
            local_root=tmp_path,
            remote_root="/scratch/a",
            input_sync="always",
            output_sync="missing",
        )
    ))
    assert context.input_sync == "always"
    assert context.output_sync == "missing"
    assert context.local_root == tmp_path.resolve()
    assert context.remote_root == "/scratch/a"



def test_different_run_staging_policies_share_one_endpoint(tmp_path):
    endpoint = RemoteEndpointExecutor(_Transport())
    first = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=tmp_path, remote_root="/scratch/one"),
    ))
    second = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=tmp_path, remote_root="/scratch/two"),
    ))

    assert first is not second
    assert first.remote_root == "/scratch/one"
    assert second.remote_root == "/scratch/two"


def test_attach_restores_typed_manifest_without_sync_dictionary(tmp_path, monkeypatch):
    from ase2sprkkr.execution import RunState
    from ase2sprkkr.execution.endpoint import endpoint_for_executor
    from ase2sprkkr.execution.manager import RunRecord
    from ase2sprkkr.execution.run_model import (
        FileEntry, FileManifest, Execution, WorkingDirectory,
    )
    from ase2sprkkr.execution.transport import LocalTransport

    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()

    def entry(name, **kwargs):
        return FileEntry(str(local / name), str(remote / name), **kwargs)

    files = FileManifest(
        stdout=entry("stdout.txt"),
        outputs=(
            entry("exact.out"),
            entry("data-*.out"),
            entry("data-a.out", source=str(local / "data-*.out")),
            FileEntry(str(local / "local-only.out")),
        ),
        caches=(
            entry("cache.dat", pre_run_signature=(2, 3)),
            entry("restart-*.pot"),
            entry("restart-a.pot", source=str(local / "restart-*.pot")),
            FileEntry(remote_path=str(remote / "remote-only.dat"), pre_run_signature=(5, 7)),
            FileEntry(str(local / "local-only-cache.dat")),
        ),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=local, remote_root=str(remote)),
    ))
    files = FileManifest(**{**asdict(files),
                            "mapping": asdict(context)})
    run = RunRecord(
        id="test-manifest", task="SCF", state=RunState.RUNNING,
        created_at="2026-10-13T08:00:00+00:00",
        updated_at="2026-10-13T08:00:00+00:00",
        execution=Execution(
            endpoint_for_executor(endpoint), WorkingDirectory(str(local), str(remote)),
            "opaque-id",
        ),
        files=files,
    )

    async def status(*args, **kwargs):
        return {"state": "running"}

    monkeypatch.setattr(endpoint.connection, "status", status)
    runner = asyncio.run(endpoint._attach(run, start=False))
    try:
        assert runner.files is run.files
        assert runner.files.caches[0].pre_run_signature == (2, 3)
    finally:
        runner.spec.output_file.close()


def test_absolute_remote_root_needs_no_remote_home_query(tmp_path, monkeypatch):
    endpoint = RemoteEndpointExecutor(_Transport())
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=tmp_path, remote_root="/scratch/../work"),
    ))
    calls = []

    async def remote_home():
        calls.append("home")
        raise AssertionError("Absolute remote roots must not query remote HOME")

    monkeypatch.setattr(endpoint.connection, "remote_home", remote_home)
    directory = tmp_path / "calculation"
    directory.mkdir()
    assert context.map_path(directory) == "/work/calculation"
    assert context.map_path(directory) == "/work/calculation"
    assert context.unmap_path("/work/calculation") == directory
    assert context.remote_root == "/work"
    assert calls == []


@pytest.mark.parametrize(
    "root, expected",
    [
        (None, "/home/user"),
        ("~", "/home/user"),
        ("~/projects", "/home/user/projects"),
        ("projects", "/home/user/projects"),
    ],
)
def test_home_relative_remote_root_still_resolves_home_once(
    tmp_path, monkeypatch, root, expected,
):
    endpoint = RemoteEndpointExecutor(_Transport())
    calls = []

    async def remote_home():
        calls.append("home")
        return "/home/user"

    monkeypatch.setattr(endpoint.connection, "remote_home", remote_home)
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=tmp_path, remote_root=root),
    ))
    assert context.remote_root == expected
    assert calls == ["home"]


def test_attach_closes_stdout_if_opening_stderr_fails(tmp_path, monkeypatch):
    """A failed second open must release the first stream immediately."""
    import builtins
    from ase2sprkkr.execution import RunState
    from ase2sprkkr.execution.endpoint import endpoint_for_executor
    from ase2sprkkr.execution.manager import RunRecord
    from ase2sprkkr.execution.run_model import (
        Execution, FileEntry, FileManifest, WorkingDirectory,
    )
    from ase2sprkkr.execution.transport import LocalTransport
    from ase2sprkkr.execution import remote_endpoint

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    stdout = local / "stdout.txt"
    stderr = local / "stderr.txt"
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_mapping(LaunchSpec(
        local_root=local, remote_root=str(remote),
    )))
    run = RunRecord(
        id="stream-open-failure", task="SCF", state=RunState.RUNNING,
        created_at="2026-10-13T08:00:00+00:00",
        updated_at="2026-10-13T08:00:00+00:00",
        execution=Execution(
            endpoint_for_executor(endpoint),
            WorkingDirectory(str(local), str(remote)), "remote-id",
        ),
        files=FileManifest(
            stdout=FileEntry(str(stdout), str(remote / "stdout.txt")),
            stderr=FileEntry(str(stderr), str(remote / "stderr.txt")),
            mapping=context,
        ),
    )

    async def status(*args, **kwargs):
        return {"state": "running"}

    monkeypatch.setattr(endpoint.connection, "status", status)
    opened = []
    actual_open = builtins.open

    def failing_open(path, *args, **kwargs):
        if str(path) == str(stderr):
            raise PermissionError("stderr not writable")
        stream = actual_open(path, *args, **kwargs)
        if str(path) == str(stdout):
            opened.append(stream)
        return stream

    monkeypatch.setattr(remote_endpoint, "open", failing_open, raising=False)
    with pytest.raises(PermissionError, match="stderr not writable"):
        asyncio.run(endpoint._attach(run, start=False))
    assert len(opened) == 1
    assert opened[0].closed


def test_unmap_rejects_symlink_parent_outside_local_root(tmp_path):
    root = tmp_path / "local"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    endpoint = RemoteEndpointExecutor(_Transport())
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=root, remote_root="/remote"),
    ))
    with pytest.raises(ValueError, match="outside mapped root"):
        context.unmap_path("/remote/redirect/output.dat")



def test_unmap_preserves_internal_symlink_spelling(tmp_path):
    root = tmp_path / "local"
    actual = root / "actual"
    actual.mkdir(parents=True)
    (root / "alias").symlink_to(actual, target_is_directory=True)
    endpoint = RemoteEndpointExecutor(_Transport())
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=root, remote_root="/remote"),
    ))
    assert context.unmap_path("/remote/alias/output.dat") == (
        root / "alias" / "output.dat"
    )

