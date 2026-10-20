import asyncio

import pytest

from ase2sprkkr.execution import Executor, LaunchSpec
from ase2sprkkr.execution.remote_endpoint import RemoteEndpointExecutor


class _Transport:
    host = "cluster"



def test_run_context_is_staging_policy_not_executor(tmp_path):
    endpoint = RemoteEndpointExecutor(_Transport())
    context = endpoint._context(
        endpoint,
        LaunchSpec(
            local_root=tmp_path,
            remote_root="/scratch/a",
            input_sync="always",
            output_sync="missing",
        ),
    )

    assert not isinstance(context, Executor)
    assert context.endpoint is endpoint
    assert context.transport is endpoint.transport
    assert context.local_root == tmp_path.resolve()
    assert context.remote_root == "/scratch/a"



def test_different_run_staging_policies_share_one_endpoint(tmp_path):
    endpoint = RemoteEndpointExecutor(_Transport())
    first = endpoint._context(
        endpoint,
        LaunchSpec(local_root=tmp_path, remote_root="/scratch/one"),
    )
    second = endpoint._context(
        endpoint,
        LaunchSpec(local_root=tmp_path, remote_root="/scratch/two"),
    )

    assert first is not second
    assert first.endpoint is endpoint
    assert second.endpoint is endpoint
    assert first.remote_root == "/scratch/one"
    assert second.remote_root == "/scratch/two"


def test_attach_restores_typed_manifest_without_sync_dictionary(tmp_path, monkeypatch):
    from ase2sprkkr.execution import RunState
    from ase2sprkkr.execution.endpoint import endpoint_for_executor
    from ase2sprkkr.execution.manager import RunRecord
    from ase2sprkkr.execution.run_model import (
        FileEntry, FileManifest, CacheEntry, Execution, WorkingDirectory,
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
            entry("data-a.out", source_pattern=str(local / "data-*.out")),
            FileEntry(str(local / "local-only.out")),
        ),
        caches=(
            CacheEntry(entry("cache.dat"), (2, 3)),
            CacheEntry(entry("restart-*.pot")),
            CacheEntry(entry("restart-a.pot",
                             source_pattern=str(local / "restart-*.pot"))),
            CacheEntry(FileEntry(remote_path=str(remote / "remote-only.dat")), (5, 7)),
            CacheEntry(FileEntry(str(local / "local-only-cache.dat"))),
        ),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = endpoint._context(endpoint, LaunchSpec(local_root=local, remote_root=remote))
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

    monkeypatch.setattr(context, "_status", status)
    # The original observer's file handle need not survive a disconnection.
    original_stream = (local / "stdout.txt").open("ab")
    original_stream.write(b"previous bytes")
    original_stream.close()
    runner = asyncio.run(context.attach(run, start=False))
    try:
        assert runner.spec.output_file is not None
        assert not runner.spec.output_file.closed
        assert (local / "stdout.txt").read_bytes() == b"previous bytes"
        assert runner._output_targets == {
            str(local / "exact.out"): str(remote / "exact.out"),
        }
        assert runner._output_patterns == (str(remote / "data-*.out"),)
        assert runner._cache_targets == {
            str(local / "cache.dat"): (str(remote / "cache.dat"), (2, 3)),
        }
        assert runner._cache_patterns == (str(remote / "restart-*.pot"),)
        assert runner._cache_before == {
            str(remote / "remote-only.dat"): (5, 7),
        }
    finally:
        runner.spec.output_file.close()


def test_absolute_remote_root_needs_no_remote_home_query(tmp_path, monkeypatch):
    endpoint = RemoteEndpointExecutor(_Transport())
    context = endpoint._context(
        endpoint,
        LaunchSpec(local_root=tmp_path, remote_root="/scratch/../work"),
    )
    calls = []

    async def remote_home():
        calls.append("home")
        raise AssertionError("Absolute remote roots must not query remote HOME")

    monkeypatch.setattr(endpoint, "_remote_home", remote_home)
    directory = tmp_path / "calculation"
    directory.mkdir()
    assert asyncio.run(context.map_path(directory)) == "/work/calculation"
    assert asyncio.run(context.map_directory(directory)) == "/work/calculation"
    assert asyncio.run(context.unmap_path("/work/calculation")) == directory
    assert asyncio.run(context._resolved_remote_root()) == "/work"
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
    context = endpoint._context(
        endpoint, LaunchSpec(local_root=tmp_path, remote_root=root),
    )
    calls = []

    async def remote_home():
        calls.append("home")
        return "/home/user"

    monkeypatch.setattr(endpoint, "_remote_home", remote_home)
    assert asyncio.run(context._resolved_remote_root()) == expected
    assert calls == ["home"]


def test_ephemeral_ssh_staging_uses_remote_mktemp_and_cleans_failed_preflight(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace
    from ase2sprkkr.execution.base import ExecutionSpec, SubmissionNotStarted
    from ase2sprkkr.execution.transport import LocalTransport

    endpoint = RemoteEndpointExecutor(LocalTransport())
    calls = []

    async def remote_run(argv, **kwargs):
        calls.append(tuple(argv))
        return SimpleNamespace(stdout=b"/tmp/ase2sprkkr-test1234" + bytes([10]))

    monkeypatch.setattr(endpoint.transport, "run", remote_run)
    launch = LaunchSpec(
        local_root=str(tmp_path), remote_root="/scratch/ignored",
        temporary_directory=True, output_sync="never",
    )
    context = endpoint._context(endpoint, launch)
    spec = ExecutionSpec(
        argv=("true",), input_file=None, output_file=None,
        directory=str(tmp_path), launch=launch,
    )
    runner = context.create_run(spec)

    async def prepared(*args):
        assert await context.map_path(tmp_path / "input.inp") == (
            "/tmp/ase2sprkkr-test1234/input.inp"
        )
        raise OSError("staging interrupted")

    monkeypatch.setattr(runner, "_prepare_request", prepared)
    with pytest.raises(SubmissionNotStarted, match="staging interrupted"):
        asyncio.run(runner._prepare_async())
    assert calls == [
        ("mktemp", "-d", "-t", "ase2sprkkr-XXXXXXXX"),
        ("rm", "-rf", "--", "/tmp/ase2sprkkr-test1234"),
    ]
    assert context._temporary_root is None
