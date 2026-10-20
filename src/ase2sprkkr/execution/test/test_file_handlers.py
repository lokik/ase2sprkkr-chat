"""Shared file strategies have no mutable state belonging to individual runs."""
import asyncio

import pytest

from ase2sprkkr.execution import (
    ExecutionSpec, FileEntry, FileManifest, LaunchSpec, LocalTransport,
    RemoteEndpointExecutor,
)
from ase2sprkkr.execution.local_file_handler import LocalFileHandler


def test_local_file_handler_preserves_input_and_cache(tmp_path):
    source = tmp_path / "input.dat"
    cache = tmp_path / "cache.dat"
    old = tmp_path / "old.out"
    for p in (source, cache, old):
        p.write_text(p.name)
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=str(source),
        output_file=None, input_files=(), output_files=("*.dat", "old.out"),
        cache_files=(str(cache),),
    )

    LocalFileHandler.prepare_target(spec)
    assert source.read_text() == "input.dat"
    assert cache.read_text() == "cache.dat"
    assert not old.exists()


def test_local_file_handler_rejects_overwriting_input(tmp_path):
    original = tmp_path / "same.in"
    original.write_text("preserve")
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=str(original),
        output_file=str(original),
    )
    with pytest.raises(ValueError, match="overlaps protected"):
        LocalFileHandler.prepare_target(spec)
    assert original.read_text() == "preserve"


def test_local_file_handler_discovers_without_mutating_manifest(tmp_path):
    out = tmp_path / "done.dat"
    out.write_bytes(b"done")
    original = FileManifest(outputs=(
        FileEntry(local_path=str(out), availability_state="pending"),
    ))
    observed = LocalFileHandler.collect(original)
    assert observed.outputs[0].availability_state == "available"
    assert original.outputs[0].availability_state == "pending"


def test_remote_handler_is_shared_without_run_state(tmp_path):
    endpoint = RemoteEndpointExecutor(LocalTransport())
    spec_a = ExecutionSpec(
        argv=("echo",), input_file=None, output_file=None,
        directory=str(tmp_path / "one"),
        launch=LaunchSpec(local_root=str(tmp_path),
                          remote_root=str(tmp_path / "target")),
    )
    spec_b = ExecutionSpec(
        argv=("echo",), input_file=None, output_file=None,
        directory=str(tmp_path / "two"),
        launch=spec_a.launch,
    )
    a, b = endpoint.create_run(spec_a), endpoint.create_run(spec_b)
    assert a.file_handler is endpoint.file_handler is b.file_handler
    assert a.files is None and b.files is None

    async def prepare():
        first = await a.file_handler.prepare(a.executor, a.spec, a.result)
        second = await b.file_handler.prepare(b.executor, b.spec, b.result)
        return first, second

    first, second = asyncio.run(prepare())
    assert first is not second
    assert first.mapping == second.mapping


def test_terminal_wildcard_membership_is_immutable_after_later_files(tmp_path):
    from ase2sprkkr.execution.base import ExecutionOutcome
    from ase2sprkkr.execution import FileMapping

    local = tmp_path / "local"
    target = tmp_path / "remote"
    local.mkdir()
    target.mkdir()
    pattern = str(local / "result-*.dat")
    remote_pattern = str(target / "result-*.dat")
    first = str(target / "result-a.dat")
    manifest = FileManifest(
        outputs=(FileEntry(pattern, remote_pattern, availability_state="pending"),),
        mapping=FileMapping(str(local), str(target)),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    spec = ExecutionSpec(argv=(), directory=str(local), input_file=None, output_file=None)
    manifest, error = endpoint.file_handler.freeze_from_outcome(
        manifest, {"outputs": [{"source": remote_pattern, "matches": [first], "error": None}], "caches": []},
    )
    (target / "result-a.dat").write_text("first")
    (target / "result-b.dat").write_text("later")
    asyncio.run(endpoint.file_handler.collect(
        endpoint._context(endpoint, spec.launch), manifest, ExecutionOutcome(),
        required=False,
    ))
    assert (local / "result-a.dat").read_text() == "first"
    assert not (local / "result-b.dat").exists()
    assert error is None
    assert not manifest.has_wildcards()
    assert tuple(x.remote_path for x in manifest.outputs if x.source) == (first,)
    saved = manifest
    asyncio.run(endpoint.file_handler.collect(
        endpoint._context(endpoint, spec.launch), saved, ExecutionOutcome(),
        required=False,
    ))
    assert not (local / "result-b.dat").exists()


def test_frozen_empty_wildcard_does_not_match_future_files(tmp_path):
    from ase2sprkkr.execution.base import ExecutionOutcome
    from ase2sprkkr.execution.remote_file_handler import FileManifest
    from ase2sprkkr.execution import FileMapping
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.out"), str(remote / "*.out"),
                           availability_state="pending"),),
        mapping=FileMapping(str(local), str(remote)),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    spec = ExecutionSpec(argv=(), directory=str(local), input_file=None, output_file=None)
    manifest, error = endpoint.file_handler.freeze_from_outcome(
        manifest, {"outputs": [{"source": str(remote / "*.out"), "matches": [], "error": None}], "caches": []},
    )
    (remote / "too-late.out").write_text("late")
    asyncio.run(endpoint.file_handler.collect(
        endpoint._context(endpoint, spec.launch), manifest, ExecutionOutcome(),
        required=False,
    ))
    assert not (local / "too-late.out").exists()
    assert error is None
    assert not manifest.has_wildcards()

def test_wildcard_freeze_returns_partial_manifest_and_error(tmp_path):
    from ase2sprkkr.execution import FileMapping

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    declaration = FileEntry(str(local / "out-*.dat"), str(remote / "out-*.dat"))
    missing = FileEntry(str(local / "broken-*.dat"), str(remote / "broken-*.dat"))
    manifest = FileManifest(
        outputs=(declaration, missing),
        mapping=FileMapping(str(local), str(remote)),
    )
    first = str(remote / "out-one.dat")
    frozen, error = RemoteEndpointExecutor(LocalTransport()).file_handler.freeze_from_outcome(
        manifest,
        {"outputs": [
            {"source": declaration.remote_path, "matches": (first,), "error": None},
            {"source": missing.remote_path, "matches": (), "error": "permission denied"},
        ], "caches": []},
    )
    assert isinstance(error, RuntimeError)
    assert "permission denied" in str(error)
    assert len(frozen.outputs) == 1
    assert frozen.outputs[0].remote_path == first
    assert frozen.resolution_errors == ((missing.local_path, "permission denied"),)
    assert frozen.outputs[0].source == declaration.local_path
    assert not frozen.has_wildcards()
    assert manifest.has_wildcards()


def test_freeze_with_no_wildcards_does_not_require_backend_metadata(tmp_path):
    manifest = FileManifest(
        outputs=(FileEntry(str(tmp_path / "one.out"), "/target/one.out"),),
    )
    frozen, error = RemoteEndpointExecutor(LocalTransport()).file_handler.freeze_from_outcome(
        manifest, None,
    )
    assert frozen is manifest
    assert error is None


def test_collect_continues_after_one_output_download_fails(tmp_path):
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.base import ExecutionOutcome
    import types

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    first, second = remote / "first.out", remote / "second.out"
    first.write_text("one")
    second.write_text("two")
    manifest = FileManifest(
        outputs=(
            FileEntry(str(local / first.name), str(first), availability_state="pending"),
            FileEntry(str(local / second.name), str(second), availability_state="pending"),
        ),
        mapping=FileMapping(str(local), str(remote)),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = endpoint._context(endpoint, LaunchSpec(
        local_root=str(local), remote_root=str(remote),
    ))
    original = endpoint.file_handler._sync_file

    async def fail_first(*args, **kwargs):
        if args[1].remote_path == str(first):
            raise OSError("broken transfer")
        return await original(*args, **kwargs)

    endpoint.file_handler._sync_file = fail_first
    captured, error = asyncio.run(
        endpoint.file_handler.collect(context, manifest, ExecutionOutcome())
    )
    assert isinstance(error, OSError)
    assert captured.outputs[0].availability_state == "failed"
    assert captured.outputs[1].availability_state == "available"
    assert (local / second.name).read_text() == "two"


def test_collect_keeps_later_outputs_and_caches_after_missing_required_file(tmp_path):
    """A missing mandatory output fails the run, not subsequent transfers."""
    from ase2sprkkr.execution import CacheEntry
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "first.out").write_text("first")
    (remote / "last.out").write_text("last")
    (remote / "restart.pot").write_text("restart")

    manifest = FileManifest(
        outputs=tuple(
            FileEntry(
                str(local / name), str(remote / name),
                availability_state="pending",
            )
            for name in ("first.out", "missing.out", "last.out")
        ),
        caches=(
            CacheEntry(FileEntry(
                str(local / "restart.pot"), str(remote / "restart.pot"),
                availability_state="pending",
            )),
        ),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = endpoint._context(endpoint, LaunchSpec(
        local_root=str(local), remote_root=str(remote), output_sync="always",
    ))
    collected, error = asyncio.run(endpoint.file_handler.collect(
        context, manifest, ExecutionOutcome(), required=True,
    ))

    assert isinstance(error, FileNotFoundError)
    assert [entry.availability_state for entry in collected.outputs] == [
        "available", "failed", "available",
    ]
    assert collected.caches[0].file.availability_state == "available"
    assert (local / "first.out").read_text() == "first"
    assert (local / "last.out").read_text() == "last"
    assert (local / "restart.pot").read_text() == "restart"
    assert all(entry.availability_state == "pending" for entry in manifest.outputs)
