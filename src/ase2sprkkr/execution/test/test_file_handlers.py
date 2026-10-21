"""Shared file strategies have no mutable state belonging to individual runs."""
import asyncio

import pytest

from ase2sprkkr.execution import (
    ExecutionSpec, FileEntry, FileManifest, LaunchSpec, LocalTransport,
    RemoteEndpointExecutor,
)
from ase2sprkkr.execution.file_availability import prepare_local_files, discover_local_files


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

    prepare_local_files(spec)
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
        prepare_local_files(spec)
    assert original.read_text() == "preserve"


def test_local_file_handler_discovers_without_mutating_manifest(tmp_path):
    out = tmp_path / "done.dat"
    out.write_bytes(b"done")
    original = FileManifest(outputs=(
        FileEntry(local_path=str(out), availability_state="pending"),
    ))
    observed = discover_local_files(original)
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
        first = await a.file_handler.prepare(await endpoint.resolve_mapping(a.spec.launch), a.spec)
        second = await b.file_handler.prepare(await endpoint.resolve_mapping(b.spec.launch), b.spec)
        return first, second

    first, second = asyncio.run(prepare())
    assert first is not second
    assert first.mapping == second.mapping


def test_materialized_manifest_freezes_membership(tmp_path):
    from ase2sprkkr.execution.file_availability import materialize_manifest
    from ase2sprkkr.execution import FileMapping

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "result-a.dat").write_text("first")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "result-*.dat"),
                           str(remote / "result-*.dat"), availability_state="pending"),),
        mapping=FileMapping(str(local), str(remote)),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    assert len(frozen.outputs) == 1
    assert frozen.outputs[0].source == str(local / "result-*.dat")
    (remote / "result-b.dat").write_text("late")
    second, error = materialize_manifest(frozen, remote=True)
    assert error is None
    assert second == frozen


def test_empty_wildcard_is_not_materialization_failure(tmp_path):
    from ase2sprkkr.execution.file_availability import materialize_manifest
    from ase2sprkkr.execution import FileMapping

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.out"), str(remote / "*.out"),
                           availability_state="pending"),),
        mapping=FileMapping(str(local), str(remote)),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    assert not frozen.outputs
    assert not frozen.has_wildcards()


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
    context = asyncio.run(endpoint.resolve_mapping(LaunchSpec(
        local_root=str(local), remote_root=str(remote),
    )))
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
            FileEntry(
                str(local / "restart.pot"), str(remote / "restart.pot"),
                availability_state="pending",
            ),
        ),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_mapping(LaunchSpec(
        local_root=str(local), remote_root=str(remote), output_sync="always",
    )))
    collected, error = asyncio.run(endpoint.file_handler.collect(
        context, manifest, ExecutionOutcome(), required=True,
    ))

    assert isinstance(error, FileNotFoundError)
    assert [entry.availability_state for entry in collected.outputs] == [
        "available", "failed", "available",
    ]
    assert collected.caches[0].availability_state == "available"
    assert (local / "first.out").read_text() == "first"
    assert (local / "last.out").read_text() == "last"
    assert (local / "restart.pot").read_text() == "restart"
    assert all(entry.availability_state == "pending" for entry in manifest.outputs)


def test_materialized_manifest_can_download_and_rehydrate(tmp_path):
    from dataclasses import asdict
    from ase2sprkkr.execution.file_availability import materialize_manifest
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "one.out").write_text("one")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.out"), str(remote / "*.out"),
                           availability_state="pending"),),
        mapping=FileMapping(str(local), str(remote)),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_mapping(
        LaunchSpec(local_root=str(local), remote_root=str(remote)),
    ))
    completed, error = asyncio.run(
        endpoint.file_handler.collect(context, frozen, ExecutionOutcome())
    )
    assert error is None
    assert (local / "one.out").read_text() == "one"
    restored = FileManifest(**asdict(completed))
    (remote / "two.out").write_text("late")
    assert materialize_manifest(restored, remote=True) == (restored, None)


def test_collect_rejects_materialized_cache_output_collision_before_download(tmp_path):
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.base import ExecutionOutcome
    from ase2sprkkr.execution.file_availability import materialize_manifest

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "result.dat").write_text("remote value")
    destination = local / "result.dat"
    destination.write_text("original local value")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.dat"), str(remote / "*.dat"),
                           availability_state="pending"),),
        caches=(FileEntry(str(destination), str(remote / "result.dat"),
                                     availability_state="pending"),),
        mapping=FileMapping(str(local), str(remote)),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_mapping(LaunchSpec(
        local_root=str(local), remote_root=str(remote),
    )))
    with pytest.raises(ValueError, match="Conflicting output/cache file destination"):
        asyncio.run(endpoint.file_handler.collect(
            context, frozen, ExecutionOutcome(), required=True,
        ))
    assert destination.read_text() == "original local value"


def test_materialization_assigns_cache_baselines_to_concrete_members(tmp_path):
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.file_availability import materialize_manifest

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "restart-1.pot").write_text("updated")
    (remote / "restart-2.pot").write_text("new")
    previous = str(remote / "restart-1.pot")
    manifest = FileManifest(
        caches=(
            FileEntry(str(local / "restart-*.pot"),
                                 str(remote / "restart-*.pot"),
                                 availability_state="pending"),
            FileEntry(remote_path=previous, pre_run_signature=(3, 17)),
        ),
        mapping=FileMapping(str(local), str(remote)),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    assert {c.local_path for c in frozen.caches} == {
        str(local / "restart-1.pot"), str(local / "restart-2.pot"),
    }
    assert {c.remote_path: c.pre_run_signature for c in frozen.caches} == {
        previous: (3, 17), str(remote / "restart-2.pot"): None,
    }


def test_materialization_keeps_failed_wildcard_declaration(tmp_path, monkeypatch):
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution import file_availability

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    manifest = FileManifest(
        outputs=(
            FileEntry(str(local / "broken-*.out"), str(remote / "broken-*.out"),
                      availability_state="pending"),
            FileEntry(str(local / "complete.out"), str(remote / "complete.out"),
                      availability_state="pending"),
        ),
        mapping=FileMapping(str(local), str(remote)),
    )

    def broken_scan(source, root):
        raise PermissionError("directory cannot be listed")

    monkeypatch.setattr(file_availability, "expand_local_files", broken_scan)
    frozen, error = file_availability.materialize_manifest(manifest, remote=True)
    assert isinstance(error, PermissionError)
    assert frozen.outputs == manifest.outputs
    assert frozen.has_wildcards()


def test_collect_rejects_two_remote_cache_sources_with_one_destination(tmp_path):
    from ase2sprkkr.execution.base import ExecutionOutcome

    local = tmp_path / "restart.pot"
    local.write_text("preserve")
    manifest = FileManifest(caches=(
        FileEntry(str(local), "/remote/cache-a.pot"),
        FileEntry(str(local), "/remote/cache-b.pot"),
    ))
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_mapping(LaunchSpec(
        local_root=str(tmp_path), remote_root="/remote",
    )))
    with pytest.raises(ValueError, match="Conflicting cache sources"):
        asyncio.run(endpoint.file_handler.collect(
            context, manifest, ExecutionOutcome(),
        ))
    assert local.read_text() == "preserve"
