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
        first = await a.file_handler.prepare(
            a.executor, a.spec, a.result, "run-a", str(tmp_path / "target" / "one"),
        )
        second = await b.file_handler.prepare(
            b.executor, b.spec, b.result, "run-b", str(tmp_path / "target" / "two"),
        )
        return first, second

    first, second = asyncio.run(prepare())
    assert first[0]["directory"] != second[0]["directory"]
    assert first[0]["id"] == "run-a" and second[0]["id"] == "run-b"


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
    manifest = endpoint.file_handler.freeze_from_outcome(
        manifest, {"outputs": [first], "caches": []},
    )
    (target / "result-a.dat").write_text("first")
    (target / "result-b.dat").write_text("later")
    asyncio.run(endpoint.file_handler.collect(
        endpoint._context(endpoint, spec.launch), manifest, ExecutionOutcome(),
        required=False,
    ))
    assert (local / "result-a.dat").read_text() == "first"
    assert not (local / "result-b.dat").exists()
    assert manifest.wildcards_frozen
    assert tuple(x.remote_path for x in manifest.outputs if x.source_pattern) == (first,)
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
    manifest = endpoint.file_handler.freeze_from_outcome(
        manifest, {"outputs": [], "caches": []},
    )
    (remote / "too-late.out").write_text("late")
    asyncio.run(endpoint.file_handler.collect(
        endpoint._context(endpoint, spec.launch), manifest, ExecutionOutcome(),
        required=False,
    ))
    assert not (local / "too-late.out").exists()
    assert manifest.wildcards_frozen


def test_remote_collect_keeps_transferring_after_required_output_fails(tmp_path):
    from ase2sprkkr.execution import CacheEntry
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "first.out").write_text("first")
    (remote / "last.out").write_text("last")
    (remote / "restart.pot").write_text("restart")
    manifest = FileManifest(
        outputs=tuple(FileEntry(
            local_path=str(local / name), remote_path=str(remote / name),
            availability_state="pending",
        ) for name in ("first.out", "missing.out", "last.out")),
        caches=(CacheEntry(FileEntry(
            local_path=str(local / "restart.pot"),
            remote_path=str(remote / "restart.pot"),
            availability_state="pending",
        )),),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = endpoint._context(endpoint, LaunchSpec(
        local_root=local, remote_root=str(remote), output_sync="always",
    ))
    updated, error = asyncio.run(endpoint.file_handler.collect(
        context, manifest, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, FileNotFoundError)
    assert [entry.availability_state for entry in updated.outputs] == [
        "available", "failed", "available",
    ]
    assert updated.caches[0].file.availability_state == "available"
    assert (local / "first.out").read_text() == "first"
    assert (local / "last.out").read_text() == "last"
    assert (local / "restart.pot").read_text() == "restart"
    assert all(entry.availability_state == "pending" for entry in manifest.outputs)


def test_remote_collect_marks_partial_wildcard_failed_but_keeps_children(tmp_path):
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "present.dat").write_text("present")
    manifest = FileManifest(
        outputs=(FileEntry(
            local_path=str(local / "*.dat"),
            remote_path=str(remote / "*.dat"),
            availability_state="pending",
        ),),
        mapping=FileMapping(str(local), str(remote)),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    manifest = endpoint.file_handler.freeze_from_outcome(manifest, {
        "outputs": [str(remote / "missing.dat"), str(remote / "present.dat")],
        "caches": [],
    })
    context = endpoint._context(endpoint, LaunchSpec(
        local_root=local, remote_root=str(remote), output_sync="always",
    ))
    updated, error = asyncio.run(endpoint.file_handler.collect(
        context, manifest, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, FileNotFoundError)
    by_path = {entry.local_path: entry.availability_state
               for entry in updated.outputs}
    assert by_path[str(local / "*.dat")] == "failed"
    assert by_path[str(local / "missing.dat")] == "failed"
    assert by_path[str(local / "present.dat")] == "available"
    assert (local / "present.dat").read_text() == "present"
