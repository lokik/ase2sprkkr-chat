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
        first = await a.file_handler.prepare(FileManifest.from_spec(a.spec).with_remote(local_root=str(tmp_path), remote_root=str(tmp_path / 'target')), a.spec)
        second = await b.file_handler.prepare(FileManifest.from_spec(b.spec).with_remote(local_root=str(tmp_path), remote_root=str(tmp_path / 'target')), b.spec)
        return first, second

    first, second = asyncio.run(prepare())
    assert first is not second
    assert (first.local_root, first.remote_root) == (second.local_root, second.remote_root)


def test_materialized_manifest_freezes_membership(tmp_path):
    from ase2sprkkr.execution.file_availability import materialize_manifest

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "result-a.dat").write_text("first")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "result-*.dat"),
                           str(remote / "result-*.dat"), availability_state="pending"),),
        local_root=str(local), remote_root=str(remote),
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

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.out"), str(remote / "*.out"),
                           availability_state="pending"),),
        local_root=str(local), remote_root=str(remote),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    assert not frozen.outputs
    assert not frozen.has_wildcards()


def test_collect_continues_after_one_output_download_fails(tmp_path):
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
        local_root=str(local), remote_root=str(remote),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_files(LaunchSpec(
        local_root=str(local), remote_root=str(remote),
    )))
    original = endpoint.file_handler._sync_file

    async def fail_first(*args, **kwargs):
        if args[1].remote_path == str(first):
            raise OSError("broken transfer")
        return await original(*args, **kwargs)

    endpoint.file_handler._sync_file = fail_first
    captured, error = asyncio.run(
        endpoint.file_handler.collect(manifest, ExecutionOutcome())
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
        local_root=str(local), remote_root=str(remote), output_sync="always",
        caches=(
            FileEntry(
                str(local / "restart.pot"), str(remote / "restart.pot"),
                availability_state="pending",
            ),
        ),
    )
    endpoint = RemoteEndpointExecutor(LocalTransport())

    collected, error = asyncio.run(endpoint.file_handler.collect(
        manifest, ExecutionOutcome(), required=True,
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
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "one.out").write_text("one")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.out"), str(remote / "*.out"),
                           availability_state="pending"),),
        local_root=str(local), remote_root=str(remote),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    endpoint = RemoteEndpointExecutor(LocalTransport())
    context = asyncio.run(endpoint.resolve_files(
        LaunchSpec(local_root=str(local), remote_root=str(remote)),
    ))
    completed, error = asyncio.run(
        endpoint.file_handler.collect(frozen, ExecutionOutcome())
    )
    assert error is None
    assert (local / "one.out").read_text() == "one"
    restored = FileManifest(**asdict(completed))
    (remote / "two.out").write_text("late")
    assert materialize_manifest(restored, remote=True) == (restored, None)


def test_collect_rejects_materialized_cache_output_collision_before_download(tmp_path):
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
        local_root=str(local), remote_root=str(remote),
    )
    # Wildcard expansion creates the collision, which must be rejected
    # immediately rather than producing a manifest that can be downloaded.
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        materialize_manifest(manifest, remote=True)
    assert destination.read_text() == "original local value"


def test_materialization_assigns_cache_baselines_to_concrete_members(tmp_path):
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
        local_root=str(local), remote_root=str(remote),
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
        local_root=str(local), remote_root=str(remote),
    )

    def broken_scan(source, root):
        raise PermissionError("directory cannot be listed")

    monkeypatch.setattr(file_availability, "expand_local_files", broken_scan)
    frozen, error = file_availability.materialize_manifest(manifest, remote=True)
    assert isinstance(error, PermissionError)
    assert frozen.outputs == manifest.outputs
    assert frozen.has_wildcards()


def test_manifest_rejects_two_remote_cache_sources_with_one_destination(tmp_path):
    local = tmp_path / "restart.pot"
    local.write_text("preserve")
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(caches=(
            FileEntry(str(local), "/remote/cache-a.pot"),
            FileEntry(str(local), "/remote/cache-b.pot"),
        ))
    assert local.read_text() == "preserve"


def test_manifest_rejects_shared_remote_source_with_distinct_local_destinations(tmp_path):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = remote / "shared.out"
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(outputs=(
            FileEntry(str(local / "first.out"), str(source)),
            FileEntry(str(local / "second.out"), str(source)),
        ))


def test_always_input_sync_uploads_without_unnecessary_digest_checks(tmp_path):
    """ALWAYS does not need hashes of either the local or remote input."""

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = local / "input.dat"
    source.write_text("input")

    class NoDigestTransport(LocalTransport):
        async def file_digest(self, path):
            pytest.fail(f"ALWAYS must not hash input files: {path}")

    handler = RemoteEndpointExecutor(NoDigestTransport()).file_handler
    spec = ExecutionSpec(
        argv=("true",), directory=str(local), input_file=str(source),
    )
    manifest = asyncio.run(handler.prepare(FileManifest.from_spec(spec).with_remote(local_root=str(local), remote_root=str(remote), input_sync="always"), spec))

    assert (remote / source.name).read_text() == "input"
    assert manifest.inputs[0].remote_path == str(remote / source.name)


def test_cheap_remote_sync_decisions_do_not_hash_files(tmp_path, monkeypatch):
    """Existence checks suffice for ALWAYS/MISSING and absent CHANGED outputs."""
    from ase2sprkkr.execution.base import ExecutionOutcome
    from ase2sprkkr.execution import remote_file_handler

    def unexpected_digest(*args, **kwargs):
        pytest.fail("A content digest is unnecessary for this sync policy")

    class NoDigestTransport(LocalTransport):
        async def file_digest(self, path):
            unexpected_digest(path)

    monkeypatch.setattr(remote_file_handler, "file_digest", unexpected_digest)
    handler = RemoteEndpointExecutor(NoDigestTransport()).file_handler

    cases = (
        ("always", True, "remote"),
        ("missing", True, "local"),
        ("missing", False, "remote"),
        ("changed", False, "remote"),
    )
    for index, (policy, local_exists, expected) in enumerate(cases):
        local, remote = tmp_path / f"local-{index}", tmp_path / f"remote-{index}"
        local.mkdir()
        remote.mkdir()
        target, source = local / "output.dat", remote / "output.dat"
        source.write_text("remote")
        if local_exists:
            target.write_text("local")
        manifest = FileManifest(local_root=str(local), remote_root=str(remote), output_sync=policy, outputs=(
            FileEntry(str(target), str(source), availability_state="pending"),
        ))
        result, error = asyncio.run(handler.collect(
            manifest, ExecutionOutcome(), required=True,
        ))
        assert error is None
        assert target.read_text() == expected
        assert result.outputs[0].availability_state == "available"


def test_cache_missing_policy_uses_signature_and_local_existence(tmp_path, monkeypatch):
    """Cache baselines matter even when output content hashes are unnecessary."""
    from ase2sprkkr.execution.base import ExecutionOutcome
    from ase2sprkkr.execution import remote_file_handler

    def unexpected_digest(*args, **kwargs):
        pytest.fail("Cache sync must not hash file contents for MISSING")

    class NoDigestTransport(LocalTransport):
        async def file_digest(self, path):
            unexpected_digest(path)

    monkeypatch.setattr(remote_file_handler, "file_digest", unexpected_digest)
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = remote / "restart.pot"
    source.write_text("updated")
    target = local / "restart.pot"
    target.write_text("existing")
    manifest = FileManifest(
        caches=(FileEntry(str(target), str(source), pre_run_signature=(0, 0)),),
        local_root=str(local), remote_root=str(remote), output_sync="missing",
    )
    handler = RemoteEndpointExecutor(NoDigestTransport()).file_handler

    first, error = asyncio.run(handler.collect(manifest, ExecutionOutcome()))
    assert error is None
    assert first.caches[0].availability_state == "skipped"
    assert target.read_text() == "existing"

    target.unlink()
    second, error = asyncio.run(handler.collect(manifest, ExecutionOutcome()))
    assert error is None
    assert second.caches[0].availability_state == "available"
    assert target.read_text() == "updated"


def test_collect_retries_failed_files_without_redownloading_successes(tmp_path):
    """A terminal reattach preserves a completed transfer even if its source vanished."""
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    first, second = remote / "first.out", remote / "second.out"
    first.write_text("first")
    second.write_text("second")

    class FlakyTransport(LocalTransport):
        fail_first = True

        async def get_file(self, source, destination):
            if str(source) == str(first) and self.fail_first:
                raise OSError("temporary transfer failure")
            await super().get_file(source, destination)

    transport = FlakyTransport()
    handler = RemoteEndpointExecutor(transport).file_handler
    manifest = FileManifest(local_root=str(local), remote_root=str(remote),
                            output_sync="always", outputs=tuple(
        FileEntry(str(local / name), str(remote / name),
                  availability_state="pending")
        for name in ("first.out", "second.out")
    ))

    partial, error = asyncio.run(handler.collect(
        manifest, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, OSError)
    assert [e.availability_state for e in partial.outputs] == ["failed", "available"]
    assert (local / "second.out").read_text() == "second"

    second.unlink()  # The remote file is no longer accessible on reattach.
    transport.fail_first = False
    completed, error = asyncio.run(handler.collect(
        partial, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert [e.availability_state for e in completed.outputs] == ["available", "available"]
    assert (local / "first.out").read_text() == "first"
    assert (local / "second.out").read_text() == "second"
    assert [e.availability_state for e in partial.outputs] == ["failed", "available"]


def test_collect_retries_available_file_only_if_local_copy_is_missing(tmp_path):
    """An available receipt does not hide a deleted local file."""
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = remote / "result.dat"
    source.write_text("restored")
    destination = local / "result.dat"
    manifest = FileManifest(
        outputs=(FileEntry(str(destination), str(source), availability_state="available"),),
        local_root=str(local), remote_root=str(remote), output_sync="always",
    )
    handler = RemoteEndpointExecutor(LocalTransport()).file_handler
    received, error = asyncio.run(handler.collect(
        manifest, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert destination.read_text() == "restored"
    assert received.outputs[0].availability_state == "available"


def test_collect_does_not_recheck_skipped_terminal_files(tmp_path):
    """Skipping an optional cache/output freezes that decision across attach."""
    from ase2sprkkr.execution.base import ExecutionOutcome

    class NoRemoteChecks(LocalTransport):
        async def file_signature(self, path):
            pytest.fail(f"Skipped files must not be rechecked: {path}")

        async def file_digest(self, path):
            pytest.fail(f"Skipped files must not be rehashed: {path}")

        async def get_file(self, source, destination):
            pytest.fail(f"Skipped files must not be downloaded: {source}")

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "missing.out"), str(remote / "missing.out"),
                           availability_state="skipped"),),
        caches=(FileEntry(str(local / "restart.pot"), str(remote / "restart.pot"),
                          availability_state="skipped"),),
    )
    handler = RemoteEndpointExecutor(NoRemoteChecks()).file_handler
    result, error = asyncio.run(handler.collect(
        manifest, ExecutionOutcome(),
    ))
    assert error is None
    assert result == manifest
