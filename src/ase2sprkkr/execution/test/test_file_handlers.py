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

def test_collect_downloads_shared_remote_source_to_both_local_destinations(tmp_path):
    """Deduplicate transfers, not distinct destinations for one source."""
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = remote / "shared.out"
    source.write_text("result")
    targets = (local / "first.out", local / "second.out")
    manifest = FileManifest(outputs=tuple(
        FileEntry(str(target), str(source), availability_state="pending")
        for target in targets
    ))
    handler = RemoteEndpointExecutor(LocalTransport()).file_handler
    mapping = FileMapping(str(local), str(remote))
    collected, error = asyncio.run(
        handler.collect(mapping, manifest, ExecutionOutcome())
    )

    assert error is None
    assert all(target.read_text() == "result" for target in targets)
    assert [entry.availability_state for entry in collected.outputs] == [
        "available", "available",
    ]

def test_always_input_sync_uploads_without_unnecessary_digest_checks(tmp_path):
    """ALWAYS does not need hashes of either the local or remote input."""
    from ase2sprkkr.execution import FileMapping

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = local / "input.dat"
    source.write_text("input")

    class NoDigestTransport(LocalTransport):
        async def file_digest(self, path):
            pytest.fail(f"ALWAYS must not hash input files: {path}")

    handler = RemoteEndpointExecutor(NoDigestTransport()).file_handler
    mapping = FileMapping(str(local), str(remote), input_sync="always")
    spec = ExecutionSpec(
        argv=("true",), directory=str(local), input_file=str(source),
        output_file=None,
    )
    manifest = asyncio.run(handler.prepare(mapping, spec))

    assert (remote / source.name).read_text() == "input"
    assert manifest.inputs[0].remote_path == str(remote / source.name)


def test_cheap_remote_sync_decisions_do_not_hash_files(tmp_path, monkeypatch):
    """Existence checks suffice for ALWAYS/MISSING and absent CHANGED outputs."""
    from ase2sprkkr.execution import FileMapping
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
        mapping = FileMapping(str(local), str(remote), output_sync=policy)
        manifest = FileManifest(outputs=(
            FileEntry(str(target), str(source), availability_state="pending"),
        ))
        result, error = asyncio.run(handler.collect(
            mapping, manifest, ExecutionOutcome(), required=True,
        ))
        assert error is None
        assert target.read_text() == expected
        assert result.outputs[0].availability_state == "available"


def test_cache_missing_policy_uses_signature_and_local_existence(tmp_path, monkeypatch):
    """Cache baselines matter even when output content hashes are unnecessary."""
    from ase2sprkkr.execution import FileMapping
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
    manifest = FileManifest(caches=(
        FileEntry(str(target), str(source), pre_run_signature=(0, 0)),
    ))
    handler = RemoteEndpointExecutor(NoDigestTransport()).file_handler
    mapping = FileMapping(str(local), str(remote), output_sync="missing")

    first, error = asyncio.run(handler.collect(mapping, manifest, ExecutionOutcome()))
    assert error is None
    assert first.caches[0].availability_state == "skipped"
    assert target.read_text() == "existing"

    target.unlink()
    second, error = asyncio.run(handler.collect(mapping, manifest, ExecutionOutcome()))
    assert error is None
    assert second.caches[0].availability_state == "available"
    assert target.read_text() == "updated"


def test_collect_retries_failed_files_without_redownloading_successes(tmp_path):
    """A terminal reattach preserves a completed transfer even if its source vanished."""
    from ase2sprkkr.execution import FileMapping
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
    mapping = FileMapping(str(local), str(remote), output_sync="always")
    manifest = FileManifest(outputs=tuple(
        FileEntry(str(local / name), str(remote / name),
                  availability_state="pending")
        for name in ("first.out", "second.out")
    ))

    partial, error = asyncio.run(handler.collect(
        mapping, manifest, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, OSError)
    assert [e.availability_state for e in partial.outputs] == ["failed", "available"]
    assert (local / "second.out").read_text() == "second"

    second.unlink()  # The remote file is no longer accessible on reattach.
    transport.fail_first = False
    completed, error = asyncio.run(handler.collect(
        mapping, partial, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert [e.availability_state for e in completed.outputs] == ["available", "available"]
    assert (local / "first.out").read_text() == "first"
    assert (local / "second.out").read_text() == "second"
    assert [e.availability_state for e in partial.outputs] == ["failed", "available"]


def test_collect_retries_available_file_only_if_local_copy_is_missing(tmp_path):
    """An available receipt does not hide a deleted local file."""
    from ase2sprkkr.execution import FileMapping
    from ase2sprkkr.execution.base import ExecutionOutcome

    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    source = remote / "result.dat"
    source.write_text("restored")
    destination = local / "result.dat"
    manifest = FileManifest(outputs=(
        FileEntry(str(destination), str(source), availability_state="available"),
    ))
    handler = RemoteEndpointExecutor(LocalTransport()).file_handler
    mapping = FileMapping(str(local), str(remote), output_sync="always")
    received, error = asyncio.run(handler.collect(
        mapping, manifest, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert destination.read_text() == "restored"
    assert received.outputs[0].availability_state == "available"


def test_collect_does_not_recheck_skipped_terminal_files(tmp_path):
    """Skipping an optional cache/output freezes that decision across attach."""
    from ase2sprkkr.execution import FileMapping
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
        FileMapping(str(local), str(remote)), manifest, ExecutionOutcome(),
    ))
    assert error is None
    assert result == manifest


def test_remote_file_handler_reconciles_terminal_delivery(tmp_path):
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileManifest, FileEntry

    path = tmp_path / "result.dat"
    path.write_text("downloaded")
    available = FileEntry(str(path), "/remote/result.dat", "available")
    pending = FileEntry(str(path), "/remote/result.dat", "pending")
    old = FileManifest(outputs=(available,))
    frozen = FileManifest(outputs=(pending,))

    result = RemoteFileHandler.merge_terminal_files(old, frozen)
    assert result.outputs[0].availability_state == "available"
    assert RemoteFileHandler.mark_pending_failed(frozen, "transfer error").failed_files
    assert RemoteFileHandler.mark_pending_available(frozen).outputs[0].availability_state == "available"
    assert old.outputs == (available,)

    path.unlink()
    assert RemoteFileHandler.merge_terminal_files(old, frozen).outputs == (pending,)


def test_remote_file_handler_does_not_add_files_on_availability_change(tmp_path):
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileManifest, FileEntry

    declared = FileEntry(str(tmp_path / "known"), "/remote/known", "pending")
    original = FileManifest(outputs=(declared,))
    updated = RemoteFileHandler.mark_pending_failed(original, "missing")
    assert len(updated.outputs) == 1
    assert updated.outputs[0].availability_state == "failed"
    assert original.outputs == (declared,)

def test_file_delivery_status_is_not_inferred_from_global_success(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    class Transport:
        async def file_signature(self, path):
            return None
    out = str(tmp_path / "missing.out")
    manifest = FileManifest(stdout=FileEntry(out, "/remote/missing.out", "pending"))
    updated, error = asyncio.run(RemoteFileHandler(Transport()).verify_stream_availability(
        manifest, SimpleNamespace(output_file=None, error_file=None),
    ))
    assert isinstance(error, FileNotFoundError)
    assert updated.stdout.availability_state == "failed"
    assert manifest.stdout.availability_state == "pending"


def test_stream_verification_records_success_and_failure_separately(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest
    from ase2sprkkr.common.file_versions import file_digest

    good = tmp_path / "good.out"
    good.write_bytes(b"good")
    bad = tmp_path / "bad.err"
    class Transport:
        async def file_signature(self, path):
            return (4, good.stat().st_mtime_ns) if path == "/remote/good" else None
        async def file_digest(self, path):
            return file_digest(good) if path == "/remote/good" else None
    manifest = FileManifest(
        stdout=FileEntry(str(good), "/remote/good", "pending"),
        stderr=FileEntry(str(bad), "/remote/bad", "pending"),
    )
    updated, error = asyncio.run(RemoteFileHandler(Transport()).verify_stream_availability(
        manifest, SimpleNamespace(output_file=None, error_file=None),
    ))
    assert isinstance(error, FileNotFoundError)
    assert updated.stdout.availability_state == "available"
    assert updated.stderr.availability_state == "failed"



def test_collect_does_not_mark_untouched_streams_available(tmp_path):
    """Stream availability is owned by FOLLOW verification, not collect."""
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest, FileMapping

    local = tmp_path / "local"
    local.mkdir()
    remote = tmp_path / "remote"
    remote.mkdir()
    manifest = FileManifest(
        stdout=FileEntry(str(local / "out.txt"), str(remote / "out.txt"), "pending"),
        mapping=FileMapping(str(local), str(remote)),
    )

    class NoTransfer:
        pass

    updated, error = asyncio.run(
        RemoteFileHandler(NoTransfer()).collect(
            manifest.mapping, manifest,
            SimpleNamespace(output_file=str(local / "out.txt"), error_file=None),
        )
    )
    assert error is None
    assert updated.stdout.availability_state == "pending"


def test_preparation_uses_manifest_paths_and_effective_streams(tmp_path):
    """Relative stdin, extra inputs and caches survive wildcard cleanup."""
    protected = ("main.in", "stdin.in", "extra.in", "cache.pot")
    removed = ("obsolete.dat", "redirect.out", "redirect.err")
    untouched = ("original.out", "original.err")
    for name in (*protected, *removed, *untouched):
        (tmp_path / name).write_text(name)
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file="main.in", stdin="stdin.in",
        input_files=("extra.in",), cache_files=("cache.pot",),
        output_files=("*.dat",), output_file="original.out", error_file="original.err",
    )

    prepare_local_files(spec, output_file="redirect.out", error_file="redirect.err")

    assert all((tmp_path / name).read_text() == name for name in (*protected, *untouched))
    assert all(not (tmp_path / name).exists() for name in removed)


def test_preparation_preserves_open_streams_inside_output_wildcard(tmp_path):
    """Keep opened stdout/stderr while clearing other matching old outputs."""
    for name in ("stdout.log", "stderr.log", "obsolete.log"):
        (tmp_path / name).write_text(name)
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None,
        output_file="stdout.log", error_file="stderr.log",
        output_files=("*.log",),
    )

    prepare_local_files(spec, preserve_open_streams=True)

    assert (tmp_path / "stdout.log").read_text() == "stdout.log"
    assert (tmp_path / "stderr.log").read_text() == "stderr.log"
    assert not (tmp_path / "obsolete.log").exists()
