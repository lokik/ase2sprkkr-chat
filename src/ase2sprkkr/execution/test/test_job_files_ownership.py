"""Stable ownership boundary for run file operations."""

from ase2sprkkr.execution import JobFiles
from ase2sprkkr.execution.job_files import JobFiles as ModuleJobFiles
from ase2sprkkr.execution.manifest_merge import merge_snapshot, merge_terminal_files, discard_capture
from ase2sprkkr.execution.run_record import RunRecord


def test_job_files_public_identity():
    assert JobFiles is ModuleJobFiles
    assert JobFiles.__module__ == "ase2sprkkr.execution.job_files"
    assert not hasattr(RunRecord, "job_files")


def test_terminal_manifest_merge_preserves_only_verified_local_delivery(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    good = tmp_path / "good.dat"
    good.write_text("complete")
    vanished = tmp_path / "gone.dat"
    previous = FileManifest(outputs=(
        FileEntry(str(good), "/remote/good", availability_state="available"),
        FileEntry(str(vanished), "/remote/gone", availability_state="available"),
    ))
    frozen = FileManifest(outputs=(
        FileEntry(str(good), "/remote/good", availability_state="pending"),
        FileEntry(str(vanished), "/remote/gone", availability_state="pending"),
    ))
    merged = merge_terminal_files(previous, frozen)
    assert [file.availability_state for file in merged.outputs] == [
        "available", "pending",
    ]
    assert previous.outputs[1].availability_state == "available"


def test_terminal_manifest_merge_never_overwrites_frozen_failure(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = tmp_path / "output.dat"
    path.write_text("previous")
    previous = FileManifest(outputs=(
        FileEntry(str(path), "/remote/output", availability_state="available"),
    ))
    frozen = FileManifest(outputs=(
        FileEntry(str(path), "/remote/output", availability_state="failed"),
    ))
    assert merge_terminal_files(previous, frozen) == frozen


def test_job_files_coordinates_remote_collection_and_verification():
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    entry = FileEntry("/tmp/output.dat", "/remote/output.dat", "pending")
    pending = FileManifest(outputs=(entry,))

    class Handler:
        def __init__(self):
            self.calls = []
        async def collect(self, mapping, manifest, outcome, *, required):
            self.calls.append(("collect", required))
            return manifest.with_file_availability(
                entry.local_path, "available", remote_path=entry.remote_path,
            ), None
        async def verify_stream_availability(self, manifest, spec):
            self.calls.append(("verify",))
            return manifest, None

    handler = Handler()
    owner = JobFiles(pending)
    error = asyncio.run(owner.collect_remote(
        pending, handler, SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    out = owner.manifest
    assert error is None
    assert out.outputs[0].availability_state == "available"
    assert handler.calls == [("collect", True), ("verify",)]
    assert pending.outputs[0].availability_state == "pending"


def test_job_files_verifies_streams_after_failed_transfer():
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    entry = FileEntry("/tmp/output.dat", "/remote/output.dat", "pending")
    files = FileManifest(outputs=(entry,))
    cause = OSError("download failed")

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            return manifest.with_file_availability(
                entry.local_path, "failed", str(cause), remote_path=entry.remote_path,
            ), cause
        async def verify_stream_availability(self, manifest, spec):
            assert manifest.outputs[0].availability_state == "failed"
            return manifest, None

    owner = JobFiles(files)
    error = asyncio.run(owner.collect_remote(
        files, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    updated = owner.manifest
    assert error is cause
    assert updated.outputs[0].availability_state == "failed"


def test_job_files_preserves_transfer_error_and_verified_stream(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    stream = FileEntry(str(tmp_path / "stdout"), "/remote/stdout", "pending")
    output = FileEntry(str(tmp_path / "result"), "/remote/result", "pending")
    files = FileManifest(stdout=stream, outputs=(output,))
    cause = OSError("result transfer failed")

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            return manifest.with_file_availability(
                output.local_path, "failed", str(cause),
                remote_path=output.remote_path,
            ), cause

        async def verify_stream_availability(self, manifest, spec):
            return manifest.with_file_availability(
                stream.local_path, "available", remote_path=stream.remote_path,
            ), None

    owner = JobFiles(files)
    error = asyncio.run(owner.collect_remote(
        files, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    final = owner.manifest
    assert error is cause
    assert final.outputs[0].availability_state == "failed"
    assert final.stdout.availability_state == "available"


def test_job_files_verifies_streams_after_collection_preflight_error(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    output = FileEntry(str(tmp_path / "result"), "/remote/result", "pending")
    stream = FileEntry(str(tmp_path / "stdout"), "/remote/stdout", "pending")
    initial = FileManifest(stdout=stream, outputs=(output,))
    cause = ValueError("destination collision")

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            raise cause

        async def verify_stream_availability(self, manifest, spec):
            assert manifest == initial
            return manifest.with_file_availability(
                stream.local_path, "available", remote_path=stream.remote_path,
            ), None

    owner = JobFiles(initial)
    error = asyncio.run(owner.collect_remote(
        initial, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    actual = owner.manifest
    assert error is cause
    assert actual.stdout.availability_state == "available"
    assert actual.outputs[0].availability_state == "pending"


def test_captured_snapshot_rejects_conflicting_update():
    import pytest
    from ase2sprkkr.execution.run_model import FileManifest

    captured = FileManifest(captured_at="2026-10-22T00:00:00+00:00")
    fresh = FileManifest()
    assert merge_snapshot(captured, captured) is captured
    with pytest.raises(ValueError, match="Cannot replace a captured"):
        merge_snapshot(captured, fresh)
    assert merge_snapshot(fresh, captured) == captured

def test_capture_failure_does_not_falsely_mark_unprocessed_files_failed(monkeypatch):
    import pytest
    from dataclasses import replace
    from types import SimpleNamespace
    from ase2sprkkr.execution import manager as manager_module
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest
    from ase2sprkkr.execution.base import RunState, FailurePhase

    files = FileManifest(outputs=(
        FileEntry("/tmp/untouched", "/remote/untouched", "pending"),
    ))
    class Store:
        def terminal_record(self, record):
            return None
    manager = manager_module.ExecutionManager(Store())
    record = SimpleNamespace(files=files)
    monkeypatch.setattr(
        manager_module, "capture_result_files",
        lambda manifest: (_ for _ in ()).throw(OSError("capture failed")),
    )
    seen = {}
    def transition(record, **kwargs):
        seen.update(kwargs)
        return kwargs
    monkeypatch.setattr(manager, "transition", transition)
    manager.complete(record, state=RunState.FINISHED, files_available=True)
    assert seen["state"] == RunState.FAILED
    assert seen["failure_phase"] == FailurePhase.FILES
    assert seen["files"].outputs[0].availability_state == "pending"


def test_terminal_merge_does_not_restore_vanished_available_file(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    missing = str(tmp_path / "vanished.dat")
    previous = FileManifest(outputs=(
        FileEntry(missing, "/remote/vanished.dat", availability_state="available"),
    ))
    received = FileManifest(outputs=(
        FileEntry(missing, "/remote/vanished.dat", availability_state="pending"),
    ))
    assert merge_terminal_files(previous, received) == received


def test_snapshot_merge_preserves_failed_transfer_over_stale_captured_success(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "result.dat")
    current = FileManifest(outputs=(
        FileEntry(path, "/remote/result.dat", availability_state="failed",
                  availability_message="download failed"),
    ))
    stale = FileManifest(outputs=(
        FileEntry(path, "/remote/result.dat", availability_state="available"),
    ), captured_at="2026-10-22T00:00:00+00:00")
    merged = merge_snapshot(current, stale)
    assert merged.captured_at == stale.captured_at
    assert merged.outputs[0].availability_state == "failed"
    assert merged.outputs[0].availability_message == "download failed"


def test_file_manifest_aggregates_distinct_cache_and_output_states(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    output = FileEntry(str(tmp_path / "run.out"), "/remote/run.out",
                       availability_state="available")
    cache = FileEntry(str(tmp_path / "restart.pot"), "/remote/restart.pot",
                      availability_state="failed", availability_message="download failed")
    files = FileManifest(stdout=output, caches=(cache,))
    assert len(files.files) == 2
    assert files.availability_state == "failed"
    assert files.failed_files == (cache,)


def test_merge_snapshot_preserves_remote_only_cache_baseline_status():
    """A remote-only cache is excluded from the availability summary."""
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    previous = FileManifest(caches=(
        FileEntry(remote_path="/remote/cache.dat", availability_state="failed",
                  availability_message="remote scan failed"),
    ))
    pending = FileManifest(caches=(
        FileEntry(remote_path="/remote/cache.dat", availability_state="pending"),
    ))
    assert previous.files == ()
    merged = merge_snapshot(previous, pending)
    assert merged.caches[0].availability_state == "failed"
    assert merged.caches[0].availability_message == "remote scan failed"

def test_stream_verification_exception_preserves_partial_download(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "output.dat")
    initial = FileManifest(outputs=(FileEntry(path, "/remote/output", "pending"),))

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            return manifest.with_file_availability(path, "available"), None

        async def verify_stream_availability(self, manifest, spec):
            raise OSError("stream verification unavailable")

    owner = JobFiles(initial)
    error = asyncio.run(owner.collect_remote(
        initial, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    saved = owner.manifest
    assert isinstance(error, OSError)
    assert str(error) == "stream verification unavailable"
    assert saved.outputs[0].availability_state == "available"


def test_snapshot_merge_preserves_concurrent_failure_over_captured_skip(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "result.dat")
    previous = FileManifest(outputs=(
        FileEntry(path, "/remote/result", availability_state="failed",
                  availability_message="transfer failed"),
    ))
    captured = FileManifest(
        outputs=(FileEntry(path, "/remote/result", availability_state="skipped"),),
        captured_at="2026-10-22T00:00:00+00:00",
    )
    merged = merge_snapshot(previous, captured)
    assert merged.outputs[0].availability_state == "failed"
    assert merged.outputs[0].availability_message == "transfer failed"

def test_atomic_completion_downgrades_stale_capture_after_download_failure(tmp_path):
    from dataclasses import replace
    from ase2sprkkr.execution.base import FailurePhase, RunState
    from ase2sprkkr.execution.manager import ExecutionManager
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "result.dat")
    files = FileManifest(outputs=(
        FileEntry(path, "/remote/result", availability_state="pending"),
    ))
    record = ExecutionManager().new_record(
        task="SCF", directory=tmp_path, files=files,
    )
    # An independent download observer persists failure while an older
    # completion observer still holds its speculative captured success.
    latest = replace(
        record, state=RunState.POSTPROCESSING,
        files=files.with_file_availability(
            path, "failed", "transfer failed", remote_path="/remote/result",
        ),
    )
    stale_capture = FileManifest(
        outputs=(FileEntry(path, "/remote/result", availability_state="skipped"),),
        captured_at="2026-10-22T00:00:00+00:00",
    )
    completed = latest.with_transition(
        state=RunState.FINISHED, files=stale_capture,
        validate_completed_files=True,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.DOWNLOAD
    assert completed.files.outputs[0].availability_state == "failed"
    assert completed.files.outputs[0].availability_message == "transfer failed"
    assert completed.files.captured_at is None


def test_discard_capture_keeps_file_delivery_facts(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest
    entry = FileEntry(str(tmp_path / "result"), "/remote/result",
                      availability_state="failed")
    original = FileManifest(outputs=(entry,),
                            captured_at="2026-10-22T00:00:00+00:00")
    updated = discard_capture(original)
    assert updated.captured_at is None
    assert updated.input_versions == updated.output_versions == ()
    assert updated.outputs == original.outputs

def test_failed_atomic_completion_discards_captured_versions(tmp_path):
    from dataclasses import replace
    from ase2sprkkr.common.file_versions import FileVersion
    from ase2sprkkr.execution.base import RunState, FailurePhase
    from ase2sprkkr.execution.manager import ExecutionManager
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    result_path = tmp_path / "result"
    result_path.write_bytes(b"stale captured contents")
    version = FileVersion.capture(result_path)
    path = str(result_path)
    entry = FileEntry(path, "/remote/result", availability_state="pending")
    base = ExecutionManager().new_record(
        task="SCF", directory=tmp_path,
        files=FileManifest(outputs=(entry,)),
    )
    latest = replace(
        base, state=RunState.POSTPROCESSING,
        files=base.files.with_file_availability(
            path, "failed", "transfer failed", remote_path="/remote/result",
        ),
    )
    captured = replace(
        base.files.with_file_availability(
            path, "available", remote_path="/remote/result",
        ),
        captured_at="2026-10-22T00:00:00+00:00",
        input_versions=(version,), output_versions=(version,),
    )
    completed = latest.with_transition(
        state=RunState.FINISHED, files=captured,
        validate_completed_files=True,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.DOWNLOAD
    assert captured.output_versions == (version,)
    assert completed.files.captured_at is None
    assert completed.files.input_versions == ()
    assert completed.files.output_versions == ()
    assert completed.files.outputs[0].availability_state == "failed"


def test_discard_capture_removes_real_version_records(tmp_path):
    from dataclasses import replace
    from ase2sprkkr.common.file_versions import FileVersion
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = tmp_path / "result.dat"
    path.write_bytes(b"completed")
    version = FileVersion.capture(path)
    original = FileManifest(
        outputs=(FileEntry(str(path), "/remote/result",
                           availability_state="failed",
                           availability_message="transfer failed"),),
        input_versions=(version,), output_versions=(version,),
        captured_at="2026-10-22T00:00:00+00:00",
    )
    cleared = discard_capture(original)
    assert cleared.captured_at is None
    assert cleared.input_versions == ()
    assert cleared.output_versions == ()
    assert cleared.outputs == original.outputs
    assert original.input_versions == (version,)
    assert original.output_versions == (version,)


def test_job_files_working_updates_do_not_modify_old_snapshots(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "output.dat")
    initial = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    files = JobFiles(initial)
    first = files.manifest
    files.mark_availability(path, "available")
    second = files.manifest

    assert first is initial
    assert second is not first
    assert first.outputs[0].availability_state == "pending"
    assert second.outputs[0].availability_state == "available"
    assert files.manifest is second


def test_job_files_reconciles_stale_observation_with_working_changes(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    first, second = (str(tmp_path / name) for name in ("one.dat", "two.dat"))
    original = FileManifest(outputs=(
        FileEntry(first, availability_state="pending"),
        FileEntry(second, availability_state="pending"),
    ))
    files = JobFiles(original)
    observed = original.with_file_availability(first, "available")

    files.mark_availability(second, "failed", "download failed")
    combined = files.merge(observed)

    assert combined.outputs[0].availability_state == "available"
    assert combined.outputs[1].availability_state == "failed"
    assert combined.outputs[1].availability_message == "download failed"
    assert original.outputs[1].availability_state == "pending"


def test_runner_owns_job_files_and_preserves_immutable_snapshots(tmp_path):
    from ase2sprkkr.execution.base import ExecutionSpec
    from ase2sprkkr.execution.local import LocalRunner
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None,
        output_file=None, error_file=None, stdin=None, cache_files=(),
    )
    runner = LocalRunner(spec)
    assert isinstance(runner.job_files, JobFiles)

    path = str(tmp_path / "output.dat")
    manifest = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    assert runner.job_files.replace(manifest) is manifest
    assert runner.job_files.manifest is manifest

    observed = manifest.with_file_availability(path, "available")
    assert runner.job_files.merge(observed).outputs[0].availability_state == "available"
    assert runner.job_files.manifest.outputs[0].availability_state == "available"
    assert manifest.outputs[0].availability_state == "pending"

def test_collect_updates_same_job_files_without_changing_old_snapshot(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "result")
    initial = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    owner = JobFiles(initial)

    class Handler:
        async def collect(self, mapping, manifest, outcome, *, required):
            assert manifest is initial
            return manifest.with_file_availability(path, "available"), None

        async def verify_stream_availability(self, manifest, spec):
            raise OSError("stream verification failed")

    failure = asyncio.run(owner.collect_remote(
        initial, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    assert isinstance(failure, OSError)
    assert owner.manifest.outputs[0].availability_state == "available"
    assert initial.outputs[0].availability_state == "pending"
