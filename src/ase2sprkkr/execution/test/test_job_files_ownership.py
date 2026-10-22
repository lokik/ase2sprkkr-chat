"""Stable ownership boundary for run file operations."""

from ase2sprkkr.execution import JobFiles
from ase2sprkkr.execution.job_files import JobFiles as ModuleJobFiles
from ase2sprkkr.execution.manager import RunRecord


def test_job_files_public_identity():
    assert JobFiles is ModuleJobFiles
    assert JobFiles.__module__ == "ase2sprkkr.execution.job_files"
    assert isinstance(RunRecord.job_files, property)


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
    merged = JobFiles.merge_terminal_files(previous, frozen)
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
    assert JobFiles.merge_terminal_files(previous, frozen) == frozen


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
    out, error = asyncio.run(JobFiles.collect_remote(
        pending, handler, SimpleNamespace(), SimpleNamespace(), required=True,
    ))
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

    updated, error = asyncio.run(JobFiles.collect_remote(
        files, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
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

    final, error = asyncio.run(JobFiles.collect_remote(
        files, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
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

    actual, error = asyncio.run(JobFiles.collect_remote(
        initial, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    assert error is cause
    assert actual.stdout.availability_state == "available"
    assert actual.outputs[0].availability_state == "pending"


def test_captured_snapshot_rejects_conflicting_update():
    import pytest
    from ase2sprkkr.execution.run_model import FileManifest

    captured = FileManifest(captured_at="2026-10-22T00:00:00+00:00")
    fresh = FileManifest()
    assert JobFiles.merge_snapshot(captured, captured) is captured
    with pytest.raises(ValueError, match="Cannot replace a captured"):
        JobFiles.merge_snapshot(captured, fresh)
    assert JobFiles.merge_snapshot(fresh, captured) == captured

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
    record = SimpleNamespace(files=files, job_files=SimpleNamespace(
        capture=lambda manifest: (_ for _ in ()).throw(OSError("capture failed")),
    ))
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
    assert JobFiles.merge_terminal_files(previous, received) == received


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
    merged = JobFiles.merge_snapshot(current, stale)
    assert merged.captured_at == stale.captured_at
    assert merged.outputs[0].availability_state == "failed"
    assert merged.outputs[0].availability_message == "download failed"


def test_snapshot_merge_preserves_shared_stdout_and_output_identity(tmp_path):
    """Aliases of a single output file keep one state through reconciliation."""
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    path = str(tmp_path / "run.out")
    previous = FileManifest(
        stdout=FileEntry(path, "/remote/run.out", availability_state="failed",
                         availability_message="transfer interrupted"),
        outputs=(FileEntry(path, "/remote/run.out", availability_state="failed",
                           availability_message="transfer interrupted"),),
    )
    pending = FileEntry(path, "/remote/run.out", availability_state="pending")
    received = FileManifest(stdout=pending, outputs=(pending,))
    updated = JobFiles.merge_snapshot(previous, received)

    assert len(updated.files) == 1
    assert updated.stdout.availability_state == "failed"
    assert updated.outputs[0].availability_state == "failed"
    assert updated.stdout.availability_message == "transfer interrupted"


def test_file_manifest_aggregates_distinct_cache_and_output_states(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    output = FileEntry(str(tmp_path / "run.out"), "/remote/run.out",
                       availability_state="available")
    cache = FileEntry(str(tmp_path / "restart.pot"), "/remote/restart.pot",
                      availability_state="failed", availability_message="download failed")
    files = FileManifest(stdout=output, outputs=(output,), caches=(cache,))
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
    merged = JobFiles.merge_snapshot(previous, pending)
    assert merged.caches[0].availability_state == "failed"
    assert merged.caches[0].availability_message == "remote scan failed"

def test_capture_failure_does_not_change_individual_file_availability(monkeypatch):
    from types import SimpleNamespace
    from ase2sprkkr.execution import manager as manager_module
    from ase2sprkkr.execution.base import FailurePhase, RunState
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    manifest = FileManifest(outputs=(
        FileEntry("/tmp/untouched", "/remote/untouched", "pending"),
    ))

    class Store:
        def terminal_record(self, record):
            return None

    manager = manager_module.ExecutionManager(Store())
    record = SimpleNamespace(files=manifest, job_files=SimpleNamespace(
        capture=lambda files: (_ for _ in ()).throw(OSError("capture failed")),
    ))
    result = {}

    def transition(record, **kwargs):
        result.update(kwargs)
        return kwargs

    monkeypatch.setattr(manager, "transition", transition)
    manager.complete(record, state=RunState.FINISHED, files_available=True)
    assert result["state"] == RunState.FAILED
    assert result["failure_phase"] == FailurePhase.FILES
    assert result["files"].outputs[0].availability_state == "pending"


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

    saved, error = asyncio.run(JobFiles.collect_remote(
        initial, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    assert isinstance(error, OSError)
    assert str(error) == "stream verification unavailable"
    assert saved.outputs[0].availability_state == "available"
