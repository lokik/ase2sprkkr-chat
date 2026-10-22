"""Protect immutable file snapshots from conflicting backend observations."""

from dataclasses import replace

import pytest

from ase2sprkkr.execution.manager import ExecutionManager, RunStore
from ase2sprkkr.execution.run_model import FileManifest


def _captured_run(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = manager.new_record(task="DOS", directory=tmp_path)
    record = replace(record, files=FileManifest(captured_at="2026-10-22T00:00:00+00:00"))
    return manager, manager.store.save(record)


def test_conflicting_captured_manifest_update_is_rejected(tmp_path):
    manager, record = _captured_run(tmp_path)
    with pytest.raises(ValueError, match="Cannot replace a captured file manifest"):
        manager.transition(record, files=FileManifest())
    assert manager.store.get(record.id).files == record.files


def test_identical_captured_manifest_update_is_idempotent(tmp_path):
    manager, record = _captured_run(tmp_path)
    updated = manager.transition(record, files=record.files)
    assert updated.files == record.files


def test_stale_observation_is_filtered_before_manifest_conflict(tmp_path):
    manager, record = _captured_run(tmp_path)
    updated = manager.transition(
        record, files=FileManifest(), expected_updated_at="outdated timestamp",
    )
    assert updated == record
    assert manager.store.get(record.id) == record


def test_stale_pending_observation_preserves_concurrent_file_availability(tmp_path):
    from ase2sprkkr.execution.job_files import JobFiles
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "result.out")
    pending = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    recorded = pending.with_file_availability(path, "failed", "download error")
    merged = JobFiles.merge_snapshot(recorded, pending)
    assert merged.outputs[0].availability_state == "failed"
    assert merged.outputs[0].availability_message == "download error"


def test_explicit_file_status_update_is_not_discarded(tmp_path):
    from ase2sprkkr.execution.job_files import JobFiles
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "result.out")
    pending = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    failed = pending.with_file_availability(path, "failed", "download error")
    merged = JobFiles.merge_snapshot(pending, failed)
    assert merged.outputs[0].availability_state == "failed"


def test_snapshot_capture_failure_preserves_unverified_files(tmp_path, monkeypatch):
    from ase2sprkkr.execution.base import FailurePhase, RunState
    from ase2sprkkr.execution.job_files import JobFiles
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "result.dat")
    manager = ExecutionManager(RunStore(tmp_path / "failure-runs"))
    record = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(outputs=(FileEntry(path, availability_state="pending"),)),
    )
    record = manager.store.save(record)

    def fail_capture(self, manifest=None):
        raise OSError("capture failed")

    monkeypatch.setattr(JobFiles, "capture", fail_capture)
    completed = manager.complete(
        record, state=RunState.FINISHED, files_available=True,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.FILES
    assert completed.files.outputs[0].availability_state == "pending"
    assert completed.files.outputs[0].availability_message is None


def test_concurrent_persisted_download_failure_overrides_stale_success(tmp_path):
    from ase2sprkkr.execution.base import FailurePhase, RunState
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "output.dat")
    original = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    manager = ExecutionManager(RunStore(tmp_path / "concurrent-runs"))
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, files=original,
    ))
    # Another client has already persisted a failed transfer, while the
    # completing client still holds the previous pending manifest.
    current = manager.update_file_availability(record, path, "failed", "lost download")
    completed = manager.complete(
        record, state=RunState.FINISHED, files_available=False,
        return_code=0,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.DOWNLOAD
    assert completed.message == "lost download"
    assert completed.files == current.files
    assert manager.store.get(record.id).files == current.files


def test_concurrent_delivery_does_not_force_failed_completion(tmp_path):
    from ase2sprkkr.execution.base import RunState
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "output.dat")
    manifest = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    manager = ExecutionManager(RunStore(tmp_path / "available-runs"))
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, files=manifest,
    ))
    current = manager.update_file_availability(record, path, "available")
    completed = manager.complete(
        record, state=RunState.FINISHED, files_available=False, return_code=0,
    )
    assert completed.state == RunState.FINISHED
    assert completed.files == current.files


def test_captured_completion_respects_newer_failed_file_state(tmp_path, monkeypatch):
    from ase2sprkkr.execution.base import FailurePhase, RunState
    from ase2sprkkr.execution.job_files import JobFiles
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "output.dat")
    manifest = FileManifest(outputs=(FileEntry(path, availability_state="pending"),))
    manager = ExecutionManager(RunStore(tmp_path / "snapshot-runs"))
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, files=manifest,
    ))
    manager.update_file_availability(record, path, "failed", "download failed")

    def capture(self, manifest=None):
        return replace(manifest, captured_at="2026-10-22T00:00:00+00:00")

    monkeypatch.setattr(JobFiles, "capture", capture)
    completed = manager.complete(
        record, state=RunState.FINISHED, files_available=True, return_code=0,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.DOWNLOAD
    assert completed.files.captured_at is None
    assert completed.files.outputs[0].availability_state == "failed"


def test_snapshot_discard_preserves_failed_transfer_for_retry(tmp_path):
    from ase2sprkkr.execution.job_files import JobFiles
    from ase2sprkkr.execution.run_model import FileEntry

    path = str(tmp_path / "result.dat")
    current = FileManifest(outputs=(FileEntry(path, availability_state="failed", availability_message="transfer failed"),))
    snapshot = FileManifest(
        outputs=(FileEntry(path, availability_state="available"),),
        captured_at="2026-10-22T00:00:00+00:00",
    )
    merged = JobFiles.merge_snapshot(current, snapshot)
    assert merged.failed_files
    assert merged.outputs[0].availability_message == "transfer failed"
    discarded = JobFiles.discard_capture(merged)
    assert discarded.captured_at is None
    assert discarded.outputs[0].availability_state == "failed"
