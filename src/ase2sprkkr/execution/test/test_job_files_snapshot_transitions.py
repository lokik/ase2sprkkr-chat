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


def test_snapshot_capture_failure_records_failed_files(tmp_path, monkeypatch):
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
    assert completed.files.outputs[0].availability_state == "failed"
    assert completed.files.outputs[0].availability_message == "capture failed"
