"""Protect immutable file snapshots from conflicting backend observations."""

from dataclasses import replace

import pytest

from ase2sprkkr.execution.manager import ExecutionManager
from ase2sprkkr.execution.run_store import RunStore
from ase2sprkkr.execution.job_files import JobFiles
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

    def fail_capture(manifest):
        raise OSError("capture failed")

    import ase2sprkkr.execution.manager as manager_module
    monkeypatch.setattr(manager_module, "capture_result_files", fail_capture)
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
    current = JobFiles.update_availability(record, manager.store, path, "failed", "lost download")
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
    current = JobFiles.update_availability(record, manager.store, path, "available")
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
    JobFiles.update_availability(record, manager.store, path, "failed", "download failed")

    from ase2sprkkr.common.file_versions import FileVersion

    version = FileVersion(path=path, fingerprint=None)

    def capture(manifest):
        return replace(
            manifest, captured_at="2026-10-22T00:00:00+00:00",
            input_versions=(version,), output_versions=(version,),
        )

    import ase2sprkkr.execution.manager as manager_module
    monkeypatch.setattr(manager_module, "capture_result_files", capture)
    completed = manager.complete(
        record, state=RunState.FINISHED, files_available=True, return_code=0,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.DOWNLOAD
    assert completed.files.captured_at is None
    assert completed.files.input_versions == ()
    assert completed.files.output_versions == ()
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
    discarded = replace(merged, captured_at=None)
    assert discarded.captured_at is None
    assert discarded.outputs[0].availability_state == "failed"


def test_independent_availability_update_does_not_change_terminal_outcome(tmp_path):
    from ase2sprkkr.execution.base import RunState
    from ase2sprkkr.execution.run_model import FileEntry

    output = str(tmp_path / "output.dat")
    manager = ExecutionManager(RunStore(tmp_path / "availability-runs"))
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(outputs=(FileEntry(output, availability_state="pending"),)),
    ))
    finished = manager.complete(record, state=RunState.FINISHED, return_code=0)
    changed = JobFiles.update_availability(
        record, manager.store, output, "failed", "historical fetch error",
    )
    assert changed.state == RunState.FINISHED
    assert changed.finished_at == finished.finished_at
    assert changed.files.failed_files[0].availability_message == "historical fetch error"


def test_stale_job_files_updater_preserves_other_file_status(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry

    first = str(tmp_path / "first.dat")
    second = str(tmp_path / "second.dat")
    manager = ExecutionManager(RunStore(tmp_path / "parallel-availability"))
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, files=FileManifest(outputs=(
            FileEntry(first, availability_state="pending"),
            FileEntry(second, availability_state="pending"),
        )),
    ))
    first_update = JobFiles.update_availability(record, manager.store, first, "available")
    second_update = JobFiles.update_availability(
        record, manager.store, second, "failed", "missing",
    )
    assert first_update.files.outputs[0].availability_state == "available"
    assert second_update.files.outputs[0].availability_state == "available"
    assert second_update.files.outputs[1].availability_state == "failed"


def test_parallel_run_stores_merge_independent_file_updates(tmp_path):
    """Concurrent clients with stale records must not overwrite each other."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from ase2sprkkr.execution.run_model import FileEntry

    first = str(tmp_path / "first.dat")
    second = str(tmp_path / "second.dat")
    directory = tmp_path / "parallel-run-stores"
    store = RunStore(directory)
    manager = ExecutionManager(store)
    record = store.save(manager.new_record(
        task="DOS", directory=tmp_path, files=FileManifest(outputs=(
            FileEntry(first, availability_state="pending"),
            FileEntry(second, availability_state="pending"),
        )),
    ))
    barrier = Barrier(3)

    def update(path, state, message=None):
        # Independent stores model clients reopening the same persisted run.
        client_store = RunStore(directory)
        barrier.wait(timeout=10)
        return JobFiles.update_availability(
            record, client_store, path, state, message,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        delivered = pool.submit(update, first, "available")
        failed = pool.submit(update, second, "failed", "lost download")
        barrier.wait(timeout=10)
        delivered.result(timeout=15)
        failed.result(timeout=15)

    persisted = store.get(record.id)
    assert persisted.files.outputs[0].availability_state == "available"
    assert persisted.files.outputs[1].availability_state == "failed"
    assert persisted.files.outputs[1].availability_message == "lost download"
    assert persisted.state == record.state
    assert persisted.execution == record.execution
