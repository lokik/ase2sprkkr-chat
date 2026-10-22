"""No-op file observations must not invalidate durable run versions."""

from ase2sprkkr.execution import (
    ExecutionManager, FileEntry, FileManifest, RunStore,
)
from ase2sprkkr.execution.historical_restore import update_historical_availability


def test_availability_transformations_reuse_unchanged_manifest(tmp_path):
    path = str(tmp_path / "result.dat")
    pending = FileManifest(
        outputs=(FileEntry(local_path=path, availability_state="pending"),),
    )
    assert pending.initialize_availability() is pending
    assert pending.retry_failed_downloads() is pending

    available = pending.with_file_availability(path, "available")
    assert available is not pending
    assert available.with_file_availability(path, "available") is available
    assert available.retry_failed_downloads() is available

    failed = available.with_file_availability(path, "failed", "download failed")
    assert failed.with_file_availability(path, "failed", "download failed") is failed
    retried = failed.retry_failed_downloads()
    assert retried.outputs[0].availability_state == "pending"
    assert retried.outputs[0].availability_message is None
    assert retried.retry_failed_downloads() is retried


def test_unchanged_file_status_does_not_bump_run_record_version(tmp_path):
    path = str(tmp_path / "result.dat")
    files = FileManifest(
        outputs=(FileEntry(local_path=path, availability_state="pending"),),
    )
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = manager.store.save(
        manager.new_record(task="DOS", directory=tmp_path, files=files),
    )
    updated = update_historical_availability(record, manager.store, path, "available")
    repeated = update_historical_availability(record, manager.store, path, "available")
    assert repeated == updated
    assert repeated.updated_at == updated.updated_at
    assert manager.store.get(record.id) == updated

    failed = update_historical_availability(
        record,        manager.store, path, "failed", "transfer failed",
    )
    assert failed.updated_at != updated.updated_at
    assert failed.files.outputs[0].availability_state == "failed"


def test_remote_stream_recovery_skips_unchanged_paths(tmp_path):
    stdout = FileEntry(local_path=str(tmp_path / "out"), remote_path="/remote/out")
    stderr = FileEntry(local_path=str(tmp_path / "err"), remote_path="/remote/err")
    original = FileManifest(stdout=stdout, stderr=stderr)
    assert original.with_remote_stream_paths(
        stdout="/remote/out", stderr="/remote/err",
    ) is original
    assert original.with_remote_stream_paths(stdout="", stderr=None) is original
    updated = original.with_remote_stream_paths(stderr="/remote/new-err")
    assert updated.stdout is stdout
    assert updated.stderr.remote_path == "/remote/new-err"
    assert original.stderr.remote_path == "/remote/err"
