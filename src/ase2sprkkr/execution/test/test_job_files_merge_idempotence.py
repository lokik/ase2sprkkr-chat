"""Manifest reconciliation does not make copies without changing file facts."""

from ase2sprkkr.execution.job_files import JobFiles
from ase2sprkkr.execution.run_model import FileEntry, FileManifest


def test_snapshot_merge_reuses_received_manifest_if_delivery_unchanged(tmp_path):
    path = str(tmp_path / "output")
    available = FileManifest(outputs=(
        FileEntry(path, "/remote/output", availability_state="available"),
    ))
    assert available.merge(available) is available

    pending = FileManifest(outputs=(
        FileEntry(path, "/remote/output", availability_state="pending"),
    ))
    merged = available.merge(pending)
    assert merged is not pending
    assert merged.outputs[0].availability_state == "available"
    assert available.merge(merged) is merged


def test_terminal_merge_reuses_manifest_when_status_is_already_correct(tmp_path):
    path = tmp_path / "output"
    path.write_text("complete")
    received = FileManifest(outputs=(
        FileEntry(str(path), "/remote/output", availability_state="available"),
    ))
    assert JobFiles(received).merge_terminal(received) is received


def test_shared_stream_failure_has_priority_without_overwriting_membership(tmp_path):
    path = str(tmp_path / "combined.log")
    previous = FileManifest(
        stdout=FileEntry(path, "/remote/combined", availability_state="failed",
                         availability_message="transfer lost"),
        stderr=FileEntry(path, "/remote/combined", availability_state="pending"),
    )
    observed = FileManifest(
        stdout=FileEntry(path, "/remote/combined", availability_state="pending"),
        stderr=FileEntry(path, "/remote/combined", availability_state="pending"),
    )
    merged = previous.merge(observed)
    assert merged.stdout.availability_state == "failed"
    assert merged.stderr.availability_state == "failed"
    assert merged.stdout.availability_message == "transfer lost"
    assert merged.stderr.availability_message == "transfer lost"
    assert previous.stderr.availability_state == "pending"
