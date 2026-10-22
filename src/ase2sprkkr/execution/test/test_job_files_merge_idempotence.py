"""Manifest reconciliation does not make copies without changing file facts."""

from ase2sprkkr.execution.job_files import JobFiles
from ase2sprkkr.execution.manifest_merge import merge_snapshot, merge_terminal_files
from ase2sprkkr.execution.run_model import FileEntry, FileManifest


def test_snapshot_merge_reuses_received_manifest_if_delivery_unchanged(tmp_path):
    from ase2sprkkr.common.file_versions import FileVersion
    path = tmp_path / "output"
    path.write_bytes(b"valid")
    available = FileManifest(outputs=(
        FileEntry(str(path), "/remote/output", availability_state="available",
                  verified_local_version=FileVersion.capture(path, require_digest=True)),
    ))
    assert merge_snapshot(available, available) is available

    pending = FileManifest(outputs=(
        FileEntry(str(path), "/remote/output", availability_state="pending"),
    ))
    merged = merge_snapshot(available, pending)
    assert merged is not pending
    assert merged.outputs[0].availability_state == "available"
    assert merge_snapshot(available, merged) is merged


def test_terminal_merge_reuses_manifest_when_status_is_already_correct(tmp_path):
    path = tmp_path / "output"
    path.write_text("complete")
    received = FileManifest(outputs=(
        FileEntry(str(path), "/remote/output", availability_state="available"),
    ))
    assert merge_terminal_files(received, received) is received


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
    merged = merge_snapshot(previous, observed)
    assert merged.stdout.availability_state == "failed"
    assert merged.stderr.availability_state == "failed"
    assert merged.stdout.availability_message == "transfer lost"
    assert merged.stderr.availability_message == "transfer lost"
    assert previous.stderr.availability_state == "pending"


def test_status_only_available_merge_preserves_verified_identity(tmp_path):
    from ase2sprkkr.common.file_versions import FileVersion
    path = tmp_path / "output.dat"
    path.write_bytes(b"verified")
    version = FileVersion.capture(path, require_digest=True)
    current = FileManifest(outputs=(FileEntry(
        str(path), "/remote/output", availability_state="available",
        verified_local_version=version,
    ),))
    status_only = FileManifest(outputs=(FileEntry(
        str(path), "/remote/output", availability_state="available",
    ),))
    merged = merge_snapshot(current, status_only)
    assert merged.outputs[0].verified_local_version == version
    assert merged.outputs[0].availability_state == "available"
    assert merge_terminal_files(current, status_only).outputs[0].verified_local_version == version
    path.write_bytes(b"changed")
    stale = merge_snapshot(current, status_only)
    assert stale.outputs[0].verified_local_version is None
