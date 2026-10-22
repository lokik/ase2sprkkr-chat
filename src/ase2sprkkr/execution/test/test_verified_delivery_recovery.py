"""A delivered output must never be re-blessed after external modification."""

import asyncio
from dataclasses import asdict

import pytest

from ase2sprkkr.common.file_versions import FileVersion, FileVersionCaptureError
from ase2sprkkr.execution.base import ExecutionOutcome
from ase2sprkkr.execution.file_history import capture_result_files
from ase2sprkkr.execution.manifest_merge import merge_snapshot, merge_terminal_files
from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
from ase2sprkkr.execution.run_model import FileEntry, FileManifest, FileMapping
from ase2sprkkr.execution.transport import LocalTransport


def _case(tmp_path):
    root = tmp_path / "local"
    target = tmp_path / "remote"
    root.mkdir()
    target.mkdir()
    local = root / "result.dat"
    remote = target / "result.dat"
    local.write_bytes(b"original result")
    remote.write_bytes(b"original result")
    version = FileVersion.capture(local, require_digest=True)
    mapping = FileMapping(str(root), str(target), output_sync="always")
    available = FileManifest(
        outputs=(FileEntry(
            str(local), str(remote), availability_state="available",
            verified_local_version=version,
        ),),
        mapping=mapping,
    )
    pending = available.with_file_availability(str(local), "pending")
    return local, remote, available, pending, mapping


def test_reattach_preserves_only_verified_contents(tmp_path):
    local, _, available, pending, _ = _case(tmp_path)
    assert merge_terminal_files(available, pending).outputs[0].verified_delivery_is_current()
    assert merge_snapshot(available, pending).outputs[0].availability_state == "available"
    assert FileManifest(**asdict(available)).outputs[0].verified_delivery_is_current()

    local.write_bytes(b"altered contents")
    assert merge_terminal_files(available, pending).outputs[0].availability_state == "pending"
    assert merge_snapshot(available, pending).outputs[0].availability_state == "pending"


def test_reattach_does_not_trust_legacy_available_without_digest(tmp_path):
    local, _, available, pending, _ = _case(tmp_path)
    unverified = available.with_file_availability(str(local), "available")
    assert unverified.outputs[0].verified_local_version is None
    assert merge_terminal_files(unverified, pending).outputs[0].availability_state == "pending"
    assert merge_snapshot(unverified, pending).outputs[0].availability_state == "pending"


def test_corrupted_local_output_is_downloaded_again(tmp_path):
    local, _, available, pending, mapping = _case(tmp_path)
    local.write_bytes(b"corrupted copy")
    collected, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        mapping, available, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert local.read_bytes() == b"original result"
    assert collected.outputs[0].verified_delivery_is_current()


def test_missing_policy_requires_matching_remote_copy(tmp_path):
    local, remote, available, pending, mapping = _case(tmp_path)
    mapping = FileMapping(mapping.local_root, mapping.remote_root, output_sync="missing")
    handler = RemoteFileHandler(LocalTransport())

    matching, error = asyncio.run(handler.collect(
        mapping, pending, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert matching.outputs[0].verified_delivery_is_current()
    local.write_bytes(b"stale local data")
    altered, error = asyncio.run(handler.collect(
        mapping, pending, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, RuntimeError)
    assert altered.outputs[0].availability_state == "failed"
    assert local.read_bytes() == b"stale local data"
    assert remote.read_bytes() == b"original result"


def test_final_capture_rejects_replaced_remote_copy(tmp_path):
    local, _, available, _, _ = _case(tmp_path)
    assert capture_result_files(available).output_versions[0].digest == (
        available.outputs[0].verified_local_version.digest
    )
    local.write_bytes(b"modified")
    with pytest.raises(FileVersionCaptureError, match="changed"):
        capture_result_files(available)


def test_final_capture_excludes_skipped_remote_file(tmp_path):
    local, _, available, _, _ = _case(tmp_path)
    skipped = available.with_file_availability(str(local), "skipped")
    captured = capture_result_files(skipped)
    assert captured.output_versions == ()
    assert local.is_file()


def test_final_capture_rejects_unverified_available(tmp_path):
    local, _, available, _, _ = _case(tmp_path)
    unverified = available.with_file_availability(str(local), "available")
    with pytest.raises(FileVersionCaptureError, match="verified"):
        capture_result_files(unverified)
