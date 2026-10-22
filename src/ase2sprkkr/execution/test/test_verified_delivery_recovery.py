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


def test_missing_policy_skips_unverified_existing_copy_without_remote_comparison(tmp_path):
    local, remote, available, pending, mapping = _case(tmp_path)
    mapping = FileMapping(mapping.local_root, mapping.remote_root, output_sync="missing")
    handler = RemoteFileHandler(LocalTransport())

    matching, error = asyncio.run(handler.collect(
        mapping, pending, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert matching.outputs[0].availability_state == "skipped"
    assert matching.outputs[0].verified_local_version is None
    local.write_bytes(b"stale local data")
    altered, error = asyncio.run(handler.collect(
        mapping, pending, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert altered.outputs[0].availability_state == "skipped"
    assert altered.outputs[0].availability_message is None
    assert altered.outputs[0].verified_local_version is None
    assert local.read_bytes() == b"stale local data"
    assert remote.read_bytes() == b"original result"
    assert merge_terminal_files(altered, pending).outputs[0].availability_state == "skipped"
    assert merge_snapshot(altered, pending).outputs[0].availability_state == "skipped"
    # Reattach does not re-download an intentionally skipped file.
    again, error = asyncio.run(handler.collect(
        mapping, merge_terminal_files(altered, pending),
        ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert again.outputs[0].availability_state == "skipped"


def test_missing_skip_can_download_after_local_conflict_is_removed(tmp_path):
    local, _, _, pending, mapping = _case(tmp_path)
    mapping = FileMapping(mapping.local_root, mapping.remote_root, output_sync="missing")
    local.write_bytes(b"conflicting local output")
    handler = RemoteFileHandler(LocalTransport())
    skipped, error = asyncio.run(handler.collect(
        mapping, pending, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert skipped.outputs[0].availability_state == "skipped"

    local.unlink()
    recovered, error = asyncio.run(handler.collect(
        mapping, merge_terminal_files(skipped, pending),
        ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert recovered.outputs[0].verified_delivery_is_current()
    assert local.read_bytes() == b"original result"


def test_missing_skip_allows_finished_run_without_capturing_stale_file(tmp_path):
    from ase2sprkkr.execution.base import RunState
    from ase2sprkkr.execution.run_manager import RunManager
    from ase2sprkkr.execution.run_store import RunStore
    from ase2sprkkr.execution.file_history import missing_result_files

    local, remote, _, pending, mapping = _case(tmp_path)
    local.write_bytes(b"stale result from an earlier run")
    valid_remote = remote.with_name("other.dat")
    valid_remote.write_bytes(b"new result")
    valid_local = local.with_name("other.dat")
    files = FileManifest(
        outputs=(
            pending.outputs[0],
            FileEntry(str(valid_local), str(valid_remote), availability_state="pending"),
        ),
        mapping=FileMapping(mapping.local_root, mapping.remote_root, output_sync="missing"),
    )
    observed, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        files.mapping, files, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert [entry.availability_state for entry in observed.outputs] == [
        "skipped", "available",
    ]
    assert observed.availability_state == "partial"
    assert [entry.availability_state for entry in observed.download_files] == [
        "skipped", "available",
    ]
    assert valid_local.read_bytes() == b"new result"
    assert local.read_bytes() == b"stale result from an earlier run"

    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    record = store.save(manager.new_record(
        task="DOS", directory=local.parent, files=files,
    ))
    finished = manager.complete(
        record, state=RunState.FINISHED, return_code=0,
        files_available=True, files=observed,
    )
    assert finished.state == RunState.FINISHED
    assert finished.files.outputs[0].availability_state == "skipped"
    assert finished.files.outputs[0].verified_local_version is None
    assert finished.files.captured_at is not None
    assert [str(version.path) for version in finished.files.output_versions] == [
        str(valid_local),
    ]
    assert missing_result_files(finished.files) == ()


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
