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


def test_job_files_does_not_verify_after_failed_transfer():
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
            raise AssertionError("must not verify after download failure")

    updated, error = asyncio.run(JobFiles.collect_remote(
        files, Handler(), SimpleNamespace(), SimpleNamespace(), required=True,
    ))
    assert error is cause
    assert updated.outputs[0].availability_state == "failed"


def test_merge_snapshot_never_overwrites_captured_history():
    from dataclasses import replace
    from ase2sprkkr.execution.run_model import FileManifest

    captured = FileManifest(captured_at="2026-10-22T00:00:00+00:00")
    fresh = FileManifest()
    assert JobFiles.merge_snapshot(captured, fresh) is captured
    assert JobFiles.merge_snapshot(fresh, captured) is captured
    assert JobFiles.merge_snapshot(None, fresh) is fresh
