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
