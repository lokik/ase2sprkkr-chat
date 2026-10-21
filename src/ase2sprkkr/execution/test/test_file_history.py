"""Focused regression tests for historical file file operations."""

from ase2sprkkr.execution.file_history import (
    capture_result_files, file_is_current, missing_result_files,
)
from ase2sprkkr.execution.run_model import FileEntry, FileManifest


def test_capture_freezes_wildcard_membership_and_detects_changed_bytes(tmp_path):
    first = tmp_path / "part-1.dat"
    first.write_text("original")
    declared = FileManifest(outputs=(FileEntry(local_path=str(tmp_path / "part-*.dat")),))

    snapshot = capture_result_files(declared)
    assert snapshot.captured_at is not None
    assert len(snapshot.output_versions) == 1
    assert snapshot.output_versions[0].path == str(first)
    assert capture_result_files(snapshot) is snapshot
    assert missing_result_files(snapshot) == ()
    assert file_is_current(snapshot, str(first), snapshot.output_versions)

    # Later files do not become members of a historical snapshot.
    (tmp_path / "part-2.dat").write_text("new")
    assert len(snapshot.output_versions) == 1

    first.write_text("changed")
    assert not file_is_current(snapshot, str(first), snapshot.output_versions)
    first.unlink()
    assert missing_result_files(snapshot) == (str(first),)


def test_missing_uncaptured_patterns_are_not_treated_as_empty_history(tmp_path):
    pattern = str(tmp_path / "results-*.dat")
    manifest = FileManifest(outputs=(FileEntry(local_path=pattern),))
    assert missing_result_files(manifest) == (pattern,)
    assert missing_result_files(None) == ()
    (tmp_path / "results-1.dat").write_text("data")
    assert missing_result_files(manifest) == ()
