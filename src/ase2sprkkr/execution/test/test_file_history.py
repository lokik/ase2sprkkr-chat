"""Focused regression tests for historical file file operations."""

import pytest

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
    assert snapshot.output_versions[0].path == first
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


def test_uncaptured_wildcard_requires_regular_file(tmp_path):
    """A directory or broken symlink cannot satisfy a file declaration."""
    declaration = str(tmp_path / "result-*")
    (tmp_path / "result-dir").mkdir()
    (tmp_path / "result-link").symlink_to(tmp_path / "absent")
    files = FileManifest(outputs=(FileEntry(local_path=declaration),))
    assert missing_result_files(files) == (declaration,)

    result = tmp_path / "result-file"
    result.write_text("done")
    assert missing_result_files(files) == ()
    result.unlink()
    assert missing_result_files(files) == (declaration,)


def test_capture_reuses_verified_output_without_hashing_again(tmp_path, monkeypatch):
    from ase2sprkkr.common.file_versions import FileVersion
    import ase2sprkkr.common.file_versions as versions

    output = tmp_path / "output.dat"
    output.write_bytes(b"delivered result")
    verified = FileVersion.capture(output, require_digest=True)
    files = FileManifest(outputs=(FileEntry(
        local_path=str(output), availability_state="available",
        verified_local_version=verified,
    ),))

    def unexpected_digest(*args, **kwargs):
        raise AssertionError("Unchanged delivery must not be hashed twice")

    monkeypatch.setattr(versions, "file_digest", unexpected_digest)
    captured = capture_result_files(files)
    assert captured.output_versions == (verified,)
    assert captured.output_versions[0].digest == verified.digest


def test_capture_rehashes_changed_verified_output(tmp_path):
    from ase2sprkkr.common.file_versions import FileVersion, file_digest

    output = tmp_path / "output.dat"
    output.write_bytes(b"delivered result")
    verified = FileVersion.capture(output, require_digest=True)
    files = FileManifest(outputs=(FileEntry(
        local_path=str(output), availability_state="available",
        verified_local_version=verified,
    ),))
    output.write_bytes(b"changed result")
    captured = capture_result_files(files)
    assert captured.output_versions[0].digest == file_digest(output)
    assert captured.output_versions[0].digest != verified.digest

@pytest.mark.parametrize("verified_stream", ("stdout", "stderr"))
def test_capture_merged_stream_keeps_verified_delivery(tmp_path, verified_stream):
    """A pending alias cannot erase a verified stream sharing its destination."""
    from ase2sprkkr.common.file_versions import FileVersion
    from ase2sprkkr.execution.run_model import FileMapping

    local = tmp_path / "merged.log"
    local.write_bytes(b"one shared output")
    version = FileVersion.capture(local, require_digest=True)
    verified = FileEntry(str(local), "/remote/merged.log", "available",
                         verified_local_version=version)
    pending = FileEntry(str(local), "/remote/merged.log", "pending")
    files = FileManifest(
        stdout=verified if verified_stream == "stdout" else pending,
        stderr=pending if verified_stream == "stdout" else verified,
        mapping=FileMapping(str(tmp_path), "/remote"),
    )
    captured = capture_result_files(files)
    assert captured.output_versions == (version,)
    assert captured.captured_at is not None


def test_capture_merged_stream_rejects_changed_verified_alias(tmp_path):
    """A damaged verified alias is never overridden by a successful peer."""
    from dataclasses import replace
    from ase2sprkkr.common.file_versions import FileVersion, FileVersionCaptureError
    from ase2sprkkr.execution.run_model import FileMapping

    local = tmp_path / "merged.log"
    local.write_bytes(b"one shared output")
    current = FileVersion.capture(local, require_digest=True)
    stale = replace(current, digest="0" * 64)
    files = FileManifest(
        stdout=FileEntry(str(local), "/remote/merged.log", "available",
                         verified_local_version=current),
        stderr=FileEntry(str(local), "/remote/merged.log", "available",
                         verified_local_version=stale),
        mapping=FileMapping(str(tmp_path), "/remote"),
    )
    with pytest.raises(FileVersionCaptureError, match="unverified or changed"):
        capture_result_files(files)
