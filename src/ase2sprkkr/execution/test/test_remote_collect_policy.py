"""File collection obeys one sync policy and preserves per-file outcomes."""

import asyncio

import pytest

from ase2sprkkr.execution import FileEntry, FileManifest, FileMapping
from ase2sprkkr.execution.base import ExecutionOutcome
from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
from ase2sprkkr.execution.transport import LocalTransport


class NoRemoteAccess(LocalTransport):
    async def file_signature(self, path):
        raise AssertionError(f"Unexpected remote signature lookup: {path}")

    async def file_digest(self, path):
        raise AssertionError(f"Unexpected remote digest lookup: {path}")

    async def get_file(self, source, destination):
        raise AssertionError(f"Unexpected download: {source}")


@pytest.mark.parametrize("policy, expected", (
    ("never", "skipped"),
    ("missing", "available"),
))
def test_collect_existing_output_without_remote_access(tmp_path, policy, expected):
    local = tmp_path / "existing.out"
    local.write_text("retained")
    files = FileManifest(outputs=(
        FileEntry(str(local), str(tmp_path / "missing-remote.out"),
                  availability_state="pending"),
    ))
    mapping = FileMapping(str(tmp_path), str(tmp_path), output_sync=policy)
    result, error = asyncio.run(RemoteFileHandler(NoRemoteAccess()).collect(
        mapping, files, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert local.read_text() == "retained"
    assert result.outputs[0].availability_state == expected
    assert files.outputs[0].availability_state == "pending"


def test_collect_missing_cache_still_checks_remote_baseline(tmp_path):
    local = tmp_path / "cache.pot"
    local.write_text("original")
    remote = tmp_path / "remote-cache.pot"
    files = FileManifest(caches=(
        FileEntry(str(local), str(remote), pre_run_signature=(1, 2),
                  availability_state="pending"),
    ))
    mapping = FileMapping(str(tmp_path), str(tmp_path), output_sync="missing")

    class SignatureOnly(LocalTransport):
        calls = 0

        async def file_signature(self, path):
            self.calls += 1
            return (9, 10)

        async def file_digest(self, path):
            raise AssertionError("MISSING must not hash cache files")

        async def get_file(self, source, destination):
            raise AssertionError("MISSING must retain an existing local cache")

    transport = SignatureOnly()
    result, error = asyncio.run(RemoteFileHandler(transport).collect(
        mapping, files, ExecutionOutcome(),
    ))
    assert error is None
    assert transport.calls == 1
    assert result.caches[0].availability_state == "skipped"
    assert local.read_text() == "original"


def test_collect_records_first_error_and_continues_other_outputs(tmp_path):
    root = tmp_path
    remote = root / "remote"
    remote.mkdir()
    (remote / "second.out").write_text("second")
    files = FileManifest(outputs=(
        FileEntry(str(root / "first.out"), str(remote / "missing.out"),
                  availability_state="pending"),
        FileEntry(str(root / "second.out"), str(remote / "second.out"),
                  availability_state="pending"),
    ))
    mapping = FileMapping(str(root), str(remote), output_sync="always")
    result, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        mapping, files, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, FileNotFoundError)
    assert [e.availability_state for e in result.outputs] == ["failed", "available"]
    assert (root / "second.out").read_text() == "second"


def test_successful_remote_download_stores_revalidatable_sha256(tmp_path):
    from ase2sprkkr.common.file_versions import file_digest

    remote = tmp_path / "remote"
    remote.mkdir()
    source = remote / "result.dat"
    source.write_bytes(b"remote result")
    local = tmp_path / "result.dat"
    entry = FileEntry(str(local), str(source), availability_state="pending")
    files = FileManifest(outputs=(entry,))
    observed, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        FileMapping(str(tmp_path), str(remote)),
        files, ExecutionOutcome(), required=True,
    ))
    assert error is None
    version = observed.outputs[0].verified_local_version
    assert observed.outputs[0].availability_state == "available"
    assert version.digest == file_digest(source)
    assert version.is_current()

    local.write_bytes(b"changed local copy")
    assert not version.is_current()


@pytest.mark.parametrize("initial_state", ("verified", "unverified"))
def test_collect_rechecks_available_output_when_local_identity_is_not_verified(
    tmp_path, initial_state,
):
    """Only a current SHA-256 receipt can skip another collection attempt."""
    from ase2sprkkr.common.file_versions import FileVersion

    remote = tmp_path / "remote"
    remote.mkdir()
    source = remote / "result.dat"
    source.write_bytes(b"authoritative output")
    local = tmp_path / "result.dat"
    local.write_bytes(b"old local contents")
    receipt = FileVersion.capture(local, require_digest=True)
    if initial_state == "verified":
        # A delivery was verified before the local copy was tampered with.
        local.write_bytes(b"tampered after delivery")
    else:
        # Imported legacy availability has no trustworthy receipt.
        receipt = None

    original = FileManifest(outputs=(
        FileEntry(
            str(local), str(source), availability_state="available",
            verified_local_version=receipt,
        ),
    ))
    observed, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        FileMapping(str(tmp_path), str(remote), output_sync="changed"),
        original, ExecutionOutcome(), required=True,
    ))

    assert error is None
    assert local.read_bytes() == source.read_bytes()
    assert observed.outputs[0].verified_local_version is not None
    assert observed.outputs[0].verified_local_version.is_current()
    assert original.outputs[0].verified_local_version == receipt


def test_collect_current_verified_output_skips_remote_even_if_missing(tmp_path):
    """An intact receipt survives reattachment without contacting the server."""
    from ase2sprkkr.common.file_versions import FileVersion

    path = tmp_path / "verified.dat"
    path.write_bytes(b"already downloaded")
    receipt = FileVersion.capture(path, require_digest=True)
    files = FileManifest(outputs=(
        FileEntry(
            str(path), str(tmp_path / "deleted-remote.dat"),
            availability_state="available", verified_local_version=receipt,
        ),
    ))
    observed, error = asyncio.run(RemoteFileHandler(NoRemoteAccess()).collect(
        FileMapping(str(tmp_path), str(tmp_path), output_sync="changed"),
        files, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert observed == files


def test_download_sha256_capture_runs_off_event_loop(tmp_path, monkeypatch):
    from threading import get_ident
    from ase2sprkkr.common.file_versions import FileVersion

    remote = tmp_path / "remote"
    remote.mkdir()
    source = remote / "result.dat"
    source.write_bytes(b"result")
    destination = tmp_path / "result.dat"
    observed_thread = []
    loop_thread = get_ident()
    capture = FileVersion.capture

    def worker_capture(*args, **kwargs):
        observed_thread.append(get_ident())
        return capture(*args, **kwargs)

    monkeypatch.setattr(FileVersion, "capture", worker_capture)
    files = FileManifest(outputs=(
        FileEntry(str(destination), str(source), availability_state="pending"),
    ))
    observed, error = asyncio.run(RemoteFileHandler(LocalTransport()).collect(
        FileMapping(str(tmp_path), str(remote), output_sync="always"),
        files, ExecutionOutcome(), required=True,
    ))
    assert error is None
    assert observed.outputs[0].verified_local_version is not None
    assert observed_thread and all(t != loop_thread for t in observed_thread)
