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
