"""Verify downloaded stdout/stderr directly in their owning file handler."""
import asyncio

import pytest

from ..base import ExecutionSpec
from ..remote_file_handler import RemoteFileHandler
from ..run_model import FileEntry, FileManifest
from ..transport import LocalTransport


def _case(tmp_path, *, stdout=b"same", stderr=b"error"):
    local_out = tmp_path / "local.out"
    local_err = tmp_path / "local.err"
    remote_out = tmp_path / "remote.out"
    remote_err = tmp_path / "remote.err"
    local_out.write_bytes(stdout)
    local_err.write_bytes(stderr)
    remote_out.write_bytes(b"same")
    remote_err.write_bytes(b"error")
    manifest = FileManifest(
        stdout=FileEntry(str(local_out), str(remote_out)),
        stderr=FileEntry(str(local_err), str(remote_err)),
    )
    spec = ExecutionSpec(
        argv=(), input_file=None, output_file=str(local_out),
        error_file=str(local_err), directory=str(tmp_path),
    )
    return RemoteFileHandler(LocalTransport()), manifest, spec, local_out, local_err, remote_out, remote_err


@pytest.mark.parametrize("channel", ("stdout", "stderr"))
def test_equal_size_corruption_fails_even_if_length_matches(tmp_path, channel):
    handler, manifest, spec, local_out, local_err, _, _ = _case(
        tmp_path,
        stdout=b"bad!" if channel == "stdout" else b"same",
        stderr=b"wrong" if channel == "stderr" else b"error",
    )
    with pytest.raises(RuntimeError, match=f"{channel} contents differ"):
        asyncio.run(handler.verify_streams(manifest, spec))
    assert (local_out if channel == "stdout" else local_err).exists()


@pytest.mark.parametrize("channel", ("stdout", "stderr"))
def test_missing_authoritative_stream_is_not_success(tmp_path, channel):
    handler, manifest, spec, _, _, remote_out, remote_err = _case(tmp_path)
    (remote_out if channel == "stdout" else remote_err).unlink()
    with pytest.raises(FileNotFoundError, match=f"Remote {channel} output is missing"):
        asyncio.run(handler.verify_streams(manifest, spec))


def test_successful_integrity_check_preserves_timestamps(tmp_path):
    handler, manifest, spec, local_out, local_err, remote_out, remote_err = _case(tmp_path)
    asyncio.run(handler.verify_streams(manifest, spec))
    assert local_out.stat().st_mtime_ns == remote_out.stat().st_mtime_ns
    assert local_err.stat().st_mtime_ns == remote_err.stat().st_mtime_ns
