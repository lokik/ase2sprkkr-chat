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
    checked, error = asyncio.run(handler.verify_stream_availability(manifest, spec))
    assert isinstance(error, RuntimeError)
    assert f"{channel} contents differ" in str(error)
    assert getattr(checked, channel).availability_state == "failed"
    assert (local_out if channel == "stdout" else local_err).exists()


@pytest.mark.parametrize("channel", ("stdout", "stderr"))
def test_missing_authoritative_stream_is_not_success(tmp_path, channel):
    handler, manifest, spec, _, _, remote_out, remote_err = _case(tmp_path)
    (remote_out if channel == "stdout" else remote_err).unlink()
    checked, error = asyncio.run(handler.verify_stream_availability(manifest, spec))
    assert isinstance(error, FileNotFoundError)
    assert f"Remote {channel} output is missing" in str(error)
    assert getattr(checked, channel).availability_state == "failed"


def test_successful_integrity_check_preserves_timestamps(tmp_path):
    handler, manifest, spec, local_out, local_err, remote_out, remote_err = _case(tmp_path)
    checked, error = asyncio.run(handler.verify_stream_availability(manifest, spec))
    assert error is None
    assert checked.stdout.availability_state == "available"
    assert checked.stderr.availability_state == "available"
    assert local_out.stat().st_mtime_ns == remote_out.stat().st_mtime_ns
    assert local_err.stat().st_mtime_ns == remote_err.stat().st_mtime_ns

def test_remote_stream_modified_during_integrity_check_is_rejected(tmp_path, monkeypatch):
    """Matching bytes are not enough if the authoritative file changes mid-check."""
    handler, manifest, spec, _, _, remote_out, _ = _case(tmp_path)
    from dataclasses import replace

    manifest = replace(manifest, stderr=None)
    original_signature = handler.transport.file_signature
    calls = 0

    async def changing_signature(path):
        nonlocal calls
        signature = await original_signature(path)
        if path == str(remote_out):
            calls += 1
            if calls == 2:
                return signature[0], signature[1] + 1
        return signature

    monkeypatch.setattr(handler.transport, "file_signature", changing_signature)
    checked, error = asyncio.run(handler.verify_stream_availability(manifest, spec))
    assert isinstance(error, RuntimeError)
    assert "changed during verification" in str(error)
    assert checked.stdout.availability_state == "failed"
    assert calls == 2


def test_stream_availability_preserves_successful_peer_after_corruption(tmp_path):
    """A damaged stdout must not conceal an independently valid stderr."""
    handler, manifest, spec, _, _, _, _ = _case(tmp_path, stdout=b"bad!")
    checked, error = asyncio.run(handler.verify_stream_availability(manifest, spec))

    assert isinstance(error, RuntimeError)
    assert "stdout contents differ" in str(error)
    assert checked.stdout.availability_state == "failed"
    assert "stdout contents differ" in checked.stdout.availability_message
    assert checked.stderr.availability_state == "available"
    assert checked.stderr.availability_message is None
