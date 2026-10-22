"""File delivery records expected failures without hiding programming errors."""

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import (
    ExecutionOutcome, FileEntry, FileManifest, FileMapping, RunFiles,
)
from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler
from ase2sprkkr.execution.transport import LocalTransport


def _scenario(tmp_path):
    local = tmp_path / "result.out"
    remote = tmp_path / "remote" / "result.out"
    files = FileManifest(outputs=(
        FileEntry(str(local), str(remote), availability_state="pending"),
    ))
    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"))
    return files, mapping


def test_transport_failure_is_recorded_per_file(tmp_path):
    files, mapping = _scenario(tmp_path)
    handler = RemoteFileHandler(LocalTransport())
    cause = subprocess.CalledProcessError(255, ["ssh", "remote-host"])

    async def fail_transfer(*args, **kwargs):
        raise cause

    handler._sync_file = fail_transfer
    observed, error = asyncio.run(handler.collect(
        mapping, files, ExecutionOutcome(), required=True,
    ))
    assert error is cause
    assert observed.outputs[0].availability_state == "failed"
    assert str(cause) in observed.outputs[0].availability_message
    assert files.outputs[0].availability_state == "pending"


@pytest.mark.parametrize("error", [TypeError("bad signature"), AttributeError("bad field")])
def test_programming_errors_escape_file_collection(tmp_path, error):
    files, mapping = _scenario(tmp_path)
    handler = RemoteFileHandler(LocalTransport())

    async def broken_transfer(*args, **kwargs):
        raise error

    handler._sync_file = broken_transfer
    with pytest.raises(type(error), match=str(error)):
        asyncio.run(handler.collect(
            mapping, files, ExecutionOutcome(), required=True,
        ))


def test_programming_error_escapes_stream_verification(tmp_path):
    stream = FileEntry(str(tmp_path / "stdout"), "/remote/stdout")
    files = FileManifest(stdout=stream)
    handler = RemoteFileHandler(LocalTransport())

    async def broken_verification(*args, **kwargs):
        raise TypeError("invalid stream reader")

    handler._verify_stream = broken_verification
    with pytest.raises(TypeError, match="invalid stream reader"):
        asyncio.run(handler.verify_stream_availability(files, SimpleNamespace()))


def test_programming_error_escapes_collection_coordinator(tmp_path):
    files, _ = _scenario(tmp_path)

    class BrokenHandler:
        async def collect(self, *args, **kwargs):
            raise TypeError("internal typo")

        async def verify_stream_availability(self, *args, **kwargs):
            raise AssertionError("Should never be reached")

    with pytest.raises(TypeError, match="internal typo"):
        asyncio.run(RunFiles(files).collect_remote(
            files, BrokenHandler(), ExecutionOutcome(), SimpleNamespace(),
            required=True,
        ))
