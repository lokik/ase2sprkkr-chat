"""Focused tests for optional remote file availability."""

import asyncio

from ..base import ExecutionOutcome
from ..remote_file_handler import RemoteFileHandler
from ..run_model import FileEntry, FileManifest, FileMapping
from ..transport import LocalTransport


def _case(tmp_path):
    local = tmp_path / "local-optional.out"
    remote = tmp_path / "remote-optional.out"
    manifest = FileManifest(outputs=(
        FileEntry(str(local), str(remote), availability_state="pending"),
    ))
    mapping = FileMapping(str(tmp_path), str(tmp_path))
    return RemoteFileHandler(LocalTransport()), mapping, manifest, local


def test_optional_missing_remote_output_is_skipped(tmp_path):
    handler, mapping, manifest, local = _case(tmp_path)
    result, error = asyncio.run(handler.collect(
        mapping, manifest, ExecutionOutcome(), required=False,
    ))
    assert error is None
    assert result.outputs[0].availability_state == "skipped"
    assert not local.exists()


def test_required_missing_remote_output_is_failed(tmp_path):
    handler, mapping, manifest, local = _case(tmp_path)
    result, error = asyncio.run(handler.collect(
        mapping, manifest, ExecutionOutcome(), required=True,
    ))
    assert isinstance(error, FileNotFoundError)
    assert result.outputs[0].availability_state == "failed"
    assert not local.exists()
