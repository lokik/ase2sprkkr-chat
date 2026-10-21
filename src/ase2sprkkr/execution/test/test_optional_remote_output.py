"""Focused tests for optional remote file availability."""

import asyncio

import pytest

from ..base import ExecutionSpec
from ..remote import RemoteRunner, FileSyncPolicy
from ..run_model import FileEntry, FileManifest
from ..transport import LocalTransport


class _Executor:
    output_files = FileSyncPolicy.CHANGED
    transport = LocalTransport()


def _runner(tmp_path):
    local = str(tmp_path / "local-optional.out")
    remote = str(tmp_path / "remote-optional.out")
    manifest = FileManifest(outputs=(
        FileEntry(local_path=local, remote_path=remote, availability_state="pending"),
    ))
    runner = RemoteRunner(
        _Executor(), ExecutionSpec(argv=(), input_file=None, output_file=None, directory=str(tmp_path)),
        files=manifest,
    )
    return runner, local


def test_optional_missing_remote_output_is_skipped(tmp_path):
    runner, local = _runner(tmp_path)
    manifest, error = asyncio.run(runner.file_handler.collect(runner.executor, runner.files, runner.result, required=False))
    assert error is None
    assert manifest.outputs[0].availability_state == "skipped"
    assert not (tmp_path / "local-optional.out").exists()


def test_required_missing_remote_output_is_failed(tmp_path):
    runner, local = _runner(tmp_path)
    manifest, error = asyncio.run(runner.file_handler.collect(runner.executor, runner.files, runner.result, required=True))
    assert isinstance(error, FileNotFoundError)
    assert manifest.outputs[0].availability_state == "failed"
