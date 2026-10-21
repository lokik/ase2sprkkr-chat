"""Restore requests must be safe before any transfer begins."""

import asyncio
from types import SimpleNamespace

import pytest

from ..remote_file_handler import RemoteFileHandler
from ..run_model import FileEntry, FileManifest
from ..transport import LocalTransport


class RecordingTransport(LocalTransport):
    def __init__(self):
        self.downloads = []

    async def get_file(self, source, destination):
        self.downloads.append((source, destination))
        return await super().get_file(source, destination)


def test_restore_rejects_bad_later_entry_before_first_transfer(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    outside = tmp_path / "outside"
    for directory in (local, remote, outside):
        directory.mkdir()
    (remote / "first.dat").write_bytes(b"first")
    good = FileEntry(str(local / "first.dat"), str(remote / "first.dat"))
    bad = FileEntry(str(local / "second.dat"), str(outside / "second.dat"))
    record = SimpleNamespace(files=FileManifest(
        outputs=(good, bad), local_root=str(local), remote_root=str(remote),
    ))
    transport = RecordingTransport()
    with pytest.raises(ValueError, match="outside mapped root"):
        asyncio.run(RemoteFileHandler(transport).restore(
            record, (good.local_path, bad.local_path),
        ))
    assert transport.downloads == []
    assert not (local / "first.dat").exists()


def test_restore_rejects_conflicting_staging_targets_before_transfer(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    first = FileEntry(str(local / "first.dat"), str(remote / "first.dat"))
    second = FileEntry(str(local / "second.dat"), str(remote / "second.dat"))
    record = SimpleNamespace(files=FileManifest(
        outputs=(first, second), local_root=str(local), remote_root=str(remote),
    ))
    staged = tmp_path / "staged.dat"
    transport = RecordingTransport()
    with pytest.raises(ValueError, match="Conflicting restore destinations"):
        asyncio.run(RemoteFileHandler(transport).restore(
            record, (first.local_path, second.local_path),
            destinations={first.local_path: str(staged),
                          second.local_path: str(staged)},
        ))
    assert transport.downloads == []
    assert not staged.exists()


def test_restore_rejects_local_path_outside_project_before_download(tmp_path):
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    outside = tmp_path / "outside"
    for directory in (local, remote, outside):
        directory.mkdir()
    (remote / "good.dat").write_bytes(b"good")
    good = FileEntry(str(local / "good.dat"), str(remote / "good.dat"))
    escaped = FileEntry(str(outside / "bad.dat"), str(remote / "bad.dat"))
    record = SimpleNamespace(files=FileManifest(
        outputs=(good, escaped), local_root=str(local), remote_root=str(remote),
    ))
    transport = RecordingTransport()
    with pytest.raises(ValueError, match="outside mapped root"):
        asyncio.run(RemoteFileHandler(transport).restore(
            record, (good.local_path, escaped.local_path),
        ))
    assert transport.downloads == []
    assert not (local / "good.dat").exists()
