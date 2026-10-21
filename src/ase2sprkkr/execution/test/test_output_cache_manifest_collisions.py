"""Cross-kind download collision regressions for materialized file manifests.

A single terminal materialization decides membership; collection never globs.
"""
import asyncio

import pytest

from ..base import ExecutionOutcome
from ..file_availability import materialize_manifest
from ..remote_file_handler import RemoteFileHandler
from ..run_model import FileEntry, FileManifest, FileMapping
from ..transport import LocalTransport


class RecordingTransport(LocalTransport):
    def __init__(self):
        self.downloads = []

    async def get_file(self, source, destination):
        self.downloads.append((source, destination))
        await super().get_file(source, destination)


@pytest.mark.parametrize(
    "output_pattern,cache_pattern",
    [
        (False, True),   # exact output versus expanded cache
        (True, False),   # expanded output versus exact cache
        (True, True),    # both independently expanded
        (False, False),  # both exact
    ],
)
def test_output_cache_overlap_rejected_before_first_download(
    tmp_path, output_pattern, cache_pattern,
):
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    remote_file = remote / "result-1.dat"
    remote_file.write_text("from remote")
    destination = local / remote_file.name
    destination.write_text("preserve me")

    def entry(wildcard):
        name = "result-*.dat" if wildcard else remote_file.name
        return FileEntry(
            local_path=str(local / name), remote_path=str(remote / name),
            availability_state="pending",
        )

    transport = RecordingTransport()
    handler = RemoteFileHandler(transport)
    # Exact duplicates fail at declaration. Overlaps introduced by wildcard
    # materialization fail at the freeze boundary, before any download.
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        manifest = FileManifest(
            outputs=(entry(output_pattern),),
            caches=(entry(cache_pattern),),
            mapping=FileMapping(str(local), str(remote)),
        )
        frozen, error = materialize_manifest(manifest, remote=True)
        assert error is None
        asyncio.run(handler.collect(
            manifest.mapping, frozen, ExecutionOutcome(), required=True,
        ))
    assert transport.downloads == []
    assert destination.read_text() == "preserve me"


@pytest.mark.parametrize("output_pattern,cache_pattern", [(True, False), (False, True)])
def test_disjoint_output_cache_members_are_downloaded(
    tmp_path, output_pattern, cache_pattern,
):
    """Sharing a wildcard prefix alone is not a collision."""
    local, remote = tmp_path / "local", tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    (remote / "output.dat").write_text("out")
    (remote / "cache.dat").write_text("cache")

    def entry(name, wildcard):
        if wildcard:
            # A wildcard that matches only the declared file.
            return FileEntry(str(local / (name + "*.dat")),
                             str(remote / (name + "*.dat")),
                             availability_state="pending")
        return FileEntry(str(local / (name + ".dat")),
                         str(remote / (name + ".dat")),
                         availability_state="pending")

    manifest = FileManifest(
        outputs=(entry("output", output_pattern),),
        caches=(entry("cache", cache_pattern),),
        mapping=FileMapping(str(local), str(remote)),
    )
    frozen, error = materialize_manifest(manifest, remote=True)
    assert error is None
    handler = RemoteFileHandler(LocalTransport())
    collected, error = asyncio.run(
        handler.collect(manifest.mapping, frozen, ExecutionOutcome(), required=True),
    )
    assert error is None
    assert (local / "output.dat").read_text() == "out"
    assert (local / "cache.dat").read_text() == "cache"
    assert collected.outputs[0].availability_state == "available"
    assert collected.caches[0].availability_state == "available"
