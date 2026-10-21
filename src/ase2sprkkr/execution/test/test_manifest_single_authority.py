"""The persisted file manifest is the sole remote-path authority."""

from dataclasses import asdict

import pytest

from ase2sprkkr.execution.run_model import FileEntry, FileManifest


def test_local_manifest_does_not_store_remote_mapping():
    manifest = FileManifest(outputs=(FileEntry("/local/output.dat"),))
    assert manifest.remote_root is None
    assert manifest.outputs[0].remote_path is None
    assert "mapping" not in asdict(manifest)
    with pytest.raises(ValueError, match="no remote workspace"):
        manifest.map_path("/local/output.dat")


def test_remote_manifest_resolves_paths_once_and_roundtrips(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    original = FileManifest(
        inputs=(FileEntry(str(local / "input.dat")),),
        outputs=(FileEntry(str(local / "output.dat")),),
    )
    resolved = original.with_remote(
        local_root=str(local), remote_root="/scratch/job",
        input_sync="always", output_sync="missing",
    )
    assert original.inputs[0].remote_path is None
    assert resolved.inputs[0].remote_path == "/scratch/job/input.dat"
    assert resolved.outputs[0].remote_path == "/scratch/job/output.dat"
    assert resolved.input_sync == "always"
    assert resolved.output_sync == "missing"
    restored = FileManifest(**asdict(resolved))
    assert restored == resolved
    assert "mapping" not in asdict(restored)


def test_remote_reservation_does_not_silently_remap_existing_entry(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    file = FileEntry(str(local / "result.dat"), "/other/target.dat")
    manifest = FileManifest(outputs=(file,))
    with pytest.raises(ValueError, match="differs from manifest"):
        manifest.with_remote(local_root=str(local), remote_root="/scratch")
