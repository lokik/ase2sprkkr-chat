from types import SimpleNamespace

from kkrunner.files.versions import FileState
from ase2sprkkr.outputs import TaskResult, capture_result_file_versions


def test_capture_result_file_versions_uses_content_identity(tmp_path):
    path = tmp_path / "result.dat"
    path.write_bytes(b"result")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    result.files.add_file("DATA", path)

    versions = capture_result_file_versions(result)

    version = versions["DATA"]
    assert version.path == path.absolute()
    assert version.digest is not None
    assert version.available
    assert version.state() is FileState.CURRENT


def test_capture_result_file_versions_ignores_non_path_outputs(tmp_path):
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    assert result.file_manifest is None

    assert dict(capture_result_file_versions(result)) == {}


def test_historical_versions_survive_overwrite_without_loader_annotation(tmp_path):
    from kkrunner.files import FileEntry, FileManifest, FileVersion

    output, curve, missing, unrelated = (
        tmp_path / name for name in ("run.out", "curve.agr", "missing.dat", "new.dat")
    )
    for path in (output, curve, missing, unrelated):
        path.write_text("original")
    recorded = tuple(
        FileVersion.capture(path, require_digest=True)
        for path in (output, curve, missing)
    )
    result = TaskResult(None, None, str(tmp_path), output_file=str(output), task_name="DOS")
    result.files.add_file("curve", str(curve), "agr")
    result.files.add_file("missing", str(missing))
    result.files.add_file("unrelated", str(unrelated))
    result.run_record = SimpleNamespace(files=FileManifest(
        stdout=FileEntry(str(output)),
        outputs=(FileEntry(str(curve)), FileEntry(str(missing))),
        output_versions=recorded,
        captured_at="2026-11-26T00:00:00+00:00",
    ))

    curve.write_text("replacement")
    missing.unlink()

    versions = capture_result_file_versions(result)
    assert versions["output"] == recorded[0]
    assert versions["curve"] == recorded[1]
    assert versions["missing"] == recorded[2]
    assert "unrelated" not in versions
    assert versions["curve"].state() is FileState.OVERWRITTEN
    assert versions["missing"].state() is FileState.MISSING
    assert result.files["curve"].actions() == ("open_directory",)
