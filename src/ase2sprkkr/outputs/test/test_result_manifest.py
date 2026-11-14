"""Build manifests from parsed SPR-KKR file roles without a live execution."""
from pathlib import Path

from kkrunner.files import FileManifest, FileState

from ase2sprkkr.outputs import TaskResult, capture_result_file_manifest


def _result(tmp_path):
    output = tmp_path / "run.out"
    output.write_text("OUT", encoding="utf-8")
    result = TaskResult(None, None, tmp_path, output_file="run.out", task_name="SCF")
    result.files.add_file("input", "run.inp")
    result.files.add_file("potential", "start.pot")
    result.files.add_file("converged", "start.pot_new")
    result.files.add_file("DOS", "run_DOS.dos", "dos")
    return result


def test_task_result_manifest_uses_native_files_section(tmp_path):
    result = _result(tmp_path)
    for name in ("run.inp", "start.pot", "start.pot_new", "run_DOS.dos"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    manifest = result.to_file_manifest()

    assert isinstance(manifest, FileManifest)
    assert manifest.stdout.local_path == str(tmp_path / "run.out")
    assert {Path(entry.local_path).name for entry in manifest.inputs} == {
        "run.inp", "start.pot",
    }
    assert {Path(entry.local_path).name for entry in manifest.outputs} == {
        "start.pot_new", "run_DOS.dos",
    }
    assert {version.path.name for version in manifest.output_versions} == {
        "run.out", "start.pot_new", "run_DOS.dos",
    }
    assert manifest.state is FileState.CURRENT


def test_manifest_keeps_missing_declarations_without_discovering_neighbor_files(tmp_path):
    result = _result(tmp_path)
    (tmp_path / "other.out").write_text("unrelated", encoding="utf-8")
    manifest = capture_result_file_manifest(result)
    paths = {Path(entry.local_path).name for entry in (*manifest.inputs, *manifest.outputs)}
    assert paths == {"run.inp", "start.pot", "start.pot_new", "run_DOS.dos"}
    assert tuple(version.path.name for version in manifest.output_versions) == ("run.out",)
    assert manifest.state is FileState.CURRENT


def test_sfn_role_tracks_generation(tmp_path):
    result = _result(tmp_path)
    result.files.add_file("SFN", "structure.sfn", "sfn")
    result.sfn_generated = False
    previous = result.to_file_manifest()
    assert "structure.sfn" in {Path(entry.local_path).name for entry in previous.inputs}
    assert "structure.sfn" not in {Path(entry.local_path).name for entry in previous.outputs}

    result.sfn_generated = True
    generated = result.to_file_manifest()
    assert "structure.sfn" in {Path(entry.local_path).name for entry in generated.outputs}


def test_provided_versions_are_reused(tmp_path):
    from kkrunner.files import FileVersion

    result = _result(tmp_path)
    out = tmp_path / "run.out"
    version = FileVersion.capture(out)
    manifest = result.to_file_manifest(output_version=version)
    assert manifest.output_versions[0] is version

    out.write_text("overwritten", encoding="utf-8")
    assert manifest.state is FileState.OVERWRITTEN


def test_effective_task_declarations_supply_unlisted_outputs(tmp_path):
    from ase2sprkkr import InputParameters

    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "Fe"
    parameters.CONTROL.POTFIL = "Fe.pot"
    (tmp_path / "run.out").write_text("output", encoding="utf-8")
    result = TaskResult(parameters, None, str(tmp_path), output_file="run.out", task_name="DOS")

    # The result parser observed no DOS file. The manifest must nevertheless
    # preserve the canonical file declaration from the effective task.
    manifest = result.to_file_manifest()
    assert {Path(entry.local_path).name for entry in manifest.outputs} == {"Fe_DOS.dos"}
    assert "DOS" not in result.files
    assert tuple(version.path.name for version in manifest.output_versions) == ("run.out",)

    (tmp_path / "Fe_DOS.dos").write_text("actual data", encoding="utf-8")
    materialized = result.to_file_manifest()
    assert {version.path.name for version in materialized.output_versions} == {
        "run.out", "Fe_DOS.dos",
    }
    assert "DOS" not in result.files


def test_task_cache_and_optional_output_pattern_are_resolved(tmp_path):
    from ase2sprkkr import InputParameters

    parameters = InputParameters.create("SCF")
    parameters.CONTROL.DATASET = "Fe"
    parameters.CONTROL.POTFIL = "Fe.pot"
    parameters.SCF.FULLPOT = True
    parameters.CONTROL.SFNFIL = "Fe.sfn"
    (tmp_path / "run.out").write_text("output", encoding="utf-8")
    (tmp_path / "Fe_SCFSTART.dos").write_text("startup DOS", encoding="utf-8")
    (tmp_path / "unrelated.dat").write_text("other", encoding="utf-8")
    result = TaskResult(parameters, None, str(tmp_path), output_file="run.out", task_name="SCF")

    manifest = result.to_file_manifest()
    assert {Path(cache.file.local_path).name for cache in manifest.caches} == {"Fe.sfn"}
    assert {Path(entry.local_path).name for entry in manifest.outputs} == {
        "Fe.pot_new", "Fe_SCFSTART.dos",
    }
    optional = next(entry for entry in manifest.outputs if entry.local_path.endswith("Fe_SCFSTART.dos"))
    assert optional.optional
    assert optional.source_pattern == str(tmp_path / "*_SCFSTART.dos")
    assert "unrelated.dat" not in {
        Path(entry.local_path).name for entry in (*manifest.inputs, *manifest.outputs)
    }
    # A missing pattern does not become an unsafe literal pathname.
    (tmp_path / "Fe_SCFSTART.dos").unlink()
    absent = result.to_file_manifest()
    assert all("*" not in entry.local_path for entry in absent.outputs)
    assert {Path(entry.local_path).name for entry in absent.outputs} == {"Fe.pot_new"}


def test_explicit_sfn_origin_overrides_task_cache_classification(tmp_path):
    from ase2sprkkr import InputParameters

    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "Fe.pot"
    parameters.SCF.FULLPOT = True
    parameters.CONTROL.SFNFIL = "Fe.sfn"
    (tmp_path / "run.out").write_text("output", encoding="utf-8")
    result = TaskResult(parameters, None, str(tmp_path), output_file="run.out", task_name="SCF")
    result.files.add_file("SFN", "Fe.sfn", "sfn")

    result.sfn_generated = True
    generated = result.to_file_manifest()
    assert {Path(entry.local_path).name for entry in generated.outputs} >= {"Fe.sfn"}
    assert all(cache.file.local_path != str(tmp_path / "Fe.sfn") for cache in generated.caches)

    result.sfn_generated = False
    reused = result.to_file_manifest()
    assert "Fe.sfn" in {Path(entry.local_path).name for entry in reused.inputs}
    assert all(cache.file.local_path != str(tmp_path / "Fe.sfn") for cache in reused.caches)

def test_internal_tau_cache_stays_in_manifest_not_result_actions(tmp_path):
    from ase2sprkkr import InputParameters

    parameters = InputParameters.create("BSFEK")
    parameters.CONTROL.DATASET = "Fe"
    (tmp_path / "run.out").write_text("output", encoding="utf-8")
    tau = tmp_path / "Fe_BLOCHSF.tau.dir"
    tau.write_bytes(b"cached tau")
    result = TaskResult(parameters, None, str(tmp_path), output_file="run.out", task_name="BSFEK")
    result.files.add_file("TAUDIR", tau.name)

    # The old UI displayed and offered actions for every FILES entry.
    assert "TAUDIR" not in result.output_values
    manifest = result.to_file_manifest()
    assert tau.as_posix() in {item.file.local_path for item in manifest.caches}
    assert tau.as_posix() not in {item.local_path for item in manifest.outputs}


def test_missing_mandatory_output_reported_but_remains_in_manifest(tmp_path):
    from ase2sprkkr import InputParameters

    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "Fe"
    parameters.CONTROL.POTFIL = "Fe.pot"
    (tmp_path / "run.out").write_text("output", encoding="utf-8")
    result = TaskResult(parameters, None, str(tmp_path), output_file="run.out", task_name="DOS")
    manifest = result.to_file_manifest()
    required = next(file for file in manifest.outputs if file.local_path.endswith("Fe_DOS.dos"))
    assert not required.optional
    assert required.availability_state == "failed"
    assert "missing" in required.availability_message
    assert "DOS" not in result.output_values


def test_missing_parsed_result_file_has_no_actions(tmp_path):
    result = TaskResult(None, None, str(tmp_path), output_file="run.out", task_name="DOS")
    file = result.files.add_file("DOS", "missing.dos", "dos")
    assert file.actions() == ()
    (tmp_path / "missing.dos").write_text("some data", encoding="utf-8")
    assert "open" in file.actions()
