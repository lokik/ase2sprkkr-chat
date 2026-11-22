"""ResultLoader discovery, caching and per-file provenance."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kkrunner.files import FileVersion
from ase2sprkkr.outputs import FileAssociation, ResultLoader
from ase2sprkkr.outputs.task_result import TaskResult
from kkrunner.runs import HistoricalFileMismatchError


class _SCF(TaskResult):
    produces_potential = True

    @property
    def potential_filename(self):
        return str(Path(self.directory) / "Fe.pot_new")


class _DOS(TaskResult):
    @property
    def potential_filename(self):
        return str(Path(self.directory) / "Fe.pot_new")


def _output(path, *, scf=True):
    if scf:
        path.write_text(
            " * KKRSCF VERSION 9.7 *\nTASK = SCF\npot-file: Fe.pot_new\n"
        )
    else:
        path.write_text(" * KKRGEN VERSION 9.7 *\nTASK = DOS\n")
    return path


def test_loader_indexes_only_out_and_identifies_overwritten_scf(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("current")
    first = _output(tmp_path / "older.out")
    second = _output(tmp_path / "newer.out")
    _output(tmp_path / "dos.out", scf=False)
    (tmp_path / "ignored.log").touch()
    os.utime(first, ns=(1_000_000_000, 1_000_000_000))
    os.utime(second, ns=(2_000_000_000, 2_000_000_000))

    reads = []

    def parse(_cls, path):
        reads.append(Path(path))
        cls = _DOS if Path(path).name == "dos.out" else _SCF
        return cls(None, None, tmp_path, output_file=path,
                   task_name="DOS" if cls is _DOS else "SCF")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    assert [p.name for p in loader.find_outputs(tmp_path)] == [
        "dos.out", "newer.out", "older.out",
    ]
    batch = loader.load_files((first, second, tmp_path / "dos.out"))
    assert not batch.failures
    older, newer, dos = batch.results
    assert older.files["converged"].association is FileAssociation.NOT_THIS_RESULT
    assert older.files["converged"].associated_with == second.resolve()
    assert newer.files["converged"].association is FileAssociation.LIKELY
    assert dos.files["potential"].association is FileAssociation.LIKELY
    with pytest.raises(HistoricalFileMismatchError):
        older.require_current_file(potential)
    assert loader.companion_scf(dos) is newer
    assert loader.load_file(second) is newer
    assert len(reads) == 3


def test_tied_scf_mtimes_are_unknown(tmp_path, monkeypatch):
    (tmp_path / "Fe.pot_new").write_text("data")
    first = _output(tmp_path / "one.out")
    second = _output(tmp_path / "two.out")
    for path in (first, second):
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))

    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path:
                    _SCF(None, None, tmp_path, output_file=path, task_name="SCF")),
    )
    results = ResultLoader.load_directory(tmp_path).results
    assert len(results) == 2
    assert all(r.files["converged"].association is FileAssociation.UNKNOWN
               for r in results)


def test_historical_hash_is_stronger_than_mtime_heuristic(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("original")
    old = _output(tmp_path / "old.out")
    new = _output(tmp_path / "new.out")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new, ns=(2_000_000_000, 2_000_000_000))
    version = FileVersion.capture(potential, require_digest=True)

    def parse(_cls, path):
        result = _SCF(None, None, tmp_path, output_file=path, task_name="SCF")
        if Path(path) == old:
            result.run_record = SimpleNamespace(
                files=SimpleNamespace(captured_at="historical",
                                      all_versions=(version,))
            )
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    old_result = loader.load_file(old)
    option = old_result.files["converged"]
    assert option.association is FileAssociation.VERIFIED
    assert option.recorded_version == version
    potential.write_text("overwritten")
    # Querying the existing result must not keep claiming VERIFIED bytes.
    assert option.association is FileAssociation.NOT_THIS_RESULT
    assert option.association_evidence == "recorded_version_changed"
    loader.load_file(old)
    assert option.association is FileAssociation.NOT_THIS_RESULT
    potential.write_text("original")
    assert option.association is FileAssociation.VERIFIED


def test_file_association_reset_when_repointed(tmp_path):
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    file = result.files.add_file("potential", "old.pot")
    file.set_association(FileAssociation.LIKELY, evidence="native_input")
    assert result.files.add_file("potential", "old.pot") is file
    assert file.association is FileAssociation.LIKELY
    # The option already stores a canonical absolute FilePath. A second
    # registration with a different spelling of the *same* path is not a
    # change of file identity and must preserve the association.
    result.files.add_file("potential", str(tmp_path / "old.pot"))
    assert file.association is FileAssociation.LIKELY
    result.files.add_file("potential", "other.pot")
    assert file.association is FileAssociation.UNKNOWN
    file.set_association(FileAssociation.LIKELY, evidence="repointed")
    file.set(str(tmp_path / "other.pot"))
    assert file.association is FileAssociation.LIKELY
    file.set("third.pot")
    assert file.association is FileAssociation.UNKNOWN


def test_reuse_invalidated_when_input_changes_without_output_change(tmp_path, monkeypatch):
    output = _output(tmp_path / "dos.out", scf=False)
    inp = tmp_path / "dos.inp"
    inp.write_text("old")
    reads = []

    def parse(_cls, path):
        reads.append(path)
        return TaskResult(None, None, tmp_path, output_file=path,
                          input_file=inp, task_name="DOS")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    original = loader.load_file(output)
    assert loader.load_file(output) is original
    assert len(reads) == 1
    inp.write_text("newer input parameters")
    changed = loader.load_file(output)
    assert changed is not original
    assert len(reads) == 2


def test_batch_preserves_good_results_after_one_bad_file(tmp_path, monkeypatch):
    valid = _output(tmp_path / "valid.out", scf=False)
    missing = tmp_path / "missing.out"
    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path:
                    TaskResult(None, None, tmp_path, output_file=path, task_name="DOS")),
    )
    batch = ResultLoader().load_files((missing, valid))
    assert len(batch.results) == 1
    assert batch.results[0].task_name == "dos"
    assert len(batch.failures) == 1
    assert batch.failures[0].path == missing


def test_companion_ignores_header_false_positive_and_tries_older(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("current")
    consumer = _output(tmp_path / "dos.out", scf=False)
    correct = _output(tmp_path / "correct.out")
    misleading = _output(tmp_path / "misleading.out")
    os.utime(correct, ns=(1_000_000_000, 1_000_000_000))
    os.utime(misleading, ns=(2_000_000_000, 2_000_000_000))

    class OtherSCF(_SCF):
        @property
        def potential_filename(self):
            return str(tmp_path / "other.pot_new")

    reads = []

    def parse(_cls, path):
        reads.append(Path(path).name)
        if Path(path) == consumer:
            kind = _DOS
        elif Path(path) == misleading:
            kind = OtherSCF
        else:
            kind = _SCF
        return kind(None, None, tmp_path, output_file=path,
                    task_name="DOS" if kind is _DOS else "SCF")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    dos = loader.load_file(consumer)
    found = loader.companion_scf(dos)
    assert found is not None
    assert Path(found.path_to("output")) == correct.resolve()
    assert reads == ["dos.out", "misleading.out", "correct.out"]


def test_companion_rejects_disagreement_between_registered_file_and_result(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("current")
    consumer = _output(tmp_path / "dos.out", scf=False)
    scf = _output(tmp_path / "scf.out")

    def parse(_cls, path):
        kind = _DOS if Path(path) == consumer else _SCF
        result = kind(None, None, tmp_path, output_file=path,
                      task_name="DOS" if kind is _DOS else "SCF")
        if kind is _SCF:
            result.files.add_file("converged", "wrong.pot_new")
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    assert loader.companion_scf(loader.load_file(consumer)) is None


def test_historical_missing_file_is_unknown_not_wrong_result(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("original")
    output = _output(tmp_path / "scf.out")
    captured = FileVersion.capture(potential, require_digest=True)

    def parse(_cls, path):
        result = _SCF(None, None, tmp_path, output_file=path, task_name="SCF")
        result.run_record = SimpleNamespace(files=SimpleNamespace(
            captured_at="historical", all_versions=(captured,),
        ))
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    result = ResultLoader().load_file(output)
    entry = result.files["converged"]
    assert entry.association is FileAssociation.VERIFIED

    potential.unlink()
    assert entry.association is FileAssociation.UNKNOWN
    assert entry.association_evidence == "recorded_file_missing"
    assert not entry.is_current()

    potential.write_text("overwritten")
    assert entry.association is FileAssociation.NOT_THIS_RESULT


def test_imported_result_refuses_potential_modified_after_loading(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("at import")
    output = _output(tmp_path / "scf.out")
    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path:
                    _SCF(None, None, tmp_path, output_file=path, task_name="SCF")),
    )
    result = ResultLoader().load_file(output)
    file = result.files["converged"]
    assert file.association is FileAssociation.LIKELY
    assert result.file_is_current(potential)

    potential.write_text("later modification")
    assert file.association is FileAssociation.UNKNOWN
    assert not file.is_current()
    assert not result.file_is_current(potential)
    with pytest.raises(HistoricalFileMismatchError):
        result.require_current_file(potential)


def test_overwritten_scf_does_not_capture_wrong_potential_as_history(tmp_path, monkeypatch):
    from ase2sprkkr.outputs import capture_result_file_versions

    potential = tmp_path / "Fe.pot_new"
    potential.write_text("current")
    older = _output(tmp_path / "old.out")
    newer = _output(tmp_path / "new.out")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path:
                    _SCF(None, None, tmp_path, output_file=path, task_name="SCF")),
    )
    old_result = ResultLoader().load_file(older)
    assert old_result.files["converged"].association is FileAssociation.NOT_THIS_RESULT
    assert "converged" not in capture_result_file_versions(old_result)


def test_historical_missing_potential_retains_recorded_version(tmp_path, monkeypatch):
    from ase2sprkkr.outputs import capture_result_file_versions

    potential = tmp_path / "Fe.pot_new"
    potential.write_text("historical")
    output = _output(tmp_path / "historical.out")
    captured = FileVersion.capture(potential, require_digest=True)
    def parse(_cls, path):
        result = _SCF(None, None, tmp_path, output_file=path, task_name="SCF")
        result.run_record = SimpleNamespace(files=SimpleNamespace(
            captured_at="historical", all_versions=(captured,),
        ))
        return result
    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    result = ResultLoader().load_file(output)
    potential.unlink()
    versions = capture_result_file_versions(result)
    assert versions["converged"] == captured


def test_cached_unmatched_input_is_rediscovered_after_repair(tmp_path, monkeypatch):
    output = _output(tmp_path / "dos.out", scf=False)
    inp = tmp_path / "dos.inp"
    inp.write_text("invalid input")
    reads = []

    def parse_output(_cls, path):
        reads.append(path)
        result = _DOS(None, None, tmp_path, output_file=path, task_name="DOS")
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse_output))
    loader = ResultLoader()

    def parse_input(path):
        if inp.read_text() == "invalid input":
            raise ValueError("invalid input")
        return SimpleNamespace(task_name="DOS")

    monkeypatch.setattr(loader.inputs, "parse", parse_input)
    first = loader.load_file(output)
    assert "input" not in first.files
    assert loader.load_file(output) is first

    inp.write_text("repaired native input")
    second = loader.load_file(output)
    assert second is not first
    assert Path(second.path_to("input")) == inp
    assert len(reads) == 2
    assert loader.load_file(output) is second


def test_reparse_after_potential_rewrite_drops_cached_task_result(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("before")
    output = _output(tmp_path / "scf.out")
    reads = []

    def parse(_cls, path):
        reads.append(path)
        return _SCF(None, None, tmp_path, output_file=path, task_name="SCF")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    first = loader.load_file(output)
    assert loader.load_file(output) is first
    assert len(reads) == 1
    potential.write_text("after rewrite")
    second = loader.load_file(output)
    assert second is not first
    assert len(reads) == 2
    assert second.files["converged"].association is FileAssociation.LIKELY


def test_retained_result_does_not_keep_entire_loader_alive(tmp_path, monkeypatch):
    import gc
    import weakref
    from ase2sprkkr import InputParameters

    output = _output(tmp_path / "dos.out", scf=False)
    inp = tmp_path / "dos.inp"
    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "original"
    inp.write_text(parameters.to_string(validate=False))

    def parse(_cls, path):
        return _DOS(None, None, tmp_path, output_file=path, task_name="DOS")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    result = loader.load_file(output)
    assert result.input_parameters_file == str(inp)
    assert result.input_parameters.CONTROL.DATASET() == "original"
    assert "_input_cache" not in vars(result)

    loader_ref = weakref.ref(loader)
    inputs_ref = weakref.ref(loader.inputs)
    del loader
    gc.collect()
    assert loader_ref() is None
    assert inputs_ref() is None

    # The parsed parameters are detached and remain usable without a loader.
    inp.write_text("overwritten after import")
    assert result.input_parameters.CONTROL.DATASET() == "original"


def test_standalone_from_file_uses_a_new_loader_each_time(tmp_path, monkeypatch):
    from ase2sprkkr import InputParameters

    output = _output(tmp_path / "dos.out", scf=False)
    native_input = tmp_path / "dos.inp"
    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "fresh"
    native_input.write_text(parameters.to_string(validate=False))

    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path: _DOS(
            None, None, tmp_path, output_file=path, task_name="DOS"
        )),
    )
    first = TaskResult.from_file(output)
    second = TaskResult.from_file(output)

    assert first is not second
    assert first.input_parameters is not second.input_parameters
    assert first.input_parameters_file == second.input_parameters_file == str(native_input)
    assert first.input_parameters.CONTROL.DATASET() == "fresh"
    assert "_input_cache" not in vars(first)
