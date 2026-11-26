"""ResultLoader discovery, caching and per-file provenance."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from kkrunner.files import FileOperations, FileVersion
from ase2sprkkr.outputs import FileAttribution, ResultLoader
from ase2sprkkr.outputs.task_result import TaskResult
from kkrunner.runs import HistoricalFileMismatchError


def test_associate_incomplete_result_records_file_provenance(tmp_path):
    output = tmp_path / "incomplete.out"
    output.write_text("partial output")
    result = TaskResult(None, None, tmp_path, output_file=output, task_name="DOS")

    assert ResultLoader().associate_result(result) is result
    assert result.files["output"].observed_version.is_current()
    assert result.return_code is None


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

    def parse(_cls, path, *, on_files_parsed=None):
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
    assert older.files["converged"].attribution is FileAttribution.NOT_THIS_RESULT
    assert older.files["converged"].attributed_to == second.resolve()
    assert newer.files["converged"].attribution is FileAttribution.LIKELY
    assert dos.files["potential"].attribution is FileAttribution.LIKELY
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
        classmethod(lambda _cls, path, *, on_files_parsed=None:
                    _SCF(None, None, tmp_path, output_file=path, task_name="SCF")),
    )
    results = ResultLoader.load_directory(tmp_path).results
    assert len(results) == 2
    assert all(r.files["converged"].attribution is FileAttribution.UNKNOWN
               for r in results)


def test_historical_hash_is_stronger_than_mtime_heuristic(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("original")
    old = _output(tmp_path / "old.out")
    new = _output(tmp_path / "new.out")
    os.utime(old, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new, ns=(2_000_000_000, 2_000_000_000))
    version = FileVersion.capture(potential, require_digest=True)

    def parse(_cls, path, *, on_files_parsed=None):
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
    assert option.attribution is FileAttribution.VERIFIED
    assert option.recorded_version == version
    potential.write_text("overwritten")
    # Querying the existing result must not keep claiming VERIFIED bytes.
    assert option.attribution is FileAttribution.NOT_THIS_RESULT
    assert option.attribution_evidence == "recorded_version_changed"
    loader.load_file(old)
    assert option.attribution is FileAttribution.NOT_THIS_RESULT
    potential.write_text("original")
    assert option.attribution is FileAttribution.VERIFIED


def test_file_attribution_reset_when_repointed(tmp_path):
    (tmp_path / "old.pot").write_text("old")
    (tmp_path / "other.pot").write_text("other")
    result = TaskResult(None, None, tmp_path, task_name="DOS")
    file = result.files.add_file("potential", "old.pot")
    file.set_attribution(FileAttribution.LIKELY, evidence="native_input")
    assert result.files.add_file("potential", "old.pot") is file
    assert file.attribution is FileAttribution.LIKELY
    # The option already stores a canonical absolute FilePath. A second
    # registration with a different spelling of the *same* path is not a
    # change of file identity and must preserve the association.
    result.files.add_file("potential", str(tmp_path / "old.pot"))
    assert file.attribution is FileAttribution.LIKELY
    result.files.add_file("potential", "other.pot")
    assert file.attribution is FileAttribution.UNKNOWN
    file.set_attribution(FileAttribution.LIKELY, evidence="repointed")
    file.set(str(tmp_path / "other.pot"))
    assert file.attribution is FileAttribution.LIKELY
    file.set("third.pot")
    assert file.attribution is FileAttribution.UNKNOWN


def test_reuse_invalidated_when_input_changes_without_output_change(tmp_path, monkeypatch):
    output = _output(tmp_path / "dos.out", scf=False)
    inp = tmp_path / "dos.inp"
    inp.write_text("old")
    reads = []

    def parse(_cls, path, *, on_files_parsed=None):
        reads.append(path)
        return TaskResult(None, None, tmp_path, output_file=path,
                          input_file=inp, task_name="DOS")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    original = loader.load_file(output)
    assert loader.load_file(output) is original
    assert len(reads) == 1
    inp.write_text("newer input parameters")
    loader.file_operations.invalidate()
    changed = loader.load_file(output)
    assert changed is not original
    assert len(reads) == 2


def test_batch_preserves_good_results_after_one_bad_file(tmp_path, monkeypatch):
    valid = _output(tmp_path / "valid.out", scf=False)
    missing = tmp_path / "missing.out"
    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path, *, on_files_parsed=None:
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

    def parse(_cls, path, *, on_files_parsed=None):
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

    def parse(_cls, path, *, on_files_parsed=None):
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

    def parse(_cls, path, *, on_files_parsed=None):
        result = _SCF(None, None, tmp_path, output_file=path, task_name="SCF")
        result.run_record = SimpleNamespace(files=SimpleNamespace(
            captured_at="historical", all_versions=(captured,),
        ))
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    result = ResultLoader().load_file(output)
    entry = result.files["converged"]
    assert entry.attribution is FileAttribution.VERIFIED

    potential.unlink()
    assert entry.attribution is FileAttribution.UNKNOWN
    assert entry.attribution_evidence == "recorded_file_missing"
    assert not entry.is_current()

    potential.write_text("overwritten")
    assert entry.attribution is FileAttribution.NOT_THIS_RESULT


def test_imported_result_refuses_potential_modified_after_loading(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("at import")
    output = _output(tmp_path / "scf.out")
    monkeypatch.setattr(
        TaskResult, "_parse_file",
        classmethod(lambda _cls, path, *, on_files_parsed=None:
                    _SCF(None, None, tmp_path, output_file=path, task_name="SCF")),
    )
    result = ResultLoader().load_file(output)
    file = result.files["converged"]
    assert file.attribution is FileAttribution.LIKELY
    assert result.file_is_current(potential)

    potential.write_text("later modification")
    assert file.attribution is FileAttribution.UNKNOWN
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
        classmethod(lambda _cls, path, *, on_files_parsed=None:
                    _SCF(None, None, tmp_path, output_file=path, task_name="SCF")),
    )
    old_result = ResultLoader().load_file(older)
    assert old_result.files["converged"].attribution is FileAttribution.NOT_THIS_RESULT
    assert "converged" not in capture_result_file_versions(old_result)


def test_historical_missing_potential_retains_recorded_version(tmp_path, monkeypatch):
    from ase2sprkkr.outputs import capture_result_file_versions

    potential = tmp_path / "Fe.pot_new"
    potential.write_text("historical")
    output = _output(tmp_path / "historical.out")
    captured = FileVersion.capture(potential, require_digest=True)
    def parse(_cls, path, *, on_files_parsed=None):
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

    def parse_output(_cls, path, *, on_files_parsed=None):
        reads.append(path)
        result = _DOS(None, None, tmp_path, output_file=path, task_name="DOS")
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse_output))
    loader = ResultLoader()

    def parse_input(path):
        if inp.read_text() == "invalid input":
            raise ValueError("invalid input")
        from ase2sprkkr import InputParameters
        return InputParameters.create("DOS")

    monkeypatch.setattr(loader.inputs, "parse", parse_input)
    first = loader.load_file(output)
    assert "input" not in first.files
    assert loader.load_file(output) is first

    inp.write_text("repaired native input")
    loader.file_operations.invalidate()
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

    def parse(_cls, path, *, on_files_parsed=None):
        reads.append(path)
        return _SCF(None, None, tmp_path, output_file=path, task_name="SCF")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    loader = ResultLoader()
    first = loader.load_file(output)
    assert loader.load_file(output) is first
    assert len(reads) == 1
    potential.write_text("after rewrite")
    loader.file_operations.invalidate()
    second = loader.load_file(output)
    assert second is not first
    assert len(reads) == 2
    assert second.files["converged"].attribution is FileAttribution.LIKELY


def test_retained_result_does_not_keep_entire_loader_alive(tmp_path, monkeypatch):
    import gc
    import weakref
    from ase2sprkkr import InputParameters

    output = _output(tmp_path / "dos.out", scf=False)
    inp = tmp_path / "dos.inp"
    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "original"
    inp.write_text(parameters.to_string(validate=False))

    def parse(_cls, path, *, on_files_parsed=None):
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
        classmethod(lambda _cls, path, *, on_files_parsed=None: _DOS(
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


def test_native_input_is_available_after_files_header(tmp_path, monkeypatch):
    from ase2sprkkr import InputParameters

    output = _output(tmp_path / "dos.out", scf=False)
    native_input = tmp_path / "dos.inp"
    parameters = InputParameters.create("DOS")
    parameters.CONTROL.DATASET = "prepared"
    native_input.write_text(parameters.to_string(validate=False))
    observed = []

    def parse(_cls, path, *, on_files_parsed=None):
        result = _DOS(None, None, tmp_path, output_file=path, task_name="DOS")
        assert result.input_parameters is None
        assert on_files_parsed is not None
        on_files_parsed(result)
        observed.append(result.input_parameters.CONTROL.DATASET())
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    result = ResultLoader().load_file(output)
    assert observed == ["prepared"]
    assert result.input_parameters_file == str(native_input)


def test_file_operations_cache_is_shared_by_nested_import(tmp_path, monkeypatch):
    first = _output(tmp_path / "one.out", scf=False)
    second = _output(tmp_path / "two.out", scf=False)
    operations = FileOperations()
    loader = ResultLoader(file_operations=operations)
    assert loader.file_operations is operations
    assert loader.inputs._file_operations is operations

    def parse(_cls, path, *, on_files_parsed=None):
        return _DOS(None, None, tmp_path, output_file=path, task_name="DOS")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    invalidations = []
    original_invalidate = operations.invalidate

    def invalidate():
        invalidations.append(1)
        original_invalidate()

    monkeypatch.setattr(operations, "invalidate", invalidate)
    batch = loader.load_files((first, second))
    assert len(batch.results) == 2
    assert not batch.failures
    # Neither nested nor independent calls may discard a supplied cache.
    assert len(invalidations) == 0

    assert loader.load_file(first) is batch.results[0]
    assert len(invalidations) == 0


def test_file_operations_loader_preserves_cache_until_explicit_invalidation(tmp_path):
    loader = ResultLoader()
    first = _output(tmp_path / "first.out", scf=False)
    assert loader.find_outputs(tmp_path) == (first,)
    (tmp_path / "artifact.pot").write_bytes(b"ignore")
    second = _output(tmp_path / "second.out", scf=False)
    assert loader.find_outputs(tmp_path) == (first,)
    loader.file_operations.invalidate()
    assert loader.find_outputs(tmp_path) == (first, second)


def test_directory_scan_stats_only_candidate_files(tmp_path, monkeypatch):
    loader = ResultLoader()
    _output(tmp_path / "task.out", scf=False)
    (tmp_path / "task.inp").write_text("control")
    (tmp_path / "bulk.dat").write_bytes(b"unrelated")
    (tmp_path / "large.pot").write_bytes(b"unrelated")
    examined = []
    real_is_file = loader.file_operations.is_file

    def observe(path, *, refresh=False):
        examined.append(Path(path).name)
        return real_is_file(path, refresh=refresh)

    monkeypatch.setattr(loader.file_operations, "is_file", observe)
    assert [path.name for path in loader.find_outputs(tmp_path)] == ["task.out"]
    assert set(examined) == {"task.out", "task.inp"}


def test_loader_rejects_stdout_rewritten_while_parsing(tmp_path, monkeypatch):
    path = _output(tmp_path / "task.out", scf=False)

    def parse(_cls, output, *, on_files_parsed=None):
        Path(output).write_text("changed during parsing")
        return _DOS(None, None, tmp_path, output_file=output, task_name="DOS")

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    with pytest.raises(RuntimeError, match="Output changed during parsing"):
        ResultLoader().load_file(path)


def test_native_input_changed_during_parse_is_not_cached(tmp_path, monkeypatch):
    from ase2sprkkr.outputs import result_loader as module

    native_input = tmp_path / "calculation.inp"
    native_input.write_text("old")
    loader = ResultLoader()

    def mutate(_cls, path):
        native_input.write_text("changed during read")
        return SimpleNamespace(task_name="SCF")

    monkeypatch.setattr(module.input_parameters.InputParameters, "from_file", classmethod(mutate))
    with pytest.raises(RuntimeError, match="Native input changed during parsing"):
        loader.inputs.parse(native_input)

    # The old fingerprint's failed parse must not conceal a repaired input.
    monkeypatch.setattr(
        module.input_parameters.InputParameters,
        "from_file",
        classmethod(lambda _cls, path: SimpleNamespace(task_name="SCF")),
    )
    assert loader.inputs.parse(native_input).task_name == "SCF"


def test_file_index_refreshes_symlink_target_without_parent_change(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = root / "linked.out"
    try:
        link.symlink_to(target / "linked.out")
    except OSError:
        pytest.skip("Symlinks not supported")
    loader = ResultLoader()
    assert loader.find_outputs(root) == ()
    before = root.stat()

    _output(target / "linked.out", scf=False)
    assert root.stat().st_mtime_ns == before.st_mtime_ns
    assert loader.find_outputs(root) == (link,)


def test_load_directory_scans_candidates_once_across_nested_calls(tmp_path, monkeypatch):
    from kkrunner.files import FileOperations

    first = _output(tmp_path / "first.out", scf=False)
    second = _output(tmp_path / "second.out", scf=False)
    (tmp_path / "irrelevant.dat").write_bytes(b"skip")
    scans = []
    original_listdir = FileOperations.listdir

    def listdir(self, directory, *, refresh=False):
        scans.append(Path(directory))
        return original_listdir(self, directory, refresh=refresh)

    monkeypatch.setattr(FileOperations, "listdir", listdir)
    monkeypatch.setattr(
        TaskResult,
        "_parse_file",
        classmethod(lambda _cls, path, *, on_files_parsed=None:
                    _DOS(None, None, tmp_path, output_file=path, task_name="DOS"))
    )
    batch = ResultLoader.load_directory(tmp_path)
    assert len(batch.results) == 2
    assert not batch.failures
    assert scans == [tmp_path]

def test_native_input_cache_is_caller_scoped(tmp_path, monkeypatch):
    from ase2sprkkr.outputs import result_loader as module

    native_input = tmp_path / "native.inp"
    native_input.write_text("initial")
    loader = ResultLoader()
    parsed = []

    def parse(_cls, path):
        value = Path(path).read_text()
        parsed.append(value)
        return SimpleNamespace(task_name="DOS", value=value)

    monkeypatch.setattr(module.input_parameters.InputParameters, "from_file", classmethod(parse))
    assert loader.inputs.parse(native_input).value == "initial"
    native_input.write_text("modified")
    # This is an intentional cache: no automatic stat or parse on repeat.
    assert loader.inputs.parse(native_input).value == "initial"
    assert parsed == ["initial"]

    loader.file_operations.invalidate()
    assert loader.inputs.parse(native_input).value == "modified"
    assert parsed == ["initial", "modified"]


def test_run_record_without_available_manifest_never_adopts_scf_potential(tmp_path, monkeypatch):
    potential = tmp_path / "Fe.pot_new"
    potential.write_text("potential")
    output = _output(tmp_path / "historical.out")

    def parse(_cls, path, *, on_files_parsed=None):
        result = _SCF(None, None, tmp_path, output_file=path, task_name="SCF")
        result.run_record = SimpleNamespace(files=None)
        return result

    monkeypatch.setattr(TaskResult, "_parse_file", classmethod(parse))
    result = ResultLoader().load_file(output)
    option = result.files["converged"]
    assert option.attribution is FileAttribution.UNKNOWN
    assert not result.file_is_current(potential)
