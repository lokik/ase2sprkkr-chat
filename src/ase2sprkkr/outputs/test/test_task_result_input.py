from types import SimpleNamespace

from ase2sprkkr import InputParameters
from ase2sprkkr.outputs import ResultLoader
from ase2sprkkr.outputs.task_result import KkrOutputReader, TaskResult


class _Parser:
    def __init__(self, print_info, on_info=None):
        self.print_info = print_info

    def read_from_file(self, output, error, read_args, return_code):
        on_files_parsed = getattr(self, "_on_files_parsed", None)
        if on_files_parsed is not None:
            on_files_parsed(read_args[0])
        return None, error, return_code


class _Reader(KkrOutputReader):
    parser_class = _Parser
    result_class = TaskResult


def test_explicit_input_file_is_registered(tmp_path):
    input_file = tmp_path / "run.inp"
    input_file.write_text("input")

    result = TaskResult(
        None,
        None,
        str(tmp_path),
        output_file="run.out",
        input_file=str(input_file),
    )

    assert result.path_to("input") == str(input_file)
    assert result.input_parameters_file == str(input_file)


def test_result_uses_effective_task_identity(tmp_path):
    parameters = InputParameters.create("ARPES")
    parameters.TASK.TASK = "SPLEED"

    assert parameters.requested_task_name == "ARPES"
    assert parameters.task_name == "SPLEED"
    assert parameters.task_definition.name == "SPLEED"

    result = TaskResult(parameters, None, str(tmp_path))
    assert result.task_name == "spleed"

    reader = _Reader(parameters, None, str(tmp_path))
    assert reader.task_name == "SPLEED"
    assert reader.create_result("run.out").task_name == "spleed"


def test_jxc_continuation_uses_source_result_energy(tmp_path):
    source_parameters = SimpleNamespace(
        CONTROL=SimpleNamespace(POTFIL=lambda: "Fe.pot")
    )
    result = TaskResult(
        source_parameters,
        None,
        str(tmp_path),
        task_name="SCF",
    )
    result.last_iteration = SimpleNamespace(
        energy=SimpleNamespace(EMIN=lambda: -0.35)
    )

    continuation = result.new_task("JXC")

    assert continuation.input_parameters.task_definition.name == "JXC"
    assert continuation.input_parameters.ENERGY.EMIN() == -0.35

def test_autodetect_single_matching_native_input_file(tmp_path):
    parameters = InputParameters.create("ARPES")
    parameters.CONTROL.DATASET = "sample"
    parameters.CONTROL.POTFIL = "sample.pot"
    source = tmp_path / "sample_SPEC.inp"
    source.write_text(parameters.to_string(validate=False))
    result = TaskResult(None, None, str(tmp_path), output_file="sample_SPEC.out", task_name="ARPES")
    assert result.input_parameters_file is None
    ResultLoader().associate_result(result)
    assert result.input_parameters_file == str(source)
    assert result.input_parameters.task_name == "ARPES"


def test_autodetect_is_not_ambiguous_or_cross_task(tmp_path):
    for name in ("first.inp", "second.inp"):
        (tmp_path / name).write_text("invalid")
    result = TaskResult(None, None, str(tmp_path), output_file="unknown.out", task_name="ARPES")
    ResultLoader().associate_result(result)
    assert result.input_parameters is None

def test_no_result_task_identity_raises_clear_error(tmp_path):
    import pytest

    result = TaskResult(None, None, str(tmp_path))
    assert result.input_parameters is None
    with pytest.raises(ValueError, match="Cannot determine the task"):
        _ = result.task_name


def test_historical_input_autodetect_accepts_in_suffix(tmp_path):
    params = InputParameters.create("ARPES")
    params.CONTROL.DATASET = "sample"
    params.CONTROL.POTFIL = "sample.pot"
    source = tmp_path / "sample_SPEC.in"
    source.write_text(params.to_string(validate=False))
    result = TaskResult(None, None, str(tmp_path), output_file="sample_SPEC.out", task_name="ARPES")
    assert result.input_parameters_file is None
    ResultLoader().associate_result(result)
    assert result.input_parameters_file == str(source)
    assert result.input_parameters.task_name == "ARPES"


def test_invalid_guessed_input_is_nonfatal_but_explicit_input_error_propagates(
    tmp_path, monkeypatch
):
    import pytest

    path = tmp_path / "legacy.inp"
    path.write_text("invalid input")

    def invalid_input(_cls, filename):
        raise ValueError("invalid historical input")

    monkeypatch.setattr(InputParameters, "from_file", classmethod(invalid_input))
    guessed = TaskResult(None, None, str(tmp_path), output_file="legacy.out", task_name="SCF")
    ResultLoader().associate_result(guessed)
    assert guessed.input_parameters is None
    assert guessed.input_parameters_file is None

    explicit = TaskResult(None, None, str(tmp_path), input_file=str(path))
    with pytest.raises(ValueError, match="invalid historical input"):
        _ = explicit.input_parameters


def test_loader_input_cache_parses_once_and_returns_detached_parameters(
    tmp_path, monkeypatch,
):
    source = tmp_path / "shared.inp"
    parameters = InputParameters.create("ARPES")
    parameters.CONTROL.DATASET = "shared"
    source.write_text(parameters.to_string(validate=False))
    original = InputParameters.from_file
    calls = []

    def read(_cls, filename):
        calls.append(filename)
        return original(filename)

    monkeypatch.setattr(InputParameters, "from_file", classmethod(read))
    loader = ResultLoader()
    first = TaskResult(None, None, str(tmp_path), output_file="one.out", task_name="ARPES")
    second = TaskResult(None, None, str(tmp_path), output_file="two.out", task_name="ARPES")
    loader.associate_result(first)
    loader.associate_result(second)

    a, b = first.input_parameters, second.input_parameters
    assert len(calls) == 1
    assert a is not b
    assert a.task_name == b.task_name == "ARPES"
    assert "_input_cache" not in vars(first)
    a.CONTROL.DATASET = "changed"
    assert b.CONTROL.DATASET() == "shared"

    # A changed file must never inherit the earlier parsed input.
    parameters.CONTROL.DATASET = "replaced-by-new-content"
    source.write_text(parameters.to_string(validate=False))
    third = TaskResult(None, None, str(tmp_path), output_file="three.out", task_name="ARPES")
    loader.associate_result(third)
    assert third.input_parameters.CONTROL.DATASET() == "replaced-by-new-content"
    assert len(calls) == 2

    # An explicit re-import starts a fresh batch, not a retained cache.
    fresh = TaskResult(None, None, str(tmp_path), output_file="four.out", task_name="ARPES")
    ResultLoader().associate_result(fresh)
    assert fresh.input_parameters.CONTROL.DATASET() == "replaced-by-new-content"
    assert len(calls) == 3


def test_loader_caches_invalid_guessed_input_but_explicit_error_propagates(
    tmp_path, monkeypatch,
):
    import pytest

    path = tmp_path / "bad.inp"
    path.write_text("invalid")
    calls = []

    def fail(_cls, _filename):
        calls.append(True)
        raise ValueError("invalid input")

    monkeypatch.setattr(InputParameters, "from_file", classmethod(fail))
    loader = ResultLoader()
    for index in range(2):
        guessed = TaskResult(None, None, str(tmp_path), output_file=f"guess-{index}.out", task_name="SCF")
        loader.associate_result(guessed)
        assert guessed.input_parameters is None
    # Repeated speculative guesses reuse the one cached parse failure.
    assert len(calls) == 1

    explicit = TaskResult(None, None, str(tmp_path), input_file=str(path))
    with pytest.raises(ValueError, match="invalid input"):
        _ = explicit.input_parameters


def test_reader_files_parsed_hook_is_ephemeral_and_ordered(tmp_path, monkeypatch):
    reader = _Reader(None, None, str(tmp_path), task_name="DOS")
    observed = []

    def prepare(result):
        observed.append("prepare")
        assert result.input_parameters is None
        result.input_parameters = InputParameters.create("DOS")

    def complete(self, error, return_code):
        assert self.input_parameters.task_name == "DOS"
        observed.append("complete")

    monkeypatch.setattr(TaskResult, "complete", complete)
    result = reader.read_from_file(
        str(tmp_path / "example.out"), on_files_parsed=prepare
    )
    assert observed == ["prepare", "complete"]
    assert result.input_parameters.task_name == "DOS"
    assert "_on_files_parsed" not in vars(reader.parser)
    assert "_on_files_parsed" not in vars(result)
