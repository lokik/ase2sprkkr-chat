"""Historical OUT fallback is opt-in and does not hide unrelated errors."""

from types import SimpleNamespace

import pyparsing as pp
import pytest
from kkrunner import RunState
from kkrunner.files import FileEntry, FileManifest, FileVersion

from ase2sprkkr.common.process_output_reader import ProcessExitError, ProcessOutputParser
from ase2sprkkr.common.warnings import DataValidityError
from ase2sprkkr.outputs import OutputOnlyResult, project_results


@pytest.fixture
def saved_run(tmp_path):
    output = tmp_path / "crashed.out"
    output.write_text("VERSION SPR-KKR\nSCF iteration 1\n")
    files = FileManifest(
        stdout=FileEntry(str(output)),
        output_versions=(FileVersion.capture(output, require_digest=True),),
        captured_at="2026-11-18T00:00:00+00:00",
    )
    record = SimpleNamespace(
        id="run", state=RunState.FAILED, files=files,
        metadata={"ase2sprkkr": {"task": "SCF"}},
        execution=SimpleNamespace(directory=SimpleNamespace(local=str(tmp_path))),
        return_code=139,
    )

    class Manager:
        def __init__(self):
            self.store = SimpleNamespace(get=lambda _id: record)
            self.current = True
            self.retrieve_calls = 0

        def retrieve_result_files(self, value):
            self.retrieve_calls += 1
            return value

        def output_file_is_current(self, value):
            return self.current

        def input_file_is_current(self, value):
            return False

    return Manager(), record, output


def fail_read(monkeypatch, error, *, on_read=None):
    class Reader:
        input_parameters = None

        def read_from_file(self, *_args, **_kwargs):
            if on_read is not None:
                on_read()
            raise error

    monkeypatch.setattr(project_results, "_reader_for_record",
                        lambda _manager, _record: Reader())


@pytest.mark.parametrize("error", [
    EOFError("Truncated OUT"),
    UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte"),
    pp.ParseException("Malformed native output"),
    DataValidityError("Invalid native data"),
    ProcessExitError("The process ended with return value 139"),
])
def test_opt_in_exposes_original_output(saved_run, monkeypatch, error):
    manager, record, output = saved_run
    fail_read(monkeypatch, error)
    result = project_results.load_project_result(manager, record.id, allow_unparsed=True)
    assert isinstance(result, OutputOnlyResult)
    assert result.run_record is record
    assert result.return_code == 139
    assert result.task_name == "scf"
    assert not result.produces_potential
    assert result.input_parameters is None
    assert tuple(result.output_values) == ("output",)
    assert result.output_values["output"].path() == str(output)
    assert "open" in result.output_values["output"].actions()
    assert "plot" not in result.output_values["output"].actions()
    assert type(error).__name__ in result.parse_error
    assert manager.retrieve_calls == 1


def test_strict_loading_reraises_parse_failure(saved_run, monkeypatch):
    manager, record, _ = saved_run
    fail_read(monkeypatch, EOFError("Truncated OUT"))
    with pytest.raises(EOFError, match="Truncated OUT"):
        project_results.load_project_result(manager, record.id)


@pytest.mark.parametrize("error", [
    ValueError("Bug"), RuntimeError("Bug"), TypeError("Bug"),
    KeyError("Bug"), AttributeError("Bug"), IndexError("Bug"),
    OSError("Cannot read file"),
])
def test_unrecognized_failures_still_raise(saved_run, monkeypatch, error):
    manager, record, _ = saved_run
    fail_read(monkeypatch, error)
    with pytest.raises(type(error)) as caught:
        project_results.load_project_result(manager, record.id, allow_unparsed=True)
    assert caught.value is error


def test_reader_construction_errors_do_not_fall_back(saved_run, monkeypatch):
    manager, record, _ = saved_run
    def broken_reader(*_args):
        raise EOFError("Reader setup failed")
    monkeypatch.setattr(project_results, "_reader_for_record", broken_reader)
    with pytest.raises(EOFError, match="Reader setup failed"):
        project_results.load_project_result(manager, record.id, allow_unparsed=True)


def test_unverified_output_is_rejected(saved_run, monkeypatch):
    manager, record, _ = saved_run
    manager.current = False
    fail_read(monkeypatch, EOFError("Truncated OUT"))
    with pytest.raises(project_results.ProjectResultUnavailableError):
        project_results.load_project_result(manager, record.id, allow_unparsed=True)


def test_output_changed_during_parse_is_rejected(saved_run, monkeypatch):
    manager, record, _ = saved_run
    fail_read(monkeypatch, EOFError("Truncated OUT"),
              on_read=lambda: setattr(manager, "current", False))
    with pytest.raises(project_results.ProjectResultUnavailableError,
                       match="changed during parsing"):
        project_results.load_project_result(manager, record.id, allow_unparsed=True)


def test_output_disappearing_during_parse_is_rejected(saved_run, monkeypatch):
    manager, record, output = saved_run
    fail_read(monkeypatch, EOFError("Truncated OUT"), on_read=output.unlink)
    with pytest.raises(project_results.ProjectResultUnavailableError,
                       match="changed during parsing"):
        project_results.load_project_result(manager, record.id, allow_unparsed=True)


def test_active_run_has_no_fallback(saved_run, monkeypatch):
    manager, record, _ = saved_run
    record.state = RunState.RUNNING
    fail_read(monkeypatch, EOFError("Truncated OUT"))
    with pytest.raises(project_results.ProjectResultUnavailableError, match="still active"):
        project_results.load_project_result(manager, record.id, allow_unparsed=True)
    assert manager.retrieve_calls == 0


def test_output_action_rechecks_saved_version(saved_run, monkeypatch):
    manager, record, output = saved_run
    fail_read(monkeypatch, EOFError("Truncated OUT"))
    result = project_results.load_project_result(manager, record.id, allow_unparsed=True)
    output.write_text("overwritten")
    assert result.output_values["output"].actions() == ()


def test_process_exit_error_remains_a_value_error():
    with pytest.raises(ProcessExitError, match="return value 3") as caught:
        ProcessOutputParser().result(None, None, 3)
    assert isinstance(caught.value, ValueError)
    assert caught.value.output is None
