"""The planned filename/parameter snapshot is independent of file writes."""

from pathlib import Path
import os
from kkrunner import TaskFile

from ase.build import bulk
import pytest

from ..calculator import SPRKKR


def _calculator(tmp_path):
    return SPRKKR(atoms=bulk("Li"), directory=tmp_path, empty_spheres=False, mpi=False)


def test_plan_resolves_exact_paths_before_materializing(tmp_path):
    calculator = _calculator(tmp_path)
    plan = calculator.plan(input_file="case.inp", potential_file="case.pot", output_file="case.out")
    try:
        paths = (*plan.written_inputs, plan.output_file)
        assert set(paths) == {
            str(tmp_path / name) for name in ("case.inp", "case.pot", "case.out")
        }
        assert plan.input_parameters.CONTROL.DATASET.result == "case"
        assert plan.input_parameters.CONTROL.POTFIL.result == str(tmp_path / "case.pot")
        assert not any(Path(path).exists() for path in paths)
        prepared = plan.prepare()
        assert prepared.input_file.name == str(tmp_path / "case.inp")
        assert prepared.potential_file == str(tmp_path / "case.pot")
        assert prepared.output_file.name == str(tmp_path / "case.out")
        assert all(Path(path).exists() for path in paths)
        with pytest.raises(RuntimeError, match="already materialized"):
            plan.prepare()
    finally:
        plan.close()


def test_closing_unmaterialized_plan_does_not_modify_existing_files(tmp_path):
    for name in ("case.inp", "case.pot", "case.out"):
        (tmp_path / name).write_text("original")
    plan = _calculator(tmp_path).plan(
        input_file="case.inp", potential_file="case.pot", output_file="case.out",
    )
    plan.close()
    assert all((tmp_path / name).read_text() == "original" for name in ("case.inp", "case.pot", "case.out"))


def test_temporary_file_paths_are_chosen_before_creation(tmp_path):
    plan = _calculator(tmp_path).plan(input_file=False, potential_file=False, output_file=False)
    paths = tuple(map(Path, plan.written_inputs))
    try:
        assert len(paths) == 2
        assert not any(path.exists() for path in paths)
        prepared = plan.prepare()
        assert prepared.temporary_potential
        assert all(path.exists() for path in paths)
    finally:
        plan.close()
    assert not any(path.exists() for path in paths)


def test_save_input_cannot_overwrite_reserved_potential(tmp_path):
    from kkrunner import ExecutionSpec, MemoryRunStore, RunManager
    from kkrunner.runs import DuplicateSubmissionError

    calculator = _calculator(tmp_path)
    potential = tmp_path / "case.pot"
    potential.write_text("original")
    spec = ExecutionSpec(argv=(), directory=str(tmp_path), input_files=(TaskFile(potential, write=True),))
    manager = RunManager(MemoryRunStore())
    with manager.preprocessing_reservation(spec):
        with pytest.raises(DuplicateSubmissionError):
            calculator.save_input(input_file="case.inp", potential_file="case.pot",
                                  output_file=False, run_manager=manager)
    assert potential.read_text() == "original"
    assert not (tmp_path / "case.inp").exists()


def test_unstarted_runner_does_not_truncate_and_conflicts_before_materialization(tmp_path):
    from kkrunner import ExecutionSpec, MemoryRunStore, RunManager, RunStore
    from kkrunner.files import FileManifest
    from kkrunner.runs import DuplicateSubmissionError

    disk = RunStore(tmp_path / "runs")
    memory = MemoryRunStore()
    manager = RunManager(memory, conflict_stores=(disk,))
    persistent = RunManager(disk, conflict_stores=(memory,))
    calculator = _calculator(tmp_path)
    for name in ("case.inp", "case.pot", "case.out", "case.out.err"):
        (tmp_path / name).write_text("original")
    runner = calculator.create_run(input_file="case.inp", potential_file="case.pot",
                                   output_file="case.out", project=manager)
    assert runner.record.store is memory
    try:
        assert all((tmp_path / name).read_text() == "original"
                   for name in ("case.inp", "case.pot", "case.out", "case.out.err"))
        old = persistent.reserve(persistent.new_record(
            directory=tmp_path, files=FileManifest.from_spec(
                ExecutionSpec(argv=(), directory=tmp_path, input_file="case.pot"))))
        with pytest.raises(DuplicateSubmissionError):
            runner.submit()
        assert all((tmp_path / name).read_text() == "original"
                   for name in ("case.inp", "case.pot", "case.out", "case.out.err"))
        assert disk.runs() == (old,)
        assert not memory.runs()
    finally:
        runner.close()


def test_relative_open_file_has_one_absolute_execution_and_reservation_path(tmp_path, monkeypatch):
    from kkrunner import MemoryRunStore, RunManager

    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "work"
    directory.mkdir()
    with open("outside.inp", "w+") as handle:
        plan = _calculator(directory).plan(
            input_file=handle, potential_file="case.pot", output_file=False,
        )
        assert plan.input_file == str(tmp_path / "outside.inp")
        manager = RunManager(MemoryRunStore())
        runner = plan.create_run(profile=_calculator(directory).profile, project=manager)
        try:
            assert os.fspath(runner.spec.input_file) == str(tmp_path / "outside.inp")
            assert runner.record.files.inputs[0].write
            assert str(directory / "outside.inp") not in runner.record.files.input_files
            assert not (directory / "outside.inp").exists()
        finally:
            runner.close()


def test_relative_prepared_output_uses_same_directory_for_default_stderr(tmp_path, monkeypatch):
    from kkrunner import RunManager, MemoryRunStore

    monkeypatch.chdir(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    with open("outside.out", "w+b") as output:
        calculator = _calculator(work)
        prepared = calculator.prepare(input_file="case.inp", potential_file="case.pot", output_file=output)
        runner = prepared.create_run(profile=calculator.profile, project=RunManager(MemoryRunStore()))
        try:
            assert runner.spec.output_file == str(tmp_path / "outside.out")
            assert runner.spec.error_file == str(tmp_path / "outside.out.err")
        finally:
            runner.close()
            prepared.close()


def test_rejected_plan_does_not_close_or_flush_borrowed_stream(tmp_path):
    from kkrunner import ExecutionSpec, RunManager, MemoryRunStore
    from kkrunner.runs import DuplicateSubmissionError

    manager = RunManager(MemoryRunStore())
    path = tmp_path / "input.inp"
    path.write_text("original")
    with path.open("r+") as handle:
        handle.write("buffered")
        calculator = _calculator(tmp_path)
        plan = calculator.plan(input_file=handle, potential_file="case.pot", output_file=False)
        runner = plan.create_run(profile=calculator.profile, project=manager)
        with manager.preprocessing_reservation(
            ExecutionSpec(argv=(), directory=tmp_path, input_files=(TaskFile(path, write=True),))
        ):
            with pytest.raises(DuplicateSubmissionError):
                runner.submit()
            assert not handle.closed
            assert path.read_text() == "original"
        runner.close()
        assert not handle.closed


def test_execution_preprocessing_does_not_open_output_streams(tmp_path, monkeypatch):
    from ase2sprkkr.sprkkr import _preparation
    from kkrunner.client.file_session import FileSession

    calculator = _calculator(tmp_path)
    runner = calculator.create_run(input_file="case.inp", potential_file="case.pot",
                                   output_file="case.out")
    opened = []
    original = _preparation.open_file

    def record(filename, *args, **kwargs):
        opened.append(str(filename))
        return original(filename, *args, **kwargs)

    monkeypatch.setattr(_preparation, "open_file", record)
    def preflight(self):
        assert (tmp_path / "case.inp").is_file()
        assert (tmp_path / "case.pot").is_file()
        assert not (tmp_path / "case.out").exists()
        assert not (tmp_path / "case.out.err").exists()
        raise ValueError("stop after preprocessing")

    monkeypatch.setattr(FileSession, "prepare_submission", preflight)
    with pytest.raises(ValueError, match="stop after preprocessing"):
        runner.submit()
    assert set(opened) == {str(tmp_path / "case.inp"), str(tmp_path / "case.pot")}
    assert {entry.local_path for entry in runner.record.files.inputs if entry.write} == set(opened)


def test_uncertain_submission_keeps_temporary_paths_and_closes_handles(tmp_path, monkeypatch):
    import shutil
    from kkrunner import RunState

    calculator = _calculator(tmp_path)
    plan = calculator.plan(directory=False, input_file=False, potential_file=False, output_file="case.out")
    runner = plan.create_run(profile=calculator.profile, error_file=False)
    directory = Path(plan.directory)
    try:
        def uncertain():
            raise ConnectionError("lost submission reply")
        monkeypatch.setattr(runner._job, "submit", uncertain)
        with pytest.raises(ConnectionError, match="lost submission reply"):
            runner.submit()
        # The durable submission ticket remains SUBMITTING until recovery
        # can determine whether a native job was actually accepted.
        assert runner.record.state == RunState.SUBMITTING
        assert plan._prepared.closed
        assert plan._prepared.input_file.closed
        assert directory.is_dir()
        assert Path(plan.input_file).is_file()
        assert Path(plan.potential_file).is_file()
        assert runner.record is not None and not runner.record.state.is_terminal
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_detached_calculation_requires_persistent_inputs(tmp_path):
    from kkrunner.backends import SupervisedBackend
    from kkrunner.client import EndpointClient

    calculator = _calculator(tmp_path)
    plan = calculator.plan(input_file=False, potential_file="case.pot", output_file="case.out")
    with pytest.raises(ValueError, match="persistent input and potential"):
        plan.create_run(profile=calculator.profile, executor=EndpointClient(SupervisedBackend))
    assert not Path(plan.input_file).exists()
    assert not (tmp_path / "case.pot").exists()


@pytest.mark.parametrize("execute", [False, True])
def test_reused_input_changed_after_planning_is_rejected_before_writes(tmp_path, execute):
    calculator = _calculator(tmp_path)
    prepared = calculator.prepare(input_file="source.inp", potential_file="source.pot", output_file=False)
    prepared.close()
    source = tmp_path / "source.inp"
    plan = calculator.plan(input_parameters=source, potential=False, output_file="new.out")
    assert not plan.resolved_input.write
    source.write_text(source.read_text() + "\\n# changed after planning\\n")
    try:
        if execute:
            runner = plan.create_run(profile=calculator.profile)
            with pytest.raises(ValueError, match="Input parameters changed after planning"):
                runner.submit()
        else:
            with pytest.raises(ValueError, match="Input parameters changed after planning"):
                plan.prepare()
        assert not (tmp_path / "new.out").exists()
    finally:
        plan.close()


def test_empty_sphere_generation_belongs_to_materialization(tmp_path, monkeypatch):
    from ase2sprkkr.sprkkr import _preparation

    calls = []
    monkeypatch.setattr(_preparation, "add_empty_spheres",
                        lambda atoms, **kwargs: calls.append(atoms))
    plan = _calculator(tmp_path).plan(input_file="case.inp", potential_file="case.pot",
                                     empty_spheres=True)
    try:
        assert calls == []
        plan.prepare()
        assert calls == [plan.resolved_potential.atoms]
    finally:
        plan.close()
