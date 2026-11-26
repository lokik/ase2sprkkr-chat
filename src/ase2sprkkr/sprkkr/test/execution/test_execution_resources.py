"""Prepared SPR-KKR resources belong to the generic Run lifecycle."""

import sys
from contextlib import ExitStack
from threading import Event
from types import SimpleNamespace

import pytest

from kkrunner import RunState
from kkrunner.backends.contracts import SubmissionNotStarted
from kkrunner.client import EndpointClient
from kkrunner.runs import Run

from ase2sprkkr import InputParameters
from ase2sprkkr.input_parameters.task_definition import TaskDefinition
from ase2sprkkr.sprkkr import calculation as calculation_module
from ase2sprkkr.sprkkr.calculation import PreparedCalculation
from ase2sprkkr.sprkkr.profile import SPRKKRProfile


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    parameters = InputParameters.create("DOS")
    for name in ("input_files", "output_files", "cache_files"):
        monkeypatch.setattr(InputParameters, name, lambda self, directory: ())
    monkeypatch.setattr(
        TaskDefinition, "output_reader",
        lambda self, **kwargs: SimpleNamespace(
            read_from_file=lambda *args, **options: SimpleNamespace(
                return_code=options.get("return_code"),
            ),
        ),
    )
    monkeypatch.setattr(calculation_module, "resolve_task_executable", lambda *_: sys.executable)
    input_path = tmp_path / "run.inp"
    input_path.write_text("print('ok')\n")
    with ExitStack() as stack:
        input_file = stack.enter_context(input_path.open("rb"))
        output_file = stack.enter_context((tmp_path / "run.out").open("w+b"))
        calculation = PreparedCalculation(
            parameters, input_file, None, output_file, str(tmp_path),
            0, stack.pop_all(), None,
        )
        try:
            yield calculation
        finally:
            calculation.close()


def _create_run(prepared, *, executor=None, callback=None):
    return prepared.create_run(
        result_context=None,
        profile=SPRKKRProfile(mpi=False),
        executor=executor or EndpointClient(),
        callback=callback,
    )


def test_files_are_closed_when_runner_creation_fails(prepared, monkeypatch):
    executor = EndpointClient()

    def fail_factory(*args, **kwargs):
        raise RuntimeError("factory failed")

    monkeypatch.setattr(executor, "create_run_parts", fail_factory)
    with pytest.raises(RuntimeError, match="factory failed"):
        _create_run(prepared, executor=executor)
    assert prepared.input_file.closed
    assert prepared.output_file.closed


def test_files_are_closed_when_submission_fails_before_start(prepared, monkeypatch):
    runner = _create_run(prepared)
    assert isinstance(runner, Run)
    assert not prepared.input_file.closed

    def fail_submission():
        raise SubmissionNotStarted("staging failed")

    monkeypatch.setattr(runner._job, "submit", fail_submission)
    with pytest.raises(SubmissionNotStarted, match="staging failed"):
        runner.submit()
    assert runner.state is RunState.FAILED
    assert prepared.input_file.closed
    assert prepared.output_file.closed


def test_completion_releases_files_without_wait(prepared):
    completed = Event()
    runner = _create_run(prepared, callback=lambda result: completed.set())
    runner.submit()
    assert completed.wait(10), "completion callback was not called"
    assert runner.state is RunState.FINISHED
    assert prepared.input_file.closed
    assert prepared.output_file.closed


def test_unstarted_runner_can_release_owned_files_explicitly(prepared):
    runner = _create_run(prepared)
    runner.close()
    runner.close()
    assert prepared.input_file.closed
    assert prepared.output_file.closed
