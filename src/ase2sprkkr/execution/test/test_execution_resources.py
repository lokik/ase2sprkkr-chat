import io
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution.base import (
    ExecutionResources,
    ExecutionOutcome,
    ResourceOwnedRunner,
    RunState,
    SubmissionNotStarted,
)
from ase2sprkkr.sprkkr.calculation import PreparedCalculation
from ase2sprkkr.sprkkr.profile import SPRKKRProfile


class _Task:
    name = "SCF"
    executable = "sprkkr"
    mpi = False

    def output_reader(self, **kwargs):
        return SimpleNamespace(
            read_from_file=lambda *args, **options: SimpleNamespace(
                return_code=options.get("return_code"),
            ),
        )


class _Parameters:
    CONTROL = SimpleNamespace(
        POTFIL=SimpleNamespace(result=None),
        DATASET=SimpleNamespace(result=None),
    )

    def task_definition_required(self):
        return _Task()

    def input_files(self, directory):
        return ()

    def output_files(self, directory):
        return ()

    def cache_files(self, directory):
        return ()


class _Prepared:
    create_run = PreparedCalculation.create_run

    def __init__(self, input_file, output_file):
        self.input_parameters = _Parameters()
        self.input_file = input_file
        self.potential_file = None
        self.output_file = output_file
        self.directory = "."
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.input_file.close()
        self.output_file.close()


class _Runner:
    exception = None

    def __init__(self, callback=None, *, submit_error=None, spec=None):
        self.callback = callback
        self.submit_error = submit_error
        self._state = RunState.NEW
        self.result = ExecutionOutcome(
            input_file=None if spec is None else spec.input_file,
            output_file=None if spec is None else spec.output_file,
            error_file=None if spec is None else spec.error_file,
            return_code=0,
        )

    @property
    def state(self):
        return self._state

    def submit(self):
        if self.submit_error is not None:
            raise self.submit_error
        self._state = RunState.FINISHED
        if self.callback:
            self.callback(self.result)

    def wait(self):
        return self.result

    def cancel(self):
        self._state = RunState.CANCELLED

    def add_output_listener(self, callback):
        pass

    def add_output_error_listener(self, callback):
        pass


class _Executor:
    def __init__(self, *, factory_error=None, submit_error=None):
        self.factory_error = factory_error
        self.submit_error = submit_error

    def create_run(self, spec, callback=None):
        if self.factory_error is not None:
            raise self.factory_error
        return _Runner(callback, submit_error=self.submit_error, spec=spec)


def _create_run(executor, input_file, output_file):
    prepared = _Prepared(input_file, output_file)
    return prepared.create_run(
        result_context=None,
        profile=SPRKKRProfile(mpi=False),
        executor=executor,
    )


def test_files_are_closed_when_runner_creation_fails():
    input_file = io.BytesIO(b"input")
    output_file = io.BytesIO()

    with pytest.raises(RuntimeError, match="factory failed"):
        _create_run(
            _Executor(factory_error=RuntimeError("factory failed")),
            input_file,
            output_file,
        )

    assert input_file.closed
    assert output_file.closed


def test_files_are_closed_when_submission_fails_before_start():
    input_file = io.BytesIO(b"input")
    output_file = io.BytesIO()
    runner = _create_run(
        _Executor(submit_error=SubmissionNotStarted("staging failed")),
        input_file,
        output_file,
    )

    assert isinstance(runner, ResourceOwnedRunner)
    assert not input_file.closed
    assert not output_file.closed

    with pytest.raises(SubmissionNotStarted, match="staging failed"):
        runner.submit()

    assert input_file.closed
    assert output_file.closed


def test_completion_releases_files_without_wait():
    input_file = io.BytesIO(b"input")
    output_file = io.BytesIO()
    runner = _create_run(_Executor(), input_file, output_file)

    runner.submit()

    # Finalization is asynchronous, but owning resources must be released
    # even when the caller never invokes the public wait() method.
    assert runner._runner._done.wait(3)
    assert runner.state == RunState.FINISHED
    assert input_file.closed
    assert output_file.closed


def test_unstarted_runner_can_release_owned_files_explicitly():
    input_file = io.BytesIO(b"input")
    output_file = io.BytesIO()
    runner = _create_run(_Executor(), input_file, output_file)

    runner.close()
    runner.close()

    assert input_file.closed
    assert output_file.closed


def test_cancel_started_run_does_not_close_resources_before_monitor_finishes():
    input_file = io.BytesIO(b"input")
    underlying = _Runner()
    underlying._state = RunState.RUNNING
    resources = ExecutionResources(input_file)
    runner = ResourceOwnedRunner(underlying, resources)

    runner.cancel()

    assert runner.state == RunState.CANCELLED
    assert not resources.closed
    assert not input_file.closed


def test_cancel_unstarted_run_releases_resources():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    runner = ResourceOwnedRunner(_Runner(), resources)

    runner.cancel()

    assert runner.state == RunState.CANCELLED
    assert resources.closed
    assert input_file.closed


def test_uncertain_submit_keeps_resources_and_prevents_resubmit():
    """A lost submit reply is not proof that the backend rejected the job."""
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    attempts = []

    class AmbiguousRunner(_Runner):
        def submit(self):
            attempts.append("submitted")
            raise ConnectionError("submit reply lost")

        def cancel(self):
            attempts.append("cancel requested")

    runner = ResourceOwnedRunner(AmbiguousRunner(), resources)
    with pytest.raises(ConnectionError, match="reply lost"):
        runner.submit()
    assert not resources.closed
    assert not input_file.closed

    with pytest.raises(RuntimeError, match="submission uncertain"):
        runner.close()
    with pytest.raises(RuntimeError, match="Submission outcome is uncertain"):
        runner.run()
    assert attempts == ["submitted"]

    runner.cancel()
    assert attempts == ["submitted", "cancel requested"]
    assert not resources.closed


def test_certain_submit_failure_still_releases_resources():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    runner = ResourceOwnedRunner(
        _Runner(submit_error=SubmissionNotStarted("not launched")), resources,
    )
    with pytest.raises(SubmissionNotStarted, match="not launched"):
        runner.submit()
    assert resources.closed
    assert input_file.closed
