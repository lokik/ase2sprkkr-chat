import asyncio
import io
from types import SimpleNamespace
from threading import Event, Thread

import pytest

from ase2sprkkr.execution.base import (
    ExecutionResources,
    ExecutionOutcome,
    RunState,
    SubmissionNotStarted,
)
from ase2sprkkr.execution import FileManifest
from ase2sprkkr.execution.run_files import RunFiles
from ase2sprkkr.execution.postprocessing import CalculationRun
from ase2sprkkr.execution.run_persistence import RunPersistence
from .submission_store_double import SubmissionStoreDouble


def _tracked(manager, executor, processed, record):
    manager.store = SubmissionStoreDouble()
    processed.persistence = RunPersistence(manager, executor, record)
    return processed
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
    definition_name = "SCF"
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
        self.backend_id = None
        self.failure_phase = None
        self.failure_message = None
        self.job_files = RunFiles(FileManifest())
        self.result = ExecutionOutcome(
            input_file=None if spec is None else spec.input_file,
            output_file=None if spec is None else spec.output_file,
            error_file=None if spec is None else spec.error_file,
            return_code=0,
        )

    @property
    def state(self):
        return self._state

    def prepare_submission(self):
        return None

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

    def resolve_spec(self, spec):
        return spec

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

    assert isinstance(runner, CalculationRun)
    assert not input_file.closed
    assert not output_file.closed

    with pytest.raises(SubmissionNotStarted, match="staging failed"):
        runner.submit()

    assert input_file.closed
    assert output_file.closed
    assert runner.state is RunState.FAILED
    with pytest.raises(RuntimeError, match="already failed"):
        runner.submit()


def test_completion_releases_files_without_wait():
    input_file = io.BytesIO(b"input")
    output_file = io.BytesIO()
    runner = _create_run(_Executor(), input_file, output_file)

    runner.submit()

    # Finalization is asynchronous, but owning resources must be released
    # even when the caller never invokes the public wait() method.
    assert runner._done.wait(3)
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
    runner = CalculationRun(underlying, lambda outcome: outcome, resources=resources)

    runner.cancel()

    assert runner.state == RunState.CANCELLED
    assert not resources.closed
    assert not input_file.closed


def test_cancel_unstarted_run_releases_resources():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    runner = CalculationRun(_Runner(), lambda outcome: outcome, resources=resources)

    runner.cancel()

    assert runner.state == RunState.CANCELLED
    assert resources.closed
    assert input_file.closed


def test_uncertain_submission_preserves_resources_and_prevents_resubmit():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    attempts = []

    class AmbiguousRunner(_Runner):
        def submit(self):
            attempts.append("submit")
            raise ConnectionError("backend reply lost")

    runner = CalculationRun(
        AmbiguousRunner(), lambda outcome: outcome, resources=resources,
    )
    with pytest.raises(ConnectionError, match="reply lost"):
        runner.submit()
    assert attempts == ["submit"]
    assert not resources.closed
    assert not input_file.closed

    with pytest.raises(RuntimeError, match="Submission outcome is uncertain"):
        runner.submit()
    with pytest.raises(RuntimeError, match="Submission outcome is uncertain"):
        runner.run()
    with pytest.raises(RuntimeError, match="Cannot close resources"):
        runner.close()
    assert attempts == ["submit"]


def test_cancel_before_submission_never_starts_backend():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    runner = CalculationRun(_Runner(), lambda outcome: outcome, resources=resources)
    runner.cancel()
    assert runner.state is RunState.CANCELLED
    assert resources.closed
    assert input_file.closed
    with pytest.raises(asyncio.CancelledError):
        runner.run()


def test_failed_project_reservation_releases_prepared_resources():

    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    postprocessed = CalculationRun(
        _Runner(), lambda outcome: outcome, resources=resources,
    )

    class FailedReservation:
        def reserve(self, record, *, spec=None, executor=None):
            raise ValueError("reservation rejected")

    project = _tracked(
        FailedReservation(), None, postprocessed,
        SimpleNamespace(state=RunState.NEW, updated_at="new", files=FileManifest()),
    )
    with pytest.raises(ValueError, match="reservation rejected"):
        project.submit()
    assert resources.closed
    assert input_file.closed


def test_preparation_error_releases_resources_before_submission():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    submitted = []

    class FailedPreflight(_Runner):
        def prepare_submission(self):
            raise OSError("staging failed")

        def submit(self):
            submitted.append(True)

    runner = CalculationRun(
        FailedPreflight(), lambda outcome: outcome, resources=resources,
    )
    with pytest.raises(SubmissionNotStarted, match="staging failed") as error:
        runner.submit()
    assert isinstance(error.value.__cause__, OSError)
    assert resources.closed
    assert not submitted
    assert runner.state is RunState.FAILED
    runner.close()
    assert runner.state is RunState.FAILED
    with pytest.raises(RuntimeError, match="already failed"):
        runner.start()


def test_explicit_close_prevents_later_run():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    submitted = []

    class ObservedRun(_Runner):
        def submit(self):
            submitted.append(True)
            super().submit()

    runner = CalculationRun(
        ObservedRun(), lambda outcome: outcome, resources=resources,
    )
    runner.close()
    assert input_file.closed
    with pytest.raises(RuntimeError, match="already cancelled"):
        runner.run()
    assert not submitted


def test_cancel_during_submit_waits_for_submission_then_requests_cancel():
    """A pre-submit cancellation must not miss a job accepted concurrently."""
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    submitting = Event()
    release_submission = Event()
    stopped = Event()
    attempts = []
    errors = []

    class RacingRunner(_Runner):
        def submit(self):
            submitting.set()
            if not release_submission.wait(5):
                raise TimeoutError("submission not released")
            self._state = RunState.RUNNING

        def wait(self):
            if not stopped.wait(5):
                raise TimeoutError("cancellation not delivered")
            return self.result

        def cancel(self):
            attempts.append("cancel")
            self._state = RunState.CANCELLED
            stopped.set()
            if self.callback is not None:
                self.callback(self.result)

    runner = CalculationRun(
        RacingRunner(), lambda outcome: outcome, resources=resources,
    )

    def submit():
        try:
            runner.submit()
        except BaseException as exc:
            errors.append(exc)

    submitting_thread = Thread(target=submit)
    submitting_thread.start()
    cancel_thread = None
    try:
        assert submitting.wait(3)
        cancel_thread = Thread(target=runner.cancel)
        cancel_thread.start()
    finally:
        release_submission.set()
        submitting_thread.join(5)
        if cancel_thread is not None:
            cancel_thread.join(5)

    assert not submitting_thread.is_alive()
    assert cancel_thread is not None and not cancel_thread.is_alive()
    assert not errors
    assert attempts == ["cancel"]
    assert runner._done.wait(3)
    assert runner.state == RunState.CANCELLED
    assert resources.closed
    assert input_file.closed


def test_project_cancel_during_reservation_does_not_close_in_use_inputs():

    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    entered = Event()
    release = Event()
    failures = []

    class SlowRejectedReservation:
        def reserve(self, record, *, spec=None, executor=None):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("reservation not released")
            raise ValueError("reservation rejected")

    raw = _Runner()
    processed = CalculationRun(
        raw, lambda outcome: outcome, resources=resources,
    )
    project = _tracked(
        SlowRejectedReservation(), None, processed,
        SimpleNamespace(state=RunState.NEW, updated_at="new", files=FileManifest()),
    )

    def reserve():
        try:
            project.submit()
        except ValueError:
            pass
        except BaseException as exc:
            failures.append(exc)

    submitting = Thread(target=reserve)
    submitting.start()
    cancelling = None
    try:
        assert entered.wait(3)
        cancelling = Thread(target=project.cancel)
        cancelling.start()
        assert not resources.closed
        assert not input_file.closed
    finally:
        release.set()
        submitting.join(5)
        if cancelling is not None:
            cancelling.join(5)

    assert not submitting.is_alive()
    assert cancelling is not None and not cancelling.is_alive()
    assert not failures
    assert resources.closed
    assert input_file.closed
    assert raw.state is RunState.NEW


def test_callback_persists_before_prepared_resources_are_closed():
    resources = ExecutionResources(io.BytesIO(b"input"))
    observed = []

    class CompletedRunner(_Runner):
        def submit(self):
            self._state = RunState.FINISHED

    runner = CalculationRun(
        CompletedRunner(), lambda outcome: outcome,
        callback=lambda result: observed.append(resources.closed),
        resources=resources,
    )
    runner.submit()
    runner.wait()
    assert observed == [False]
    assert resources.closed


def test_manifest_persistence_failure_does_not_submit_backend():

    resources = ExecutionResources(io.BytesIO(b"input"))
    submissions = []
    transitions = []
    completed = []

    class PreparedBackend(_Runner):
        def prepare_submission(self):
            return True

        def submit(self):
            submissions.append(True)
            super().submit()

    class FailingStore:
        def reserve(self, record, *, spec=None, executor=None):
            return SimpleNamespace(state=RunState.SUBMITTING, id="reserved",
                                   updated_at="reserved", files=FileManifest())

        def transition(self, record, **kwargs):
            transitions.append(kwargs)
            raise OSError("manifest write failed")

        def complete(self, record, **kwargs):
            completed.append(kwargs)
            return SimpleNamespace(state=kwargs["state"], id="reserved")

        def record_submission_failure(self, record, *, error, uncertain, **kwargs):
            assert not uncertain
            return self.complete(
                record, state=RunState.FAILED, failure_phase=kwargs["failure_phase"],
            )



    processed = CalculationRun(
        PreparedBackend(), lambda outcome: outcome, resources=resources,
    )
    project = _tracked(
        FailingStore(), None, processed,
        SimpleNamespace(state=RunState.NEW, id="reserved",
                        updated_at="new", files=FileManifest()),
    )

    with pytest.raises(OSError, match="manifest write failed"):
        project.submit()
    assert len(transitions) == 1
    assert not submissions
    assert resources.closed
    assert len(completed) == 1
    assert completed[0]["state"] is RunState.FAILED
    assert completed[0]["failure_phase"].value == "submission"


def test_cancel_during_preflight_does_not_close_inputs_until_preflight_ends():
    input_file = io.BytesIO(b"input")
    resources = ExecutionResources(input_file)
    entered, release = Event(), Event()
    failures = []

    class BlockingPreflight(_Runner):
        def prepare_submission(self):
            entered.set()
            assert not input_file.closed
            assert release.wait(5)
            assert not input_file.closed

    runner = CalculationRun(
        BlockingPreflight(), lambda outcome: outcome, resources=resources,
    )

    def prepare():
        try:
            runner.submit()
        except BaseException as exc:
            failures.append(exc)

    preparing = Thread(target=prepare)
    preparing.start()
    cancelling = None
    try:
        assert entered.wait(3)
        cancelling = Thread(target=runner.cancel)
        cancelling.start()
        assert not resources.closed
        assert not input_file.closed
    finally:
        release.set()
        preparing.join(5)
        if cancelling is not None:
            cancelling.join(5)
    assert not preparing.is_alive()
    assert cancelling is not None and not cancelling.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], asyncio.CancelledError)
    assert resources.closed
    assert runner.state is RunState.CANCELLED


def test_project_cancel_during_manifest_staging_never_submits_backend():

    resources = ExecutionResources(io.BytesIO(b"input"))
    entered, release = Event(), Event()
    submissions, completions, failures = [], [], []

    class Backend(_Runner):
        def prepare_submission(self):
            entered.set()
            assert release.wait(5)

        def submit(self):
            submissions.append(True)
            super().submit()

    class Store:
        def reserve(self, record, *, spec=None, executor=None):
            return SimpleNamespace(state=RunState.SUBMITTING, id="run",
                                   updated_at="reserved", files=FileManifest())

        def transition(self, record, **changes):
            return record

        def complete(self, record, **changes):
            completions.append(changes)
            return SimpleNamespace(state=changes["state"], id="run")

        def record_submission_failure(self, record, *, error, uncertain, **kwargs):
            assert not uncertain
            assert isinstance(error, asyncio.CancelledError)
            return self.complete(record, state=RunState.CANCELLED)



    processed = CalculationRun(Backend(), lambda outcome: outcome, resources=resources)
    project = _tracked(Store(), None, processed, SimpleNamespace(
        state=RunState.NEW, id="run", updated_at="new", files=FileManifest(),
    ))

    def submit():
        try:
            project.submit()
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            failures.append(exc)

    submit_thread = Thread(target=submit)
    submit_thread.start()
    cancel_thread = None
    try:
        assert entered.wait(3)
        cancel_thread = Thread(target=project.cancel)
        cancel_thread.start()
        assert project._cancel_requested.wait(3)
    finally:
        release.set()
        submit_thread.join(5)
        if cancel_thread is not None:
            cancel_thread.join(5)

    assert not submit_thread.is_alive()
    assert cancel_thread is not None and not cancel_thread.is_alive()
    assert not failures
    assert not submissions
    assert resources.closed
    assert completions and completions[0]["state"] is RunState.CANCELLED


def test_ambiguous_submit_without_resources_cannot_be_retried():
    attempts = []
    class LostReply(_Runner):
        def submit(self):
            attempts.append("submitted")
            raise ConnectionError("reply lost")

    runner = CalculationRun(LostReply(), lambda result: result)
    with pytest.raises(ConnectionError, match="reply lost"):
        runner.submit()
    with pytest.raises(RuntimeError, match="Submission outcome is uncertain"):
        runner.submit()
    with pytest.raises(RuntimeError, match="Submission outcome is uncertain"):
        runner.run()
    assert attempts == ["submitted"]




