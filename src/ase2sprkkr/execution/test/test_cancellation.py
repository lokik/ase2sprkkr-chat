from __future__ import annotations

import asyncio
from threading import Event, Thread

import pytest

from ase2sprkkr.execution import FileManifest
from ase2sprkkr.execution.run_files import RunFiles
from ase2sprkkr.execution.base import ExecutionResources, RunState
from ase2sprkkr.execution.postprocessing import CalculationRun
from ase2sprkkr.execution.run_persistence import RunPersistence
from .submission_store_double import SubmissionStoreDouble


def _tracked(manager, executor, processed, record):
    manager.store = SubmissionStoreDouble()
    processed.persistence = RunPersistence(manager, executor, record)
    return processed


class _Resource:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _BackendRunner:
    """Backend double intentionally lacking pre-submit cancellation support."""

    def __init__(self):
        self.callback = None
        self.state = RunState.NEW
        self.backend_id = None
        self.failure_phase = None
        self.failure_message = None
        self.job_files = RunFiles(FileManifest())
        self.exception = None
        self.result = object()
        self.submitted = 0
        self.cancelled = 0
        self.submit_entered = Event()
        self.release_submit = Event()
        self.completed = Event()

    @property
    def return_code(self):
        # Cancellation before the backend's terminal observation has no exit code.
        return None

    def prepare_submission(self):
        return None

    def submit(self):
        self.submit_entered.set()
        if not self.release_submit.wait(5):
            raise TimeoutError("test did not release submit")
        self.submitted += 1
        self.state = RunState.QUEUED

    def wait(self):
        if not self.completed.wait(5):
            raise TimeoutError("test did not complete backend")
        return self.result

    def cancel(self):
        if self.state == RunState.NEW:
            raise RuntimeError("backend cannot cancel before submission")
        self.cancelled += 1
        self.state = RunState.CANCELLED
        self.completed.set()
        # Real monitored backends publish their terminal observation before
        # CalculationRun's waiters are released. This double must do so too.
        if self.callback is not None:
            self.callback(self.result)

    def add_output_listener(self, _callback):
        pass

    def add_output_error_listener(self, _callback):
        pass


def _runner():
    backend = _BackendRunner()
    backend.release_submit.set()
    resource = _Resource()
    return CalculationRun(backend, lambda outcome: outcome, resources=ExecutionResources(resource)), backend, resource


def test_public_runner_latches_cancel_before_submit():
    runner, backend, resource = _runner()

    runner.cancel()

    assert runner.state == RunState.CANCELLED
    assert resource.closed
    assert backend.submitted == 0
    assert backend.cancelled == 0
    with pytest.raises(asyncio.CancelledError):
        runner.submit()
    assert backend.submitted == 0


def test_cancel_during_submit_is_dispatched_after_submission():
    backend = _BackendRunner()
    resource = _Resource()
    runner = CalculationRun(backend, lambda outcome: outcome, resources=ExecutionResources(resource))
    errors = []

    def submit():
        try:
            runner.submit()
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    thread = Thread(target=submit)
    thread.start()
    assert backend.submit_entered.wait(5)

    cancel_thread = Thread(target=runner.cancel)
    cancel_thread.start()
    assert runner._cancel_requested.wait(5)
    try:
        # Cancellation is accepted *before* the gated backend submit returns.
        cancel_thread.join(1)
        assert not cancel_thread.is_alive(), "cancel waited for backend submission"
        assert backend.submitted == 0
        assert backend.cancelled == 0
    finally:
        backend.release_submit.set()

    thread.join(5)
    cancel_thread.join(5)
    assert not thread.is_alive()
    assert not cancel_thread.is_alive()
    assert not errors
    assert backend.submitted == 1
    assert backend.cancelled == 1
    assert runner._done.wait(5)
    assert runner.state == RunState.CANCELLED

    # Repeated cancellation is idempotent at the public runner boundary.
    runner.cancel()
    assert backend.cancelled == 1


@pytest.mark.parametrize("owned", (False, True))
@pytest.mark.parametrize("final_state", (RunState.FINISHED, RunState.FAILED, RunState.CANCELLED))
def test_late_cancel_never_reaches_terminal_backend(owned, final_state):
    backend = _BackendRunner()
    backend.state = final_state
    resource = _Resource()
    resources = ExecutionResources(resource) if owned else None
    runner = CalculationRun(backend, lambda outcome: outcome, resources=resources)

    assert runner.cancel() is None
    assert runner.cancel() is None
    assert backend.cancelled == 0
    assert backend.state == final_state


def test_project_cancel_returns_while_backend_submission_is_blocked():
    """Project persistence must not hold cancellation behind raw submission."""
    from types import SimpleNamespace


    class Execution:
        endpoint = "test"

        def __init__(self):
            self.backend_id = None

        def submitted(self, backend_id):
            assert backend_id == "job-1"
            self.backend_id = backend_id
            return self

    class Backend(_BackendRunner):
        execution = Execution()
        exception = None

        def __init__(self):
            super().__init__()
            self.backend_id = "job-1"

    class Store:
        def __init__(self):
            self.transitions = []

        def reserve(self, record, *, spec=None, executor=None):
            return SimpleNamespace(state=RunState.SUBMITTING, id=record.id,
                                   updated_at="reserved", execution=record.execution,
                                   files=record.files)

        def transition(self, record, **changes):
            self.transitions.append(changes)
            return SimpleNamespace(state=changes.get("state", record.state),
                                   id=record.id, updated_at="transition",
                                   execution=record.execution, files=record.files)

        def complete(self, record, **changes):
            return self.transition(record, **changes)

        def register_executor(self, executor, endpoint):
            pass

        def record_submitted(self, record, *, backend_id, state, **kwargs):
            return self.transition(
                record, state=state,
                execution=record.execution.submitted(backend_id),
            )

        def record_observation(self, record, *, state, **kwargs):
            return self.complete(record, state=state)

        @staticmethod
        def acknowledge_terminal(executor, record):
            executor.acknowledge(record)
            return True

        def _acknowledge_submission(self, executor, record):
            pass

    backend = Backend()
    resource = _Resource()
    processed = CalculationRun(
        backend, lambda outcome: outcome, resources=ExecutionResources(resource),
    )
    store = Store()
    project = _tracked(
        store, SimpleNamespace(acknowledge=lambda record: None), processed,
        SimpleNamespace(state=RunState.NEW, id="test-run", updated_at="new",
                        execution=Execution(), files=FileManifest()),
    )
    errors = []

    def submit():
        try:
            project.submit()
        except BaseException as exc:
            errors.append(exc)

    submitting = Thread(target=submit)
    submitting.start()
    assert backend.submit_entered.wait(5)
    cancelling = Thread(target=project.cancel)
    cancelling.start()
    try:
        cancelling.join(1)
        assert not cancelling.is_alive(), "project cancel waited for submit"
        assert backend.submitted == 0
        assert backend.cancelled == 0
        assert not resource.closed
    finally:
        backend.release_submit.set()
        submitting.join(5)
        cancelling.join(5)

    assert not submitting.is_alive()
    assert not cancelling.is_alive()
    assert not errors
    assert backend.submitted == 1
    assert backend.cancelled == 1
    project.wait()
    assert resource.closed
    assert project.record.state == RunState.CANCELLED
    assert any(change.get("state") == RunState.CANCELLED for change in store.transitions)
    project.cancel()
    assert backend.cancelled == 1


@pytest.mark.parametrize("phase", ("reservation", "preparation"))
def test_project_cancellation_during_prelaunch_is_durable_and_nonblocking(phase):
    """One runtime cancel latch settles a project even before backend submit."""
    from types import SimpleNamespace

    entered, release = Event(), Event()

    class Backend(_BackendRunner):
        def prepare_submission(self):
            if phase == "preparation":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("preparation was not released")

    class Manager:
        def __init__(self):
            self.completions = []

        @staticmethod
        def changed(record, **changes):
            return SimpleNamespace(**{
                **vars(record), **changes, "updated_at": "later",
            })

        def reserve(self, record, *, spec=None, executor=None):
            if phase == "reservation":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("reservation was not released")
            return self.changed(record, state=RunState.SUBMITTING)

        def transition(self, record, **changes):
            return self.changed(record, **changes)

        def complete(self, record, **changes):
            self.completions.append(changes)
            return self.changed(record, **changes)

        def record_submission_failure(self, record, *, error, uncertain, **kwargs):
            assert not uncertain
            assert isinstance(error, asyncio.CancelledError)
            return self.complete(record, state=RunState.CANCELLED)



    backend = Backend()
    resource = _Resource()
    processed = CalculationRun(
        backend, lambda result: result, resources=ExecutionResources(resource),
    )
    manager = Manager()
    project = _tracked(
        manager, object(), processed,
        SimpleNamespace(
            state=RunState.NEW, id="run", updated_at="new",
            execution=SimpleNamespace(), files=FileManifest(),
        ),
    )
    errors = []

    def submit():
        try:
            project.submit()
        except BaseException as exc:
            errors.append(exc)

    submitting = Thread(target=submit)
    submitting.start()
    try:
        assert entered.wait(5)
        cancelling = Thread(target=project.cancel)
        cancelling.start()
        cancelling.join(1)
        assert not cancelling.is_alive(), "cancel blocked on prelaunch I/O"
        assert backend.submitted == 0
    finally:
        release.set()
        submitting.join(5)

    assert not submitting.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], asyncio.CancelledError)
    assert project.record.state == RunState.CANCELLED
    assert processed.state == RunState.CANCELLED
    assert len(manager.completions) == 1
    assert resource.closed
    assert backend.submitted == 0
    assert backend.backend_id is None
    assert backend.cancelled == 0
