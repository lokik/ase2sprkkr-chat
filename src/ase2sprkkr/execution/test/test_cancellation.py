from __future__ import annotations

import asyncio
from threading import Event, Thread

import pytest

from ase2sprkkr.execution.base import ExecutionResources, RunState
from ase2sprkkr.execution.postprocessing import PostprocessingRunner


class _Resource:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _BackendRunner:
    """Backend double intentionally lacking pre-submit cancellation support."""

    def __init__(self):
        self.state = RunState.NEW
        self.backend_id = None
        self.result = object()
        self.submitted = 0
        self.cancelled = 0
        self.submit_entered = Event()
        self.release_submit = Event()
        self.completed = Event()

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

    def add_output_listener(self, _callback):
        pass

    def add_output_error_listener(self, _callback):
        pass


def _runner():
    backend = _BackendRunner()
    backend.release_submit.set()
    resource = _Resource()
    return PostprocessingRunner(backend, lambda outcome: outcome, resources=ExecutionResources(resource)), backend, resource


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
    runner = PostprocessingRunner(backend, lambda outcome: outcome, resources=ExecutionResources(resource))
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
    runner = PostprocessingRunner(backend, lambda outcome: outcome, resources=resources)

    assert runner.cancel() is None
    assert runner.cancel() is None
    assert backend.cancelled == 0
    assert backend.state == final_state
