"""One monitor owns execution completion, result parsing and publication."""
import threading
from threading import Event, Thread

import pytest

from ase2sprkkr.execution.base import (
    ExecutionOutcome, FailurePhase, RunState,
)
from ase2sprkkr.execution.postprocessing import CalculationRun


class _CallbackBackend:
    """Model a backend monitor that owns exactly one monitor thread."""

    def __init__(self, state=RunState.FINISHED, *, error=None):
        self.callback = None
        self.state = RunState.NEW
        self.terminal_state = state
        self.result = ExecutionOutcome(return_code=0)
        self.exception = error
        self.failure_phase = None
        self.failure_message = None
        self.backend_id = "existing-backend"
        self.submitted = Event()
        self.release = Event()
        self._done = Event()
        self.thread = None
        self.monitor_ident = None
        self.wait_called_from_monitor = False

    def submit(self):
        self.start()

    def start(self):
        self.state = RunState.RUNNING
        self.thread = Thread(target=self._monitor, daemon=True)
        self.thread.start()
        self.submitted.set()

    def _monitor(self):
        self.monitor_ident = threading.get_ident()
        assert self.release.wait(3)
        self.state = self.terminal_state
        if self.state == RunState.FAILED:
            self.failure_phase = FailurePhase.EXECUTION
        if self.callback is not None:
            self.callback(self.result)
        self._done.set()

    def wait(self):
        if threading.get_ident() == self.monitor_ident:
            self.wait_called_from_monitor = True
            raise AssertionError("Completion called wait() on its own monitor")
        if not self._done.wait(3):
            raise TimeoutError("Monitor failed to finish")
        return self.result

    def cancel(self):
        self.state = RunState.CANCELLED
        self.release.set()

    def add_output_listener(self, callback):
        pass

    def add_output_error_listener(self, callback):
        pass


def test_one_monitor_handles_processing_and_persistent_callback_before_wait():
    backend = _CallbackBackend()
    processing_entered, continue_processing = Event(), Event()
    published = Event()
    actions = []

    def process(outcome):
        actions.append(("process", threading.get_ident()))
        processing_entered.set()
        assert continue_processing.wait(3)
        return outcome

    def persist(result):
        actions.append(("persist", threading.get_ident()))
        published.set()

    run = CalculationRun(backend, process, callback=persist)
    run.submit()
    backend.release.set()
    try:
        assert processing_entered.wait(3)
        assert run.state == RunState.POSTPROCESSING
        assert not published.is_set()
    finally:
        continue_processing.set()

    assert run.wait() is backend.result
    assert backend.thread is not None
    backend.thread.join(3)
    assert not backend.thread.is_alive()
    assert len(actions) == 2
    assert actions == [
        ("process", backend.monitor_ident),
        ("persist", backend.monitor_ident),
    ]
    assert run.state == RunState.FINISHED
    assert not backend.wait_called_from_monitor
    assert not hasattr(run, "_thread")


@pytest.mark.parametrize("state", [
    RunState.UNKNOWN, RunState.FAILED, RunState.CANCELLED,
])
def test_non_successful_observation_does_not_parse(state):
    backend = _CallbackBackend(state)
    parsed, delivered = [], []
    run = CalculationRun(
        backend, lambda outcome: parsed.append(outcome),
        callback=lambda result: delivered.append(result),
    )
    run.submit()
    backend.release.set()
    assert run.wait() is backend.result
    assert parsed == []
    assert delivered == [backend.result]
    assert run.state == state
    assert run.failure_phase == (
        FailurePhase.EXECUTION if state == RunState.FAILED else None
    )


def test_postprocessing_error_is_failure_and_callback_sees_it():
    backend = _CallbackBackend()
    reason = ValueError("reader failed")
    observed = []

    def fail(outcome):
        raise reason

    run = CalculationRun(
        backend, fail, callback=lambda result: observed.append(run.state),
    )
    run.submit()
    backend.release.set()
    with pytest.raises(ValueError, match="reader failed") as exc:
        run.wait()
    assert exc.value is reason
    assert observed == [RunState.FAILED]
    assert run.failure_phase == FailurePhase.POSTPROCESSING
    assert run.failure_message == "reader failed"


def test_callback_failure_cannot_turn_successful_execution_into_failure():
    backend = _CallbackBackend()

    def fail(_result):
        raise RuntimeError("user notification failed")

    run = CalculationRun(backend, lambda outcome: outcome, callback=fail)
    run.submit()
    backend.release.set()
    with pytest.raises(RuntimeError, match="user notification failed"):
        run.wait()
    assert run.state == RunState.FINISHED
    assert run.failure_phase is None


def test_fallback_wait_and_late_backend_callback_finalize_once():
    """A small external runner may only publish the result through wait()."""
    class Runner:
        def __init__(self):
            self.callback = None
            self.result = ExecutionOutcome(return_code=0)
            self.state = RunState.NEW
            self.exception = None

        def submit(self):
            self.state = RunState.FINISHED

        def wait(self):
            return self.result

    backend = Runner()
    process_calls, callback_calls = [], []
    run = CalculationRun(
        backend, lambda value: process_calls.append(value) or value,
        callback=lambda value: callback_calls.append(value),
    )
    run.submit()
    assert run.wait() is backend.result
    backend.callback(backend.result)
    assert process_calls == [backend.result]
    assert callback_calls == [backend.result]
