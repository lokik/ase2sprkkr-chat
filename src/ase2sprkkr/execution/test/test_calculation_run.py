"""Callback-based calculation completion never adds a second monitor."""
from threading import Event, Thread, get_ident
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution.base import ExecutionOutcome, FailurePhase, RunState
from ase2sprkkr.execution.postprocessing import CalculationRun


class Backend:
    def __init__(self, state=RunState.FINISHED):
        self.callback = None
        self.state = RunState.NEW
        self.terminal = state
        self.result = ExecutionOutcome(return_code=0)
        self.exception = None
        self.failure_phase = None
        self.failure_message = None
        self.backend_id = "job-1"
        self._finished = Event()
        self.release = Event()
        self.thread = None
        self.monitor_thread_id = None

    def submit(self):
        self.state = RunState.RUNNING
        self.thread = Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def monitor(self):
        self.monitor_thread_id = get_ident()
        assert self.release.wait(5)
        self.state = self.terminal
        if self.state == RunState.FAILED:
            self.failure_phase = FailurePhase.EXECUTION
        try:
            if self.callback:
                self.callback(self.result)
        finally:
            self._finished.set()

    def wait(self):
        assert get_ident() != self.monitor_thread_id
        assert self._finished.wait(5)
        return self.result

    def cancel(self):
        self.terminal = RunState.CANCELLED
        self.release.set()

    def add_output_listener(self, callback):
        pass

    def add_output_error_listener(self, callback):
        pass


def test_one_monitor_runs_parser_and_persistence_before_waking_waiters():
    backend = Backend()
    entered, proceed = Event(), Event()
    operations = []

    def parse(value):
        operations.append(("parse", get_ident()))
        entered.set()
        assert proceed.wait(5)
        return value

    def persist(value):
        operations.append(("persist", get_ident()))

    run = CalculationRun(backend, parse, callback=persist)
    run.submit()
    backend.release.set()
    try:
        assert entered.wait(5)
        assert run.state is RunState.POSTPROCESSING
        assert not run._done.is_set()
    finally:
        proceed.set()

    assert run.wait() is backend.result
    backend.thread.join(5)
    assert len(operations) == 2
    assert operations == [
        ("parse", backend.monitor_thread_id),
        ("persist", backend.monitor_thread_id),
    ]
    assert run.state == RunState.FINISHED
    assert not hasattr(run, "_thread")


@pytest.mark.parametrize("state", [
    RunState.UNKNOWN, RunState.FAILED, RunState.CANCELLED,
])
def test_non_successful_observation_skips_postprocessing(state):
    backend = Backend(state)
    parsed, callbacks = [], []
    run = CalculationRun(
        backend, lambda value: parsed.append(value),
        callback=lambda value: callbacks.append(value),
    )
    run.submit()
    backend.release.set()
    assert run.wait() is backend.result
    assert parsed == []
    assert callbacks == [backend.result]
    assert run.state is state


def test_parser_exception_is_postprocessing_failure():
    backend = Backend()
    error = ValueError("parser failed")
    reported = []

    def parse(_value):
        raise error

    run = CalculationRun(
        backend, parse, callback=lambda value: reported.append(run.state),
    )
    run.submit()
    backend.release.set()
    with pytest.raises(ValueError, match="parser failed") as caught:
        run.wait()
    assert caught.value is error
    assert reported == [RunState.FAILED]
    assert run.failure_phase is FailurePhase.POSTPROCESSING


def test_external_wait_only_backend_and_late_callback_finalize_once():
    class PlainBackend:
        def __init__(self):
            self.callback = None
            self.state = RunState.NEW
            self.result = ExecutionOutcome(return_code=0)
            self.exception = None

        def submit(self):
            self.state = RunState.FINISHED

        def wait(self):
            return self.result

    backend = PlainBackend()
    parsed, notifications = [], []
    run = CalculationRun(
        backend, lambda value: parsed.append(value) or value,
        callback=lambda value: notifications.append(value),
    )
    run.submit()
    assert run.wait() is backend.result
    backend.callback(backend.result)
    assert len(parsed) == len(notifications) == 1


def test_callback_failure_does_not_change_finished_state():
    backend = Backend()
    run = CalculationRun(
        backend, lambda value: value,
        callback=lambda value: (_ for _ in ()).throw(ValueError("user callback")),
    )
    run.submit()
    backend.release.set()
    with pytest.raises(ValueError, match="user callback"):
        run.wait()
    assert run.state is RunState.FINISHED
