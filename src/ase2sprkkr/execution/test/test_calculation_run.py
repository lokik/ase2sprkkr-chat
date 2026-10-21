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

def test_persistence_failure_is_distinct_from_calculation_and_can_be_retried():
    from ase2sprkkr.execution.run_hooks import PersistenceError

    backend = Backend()
    attempts, observed, processed = [], [], []

    class Hooks:
        def before_submit(self, run):
            pass

        def prepared(self, run):
            pass

        def submitted(self, run):
            pass

        def submission_failed(self, run, error, *, uncertain):
            pytest.fail(f"Unexpected submission failure: {error}")

        def completed(self, run):
            attempts.append(run.state)
            if len(attempts) == 1:
                raise OSError("store is unavailable")

    run = CalculationRun(
        backend, lambda result: processed.append(result) or result,
        callback=lambda result: observed.append(result), hooks=Hooks(),
    )
    run.submit()
    backend.release.set()

    with pytest.raises(PersistenceError, match="persist calculation"):
        run.wait()
    assert run.state is RunState.FINISHED
    assert len(processed) == 1
    assert observed == []
    assert isinstance(run.persistence_error.__cause__, OSError)

    run.retry_persistence()
    assert run.persistence_error is None
    assert run.wait() is backend.result
    assert len(attempts) == 2
    assert len(processed) == 1
    assert observed == [backend.result]
    run.retry_persistence()
    assert len(attempts) == 2


def test_persistent_hook_transaction_order_and_completion():
    backend = Backend()
    stages = []

    def preflight():
        stages.append("preflight")
    backend.prepare_submission = preflight
    original_submit = backend.submit

    def submit():
        stages.append("backend-submit")
        original_submit()
    backend.submit = submit

    class Hooks:
        def before_submit(self, run):
            stages.append("reserved")

        def prepared(self, run):
            stages.append("prepared")

        def submitted(self, run):
            stages.append("submitted")

        def submission_failed(self, run, error, *, uncertain):
            pytest.fail(f"Unexpected submission failure: {error!r}")

        def completed(self, run):
            stages.append("persisted")

    run = CalculationRun(
        backend, lambda result: stages.append("processed") or result,
        hooks=Hooks(), callback=lambda result: stages.append("callback"),
    )
    run.submit()
    backend.release.set()
    run.wait()
    assert stages == [
        "reserved", "preflight", "prepared", "backend-submit",
        "submitted", "processed", "persisted", "callback",
    ]


def test_persistent_pre_submit_hook_failure_does_not_launch_or_allow_retry():
    backend = Backend()
    phases = []

    class Hooks:
        def before_submit(self, run):
            phases.append("reservation")
            raise OSError("reservation failed")

        def submission_failed(self, run, error, *, uncertain):
            assert isinstance(error, OSError)
            assert uncertain is False
            phases.append("failed")

    run = CalculationRun(backend, lambda result: result, hooks=Hooks())
    with pytest.raises(OSError, match="reservation failed"):
        run.submit()
    assert phases == ["reservation", "failed"]
    assert run.state is RunState.FAILED
    assert not run._submission_uncertain
    assert backend.thread is None
    with pytest.raises(RuntimeError, match="already"):
        run.submit()


def test_persistent_cancel_after_preflight_does_not_launch():
    import asyncio

    backend = Backend()
    phases = []

    class Hooks:
        def before_submit(self, run):
            phases.append("reserved")

        def prepared(self, run):
            phases.append("prepared")
            run.request_cancel()

        def submission_failed(self, run, error, *, uncertain):
            assert isinstance(error, asyncio.CancelledError)
            assert uncertain is False
            phases.append("cancelled")

    backend.prepare_submission = lambda: None
    run = CalculationRun(backend, lambda result: result, hooks=Hooks())
    with pytest.raises(asyncio.CancelledError):
        run.submit()
    assert phases == ["reserved", "prepared", "cancelled"]
    assert run.state is RunState.CANCELLED
    assert not run._submission_uncertain
    assert backend.thread is None


def test_persistent_prepared_hook_runs_without_backend_preflight():
    backend = Backend()
    events = []

    class Hooks:
        def before_submit(self, run):
            events.append("reserved")

        def prepared(self, run):
            events.append("prepared")

        def submitted(self, run):
            events.append("submitted")

        def submission_failed(self, run, error, *, uncertain):
            pytest.fail(f"Unexpected submission failure: {error}")

        def completed(self, run):
            events.append("completed")

    run = CalculationRun(backend, lambda outcome: outcome, hooks=Hooks())
    run.submit()
    backend.release.set()
    run.wait()
    assert events == ["reserved", "prepared", "submitted", "completed"]


def test_synchronous_completion_waits_for_durable_backend_identity():
    """Backend callbacks in submit() cannot commit completion before identity."""
    from types import SimpleNamespace

    class ImmediateBackend:
        callback = None
        state = RunState.NEW
        backend_id = None
        exception = None
        failure_phase = None
        failure_message = None

        def __init__(self):
            self.result = SimpleNamespace(return_code=0)

        def submit(self):
            self.backend_id = "job-1"
            self.state = RunState.FINISHED
            self.callback(self.result)

        def wait(self):
            return self.result

    order = []

    class Hooks:
        def before_submit(self, run):
            order.append("reserve")

        def prepared(self, run):
            order.append("prepared")

        def submitted(self, run):
            assert run.backend_id == "job-1"
            order.append("identity")

        def submission_failed(self, run, error, *, uncertain):
            pytest.fail(f"Unexpected submission error: {error}")

        def completed(self, run):
            assert "identity" in order
            order.append("complete")

    run = CalculationRun(
        ImmediateBackend(), lambda result: result,
        hooks=Hooks(), callback=lambda result: order.append("callback"),
    )
    run.submit()
    assert run.wait().return_code == 0
    assert order == ["reserve", "prepared", "identity", "complete", "callback"]


def test_fast_completion_does_not_publish_if_identity_persistence_fails():
    """A failed identity commit must not unlock terminal persistence or ACK."""
    from types import SimpleNamespace
    from ase2sprkkr.execution.run_hooks import PersistenceError

    class ImmediateBackend:
        callback = None
        state = RunState.NEW
        backend_id = None
        exception = None
        failure_phase = None
        failure_message = None

        def __init__(self):
            self.result = SimpleNamespace(return_code=0)

        def submit(self):
            self.backend_id = "job-1"
            self.state = RunState.FINISHED
            self.callback(self.result)

        def wait(self):
            return self.result

    events = []

    class Hooks:
        def before_submit(self, run):
            pass

        def prepared(self, run):
            pass

        def submitted(self, run):
            events.append("identity-attempted")
            raise OSError("store is unavailable")

        def submission_failed(self, run, error, *, uncertain):
            assert uncertain is True
            events.append("submission-failed")

        def completed(self, run):
            events.append("terminal-persisted")

    run = CalculationRun(
        ImmediateBackend(), lambda outcome: outcome,
        hooks=Hooks(), callback=lambda outcome: events.append("callback"),
    )
    with pytest.raises(OSError, match="store is unavailable"):
        run.submit()
    assert run.state is RunState.FINISHED
    assert isinstance(run.persistence_error, PersistenceError)
    with pytest.raises(PersistenceError):
        run.wait()
    assert events == ["identity-attempted", "submission-failed"]
