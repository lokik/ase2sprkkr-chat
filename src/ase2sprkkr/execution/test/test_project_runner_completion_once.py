"""PersistentRun transaction hooks: idempotent durable finalization."""

from threading import Event, Thread
from types import SimpleNamespace

from ..base import RunState
from ..manager import PersistentRun


class _Manager:
    def __init__(self):
        self.completions = 0
        self.transitions = 0

    def complete(self, record, **kwargs):
        self.completions += 1
        return SimpleNamespace(**{**vars(record), "state": kwargs["state"]})

    def transition(self, record, **kwargs):
        self.transitions += 1
        return record


class _Runner:
    state = RunState.RUNNING
    backend_id = "backend-id"
    result = SimpleNamespace(return_code=0)
    exception = None
    failure_phase = None
    failure_message = None
    return_code = 0
    files = None
    execution = None

    def wait(self):
        self.state = RunState.FINISHED
        return self.result


class _Executor:
    def __init__(self):
        self.acknowledgements = 0

    def acknowledge(self, record):
        self.acknowledgements += 1


def test_reading_state_does_not_write_to_store():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = PersistentRun(manager, executor, record)
    assert run.state == RunState.RUNNING
    assert manager.transitions == 0


def test_finalization_is_once_even_if_wait_and_callback_both_request_it():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = PersistentRun(manager, executor, record)
    raw.state = RunState.FINISHED
    run.completed(raw)
    run.completed(raw)
    assert manager.completions == 1
    assert executor.acknowledgements == 1


def test_unknown_observation_is_written_once_but_not_finalized():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = PersistentRun(manager, executor, record)
    raw.state = RunState.UNKNOWN
    raw.failure_message = "scheduler disconnected"

    run.completed(raw)
    run.completed(raw)

    assert manager.transitions == 1
    assert manager.completions == 0
    assert executor.acknowledgements == 0


def test_unknown_observation_does_not_fire_completion_callback():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = PersistentRun(manager, executor, record)
    delivered = []
    raw.state = RunState.UNKNOWN
    run.completed(raw)
    delivered.append(raw.result)
    assert delivered == [raw.result]

    raw.state = RunState.FINISHED
    new_run = PersistentRun(manager, executor, record)
    new_run.completed(raw)
    delivered.append(raw.result)
    assert delivered == [raw.result, raw.result]

def test_concurrent_wait_and_completion_callback_persist_and_ack_once():
    """The callback must finish persisting before a simultaneous wait returns."""
    entered, release, wait_entered = Event(), Event(), Event()
    errors, returned = [], []

    class SlowManager(_Manager):
        def complete(self, record, **kwargs):
            entered.set()
            if not release.wait(5):
                raise AssertionError("Timed out waiting to release persistence")
            return super().complete(record, **kwargs)

    class WaitingRunner(_Runner):
        pass

    manager, raw, executor = SlowManager(), WaitingRunner(), _Executor()
    record = SimpleNamespace(
        id="run", state=RunState.RUNNING,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=SimpleNamespace(submitted=lambda native_id: native_id),
    )
    run = PersistentRun(manager, executor, record)
    raw.state = RunState.FINISHED

    def invoke(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    callback = Thread(target=lambda: invoke(lambda: run.completed(raw)))

    def await_persistence():
        wait_entered.set()
        run.completed(raw)
        returned.append(raw.result)

    waiter = Thread(target=lambda: invoke(await_persistence))
    try:
        callback.start()
        assert entered.wait(5), "Completion callback did not enter persistence"
        waiter.start()
        assert wait_entered.wait(5), "wait() did not reach the runner"
        assert manager.completions == 0
        assert executor.acknowledgements == 0
    finally:
        release.set()
        if callback.ident is not None:
            callback.join(5)
        if waiter.ident is not None:
            waiter.join(5)

    assert not callback.is_alive() and not waiter.is_alive()
    assert not errors
    assert returned == [raw.result]
    assert manager.completions == 1
    assert executor.acknowledgements == 1
    assert run.record.state is RunState.FINISHED


def test_terminal_record_deduplicates_after_recreating_persistence():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(
        id="run", state=RunState.RUNNING,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=SimpleNamespace(submitted=lambda value: value),
    )
    raw.state = RunState.FINISHED
    first = PersistentRun(manager, executor, record)
    first.completed(raw)
    assert manager.completions == 1

    restarted = PersistentRun(manager, executor, first.record)
    assert restarted.completed(raw) is RunState.FINISHED
    assert manager.completions == 1
    assert executor.acknowledgements == 1


def test_unknown_is_observation_not_terminal_state():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(
        id="run", state=RunState.RUNNING,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=SimpleNamespace(submitted=lambda value: value),
    )
    raw.state = RunState.UNKNOWN
    first = PersistentRun(manager, executor, record)
    first.completed(raw)
    first.completed(raw)
    assert manager.transitions == 1

    raw.state = RunState.FINISHED
    recovered = PersistentRun(manager, executor, first.record)
    recovered.completed(raw)
    assert manager.completions == 1
    assert executor.acknowledgements == 1
