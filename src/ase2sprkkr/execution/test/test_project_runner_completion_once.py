"""ProjectRunner state observation and exactly-once finalization."""

from threading import Event, Thread
from types import SimpleNamespace

from ..base import RunState
from ..manager import ProjectRunner


class _Manager:
    def __init__(self):
        self.completions = 0
        self.transitions = 0

    def complete(self, record, **kwargs):
        self.completions += 1
        return SimpleNamespace(id=record.id, state=kwargs["state"])

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
    run = ProjectRunner(manager, executor, raw, record)
    assert run.state == RunState.RUNNING
    assert manager.transitions == 0


def test_finalization_is_once_even_if_wait_and_callback_both_request_it():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = ProjectRunner(manager, executor, raw, record)
    raw.state = RunState.FINISHED
    run._sync_completion(raw.result)
    run._sync_completion(raw.result)
    run.wait()
    assert manager.completions == 1
    assert executor.acknowledgements == 1


def test_unknown_observation_is_written_once_but_not_finalized():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = ProjectRunner(manager, executor, raw, record)
    raw.state = RunState.UNKNOWN
    raw.failure_message = "scheduler disconnected"

    run._sync_completion(raw.result)
    run._sync_completion(raw.result)
    run.wait = lambda: run._sync_completion(raw.result)
    run.wait()

    assert manager.transitions == 1
    assert manager.completions == 0
    assert executor.acknowledgements == 0


def test_unknown_observation_does_not_fire_completion_callback():
    from ..manager import ExecutionManager

    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=SimpleNamespace(submitted=lambda x: x))
    run = ProjectRunner(manager, executor, raw, record)
    delivered = []
    raw.state = RunState.UNKNOWN
    ExecutionManager._complete(run, raw.result, delivered.append)
    assert delivered == [raw.result]

    raw.state = RunState.FINISHED
    new_run = ProjectRunner(manager, executor, raw, record)
    ExecutionManager._complete(new_run, raw.result, delivered.append)
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
        def wait(self):
            wait_entered.set()
            return super().wait()

    manager, raw, executor = SlowManager(), WaitingRunner(), _Executor()
    record = SimpleNamespace(
        id="run", state=RunState.RUNNING,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=SimpleNamespace(submitted=lambda native_id: native_id),
    )
    run = ProjectRunner(manager, executor, raw, record)
    raw.state = RunState.FINISHED

    def invoke(action):
        try:
            action()
        except BaseException as exc:
            errors.append(exc)

    callback = Thread(target=lambda: invoke(lambda: run._sync_completion(raw.result)))
    waiter = Thread(target=lambda: invoke(lambda: returned.append(run.wait())))
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
