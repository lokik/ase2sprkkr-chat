"""RunPersistence transaction hooks: idempotent durable finalization."""

from threading import Event, Thread
from types import SimpleNamespace

from ..base import RunState
from ..job_files import JobFiles
from ..run_model import FileManifest
from ..endpoint import ExecutorEndpoint
from ..run_model import Execution, WorkingDirectory
from ..manager import RunPersistence


def _execution():
    return Execution(ExecutorEndpoint(), WorkingDirectory("/virtual/run"), "backend-id")


class _Manager:
    def __init__(self):
        self.completions = 0
        self.transitions = 0

    def complete(self, record, **kwargs):
        self.completions += 1
        return SimpleNamespace(**{**vars(record), "state": kwargs["state"]})

    def transition(self, record, **kwargs):
        self.transitions += 1
        # Model the committed state. Returning the untouched record would
        # make repeated UNKNOWN observations look like new transitions.
        changes = {
            name: value for name, value in kwargs.items()
            if name in ("state", "message") and value is not None
        }
        return SimpleNamespace(**{**vars(record), **changes})


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

    def __init__(self):
        self._job_files = JobFiles(FileManifest())

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
                             execution=_execution())
    run = RunPersistence(manager, executor, record)
    assert run.record.state == RunState.RUNNING
    assert manager.transitions == 0


def test_finalization_is_once_even_if_wait_and_callback_both_request_it():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=_execution())
    run = RunPersistence(manager, executor, record)
    raw.state = RunState.FINISHED
    run.completed(raw)
    run.completed(raw)
    assert manager.completions == 1
    assert executor.acknowledgements == 1


def test_unknown_observation_is_written_once_but_not_finalized():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(id="run", state=RunState.RUNNING,
                             updated_at="2026-10-20T10:00:00+00:00",
                             execution=_execution())
    run = RunPersistence(manager, executor, record)
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
                             execution=_execution())
    run = RunPersistence(manager, executor, record)
    delivered = []
    raw.state = RunState.UNKNOWN
    run.completed(raw)
    delivered.append(raw.result)
    assert delivered == [raw.result]

    raw.state = RunState.FINISHED
    new_run = RunPersistence(manager, executor, record)
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
        execution=_execution(),
    )
    run = RunPersistence(manager, executor, record)
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
        execution=_execution(),
    )
    raw.state = RunState.FINISHED
    first = RunPersistence(manager, executor, record)
    first.completed(raw)
    assert manager.completions == 1

    restarted = RunPersistence(manager, executor, first.record)
    assert restarted.completed(raw) is RunState.FINISHED
    assert manager.completions == 1
    # Terminal persistence is deduplicated; ACK deliberately retries after
    # reconstructing a client, since its previous delivery is not durable.
    assert executor.acknowledgements == 2


def test_unknown_is_observation_not_terminal_state():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(
        id="run", state=RunState.RUNNING,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=_execution(),
    )
    raw.state = RunState.UNKNOWN
    first = RunPersistence(manager, executor, record)
    first.completed(raw)
    first.completed(raw)
    assert manager.transitions == 1

    raw.state = RunState.FINISHED
    recovered = RunPersistence(manager, executor, first.record)
    recovered.completed(raw)
    assert manager.completions == 1
    assert executor.acknowledgements == 1


def test_restart_retries_ack_without_repeating_terminal_commit():
    manager, raw, executor = _Manager(), _Runner(), _Executor()
    record = SimpleNamespace(
        id="run", state=RunState.FINISHED,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=_execution(),
    )
    recovered = RunPersistence(manager, executor, record)
    raw.state = RunState.FINISHED
    assert recovered.completed(raw) is RunState.FINISHED
    assert recovered.completed(raw) is RunState.FINISHED
    assert manager.completions == 0
    assert executor.acknowledgements == 1


def test_failed_ack_is_retryable_without_rewriting_completion():
    manager, raw = _Manager(), _Runner()
    record = SimpleNamespace(
        id="run", state=RunState.FINISHED,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=_execution(),
    )

    class FlakyExecutor:
        def __init__(self):
            self.attempts = 0

        def acknowledge(self, record):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("lost reply")

    executor = FlakyExecutor()
    recovered = RunPersistence(manager, executor, record)
    recovered.completed(raw)
    recovered.completed(raw)
    recovered.completed(raw)
    assert executor.attempts == 2
    assert manager.completions == 0


def test_lost_ack_reply_after_receipt_deletion_is_safe_on_retry(tmp_path, monkeypatch):
    """A lost reply after a successful delete must not repeat durable completion."""
    from uuid import uuid4

    from .. import remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    receipt_id = uuid4().hex
    assert remote_mailbox.publish_outcome(receipt_id, RunState.FINISHED, 0)

    class LostReplyExecutor:
        def __init__(self):
            self.attempts = []
            self.fail_reply = True

        def acknowledge(self, record):
            self.attempts.append(remote_mailbox.acknowledge_outcome(receipt_id))
            if self.fail_reply:
                self.fail_reply = False
                raise OSError("response lost after durable receipt deletion")

    manager, raw, executor = _Manager(), _Runner(), LostReplyExecutor()
    record = SimpleNamespace(
        id="run", state=RunState.RUNNING,
        updated_at="2026-10-20T10:00:00+00:00",
        execution=_execution(),
    )
    raw.state = RunState.FINISHED
    persistence = RunPersistence(manager, executor, record)
    assert persistence.completed(raw) is RunState.FINISHED
    assert remote_mailbox.read_outcome(receipt_id) is None
    assert manager.completions == 1
    assert executor.attempts == [True]

    # The server has no receipt left. Its idempotent ACK returns False, but
    # successful communication still resolves the client's uncertainty.
    assert persistence.completed(raw) is RunState.FINISHED
    assert persistence.completed(raw) is RunState.FINISHED
    assert manager.completions == 1
    assert executor.attempts == [True, False]

def test_persistent_reservation_rejects_duplicate_before_submitting(tmp_path):
    """The store claim, not another hook-local flag, prevents duplicates."""
    import pytest
    from ..manager import ExecutionManager, RunStore

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = manager.store.save(manager.new_record(task="DOS", directory=tmp_path))
    persistence = RunPersistence(manager, _Executor(), record)
    runner = SimpleNamespace(job_files=JobFiles(record.files))
    persistence.before_submit(runner)
    assert persistence.record.state is RunState.SUBMITTING
    with pytest.raises(RuntimeError, match="already"):
        persistence.before_submit(runner)
    assert runner.job_files.manifest == record.files
    assert manager.store.get(record.id).state is RunState.SUBMITTING
