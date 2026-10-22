from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import subprocess

from ase2sprkkr.execution import (
    DuplicateSubmissionError,
    ExecutionOutcome,
    ExecutionSpec,
    Executor,
    FailurePhase,
    RunState,
    RunStatus,
    RunStore,
    FileManifest,
)

from ase2sprkkr.execution.manager import ExecutionManager
from ase2sprkkr.execution.job_files import JobFiles
from ase2sprkkr.execution.pool import Executors


class _RawRunner:
    def __init__(self, spec, executor, on_submit=None):
        self.spec = spec
        self.executor = executor
        self.job_files = JobFiles(FileManifest.from_spec(spec))
        self.callback = None
        self.on_submit = on_submit
        self.state = RunState.NEW
        self.backend_id = None
        self.result = ExecutionOutcome(return_code=None)
        self.exception = None
        self.failure_phase = None
        self.failure_message = None
        self._done = Event()

    @property
    def return_code(self):
        return self.result.return_code

    def prepare_submission(self):
        pass

    def submit(self):
        if self.on_submit:
            self.on_submit()
        self.backend_id = "native-1"
        self.state = RunState.QUEUED

    def wait(self):
        self._done.wait()
        return self.result

    def finish(self, return_code=0):
        self.result.return_code = return_code
        if return_code == 0:
            self.state = RunState.FINISHED
            self.failure_phase = None
            self.failure_message = None
        else:
            self.state = RunState.FAILED
            self.failure_phase = FailurePhase.EXECUTION
            self.failure_message = f"process exited with {return_code}"
        if self.callback is not None:
            self.callback(self.result)
        self._done.set()

    def cancel(self):
        self.state = RunState.CANCELLED
        self._done.set()

    def add_output_listener(self, callback):
        pass

    def add_output_error_listener(self, callback):
        pass


class _Executor(Executor):
    name = "test"

    def __init__(self):
        self.runners = []
        self.status_calls = []
        self.on_submit = None

    def create_project_run(self, spec, *, run_id, callback=None):
        runner = _RawRunner(spec, self, on_submit=self.on_submit)
        self.runners.append(runner)
        return runner

    def status(self, *runs):
        self.status_calls.append(runs)
        return tuple(RunStatus(RunState.RUNNING, backend_id=run.execution.backend_id) for run in runs)


def _spec(tmp_path, *, dataset="case"):
    return ExecutionSpec(
        argv=(),
        directory=str(tmp_path),
        input_file=str(tmp_path / f"{dataset}.inp"),
        output_file=str(tmp_path / f"{dataset}.out"),
    )


def _manager(tmp_path):
    return ExecutionManager(
        RunStore(tmp_path / "runs"), executors=Executors()
    )


def test_submit_persists_submitting_before_backend_call(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome,
        task="ARPES",
        dataset="case",
    )
    seen = []
    executor.on_submit = lambda: seen.append(manager.store.get(run.id).state)
    # The runner already captured the callable during construction.
    executor.runners[0].on_submit = executor.on_submit

    run.submit()

    assert seen == [RunState.SUBMITTING]
    stored = manager.store.get(run.id)
    assert stored.state == RunState.QUEUED
    assert stored.finished_at is None
    assert stored.execution.backend_id == executor.runners[0].backend_id
    assert stored.task == "ARPES"
    assert stored.dataset == "case"


@pytest.mark.parametrize(
    "ack_error",
    [
        pytest.param(OSError("SSH disconnected"), id="os-error"),
        pytest.param(
            subprocess.CalledProcessError(255, ("ssh", "ack-submit")),
            id="ssh-command-error",
        ),
    ],
)
def test_submission_ack_failure_does_not_undo_durable_submit(
    tmp_path, monkeypatch, ack_error,
):
    manager = _manager(tmp_path)
    executor = _Executor()
    observed = []

    def fail_ack(record):
        stored = manager.store.get(record.id)
        observed.append((stored.state, stored.execution.backend_id))
        raise ack_error

    monkeypatch.setattr(executor, "acknowledge_submission", fail_ack)
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome,
        task="ARPES",
    )

    run.submit()

    assert observed == [(RunState.QUEUED, "native-1")]
    assert run.state == RunState.QUEUED
    assert manager.store.get(run.id).execution.backend_id == "native-1"


@pytest.mark.parametrize(
    "ack_error",
    [
        pytest.param(OSError("SSH disconnected"), id="os-error"),
        pytest.param(
            subprocess.CalledProcessError(255, ("ssh", "ack")),
            id="ssh-command-error",
        ),
    ],
)
def test_terminal_ack_failure_preserves_completion_and_retries(
    tmp_path, monkeypatch, ack_error,
):
    manager = _manager(tmp_path)
    executor = _Executor()
    callback_results = []
    observed_states = []
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome,
        task="DOS", callback=callback_results.append,
    )
    run.submit()

    def fail_ack(record):
        observed_states.append(manager.store.get(record.id).state)
        raise ack_error

    monkeypatch.setattr(executor, "acknowledge", fail_ack)
    executor.runners[0].finish()
    result = run.wait()

    assert result.return_code == 0
    assert callback_results == [result]
    assert run.persistence_error is None
    assert manager.store.get(run.id).state == RunState.FINISHED
    assert observed_states == [RunState.FINISHED]

    def successful_ack(record):
        observed_states.append(manager.store.get(record.id).state)
        return True

    monkeypatch.setattr(executor, "acknowledge", successful_ack)
    assert run.persistence.completed(run) == RunState.FINISHED
    assert run.persistence.completed(run) == RunState.FINISHED
    assert observed_states == [RunState.FINISHED, RunState.FINISHED]
    assert callback_results == [result]


def test_submission_ack_does_not_hide_programming_errors(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome, task="DOS",
    )
    run.submit()

    def bad_ack(_record):
        raise ValueError("invalid ACK payload")

    monkeypatch.setattr(executor, "acknowledge_submission", bad_ack)
    with pytest.raises(ValueError, match="invalid ACK payload"):
        manager._acknowledge_submission(executor, manager.store.get(run.id))


@pytest.mark.parametrize(
    "return_code, expected_state, expected_phase",
    [
        pytest.param(0, RunState.FINISHED, None, id="postprocess-success"),
        pytest.param(7, RunState.FAILED, FailurePhase.EXECUTION,
                     id="skip-processing-on-failure"),
    ],
)
def test_completion_postprocesses_only_successful_execution(
    tmp_path, return_code, expected_state, expected_phase,
):
    manager = _manager(tmp_path)
    executor = _Executor()
    processed = []
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: processed.append(outcome) or outcome,
        task="DOS",
    )
    run.submit()
    executor.runners[0].finish(return_code)

    result = run.wait()

    assert result.return_code == return_code
    assert processed == ([result] if return_code == 0 else [])
    stored = manager.store.get(run.id)
    assert stored.state == expected_state
    assert stored.failure_phase == expected_phase
    if return_code == 0:
        assert stored.finished_at is not None
    else:
        assert stored.return_code == return_code


def test_refresh_batches_status_queries(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()
    endpoint = manager.executors.register(executor)
    records = []
    for index in range(2):
        record = manager.new_record(
            task="DOS",
            directory=tmp_path / str(index),
            endpoint=endpoint,
        )
        record = manager.transition(
            record,
            state=RunState.QUEUED,
            execution=record.execution.submitted(str(index)),
        )
        records.append(record)

    refreshed = manager.refresh(records)

    assert [record.state for record in refreshed] == [
        RunState.RUNNING,
        RunState.RUNNING,
    ]
    assert executor.status_calls == [tuple(records)]


def test_duplicate_submission_is_blocked_by_persisted_namespace(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()
    first = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome,
        task="DOS",
        dataset="case",
    )
    first.submit()

    second = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome,
        task="DOS",
        dataset="case",
    )

    with pytest.raises(DuplicateSubmissionError):
        second.submit()


def test_project_manager_returns_one_public_calculation_runner(tmp_path):
    from ase2sprkkr.execution import CalculationRun, RunPersistence

    manager = _manager(tmp_path)
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(_Executor()), task="DOS",
        process_result=lambda value: value,
    )

    assert type(run) is CalculationRun
    assert isinstance(run.persistence, RunPersistence)
    assert not hasattr(run.persistence, "_runner")
    assert run.record is run.persistence.record
    assert run.id == run.record.id
    assert run.record.state is RunState.NEW
    assert run.refresh().state is RunState.NEW

    run.submit()
    assert run.record.state is RunState.QUEUED
    assert run.backend_id == "native-1"


def test_before_submit_returns_reserved_files_without_mutating_runner(tmp_path, monkeypatch):
    """Persistence reserves a snapshot; only CalculationRun installs it."""
    from dataclasses import replace
    from ase2sprkkr.execution.run_model import FileMapping

    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome, task="DOS",
    )
    original = run.job_files.manifest
    reserved = replace(original, mapping=FileMapping(str(tmp_path), "/remote/dos"))
    monkeypatch.setattr(
        executor, "reservation_context",
        lambda spec, files: (reserved, "/remote/dos"),
    )

    assert run.persistence.before_submit(run) is reserved
    assert run.job_files.manifest is original
    assert run.record.files is reserved
    assert run.record.state == RunState.SUBMITTING
    assert manager.store.get(run.id).files == reserved


def test_calculation_run_installs_reserved_files_before_staging(tmp_path, monkeypatch):
    from dataclasses import replace
    from ase2sprkkr.execution.run_model import FileMapping

    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome, task="DOS",
    )
    original = run.job_files.manifest
    reserved = replace(original, mapping=FileMapping(str(tmp_path), "/remote/dos"))
    monkeypatch.setattr(
        executor, "reservation_context",
        lambda spec, files: (reserved, "/remote/dos"),
    )
    raw = executor.runners[0]
    staged = []
    monkeypatch.setattr(raw, "prepare_submission", lambda: staged.append(raw.job_files.manifest))

    run.submit()

    assert staged == [reserved]
    assert staged[0] is reserved
    assert run.job_files is raw.job_files
    assert run.job_files.manifest is reserved
    assert run.record.files == reserved


def test_calculation_run_job_files_is_read_only_runner_view(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path),
        endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome,
        task="DOS",
    )
    files = executor.runners[0].job_files
    assert run.job_files is files
    with pytest.raises(AttributeError):
        run.job_files = JobFiles(FileManifest())
    assert run.job_files is files


def test_executors_are_owned_by_manager_and_keyed_by_endpoint(tmp_path):
    manager = _manager(tmp_path)
    assert isinstance(manager.executors, Executors)
    assert not hasattr(manager.executors, 'pool')
    executor = _Executor()
    endpoint = manager.executors.register(executor)
    record = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    assert manager.executors.for_endpoint(record.execution.endpoint) is executor


def test_executor_registries_isolate_endpoint_instances():
    from ase2sprkkr.execution import ExecutorEndpoint

    first, second = Executors(), Executors()
    endpoint = ExecutorEndpoint()
    first_executor, second_executor = _Executor(), _Executor()
    first.register(first_executor, endpoint)
    second.register(second_executor, endpoint)
    assert first.for_endpoint(endpoint) is first_executor
    assert second.for_endpoint(endpoint) is second_executor


def test_register_executor_rejects_conflicting_instance():
    from ase2sprkkr.execution import ExecutorEndpoint

    executors = Executors()
    endpoint = ExecutorEndpoint()
    original, conflicting = _Executor(), _Executor()
    assert executors.register(original, endpoint) == endpoint
    assert executors.register(original, endpoint) == endpoint
    with pytest.raises(ValueError, match="already registered"):
        executors.register(conflicting, endpoint)
    assert executors.for_endpoint(endpoint) is original


def test_explicit_executor_for_run_is_the_registered_instance(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor), process_result=lambda value: value, task="DOS",
    )
    assert run.persistence.executor is manager.executors.for_endpoint(run.record.execution.endpoint)


def test_executor_registration_normalizes_serialized_endpoint():
    from dataclasses import asdict
    from ase2sprkkr.execution import ExecutorEndpoint

    executors = Executors()
    endpoint = ExecutorEndpoint(executor="custom:test")
    executor = _Executor()
    assert executors.register(executor, asdict(endpoint)) == endpoint
    assert executors.for_endpoint(endpoint) is executor
    assert executors.for_endpoint(asdict(endpoint)) is executor
    assert executors.register(executor, endpoint) == endpoint
    with pytest.raises(ValueError, match="already registered"):
        executors.register(_Executor(), asdict(endpoint))


def test_parallel_endpoint_lookup_constructs_one_executor(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock
    from ase2sprkkr.execution import ExecutorEndpoint
    from ase2sprkkr.execution import endpoint as endpoint_module

    registry = Executors()
    endpoint = ExecutorEndpoint(executor="local")
    start = Barrier(8)
    lock = Lock()
    instances = []

    def factory(actual_endpoint):
        assert actual_endpoint == endpoint
        instance = _Executor()
        with lock:
            instances.append(instance)
        return instance

    monkeypatch.setattr(endpoint_module, "create_executor_from_endpoint", factory)

    def lookup(_):
        start.wait(timeout=10)
        return registry.for_endpoint(endpoint)

    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(lookup, range(8)))
    assert len(instances) == 1
    assert all(instance is instances[0] for instance in results)


def test_executor_factory_failure_is_not_cached(monkeypatch):
    from ase2sprkkr.execution import ExecutorEndpoint
    from ase2sprkkr.execution import endpoint as endpoint_module

    executors = Executors()
    endpoint = ExecutorEndpoint()
    attempts = []

    def factory(value):
        assert value == endpoint
        attempts.append(_Executor())
        if len(attempts) == 1:
            raise RuntimeError("temporary backend failure")
        return attempts[-1]

    monkeypatch.setattr(endpoint_module, "create_executor_from_endpoint", factory)
    with pytest.raises(RuntimeError, match="temporary backend failure"):
        executors.for_endpoint(endpoint)
    assert executors.for_endpoint(endpoint) is attempts[1]
    assert executors.for_endpoint(endpoint) is attempts[1]
    assert len(attempts) == 2


def test_parallel_registration_of_same_executor_is_idempotent():
    from concurrent.futures import ThreadPoolExecutor
    from ase2sprkkr.execution import ExecutorEndpoint

    executors = Executors()
    endpoint = ExecutorEndpoint()
    executor = _Executor()
    with ThreadPoolExecutor(max_workers=8) as workers:
        registered = tuple(workers.map(
            lambda _: executors.register(executor, endpoint), range(32)
        ))
    assert all(value == endpoint for value in registered)
    assert executors.for_endpoint(endpoint) is executor


def test_refresh_groups_by_endpoint_not_executor_hash(tmp_path):
    from ase2sprkkr.execution import ExecutorEndpoint

    class UnhashableExecutor:
        __hash__ = None

        def acknowledge_submission(self, record):
            pass

        def status(self, *records):
            return tuple(RunStatus(state=RunState.RUNNING) for _ in records)

    manager = _manager(tmp_path)
    endpoint = ExecutorEndpoint(executor="custom:unhashable")
    manager.executors.register(UnhashableExecutor(), endpoint)
    record = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    record = manager.transition(
        record, state=RunState.QUEUED,
        execution=record.execution.submitted("native-1"),
    )
    assert manager.refresh((record,))[0].state is RunState.RUNNING


def test_refresh_does_not_query_terminal_runs(tmp_path):
    from ase2sprkkr.execution import ExecutorEndpoint

    class RecordingExecutor:
        def __init__(self):
            self.polled = []

        def acknowledge_submission(self, record):
            pass

        def status(self, *records):
            self.polled.extend(record.id for record in records)
            return tuple(RunStatus(state=RunState.RUNNING) for _ in records)

    manager = _manager(tmp_path)
    endpoint = ExecutorEndpoint(executor="custom:recording")
    executor = RecordingExecutor()
    manager.executors.register(executor, endpoint)
    active = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    active = manager.transition(active, state=RunState.QUEUED,
                                execution=active.execution.submitted("job-active"))
    finished = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    finished = manager.transition(finished, state=RunState.FINISHED,
                                  execution=finished.execution.submitted("job-finished"))
    result = manager.refresh((finished, active))
    assert [item.state for item in result] == [RunState.FINISHED, RunState.RUNNING]
    assert executor.polled == [active.id]


def test_update_without_backend_does_not_reconstruct_executor(tmp_path):
    from ase2sprkkr.execution import ExecutorEndpoint

    manager = _manager(tmp_path)
    endpoint = ExecutorEndpoint(executor="custom:cannot-reconstruct")
    record = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    updated = manager.update(record, RunStatus(state=RunState.SUBMITTING))
    assert updated.state is RunState.SUBMITTING
    assert updated.execution.backend_id is None


def test_refresh_does_not_apply_stale_status_after_terminal_commit(tmp_path):
    """A slow status query cannot add stale failure data to a newer finish."""
    manager = _manager(tmp_path)
    executor = _Executor()
    endpoint = manager.executors.register(executor)
    original = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    queued = manager.transition(
        original, state=RunState.QUEUED,
        execution=original.execution.submitted("native-1"),
    )
    finished = []

    def stale_status(*records):
        assert records == (queued,)
        finished.append(manager.transition(
            queued, state=RunState.FINISHED, return_code=0,
        ))
        return (RunStatus(
            RunState.FAILED, backend_id="native-1",
            message="outdated scheduler failure",
        ),)

    executor.status = stale_status
    assert manager.refresh((queued,)) == (finished[0],)
    assert manager.store.get(queued.id) == finished[0]
    assert finished[0].message is None


def test_refresh_ignores_stale_scheduler_observation(tmp_path):
    from ase2sprkkr.execution import ExecutorEndpoint

    manager = _manager(tmp_path)
    endpoint = ExecutorEndpoint(executor="custom:stale-refresh")
    record = manager.new_record(task="DOS", directory=tmp_path, endpoint=endpoint)
    record = manager.transition(
        record, state=RunState.QUEUED,
        execution=record.execution.submitted("native-1"),
    )

    class RacingExecutor:
        def status(self, *records):
            manager.transition(record, state=RunState.RUNNING, message="fresh")
            return (RunStatus(state=RunState.UNKNOWN, message="stale"),)

        def acknowledge_submission(self, record):
            pass

    manager.executors.register(RacingExecutor(), endpoint)
    refreshed = manager.refresh((record,))[0]
    assert refreshed.state is RunState.RUNNING
    assert refreshed.message == "fresh"
    assert manager.store.get(record.id) == refreshed

@pytest.mark.parametrize("terminal", [RunState.FINISHED, RunState.FAILED])
def test_attach_does_not_commit_terminal_before_processing(tmp_path, terminal):
    from ase2sprkkr.execution.run_persistence import RunPersistence

    manager = _manager(tmp_path)
    executor = _Executor()
    record = manager.new_record(task="DOS", directory=tmp_path)
    queued = manager.transition(
        record, state=RunState.QUEUED,
        execution=record.execution.submitted("native-1"),
    )
    persistence = RunPersistence(manager, executor, queued)
    observed = SimpleNamespace(
        state=terminal, backend_id="native-1",
        failure_phase=FailurePhase.EXECUTION if terminal == RunState.FAILED else None,
    )

    persistence.attached(observed)

    assert persistence.record == queued
    assert manager.store.get(queued.id) == queued
    assert queued.finished_at is None


def test_refresh_cannot_promote_stale_monitor_unknown(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome, task="DOS",
    )
    run.submit()
    refreshed = run.refresh()
    assert refreshed.state == RunState.RUNNING

    stale_monitor = SimpleNamespace(
        state=RunState.UNKNOWN, backend_id=run.backend_id,
        return_code=None, failure_phase=None,
        failure_message="stale monitor disconnected", exception=None,
        job_files=run.job_files,
    )
    result = run.persistence.completed(stale_monitor)

    assert result == RunState.RUNNING
    assert run.record.state == RunState.RUNNING
    assert not run.record.confirmed
    assert run.record.error == "stale monitor disconnected"
    assert run.record.last_observed == refreshed.last_observed
    assert manager.store.get(run.id) == run.record


def test_slow_terminal_ack_does_not_block_refresh_or_ack_twice(tmp_path, monkeypatch):
    manager = _manager(tmp_path)
    executor = _Executor()
    run = manager.create_run(
        _spec(tmp_path), endpoint=manager.executors.register(executor),
        process_result=lambda outcome: outcome, task="DOS",
    )
    run.submit()
    ack_entered = Event()
    ack_release = Event()
    acknowledged = []

    def slow_ack(record):
        acknowledged.append(record.id)
        ack_entered.set()
        assert ack_release.wait(timeout=5)
        return True

    monkeypatch.setattr(executor, "acknowledge", slow_ack)
    with ThreadPoolExecutor(max_workers=3) as workers:
        finish = workers.submit(executor.runners[0].finish)
        try:
            assert ack_entered.wait(timeout=3)
            # refresh must not acquire the lock currently held by remote ACK.
            refreshed = workers.submit(run.refresh)
            assert refreshed.result(timeout=2).state == RunState.FINISHED
            duplicate = workers.submit(run.persistence.completed, run)
        finally:
            ack_release.set()
        finish.result(timeout=3)
        assert duplicate.result(timeout=3) == RunState.FINISHED

    assert acknowledged == [run.id]
    assert manager.store.get(run.id).state == RunState.FINISHED

def test_attached_unknown_does_not_override_newer_scheduler_state(tmp_path):
    from ase2sprkkr.execution.run_persistence import RunPersistence

    manager = _manager(tmp_path)
    executor = _Executor()
    initial = manager.new_record(task="DOS", directory=tmp_path)
    queued = manager.transition(
        initial, state=RunState.QUEUED,
        execution=initial.execution.submitted("native-1"),
    )
    persistence = RunPersistence(manager, executor, queued)
    newer = manager.transition(
        queued, state=RunState.RUNNING, message="scheduler confirmed",
    )

    persistence.attached(SimpleNamespace(
        state=RunState.UNKNOWN, backend_id="native-1",
        failure_phase=None,
    ))

    assert persistence.record == newer
    assert manager.store.get(initial.id) == newer
    assert persistence.record.message == "scheduler confirmed"
