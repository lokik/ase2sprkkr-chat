"""Submission ownership and crash/recovery regression tests."""

from threading import Event, Thread

import pytest

from ase2sprkkr.execution import (
    DuplicateSubmissionError, ExecutionSpec, RunState, RunStatus, RunStore,
)
from ase2sprkkr.execution.run_manager import RunManager
from ase2sprkkr.execution.pool import Executors
from ase2sprkkr.execution.run_record import SubmissionPhase
from ase2sprkkr.execution.test.test_run_manager import _Executor, _manager


def _run(manager, executor, directory):
    return manager.create_run(
        ExecutionSpec(argv=(), input_file=None, output_file=None, directory=str(directory)),
        endpoint=manager.executors.register(executor),
        task="SCF", process_result=lambda outcome: outcome,
    )


def test_owner_survives_long_staging_and_recovery_cannot_release(tmp_path):
    manager = RunManager(RunStore(tmp_path / "runs"), executors=Executors(), submitting_grace=0)
    executor = _Executor()
    first = _run(manager, executor, tmp_path)
    entered, release = Event(), Event()
    original_prepare = first._runner.prepare_submission

    def slow_prepare():
        entered.set()
        assert release.wait(5), "submit worker was not released"
        return original_prepare()

    first._runner.prepare_submission = slow_prepare
    errors = []

    def submit():
        try:
            first.submit()
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=submit)
    worker.start()
    try:
        assert entered.wait(5)
        record = manager.store.get(first.id)
        assert record.submission_phase == SubmissionPhase.PREPARING
        assert manager.refresh((record,))[0].state == RunState.SUBMITTING
        assert manager.store.get(first.id).state == RunState.SUBMITTING
        with pytest.raises(DuplicateSubmissionError):
            _run(manager, executor, tmp_path).submit()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors
    assert manager.store.get(first.id).execution.backend_id is not None


def test_owner_lock_released_after_thread_baseexception_while_gui_alive(tmp_path):
    manager = _manager(tmp_path)
    first = _run(manager, _Executor(), tmp_path)

    def fatal_prepare():
        raise KeyboardInterrupt("worker died")

    first._runner.prepare_submission = fatal_prepare
    errors = []

    def submit():
        try:
            first.submit()
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=submit)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], KeyboardInterrupt)
    assert manager.store.get(first.id).state == RunState.FAILED
    # Same Python process and GUI is alive; the OS lock was explicitly closed.
    with manager.store.submission_owner(first.record) as acquired:
        assert acquired


def test_abandoned_preparing_local_fails_and_releases(tmp_path):
    manager = _manager(tmp_path)
    original = manager.new_record(task="SCF", directory=tmp_path)
    reserved = manager.reserve(original)
    assert reserved.submission_phase == SubmissionPhase.PREPARING
    recovered = manager.refresh((reserved,))[0]
    assert recovered.state == RunState.FAILED
    assert recovered.submission_phase is None
    assert manager.reserve(manager.new_record(task="SCF", directory=tmp_path)).state == RunState.SUBMITTING


def test_abandoned_preparing_remote_remains_reserved(tmp_path):
    manager = _manager(tmp_path)
    executor = _Executor()  # No guarantee that remote staging subprocesses exited.
    endpoint = manager.executors.register(executor)
    reserved = manager.reserve(manager.new_record(task="SCF", directory=tmp_path, endpoint=endpoint))
    assert manager.refresh((reserved,))[0] == reserved


@pytest.mark.parametrize("recovery", [None, NotImplementedError(), OSError("offline")])
def test_unknown_dispatch_preserves_reservation(tmp_path, recovery):
    manager = RunManager(RunStore(tmp_path / "runs"), executors=Executors(), submitting_grace=0)
    executor = _Executor()
    endpoint = manager.executors.register(executor)
    reserved = manager.reserve(manager.new_record(task="SCF", directory=tmp_path, endpoint=endpoint))
    dispatched = manager.store.begin_dispatch(reserved)

    def recover(*args, **kwargs):
        if isinstance(recovery, BaseException):
            raise recovery
        return recovery

    executor.recover_submission = recover
    assert manager.refresh((dispatched,))[0] == dispatched
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(manager.new_record(task="SCF", directory=tmp_path, endpoint=endpoint))


def test_found_dispatch_restores_identity(tmp_path):
    manager = RunManager(RunStore(tmp_path / "runs"), executors=Executors(), submitting_grace=0)
    executor = _Executor()
    endpoint = manager.executors.register(executor)
    reserved = manager.reserve(manager.new_record(task="SCF", directory=tmp_path, endpoint=endpoint))
    dispatched = manager.store.begin_dispatch(reserved)
    executor.recover_submission = lambda *args, **kwargs: RunStatus(
        state=RunState.QUEUED, backend_id="job-123",
    )
    recovered = manager.refresh((dispatched,))[0]
    assert recovered.execution.backend_id == "job-123"
    assert recovered.state == RunState.QUEUED
    assert recovered.submission_phase is None



@pytest.mark.parametrize(
    "result,exception,pattern",
    [
        (object(), TypeError, "Invalid backend submission recovery result"),
        (RunStatus(state=RunState.RUNNING), ValueError, "lacks backend identity"),
    ],
)
def test_invalid_recovery_result_preserves_dispatch_reservation(
    tmp_path, monkeypatch, result, exception, pattern,
):
    """Malformed recovery cannot release the claim or invent backend identity."""
    manager = RunManager(
        RunStore(tmp_path / "runs"), executors=Executors(), submitting_grace=0,
    )
    executor = _Executor()
    endpoint = manager.executors.register(executor)
    first = manager.reserve(
        manager.new_record(task="SCF", directory=tmp_path, endpoint=endpoint),
    )
    dispatched = manager.store.begin_dispatch(first)
    monkeypatch.setattr(
        executor, "recover_submission", lambda *args, **kwargs: result,
    )
    with pytest.raises(exception, match=pattern):
        manager.recover_submitting(dispatched)
    assert manager.store.get(dispatched.id) == dispatched

    # Neither malformed observation may authorize a conflicting second run.
    another = manager.new_record(
        task="SCF", directory=tmp_path, endpoint=endpoint,
    )
    with pytest.raises(exception, match=pattern):
        manager.reserve(another)
    assert another.state == RunState.NEW
    assert manager.store.get(dispatched.id) == dispatched


def test_cannot_dispatch_after_reservation_was_reclaimed(tmp_path):
    manager = _manager(tmp_path)
    reserved = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    manager.recover_submitting(reserved)
    with pytest.raises(RuntimeError, match="lost"):
        manager.store.begin_dispatch(reserved)
