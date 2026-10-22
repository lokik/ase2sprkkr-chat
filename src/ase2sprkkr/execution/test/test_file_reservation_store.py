"""Atomic file namespace reservation for persistent execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution.base import RunState, RunStatus, SubmissionRecovery
from ase2sprkkr.execution.manager import DuplicateSubmissionError, ExecutionManager
from ase2sprkkr.execution.run_store import RunStore
from ase2sprkkr.execution.run_record import SubmissionPhase


def _dispatch(manager, record):
    """Commit the uncertain launch boundary after a file reservation."""
    prepared = manager.reserve(record)
    assert prepared.submission_phase == SubmissionPhase.PREPARING
    dispatched = manager.store.begin_dispatch(prepared)
    assert dispatched.submission_phase == SubmissionPhase.DISPATCHING
    return dispatched


def test_conflict_does_not_write_unclaimed_record(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    claimed, conflicts = manager.store.reserve_files(second)
    assert claimed is None
    assert [record.id for record in conflicts] == [first.id]
    assert second.state == RunState.NEW


def test_distinct_endpoints_are_independent(tmp_path):
    from ase2sprkkr.execution.endpoint import ExecutorEndpoint
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    a = manager.new_record(task="SCF", directory=tmp_path)
    b = manager.new_record(task="SCF", directory=tmp_path,
                           endpoint=ExecutorEndpoint(executor="ssh", host="example.org"))
    manager.reserve(a)
    assert manager.reserve(b).state == RunState.SUBMITTING


def test_reservation_rejects_stale_new_record(tmp_path):
    """An outdated claimant cannot replace a newer persisted run."""
    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    original = manager.new_record(task="DOS", directory=tmp_path, dataset="old")
    store.save(replace(original, dataset="updated"))

    with pytest.raises(RuntimeError, match="changed during file reservation"):
        store.reserve_files(original)

    latest = store.get(original.id)
    assert latest.dataset == "updated"
    assert latest.state == RunState.NEW



def test_recovery_does_not_fail_concurrently_refreshed_submission(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store, submitting_grace=0)
    submitted = _dispatch(manager, manager.new_record(task="DOS", directory=tmp_path))

    def recover(run_id, record, *, submitted_at):
        assert (run_id, record, submitted_at) == (
            submitted.id, submitted, submitted.updated_at,
        )
        manager.transition(
            submitted, state=RunState.SUBMITTING, message="newer submission evidence",
        )
        return SubmissionRecovery.ABSENT

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(recover_submission=recover),
    )
    result = manager.recover_submitting(submitted)
    assert result.state == RunState.SUBMITTING
    assert result.submission_phase == SubmissionPhase.DISPATCHING
    assert result.message == "newer submission evidence"
    assert store.get(submitted.id) == result


def test_unsubmitted_record_does_not_reserve_files(tmp_path):
    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    draft = manager.new_record(task="SCF", directory=tmp_path)
    store.save(draft)

    next_run = manager.new_record(task="SCF", directory=tmp_path)
    assert manager.reserve(next_run).state == RunState.SUBMITTING
    assert store.get(draft.id).state == RunState.NEW


def test_stale_claim_does_not_sync_files(tmp_path, monkeypatch):
    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    original = manager.new_record(task="DOS", directory=tmp_path)
    store.save(replace(original, dataset="newer"))

    def unexpected_sync(manifests):
        raise AssertionError("Stale run must not sync inputs")

    import ase2sprkkr.execution.run_store as run_store_module
    monkeypatch.setattr(run_store_module, "sync_result_files", unexpected_sync)
    with pytest.raises(RuntimeError, match="changed during file reservation"):
        store.reserve_files(original)


def test_reservations_are_atomic_across_store_instances(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    directory = tmp_path / "runs"
    first = ExecutionManager(RunStore(directory))
    second = ExecutionManager(RunStore(directory))
    records = (
        first.new_record(task="SCF", directory=tmp_path),
        second.new_record(task="SCF", directory=tmp_path),
    )
    barrier = Barrier(2)

    def reserve(manager, record):
        barrier.wait()
        claimed, conflicts = manager.store.reserve_files(record)
        return claimed, conflicts

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(reserve, first, records[0])
        b = pool.submit(reserve, second, records[1])
        results = (a.result(), b.result())

    assert sum(claimed is not None for claimed, _ in results) == 1
    assert sum(bool(conflicts) for _, conflicts in results) == 1
    assert len(first.store.runs(active=True)) == 1



def test_recovered_status_does_not_overwrite_newer_backend_observation(
    tmp_path, monkeypatch,
):
    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store, submitting_grace=0)
    submitted = _dispatch(manager, manager.new_record(task="DOS", directory=tmp_path))

    def recover(*args, **kwargs):
        manager.transition(
            submitted, state=RunState.RUNNING,
            execution=submitted.execution.submitted("current-backend"),
        )
        return RunStatus(
            state=RunState.FAILED, backend_id="stale-backend",
            message="stale recovery",
        )

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(
            recover_submission=recover, acknowledge_submission=lambda record: None,
        ),
    )
    result = manager.recover_submitting(submitted)
    assert result.state == RunState.RUNNING
    assert result.execution.backend_id == "current-backend"
    assert result.message is None
    assert store.get(submitted.id) == result


def test_corrupt_run_blocks_reservation_but_history_remains_readable(tmp_path):
    from ase2sprkkr.execution.run_store import CorruptRunRecord

    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    original = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    path = store._path(original)
    path.write_text("{broken json")

    with pytest.warns(RuntimeWarning, match="Corrupt run record"):
        assert store.runs(active=True) == ()

    with pytest.raises(CorruptRunRecord):
        manager.reserve(manager.new_record(task="SCF", directory=tmp_path))

    assert path.read_text() == "{broken json"



def test_stale_missing_submission_releases_reservation(tmp_path, monkeypatch):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=0)
    first = _dispatch(manager, manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    calls = []

    def recover(run_id, record, *, submitted_at):
        calls.append((run_id, record, submitted_at))
        return SubmissionRecovery.ABSENT

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(recover_submission=recover),
    )
    reserved = manager.reserve(second)
    assert reserved.state == RunState.SUBMITTING
    assert reserved.submission_phase == SubmissionPhase.PREPARING
    abandoned = manager.store.get(first.id)
    assert abandoned.state == RunState.FAILED
    assert abandoned.submission_phase is None
    assert calls == [(first.id, first, first.updated_at)]



def test_recovered_active_backend_still_blocks_reservation(tmp_path, monkeypatch):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=0)
    first = _dispatch(manager, manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    calls = []

    def recover(run_id, record, *, submitted_at):
        calls.append((run_id, submitted_at))
        return RunStatus(state=RunState.RUNNING, backend_id="still-running")

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(
            recover_submission=recover,
            acknowledge_submission=lambda record: None,
        ),
    )
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    recovered = manager.store.get(first.id)
    assert recovered.state == RunState.RUNNING
    assert recovered.execution.backend_id == "still-running"
    assert recovered.submission_phase is None
    assert second.state == RunState.NEW
    assert calls == [(first.id, first.updated_at)]



def test_unsupported_recovery_preserves_reservation(tmp_path, monkeypatch):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=0)
    first = _dispatch(manager, manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    calls = []

    def unsupported(*args, **kwargs):
        calls.append(1)
        raise NotImplementedError

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(recover_submission=unsupported),
    )
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    persisted = manager.store.get(first.id)
    assert persisted.state == RunState.SUBMITTING
    assert persisted.submission_phase == SubmissionPhase.DISPATCHING
    assert second.state == RunState.NEW
    assert calls == [1]



def test_fresh_submission_is_not_recovered(tmp_path, monkeypatch):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=3600)
    first = _dispatch(manager, manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)

    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("Fresh dispatch must not be recovered")

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(recover_submission=unexpected_recovery),
    )
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    persisted = manager.store.get(first.id)
    assert persisted.state == RunState.SUBMITTING
    assert persisted.submission_phase == SubmissionPhase.DISPATCHING



@pytest.mark.parametrize("safe", (False, True))
def test_abandoned_preparing_requires_executor_safety(tmp_path, monkeypatch, safe):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=3600)
    first = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    assert first.submission_phase == SubmissionPhase.PREPARING
    calls = []

    def preparation_is_safe():
        calls.append("preparing")
        return safe

    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("PREPARING must not query submission receipts")

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(
            abandoned_preparation_is_safe=preparation_is_safe,
            recover_submission=unexpected_recovery,
        ),
    )
    if safe:
        assert manager.reserve(second).submission_phase == SubmissionPhase.PREPARING
        assert manager.store.get(first.id).state == RunState.FAILED
    else:
        with pytest.raises(DuplicateSubmissionError):
            manager.reserve(second)
        assert manager.store.get(first.id) == first
        assert second.state == RunState.NEW
    assert calls == ["preparing"]


@pytest.mark.parametrize("phase", (SubmissionPhase.PREPARING, SubmissionPhase.DISPATCHING))
def test_submission_owner_lock_prevents_recovery(tmp_path, monkeypatch, phase):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=0)
    first = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    if phase == SubmissionPhase.DISPATCHING:
        first = manager.store.begin_dispatch(first)
    second = manager.new_record(task="SCF", directory=tmp_path)
    assert first.submission_phase == phase

    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("A live submission owner must not be recovered")

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(
            abandoned_preparation_is_safe=unexpected_recovery,
            recover_submission=unexpected_recovery,
        ),
    )
    with manager.store.submission_owner(first) as acquired:
        assert acquired
        assert manager.recover_submitting(first) == first
        with pytest.raises(DuplicateSubmissionError):
            manager.reserve(second)
        assert manager.store.get(first.id) == first
    with manager.store.submission_owner(first) as acquired:
        assert acquired  # Ownership is released when its holder exits.
    assert second.state == RunState.NEW


def test_inconclusive_dispatch_recovery_keeps_reservation(tmp_path, monkeypatch):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), submitting_grace=0)
    first = _dispatch(manager, manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)

    monkeypatch.setattr(
        manager.executors, "for_endpoint",
        lambda endpoint: SimpleNamespace(recover_submission=lambda *a, **kw: None),
    )
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    assert manager.store.get(first.id) == first
    assert second.state == RunState.NEW


@pytest.mark.parametrize("payload", ("null", "[]", '"text"', "7"))
def test_malformed_run_record_json_value_is_corrupt(tmp_path, payload):
    from ase2sprkkr.execution.run_store import CorruptRunRecord

    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    record = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    store._path(record).write_text(payload)

    with pytest.raises(CorruptRunRecord, match="JSON object"):
        store.get(record.id)
    with pytest.warns(RuntimeWarning, match="Corrupt run record"):
        assert store.runs(active=True) == ()
    with pytest.raises(CorruptRunRecord, match="JSON object"):
        manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
