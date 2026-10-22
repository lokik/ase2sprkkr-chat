"""Atomic file namespace reservation for persistent execution."""

from dataclasses import replace

import pytest

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import (
    ExecutionManager, RunStore,
)


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
    from types import SimpleNamespace

    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store, submitting_grace=0)
    record = manager.new_record(task="DOS", directory=tmp_path)
    submitted = manager.reserve(record)

    def recover(*args, **kwargs):
        manager.transition(submitted, state=RunState.SUBMITTING)
        return None

    monkeypatch.setattr(
        manager.executors, "for_record",
        lambda record: SimpleNamespace(recover_submission=recover),
    )
    result = manager.recover_submitting(submitted)
    assert result.state == RunState.SUBMITTING
    assert store.get(submitted.id).state == RunState.SUBMITTING


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

    def unexpected_sync(self):
        raise AssertionError("Stale run must not sync inputs")

    from ase2sprkkr.execution.job_files import JobFiles
    monkeypatch.setattr(JobFiles, "sync_inputs", unexpected_sync)
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
