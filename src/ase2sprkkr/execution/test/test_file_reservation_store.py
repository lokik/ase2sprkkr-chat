"""Atomic file reservations use the durable RunStore."""

import pytest

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import DuplicateSubmissionError, ExecutionManager, RunStore


def test_conflicting_submission_claim_and_release(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.new_record(task="SCF", directory=tmp_path)
    second = manager.new_record(task="SCF", directory=tmp_path)
    claimed = manager.reserve(first)
    assert claimed.state == RunState.SUBMITTING
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    manager.transition(claimed, state=RunState.FAILED)
    assert manager.reserve(second).state == RunState.SUBMITTING


def test_conflicting_claim_does_not_change_new_run(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    claimed, conflicts = manager.store.reserve_files(second)
    assert claimed is None
    assert [r.id for r in conflicts] == [first.id]
    assert second.state == RunState.NEW
