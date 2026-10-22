"""File reservations are atomic claims in the run store."""

import pytest
from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import DuplicateSubmissionError, ExecutionManager, RunStore

def test_file_reservation_conflict_and_release(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.new_record(task="SCF", directory=tmp_path)
    second = manager.new_record(task="SCF", directory=tmp_path)
    claimed = manager.reserve(first)
    assert claimed.state == RunState.SUBMITTING
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    manager.transition(claimed, state=RunState.FAILED)
    assert manager.reserve(second).state == RunState.SUBMITTING

def test_failed_claim_keeps_new_run_unmodified(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    reserved, conflicts = manager.store.reserve_files(second)
    assert reserved is None
    assert [record.id for record in conflicts] == [first.id]
    assert second.state == RunState.NEW
