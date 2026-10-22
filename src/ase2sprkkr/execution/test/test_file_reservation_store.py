"""Atomic file reservations share the durable run record."""

import pytest

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import (
    DuplicateSubmissionError, ExecutionManager, RunStore,
)


def test_conflicting_reservation_and_release(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.new_record(task="SCF", directory=tmp_path)
    second = manager.new_record(task="SCF", directory=tmp_path)
    reserved = manager.reserve(first)
    assert reserved.state == RunState.SUBMITTING
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)
    manager.transition(reserved, state=RunState.FAILED)
    assert manager.reserve(second).state == RunState.SUBMITTING


def test_conflict_does_not_modify_new_run(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.reserve(manager.new_record(task="SCF", directory=tmp_path))
    second = manager.new_record(task="SCF", directory=tmp_path)
    reserved, conflicts = manager.store.reserve_files(second)
    assert reserved is None
    assert [x.id for x in conflicts] == [first.id]
    assert second.state == RunState.NEW
