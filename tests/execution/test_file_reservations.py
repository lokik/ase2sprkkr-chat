"""Persistent run file reservations."""

import pytest

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import DuplicateSubmissionError, ExecutionManager, RunStore


def test_conflicting_files_block_second_submission_until_terminal(tmp_path):
    manager = ExecutionManager(store=RunStore(tmp_path / "history"))
    directory = tmp_path / "calculation"
    directory.mkdir()
    first = manager.new_record(task="SCF", directory=directory)
    second = manager.new_record(task="SCF", directory=directory)

    reserved = manager.reserve(first)
    assert reserved.state == RunState.SUBMITTING
    assert manager.store.get(first.id).state == RunState.SUBMITTING
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)

    manager.transition(reserved, state=RunState.FAILED)
    assert manager.reserve(second).state == RunState.SUBMITTING


def test_conflicting_reservation_does_not_change_record(tmp_path):
    manager = ExecutionManager(store=RunStore(tmp_path / "history"))
    directory = tmp_path / "calculation"
    directory.mkdir()
    first = manager.reserve(manager.new_record(task="SCF", directory=directory))
    second = manager.new_record(task="SCF", directory=directory)

    reserved, conflicts = manager.store.reserve_files(second)
    assert reserved is None
    assert [r.id for r in conflicts] == [first.id]
    assert second.state == RunState.NEW
