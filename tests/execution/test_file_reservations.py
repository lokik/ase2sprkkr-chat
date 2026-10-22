"""Reservation is a persistent file-namespace claim, not an executor slot."""

import pytest

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import (
    DuplicateSubmissionError,
    ExecutionManager,
    RunStore,
)


def test_file_reservation_blocks_conflicting_active_run(tmp_path):
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
    next_reserved = manager.reserve(second)
    assert next_reserved.state == RunState.SUBMITTING


def test_store_conflicts_are_reported_without_mutating_new_run(tmp_path):
    manager = ExecutionManager(store=RunStore(tmp_path / "history"))
    directory = tmp_path / "calculation"
    directory.mkdir()

    first = manager.reserve(manager.new_record(task="SCF", directory=directory))
    second = manager.new_record(task="SCF", directory=directory)

    reserved, conflicts = manager.store.reserve_files(second)
    assert reserved is None
    assert [r.id for r in conflicts] == [first.id]
    assert second.state == RunState.NEW
