"""Reservation protects conflicting file operations across run records."""

import pytest

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.manager import (
    DuplicateSubmissionError, ExecutionManager, RunStore,
)


def test_reservation_conflict_and_terminal_release(tmp_path):
    manager = ExecutionManager(store=RunStore(tmp_path / "history"))
    directory = tmp_path / "job"
    directory.mkdir()
    first = manager.new_record(task="SCF", directory=directory)
    second = manager.new_record(task="SCF", directory=directory)

    reserved = manager.reserve(first)
    assert manager.store.get(first.id).state == RunState.SUBMITTING
    with pytest.raises(DuplicateSubmissionError):
        manager.reserve(second)

    manager.transition(reserved, state=RunState.FAILED)
    assert manager.reserve(second).state == RunState.SUBMITTING


def test_store_conflict_leaves_other_record_unmodified(tmp_path):
    manager = ExecutionManager(store=RunStore(tmp_path / "history"))
    directory = tmp_path / "job"
    directory.mkdir()
    first = manager.reserve(manager.new_record(task="SCF", directory=directory))
    second = manager.new_record(task="SCF", directory=directory)

    reserved, conflicts = manager.store.reserve_files(second)
    assert reserved is None
    assert tuple(r.id for r in conflicts) == (first.id,)
    assert second.state == RunState.NEW
