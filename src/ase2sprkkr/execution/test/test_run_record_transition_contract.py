"""RunRecord owns pure transition rules; RunStore only commits them."""

from dataclasses import replace

import pytest

from ase2sprkkr.execution import (
    ExecutionManager, FileEntry, FileManifest, RunState, RunStore,
    WorkingDirectory,
)
from ase2sprkkr.execution.base import FailurePhase


def _record(tmp_path, *, files=None):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    return manager.new_record(
        task="DOS", directory=tmp_path, files=files,
    )


def test_run_record_transition_does_not_mutate_old_snapshots(tmp_path):
    original = _record(tmp_path)
    queued = original.with_transition(state=RunState.QUEUED)
    running = queued.with_transition(state=RunState.RUNNING, return_code=0)
    assert original.state is RunState.NEW
    assert queued.state is RunState.QUEUED
    assert running.state is RunState.RUNNING
    assert running.return_code == 0
    assert original.return_code is None


def test_run_record_transition_guards_stale_observations(tmp_path):
    original = _record(tmp_path)
    running = original.with_transition(state=RunState.RUNNING)
    stale = running.with_transition(
        state=RunState.UNKNOWN, message="stale connection failure",
        expected_updated_at="known older timestamp",
    )
    assert stale is running
    processing = running.with_transition(state=RunState.POSTPROCESSING)
    assert processing.with_transition(
        state=RunState.UNKNOWN, message="stale observation",
    ) is processing


def test_run_record_transition_uses_latest_file_failure_on_completion(tmp_path):
    path = str(tmp_path / "result.dat")
    files = FileManifest(outputs=(
        FileEntry(path, availability_state="failed",
                  availability_message="download failed"),
    ))
    record = _record(tmp_path, files=files)
    claimed = FileManifest(
        outputs=(FileEntry(path, availability_state="available"),),
        captured_at="2026-10-22T08:00:00+00:00",
    )
    complete = record.with_transition(
        state=RunState.FINISHED, files=claimed, return_code=0,
        validate_completed_files=True,
    )
    assert complete.state is RunState.FAILED
    assert complete.failure_phase is FailurePhase.DOWNLOAD
    assert complete.files.failed_files
    assert complete.files.captured_at is None
    assert record.files == files


def test_run_record_rejects_changed_execution_identity(tmp_path):
    original = _record(tmp_path)
    changed = replace(
        original.execution,
        directory=WorkingDirectory(str(tmp_path / "other")),
    )
    with pytest.raises(ValueError, match="local directory"):
        original.with_transition(execution=changed)


def test_run_record_terminal_status_is_not_reversed_by_stale_result(tmp_path):
    record = _record(tmp_path)
    finished = record.with_transition(state=RunState.FINISHED, return_code=0)
    stale = finished.with_transition(state=RunState.FAILED, return_code=1)
    assert stale.state is RunState.FINISHED
    assert stale.return_code == 0


@pytest.mark.parametrize("observation", (
    RunState.FINISHED, RunState.FAILED, RunState.CANCELLED,
))
def test_terminal_record_ignores_late_backend_observation(tmp_path, observation):
    record = _record(tmp_path)
    finished = record.with_transition(
        state=RunState.FINISHED, return_code=0, message="complete",
        observed=True,
    )
    stale = finished.with_transition(
        state=observation, return_code=2, message="stale backend status",
        failure_phase=FailurePhase.EXECUTION, observed=True,
    )
    assert stale is finished
    assert stale.updated_at == finished.updated_at
    assert stale.last_observed == finished.last_observed
    assert stale.return_code == 0
    assert stale.message == "complete"


def test_manager_transition_commits_value_object_through_store(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "record-store"))
    original = manager.store.save(manager.new_record(task="DOS", directory=tmp_path))
    final = manager.transition(original, state=RunState.RUNNING)
    assert final == manager.store.get(original.id)
    assert original.state is RunState.NEW
