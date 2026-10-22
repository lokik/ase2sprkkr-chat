"""Persistent policy belongs to the manager, not the per-run lifecycle adapter."""

import asyncio

import pytest

from ase2sprkkr.execution import FailurePhase, RunState, RunStore
from ase2sprkkr.execution.manager import ExecutionManager


@pytest.fixture
def manager(tmp_path):
    return ExecutionManager(RunStore(tmp_path / "runs"))


def test_submitted_terminal_observation_first_persists_backend_identity(manager, tmp_path):
    original = manager.new_record(task="DOS", directory=tmp_path)
    reserved = manager.reserve(original)
    submitted = manager.record_submitted(
        reserved, backend_id="backend-1", state=RunState.FINISHED,
        failure_phase=None, files=reserved.files,
    )
    assert submitted.state == RunState.SUBMITTING
    assert manager.store.get(original.id).execution.backend_id == "backend-1"
    assert manager.store.get(original.id).finished_at is None


def test_definite_cancel_is_durable_without_launch(manager, tmp_path):
    record = manager.reserve(manager.new_record(task="DOS", directory=tmp_path))
    updated = manager.record_submission_failure(
        record, error=asyncio.CancelledError(), uncertain=False,
        state=RunState.CANCELLED, backend_id=None,
    )
    assert updated.state == RunState.CANCELLED
    assert manager.store.get(record.id).failure_phase is None


def test_failed_run_without_phase_is_execution_failure(manager, tmp_path):
    record = manager.new_record(task="DOS", directory=tmp_path)
    completed = manager.record_observation(
        record, state=RunState.FAILED, execution=None,
        return_code=2, message="backend failed",
        failure_phase=None, files=record.files,
        expected_updated_at=record.updated_at,
    )
    assert completed.state == RunState.FAILED
    assert completed.failure_phase == FailurePhase.EXECUTION


def test_unknown_observation_cannot_overwrite_newer_status(manager, tmp_path):
    record = manager.new_record(task="DOS", directory=tmp_path)
    newer = manager.transition(record, state=RunState.RUNNING)
    observed = manager.record_observation(
        record, state=RunState.UNKNOWN, execution=None,
        return_code=None, message="connection lost",
        failure_phase=None, files=record.files,
        expected_updated_at=record.updated_at,
    )
    assert observed == newer
    assert manager.store.get(record.id).message is None


def test_terminal_ack_retries_transport_error_without_overwriting_record(manager, tmp_path):
    record = manager.complete(
        manager.new_record(task="DOS", directory=tmp_path),
        state=RunState.FINISHED,
    )
    class Backend:
        calls = 0
        def acknowledge(self, current):
            self.calls += 1
            assert current == record
            if self.calls == 1:
                raise OSError("disconnected")
    backend = Backend()
    assert manager.acknowledge_terminal(backend, record, backend_id="backend-1") is False
    assert manager.acknowledge_terminal(backend, record, backend_id="backend-1") is True
    assert backend.calls == 2
    assert manager.store.get(record.id) == record
