"""Protect immutable file snapshots from conflicting backend observations."""

from dataclasses import replace

import pytest

from ase2sprkkr.execution.manager import ExecutionManager, RunStore
from ase2sprkkr.execution.run_model import FileManifest


def _captured_run(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = manager.new_record(task="DOS", directory=tmp_path)
    record = replace(record, files=FileManifest(captured_at="2026-10-22T00:00:00+00:00"))
    return manager, manager.store.save(record)


def test_conflicting_captured_manifest_update_is_rejected(tmp_path):
    manager, record = _captured_run(tmp_path)
    with pytest.raises(ValueError, match="Cannot replace a captured file manifest"):
        manager.transition(record, files=FileManifest())
    assert manager.store.get(record.id).files == record.files


def test_identical_captured_manifest_update_is_idempotent(tmp_path):
    manager, record = _captured_run(tmp_path)
    updated = manager.transition(record, files=record.files)
    assert updated.files == record.files


def test_stale_observation_is_filtered_before_manifest_conflict(tmp_path):
    manager, record = _captured_run(tmp_path)
    updated = manager.transition(
        record, files=FileManifest(), expected_updated_at="outdated timestamp",
    )
    assert updated == record
    assert manager.store.get(record.id) == record
