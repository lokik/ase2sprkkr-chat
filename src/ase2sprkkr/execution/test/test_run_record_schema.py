import json
from dataclasses import replace

import pytest

from ase2sprkkr.execution import CorruptRunRecord, RunRecord, RunState, RunStore, Execution, WorkingDirectory, ExecutorEndpoint, FileManifest


def test_noncurrent_run_record_schema_is_rejected(tmp_path):
    store = RunStore(tmp_path / "runs")
    created_at = "2026-10-17T12:34:56+00:00"
    directory = store.directory / "2026-10-17"
    directory.mkdir(parents=True)
    path = directory / "foreign.json"
    path.write_text(
        json.dumps(
            {
                "id": "foreign",
                "state": "finished",
                "created_at": created_at,
                "updated_at": created_at,
                "task": "DOS",
                "directory": str(tmp_path),
                "schema_version": 4,  # Current format is 3; no legacy compatibility matrix.
            }
        )
    )

    with pytest.raises(CorruptRunRecord, match="unsupported schema version"):
        store.get("foreign")


@pytest.mark.parametrize("backend_id", (None, "backend-42"))
def test_current_schema_round_trips_with_one_execution_identity(tmp_path, backend_id):
    store = RunStore(tmp_path / "runs")
    created_at = "2026-10-17T12:34:56+00:00"
    record = RunRecord(
        id="record",
        state=RunState.RUNNING,
        created_at=created_at,
        updated_at=created_at,
        task="DOS",
        execution=Execution(
            ExecutorEndpoint(), WorkingDirectory(str(tmp_path)), backend_id,
        ),
        files=FileManifest(),
    )

    store.save(record)
    loaded = store.get(record.id)

    assert loaded == record
    assert loaded.schema_version == 3
    assert loaded.execution.backend_id == backend_id
    for obsolete in (
        "backend_state", "requested_task", "reference", "submission",
        "endpoint", "directory", "result_files",
    ):
        assert not hasattr(loaded, obsolete)


def test_persisted_execution_identity_cannot_be_replaced(tmp_path):
    from dataclasses import replace
    from ase2sprkkr.execution.manager import ExecutionManager

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = manager.new_record(task="DOS", directory=tmp_path)
    first = manager.transition(
        record, state=RunState.RUNNING,
        execution=record.execution.submitted("job-42"),
    )
    assert manager.store.get(record.id).execution.backend_id == "job-42"
    with pytest.raises(ValueError, match="Backend identity"):
        manager.transition(
            first, execution=replace(first.execution, backend_id="job-43"),
        )
    assert manager.store.get(record.id).execution.backend_id == "job-42"


def test_run_store_rejects_reused_id_with_different_creation_time(tmp_path):
    store = RunStore(tmp_path / "runs")
    created = "2026-10-17T12:34:56+00:00"
    original = RunRecord(
        id="same-id", state=RunState.RUNNING,
        created_at=created, updated_at=created, task="DOS",
        execution=Execution(ExecutorEndpoint(), WorkingDirectory(str(tmp_path))),
        files=FileManifest(),
    )
    forged = replace(
        original, created_at="2026-10-19T12:34:56+00:00",
        updated_at="2026-10-19T12:34:56+00:00",
    )
    store.save(original)
    with pytest.raises(ValueError, match="persistent creation time"):
        store.save(forged)
    (store.directory / "unrelated-file.txt").write_text("not a shard")

    assert store.get("same-id") == original


def test_run_store_lookup_preserves_missing_directory_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing-run"):
        RunStore(tmp_path / "absent").get("missing-run")
