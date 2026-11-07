"""Persisted result processors are selected without SPR-KKR dependencies."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import (
    ExecutionManager, ExecutionOutcome, FileEntry, FileManifest,
    ProjectRepository, RunState, RunStore,
)


class SyntheticResultProcessor:
    def __init__(self):
        self.calls = []

    def process(self, record, outcome):
        self.calls.append((record.id, outcome.output_file, outcome.return_code))
        return SimpleNamespace(record=None, contents=open(outcome.output_file).read())

    def record_updated(self, result, record):
        result.record = record


def _completed_run(tmp_path, processor_name="synthetic"):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Test")
    processor = SyntheticResultProcessor()
    manager = ExecutionManager(
        store=RunStore(tmp_path / "runs"), project_repository=projects,
        project=project, result_processors={"synthetic": processor},
    )
    output = tmp_path / "completed.out"
    output.write_text("calculation output")
    record = manager.new_record(
        task="OTHER", directory=tmp_path, result_processor=processor_name,
        files=FileManifest(stdout=FileEntry(local_path=str(output))),
    )
    manager.store.save(record)
    record = manager.complete(record, state=RunState.FINISHED, return_code=0,
                              artifacts_available=True)
    return manager, record, processor


def test_generic_processor_loads_and_binds_historical_result(tmp_path):
    manager, record, processor = _completed_run(tmp_path)
    result = manager.load_result(record.id)
    assert result.contents == "calculation output"
    assert result.record == record
    assert processor.calls == [(record.id, str(tmp_path / "completed.out"), 0)]


def test_registry_is_shared_by_project_views(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Test")
    processor = SyntheticResultProcessor()
    unbound = ExecutionManager(
        project_repository=projects, result_processors={"synthetic": processor}
    )
    bound = unbound.for_project(project)
    assert bound._result_processors is unbound._result_processors
    assert bound._result_processors["synthetic"] is processor


def test_unknown_or_missing_processor_fails_explicitly(tmp_path):
    manager, record, _ = _completed_run(tmp_path, processor_name="other")
    with pytest.raises(LookupError, match="not registered"):
        manager.load_result(record.id)

    absent = replace(record, result_processor=None)
    with pytest.raises(ValueError, match="no persisted result processor"):
        manager._result_processor_for(absent)


def test_current_record_schema_rejects_old_version(tmp_path):
    manager, record, _ = _completed_run(tmp_path)
    with pytest.raises(ValueError, match="unsupported schema version"):
        replace(record, schema_version=3)


def test_attach_uses_persisted_processor_without_task_specific_api(
    monkeypatch, tmp_path
):
    from ase2sprkkr.execution import project_manager

    manager, record, processor = _completed_run(tmp_path)
    captured = {}
    runner = object()

    def raw_attach(self, run, *, process_result, executor=None,
                   callback=None, output_listener=None):
        captured.update(run=run, process=process_result, callback=callback)
        return runner

    monkeypatch.setattr(project_manager._ExecutionManager, "attach", raw_attach)
    notified = []
    assert manager.attach(record.id, callback=notified.append) is runner
    result = captured["process"](
        ExecutionOutcome(
            input_file=None, output_file=str(tmp_path / "completed.out"),
            error_file=None, return_code=0,
        )
    )
    assert result.record is None
    captured["callback"](result)
    assert result.record == record
    assert notified == [result]
    assert processor.calls == [(record.id, str(tmp_path / "completed.out"), 0)]


def test_unregistered_processor_cannot_attach(tmp_path):
    manager, record, _ = _completed_run(tmp_path, processor_name="other")
    with pytest.raises(LookupError, match="not registered"):
        manager.attach(record.id)
