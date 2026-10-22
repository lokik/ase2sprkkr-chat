"""Public persisted project/job handles are identity-bound, not snapshots."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import (
    CalculationRun, Job, Project, ProjectRecord, ProjectRepository,
    RunManager, RunRecord, RunFiles, RunState,
)


def test_project_repository_exposes_one_lazy_manager(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Root")

    assert isinstance(project, Project)
    assert isinstance(project.record, ProjectRecord)
    assert isinstance(projects.manager, RunManager)
    assert projects.manager is projects.manager
    assert projects.manager.store is projects.run_store
    assert project.last_job() is None
    assert project.jobs() == ()


def test_job_history_is_scoped_to_its_project_and_reopens(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    a, b = projects.create("A"), projects.create("B")
    manager = projects.manager
    earlier = manager.new_record(task="DOS", directory=tmp_path, project_id=a.id)
    later = manager.new_record(task="ARPES", directory=tmp_path, project_id=a.id)
    other = manager.new_record(task="SCF", directory=tmp_path, project_id=b.id)
    manager.store.save(replace(earlier, created_at="2026-10-20T09:00:00+00:00"))
    manager.store.save(replace(later, created_at="2026-10-21T09:00:00+00:00"))
    manager.store.save(replace(other, created_at="2026-10-22T09:00:00+00:00"))

    assert [job.id for job in a.jobs()] == [later.id, earlier.id]
    assert a.last_job().id == later.id
    assert isinstance(a.last_job(), Job)
    assert projects.job(earlier.id).id == earlier.id
    with pytest.raises(KeyError):
        a.job(other.id)

    reopened = ProjectRepository(projects.directory)
    job = reopened.project(a.id).last_job()
    assert job.id == later.id
    assert job.record.task == "ARPES"
    assert isinstance(job.record, RunRecord)
    assert job.files == job.record.files
    assert job.input_is_current() is False
    assert job.output_is_current() is False


def test_job_restores_existing_terminal_history_without_resubmit(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Root")
    manager = projects.manager
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id,
    ))
    completed = manager.complete(record, state=RunState.FINISHED, return_code=0)
    job = project.last_job()

    assert job.refresh() == completed
    assert job.restore_files() == completed
    assert job.state is RunState.FINISHED


def test_calculation_run_job_handle_requires_persistence(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Root")
    manager = projects.manager
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id,
    ))

    calculation = CalculationRun.__new__(CalculationRun)
    calculation.persistence = None
    assert calculation.job is None
    calculation.persistence = SimpleNamespace(manager=manager, id=record.id)
    assert isinstance(calculation.job, Job)
    assert calculation.job.record == record
    assert calculation.job.id == record.id


def test_run_files_are_not_the_stored_job(tmp_path):
    from ase2sprkkr.execution.run_model import FileManifest
    working = RunFiles(FileManifest())
    assert isinstance(working.manifest, FileManifest)
    assert not hasattr(Job, "run_files")


def test_job_user_actions_delegate_with_current_identity(tmp_path, monkeypatch):
    from ase2sprkkr.outputs import project_results

    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Root")
    record = projects.manager.store.save(projects.manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id,
    ))
    job = project.job(record.id)
    calls = []

    def fake_load(manager, run):
        calls.append(("load", manager, run))
        return "loaded"

    def fake_attach(manager, run, *, executor=None, callback=None, output_listener=None):
        calls.append(("attach", manager, run, executor, callback, output_listener))
        return "attached"

    def fake_retry(manager, run):
        calls.append(("retry", manager, run))
        return "reparsed"

    monkeypatch.setattr(project_results, "load_project_result", fake_load)
    monkeypatch.setattr(project_results, "attach_project_run", fake_attach)
    monkeypatch.setattr(project_results, "retry_project_postprocessing", fake_retry)

    callback = lambda _: None
    output_listener = lambda _: None
    executor = object()
    assert job.load_result() == "loaded"
    assert job.attach(
        executor=executor, callback=callback, output_listener=output_listener,
    ) == "attached"
    assert job.retry_postprocessing() == "reparsed"
    assert calls == [
        ("load", projects.manager, record.id),
        ("attach", projects.manager, record.id, executor, callback, output_listener),
        ("retry", projects.manager, record.id),
    ]
