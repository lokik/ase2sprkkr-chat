"""Global run history and cross-project namespace regression tests."""

from dataclasses import replace

import pytest

from ..base import RunState
from ..manager import DuplicateSubmissionError, RunStore
from ..manager import ExecutionManager
from ..projects import ProjectRepository, ProjectDeleteError


def test_different_projects_share_one_store_and_keep_project_identity(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    first = projects.create("First", parent_id=root.id)
    second = projects.create("Second", parent_id=root.id)
    a = ExecutionManager(project_repository=projects)
    b = ExecutionManager(project_repository=projects)
    assert a.store.directory == b.store.directory
    first_run = a.new_record(task="DOS", directory=tmp_path / "work", dataset="same", project_id=first.id)
    second_run = b.new_record(task="DOS", directory=tmp_path / "work", dataset="same", project_id=second.id)
    assert first_run.project_id == first.id
    assert second_run.project_id == second.id
    a.store.save(first_run)
    b.store.save(second_run)
    assert {r.project_id for r in a.store.runs()} == {first.id, second.id}


def test_cross_project_active_conflict_and_project_deletion(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    first = projects.create("First", parent_id=root.id)
    second = projects.create("Second", parent_id=root.id)
    a = ExecutionManager(project_repository=projects, project=first)
    b = ExecutionManager(project_repository=projects, project=second)
    first_record = a.new_record(task="SCF", directory=tmp_path / "work", project_id=first.id)
    second_record = b.new_record(task="DOS", directory=tmp_path / "work", project_id=second.id)
    a.reserve(first_record)
    with pytest.raises(DuplicateSubmissionError):
        b.reserve(second_record)
    with pytest.raises(ProjectDeleteError, match="active"):
        projects.delete(first.id, expected_generation=first.generation)
    a.complete(first_record, state=RunState.FAILED)
    projects.delete(first.id, expected_generation=first.generation)
    assert all(r.project_id != first.id for r in b.store.runs())


def test_explicit_project_uses_shared_store_and_validates_membership(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Project")
    custom_store = RunStore(tmp_path / "custom-runs")
    manager = ExecutionManager(
        store=custom_store, project_repository=projects, submitting_grace=17,
    )
    with pytest.raises(ValueError, match="project_id"):
        manager.new_record(task="DOS", directory=tmp_path / "work")
    record = manager.new_record(
        task="DOS", directory=tmp_path / "work", project_id=project.id,
    )
    manager.store.save(record)
    assert manager.store is custom_store
    assert record.project_id == project.id
    assert custom_store.get(record.id) == record
    with pytest.raises(Exception):
        manager.new_record(task="DOS", directory=tmp_path / "work", project_id="nonexistent")


def test_default_project_store_shared_across_projects(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    one, two = projects.create("First"), projects.create("Second")
    manager = ExecutionManager(project_repository=projects)
    first = manager.new_record(task="DOS", directory=tmp_path / "a", project_id=one.id)
    second = manager.new_record(task="DOS", directory=tmp_path / "b", project_id=two.id)
    manager.store.save(first)
    manager.store.save(second)
    assert manager.store.directory == tmp_path / "projects_runs"
    assert {run.project_id for run in manager.store.runs()} == {one.id, two.id}


def test_unknown_run_prevents_project_deletion_until_terminal(tmp_path):
    """UNKNOWN is recoverable, so deleting its project must not erase history."""
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    project = projects.create("Child", parent_id=root.id)
    manager = ExecutionManager(project_repository=projects)
    record = replace(
        manager.new_record(task="DOS", directory=tmp_path / "work", project_id=project.id),
        state=RunState.UNKNOWN,
    )
    manager.store.save(record)

    with pytest.raises(ProjectDeleteError, match="active"):
        projects.delete(project.id, expected_generation=project.generation)
    assert manager.store.get(record.id).state == RunState.UNKNOWN

    manager.store.update(
        record, lambda latest: replace(latest, state=RunState.FINISHED),
    )
    projects.delete(project.id, expected_generation=project.generation)
    assert manager.store.runs() == ()
