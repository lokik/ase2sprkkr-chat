"""Public Project facade delegates to versioned project records."""

import pytest

from ase2sprkkr.execution import Project, ProjectRecord, ProjectRepository


def test_project_facade_reads_live_record_and_persists_edits(tmp_path):
    repository = ProjectRepository(tmp_path / "projects")
    root = repository.create_project("Root")
    child = repository.create_project("Child", parent_id=root.id)
    assert isinstance(root, Project)
    assert isinstance(root.record, ProjectRecord)
    assert child.parent_id == root.id
    first = child.record
    edited = child.rename("Renamed")
    assert edited.generation == first.generation + 1
    assert child.record == edited
    assert child.name == "Renamed"
    assert child.move_to(None).parent_id is None


def test_project_facade_lists_only_its_jobs(tmp_path):
    repository = ProjectRepository(tmp_path / "projects")
    first = repository.create_project("First")
    second = repository.create_project("Second")
    manager = repository.manager
    saved = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=first.id,
    ))
    manager.store.save(manager.new_record(
        task="SCF", directory=tmp_path, project_id=second.id,
    ))
    assert [job.id for job in first.jobs()] == [saved.id]
    assert first.last_job.id == saved.id
    assert first.job(saved.id).id == saved.id
    with pytest.raises(ValueError, match="another project"):
        first.job(second.last_job.id)


def test_live_project_uses_its_own_manager_for_calculations(tmp_path):
    from ase2sprkkr.execution.persistence import resolve_project_manager

    repository = ProjectRepository(tmp_path / "projects")
    project = repository.create_project("Owned")
    manager, project_id = resolve_project_manager(project)
    assert manager is repository.manager
    assert manager.projects is repository
    assert project_id == project.id
    assert resolve_project_manager((manager, project.record)) == (manager, project_id)
