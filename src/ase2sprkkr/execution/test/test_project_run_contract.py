"""Project tracking is opt-in by project identity, never by a persistent flag."""

import pytest

from ase2sprkkr.execution.persistence import resolve_project_manager
from ase2sprkkr.execution.projects import ProjectRepository


def test_untracked_run_has_no_store():
    assert resolve_project_manager(None) is None
    assert resolve_project_manager(False) is None
    with pytest.raises(TypeError):
        resolve_project_manager(True)


def test_project_schema_has_no_run_reference_list(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("test")
    assert not hasattr(project, "persistent_refs")
    text = (tmp_path / "projects" / f"{project.id}.json").read_text()
    assert "persistent_refs" not in text

def test_project_handle_reuses_repository_manager(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Run")
    manager, project_id = resolve_project_manager(project)
    assert manager is projects.manager
    assert project_id == project.id


def test_explicit_manager_preserves_execution_configuration(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Run")
    manager, project_id = resolve_project_manager((projects.manager, project))
    assert manager is projects.manager
    assert project_id == project.id
