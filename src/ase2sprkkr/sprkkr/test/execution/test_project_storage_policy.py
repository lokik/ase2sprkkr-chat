"""Existing project definitions and new runner records have independent storage."""

from kkrunner.projects import ProjectRepository

from ase2sprkkr.sprkkr.projects import default_project_repository, project_execution_manager, resolve_project_manager


def test_default_projects_remain_visible_without_loading_old_development_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    projects = ProjectRepository(tmp_path / "ase2sprkkr/projects")
    project = projects.create("Existing", directory=tmp_path)
    old = tmp_path / "ase2sprkkr/projects_runs/2026-01-01/old.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"schema_version":3}')

    manager = resolve_project_manager(project.id)
    assert manager.project_id == project.id
    assert default_project_repository().project(project.id) == project
    assert manager.store.directory == tmp_path / "kkrunner/projects_runs"
    assert manager.store.runs() == ()
    assert old.read_text() == '{"schema_version":3}'


def test_explicit_project_repository_keeps_its_explicit_store(tmp_path):
    projects = ProjectRepository(tmp_path / "custom/projects")
    manager = project_execution_manager(projects)
    assert manager.store.directory == tmp_path / "custom/projects_runs"


def test_default_memory_manager_and_projects_share_conflicts(tmp_path, monkeypatch):
    from ase2sprkkr.sprkkr.projects import memory_execution_manager
    from kkrunner import MemoryRunStore

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    memory = memory_execution_manager()
    projects = project_execution_manager()
    assert isinstance(memory.store, MemoryRunStore)
    assert projects.conflict_stores == (memory.store,)
    assert memory.conflict_stores[0].lock_key == projects.store.lock_key
    assert memory_execution_manager().store is memory.store
