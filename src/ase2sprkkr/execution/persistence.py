"""Resolve a project identity and one shared execution manager."""
from __future__ import annotations

from .run_manager import RunManager
from .pool import Executors
from .projects import Project, ProjectRepository


_nonproject_executors = Executors()


def default_executors():
    """Process-wide registry for non-project calculations."""
    return _nonproject_executors


def resolve_project_manager(project):
    """Return (manager, project_id); no manager is bound to one project."""
    if project is None or project is False:
        return None
    if isinstance(project, tuple) and len(project) == 2:
        manager, project = project
        if not isinstance(manager, RunManager) or manager.projects is None:
            raise TypeError("Project execution requires an RunManager with a repository")
    elif isinstance(project, (Project, str)):
        manager = RunManager(project_repository=ProjectRepository())
    else:
        raise TypeError("project must be a Project, project ID, or (RunManager, project)")
    project_id = project.id if isinstance(project, Project) else str(project)
    return manager, manager.projects.project(project_id).id
