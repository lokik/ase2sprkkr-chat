"""Resolve a project identity and one shared execution manager."""
from __future__ import annotations

from .run_manager import RunManager
from .pool import Executors
from .project import Project
from .projects import ProjectRecord, ProjectRepository


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
            raise TypeError("Project execution requires a RunManager with a repository")
    elif isinstance(project, Project):
        # A live Project already knows its repository; using the default
        # repository would silently move a run into another namespace.
        manager = project.manager
    elif isinstance(project, (ProjectRecord, str)):
        manager = RunManager(project_repository=ProjectRepository())
    else:
        raise TypeError(
            "project must be a Project, ProjectRecord, project ID, "
            "or (RunManager, project)"
        )
    project_id = (
        project.id if isinstance(project, (Project, ProjectRecord))
        else str(project)
    )
    return manager, manager.projects.project(project_id).id
