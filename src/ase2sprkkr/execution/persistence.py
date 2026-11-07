"""Resolve explicit project-backed execution without global persistence."""

from __future__ import annotations

from .project_manager import ExecutionManager
from .projects import Project


def resolve_project_manager(project, *, result_processors=None):
    """Only an existing project enables durable run tracking."""
    if project is None or project is False:
        return None
    if isinstance(project, ExecutionManager):
        if project.project_id is None:
            raise ValueError("ExecutionManager must be bound to a project")
        for name, processor in (result_processors or {}).items():
            project.register_result_processor(name, processor)
        return project
    if isinstance(project, (Project, str)):
        return ExecutionManager(project=project, result_processors=result_processors)
    raise TypeError("project must be a Project, project ID, or project-bound ExecutionManager")
