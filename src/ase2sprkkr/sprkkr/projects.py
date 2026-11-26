"""SPR-KKR project storage policy, independent of KKRunner's defaults."""

from pathlib import Path

from platformdirs import user_state_dir

from kkrunner.projects import Project, ProjectRepository, ProjectRunManager
from kkrunner import MemoryRunStore, RunManager, RunStore
from kkrunner.projects.resolution import resolve_project_manager as _resolve_project_manager


# Default applications share one process-local registry, never a disk history.
# A GUI can instead inject its own manager for its session.
_memory_store = MemoryRunStore()


def default_project_repository():
    """Keep existing user project definitions in the ASE2SPRKKR namespace."""
    return ProjectRepository(Path(user_state_dir("ase2sprkkr")) / "projects")


def project_execution_manager(project_repository=None, *, project=None, conflict_stores=None):
    """Retain project definitions while keeping current run records separate.

    Old development RunRecord formats in the ASE2SPRKKR state directory are
    intentionally neither loaded nor migrated. Explicit repositories retain
    their own colocated run store, including isolated application/test stores.
    """
    projects = project_repository if project_repository is not None else default_project_repository()
    legacy = Path(user_state_dir("ase2sprkkr")) / "projects"
    parent = Path(user_state_dir("kkrunner")) if projects.directory == legacy else projects.directory.parent
    return ProjectRunManager(
        project_repository=projects, project=project, store=RunStore(parent / "projects_runs"),
        conflict_stores=(_memory_store,) if conflict_stores is None else conflict_stores,
    )


def memory_execution_manager():
    """Default process-local execution registry, including project conflicts."""
    persistent = project_execution_manager()
    return RunManager(_memory_store, conflict_stores=(persistent.store,))


def resolve_project_manager(project):
    """Apply application storage policy only when constructing a new manager."""
    if isinstance(project, (Project, str)):
        return project_execution_manager(project=project)
    return _resolve_project_manager(project)
