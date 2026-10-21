"""Global run history and cross-project namespace regression tests."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from uuid import UUID

import pytest

from ..base import RunState
from ...common.file_locks import FileLock
from ..manager import DuplicateSubmissionError, RunStore
from ..project_manager import ExecutionManager
from ..projects import ProjectRepository, ProjectDeleteError


def test_different_projects_share_one_store_and_keep_project_identity(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    first = projects.create("First", parent_id=root.id)
    second = projects.create("Second", parent_id=root.id)
    a = ExecutionManager(project_repository=projects, project=first)
    b = ExecutionManager(project_repository=projects, project=second)
    assert a.store.directory == b.store.directory
    first_run = a.new_record(task="DOS", directory=tmp_path / "work", dataset="same")
    second_run = b.new_record(task="DOS", directory=tmp_path / "work", dataset="same")
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
    first_record = a.new_record(task="SCF", directory=tmp_path / "work")
    second_record = b.new_record(task="DOS", directory=tmp_path / "work")
    a.reserve(first_record)
    with pytest.raises(DuplicateSubmissionError):
        b.reserve(second_record)
    with pytest.raises(ProjectDeleteError, match="active"):
        projects.delete(first.id, expected_generation=first.generation)
    a.complete(first_record, state=RunState.FAILED)
    projects.delete(first.id, expected_generation=first.generation)
    assert all(r.project_id != first.id for r in b.store.runs())


def test_for_project_preserves_explicit_run_store(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Project")
    custom_store = RunStore(tmp_path / "custom-runs")
    original = ExecutionManager(
        store=custom_store, project_repository=projects, submitting_grace=17,
    )

    bound = original.for_project(project)

    assert bound.store is custom_store
    assert bound.submitting_grace == 17
    assert bound.executor_pool is original.executor_pool
    assert bound.for_project(project) is bound

    record = bound.new_record(task="DOS", directory=tmp_path / "work")
    bound.store.save(record)

    assert custom_store.get(record.id) == record
    assert record.project_id == project.id
    assert not (tmp_path / "projects_runs").exists()


def test_for_project_without_explicit_store_uses_shared_project_store(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Project")
    unbound = ExecutionManager(project_repository=projects)

    bound = unbound.for_project(project)

    assert unbound.store is None
    assert bound.project_id == project.id
    assert bound.store.directory == tmp_path / "projects_runs"



def test_unknown_run_prevents_project_deletion_until_terminal(tmp_path):
    """UNKNOWN is recoverable, so deleting its project must not erase history."""
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    project = projects.create("Child", parent_id=root.id)
    manager = ExecutionManager(project_repository=projects, project=project)
    record = replace(
        manager.new_record(task="DOS", directory=tmp_path / "work"),
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


def test_default_project_store_is_guarded_and_shared(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Project")
    first = ExecutionManager(project_repository=projects, project=project)
    second = ExecutionManager(project_repository=projects, project=project)
    assert first.store is second.store is projects.run_store
    assert first.store.projects_directory == projects.directory
    assert first.store.directory == projects.runs_directory

    record = first.new_record(task="DOS", directory=tmp_path / "work")
    assert len(record.project_id) == 32  # The repository stores UUID.hex filenames.
    first.store.save(record)
    assert first.store.get(record.id) == record
    assert (projects.directory / f"{project.id}.json").is_file()
    assert (projects.directory / f".runs-{project.id}.lock").is_file()
    assert not (projects.directory / f".runs-{str(UUID(project.id))}.lock").exists()


def test_deleted_project_cannot_resurrect_run_history(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    child = projects.create("Child", parent_id=root.id)
    manager = ExecutionManager(project_repository=projects, project=child)
    other = ExecutionManager(
        project_repository=ProjectRepository(projects.directory), project=child,
    )
    record = manager.new_record(task="DOS", directory=tmp_path / "work")
    manager.store.save(record)
    manager.complete(record, state=RunState.FAILED)
    projects.delete(child.id, expected_generation=child.generation)
    for store in (manager.store, other.store):
        with pytest.raises(KeyError, match="no longer exists"):
            store.save(record)
        with pytest.raises(KeyError, match="no longer exists"):
            store.update(record, lambda latest: latest)
        assert store.runs() == ()


def test_project_deletion_waits_for_in_progress_run_update(tmp_path, monkeypatch):
    projects = ProjectRepository(tmp_path / "projects")
    root = projects.create("Root")
    child = projects.create("Child", parent_id=root.id)
    manager = ExecutionManager(project_repository=projects, project=child)
    record = manager.new_record(task="DOS", directory=tmp_path / "work")
    manager.store.save(record)
    started, release, deletion_at_guard = Event(), Event(), Event()
    original_write = manager.store._write_unlocked

    def interrupted_write(value):
        started.set()
        assert release.wait(10)
        return original_write(value)

    monkeypatch.setattr(manager.store, "_write_unlocked", interrupted_write)
    independent_projects = ProjectRepository(projects.directory)
    guard_path = projects.directory / f".runs-{child.id}.lock"
    original_file_lock = independent_projects._file_lock

    def observe_deletion_lock(path):
        if path == guard_path:
            deletion_at_guard.set()
        return original_file_lock(path)

    monkeypatch.setattr(independent_projects, "_file_lock", observe_deletion_lock)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(
            manager.store.update, record,
            lambda latest: replace(latest, state=RunState.FAILED),
        )
        try:
            assert started.wait(10)
            deletion = pool.submit(
                independent_projects.delete, child.id,
                expected_generation=child.generation,
            )
            assert deletion_at_guard.wait(10)
            # An explicit non-blocking probe proves that the writer holds the
            # same guard that deletion is attempting to enter. No scheduling-
            # dependent sleep or short Future timeout is needed.
            probe = FileLock(guard_path)
            try:
                assert not probe.acquire(blocking=False)
                assert not deletion.done()
            finally:
                probe.release()
        finally:
            release.set()
        assert writer.result(timeout=10).state is RunState.FAILED
        deletion.result(timeout=10)
    assert manager.store.runs() == ()

def test_guarded_run_store_rejects_noncanonical_project_identity(tmp_path):
    """An alias must not persist a run that deletion would fail to recognize."""
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Project")
    manager = ExecutionManager(project_repository=projects, project=project)
    record = manager.new_record(task="DOS", directory=tmp_path / "work")
    aliased = replace(record, project_id=str(UUID(project.id)))
    with pytest.raises(ValueError, match="canonical project ID"):
        manager.store.save(aliased)
    with pytest.raises(ValueError, match="canonical project ID"):
        manager.store.update(aliased, lambda latest: latest)
    assert manager.store.runs() == ()

    # Standalone stores are not coupled to a ProjectRepository.
    custom = RunStore(tmp_path / "custom-runs")
    custom.save(aliased)
    assert custom.get(aliased.id) == aliased


def test_guarded_run_store_rejects_missing_or_foreign_project_id(tmp_path):
    """Stale caller metadata must not bypass the persisted run's guard."""
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("First")
    another = projects.create("Second")
    manager = ExecutionManager(project_repository=projects, project=project)
    original = manager.new_record(task="DOS", directory=tmp_path / "work")
    manager.store.save(original)

    for bad_id in (None, another.id):
        forged = replace(original, project_id=bad_id)
        with pytest.raises(ValueError, match="project ID|project identity"):
            manager.store.update(forged, lambda latest: latest)
        with pytest.raises(ValueError, match="project ID|project identity"):
            manager.store.save(forged)
        assert manager.store.get(original.id) == original

    with pytest.raises(ValueError, match="canonical project ID"):
        manager.store.save(replace(original, project_id=None))
    assert manager.store.get(original.id) == original


def test_guarded_run_identity_across_day_shards(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    first = projects.create("First")
    second = projects.create("Second")
    store = projects.run_store
    manager = ExecutionManager(project_repository=projects, project=first)
    original = manager.new_record(task="DOS", directory=tmp_path / "work")
    store.save(original)

    next_day = "2030-01-02T10:00:00+00:00"
    earlier = replace(
        original, created_at="2030-01-01T10:00:00+00:00",
        updated_at="2030-01-01T10:00:00+00:00",
    )
    # Duplicate IDs within one project retain the historical newest-first
    # semantics; only crossing project boundaries is forbidden.
    store.save(earlier)
    assert store.get(original.id) == earlier if (
        original.created_at < earlier.created_at
    ) else store.get(original.id) == original

    foreign = replace(
        original, created_at=next_day, updated_at=next_day,
        project_id=second.id,
    )
    independent = ProjectRepository(projects.directory).run_store
    for action in (
        lambda: independent.save(foreign),
        lambda: independent.update(foreign, lambda current: current),
    ):
        with pytest.raises(ValueError, match="persistent project identity"):
            action()
    assert not independent._path(foreign).exists()
    assert {r.project_id for r in independent.runs()} == {first.id}


def test_guarded_run_cross_day_save_serializes_across_store_instances(
    tmp_path, monkeypatch,
):
    projects = ProjectRepository(tmp_path / "projects")
    first = projects.create("First")
    second = projects.create("Second")
    a = projects.run_store
    b = ProjectRepository(projects.directory).run_store
    record = ExecutionManager(
        project_repository=projects, project=first,
    ).new_record(task="DOS", directory=tmp_path / "work")
    foreign = replace(
        record, created_at="2030-01-02T10:00:00+00:00",
        updated_at="2030-01-02T10:00:00+00:00",
        project_id=second.id,
    )
    entered, release = Event(), Event()
    original_write = a._write_unlocked

    def blocked_write(value):
        entered.set()
        assert release.wait(10)
        return original_write(value)

    monkeypatch.setattr(a, "_write_unlocked", blocked_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(a.save, record)
        try:
            assert entered.wait(10)
            probe = FileLock(a._run_lock_path(record))
            try:
                assert not probe.acquire(blocking=False)
                future = pool.submit(b.save, foreign)
                assert not future.done()
            finally:
                probe.release()
        finally:
            release.set()
        assert writer.result(timeout=10) == record
        with pytest.raises(ValueError, match="persistent project identity"):
            future.result(timeout=10)
    assert len(a.runs()) == 1
    assert a.get(record.id) == record
