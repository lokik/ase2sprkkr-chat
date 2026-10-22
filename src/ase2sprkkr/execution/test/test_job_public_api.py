"""Public stored-job accessors and project repository facade."""

from ase2sprkkr.execution import Job
from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.projects import ProjectRepository
from ase2sprkkr.execution.manager import ExecutionManager
from ase2sprkkr.execution.postprocessing import CalculationRun
from ase2sprkkr.execution.run_model import FileManifest


def test_repository_last_job_filters_project_before_selecting_latest(tmp_path):
    repository = ProjectRepository(tmp_path / "projects")
    first = repository.create("First")
    second = repository.create("Second")
    manager = repository.manager
    older = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=first.id,
    ))
    manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=second.id,
    ))
    newer = manager.store.save(manager.new_record(
        task="SCF", directory=tmp_path, project_id=first.id,
    ))
    assert repository.manager is manager
    assert tuple(x.id for x in repository.jobs(first.id)) == (newer.id, older.id)
    assert repository.last_job(first.id).id == newer.id
    assert repository.job(older.id).id == older.id


def test_job_refresh_reads_latest_record_and_file_integrity(tmp_path):
    repository = ProjectRepository(tmp_path / "projects")
    project = repository.create("History")
    manager = repository.manager
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id,
    ))
    job = repository.job(record.id)
    assert isinstance(job, Job)
    assert job.record == record
    assert job.files == FileManifest()
    assert not job.input_is_current()
    assert not job.output_is_current()
    updated = manager.store.update(
        record,
        lambda current: current.with_transition(state=RunState.RUNNING),
    )
    assert job.refresh() == updated
    assert job.state is RunState.RUNNING


def test_untracked_calculation_has_no_job(tmp_path):
    from ase2sprkkr.execution.local import LocalRunner
    from ase2sprkkr.execution.base import ExecutionSpec
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    run = CalculationRun(LocalRunner(spec), lambda outcome: outcome)
    assert run.job is None
