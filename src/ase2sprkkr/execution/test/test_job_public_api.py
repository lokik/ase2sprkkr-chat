"""Public stored-job accessors and project repository facade."""

from ase2sprkkr.execution import Job
from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.projects import ProjectRepository
from ase2sprkkr.execution.run_manager import RunManager
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


def test_job_delegates_public_actions_to_existing_services(tmp_path, monkeypatch):
    repository = ProjectRepository(tmp_path / "projects")
    project = repository.create("Actions")
    manager = repository.manager
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id,
    ))
    job = repository.job(record.id)
    observed = []

    def retrieve(value):
        observed.append(("restore", value))
        return record

    monkeypatch.setattr(manager, "retrieve_result_files", retrieve)
    assert job.restore_files() == record
    assert observed == [("restore", record.id)]

    from ase2sprkkr.outputs import project_results
    monkeypatch.setattr(
        project_results, "load_project_result",
        lambda owner, run_id: (owner, run_id),
    )
    assert job.load_result() == (manager, record.id)
    monkeypatch.setattr(
        project_results, "retry_project_postprocessing",
        lambda owner, run_id: ("retried", owner, run_id),
    )
    assert job.retry_postprocessing() == ("retried", manager, record.id)
    monkeypatch.setattr(
        project_results, "attach_project_run",
        lambda owner, run_id, **kwargs: (owner, run_id, kwargs),
    )
    assert job.attach(output_listener="listener") == (
        manager, record.id, {
            "executor": None,
            "callback": None,
            "output_listener": "listener",
        },
    )


def test_calculation_run_job_reuses_its_persistence_manager(tmp_path):
    from ase2sprkkr.execution.local import LocalRunner
    from ase2sprkkr.execution.base import ExecutionSpec
    from types import SimpleNamespace

    repository = ProjectRepository(tmp_path / "projects")
    project = repository.create("Tracked")
    manager = repository.manager
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id,
    ))
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    persistence = SimpleNamespace(manager=manager, id=record.id)
    run = CalculationRun(
        LocalRunner(spec), lambda outcome: outcome, persistence=persistence,
    )
    assert isinstance(run.job, Job)
    assert run.job.record == record


def test_uncaptured_existing_files_are_not_certified_as_job_results(tmp_path):
    from ase2sprkkr.execution import FileEntry

    repository = ProjectRepository(tmp_path / "projects")
    project = repository.create("Uncaptured history")
    input_path = tmp_path / "run.inp"
    output_path = tmp_path / "run.out"
    input_path.write_text("previous input")
    output_path.write_text("output from another run")

    files = FileManifest(
        inputs=(FileEntry(local_path=str(input_path)),),
        stdout=FileEntry(local_path=str(output_path)),
    )
    manager = repository.manager
    record = manager.store.save(manager.new_record(
        task="DOS", directory=tmp_path, project_id=project.id, files=files,
    ))
    job = repository.job(record.id)
    assert job.files.captured_at is None
    assert not job.input_is_current()
    assert not job.output_is_current()
