"""The public run factory freezes execution before creating file state."""
from ase2sprkkr.execution import (
    CalculationRun, ExecutionSpec, RunManager, Executors, FileManifest,
    RunFiles, LaunchSpec, ProjectRepository, RunState,
)
from ase2sprkkr.execution.base import Executor
from ase2sprkkr.execution.local_process import select_mpi_spec


class _Raw:
    def __init__(self, executor, spec):
        self.executor = executor
        self.spec = spec
        self.callback = None
        self.state = RunState.NEW
        self.job_files = RunFiles(FileManifest.from_spec(spec))


class _Executor(Executor):
    name = "fixture"

    def resolve_spec(self, spec):
        return select_mpi_spec(spec, ("mpirun",))

    def create_run(self, spec, callback=None):
        return _Raw(self, self.resolve_spec(spec))

    def create_project_run(self, spec, *, run_id, callback=None):
        return _Raw(self, self.resolve_spec(spec))


def _spec(tmp_path):
    return ExecutionSpec(
        argv=("serial",), mpi_argv=("parallel",),
        stdin="serial.in", mpi_stdin="parallel.in",
        input_file=None, output_file=None, directory=str(tmp_path),
        launch=LaunchSpec.from_mpi(True),
    )


def test_untracked_factory_chooses_only_mpi_input(tmp_path):
    run = CalculationRun.create(
        _spec(tmp_path), executor=_Executor(), process_result=lambda value: value,
    )
    assert run.spec.argv == ("mpirun", "parallel")
    assert run.spec.stdin == "parallel.in"
    assert run.job_files.manifest.input_files == (str(tmp_path / "parallel.in"),)


def test_project_factory_reserves_only_chosen_input(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("MPI factory test")
    manager = RunManager(project_repository=projects, executors=Executors())
    run = CalculationRun.create(
        _spec(tmp_path), executor=_Executor(), project=(manager, project.id),
        task="SCF", process_result=lambda value: value,
    )
    assert run.spec.stdin == "parallel.in"
    assert run.record.files.input_files == (str(tmp_path / "parallel.in"),)
    assert run.job_files.manifest.input_files == run.record.files.input_files


def test_run_factory_resolves_only_once_for_untracked_and_project(tmp_path):
    for tracked in (False, True):
        executor = _Executor()
        calls = []
        original = executor.resolve_spec

        def once(spec):
            calls.append(spec)
            assert len(calls) == 1
            return original(spec)

        executor.resolve_spec = once
        kwargs = {}
        if tracked:
            projects = ProjectRepository(tmp_path / "project-single-resolution")
            project = projects.create("Single resolution")
            manager = RunManager(project_repository=projects, executors=Executors())
            kwargs.update(project=(manager, project.id), task="SCF")
        run = CalculationRun.create(
            _spec(tmp_path), executor=executor,
            process_result=lambda value: value, **kwargs,
        )
        assert run.spec.argv == ("mpirun", "parallel")
        assert len(calls) == 1


def test_live_project_factory_uses_own_repository(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create_project("Live project")
    manager = projects.manager
    executor = _Executor()

    run = CalculationRun.create(
        _spec(tmp_path), executor=executor, project=project,
        task="SCF", process_result=lambda value: value,
    )

    assert run.persistence.manager is manager
    assert run.record.project_id == project.id
    assert run.spec.stdin == "parallel.in"
    assert run.job_files.manifest.input_files == run.record.files.input_files
