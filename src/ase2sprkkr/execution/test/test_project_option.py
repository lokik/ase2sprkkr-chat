from types import SimpleNamespace

from ase2sprkkr.execution import (
    RunManager,
    Executor,
    ExecutionOutcome,
    ProjectRepository,
    RunState,
    FileManifest,
)
from ase2sprkkr.execution.run_files import RunFiles
from ase2sprkkr.input_parameters.input_parameters import InputParameters
from ase2sprkkr.input_parameters.task_definition import TaskDefinition
from ase2sprkkr.sprkkr import calculation as calculation_module
from ase2sprkkr.sprkkr.calculation import PreparedCalculation
from ase2sprkkr.sprkkr.calculator import SPRKKR
from ase2sprkkr.sprkkr.profile import SPRKKRProfile


class _Runner:
    def __init__(self, callback=None, *, spec=None):
        self.callback = callback
        self.state = RunState.NEW
        self.backend_id = None
        self.spec = spec
        self.run_files = RunFiles(FileManifest())
        self.failure_phase = None
        self.failure_message = None
        self.result = ExecutionOutcome(
            input_file=None if spec is None else spec.input_file,
            output_file=None if spec is None else spec.output_file,
            error_file=None if spec is None else spec.error_file,
            return_code=0,
        )
        self.exception = None

    def prepare_submission(self):
        return None

    @property
    def return_code(self):
        return self.result.return_code

    def submit(self):
        self.backend_id = "job-1"
        self.state = RunState.QUEUED

    def wait(self):
        self.state = RunState.FINISHED
        if self.callback:
            self.callback(self.result)
        return self.result

    def run(self):
        self.submit()
        return self.wait()

    def cancel(self):
        self.state = RunState.CANCELLED

    def add_output_listener(self, callback):
        pass

    def add_output_error_listener(self, callback):
        pass


class _Executor(Executor):
    name = "fake"

    def __init__(self):
        self.runners = []

    def resolve_mpi(self, mpi=None):
        return False

    def create_run(self, spec, callback=None):
        runner = _Runner(callback, spec=spec)
        self.runners.append(runner)
        return runner


class _Prepared:
    create_run = PreparedCalculation.create_run

    def __init__(self, parameters, input_file, output_file, directory):
        self.input_parameters = parameters
        self.input_file = input_file
        self.potential_file = None
        self.output_file = output_file
        self.directory = str(directory)
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.input_file.close()
        self.output_file.close()


def _parameters(monkeypatch, task="DOS"):
    parameters = InputParameters.create(task)
    parameters.CONTROL.DATASET = "sample"
    parameters.CONTROL.POTFIL = "sample.pot"
    monkeypatch.setattr(
        TaskDefinition,
        "output_reader",
        lambda self, *args, **kwargs: SimpleNamespace(
            read_from_file=lambda *args, **kwargs: SimpleNamespace(return_code=0),
        ),
    )
    monkeypatch.setattr(InputParameters, "input_files", lambda self, directory: ())
    monkeypatch.setattr(InputParameters, "output_files", lambda self, directory: ())
    monkeypatch.setattr(InputParameters, "cache_files", lambda self, directory: ())
    return parameters


def _manager(tmp_path):
    projects = ProjectRepository(tmp_path / "projects")
    project = projects.create("Calculation project")
    return RunManager(project_repository=projects), project.id


def _runner(parameters, tmp_path, manager, executor=None):
    executor = executor or _Executor()
    input_file = (tmp_path / "test.inp").open("w+")
    output_file = (tmp_path / "test.out").open("wb")
    prepared = _Prepared(parameters, input_file, output_file, tmp_path)
    return prepared.create_run(
        result_context=SimpleNamespace(directory=str(tmp_path)),
        profile=SPRKKRProfile(),
        executor=executor,
        project=manager,
    )


def _project_record(parameters, tmp_path):
    manager = _manager(tmp_path)
    runner = _runner(parameters, tmp_path, manager)
    assert runner.id
    runner.run()
    return manager[0].store.get(runner.id)


def test_explicit_manager_persists_task_and_dataset(tmp_path, monkeypatch):
    parameters = _parameters(monkeypatch)

    record = _project_record(parameters, tmp_path)

    assert record.task == parameters.task_name
    assert record.input_definition == parameters.definition_name.upper()
    assert not hasattr(record, "requested_task")
    assert record.dataset == "sample"
    assert record.state == RunState.FINISHED


def test_persistence_uses_task_selected_by_shared_schema_values(tmp_path, monkeypatch):
    parameters = _parameters(monkeypatch, "ARPES")
    parameters.TASK.TASK = "SPLEED"

    record = _project_record(parameters, tmp_path)

    assert parameters.requested_task_name == "ARPES"
    assert parameters.task_name == "SPLEED"
    assert record.task == "SPLEED"
    assert record.input_definition == "ARPES"
    assert not hasattr(record, "requested_task")


def test_persistence_keeps_ase2sprkkr_task_distinct_from_sprkkr_task_value(
    tmp_path, monkeypatch
):
    parameters = _parameters(monkeypatch, "BSFEK")

    record = _project_record(parameters, tmp_path)

    assert parameters.TASK.TASK() == "BSF"
    assert parameters.task_name == "BSFEK"
    assert record.task == "BSFEK"
    assert record.input_definition == "BSF"


def test_explicit_project_identity_is_preserved_by_execution_adapter(tmp_path, monkeypatch):
    parameters = _parameters(monkeypatch)
    manager = _manager(tmp_path)
    executor_manager, project_id = manager
    project = executor_manager.projects.project(project_id)
    runner = _runner(parameters, tmp_path, manager)

    runner.submit()

    assert executor_manager.store.get(runner.id).id == runner.id
    runner.cancel()


def test_persistence_defaults_to_calculator_setting(tmp_path, monkeypatch):
    executor = _Executor()
    manager = _manager(tmp_path)
    calculator = SPRKKR(directory=tmp_path, executor=executor, project=manager)
    captured = {}

    class Prepared:
        def create_run(self, **kwargs):
            captured.update(kwargs)
            return object()

    prepared = Prepared()
    monkeypatch.setattr(calculator, "prepare", lambda **kwargs: prepared)

    calculator.create_run()

    assert captured["project"] is manager
    assert captured["executor"] is executor


def test_run_level_profile_selects_executor_instead_of_calculator_executor(
    tmp_path, monkeypatch
):
    parameters = _parameters(monkeypatch)
    calculator_executor = _Executor()
    profile_executor = _Executor()
    profile = SPRKKRProfile(executor="ssh", host="cluster")
    calculator = SPRKKR(directory=tmp_path, executor=calculator_executor)
    input_file = (tmp_path / "test.inp").open("w+")
    output_file = (tmp_path / "test.out").open("wb")
    prepared = _Prepared(parameters, input_file, output_file, tmp_path)

    monkeypatch.setattr(calculator, "prepare", lambda **kwargs: prepared)
    selected_endpoints = []

    def select_profile_endpoint(endpoint):
        selected_endpoints.append(endpoint)
        return profile_executor

    monkeypatch.setattr(
        "ase2sprkkr.execution.persistence.default_executors",
        lambda: SimpleNamespace(for_endpoint=select_profile_endpoint),
    )

    runner = calculator.create_run(profile=profile)
    try:
        assert len(profile_executor.runners) == 1
        assert calculator_executor.runners == []
        assert selected_endpoints == [profile.endpoint]
    finally:
        runner.close()
