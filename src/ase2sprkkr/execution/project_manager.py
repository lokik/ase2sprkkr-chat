"""Project-bound facade over the shared global run store.

Result processors are supplied by callers and never import specific calculators
into the generic execution subsystem.
"""

from __future__ import annotations

from .base import ExecutionOutcome
from .manager import ExecutionManager as _ExecutionManager, RunStore
from .projects import Project, ProjectRepository


class ExecutionManager(_ExecutionManager):
    """Track only project-owned runs, with a durable result-processor identity.

    Processors are registered by applications; their stable keys are stored in
    RunRecord and used for reconnecting and historical result loading.
    """

    def __init__(
        self, store=None, *, executor_pool=None, submitting_grace=60.0,
        project_repository=None, project=None, result_processors=None,
    ):
        self.projects = project_repository if project_repository is not None else ProjectRepository()
        self._project_id = None
        if project is not None:
            identity = project.id if isinstance(project, Project) else str(project)
            current = self.projects.project(identity)
            self._project_id = current.id
        if store is None:
            store = RunStore(self.projects.directory.parent / "projects_runs")
        self._result_processors = {} if result_processors is None else result_processors
        super().__init__(
            store, executor_pool=executor_pool,
            submitting_grace=submitting_grace,
        )

    @property
    def project_id(self):
        return self._project_id

    def register_result_processor(self, name, processor):
        """Register one stable processor identity on this manager and its views."""
        key = str(name).strip().lower()
        if not key:
            raise ValueError("result processor name must not be empty")
        previous = self._result_processors.get(key)
        if previous is not None and previous is not processor:
            raise ValueError(f"Result processor {key!r} is already registered")
        self._result_processors[key] = processor

    def _result_processor_for(self, record):
        key = record.result_processor
        if key is None:
            raise ValueError(f"Run {record.id} has no persisted result processor")
        try:
            return self._result_processors[key]
        except KeyError as exc:
            raise LookupError(
                f"Result processor {key!r} for run {record.id} is not registered"
            ) from exc

    def new_record(self, **kwargs):
        if self._project_id is None:
            raise ValueError("Tracked runs require a project")
        return super().new_record(project_id=self._project_id, **kwargs)

    def for_project(self, project):
        identity = project.id if isinstance(project, Project) else str(project)
        current = self.projects.project(identity)
        if self._project_id is not None and current.id != self._project_id:
            raise ValueError("Cannot replace the project of a project-bound execution manager")
        if self._project_id == current.id:
            return self
        return type(self)(
            store=self.store,
            executor_pool=self.executor_pool,
            submitting_grace=self.submitting_grace,
            project_repository=self.projects,
            project=current,
            result_processors=self._result_processors,
        )

    def create_run(
        self, executor, spec, *, process_result, task,
        input_definition=None, result_processor=None, dataset=None,
        callback=None, run_id=None, project=None,
    ):
        manager = self if project is None else self.for_project(project)
        if manager.project_id is None:
            raise ValueError("A tracked run must belong to an existing project")
        processor = None
        if result_processor is not None:
            key = str(result_processor).strip().lower()
            if not key:
                raise ValueError("result_processor must not be empty")
            try:
                processor = manager._result_processors[key]
            except KeyError as exc:
                raise LookupError(f"Result processor {key!r} is not registered") from exc

        def completed(result):
            if processor is not None and result is not None:
                processor.record_updated(result, manager.store.get(runner.id))
            if callback is not None:
                callback(result)

        runner = _ExecutionManager.create_run(
            manager, executor, spec,
            process_result=process_result, task=task,
            input_definition=input_definition,
            result_processor=result_processor, dataset=dataset,
            callback=completed, run_id=run_id,
        )
        return runner

    def attach(
        self, run, *, process_result=None, executor=None,
        callback=None, output_listener=None,
    ):
        """Reconnect using the persisted processor, unless one was explicit."""
        if process_result is not None:
            return super().attach(
                run, process_result=process_result, executor=executor,
                callback=callback, output_listener=output_listener,
            )
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        processor = self._result_processor_for(record)
        if (record.state.value == "failed"
                and record.failure_phase is not None
                and record.failure_phase.value == "download"
                and record.return_code == 0):
            record = self.resume_download(record)

        def process(outcome):
            return processor.process(record, outcome)

        def completed(result):
            if result is not None:
                processor.record_updated(result, self.store.get(record.id))
            if callback is not None:
                callback(result)

        return super().attach(
            record, process_result=process, executor=executor,
            callback=completed, output_listener=output_listener,
        )

    def load_result(self, run):
        """Retrieve verified historical files, then use their saved processor."""
        record = self.retrieve_result_files(run)
        if not self.output_file_is_current(record):
            raise OSError(f"Historical output of run {record.id} is missing or modified")
        processor = self._result_processor_for(record)
        files = record.files
        outcome = ExecutionOutcome(
            input_file=files.input_file,
            output_file=files.output_file,
            error_file=None if files.stderr is None else files.stderr.local_path,
            return_code=record.return_code,
        )
        result = processor.process(record, outcome)
        if result is not None:
            processor.record_updated(result, record)
        return result

    def retry_postprocessing(self, run, process_result=None):
        """Retry processing captured outputs without restarting the computation."""
        if process_result is not None:
            return super().retry_postprocessing(run, process_result)
        record = self.store.get(run) if isinstance(run, str) else self.store.get(run.id)
        processor = self._result_processor_for(record)
        result = super().retry_postprocessing(
            record, lambda outcome: processor.process(record, outcome)
        )
        if result is not None:
            processor.record_updated(result, self.store.get(record.id))
        return result
