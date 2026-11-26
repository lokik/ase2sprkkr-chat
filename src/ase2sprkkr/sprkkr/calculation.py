"""Prepared SPR-KKR calculations and their execution adapter.

``SPRKKR`` supplies user defaults; ``_preparation`` resolves state and writes
files. This module orchestrates preparation, owns the resulting resources,
and adapts prepared calculations to the generic execution subsystem.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, replace

from kkrunner.files import TaskFile

from ..common.file_utils import Directory
from ..common.output_printing import output_listeners
from kkrunner.runs.base import ExecutionResources
from .projects import memory_execution_manager, resolve_project_manager
from kkrunner.runs import Run
from kkrunner import ExecutionSpec, MemoryRunStore
from ..input_parameters.input_parameters import InputParameters
from ..potentials.cluster_potentials import ClusterPotential
from . import _preparation as preparation
from ._preparation import FilenameTemplator as FilenameTemplator
from .profile import SPRKKRProfile, executor_from_profile
from .executables import resolve_task_executable


@dataclass
class PreparedCalculation:
    """One fully materialized SPR-KKR calculation.

    Instances own the concrete parameter snapshot, path-backed calculation
    files and, when applicable, an entered temporary working-directory context.
    Ownership can be transferred to a runner when preprocessing begins,
    explicitly released for ``save_input(return_files=True)``, or terminated
    with :meth:`close`.

    :meth:`prepare` is the canonical factory because it resolves all public
    calculator sentinels and establishes these invariants. The generated
    dataclass constructor validates path-backed file invariants as well, so it
    cannot silently create an execution-incompatible instance.
    """

    input_parameters: InputParameters
    input_file: object
    potential_file: str | None
    output_file: object | None
    directory: str
    counter: int
    _file_resources: ExitStack | None
    _directory_resources: ExitStack | None
    _temporary_potential: bool = False
    _temporary_directory: bool = False
    _closed: bool = False
    cluster_potential_file: str | None = None
    _temporary_cluster_potential: bool = False
    clear_cache: bool = False
    _temporary_files: tuple = ()
    _directory_context: Directory | None = None

    def __post_init__(self):
        if type(self.clear_cache) is not bool:
            raise TypeError("clear_cache must be a bool")
        preparation.require_named_file(self.input_file, "input_file")
        if self.output_file is not None:
            preparation.require_named_file(self.output_file, "output_file")
        if self.potential_file is not None:
            try:
                os.fspath(self.potential_file)
            except TypeError:
                raise ValueError(
                    "SPR-KKR potential_file must be backed by a filesystem path"
                ) from None
        if self.cluster_potential_file is not None:
            try:
                os.fspath(self.cluster_potential_file)
            except TypeError:
                raise ValueError(
                    "SPR-KKR cluster_potential_file must be a filesystem path"
                ) from None
        self.directory = os.path.abspath(os.fspath(self.directory))

    @classmethod
    def prepare(cls, *, run_manager=None, **kwargs):
        """Plan and materialize an explicit standalone preparation."""
        return CalculationPlan.plan(**kwargs).prepare(run_manager=run_manager)

    def create_run(
        self,
        *,
        result_context=None,
        profile: SPRKKRProfile,
        executor=None,
        project=None,
        print_output=False,
        read_callback=None,
        callback=None,
        gdb=False,
        error_file=None,
        create_subdirs=False,
        metadata=None,
    ):
        """Create an unstarted runner borrowing this calculation.

        Its resources transfer to the run only when reserved preprocessing
        starts. If creation, cancellation or reservation fails before that
        boundary, the caller retains ownership and must close the calculation.
        The prepared calculation does not depend on the calculator facade.
        ``result_context`` is optional runtime context for output readers; the
        :class:`SPRKKR` facade supplies itself so result convenience APIs can
        construct follow-up calculations, while standalone prepared runs may
        leave the context unset.

        ``error_file=None`` stores stderr as ``<output_file>.err``. ``True``
        explicitly merges stderr into ``output_file``; ``False`` requests a
        managed temporary stderr file; a path or file object is used directly.
        Tracked runs require persistent working-directory and stderr paths so
        they can be recovered after the client exits.
        """
        return _create_sprkkr_run(
            self,
            result_context=result_context,
            profile=profile,
            executor=executor,
            project=project,
            print_output=print_output,
            read_callback=read_callback,
            callback=callback,
            gdb=gdb,
            error_file=error_file,
            create_subdirs=create_subdirs,
            metadata=metadata,
        )

    @property
    def input_declarations(self):
        # A caller-supplied prepared input may still contain buffered writes.
        path = preparation.execution_path(self.input_file, self.directory)
        return _input_declarations(self, (path,))

    @property
    def closed(self):
        """Whether this object has relinquished or closed all owned resources."""
        return self._closed

    @property
    def temporary_directory(self):
        return self._temporary_directory

    @property
    def has_temporary_inputs(self):
        return bool(self._temporary_files)

    @property
    def temporary_potential(self):
        """Whether POTFIL is a managed temporary pathname."""
        return self._temporary_potential

    @property
    def temporary_cluster_potential(self):
        """Whether POTFIL_CLU is a managed temporary pathname."""
        return self._temporary_cluster_potential

    def release_file_ownership(self):
        """Transfer ownership of open input/output files to the caller.

        The files themselves are not closed. This supports the legacy
        ``save_input(return_files=True)`` API. A temporary POTFIL cannot be
        transferred by that API because it returns only its pathname; callers
        must supply a persistent POTFIL name in that case.
        """
        stack = self._file_resources
        self._file_resources = None
        if stack is not None:
            stack.pop_all()
        self._update_closed_state()

    def close_directory_context(self):
        """Close the entered working-directory context after file transfer."""
        stack = self._directory_resources
        self._directory_resources = None
        if stack is not None:
            stack.close()
        self._update_closed_state()

    def _update_closed_state(self):
        if self._file_resources is None and self._directory_resources is None:
            self._closed = True

    def detach(self):
        """Close client handles but preserve paths after uncertain execution."""
        for file in self._temporary_files:
            file.preserve()
        if self._directory_context is not None:
            self._directory_context.preserve()
        self.close()

    def close(self):
        """Close all resources still owned by this prepared calculation."""
        if self._closed:
            return

        file_resources = self._file_resources
        directory_resources = self._directory_resources
        self._file_resources = None
        self._directory_resources = None
        self._closed = True

        error = None
        if file_resources is not None:
            try:
                file_resources.close()
            except BaseException as exc:
                error = exc
        if directory_resources is not None:
            try:
                directory_resources.close()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error


@dataclass
class CalculationPlan:
    """Write-free declaration of one calculation and its client preprocessing.

    The planned files and parameter-derived dataset do not change between
    declaration and materialization. The plan owns the resources acquired
    during resolution until they are transferred to PreparedCalculation.
    """

    input_parameters: InputParameters
    directory: str
    counter: int
    input_plan: preparation.PlannedFile
    potential_plan: preparation.PlannedFile | None
    cluster_plan: preparation.PlannedFile | None
    output_plan: preparation.PlannedFile | None
    potential_file: str | None
    cluster_potential_file: str | None
    resolved_input: preparation.ResolvedInput
    resolved_potential: preparation.ResolvedPotential
    files: preparation.CalculationFiles
    directory_resources: ExitStack
    directory_context: Directory
    task_name: str
    restart_scf: bool
    temporary_directory: bool
    clear_cache: bool
    empty_spheres: object
    _prepared: PreparedCalculation | None = None
    _closed: bool = False

    @classmethod
    def plan(
        cls,
        *,
        default_atoms,
        default_potential,
        default_input_parameters,
        default_input_file,
        default_potential_file,
        default_output_file,
        default_directory,
        default_empty_spheres,
        label,
        counter,
        atoms=None,
        input_parameters=None,
        potential=None,
        input_file=None,
        potential_file=None,
        output_file=False,
        directory=None,
        create_subdirs=False,
        options=None,
        task=None,
        empty_spheres=None,
        allow_temporary_files=False,
        restart_scf=False,
        clear_cache=False,
    ) -> CalculationPlan:
        """Resolve the physical state, parameter snapshot and exact file names.

        Existing inputs may be read to resolve the declarations. No calculation
        file is opened for writing. A private temporary working directory may be
        allocated to establish its namespace.
        """
        directory_resources = ExitStack()
        file_resources = ExitStack()

        try:
            directory_context = Directory.new(directory, default=default_directory)
            work_directory = directory_resources.enter_context(directory_context).path
            work_directory = os.path.abspath(os.fspath(work_directory))

            resolved_potential = preparation.resolve_potential_and_atoms(
                potential=potential,
                atoms=atoms,
                potential_file=potential_file,
                default_potential=default_potential,
                default_atoms=default_atoms,
                restart_scf=restart_scf,
                directory=work_directory,
            )
            resolved_input = preparation.resolve_input_parameters(
                input_parameters=input_parameters,
                default_input_parameters=default_input_parameters,
                input_file=input_file,
                potential=resolved_potential,
                options=options,
                task=task,
                directory=work_directory,
            )
            parameters = resolved_input.parameters
            atoms = resolved_potential.atoms
            task = parameters.task_definition_required()
            preparation.validate_task_system(task, resolved_input, resolved_potential)
            task_name = task.name
            preparation.validate_restart(resolved_potential.potential, task_name, restart_scf)
            if isinstance(resolved_potential.potential, ClusterPotential):
                requested_spheres = (
                    default_empty_spheres if empty_spheres is None else empty_spheres
                )
                if requested_spheres not in (None, False, "auto"):
                    raise ValueError(
                        "Empty-sphere generation is not supported for an embedded cluster"
                    )
            else:
                requested_spheres = default_empty_spheres if empty_spheres is None else empty_spheres

            templator = preparation.FilenameTemplator(
                label,
                counter,
                lambda: task_name,
                lambda: preparation.symbols_for_filenames(
                    atoms,
                    parameters,
                    resolved_input.filename,
                    resolved_input.potential_file,
                    default_input_file,
                    default_potential_file,
                ),
            )
            files = preparation.CalculationFiles(
                work_directory,
                templator,
                file_resources,
                allow_temporary=allow_temporary_files,
                create_subdirs=create_subdirs,
            )
            input_plan = files.plan_input(resolved_input, default_input_file)
            reused_input = None if resolved_input.write else input_plan.path
            cluster_plan = None
            if isinstance(resolved_potential.potential, ClusterPotential):
                potential_plan, cluster_plan = preparation.plan_cluster_potentials(
                    resolved_potential, resolved_input.potential_file, files,
                    default_filename=default_potential_file, input_parameters=parameters,
                )
            else:
                potential_plan = preparation.plan_potential(
                    resolved_potential, resolved_input.potential_file, files,
                    default_filename=default_potential_file,
                )
            potential_path = (resolved_input.potential_file if potential_plan is None
                              else potential_plan.path)
            cluster_path = None if cluster_plan is None else cluster_plan.path
            preparation.configure_input(
                resolved_input, input_plan.path, potential_path, cluster_potential_file=cluster_path,
            )
            if resolved_input.write:
                parameters.set_from_atoms(atoms if resolved_potential.potential is not None else None)
            output_plan = files.plan_output(
                output_file, default_filename=default_output_file, reused_input=reused_input,
            )

            return cls(
                input_parameters=parameters,
                directory=work_directory,
                counter=templator.counter,
                input_plan=input_plan,
                potential_plan=potential_plan,
                cluster_plan=cluster_plan,
                output_plan=output_plan,
                potential_file=potential_path,
                cluster_potential_file=cluster_path,
                resolved_input=resolved_input,
                resolved_potential=resolved_potential,
                files=files,
                directory_resources=directory_resources,
                directory_context=directory_context,
                task_name=task_name,
                restart_scf=restart_scf,
                temporary_directory=directory_context.dir is False,
                clear_cache=clear_cache,
                empty_spheres=requested_spheres,
            )
        except BaseException:
            try:
                file_resources.close()
            except BaseException:
                pass
            try:
                directory_resources.close()
            except BaseException:
                pass
            raise

    @property
    def has_temporary_inputs(self):
        return any(file is not None and file.temporary
                   for file in (self.input_plan, self.potential_plan, self.cluster_plan))

    @property
    def written_inputs(self):
        """Input paths generated by this particular calculation."""
        return tuple(file.path for file in (
            self.input_plan if self.resolved_input.write else None,
            self.potential_plan, self.cluster_plan,
        ) if file is not None)

    @property
    def input_declarations(self):
        return _input_declarations(self, self.written_inputs)

    @property
    def input_file(self):
        return self.input_plan.path

    @property
    def output_file(self):
        return None if self.output_plan is None else self.output_plan.path

    def create_run(self, **kwargs):
        """Transfer this unmaterialized plan to a runner."""
        if self._closed or self._prepared is not None:
            raise RuntimeError("Calculation plan was already materialized or closed")
        return _create_sprkkr_run(
            self, materialize=self._materialize_inputs,
            **kwargs,
        )

    def reservation(self, *, run_manager=None):
        """Reserve standalone preparation without starting a native backend."""
        primary, inputs = self.input_declarations
        spec = ExecutionSpec(
            argv=(), directory=self.directory,
            input_file=primary, input_files=inputs, output_file=self.output_file,
        )
        manager = memory_execution_manager() if run_manager is None else run_manager
        return manager.preprocessing_reservation(spec)

    def prepare(self, *, run_manager=None):
        """Materialize standalone input files while protecting their namespace."""
        try:
            with self.reservation(run_manager=run_manager):
                return self._materialize()
        except BaseException:
            self.close()
            raise

    def _materialize_inputs(self):
        """Materialize only the input side of an executing calculation."""
        prepared = self._materialize(open_output=False)
        # Native execution consumes pathnames, not the preparation descriptors.
        if isinstance(prepared.input_file, preparation.TemporaryNamedFile):
            prepared.input_file.close_file()
        else:
            prepared.input_file.close()
        if self.output_plan is not None:
            selection = self.output_plan.selection
            if not isinstance(selection, (bool, str, os.PathLike)):
                selection.close()  # Transfer a supplied named target before KKRunner opens it.
        return prepared

    def _materialize(self, *, open_output=True):
        """Open and serialize the planned files under the caller's reservation."""
        if self._closed or self._prepared is not None:
            raise RuntimeError("Calculation plan was already materialized or closed")
        try:
            version = self.resolved_input.source_version
            if version is not None and not version.is_current():
                raise ValueError(f"Input parameters changed after planning: {version.path}")
            # A path-only potential may have changed since planning. Validate
            # the geometry under reservation, before opening any write target.
            preparation.validate_task_system(
                self.input_parameters.task_definition_required(),
                self.resolved_input, self.resolved_potential,
            )
            if not isinstance(self.resolved_potential.potential, ClusterPotential):
                preparation.resolve_empty_spheres(
                    input_parameters=self.input_parameters,
                    atoms=self.resolved_potential.atoms,
                    empty_spheres=self.empty_spheres,
                    default_empty_spheres=False,
                )
                if self.resolved_input.write and self.empty_spheres:
                    self.input_parameters.set_from_atoms(
                        self.resolved_potential.atoms if self.resolved_potential.potential is not None else None
                    )
            opened_input = self.files.open(self.input_plan)
            if self.cluster_plan is not None:
                preparation.write_cluster_potentials(
                    self.resolved_potential, self.potential_plan, self.cluster_plan, self.files,
                    task_name=self.task_name, restart_scf=self.restart_scf,
                )
            else:
                preparation.write_potential(
                    self.resolved_potential, self.potential_plan, self.files,
                    task_name=self.task_name, restart_scf=self.restart_scf,
                )
            preparation.write_input(self.resolved_input, opened_input)
            opened_output = self.output_file
            if open_output and self.output_plan is not None:
                opened_output = self.files.open(self.output_plan)
            self._prepared = PreparedCalculation(
                input_parameters=self.input_parameters,
                input_file=opened_input,
                potential_file=self.potential_file,
                output_file=opened_output,
                directory=self.directory,
                counter=self.counter,
                _file_resources=self.files.resources,
                _directory_resources=self.directory_resources,
                _temporary_potential=self.potential_plan is not None and self.potential_plan.temporary,
                _temporary_directory=self.temporary_directory,
                cluster_potential_file=self.cluster_potential_file,
                _temporary_cluster_potential=self.cluster_plan is not None and self.cluster_plan.temporary,
                clear_cache=self.clear_cache,
                _temporary_files=tuple(self.files.temporary_files),
                _directory_context=self.directory_context,
            )
            return self._prepared
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def detach(self):
        """Relinquish a possibly still running calculation without deleting its files."""
        if self._prepared is not None:
            self._prepared.detach()
        self.close()

    def close(self):
        """Release unmaterialized resources or the materialized calculation."""
        if self._closed:
            return
        self._closed = True
        if self._prepared is not None:
            self._prepared.close()
        else:
            try:
                self.files.resources.close()
            finally:
                self.directory_resources.close()


def _input_declarations(calculation, written_inputs):
    """Attach the resolved write intent without duplicating input declarations."""
    directory = calculation.directory
    written = set(written_inputs)
    primary = preparation.execution_path(calculation.input_file, directory)
    primary = TaskFile(primary, write=primary in written)
    declarations = []
    for value in (calculation.potential_file, *calculation.input_parameters.input_files(directory)):
        if value is None:
            continue
        declaration = value if isinstance(value, TaskFile) else TaskFile(value)
        path = preparation.calculation_path(declaration, directory)
        declarations.append(replace(declaration, pathname=path,
                                    write=declaration.write or path in written))
    return primary, tuple(declarations)


def _plan_error_file(error_file, output_file, directory):
    """Return a stderr pathname and whether it is a temporary target."""
    if error_file is True:
        return output_file, False
    if error_file is None:
        return None if output_file is None else output_file + ".err", False
    if error_file is False:
        from uuid import uuid4
        return os.path.join(directory, "tmp-sprkkr-" + uuid4().hex), True
    return preparation.execution_path(error_file, directory), False


def _run_cache_files(parameters, directory, *, clear_cache=False):
    """Return output/cache declarations for one run without mutating task metadata."""
    outputs = tuple(parameters.output_files(directory))
    caches = tuple(parameters.cache_files(directory))
    if not clear_cache:
        return outputs, caches
    # An invalidated cache is an optional output: the native executable may
    # recreate it, but a successful task need not produce it.
    refreshed = tuple(
        replace(value, optional=True)
        if isinstance(value, TaskFile)
        else TaskFile(value, optional=True)
        for value in caches
    )
    return (*outputs, *refreshed), ()


def _create_sprkkr_run(
    calculation,
    *,
    result_context=None,
    profile: SPRKKRProfile,
    materialize=None,
    executor=None,
    project=None,
    print_output=False,
    read_callback=None,
    callback=None,
    gdb=False,
    error_file=None,
    create_subdirs=False,
    metadata=None,
):
    """Adapt declared calculation files and optional materialization to execution.

    This SPR-KKR layer owns task metadata, reader construction and TaskResult
    reconstruction. Executors see only :class:`ExecutionSpec` and produce raw
    :class:`ExecutionOutcome` values; the generic postprocessing layer merely
    invokes the callable defined here.
    """
    error_resources = ExitStack()
    calculation_resources = ExitStack()
    release = calculation.close

    def close_calculation():
        release()

    # An unmaterialized plan owns only private planning resources. A
    # prepared calculation may own buffered user files and remains borrowed
    # until reservation succeeds; its cleanup must not run on rejection.
    if materialize is not None:
        calculation_resources.callback(close_calculation)

    def detach():
        nonlocal release
        error_resources.pop_all()  # Preserve a possibly active temporary stderr.
        release = calculation.detach
        calculation_resources.close()

    resources = ExecutionResources(calculation_resources, error_resources, on_detach=detach)
    try:
        input_file = preparation.execution_path(calculation.input_file, calculation.directory)
        output_file = preparation.execution_path(calculation.output_file, calculation.directory)
        if error_file is calculation.output_file and error_file is not None:
            error_file = True
        error_path, temporary_error = _plan_error_file(
            error_file, output_file, calculation.directory,
        )
    except BaseException:
        resources.close(suppress_errors=True)
        raise
    handed_off = False
    input_parameters = calculation.input_parameters

    try:
        listeners, print_info = output_listeners(print_output)
        task = input_parameters.task_definition_required()
        reader = task.output_reader(
            input_parameters=input_parameters,
            calculator=result_context,
            print_info=print_info,
            directory=calculation.directory,
        )
        if executor is None:
            executor = executor_from_profile(profile)
        manager = resolve_project_manager(project)
        if manager is None:
            manager = memory_execution_manager()
        durable = not isinstance(manager.store, MemoryRunStore)
        if metadata is not None and "ase2sprkkr" in metadata:
            raise ValueError("The 'ase2sprkkr' run metadata namespace is reserved")
        detached = True if profile.detached and executor.backend.name == "direct" else None
        if durable or detached or executor.default_detached:
            if calculation.temporary_directory:
                raise ValueError("Recoverable runs require a persistent calculation directory; pass directory=path")
            if temporary_error:
                raise ValueError("Recoverable runs require persistent stderr; use error_file=None or a pathname")
            if calculation.has_temporary_inputs:
                raise ValueError("Recoverable runs require persistent input and potential files")

        input_file, input_files = calculation.input_declarations
        auxiliary_outputs, cache_files = _run_cache_files(
            input_parameters, calculation.directory, clear_cache=calculation.clear_cache,
        )
        output_files = tuple(
            file for file in (
                output_file,
                error_path if error_path != output_file else None,
                *auxiliary_outputs,
            ) if file is not None
        )

        spec = profile.create_execution(
            resolve_task_executable(task, input_parameters),
            input_file,
            output_file,
            calculation.directory,
            error_file=error_path,
            input_files=input_files,
            output_files=output_files,
            cache_files=cache_files,
            mpi_supported=task.mpi,
            gdb=gdb,
        )

        def preprocess():
            if materialize is not None:
                materialize()
            else:
                calculation_resources.callback(close_calculation)
                # The explicit prepared-calculation API can supply a still
                # buffered input handle. Flush only under the reservation.
                if not isinstance(calculation.input_file, (str, os.PathLike)):
                    calculation.input_file.flush()
                if calculation.output_file is not None and not isinstance(calculation.output_file, (str, os.PathLike)):
                    calculation.output_file.close()  # Standalone prepare() may have returned an open OUT.
            if error_file is not None and not isinstance(error_file, (bool, str, os.PathLike)):
                error_file.close()
            if temporary_error:
                from pathlib import Path
                error_resources.callback(Path(error_path).unlink, missing_ok=True)

        def flush_listeners():
            for listener in listeners:
                listener.flush()

        def process_result(outcome):
            flush_listeners()
            return reader.read_from_file(
                outcome.output_file,
                outcome.error_file,
                return_code=(
                    0 if outcome.return_code is None else outcome.return_code
                ),
                input_file=outcome.input_file,
            )

        def finalized(out):
            if out is not None:
                if durable:
                    out.run_record = manager.store.get(runner.id)
                else:
                    out.file_manifest = runner.files
            flush_listeners()
            # Run has already released ExecutionResources before completion.
            if callback:
                callback(out)

        record_options = {"metadata": {
            **(dict(metadata) if metadata is not None else {}),
            "ase2sprkkr": {
                "task": task.name,
                "input_definition": input_parameters.definition_name,
                "dataset": input_parameters.CONTROL.DATASET.result,
                "profile": profile.name,
            },
        }}
        runner = Run.create(
            executor, spec,
            persistent=manager,
            detached=detached,
            process_result=process_result,
            callback=finalized,
            owned_resources=resources,
            preprocess=preprocess,
            **record_options,
        )
        handed_off = True
        try:
            for listener in listeners:
                runner.add_output_listener(listener)
            if read_callback:
                runner.add_output_listener(read_callback)
        except BaseException:
            # An unstarted Run can release resources it already owns.
            runner.close()
            raise
        return runner
    finally:
        if not handed_off:
            resources.close(suppress_errors=True)
