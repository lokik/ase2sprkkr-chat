"""ASE calculator facade for SPR-KKR.

The calculator stores persistent user/ASE defaults.  Materialization of one
concrete calculation lives in :mod:`ase2sprkkr.sprkkr.calculation`; process
execution is delegated to the generic execution subsystem.
"""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union
from ase.calculators.calculator import BaseCalculator, all_changes
import numpy as np

from ..common.file_utils import Directory
from ..common.misc import config_property, first_non_none
from ..configuration import config
from ..potentials.potentials import Potential
from .profile import SPRKKRProfile, resolve_sprkkr_profile
from .sprkkr_atoms import SPRKKRAtoms


class SPRKKR(BaseCalculator):
    """ASE-facing calculator for SPR-KKR.

    ``SPRKKR`` is a mutable facade holding defaults shared by calculations:
    structure/potential, input parameters, filename templates, working directory
    and execution policy. Preparation snapshots run-specific configuration and
    materializes the selected physical state into independent calculation files.

    Selected ``Atoms`` and ``Potential`` objects are mutable physical working
    state, not per-run snapshots. This applies both to calculator defaults and
    explicitly supplied objects: preparation may promote or normalize the
    structure, add automatic empty spheres, and synchronize the potential. Once
    preparation finishes, the returned :class:`PreparedCalculation` owns its
    concrete files and resources, so execution no longer depends on subsequent
    changes to those physical objects.

    A calculator instance is not intended to be prepared concurrently from
    multiple threads because filename-template counters and physical
    preprocessing are mutable state. Prepared runners themselves own independent
    resources and may run concurrently.
    """

    implemented_properties = ["energy"]

    def __init__(
        self,
        label=None,
        directory=".",
        atoms=None,
        potential=None,
        input_file=True,
        output_file=True,
        potential_file=True,
        input_parameters=None,
        options=None,
        print_output=None,
        mpi: Optional[bool] = None,
        empty_spheres: Optional[Union[str, bool, Dict]] = None,
        executable_suffix=None,
        executable_dir: Optional[Union[str, bool]] = None,
        profile=None,
        executor=None,
        project=None,
        **kwargs,
    ):
        """Configure persistent defaults for SPR-KKR calculations.

        Initialization does not run SPR-KKR. Individual calculations can override
        these defaults using :meth:`calculate`, :meth:`create_run` or
        :meth:`save_input`. Physical structure and potential objects are retained
        rather than copied; run-specific input parameters are copied on preparation.

        Parameters
        ----------
        label : str or None, optional
            Prefix used by the `%l` placeholder in generated filenames. By
            default no prefix is added.
        directory : str, os.PathLike or False, default "."
            Working directory for calculation files and process execution.
            `False` requests a managed temporary directory for executing runs.
            Unlike execution, `save_input` requires a persistent directory.
        atoms : ase.Atoms, SPRKKRAtoms or None, optional
            Default structure. ASE atoms are promoted to `SPRKKRAtoms`; preparation
            may update this object (for example, by adding empty spheres). Do not
            supply `atoms` together with a `Potential` object.
        potential : Potential, str, bool or None, optional
            Default physical potential. A `Potential` object supplies its own atoms;
            a string names a potential file, resolved during preparation. `True`
            requests generation from atoms; `False` uses `CONTROL.POTFIL` from
            the input parameters. `None` uses the available physical state, normally
            generating a potential from atoms when necessary.
        input_file : str, os.PathLike, named file or bool, default True
            Input filename or filename template. `True` selects the default
            `%l%_%a_%T.inp` template; `False` creates a named temporary file when
            temporary artifacts are allowed. Templates recognize `%l` (label),
            `%a` (chemical symbols), `%t`/`%T` (task name, original/uppercase),
            `%d` (date and time), `%c` (incrementing counter), and `%_`
            (optional underscore separator).
        output_file : str, os.PathLike, named file or bool, default True
            Process-output destination. `True` chooses an output name derived
            from the input filename when reused, or the default `%a_%T.out`
            template. `False` disables the separate output file. Other
            filenames and templates follow the rules of `input_file`.
        potential_file : str, os.PathLike, named file or bool, default True
            Where a generated or updated potential is written. `True` uses an
            input-derived name when applicable or `%a.pot` otherwise;
            `False` requests a managed temporary file. This does not select
            which physical potential is used (see `potential`).
        input_parameters : InputParameters, str, file object or None, optional
            Persistent defaults for the SPR-KKR input file. A string may identify a
            task (such as `"SCF"` or `"DOS"`), a named input-parameter schema,
            or a file to read. `None` lazily selects the default SCF parameters.
            Each run operates on a copy, leaving these defaults unchanged.
        options : dict or None, optional
            Initial values applied to `input_parameters`, possibly specified as
            `SECTION.OPTION` keys. Additional keyword arguments are applied
            as input-parameter values by the same mechanism.
        print_output : bool, str or None, optional
            Whether to display executable output in addition to writing output
            files. `"info"` selects concise iteration information. `None` uses
            `config.running.print_output`.
        mpi : bool, int, str, sequence or None, optional
            Override the resolved execution profile's MPI launch policy. `False`
            disables MPI; `True`/`"auto"` request autodetection; an integer
            specifies the number of processes and a command or argument sequence
            can specify a launcher. `None` keeps the profile setting.
        empty_spheres : bool, str, tuple, mapping or None, optional
            Whether to insert empty spheres when preparing the structure. `"auto"`
            adds them for applicable initial SCF structures lacking vacuum sites.
            A pair `(min_radius, max_radius)` sets the radius bounds; a mapping
            supplies keyword arguments to `add_empty_spheres`. `None` uses
            `config.running.empty_spheres`.
        executable_suffix : str, False or None, optional
            Suffix appended to SPR-KKR executable names. An empty string or
            `False` disables the suffix; `None` retains the profile setting.
        executable_dir : str, os.PathLike, False or None, optional
            Directory containing SPR-KKR executables. `False` searches `PATH`;
            `None` retains the profile setting.
        profile : SPRKKRProfile, str or None, optional
            Machine/execution profile or the name of a configured profile. `None`
            resolves the configured default profile. Explicit `mpi`,
            `executable_dir` and `executable_suffix` override its launch settings.
        executor : object or None, optional
            Explicit execution backend, taking precedence over the profile's
            backend for runs that do not specify a different profile or executor.
        project : object or None, optional
            Default project context forwarded to execution runners, allowing
            project-managed runs and their associated artifacts.
        **kwargs : object
            Additional input-parameter values, passed to
            `input_parameters.set(..., unknown="find")` together with `options`.
            These are not arbitrary calculator options.

        See Also
        --------
        prepare : Materialize one calculation's inputs and resources.
        create_run : Create an unstarted runner for one task.
        calculate : Execute a task synchronously.
        """
        if potential is not None and not isinstance(potential, (bool, str)):
            if atoms is not None:
                raise ValueError(
                    "You can not specify both atoms and potential. "
                    "If you specify one of them, the other will be generated from it."
                )
            atoms = potential.atoms
        elif atoms is not None:
            atoms = SPRKKRAtoms.promote_ase_atoms(atoms)

        self._potential = None
        self.atoms = atoms
        self.potential = potential

        super().__init__(use_cache=False)
        if directory is False:
            self._directory = directory
        else:
            self.directory = directory
        self.atoms = atoms
        self.potential = potential
        self.input_file = input_file
        self.output_file = output_file
        self.potential_file = potential_file
        self.input_parameters = input_parameters
        if options or kwargs:
            self.input_parameters.set(options, **kwargs, unknown="find")
        self.print_output = print_output
        self.empty_spheres = empty_spheres
        self.profile = resolve_sprkkr_profile(profile).with_overrides(
            mpi=mpi,
            executable_dir=executable_dir,
            executable_suffix=executable_suffix,
        )
        self.executor = executor
        self.project = project
        self.label = label
        self._counter = 0

    print_output = config_property("print_output", config.running.print_output)
    empty_spheres = config_property("empty_spheres", config.running.empty_spheres)

    @property
    def input_parameters(self):
        """Default :class:`InputParameters` used for new calculations."""
        if self._input_parameters is None:
            self.input_parameters = "SCF"
        return self._input_parameters

    @input_parameters.setter
    def input_parameters(self, value):
        if value is not None:
            value = InputParameters.create_input_parameters(value)
        self._input_parameters = value

    def set(
        self,
        options: Union[Dict[str, Any], str, None] = None,
        value=None,
        *,
        unknown="find",
        **kwargs,
    ):
        """Set values on the calculator's persistent input-parameter defaults."""
        if isinstance(options, str):
            options = {options: value}
        elif value is not None:
            raise ValueError(
                "If value argument is given to SPRKKR.set, the options argument have to be string name of the value"
            )
        if kwargs or options:
            self.input_parameters.set(options, **kwargs, unknown=unknown)

    def get(self, name):
        """Return an input-parameter value by name or ``SECTION.NAME`` path."""
        return self.input_parameters.get(name)

    @property
    def potential(self) -> Potential:
        """Potential associated with the calculator, generated lazily from atoms."""
        if self._potential is None:
            if self._atoms is None:
                return None
            self._potential = Potential.from_atoms(self._atoms)
        return self._potential

    @potential.setter
    def potential(self, pot):
        self._potential = pot
        if pot is not None and not isinstance(pot, (bool, str)):
            atoms = pot.atoms
            if atoms is not None:
                self._atoms = atoms
            else:
                pot._atoms = self._atoms

    @property
    def atoms(self) -> SPRKKRAtoms:
        """Structure associated with the calculator."""
        if self._atoms is None:
            if self._potential is None or isinstance(self._potential, bool):
                return None
            if isinstance(self._potential, str):
                self._potential = Potential.from_file(self._potential)
            self._atoms = self._potential.atoms
        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        atoms = SPRKKRAtoms.promote_ase_atoms(atoms)
        self._atoms = atoms
        if self._potential is not None and self._potential is not False:
            if isinstance(self._potential, str):
                self._potential = None
            elif self._potential is not True:
                self._potential.atoms = atoms

    @property
    def directory(self):
        """Default working directory, or ``False`` for a temporary run directory."""
        return self._directory

    @directory.setter
    def directory(self, directory: Union[bool, str, os.PathLike]):
        if not isinstance(directory, bool):
            directory = str(Path(directory))
        self._directory = directory

    def _advance_counter(self):
        """Advance the filename-template counter used by the legacy templator."""
        self._counter += 1
        return str(self._counter)

    def prepare(
        self,
        atoms=None,
        input_parameters=None,
        potential=None,
        input_file=None,
        potential_file=None,
        output_file=None,
        directory: Union[str, bool, Directory, None] = None,
        create_subdirs: bool = False,
        options=None,
        task=None,
        empty_spheres=None,
        allow_temporary_files: bool = True,
        restart_scf: bool = False,
    ):
        """Materialize one task's inputs and return an owning preparation object.

        Input parameters are copied before task and option overrides are applied.
        Atoms and potentials are *not* deep-copied: the selected physical state may
        be normalized or modified during preparation. The returned object owns its
        files and directory resources and must be closed unless ownership is
        transferred to a runner.

        Parameters
        ----------
        atoms : ase.Atoms, SPRKKRAtoms or None, optional
            Structure for this preparation; `None` uses the calculator default.
        input_parameters : InputParameters, str or None, optional
            Task parameters or task/file name; `None` uses calculator defaults.
        potential : Potential, str, bool or None, optional
            Potential or input filename, using the constructor's sentinel rules.
        input_file, potential_file, output_file : str, os.PathLike, named file, bool or None
            File destinations or templates. `None` inherits each calculator
            default; `False` requests a temporary file where supported (for
            `output_file`, `False` disables its creation).
        directory : str, os.PathLike, Directory, False or None, optional
            Working directory. `None` inherits the calculator's directory;
            `False` requests a managed temporary directory.
        create_subdirs : bool, default False
            Create missing parent directories for generated files.
        options : dict or None, optional
            Input-parameter values applied only to this preparation.
        task : str or None, optional
            Change the task of the copied input parameters before writing.
        empty_spheres : bool, str, tuple, mapping or None, optional
            Per-run empty-sphere policy; `None` uses the calculator default.
        allow_temporary_files : bool, default True
            Allow managed temporary input and potential file paths.
        restart_scf : bool, default False
            Write a potential suitable for restarting an SCF run. Requires an
            SCF task and a writable or generatable potential.

        Returns
        -------
        PreparedCalculation
            Materialized task parameters, files and owned resources. Call
            `close()` if no runner takes ownership.
        """
        prepared = PreparedCalculation.prepare(
            default_atoms=self._atoms,
            default_potential=self._potential,
            default_input_parameters=self.input_parameters,
            default_input_file=self.input_file,
            default_potential_file=self.potential_file,
            default_output_file=self.output_file,
            default_directory=self._directory,
            default_empty_spheres=self.empty_spheres,
            label=self.label,
            counter=self._counter,
            atoms=atoms,
            input_parameters=input_parameters,
            potential=potential,
            input_file=input_file,
            potential_file=potential_file,
            output_file=output_file,
            directory=directory,
            create_subdirs=create_subdirs,
            options=options,
            task=task,
            empty_spheres=empty_spheres,
            allow_temporary_files=allow_temporary_files,
            restart_scf=restart_scf,
        )
        self._counter = prepared.counter
        return prepared

    @staticmethod
    def _is_temporary_directory(directory):
        if directory is False:
            return True
        return isinstance(directory, Directory) and directory.dir is False

    def save_input(
        self,
        atoms=None,
        input_parameters=None,
        potential=None,
        input_file=None,
        potential_file=None,
        output_file=False,
        directory: Union[str, bool, Directory, None] = None,
        create_subdirs: bool = False,
        options=None,
        task=None,
        empty_spheres=None,
        return_files=False,
        restart_scf: bool = False,
    ):
        """Write the input and potential files without executing SPR-KKR.

        Preparation copies `InputParameters`, so generated `POTFIL`/`DATASET`
        values do not change the calculator's persistent defaults. Unlike
        :meth:`calculate`, this method cannot use a temporary directory: the
        returned filenames must remain meaningful after the call.

        Parameters
        ----------
        atoms : ase.Atoms, SPRKKRAtoms or None, optional
            Structure for the input; `None` uses the calculator default.
        input_parameters : InputParameters, str or None, optional
            Input-parameter object, task name or input filename; `None`
            uses the calculator defaults.
        potential : Potential, str, bool or None, optional
            Selected potential, with the same rules as in the constructor.
        input_file, potential_file, output_file : str, os.PathLike, named file, bool or None
            Filenames or templates; `None` inherits calculator defaults.
            `output_file=False` (the default here) suppresses the output file.
            `input_file=False` and `potential_file=False` require temporary
            files to be allowed.
        directory : str, os.PathLike, Directory or None, optional
            Persistent directory for files. `None` uses the calculator default;
            `False` is not supported here.
        create_subdirs : bool, default False
            Create missing parent directories for generated files.
        options : dict or None, optional
            One-shot input-parameter overrides.
        task : str or None, optional
            Task to select on the copied input parameters.
        empty_spheres : bool, str, tuple, mapping or None, optional
            Empty-sphere policy; `None` inherits calculator defaults.
        return_files : bool, default False
            Return open input/output file objects instead of their paths, leaving
            their ownership with the caller. Temporary potential files cannot
            be returned in this mode: specify a persistent `potential_file`.
        restart_scf : bool, default False
            Write a restartable SCF potential; requires an SCF task.

        Returns
        -------
        tuple
            `(input_parameters, input_file, potential_file, output_file)`.
            The first entry is the copied `InputParameters`; input/output entries
            are pathnames by default, or open files if `return_files=True`.
            `potential_file` is a pathname (or `None`); `output_file` may
            be `None` when suppressed.
        """
        selected_directory = self._directory if directory is None else directory
        if self._is_temporary_directory(selected_directory):
            raise ValueError(
                "save_input() requires a persistent working directory; "
                "directory=False is supported by create_run()/calculate() instead"
            )

        prepared = self.prepare(
            atoms=atoms,
            input_parameters=input_parameters,
            potential=potential,
            input_file=input_file,
            potential_file=potential_file,
            output_file=output_file,
            directory=directory,
            create_subdirs=create_subdirs,
            options=options,
            task=task,
            empty_spheres=empty_spheres,
            allow_temporary_files=return_files,
            restart_scf=restart_scf,
        )

        if return_files:
            if prepared.temporary_potential:
                prepared.close()
                raise ValueError(
                    "save_input(return_files=True) cannot transfer ownership of "
                    "a temporary POTFIL pathname; supply potential_file as a path"
                )
            prepared.release_file_ownership()
            prepared.close_directory_context()
            return (
                prepared.input_parameters,
                prepared.input_file,
                prepared.potential_file,
                prepared.output_file,
            )

        input_name = prepared.input_file.name
        output_name = prepared.output_file.name if prepared.output_file else None
        potential_name = prepared.potential_file
        input_parameters = prepared.input_parameters
        prepared.close()
        return input_parameters, input_name, potential_name, output_name

    def create_run(
        self,
        atoms=None,
        input_parameters=None,
        potential=None,
        input_file=None,
        potential_file=None,
        output_file=None,
        directory: Union[str, bool, Directory, None] = None,
        create_subdirs: bool = False,
        options=None,
        task=None,
        empty_spheres: Optional[str | bool] = None,
        print_output=None,
        read_callback: Optional[callable] = None,
        callback=None,
        gdb=False,
        profile: Optional[SPRKKRProfile] = None,
        executor=None,
        project=None,
        restart_scf: bool = False,
        error_file=None,
    ):
        """Prepare one task and return an unstarted runner owning its resources.

        A per-run `executor` takes precedence; if a per-run `profile` is given,
        its backend takes precedence over the calculator-level executor.
        Unlike the success-only callback in :meth:`calculate`, this method's
        `callback` is a terminal callback and may receive partial results.

        Parameters
        ----------
        atoms : ase.Atoms, SPRKKRAtoms or None, optional
            Structure to use instead of the calculator default.
        input_parameters : InputParameters, str or None, optional
            Task parameters or task/input filename for this run.
        potential : Potential, str, bool or None, optional
            Potential selection; see the constructor's sentinel rules.
        input_file, potential_file, output_file : str, os.PathLike, named file, bool or None
            File destinations; `None` inherits calculator defaults.
        directory : str, os.PathLike, Directory, False or None, optional
            Working directory; `False` selects a temporary directory.
        create_subdirs : bool, default False
            Create parent directories as needed.
        options : dict or None, optional
            One-run input-parameter overrides.
        task : str or None, optional
            Override the task of the copied parameters.
        empty_spheres : bool, str, tuple, mapping or None, optional
            Policy for adding empty spheres, or `None` for the calculator default.
        print_output : bool, str or None, optional
            Whether to display output, or `"info"` for concise progress.
            `None` uses the calculator's setting.
        read_callback : callable or None, optional
            Callback observing streamed stdout/stderr.
        callback : callable or None, optional
            Terminal runner callback; may receive an incomplete result on failure.
        gdb : bool, default False
            Launch the executable under GDB for debugging.
        profile : SPRKKRProfile, str or None, optional
            Profile used by this run; `None` uses the calculator profile.
        executor : object or None, optional
            Explicit runner backend, overriding the selected profile's backend.
        project : object or None, optional
            Project context; `None` inherits the calculator project.
        restart_scf : bool, default False
            Prepare for restarting an SCF calculation.
        error_file : str, named file, bool or None, optional
            Stderr destination. `None` uses `<output>.err`; `True` merges
            stderr with stdout, `False` uses a managed temporary file, and a
            path or named file selects an explicit destination.

        Returns
        -------
        CalculationRun
            An unstarted runner. Starting or closing it transfers or releases
            ownership of the prepared resources.
        """
        explicit_profile = profile is not None
        run_profile = resolve_sprkkr_profile(
            self.profile if profile is None else profile
        )
        print_output = first_non_none(print_output, self.print_output)
        if project is None:
            project = self.project
        if executor is None and not explicit_profile:
            executor = self.executor

        prepared = self.prepare(
            atoms=atoms,
            input_parameters=input_parameters,
            potential=potential,
            input_file=input_file,
            potential_file=potential_file,
            output_file=output_file,
            directory=directory,
            create_subdirs=create_subdirs,
            empty_spheres=empty_spheres,
            options=options,
            task=task,
            allow_temporary_files=True,
            restart_scf=restart_scf,
        )
        return prepared.create_run(
            result_context=self,
            profile=run_profile,
            executor=executor,
            project=project,
            print_output=print_output,
            read_callback=read_callback,
            callback=callback,
            gdb=gdb,
            error_file=error_file,
            create_subdirs=create_subdirs,
        )

    def value_or_default(self, name, value):
        """Return calculator default *name* when *value* is ``None``."""
        if value is None:
            value = getattr(self, name)
        return value

    def _update_from_result(self, result, *, adopt_potential=False):
        """Update calculator state from a completed synchronous result."""
        if hasattr(result, "energy"):
            self.results["energy"] = result.energy

        if adopt_potential:
            try:
                potential = result.potential
            except (ValueError, FileNotFoundError):
                return
            if potential is not None:
                self.potential = potential

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
        input_parameters=None,
        potential=None,
        input_file=None,
        potential_file=None,
        output_file=None,
        directory: Union[str, bool, Directory, None] = None,
        create_subdirs: bool = False,
        empty_spheres: Optional[str | bool] = None,
        mpi: bool = None,
        options=None,
        task=None,
        print_output=None,
        read_callback: Optional[callable] = None,
        executable_suffix=None,
        callback=None,
        executable_dir=None,
        gdb=False,
        profile=None,
        executor=None,
        project=None,
        restart_scf: bool = False,
        error_file=None,
    ):
        """Execute one SPR-KKR task synchronously and return its parsed result.

        The ASE `properties` and `system_changes` arguments are accepted but
        do not choose a task or enable incremental updates. A full input state
        is materialized for each run. A completed SCF run may update the
        calculator's default potential; other tasks leave it unchanged.

        Parameters
        ----------
        atoms : ase.Atoms, SPRKKRAtoms or None, optional
            Structure for the run; `None` uses the calculator's physical state.
        properties : sequence of str, default ("energy",)
            Requested ASE properties. Currently only `"energy"` is exposed as
            an ASE property; this argument does not select the SPR-KKR task.
        system_changes : sequence of str, optional
            ASE change indicators, accepted for compatibility but not used
            to perform incremental updates.
        input_parameters : InputParameters, str or None, optional
            Parameters, task name or input filename, overriding calculator defaults.
        potential : Potential, str, bool or None, optional
            Potential selection using the constructor's rules.
        input_file, potential_file, output_file : str, os.PathLike, named file, bool or None
            Filenames or templates, inheriting calculator defaults when `None`.
        directory : str, os.PathLike, Directory, False or None, optional
            Directory for execution; `False` selects a managed temporary directory.
        create_subdirs : bool, default False
            Create missing parent directories for generated files.
        empty_spheres : bool, str, tuple, mapping or None, optional
            Empty-sphere policy; `None` uses the calculator default.
        mpi : bool, int, str, sequence or None, optional
            Override MPI launch settings for this run.
        options : dict or None, optional
            Input-parameter changes applied to this run's copy only.
        task : str or None, optional
            Task selected on the copied input parameters.
        print_output : bool, str or None, optional
            Display process output (or concise `"info"`); `None` uses
            the calculator default.
        read_callback : callable or None, optional
            Callback observing stdout/stderr as the process produces them.
        executable_suffix : str, False or None, optional
            One-run executable-name suffix override.
        callback : callable or None, optional
            Success callback receiving the parsed result after calculator state
            is updated; not invoked when the run raises an exception.
        executable_dir : str, os.PathLike, False or None, optional
            Override the executable search directory for this run.
        gdb : bool, default False
            Run the executable with GDB for debugging.
        profile : SPRKKRProfile, str or None, optional
            Per-run execution profile; `None` uses the calculator profile.
        executor : object or None, optional
            Per-run backend; when specified takes precedence over the profile.
        project : object or None, optional
            Project context; `None` uses the calculator default.
        restart_scf : bool, default False
            Restart a prior SCF calculation from a suitable writable potential.
        error_file : str, named file, bool or None, optional
            Stderr target: `None` uses `<output>.err`, `True` merges it
            with stdout, and `False` requests a managed temporary file.

        Returns
        -------
        TaskResult
            Parsed result for the executed SPR-KKR task. If an energy is available,
            it is also stored in the calculator's ASE `results` dictionary.

        See Also
        --------
        create_run : Construct a runner without executing it immediately.
        save_input : Generate input files without running a task.
        """
        if executor is None and profile is None:
            executor = self.executor
        run_profile = resolve_sprkkr_profile(
            self.profile if profile is None else profile
        ).with_overrides(
            mpi=mpi,
            executable_dir=executable_dir,
            executable_suffix=executable_suffix,
        )
        runner = self.create_run(
            atoms=atoms,
            input_parameters=input_parameters,
            potential=potential,
            input_file=input_file,
            potential_file=potential_file,
            output_file=output_file,
            directory=directory,
            create_subdirs=create_subdirs,
            options=options,
            task=task,
            empty_spheres=empty_spheres,
            print_output=print_output,
            read_callback=read_callback,
            gdb=gdb,
            profile=run_profile,
            executor=executor,
            project=project,
            restart_scf=restart_scf,
            error_file=error_file,
        )
        result = runner.run()
        self._update_from_result(
            result,
            adopt_potential=(
                result.task_name.upper() == "SCF"
                and potential is None
                and atoms is None
            ),
        )
        if callback:
            callback(result)
        return result

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                f"{name} property not implemented"
            )

        if not allow_calculation:
            return None

        self.results.clear()

        result = self.calculate(
            atoms=atoms,
            properties=(name,),
            system_changes=all_changes,
        )

        if atoms is not None and result.task_name.upper() == "SCF":
            self._update_from_result(result, adopt_potential=True)

        if name not in self.results:
            raise PropertyNotImplementedError(
                f"{name} not present in this calculation"
            )

        value = self.results[name]
        return value.copy() if isinstance(value, np.ndarray) else value

    def scf(self, *args, **kwargs):
        """Run an SCF calculation."""
        self.calculate(input_parameters="SCF", *args, **kwargs)

    def phagen(self, *args, **kwargs):
        """Run a PHAGEN calculation."""
        self.calculate(input_parameters="PHAGEN", *args, **kwargs)

    def kkrgen(self, *args, **kwargs):
        """Run a KKRGEN calculation."""
        self.calculate(input_parameters="KKRGEN", *args, **kwargs)

    def kkrspec(self, *args, **kwargs):
        """Run a KKRSPEC calculation."""
        self.calculate(input_parameters="KKRSPEC", *args, **kwargs)

    def kkrch(self, *args, **kwargs):
        """Run a KKRCH calculation."""
        self.calculate(input_parameters="KKRCH", *args, **kwargs)

    def run_xband(self, executable="xband"):
        """Start the external ``xband`` executable detached from calculator IO."""
        subprocess.Popen(executable, stdout=subprocess.DEVNULL)

    def copy_with_potential(self, potential):
        """Return a shallow calculator copy using *potential*."""
        out = copy.copy(self)
        out.potential = potential
        out.results = out.results.copy()
        return out

    @property
    def potential_object(self):
        """Return the calculator potential as a :class:`Potential` object."""
        if isinstance(self.potential, str):
            self.potential = Potential.from_file(self.potential)
        return self.potential

    def change_task(self, task, retain_values=False):
        """Change the persistent default input-parameter task."""
        self.input_parameters.change_task(task, retain_values)


# Keep these imports below the calculator class. Importing InputParameters while
# sprkkr.configuration is still initializing creates a cycle through potentials;
# calculation also imports InputParameters, so it must be delayed for the same
# reason. Method bodies resolve these globals only when called, after module init.
from ..input_parameters.input_parameters import InputParameters  # NOQA: E402
from .calculation import (  # NOQA: E402
    FilenameTemplator as _FilenameTemplator,
    PreparedCalculation,
)


class FilenameTemplator(_FilenameTemplator):
    """Backward-compatible facade for the historical calculator-based API."""

    def __init__(self, calculator, task, symbols):
        self.calculator = calculator
        super().__init__(calculator.label, calculator._counter, task, symbols)

    def __call__(self, template):
        rendered = super().__call__(template)
        self.calculator._counter = self.counter
        return rendered


SPRKKR.InputParameters = InputParameters
SPRKKR.Potential = Potential
