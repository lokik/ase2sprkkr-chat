"""Internal state resolution and file materialization for SPR-KKR calculations.

``CalculationPlan`` orchestrates these helpers and owns the
resources acquired during materialization. This module depends on neither ``PreparedCalculation``
nor the execution subsystem.
"""

from __future__ import annotations

import collections.abc
import os
import re
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pyparsing import ParseBaseException
from kkrunner.files.versions import FileVersion

from ..ase.symbols import filename_from_symbols
from ..bindings.empty_spheres import add_empty_spheres
from ..common.temporary import temporary_option_value
from ..input_parameters.input_parameters import InputParameters
from ..potentials.potentials import Potential
from ..potentials.cluster_potentials import ClusterPotential
from .executables import resolve_task_executable, task_system_error
from .sprkkr_atoms import SPRKKRAtoms, system_kind_for_atoms



class FilenameTemplator:
    """Render filename templates consistently for one prepared calculation.

    ``%%`` represents a literal percent sign. Placeholder values are evaluated
    lazily and cached, so all filenames created by one instance use the same
    date, task, symbols and counter value.
    """

    def __init__(self, label, counter, task, symbols):
        self.data = {}
        self.label = label
        self.counter = counter
        self.task = task
        self.symbols = symbols

    def _get(self, name, default):
        if name not in self.data:
            self.data[name] = default(self)
        return self.data[name]

    def _advance_counter(self):
        self.counter += 1
        return str(self.counter)

    replacements = {
        "%d": lambda self: datetime.today().strftime("%Y-%m-%d_%H:%M"),
        "%t": lambda self: str(self.task()),
        "%T": lambda self: str(self.task()).upper(),
        "%a": lambda self: str(self.symbols()),
        "%l": lambda self: self.label or "",
        "%c": lambda self: self._advance_counter(),
    }

    def __call__(self, template):
        return "%".join(
            self._render(part, leading=index == 0)
            for index, part in enumerate(template.split("%%"))
        )

    def _render(self, template, *, leading):
        for placeholder, value in self.replacements.items():
            if placeholder in template:
                template = template.replace(
                    placeholder, self._get(placeholder, value)
                )
        if leading:
            template = re.sub("^%_", "", template)
        template = re.sub("%_(%_)*", "_", template)
        return template.replace("%_", "_")


class TemporaryNamedFile:
    """Path-backed temporary file whose pathname lives until ``close()``.

    ``NamedTemporaryFile(delete=True)`` cannot be reopened reliably by another
    process on all supported platforms. SPR-KKR and remote execution both need
    to reopen prepared files by pathname, therefore the underlying file is
    created with ``delete=False`` and this small owner removes it explicitly.
    ``close_file()`` closes only the descriptor while retaining the pathname;
    it is used after POTFIL serialization while the prepared calculation still
    owns the temporary artifact.
    """

    def __init__(self, *, directory, mode, path=None):
        if path is None:
            self._file = tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=directory)
        else:
            # The planner chose a unique name without creating a file. Exclusive
            # creation protects it even if some external process used that name.
            self._file = open(path, mode.replace("w", "x", 1),
                              opener=lambda name, flags: os.open(name, flags, 0o600))
        self.name = self._file.name
        self._delete_on_close = True

    def __getattr__(self, name):
        return getattr(self._file, name)

    @property
    def closed(self):
        return self._file.closed

    def close_file(self):
        if not self._file.closed:
            self._file.close()

    def preserve(self):
        """Relinquish pathname deletion, but not ownership of the open handle."""
        self._delete_on_close = False

    def close(self):
        self.close_file()
        if not self._delete_on_close:
            return
        self._delete_on_close = False
        try:
            os.unlink(self.name)
        except FileNotFoundError:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def calculation_path(path, directory):
    """Resolve a user path relative to the concrete calculation directory."""
    return os.path.abspath(os.path.join(directory, os.fspath(path)))


def named_file_name(value):
    """Return a path-backed file name, or ``None`` for anonymous streams."""
    if value is None:
        return None
    name = value if isinstance(value, (str, os.PathLike)) else getattr(value, "name", None)
    if name is None or isinstance(name, int):
        return None
    try:
        return os.fspath(name)
    except TypeError:
        return None


def require_named_file(value, role):
    name = named_file_name(value)
    if name is None:
        raise ValueError(
            f"SPR-KKR {role} must be backed by a filesystem path. "
            f"Pass a pathname or {role}=False to request a managed temporary file."
        )
    return name


def execution_path(value, directory):
    """Normalize declarations and open handles in their respective path domains."""
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        return calculation_path(value, directory)
    return os.path.abspath(require_named_file(value, "execution file"))


def open_file(
    filename,
    directory,
    templator=None,
    mode="w+",
    allow_temporary=True,
    create_subdirs=False,
    planned_path=None,
):
    """Open one path-backed calculation file according to the public contract."""
    if filename is False:
        if not allow_temporary:
            raise ValueError(
                "Creation of temporary files is not allowed in this context"
            )
        return TemporaryNamedFile(directory=directory, mode=mode, path=planned_path)

    if not isinstance(filename, (str, os.PathLike)):
        return filename

    filename = os.fspath(filename)
    if templator:
        filename = templator(filename)
    filename = calculation_path(filename, directory or os.curdir)
    if create_subdirs:
        parent = os.path.dirname(filename)
        if parent:
            os.makedirs(parent, exist_ok=True)
    return open(filename, mode)


def own_file(stack, owned_ids, file):
    """Register ownership of a file-like object exactly once."""
    if file is None or id(file) in owned_ids:
        return
    close = getattr(file, "close", None)
    if close is None:
        return
    owned_ids.add(id(file))
    stack.callback(close)


@dataclass(frozen=True)
class ResolvedPotential:
    """A potential to serialize, or an existing POTFIL to reference.

    ``potential=None`` means no serialization is needed. In that case
    ``filename`` is either an existing potential path or ``None`` to use the
    POTFIL from the input parameters. Otherwise it selects the output file.
    """

    potential: Potential | ClusterPotential | None
    atoms: SPRKKRAtoms | None
    filename: object


@dataclass(frozen=True)
class ResolvedInput:
    """Parameter snapshot and the concrete decision to write or reuse input."""

    parameters: InputParameters
    filename: object
    potential_file: object
    write: bool
    source_version: FileVersion | None = None


def resolve_potential_and_atoms(
    *,
    potential,
    atoms,
    potential_file,
    default_potential,
    default_atoms,
    restart_scf,
    directory,
):
    """Resolve the concrete structure/potential pair for preparation."""
    if potential is None and atoms is None:
        potential = default_potential
    if potential is None:
        potential = atoms is not None or default_atoms is not None

    # With potential=False, atoms are optional geometry metadata for native
    # task validation; they never cause a potential to be generated or rewritten.
    potential_specified = potential is not None and potential is not True and potential is not False
    if atoms is not None and potential_specified:
        raise ValueError(
            "You can not provide both a potential and atoms object to the SPRKKR "
            "calculate method; one is generated from the other."
        )

    atoms = default_atoms if atoms is None else atoms
    if potential is True:
        if atoms is None:
            raise ValueError(
                "Potential is True, which requests generation from atoms, but no "
                "atoms object has been supplied."
            )
        potential = Potential.from_atoms(atoms)
    elif potential is False:
        if potential_file is not None:
            raise ValueError(
                "When potential=False, POTFIL from the input parameters is used; "
                "potential_file therefore must not be specified."
            )
        potential = None
    elif potential is None:
        raise ValueError(
            "No potential is available. Supply a Potential, atoms, potential=True "
            "to generate one from atoms, or potential=False to use POTFIL from the "
            "input parameters."
        )
    elif isinstance(potential, (str, os.PathLike)):
        source = calculation_path(potential, directory)
        if potential_file is not None or restart_scf:
            potential = Potential.from_file(source)
        else:
            potential_file = source
            potential = None

    if isinstance(potential, (Potential, ClusterPotential)):
        atoms = potential.atoms
    elif atoms is not None:
        atoms = SPRKKRAtoms.promote_ase_atoms(atoms)

    return ResolvedPotential(potential, atoms, potential_file)


def potential_from_parameters(parameters, directory):
    """Resolve POTFIL against the directory in which SPR-KKR will run.

    SPR-KKR reads the input through stdin; the input file's own location does
    not change the base directory for references inside it.
    """
    potfil = parameters.CONTROL.POTFIL.result
    if not potfil:
        return None
    return calculation_path(potfil, directory)


def resolve_input_parameters(
    *,
    input_parameters,
    default_input_parameters,
    input_file,
    potential,
    options,
    task,
    directory,
):
    """Resolve and snapshot task parameters for one concrete calculation."""
    value = default_input_parameters if input_parameters is None else input_parameters
    potential_file = potential.filename
    potential_missing = potential.potential is None
    write_input = True
    source_version = None

    if isinstance(value, (str, os.PathLike)):
        if isinstance(value, str) and InputParameters.is_task_name(value):
            value = InputParameters.create(value)
        else:
            parameter_filename = calculation_path(value, directory)
            version = FileVersion.capture(parameter_filename, require_digest=True)
            parsed = InputParameters.from_file(parameter_filename)
            if not version.is_current():
                raise ValueError(f"Input parameters changed while planning: {parameter_filename}")

            reuse_input = (
                potential_missing
                and not options
                and not task
                and input_file is None
                and (
                    potential_file is None
                    or potential_file == potential_from_parameters(parsed, directory)
                )
            )
            if reuse_input:
                input_file = parameter_filename
                write_input = False
                source_version = version
            value = parsed

    if value is None:
        raise ValueError("No input parameters are available for the calculation")

    input_parameters = value.copy(copy_values=True)
    input_parameters.directory = directory
    if task:
        input_parameters.change_task(task)
    if options:
        input_parameters.set(options, unknown="find")

    if isinstance(potential.potential, ClusterPotential):
        cluster_type = input_parameters.CONTROL.CLUTYPE()
        if cluster_type not in (None, "embedded"):
            raise ValueError(
                "ClusterPotential requires CONTROL.CLUTYPE=embedded"
            )
        input_parameters.CONTROL.CLUTYPE = "embedded"

    if potential_missing and potential_file is None:
        potential_file = potential_from_parameters(input_parameters, directory)

    if (
        not input_parameters.CONTROL.POTFIL.result
        and potential_missing
        and not potential_file
    ):
        raise ValueError(
            "Potential in the input parameters is not set and no atoms or "
            "Potential object has been supplied."
        )

    return ResolvedInput(input_parameters, input_file, potential_file, write_input, source_version)


def validate_task_system(task, resolved_input, resolved_potential):
    """Check known geometry before opening any calculation files."""
    parameters = resolved_input.parameters
    resolve_task_executable(task, parameters)
    kind = system_kind_for_atoms(resolved_potential.atoms)
    # For a path-only embedded calculation the host POTFIL is not itself
    # an embedded structure; the complete pair is checked during source loading.
    if kind is None and parameters.cluster_type == "embedded":
        kind = "embedded"
    # When a task does not support both periodic geometries, inspect an
    # available potential to reject a known unsupported system before creating
    # output files. A parser failure means the geometry remains unknown:
    # path-only native execution must not require full Python parsing.
    filename = resolved_input.potential_file
    if kind is None and filename and not {"2d", "3d"}.issubset(task.supports):
        try:
            potential = Potential.from_file(filename, atoms=False)
        except (FileNotFoundError, ParseBaseException):
            pass
        else:
            if isinstance(potential, ClusterPotential):
                kind = "embedded"
            else:
                kind = potential.LATTICE.SYSDIM().lower()
                if kind == "0d":
                    kind = "cluster"
    error = task_system_error(task, kind)
    if error:
        raise ValueError(error)


def resolve_empty_spheres(
    *, input_parameters, atoms, empty_spheres, default_empty_spheres
):
    """Apply the configured empty-sphere preparation policy."""
    if empty_spheres is None:
        empty_spheres = default_empty_spheres
    if empty_spheres == "auto":
        empty_spheres = (
            input_parameters.task_name == "SCF"
            and atoms is not None
            and (
                not atoms.has_potential()
                or atoms.potential.SCF_INFO.SCFSTATUS() == "START"
            )
        )
        if empty_spheres:
            empty_spheres = not any(site.is_vacuum() for site in atoms.sites)

    if not empty_spheres or atoms is None:
        return

    if isinstance(empty_spheres, tuple):
        if len(empty_spheres) != 2:
            raise ValueError(
                "empty_spheres tuple must contain (min_radius, max_radius)"
            )
        empty_spheres = {
            "min_radius": empty_spheres[0],
            "max_radius": empty_spheres[1],
        }
    if not isinstance(empty_spheres, collections.abc.Mapping):
        empty_spheres = {}

    add_empty_spheres(atoms, **empty_spheres)


def symbols_for_filenames(
    atoms,
    input_parameters,
    input_file,
    potential_file,
    default_input_file,
    default_potential_file,
):
    """Return a stable best-effort system name for filename templates."""
    if atoms is not None:
        return filename_from_symbols(atoms.symbols)
    for name in "POTFIL", "DATASET":
        option = input_parameters.CONTROL[name]
        if option.is_set():
            return Path(option()).stem
    for candidate in (
        input_file,
        potential_file,
        default_input_file,
        default_potential_file,
    ):
        if isinstance(candidate, (str, os.PathLike)):
            stem = Path(candidate).stem
            if "%" not in stem:
                return stem
    return "sprkkr"


def filename_from_input(selected, *, reused_input, suffix, default):
    """Choose an automatic filename, preserving the reused input's basename."""
    if selected is not True:
        return selected
    if reused_input is None:
        return default
    base = os.path.basename(reused_input)
    for extension in (".inp", ".in"):
        if base.endswith(extension):
            return base[:-len(extension)] + suffix
    return base + suffix


@dataclass(frozen=True)
class PlannedFile:
    """One concrete filename and opening intent; no descriptor is opened."""

    selection: object
    path: str
    role: str
    mode: str

    @property
    def temporary(self):
        return self.selection is False


class CalculationFiles:
    """Resolve filenames once and later materialize exactly those declarations."""

    def __init__(
        self, directory, templator, resources, *, allow_temporary, create_subdirs
    ):
        self.directory = directory
        self.templator = templator
        self.resources = resources
        self.allow_temporary = allow_temporary
        self.create_subdirs = create_subdirs
        self._owned_ids = set()
        self.temporary_files = []

    def plan(self, filename, role, *, mode="w+"):
        if filename is False:
            if not self.allow_temporary:
                raise ValueError("Creation of temporary files is not allowed in this context")
            path = os.path.join(self.directory, "tmp-sprkkr-" + uuid4().hex)
        elif isinstance(filename, (str, os.PathLike)):
            path = calculation_path(self.templator(os.fspath(filename)), self.directory)
        else:
            # Borrow the descriptor until materialization. Closing it during
            # rejected planning/submission could flush bytes without a reservation.
            path = os.path.abspath(require_named_file(filename, role))
        return PlannedFile(filename, path, role, mode)

    def open(self, planned):
        selection = planned.selection
        filename = planned.path if isinstance(selection, (str, os.PathLike)) else selection
        file = open_file(
            filename, self.directory, mode=planned.mode,
            allow_temporary=self.allow_temporary, create_subdirs=self.create_subdirs,
            planned_path=planned.path if planned.temporary else None,
        )
        own_file(self.resources, self._owned_ids, file)
        if isinstance(file, TemporaryNamedFile):
            self.temporary_files.append(file)
        require_named_file(file, planned.role)
        return file

    def plan_input(self, resolved, default_filename):
        filename = default_filename if resolved.filename is None else resolved.filename
        if filename is True:
            filename = "%l%_%a_%T.inp"
        return self.plan(filename, "input_file", mode="w+" if resolved.write else "r")

    def plan_output(self, filename, *, default_filename, reused_input):
        if filename is None:
            filename = default_filename
        if filename is False or filename is None:
            return None
        filename = filename_from_input(
            filename, reused_input=reused_input, suffix=".out", default="%a_%T.out"
        )
        return self.plan(filename, "output_file", mode="wb")


def validate_restart(potential, task_name, restart_scf):
    if not restart_scf:
        return
    if task_name != "SCF":
        raise ValueError("restart_scf=True is only valid for an SCF calculation.")
    if not isinstance(potential, (Potential, ClusterPotential)):
        raise ValueError(
            "restart_scf=True requires a writable Potential/ClusterPotential "
            "or atoms from which one can be generated."
        )


def plan_potential(resolved, filename, files, *, default_filename):
    """Choose POTFIL without creating it, or reuse an existing potential."""
    if resolved.potential is None:
        return None
    if filename is None:
        filename = default_filename
    if filename is True:
        filename = "%a.pot"
    return files.plan(filename, "potential_file")


def write_potential(resolved, planned, files, *, task_name, restart_scf):
    """Serialize a previously declared POTFIL, without changing SCF status."""
    if planned is None:
        return
    potential = resolved.potential
    handle = files.open(planned)
    status_context = nullcontext()
    if (
        isinstance(potential, Potential)
        and task_name != "SCF"
        and potential.SCF_INFO.SCFSTATUS() == "START"
        and potential.is_scf_converged(resolved.atoms)
    ):
        status_context = temporary_option_value(
            potential.SCF_INFO.SCFSTATUS, "CONVERGED"
        )
    with status_context:
        potential.save_to_file(handle, resolved.atoms, restart_scf=restart_scf)

    if isinstance(handle, TemporaryNamedFile):
        handle.close_file()
    else:
        handle.close()



def plan_cluster_potentials(resolved, host_filename, files, *, default_filename, input_parameters):
    """Resolve both embedded potential paths before either file is opened."""
    host = resolved.potential.host
    if not isinstance(host, Potential):
        raise ValueError("A ClusterPotential calculation requires a host Potential")
    host_plan = plan_potential(
        ResolvedPotential(host, host.atoms, host_filename),
        host_filename, files, default_filename=default_filename,
    )
    cluster_target = input_parameters.CONTROL.POTFIL_CLU.result
    if not cluster_target:
        cluster_target = False if host_plan.temporary else str(Path(host_plan.path).with_suffix(".clu.pot"))
    cluster_plan = files.plan(cluster_target, "cluster_potential_file")
    if host_plan.path == cluster_plan.path:
        raise ValueError("POTFIL and POTFIL_CLU must be different files")
    return host_plan, cluster_plan


def write_cluster_potentials(resolved, host_plan, cluster_plan, files, *, task_name, restart_scf=False):
    """Write the two already planned native POTFILs."""
    cluster = resolved.potential
    host = cluster.host
    write_potential(ResolvedPotential(host, host.atoms, host_plan.path), host_plan, files,
                    task_name=task_name, restart_scf=False)
    handle = files.open(cluster_plan)
    # Restart is a serialization intent; do not mutate the canonical
    # converged cluster potential retained by the calling Atoms.
    if restart_scf:
        cluster_to_write = ClusterPotential.from_atoms(
            resolved.atoms, scf_status="START"
        )
    else:
        cluster_to_write = cluster
    cluster_to_write.save_to_file(handle)
    if isinstance(handle, TemporaryNamedFile):
        handle.close_file()
    else:
        handle.close()



def configure_input(resolved, input_name, potential_file, *, cluster_potential_file=None):
    """Set file-derived parameter values as part of the write-free plan."""
    if not resolved.write:
        return
    parameters = resolved.parameters
    if potential_file:
        parameters.CONTROL.POTFIL = potential_file
    if cluster_potential_file:
        parameters.CONTROL.POTFIL_CLU = cluster_potential_file
    if not parameters.CONTROL.DATASET.is_set():
        parameters.CONTROL.DATASET.result = Path(input_name).stem


def write_input(resolved, handle):
    """Serialize the already derived parameter snapshot, without recomputing it."""
    if resolved.write:
        resolved.parameters.save_to_file(handle, False)
        handle.seek(0)
