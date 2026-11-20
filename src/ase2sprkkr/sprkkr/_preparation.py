"""Internal state resolution and file materialization for SPR-KKR calculations.

``PreparedCalculation.prepare`` orchestrates these helpers and owns the
resources they acquire. This module depends on neither ``PreparedCalculation``
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

from ..ase.symbols import filename_from_symbols
from ..bindings.empty_spheres import (
    INHOUSE_EMPTY_SPHERES_DEFAULTS, add_empty_spheres,
)
from ..common.temporary import temporary_option_value
from ..input_parameters.input_parameters import InputParameters
from ..potentials.potentials import Potential
from ..potentials.cluster_potentials import ClusterPotential
from .sprkkr_atoms import SPRKKRAtoms


class FilenameTemplator:
    """Render filename templates consistently for one prepared calculation.

    Placeholder values are evaluated lazily and cached, so all filenames
    created by one instance use the same date, task, symbols and counter value.
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
        for placeholder, value in self.replacements.items():
            if placeholder in template:
                template = template.replace(
                    placeholder, self._get(placeholder, value)
                )
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

    def __init__(self, *, directory, mode):
        self._file = tempfile.NamedTemporaryFile(
            mode=mode,
            delete=False,
            dir=directory,
        )
        self.name = self._file.name
        self._removed = False

    def __getattr__(self, name):
        return getattr(self._file, name)

    @property
    def closed(self):
        return self._file.closed

    def close_file(self):
        if not self._file.closed:
            self._file.close()

    def close(self):
        self.close_file()
        if self._removed:
            return
        self._removed = True
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
    name = getattr(value, "name", value if isinstance(value, os.PathLike) else None)
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


def open_file(
    filename,
    directory,
    templator=None,
    mode="w+",
    allow_temporary=True,
    create_subdirs=False,
):
    """Open one path-backed calculation file according to the public contract."""
    if filename is False:
        if not allow_temporary:
            raise ValueError(
                "Creation of temporary files is not allowed in this context"
            )
        return TemporaryNamedFile(directory=directory, mode=mode)

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

    potential_specified = potential is not None and potential is not True
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

    if isinstance(value, (str, os.PathLike)):
        if isinstance(value, str) and InputParameters.is_task_name(value):
            value = InputParameters.create(value)
        else:
            parameter_filename = calculation_path(value, directory)
            parsed = InputParameters.from_file(parameter_filename)

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

    return ResolvedInput(input_parameters, input_file, potential_file, write_input)


def _embedded_empty_sphere_params(value):
    """True for the legacy forms that supply parameters instead of a policy."""
    return isinstance(value, (tuple, collections.abc.Mapping)) or value == "library"


def _empty_sphere_overrides(value, name):
    """Normalize a partial override mapping or the legacy (min, max) tuple."""
    if value is None:
        return {}
    if isinstance(value, collections.abc.Mapping):
        return dict(value)
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} tuple must contain (min_radius, max_radius)")
        return {"min_radius": value[0], "max_radius": value[1]}
    raise ValueError(
        f"{name} must be a dictionary, a (min_radius, max_radius) tuple, "
        "or 'library'"
    )


def resolve_empty_spheres(
    *, input_parameters, atoms, empty_spheres, default_empty_spheres,
    empty_spheres_params=None, default_empty_spheres_params=None,
    configured_empty_spheres_params=None,
):
    """Resolve policy separately from search parameters.

    Search parameters are layered library -> configuration -> calculator ->
    per-call override. Legacy tuple/dict policies force searching and override
    configured values, while 'library' forces built-in finder defaults.
    """
    explicit_policy = empty_spheres is not None
    policy = default_empty_spheres if empty_spheres is None else empty_spheres
    legacy_params = _embedded_empty_sphere_params(policy)

    if legacy_params:
        # Parameters explicitly passed in the same call are contradictory.
        # Calculator defaults are overridable by a per-call legacy shorthand.
        if empty_spheres_params is not None or (
            not explicit_policy and default_empty_spheres_params is not None
        ):
            raise ValueError(
                "Specify empty-sphere parameters in either empty_spheres "
                "or empty_spheres_params, not both"
            )

    if policy == "library":
        enabled = True
        overrides = {}
    elif isinstance(policy, (tuple, collections.abc.Mapping)):
        enabled = True
        overrides = _empty_sphere_overrides(policy, "empty_spheres")
    else:
        enabled = policy
        overrides = {}

    if enabled == "auto":
        enabled = (
            input_parameters.task_name == "SCF"
            and atoms is not None
            and (
                not atoms.has_potential()
                or atoms.potential.SCF_INFO.SCFSTATUS() == "START"
            )
        )
        if enabled:
            enabled = not any(site.is_vacuum() for site in atoms.sites)

    if enabled not in (True, False, None):
        raise ValueError(
            "empty_spheres must be True, False, 'auto', 'library', "
            "a parameter dictionary, or a (min_radius, max_radius) tuple"
        )
    if not enabled or atoms is None:
        return

    use_library_defaults = policy == "library"
    selected_params = (
        empty_spheres_params
        if empty_spheres_params is not None
        else default_empty_spheres_params
    )
    if selected_params == "library":
        use_library_defaults = True
        selected_params = None

    params = {} if use_library_defaults else dict(configured_empty_spheres_params or {})
    if not legacy_params and not use_library_defaults:
        params.update(_empty_sphere_overrides(selected_params, "empty_spheres_params"))
    params.update(overrides)

    # XBand parameters cannot be silently ignored by an auto-selected
    # es_finder backend (which accepts a different argument vocabulary).
    xband_keys = set(params) & set(INHOUSE_EMPTY_SPHERES_DEFAULTS)
    if xband_keys:
        method = params.get("method", "auto")
        if method == "es_finder":
            raise ValueError(
                "XBand empty_spheres_params are incompatible with method='es_finder'"
            )
        if method == "auto":
            params["method"] = "inhouse"
        min_radius = params.get("min_radius", INHOUSE_EMPTY_SPHERES_DEFAULTS["min_radius"])
        max_radius = params.get("max_radius", INHOUSE_EMPTY_SPHERES_DEFAULTS["max_radius"])
        if min_radius < 0 or max_radius < min_radius:
            raise ValueError("Empty-sphere radii must satisfy 0 <= min_radius <= max_radius")
        if params.get("max_spheres", 1) < 1:
            raise ValueError("max_spheres must be positive")
        if "mesh" in params:
            mesh = params["mesh"]
            mesh = (mesh,) * 3 if isinstance(mesh, int) else tuple(mesh)
            if len(mesh) != 3 or any(value < 1 for value in mesh):
                raise ValueError("Empty-sphere mesh must contain three positive integers")
    add_empty_spheres(atoms, **params)


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


class CalculationFiles:
    """Open, validate and register files with shared preparation settings."""

    def __init__(
        self, directory, templator, resources, *, allow_temporary, create_subdirs
    ):
        self.directory = directory
        self.templator = templator
        self.resources = resources
        self.allow_temporary = allow_temporary
        self.create_subdirs = create_subdirs
        self._owned_ids = set()

    def open(self, filename, role, *, mode="w+"):
        file = open_file(
            filename,
            self.directory,
            self.templator,
            mode=mode,
            allow_temporary=self.allow_temporary,
            create_subdirs=self.create_subdirs,
        )
        # Own the descriptor before validation, so invalid streams are closed
        # along with all other acquired resources if preparation fails.
        own_file(self.resources, self._owned_ids, file)
        require_named_file(file, role)
        return file

    def open_input(self, resolved, default_filename):
        filename = default_filename if resolved.filename is None else resolved.filename
        if filename is True:
            filename = "%l%_%a_%T.inp"
        return self.open(filename, "input_file", mode="w+" if resolved.write else "r")

    def open_output(self, filename, *, default_filename, reused_input):
        if filename is None:
            filename = default_filename
        if filename is False or filename is None:
            return None
        filename = filename_from_input(
            filename, reused_input=reused_input, suffix=".out", default="%a_%T.out"
        )
        return self.open(filename, "output_file", mode="wb")


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


def write_potential(
    resolved, filename, files, *, default_filename, task_name, restart_scf
):
    """Serialize POTFIL without retaining temporary changes to SCF status."""
    potential = resolved.potential
    if potential is None:
        return filename, False

    if filename is None:
        filename = default_filename
    if filename is True:
        filename = "%a.pot"
    handle = files.open(filename, "potential_file")
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

    flush = getattr(handle, "flush", None)
    if flush is not None:
        flush()
    if isinstance(handle, TemporaryNamedFile):
        handle.close_file()
    else:
        handle.close()
    # Caller-supplied handles may have names relative to the Python process's
    # cwd. Files opened above already have absolute names.
    name = named_file_name(handle)
    if not os.path.isabs(name):
        name = os.path.abspath(name)
    return name, filename is False



def write_cluster_potentials(
    resolved, host_filename, files, *, default_filename, task_name,
    input_parameters, restart_scf=False,
):
    """Write an embedded cluster and its host as separate native POTFILs."""
    cluster = resolved.potential
    host = cluster.host
    if not isinstance(host, Potential):
        raise ValueError(
            "A ClusterPotential calculation requires a host Potential"
        )

    host_file, temporary_host = write_potential(
        ResolvedPotential(host, host.atoms, host_filename),
        host_filename,
        files,
        default_filename=default_filename,
        task_name=task_name,
        restart_scf=False,
    )
    cluster_target = input_parameters.CONTROL.POTFIL_CLU.result
    if not cluster_target:
        if temporary_host:
            cluster_target = False
        else:
            cluster_target = str(Path(host_file).with_suffix(".clu.pot"))
    if cluster_target is not False:
        intended = calculation_path(cluster_target, files.directory)
        if os.path.abspath(host_file) == intended:
            raise ValueError("POTFIL and POTFIL_CLU must be different files")

    handle = files.open(cluster_target, "cluster_potential_file")
    # Restart is a serialization intent; do not mutate the canonical
    # converged cluster potential retained by the calling Atoms.
    if restart_scf:
        cluster_to_write = ClusterPotential.from_atoms(
            resolved.atoms, scf_status="START"
        )
    else:
        cluster_to_write = cluster
    cluster_to_write.save_to_file(handle)
    flush = getattr(handle, "flush", None)
    if flush is not None:
        flush()
    if isinstance(handle, TemporaryNamedFile):
        handle.close_file()
    else:
        handle.close()
    cluster_file = named_file_name(handle)
    if not os.path.isabs(cluster_file):
        cluster_file = os.path.abspath(cluster_file)
    return host_file, cluster_file, temporary_host, cluster_target is False


def write_input(resolved, handle, potential_file, atoms, *, cluster_potential_file=None):
    """Serialize the snapshot, letting filename mappings render relative paths."""
    if not resolved.write:
        return

    parameters = resolved.parameters
    if potential_file:
        parameters.CONTROL.POTFIL = potential_file
    if cluster_potential_file:
        parameters.CONTROL.POTFIL_CLU = cluster_potential_file
    if not parameters.CONTROL.DATASET.is_set():
        parameters.CONTROL.DATASET.result = Path(named_file_name(handle)).stem
    parameters.save_to_file(handle, atoms)
    handle.seek(0)
