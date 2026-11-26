"""Resolve existing native SPR-KKR input sources before launching a calculation.

InputParameters owns the native settings, Potential owns parsing and structure,
and this boundary binds both to concrete versioned files. File staging and
calculation-file generation belong to the runner and preparation respectively.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kkrunner.files import FileVersion

from ..input_parameters.input_parameters import InputParameters
from ..potentials.potentials import PeriodicPotential, Potential
from ..potentials.cluster_potentials import EmbeddedPotential


@dataclass(frozen=True)
class ExistingPotentialInput:
    """One parsed native input and the versions used to construct its structure."""

    potential: Potential
    parameters: InputParameters
    versions: tuple[FileVersion, ...]


def existing_potential_files(parameters, directory):
    """Resolve required native potential paths relative to the execution directory."""
    control = parameters.CONTROL
    names = ["POTFIL"]
    if parameters.cluster_type == "embedded":
        names.append("POTFIL_CLU")
    elif "POTFIL_CLU" in control and control.POTFIL_CLU():
        raise ValueError("CONTROL.POTFIL_CLU requires CONTROL.CLUTYPE=embedded")

    directory = Path(directory).expanduser().absolute()
    files = {}
    for name in names:
        value = control[name]()
        if not value:
            raise ValueError(f"Set CONTROL.{name} to an existing file.")
        path = Path(os.path.abspath(directory / value))
        if not path.is_file():
            raise ValueError(f"File for CONTROL.{name} does not exist: {path}")
        files[name] = path

    # The shape function can also be produced by full-potential SCF. It is
    # not a source of the structure; task cache/input declarations own it.
    if control.SFNFIL.is_set():
        shape = Path(os.path.abspath(directory / control.SFNFIL()))
        if shape.is_file():
            files["SFNFIL"] = shape
    return files


def load_existing_potential_input(parameters, directory):
    """Read a native potential once and retain immutable input-file identities.

    The resulting versions must be compared to KKRunner's staged manifest
    immediately before submission. Never reinterpret the source files then.
    """
    files = existing_potential_files(parameters, directory)
    source_paths = [files[name] for name in ("POTFIL", "POTFIL_CLU") if name in files]
    versions = tuple(FileVersion.capture(path, require_digest=True) for path in source_paths)

    host = PeriodicPotential.from_file(files["POTFIL"], check_file_unchanged=True)
    potential = host
    if "POTFIL_CLU" in files:
        if not host.is_scf_converged():
            raise ValueError("Converge the host potential before running an embedded calculation.")
        potential = EmbeddedPotential.from_file(
            files["POTFIL_CLU"], host=host, check_file_unchanged=True,
        )

    if any(not version.is_current() for version in versions):
        raise ValueError("A source potential changed while being loaded.")

    prepared = parameters.copy(copy_values=True)
    for name, path in files.items():
        prepared.CONTROL[name].set(str(path))
    if not prepared.CONTROL.DATASET.is_set():
        prepared.CONTROL.DATASET = potential.atoms.get_chemical_formula(empirical=True)
    return ExistingPotentialInput(potential, prepared, versions)
