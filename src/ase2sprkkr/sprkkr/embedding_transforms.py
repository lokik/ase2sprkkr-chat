"""Host transformations with exact cluster-site remapping; never mutate inputs."""
from __future__ import annotations

from copy import copy
from .sprkkr_atoms import SPRKKRAtoms
from ..potentials.potentials import PeriodicPotential
import numpy as np
from .embedding import EMBEDDING_ARRAY, EMBEDDING_HOST, embedding_data, host_atoms, host_dimension, translation_basis


def embedded_host_transform_capability(cluster, action):
    """Whether an operation has a lossless host-site correspondence."""
    if not isinstance(cluster, SPRKKRAtoms) or cluster.embedding_host is None or EMBEDDING_ARRAY not in cluster.arrays:
        return False, "The embedded structure needs a host and embedding indices."
    host = cluster.embedding_host
    dim = host_dimension(host)
    if action in ("sites", "lattice", "scale"):
        atoms = host_atoms(host)
        if dim == 2 and (not isinstance(atoms, SPRKKRAtoms) or not atoms.regions):
            return False, "2D host edits require left/right bulk regions."
        return True, ""
    if action == "repeat":
        if dim != 3:
            return False, "Integer host supercells currently require a 3D host."
        if np.any(embedding_data(cluster)[:, 4:6]):
            return False, "The native extra translation vectors cannot be remapped."
        return True, ""
    if action == "rotate":
        return False, "Miller-index rotation changes the host basis and needs a site mapping."
    return False, "Unknown transformation"


def _new_host(old_host, updated):
    from ..potentials.potential_base import Potential
    original = host_atoms(old_host)
    result = updated.copy()
    SPRKKRAtoms.promote_ase_atoms(result)
    if not np.array_equal(result.pbc, original.pbc):
        raise ValueError("A host edit may not change periodicity")
    previous_regions = original.regions if isinstance(original, SPRKKRAtoms) else {}
    if previous_regions and not result.regions:
        if len(result) != len(original):
            raise ValueError("2D host-site count changed")
        matrix = np.linalg.solve(np.asarray(original.cell), np.asarray(result.cell))
        regions = []
        for previous in previous_regions.values():
            region = copy(previous)
            region.atoms = None
            region.incomplete_cell = type(previous.incomplete_cell)(
                np.asarray(previous.incomplete_cell) @ matrix
            )
            regions.append(region)
        result.set_regions(regions)
    if host_dimension(old_host) == 2 and not result.regions:
        raise ValueError("2D host requires valid left/right bulk regions")
    result._potential = None
    if result.are_sites_inited():
        result.sprkkr_reset_potential()
    return Potential.from_atoms(result) if isinstance(old_host, PeriodicPotential) else result


def rebase_embedded_host(cluster, updated_host, *, repeat=(1, 1, 1),
                         displacement_mode="affine"):
    """Replace the host, preserving substitutions and mapping old sites exactly.

    No new cluster sites are created when repeating a host. For an ordinary
    edit the host order/count must not change; for integer 3D supercells the
    embedding translations are divided into within-cell and supercell terms.
    Existing radial data become invalid: both structures restart at START.
    """
    if not isinstance(cluster, SPRKKRAtoms):
        raise TypeError("Expected an embedded SPRKKRAtoms structure")
    old_host = cluster.embedding_host
    if old_host is None:
        raise ValueError("Embedded structure has no host")
    source = host_atoms(old_host)
    rows = embedding_data(cluster).copy()
    if np.any(rows[:, 0] < 0) or np.any(rows[:, 0] >= len(source)):
        raise ValueError("Embedding host-site index out of range")
    repeat = tuple(repeat)
    if len(repeat) != 3 or any(type(x) not in (int, np.int64, np.int32) or x < 1 for x in repeat):
        raise ValueError("Host repeat requires positive integer factors")
    if displacement_mode not in ("affine", "cartesian"):
        raise ValueError("Unknown displacement transformation")
    if repeat != (1, 1, 1):
        ok, reason = embedded_host_transform_capability(cluster, "repeat")
        if not ok:
            raise ValueError(reason)
    new_atoms = host_atoms(updated_host)
    if len(new_atoms) != len(source) * np.prod(repeat):
        raise ValueError("Transformed host site count does not match the repeat")
    before = (np.asarray(source.positions)[rows[:, 0]]
              + rows[:, 1:6] @ translation_basis(old_host))
    displacement = np.asarray(cluster.positions) - before
    if displacement_mode == "affine":
        old_cell = np.asarray(source.cell, dtype=float)
        target_cell = np.asarray(new_atoms.cell, dtype=float) / np.asarray(repeat)[:, None]
        displacement = displacement @ np.linalg.solve(old_cell, target_cell)
    if repeat != (1, 1, 1):
        factors = np.asarray(repeat, dtype=int)
        q, rem = np.divmod(rows[:, 1:4], factors)
        block = (rem[:, 0] * factors[1] + rem[:, 1]) * factors[2] + rem[:, 2]
        rows[:, 0] += len(source) * block
        rows[:, 1:4] = q
    host = _new_host(old_host, new_atoms)
    result = cluster.copy()
    result._potential = None
    result.set_array(EMBEDDING_ARRAY, rows)
    result.info[EMBEDDING_HOST] = host
    result.positions[:] = (np.asarray(host_atoms(host).positions)[rows[:, 0]]
                           + rows[:, 1:6] @ translation_basis(host) + displacement)
    if result.are_sites_inited():
        result.sprkkr_reset_potential()
    result.info["sprkkr_embedding_scf_status"] = "START"
    return result


def repeat_embedded_host(cluster, factors):
    """Build an integer 3D host supercell without repeating the cluster."""
    factors = tuple(factors)
    ok, reason = embedded_host_transform_capability(cluster, "repeat")
    if not ok:
        raise ValueError(reason)
    if len(factors) != 3 or any(type(f) is not int or f < 1 for f in factors):
        raise ValueError("Only positive integer host repeats are allowed")
    return rebase_embedded_host(
        cluster, host_atoms(cluster.embedding_host).repeat(factors), repeat=factors
    )
