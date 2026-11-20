"""Materialize edited cluster sites and select native restart data."""

from dataclasses import dataclass, field

import numpy as np

from .embedding import embedding_data, host_atoms
from .embedding_geometry import HostSite
from .radial import _equivalent_meshes
from .sprkkr_atoms import SPRKKRAtoms


@dataclass(eq=False)
class EmbeddedSiteParameters:
    """Chemical and geometrical values of one site in an embedded cluster."""

    occupation: dict[str, float]
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(3))
    magnetisation: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __eq__(self, other):
        if not isinstance(other, EmbeddedSiteParameters):
            return NotImplemented
        return (
            self.occupation == other.occupation
            and np.array_equal(self.displacement, other.displacement)
            and np.array_equal(self.magnetisation, other.magnetisation)
        )


def _radial_sources(original):
    """Index original and host types, preferring original cluster data."""
    by_site, by_element = {}, {}
    host = original.embedding_host
    for row, site in zip(embedding_data(original), original.sites):
        key = HostSite(int(row[0]), tuple(int(index) for index in row[1:])).canonical(host)
        types = {atomic_type.symbol: atomic_type for atomic_type in site.occupation}
        by_site[key] = types
        for symbol, atomic_type in types.items():
            if atomic_type.has_converged_data():
                by_element.setdefault(symbol, atomic_type)

    reference = host_atoms(host)
    if isinstance(reference, SPRKKRAtoms) and reference.are_sites_inited():
        for site in reference.sites:
            for atomic_type in site.occupation:
                if atomic_type.has_converged_data():
                    by_element.setdefault(atomic_type.symbol, atomic_type)
    return by_site, by_element


def _initial_types(occupation, local_types, by_element):
    """Select complete data on a common mesh for every species at one site."""
    sources = {}
    for symbol in occupation:
        atomic_type = local_types.get(symbol)
        if atomic_type is None or not atomic_type.has_converged_data():
            atomic_type = by_element.get(symbol)
        if atomic_type is None or atomic_type.mesh is None:
            return None
        sources[symbol] = atomic_type
    if not sources:
        return None
    mesh = next(iter(sources.values())).mesh
    if any(
        atomic_type.mesh is not mesh and not _equivalent_meshes(mesh, atomic_type.mesh)
        for atomic_type in sources.values()
    ):
        return None
    return sources


def _restart_sources(original, entries):
    if original is None:
        return {}
    by_site, by_element = _radial_sources(original)
    return {
        key: _initial_types(entry.occupation, by_site.get(key, {}), by_element)
        for key, entry in entries.items()
    }


def _would_discard_radial_data(original, sources):
    return original is not None and None in sources.values() and any(
        atomic_type.potential is not None or atomic_type.charge is not None or atomic_type.moments is not None
        for site in original.sites
        for atomic_type in site.occupation
    )


def embedded_edit_discards_radial_data(original, entries):
    """Whether the edited cluster would lose its native restart data."""
    return _would_discard_radial_data(original, _restart_sources(original, entries))


def rebuild_embedded_cluster(host, entries, *, original=None, allow_radial_reset=False, shallow=False):
    """Build an edited cluster from a mapping of HostSite to EmbeddedSiteParameters.

    Use complete compatible radial data from the original cluster or its host
    as an unconverged restart. Deep-copy the data unless ``shallow=True`` is
    requested for a preview.
    """
    if not entries:
        raise ValueError("The cluster must contain at least one site")
    if original is not None and original.embedding_host is not host:
        raise ValueError("The original cluster must reference the selected host")
    sources = _restart_sources(original, entries)
    if not allow_radial_reset and _would_discard_radial_data(original, sources):
        raise ValueError(
            "Editing this cluster would discard existing radial data. "
            "Pass allow_radial_reset=True only after explicit confirmation."
        )

    ordered = sorted(entries.items())
    atoms = SPRKKRAtoms.from_embedding(
        host,
        [(key.original_site, key.cell_index, entry.occupation) for key, entry in ordered],
    )
    meshes, types = {}, {}
    for (key, entry), site in zip(ordered, atoms.sites):
        site_sources = sources.get(key)
        if site_sources is not None:
            source_mesh = next(iter(site_sources.values())).mesh
            mesh_key = id(source_mesh)
            if mesh_key not in meshes:
                meshes[mesh_key] = source_mesh.copy()
            target_mesh = meshes[mesh_key]
            site.site_type.mesh = target_mesh
            occupation = {}
            for symbol, fraction in entry.occupation.items():
                source_type = site_sources[symbol]
                type_key = (id(source_type), mesh_key)
                if type_key not in types:
                    types[type_key] = source_type.copy(for_mesh=target_mesh, shallow=shallow)
                occupation[types[type_key]] = fraction
            site.occupation = occupation
        site.embedding.displacement = entry.displacement.copy()
        site.magnetisation = entry.magnetisation.copy()

    complete = original is not None and all(value is not None for value in sources.values())
    atoms.info["sprkkr_embedding_scf_status"] = "ITERATING" if complete else "START"
    return atoms
