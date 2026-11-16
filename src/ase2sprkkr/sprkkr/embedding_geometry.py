"""Geometry and point symmetry of explicit sites in a periodic embedding host.

A HostSite identifies a *physical atom*, not just a repeated unit cell.  The
symmetry finder retains only operations fixing the specified cluster centre in
real space.  No supercell or cluster Atoms is created by these operations.
"""

from dataclasses import dataclass
from itertools import product
from numbers import Integral
import numpy as np

from ..bindings.spglib import symmetry_dataset
from .embedding import host_atoms, host_dimension, translation_basis
from .site_equivalence import site_equivalence_tags


@dataclass(frozen=True, order=True)
class HostSite:
    """Zero-based host basis site and its 2D/3D (or native N5VEC) translation."""

    original_site: int
    cell_index: tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.original_site, bool) or not isinstance(self.original_site, Integral):
            raise TypeError("original_site must be an integer")
        try:
            index = tuple(self.cell_index)
        except TypeError as exc:
            raise TypeError("cell_index must be a sequence of integers") from exc
        if len(index) not in (2, 3, 5):
            raise ValueError("cell_index must have 2, 3 or 5 components")
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in index):
            raise TypeError("cell_index must contain only integers")
        object.__setattr__(self, "original_site", int(self.original_site))
        object.__setattr__(self, "cell_index", tuple(int(i) for i in index))

    def translations(self, host):
        """Five native translation indices, with user-facing dimensions checked."""
        dim = host_dimension(host)
        if len(self.cell_index) not in (dim, 5):
            raise ValueError(f"Expected {dim} or 5 cell indices for this host")
        return np.array(self.cell_index + (0,) * (5 - len(self.cell_index)), dtype=int)

    def canonical(self, host):
        """Use short indices unless nonstandard native translations are present."""
        values = self.translations(host)
        dim = host_dimension(host)
        return HostSite(self.original_site, tuple(values if np.any(values[dim:]) else values[:dim]))

    def position(self, host):
        """Cartesian position in Angstrom; host is never copied or modified."""
        atoms = host_atoms(host)
        if not 0 <= self.original_site < len(atoms):
            raise IndexError("original_site is outside the host structure")
        return (np.asarray(atoms.positions[self.original_site], dtype=float)
                + self.translations(host) @ translation_basis(host))

    def occupation(self, host):
        """Copy of the host site's chemical fractions, including vacancies."""
        from .occupations import Occupation
        from .sprkkr_atoms import SPRKKRAtoms

        atoms = host_atoms(host)
        if not 0 <= self.original_site < len(atoms):
            raise IndexError("original_site is outside the host structure")
        if isinstance(atoms, SPRKKRAtoms) and atoms.are_sites_inited():
            return atoms.sites[self.original_site].occupation.to_dict(True)
        return Occupation.for_atom_from_ase_atoms(atoms, self.original_site).to_dict(True)

    def region(self, host):
        """2D host region name if available (left, central or right)."""
        atoms = host_atoms(host)
        if not 0 <= self.original_site < len(atoms):
            raise IndexError("original_site is outside the host structure")
        if host_dimension(host) != 2:
            return None
        regions = getattr(atoms, "regions", {})
        for name, region in regions.items():
            if self.original_site in region.ids:
                return name
        lattice = getattr(host, "LATTICE", None)
        if lattice is not None and hasattr(lattice, "NQ_L") and hasattr(lattice, "NQ_R"):
            left, right = int(lattice.NQ_L()), int(lattice.NQ_R())
            if self.original_site < left:
                return "left"
            if self.original_site >= len(atoms) - right:
                return "right"
            return "central"
        return None


def embedding_host_indices(host):
    """Allowed host basis indices (exclude semi-infinite leads for a 2D POTFIL)."""
    atoms = host_atoms(host)
    if host_dimension(host) != 2:
        return tuple(range(len(atoms)))
    lattice = getattr(host, "LATTICE", None)
    if lattice is not None and hasattr(lattice, "NQ_L") and hasattr(lattice, "NQ_R"):
        left, right = int(lattice.NQ_L()), int(lattice.NQ_R())
        if left < 0 or right < 0 or left + right > len(atoms):
            raise ValueError("Invalid 2D host lead-site counts")
        return tuple(range(left, len(atoms) - right))
    regions = getattr(atoms, "regions", {})
    if "central" in regions:
        return tuple(sorted(int(i) for i in regions["central"].ids))
    return tuple(range(len(atoms)))


@dataclass(frozen=True, eq=False)
class EmbeddingOperation:
    """One centre-fixing host symmetry; rotation acts on Cartesian *row* vectors."""

    rotation: np.ndarray
    site_permutation: tuple[int, ...]


class EmbeddingSymmetry:
    """Point subgroup of the chemical symmetry of a periodic host.

    The reference centre may be a HostSite or Cartesian (x, y, z) in Angstrom.
    'occupation' ignores coincidental SiteType identity, while 'site-types'
    respects already initialized SiteTypes and their magnetic directions.
    For 2D hosts only operations preserving the oriented surface normal are
    accepted: they cannot exchange the two semi-infinite sides.
    """

    def __init__(self, host, center, *, equivalence="occupation", symprec=1e-5,
                 angle_tolerance=-1.0):
        if symprec <= 0 or not np.isfinite(symprec):
            raise ValueError("symprec must be positive and finite")
        self.host = host
        self.atoms = host_atoms(host)
        self.dimension = host_dimension(host)
        self.symprec = float(symprec)
        if isinstance(center, HostSite):
            self.center = center.canonical(host)
            position = center.position(host)
        else:
            self.center = None
            position = np.asarray(center, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("center must be a HostSite or finite Cartesian three-vector")
        self.center_position = position.copy()
        self._basis = np.asarray(self.atoms.cell, dtype=float).copy()
        if self.dimension == 2:
            normal = np.cross(self._basis[0], self._basis[1])
            norm = np.linalg.norm(normal)
            if norm <= 1e-12:
                raise ValueError("The two periodic lattice vectors are dependent")
            self._normal = normal / norm
            # spglib requires a full-rank cell. For a flat ASE host use an
            # artificial nonperiodic third vector solely for the symmetry search.
            if abs(np.linalg.det(self._basis)) < 1e-10:
                heights = np.asarray(self.atoms.positions) @ self._normal
                height = float(np.ptp(heights)) if len(heights) else 0.0
                self._basis[2] = self._normal * max(20.0, height + 20.0)
        else:
            self._normal = None
        if abs(np.linalg.det(self._basis)) < 1e-12:
            raise ValueError("A full-rank host cell is needed for symmetry")
        inv = np.linalg.inv(self._basis)
        scaled = np.asarray(self.atoms.positions) @ inv
        centre_scaled = position @ inv

        memberships = None
        regions = getattr(self.atoms, "regions", {})
        if regions:
            memberships = [[] for _ in range(len(self.atoms))]
            for name, region in regions.items():
                for i in region.ids:
                    memberships[int(i)].append(name)
        tags = site_equivalence_tags(
            self.atoms, equivalence=equivalence, region_memberships=memberships
        )
        dataset = symmetry_dataset(self._basis, scaled, tags, symprec=symprec,
                                   angle_tolerance=angle_tolerance)
        self.operations = self._find_operations(
            dataset, inv, scaled, centre_scaled, tags
        )
        if not self.operations:
            raise ValueError("No host symmetry operation fixes the specified center")

    def _site_permutation(self, rotation):
        """Find basis permutations without identifying different 2D layers."""
        positions = np.asarray(self.atoms.positions, dtype=float)
        basis = self._basis[:self.dimension]
        pinv = np.linalg.pinv(basis)
        result = []
        for i, target in enumerate(positions @ rotation
                                   + self.center_position - self.center_position @ rotation):
            # The chemical/region constraint is enforced by _find_operations.
            choices = self._candidates[int(i)]
            matches = []
            for j in choices:
                delta = target - positions[j]
                shifts = np.rint(delta @ pinv)
                residual = delta - shifts @ basis
                if np.linalg.norm(residual) <= self.symprec * 2:
                    matches.append(j)
            if len(matches) != 1:
                return None
            result.append(matches[0])
        if len(set(result)) != len(result):
            return None
        return tuple(result)

    def _preserves_2d_native_translations(self, rotation):
        """Keep lead vectors fixed modulo integer in-plane translations.

        A layer-group operation of the central slab is not necessarily a
        symmetry of the semi-infinite leads (which can have tilted A_L3/A_R3).
        The nonperiodic N5VEC coefficients are intentionally not permuted.
        """
        periodic = np.asarray(self.atoms.cell, dtype=float)[:2]
        pinv = np.linalg.pinv(periodic)
        for vector in translation_basis(self.host)[2:]:
            delta = np.asarray(vector) @ rotation - vector
            coeffs = np.rint(delta @ pinv)
            if np.linalg.norm(delta - coeffs @ periodic) > 2 * self.symprec:
                return False
        return True

    def _find_operations(self, dataset, inv, scaled, centre_scaled, tags):
        self._candidates = {
            i: tuple(int(j) for j in np.flatnonzero(tags == tag))
            for i, tag in enumerate(tags)
        }
        ops = []
        for fractional_rotation, fractional_translation in zip(
            dataset.rotations, dataset.translations
        ):
            # spglib translations are modulo the periodic cell. Only their
            # unique lattice lift fixing the chosen physical centre is allowed.
            needed = centre_scaled - (
                centre_scaled @ fractional_rotation.T + fractional_translation
            )
            lattice_shift = np.rint(needed)
            if self.dimension == 2:
                lattice_shift[2] = 0
            if np.linalg.norm((needed - lattice_shift) @ self._basis) > self.symprec:
                continue
            rotation = inv @ fractional_rotation.T @ self._basis
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
                continue
            if self.dimension == 2:
                if not np.allclose(self._normal @ rotation, self._normal, atol=1e-6):
                    continue
                if not self._preserves_2d_native_translations(rotation):
                    continue
            permutation = self._site_permutation(rotation)
            if permutation is None:
                continue
            if any(np.allclose(rotation, old.rotation, atol=1e-10) for old in ops):
                continue
            ops.append(EmbeddingOperation(rotation.copy(), permutation))
        return tuple(ops)

    def map_site(self, site, operation):
        """Map a physical site to (new original_site, new cell_index)."""
        if not any(operation is known for known in self.operations):
            raise ValueError("operation does not belong to this host symmetry")
        site = site.canonical(self.host)
        source = site.translations(self.host)
        destination = operation.site_permutation[site.original_site]
        rotated = (site.position(self.host) - self.center_position) @ operation.rotation
        rotated += self.center_position
        # Preserve native out-of-plane N5VEC coefficients. A transformed
        # translation not expressible this way is deliberately rejected.
        extras = source.copy()
        extras[:self.dimension] = 0
        residual = (rotated - self.atoms.positions[destination]
                    - extras @ translation_basis(self.host))
        basis = np.asarray(self.atoms.cell)[:self.dimension]
        shifts, _, rank, _ = np.linalg.lstsq(basis.T, residual, rcond=None)
        rounded = np.rint(shifts).astype(int)
        if rank != self.dimension or np.linalg.norm(residual - rounded @ basis) > 2 * self.symprec:
            raise ValueError("Mapped site cannot be expressed using host translations")
        extras[:self.dimension] = rounded
        return HostSite(destination, tuple(extras if np.any(extras[self.dimension:])
                                          else extras[:self.dimension]))

    def equivalent_sites(self, site):
        """Sorted orbit of a HostSite (without changing any atoms)."""
        return tuple(sorted({self.map_site(site, op) for op in self.operations}))

    def expand_symmetry(self, sites):
        """Sorted union of the symmetry orbits of all selected sites."""
        return tuple(sorted({mapped for site in sites for mapped in self.equivalent_sites(site)}))

    def transform_displacement(self, displacement, operation):
        """Rotate a Cartesian displacement, e.g. when copying a relaxation."""
        if not any(operation is known for known in self.operations):
            raise ValueError("operation does not belong to this host symmetry")
        vector = np.asarray(displacement, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("displacement must be a finite Cartesian three-vector")
        return vector @ operation.rotation


def _center_positions(host, centers):
    if isinstance(centers, HostSite):
        values = (centers,)
    else:
        try:
            array = np.asarray(centers, dtype=float)
        except (TypeError, ValueError):
            array = None
        if array is not None and array.shape == (3,):
            values = (array,)
        elif array is not None and array.ndim == 2 and array.shape[1] == 3:
            values = tuple(array)
        else:
            values = tuple(centers)
    if not values:
        raise ValueError("At least one center is required")
    result = []
    for item in values:
        value = item.position(host) if isinstance(item, HostSite) else np.asarray(item, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("Center must be a HostSite or Cartesian three-vector")
        result.append(value)
    return tuple(result)


def _sites_in_ball(host, center, radius, tolerance, max_candidates):
    atoms = host_atoms(host)
    dim = host_dimension(host)
    basis = np.asarray(atoms.cell, dtype=float)[:dim]
    if np.linalg.matrix_rank(basis) != dim:
        raise ValueError("Periodic host lattice vectors are linearly dependent")
    pinv = np.linalg.pinv(basis)
    widths = (radius + tolerance) * np.linalg.norm(pinv, axis=0)
    selected = set()
    tested = 0
    for i in embedding_host_indices(host):
        origin = np.asarray(atoms.positions[i], dtype=float)
        nearest = (center - origin) @ pinv
        ranges = [
            range(int(np.floor(x - w)) - 1, int(np.ceil(x + w)) + 2)
            for x, w in zip(nearest, widths)
        ]
        candidates = int(np.prod([len(r) for r in ranges]))
        tested += candidates
        if tested > max_candidates:
            raise ValueError("Requested cluster region is too large to enumerate")
        for cell_index in product(*ranges):
            offset = np.asarray(cell_index) @ basis
            if np.linalg.norm(origin + offset - center) <= radius + tolerance:
                selected.add(HostSite(int(i), tuple(cell_index)))
    return selected


def embedding_sites_within_radius(host, centers, radius, *, tolerance=1e-8,
                                  max_candidates=200000):
    """Enumerate host sites within radius (Angstrom) of any specified center.

    The host may be ASE Atoms or Potential. For 2D host POTFILs, sites from
    semi-infinite leads are excluded, and repetition is only in the xy plane.
    """
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be non-negative and finite")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be non-negative and finite")
    positions = _center_positions(host, centers)
    selected = set()
    for center in positions:
        selected.update(_sites_in_ball(host, center, radius, tolerance, max_candidates))
    return tuple(sorted(selected))


def embedding_sites_within_shells(host, centers, shells, *, tolerance=1e-5,
                                  max_candidates=200000):
    """Return sites through the Nth *positive-distance* coordination shell.

    A site exactly at the centre is always included (shell zero). For several
    centres, shells are counted separately, then the site sets are united.
    """
    if isinstance(shells, bool) or not isinstance(shells, Integral) or shells < 0:
        raise ValueError("shells must be a non-negative integer")
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    basis = np.asarray(host_atoms(host).cell)[:host_dimension(host)]
    start_radius = max(float(min(np.linalg.norm(v) for v in basis)), tolerance)
    selected = set()
    for center in _center_positions(host, centers):
        radius = start_radius
        for _ in range(14):
            found = _sites_in_ball(host, center, radius, tolerance, max_candidates)
            values = sorted(
                np.linalg.norm(HostSite(s.original_site, s.cell_index).position(host) - center)
                for s in found
            )
            distances = []
            for value in values:
                if value > tolerance and (not distances or value - distances[-1] > tolerance):
                    distances.append(value)
            if shells == 0:
                selected.update(s for s in found if np.linalg.norm(s.position(host) - center) <= tolerance)
                break
            if len(distances) >= shells and distances[shells - 1] + tolerance < radius:
                cutoff = distances[shells - 1] + tolerance
                selected.update(s for s in found if np.linalg.norm(s.position(host) - center) <= cutoff)
                break
            radius *= 2
        else:
            raise ValueError("Could not resolve the requested number of coordination shells")
    return tuple(sorted(selected))


def embedding_add_sites(atoms, sites, *, occupation=None):
    """Return a cluster copy with new host sites; reject duplicate identities."""
    from .sprkkr_atoms import SPRKKRAtoms
    if not isinstance(atoms, SPRKKRAtoms) or atoms.embedding_host is None:
        raise ValueError("Expected an embedded SPRKKRAtoms with a host")
    existing = {
        HostSite(int(row[0]), tuple(int(i) for i in row[1:])).canonical(atoms.embedding_host)
        for row in atoms.arrays["sprkkr_embedding"]
    }
    requested = tuple(site.canonical(atoms.embedding_host) for site in sites)
    if len(requested) != len(set(requested)) or existing.intersection(requested):
        raise ValueError("An embedded host site may occur only once")
    if not requested:
        return atoms.copy()
    allowed = set(embedding_host_indices(atoms.embedding_host))
    if any(site.original_site not in allowed for site in requested):
        raise ValueError("Cannot embed a site in a semi-infinite 2D lead")
    added = SPRKKRAtoms.from_embedding(atoms.embedding_host, [
        (site.original_site, site.cell_index, occupation) for site in requested
    ])
    result = atoms.copy()
    result.extend(added)
    return result


def embedding_remove_sites(atoms, sites):
    """Return a cluster copy with requested physical sites removed."""
    from .sprkkr_atoms import SPRKKRAtoms
    if not isinstance(atoms, SPRKKRAtoms) or atoms.embedding_host is None:
        raise ValueError("Expected an embedded SPRKKRAtoms with a host")
    removal = {site.canonical(atoms.embedding_host) for site in sites}
    existing = [
        HostSite(int(row[0]), tuple(int(i) for i in row[1:])).canonical(atoms.embedding_host)
        for row in atoms.arrays["sprkkr_embedding"]
    ]
    if not removal.issubset(set(existing)):
        raise ValueError("Cannot remove a site that is absent from the cluster")
    return atoms[[i for i, site in enumerate(existing) if site not in removal]]
