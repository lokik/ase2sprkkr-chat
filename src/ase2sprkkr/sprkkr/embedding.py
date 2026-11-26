"""Live access to embedded-cluster geometry on an ordinary SPRKKR Site.

The only per-atom embedding data are a host basis-site index and five
integer translations. Actual positions live in ASE Atoms.positions, and
displacements are always derived from these positions.
"""

import numpy as np
from ase import Atoms

EMBEDDING_ARRAY = "sprkkr_embedding"
EMBEDDING_HOST = "sprkkr_embedding_host"


def host_atoms(host):
    """Accept an ASE Atoms or an SPR-KKR Potential as the embedding host."""
    if host is None:
        raise ValueError("sprkkr_embedding_host must be assigned before using embedding")
    if isinstance(host, Atoms):
        return host
    from ..potentials.potentials import PeriodicPotential
    if isinstance(host, PeriodicPotential):
        atoms = host.atoms
        if atoms is not None:
            return atoms
    raise TypeError("Embedding host must be Atoms or a periodic Potential containing Atoms")


def host_dimension(host):
    """The translation dimension in the user-facing API (2D or 3D)."""
    atoms = host_atoms(host)
    return 2 if tuple(bool(x) for x in atoms.pbc) == (True, True, False) else 3


def translation_basis(host):
    """Return the five SPR-KKR real-space translation vectors, in Angstrom."""
    atoms = host_atoms(host)
    basis = np.zeros((5, 3))
    basis[:3] = np.asarray(atoms.cell)
    if host_dimension(host) == 2:
        from .sprkkr_atoms import SPRKKRAtoms
        from ..potentials.potentials import PeriodicPotential
        regions = atoms.regions if isinstance(atoms, SPRKKRAtoms) else {}
        if "left" in regions and "right" in regions:
            basis[3] = np.asarray(regions["left"].cell[2])
            basis[4] = np.asarray(regions["right"].cell[2])
        elif isinstance(host, PeriodicPotential):
            from ase.units import Bohr
            lattice = host.LATTICE
            alat = lattice.ALAT() * Bohr
            basis[3] = np.asarray(lattice.A_L3()) * alat
            basis[4] = np.asarray(lattice.A_R3()) * alat
    return basis


def embedding_data(atoms):
    """Validate and return the per-atom integer (host_site, N5VEC) matrix."""
    value = atoms.arrays.get(EMBEDDING_ARRAY)
    if value is None:
        raise ValueError("This structure has no sprkkr_embedding data")
    if value.shape != (len(atoms), 6) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("sprkkr_embedding must be a (N, 6) integer array")
    return value


def repeat_shift(host, shift, tolerance=1e-8):
    """Represent a Cartesian repetition displacement as host-cell translations."""
    dim = host_dimension(host)
    basis = translation_basis(host)[:dim]
    coefficients, _, rank, _ = np.linalg.lstsq(basis.T, shift, rcond=None)
    rounded = np.rint(coefficients).astype(int)
    if rank != dim or not np.allclose(rounded @ basis, shift, atol=tolerance, rtol=0):
        raise ValueError("The repeated cell is not commensurate with the embedding host")
    result = np.zeros(5, dtype=int)
    result[:dim] = rounded
    return result


class SiteEmbedding:
    """Proxy holding only its Site; no duplicate position/displacement data."""

    def __init__(self, site):
        self.site = site

    def _context(self):
        atoms = self.site.atoms
        host = atoms.info.get(EMBEDDING_HOST)
        reference = host_atoms(host)
        row = embedding_data(atoms)[self.site.index]
        index = int(row[0])
        if not 0 <= index < len(reference):
            raise ValueError(f"Embedding host-site index {index} is out of range")
        return atoms, host, reference, row

    @staticmethod
    def _reference(host, atoms, row):
        return (np.asarray(atoms.positions[int(row[0])])
                + np.dot(row[1:6], translation_basis(host)))

    @property
    def original_site(self):
        return int(self._context()[3][0])

    @original_site.setter
    def original_site(self, index):
        atoms, host, reference, row = self._context()
        if not isinstance(index, (int, np.integer)):
            raise TypeError("original_site must be an integer host-site index")
        if not 0 <= index < len(reference):
            raise IndexError("original_site is outside the host structure")
        delta = self.displacement
        changed = row.copy()
        changed[0] = index
        position = self._reference(host, reference, changed) + delta
        row[:] = changed
        atoms.positions[self.site.index] = position

    @property
    def translation_vector(self):
        """All five native N5VEC coefficients (even for 2D/3D convenience APIs)."""
        return tuple(int(x) for x in self._context()[3][1:6])

    @property
    def cell_index(self):
        _, host, _, row = self._context()
        dim = host_dimension(host)
        # Preserve extraordinary native CLUPOT indices in a lossless view.
        size = 5 if np.any(row[1 + dim:6]) else dim
        return tuple(int(x) for x in row[1:1 + size])

    @cell_index.setter
    def cell_index(self, index):
        atoms, host, reference, row = self._context()
        try:
            index = tuple(index)
        except TypeError as exc:
            raise TypeError("cell_index must be a tuple or list of integers") from exc
        dim = host_dimension(host)
        if len(index) not in (dim, 5):
            raise ValueError(f"Expected {dim} or 5 cell indices for this host")
        if any(not isinstance(i, (int, np.integer)) for i in index):
            raise TypeError("cell_index must contain only integers")
        delta = self.displacement
        changed = row.copy()
        changed[1:6] = 0
        changed[1:1 + len(index)] = index
        position = self._reference(host, reference, changed) + delta
        row[:] = changed
        atoms.positions[self.site.index] = position

    @property
    def reference_position(self):
        _, host, reference, row = self._context()
        return self._reference(host, reference, row)

    @property
    def displacement(self):
        atoms, host, reference, row = self._context()
        return np.asarray(atoms.positions[self.site.index]) - self._reference(host, reference, row)

    @displacement.setter
    def displacement(self, value):
        atoms, host, reference, row = self._context()
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("displacement must be a finite Cartesian three-vector")
        atoms.positions[self.site.index] = self._reference(host, reference, row) + value
