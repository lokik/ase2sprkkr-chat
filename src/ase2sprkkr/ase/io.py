"""ASE read/write adapter for ordinary POTFIL and embedded-cluster CLUPOT.

ASE expects Atoms, not a potential object.  CLUPOT contains neither the
host unit cell nor its full geometry, so reading a cluster requires a
separate host potential supplied through the ``host`` argument.
"""

from ase import Atoms

from ..potentials.cluster_potentials import ClusterPotential
from ..potentials.potentials import Potential


def read_sprkkr(fd, index=None, *, host=None, allow_dangerous=False, **kwargs):
    """Read a POTFIL or CLUPOT into ASE atoms.

    The format is identified by the native NQCLU header, not the filename
    extension.  CLUPOT requires ``host``, either a Potential or a path to
    an ordinary potential file.
    """
    potential = Potential.from_file(fd, host=host, allow_dangerous=allow_dangerous)
    if isinstance(potential, ClusterPotential) and potential.host is None:
        raise ValueError(
            "Reading CLUPOT as ASE Atoms requires a host potential; "
            "pass host='host.pot' or host=Potential.from_file('host.pot')."
        )
    atoms = potential.atoms
    if index is not None:
        return [atoms][index]
    return atoms


read_SPRKKR = read_sprkkr


def _single_atoms(images):
    """Accept both ASE's image sequence and a direct single Atoms object."""
    if isinstance(images, Atoms):
        return images
    try:
        frames = list(images)
    except TypeError as exc:
        raise TypeError("SPR-KKR writer expects ASE Atoms or a sequence of Atoms") from exc
    if len(frames) != 1:
        raise ValueError("A SPR-KKR potential file represents exactly one Atoms structure")
    if not isinstance(frames[0], Atoms):
        raise TypeError("SPR-KKR writer expects ASE Atoms")
    return frames[0]


def write_sprkkr(fd, atoms, *, validate="save", **kwargs):
    """Write POTFIL or CLUPOT according to the structure's periodicity.

    Rebuild CLUPOT from the current embedded atoms so edits to positions,
    occupation, mesh data, and magnetic directions are not silently lost.
    The host is retained by reference in atoms.info['sprkkr_embedding_host'].
    """
    atoms = _single_atoms(atoms)
    if not any(atoms.pbc) and atoms.info.get("sprkkr_embedding_host") is not None:
        potential = ClusterPotential.from_atoms(atoms)
    else:
        potential = Potential.from_atoms(atoms)
    return potential.save_to_file(fd, validate=validate)


write_SPRKKR = write_sprkkr
