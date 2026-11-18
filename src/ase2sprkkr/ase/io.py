"""ASE read/write adapter for ordinary POTFIL and embedded-cluster CLUPOT.

ASE expects Atoms, not a potential object.  CLUPOT contains neither the
host unit cell nor its full geometry, so reading a cluster requires a
separate host potential supplied through the ``host`` argument.
"""

from io import StringIO
from os import PathLike

from ase import Atoms

from ..potentials.cluster_potentials import ClusterPotential
from ..potentials.potentials import Potential


def _source(fd):
    """Make file-like input reusable for format probing and parsing.

    Non-seekable streams cannot be probed and read separately.  Buffer
    file-like input once; filesystem paths continue to use streaming I/O.
    """
    if not hasattr(fd, "read"):
        return fd
    content = fd.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return StringIO(content)


def _cluster_host(host):
    if host is None:
        raise ValueError(
            "Reading CLUPOT as ASE Atoms requires a host potential; "
            "pass host='host.pot' or host=Potential.from_file('host.pot'). "
            "To inspect CLUPOT without its host, use ClusterPotential.from_file()."
        )
    if isinstance(host, (str, bytes, PathLike)) or hasattr(host, "read"):
        host = Potential.from_file(host)
    if not all(hasattr(host, attr) for attr in ("atoms", "LATTICE", "OCCUPATION")):
        raise TypeError("CLUPOT host must be an ordinary SPR-KKR Potential or its file path")
    if isinstance(host, ClusterPotential):
        raise TypeError("CLUPOT host must be an ordinary (non-cluster) potential")
    return host


def read_sprkkr(fd, index=None, *, host=None, allow_dangerous=False, **kwargs):
    """Read a POTFIL or CLUPOT into ASE atoms.

    The format is identified by the native NQCLU header, not the filename
    extension.  CLUPOT requires ``host``, either a Potential or a path to
    an ordinary potential file.
    """
    source = _source(fd)
    if Potential.is_cluster_file(source):
        potential = ClusterPotential.from_file(
            source, host=_cluster_host(host), allow_dangerous=allow_dangerous
        )
        atoms = potential.atoms
    else:
        atoms = Potential.from_file(source, allow_dangerous=allow_dangerous).atoms
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
