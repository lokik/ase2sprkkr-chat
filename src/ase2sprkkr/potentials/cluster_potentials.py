"""Representation of a native SPR-KKR embedded-cluster potential.

The host lattice and electronic environment live in a separate host POTFIL.
Unlike :class:`Potential`, this file must not be converted into a periodic
ASE Atoms object merely by reading it.
"""

from io import StringIO

from ..sprkkr.configuration import ConfigurationFile


class ClusterPotential(ConfigurationFile):
    """A POTFIL_CLU file and its host-site-to-cluster-site mapping.

    Cluster site indices (IQCLU) and atomic type indices (ITCLU) are local
    to this file.  The radial TYPE numbers and magnetisation site indices
    refer to the host-plus-cluster global numbering in the native code.

    This object parses and writes the cluster file; computing its Cartesian
    positions requires the separate host potential and is intentionally not
    inferred here.
    """

    def __init__(self, definition=None):
        if definition is None:
            from .definitions.cluster import cluster_definition

            definition = cluster_definition
        super().__init__(definition)

    @classmethod
    def from_file(cls, filename, *, allow_dangerous=False):
        """Read a native cluster potential without constructing ASE atoms."""
        from .definitions.cluster import cluster_definition

        return cluster_definition.read_from_file(filename, allow_dangerous=allow_dangerous)

    @classmethod
    def from_string(cls, content, *, allow_dangerous=False):
        return cls.from_file(StringIO(content), allow_dangerous=allow_dangerous)

    def save_to_file(self, file, *, validate="save"):
        """Write the cluster file, without ASE-atom synchronization."""
        return super().save_to_file(file, atoms=False, validate=validate)
