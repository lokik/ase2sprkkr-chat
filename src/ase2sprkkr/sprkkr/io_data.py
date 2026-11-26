"""IoData classes serves are intermediate object used as storage during
reading or writing of potential/input_parameters files."""

from ..common.decorators import cached_property
from ..common.unique_values import UniqueValuesMapping
from functools import wraps
import numpy as np


def unique_mapping(fce):
    """
    Create function (with a cached result), that returns UniqueValuesMapping from a given iterator
    """

    @wraps(fce)
    def get_unique_mapping(self):
        lst = [i for i in fce(self)]
        return UniqueValuesMapping.from_values(lst)

    return cached_property(get_unique_mapping)


class BaseIoData(dict):
    """Base class for object used during reading/writing.
    For a potential future use."""

    pass


class WriteIoData(BaseIoData):
    """During writing of potential file, some lists are needed in more sections,
    typically one section contains the list itself, while another(s) reference(s)
    the items of the list.
    So the lists are created on demand and cached for its further use.
    """

    def __init__(self, atoms):
        self.atoms = atoms
        self._has_converged_data = None

    @cached_property
    def sites_order(self):
        """Order of ASE site indices in a native periodic POTFIL."""
        atoms = self.atoms
        if atoms.pbc.sum() != 2:
            return slice(None)
        order = np.concatenate([
            atoms.regions[name].ids[np.argsort(atoms.regions[name].positions[:, 2])]
            for name in ("left", "central", "right")
        ])
        if np.array_equal(order, np.arange(len(atoms))):
            return slice(None)
        return order

    def site_number(self, site):
        """One-based native site number, not a SiteType number."""
        if site.atoms is not self.atoms:
            raise ValueError("Site reference does not belong to the supplied atoms")
        index = site.index
        order = self.sites_order
        if isinstance(order, slice):
            return index + 1
        return int(np.flatnonzero(order == index)[0]) + 1

    @unique_mapping
    def site_types(self):
        return [i.site_type for i in self.atoms.sites]

    @unique_mapping
    def types(self):
        for s in self.site_types.iter_unique():
            yield from s.occupation.atomic_types()

    @unique_mapping
    def reference_systems(self):
        return (s.reference_system for s in self.site_types.iter_unique())

    @unique_mapping
    def meshes(self):
        return (s.mesh for s in self.site_types.iter_unique())

    def has_converged_data(self, potential):

        def compute():
            fullpot = potential.SCF_INFO.FULLPOT()
            for site_type, i in self.site_types.unique_items():
                for atom in site_type.occupation.atomic_types():
                    if not atom.charge or not atom.potential or not atom.moments:
                        return False
                    if fullpot and (
                        not atom.charge.fullpot_data or not atom.potential.fullpot_data
                    ):
                        # An incomplete FULLPOT state must not be represented
                        # as a restartable converged radial potential.
                        return False

            return True

        if self._has_converged_data is None:
            self._has_converged_data = compute()
        return self._has_converged_data


class ReadIoData(BaseIoData):
    """Object to store data during reading of a potential file."""

    def __init__(self):
        self._postponed = []

    def apply_on_atoms(self, handler, atoms):
        """Apply the given function on the atoms object, if not-None is given.
        However, the store the handler, to be executed (or repeated) on a newly
        given atoms - for the case, that a new atoms object will be created later
        """
        if atoms is not None:
            handler(atoms)
        self._postponed.append(handler)

    def update_atoms(self, atoms):
        """Replay are the stored handlers on the given atoms object"""
        if atoms.are_sites_inited():
            for i in atoms.sites:
                i._clear_data()
        for handler in self._postponed:
            handler(atoms)
