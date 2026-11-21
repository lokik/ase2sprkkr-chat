import numpy as np

from ..sites import Site
from ..sprkkr_atoms import SPRKKRAtoms
from ..atoms_region import AtomsRegion


class TestSPRKKRAtomsCopyOwnership:
    def test_shallow_copy_detaches_editable_containers_and_shares_radial_values(self):
        atoms = SPRKKRAtoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]],
                            cell=[2, 2, 2], symmetry=False)
        atoms.sites[1].site_type = atoms.sites[0].site_type
        original = atoms.sites[0].atomic_type(0)
        original_coordinates = original.mesh.coors
        original.mesh.set_fullpot((1, 2, 1, [3]))
        original.potential = np.ones((2, len(original.mesh)))
        original.charge = np.ones(len(original.mesh))
        copied = atoms.copy(shallow=True)
        edited = copied.sites[0].atomic_type(0)

        assert copied.sites[0].site_type is copied.sites[1].site_type
        assert copied.sites[0].site_type is not atoms.sites[0].site_type
        assert edited is not original
        assert edited.mesh is not original.mesh
        assert not np.shares_memory(edited.mesh.coors, original_coordinates)
        edited.mesh.fullpot.jrcut[0] = 2
        assert original.mesh.fullpot.jrcut == [3]
        assert edited.potential is not original.potential
        assert edited.potential.mesh is edited.mesh
        assert np.shares_memory(edited.potential.raw_value, original.potential.raw_value)
        assert np.shares_memory(edited.charge.raw_value, original.charge.raw_value)

        copied.positions[0, 0] = 10
        copied.cell[0, 0] = 20
        copied.sites[0].site_type.occupation.set({"Cu": 1.0})
        assert atoms.positions[0, 0] == 0
        assert atoms.cell[0, 0] == 2
        assert original.symbol == "Fe"
        assert atoms.sites[0].atomic_type(0) is original

        # Replacing a radial value is local; an in-place array write is shared.
        edited.charge = np.zeros(len(edited.mesh))
        np.testing.assert_array_equal(original.charge.raw_value, 1)
        edited.potential.raw_value[0, 0] = 30
        assert original.potential.raw_value[0, 0] == 30

    def test_copy_preserves_regions_and_independent_region_caches(self):
        atoms = SPRKKRAtoms("Fe2", positions=[[0, 0, 0], [0, 0, 1]],
                            cell=[2, 2, 2], symmetry=False)
        region = AtomsRegion("left", np.array([0]), atoms.cell.copy(), atoms=atoms)
        original_info = region.spacegroup_info
        copied = atoms.copy()
        copied_region = copied.regions["left"]

        assert copied_region is not region
        assert copied_region.atoms is copied
        assert copied_region.spacegroup_info is not original_info
        assert copied_region.spacegroup_info.info is copied.spacegroup_info
        np.testing.assert_array_equal(copied_region.cell, region.cell)
        copied_region.slice[0] = 1
        copied_region.incomplete_cell[0, 0] = 10
        np.testing.assert_array_equal(region.slice, [0])
        assert region.incomplete_cell[0, 0] == 2

    def test_copy_preserves_symmetry_before_site_initialization(self):
        atoms = SPRKKRAtoms("Fe", symmetry=False)
        copied = atoms.copy()
        assert not copied.are_sites_inited()
        assert copied.symmetry is False

    def test_copy_sites_and_copy_preserve_sharing_without_reusing_source_types(self):
        atoms = SPRKKRAtoms("Fe2", positions=[[0, 0, 0], [1, 0, 0]], symmetry=False)
        atoms.sites[1].site_type = atoms.sites[0].site_type
        original_type = atoms.sites[0].site_type
        target = SPRKKRAtoms("Fe2", positions=atoms.positions, symmetry=False)

        copied_sites = Site.copy_sites(atoms.sites, target)
        assert copied_sites[0].site_type is copied_sites[1].site_type
        assert copied_sites[0].site_type.atoms is target
        assert copied_sites[0].site_type is not original_type

        copied = atoms.copy()
        assert copied.sites[0].site_type is copied.sites[1].site_type
        assert copied.sites[0].site_type is not original_type
        assert copied.sites[0].site_type.atoms is copied


    def test_copy_owns_independent_site_state(self):
        atoms = SPRKKRAtoms("Fe", symmetry=False)
        site = atoms.sites[0]
        site_type = site.site_type
        atomic_type = site_type.atomic_type(0)

        atomic_type.potential = np.ones((2, len(site_type.mesh)))
        atomic_type.charge = np.ones(len(site_type.mesh))
        atomic_type.moments = (1.0, 2.0, 3.0, 4.0, 5.0)
        atomic_type.magnetisation[:] = [1.0, 2.0, 3.0]
        site.magnetisation[:] = [4.0, 5.0]

        copied = atoms.copy()
        copied_site = copied.sites[0]
        copied_site_type = copied_site.site_type
        copied_atomic_type = copied_site_type.atomic_type(0)

        assert copied_site_type.atoms is copied
        assert copied_site_type is not site_type
        assert copied_site_type.mesh is not site_type.mesh
        assert copied_atomic_type is not atomic_type
        assert copied_atomic_type.mesh is copied_site_type.mesh
        assert copied_atomic_type.potential is not atomic_type.potential
        assert copied_atomic_type.charge is not atomic_type.charge
        assert copied_atomic_type.moments is not atomic_type.moments
        assert copied_atomic_type.potential.mesh is copied_site_type.mesh
        assert copied_atomic_type.charge.mesh is copied_site_type.mesh
        assert copied_atomic_type.magnetisation is not atomic_type.magnetisation
        assert copied_site.magnetisation is not site.magnetisation

        copied_atomic_type.potential.raw_value[0, 0] = 10.0
        copied_atomic_type.charge.raw_value[0] = 20.0
        copied_atomic_type.moments.smt = 30.0
        copied_atomic_type.magnetisation[0] = 40.0
        copied_site.magnetisation[0] = 50.0

        assert atomic_type.potential.raw_value[0, 0] == 1.0
        assert atomic_type.charge.raw_value[0] == 1.0
        assert atomic_type.moments.smt == 3.0
        assert atomic_type.magnetisation[0] == 1.0
        assert site.magnetisation[0] == 4.0
