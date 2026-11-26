"""Regression tests for physical host-site symmetry and cluster selection."""

import numpy as np
import pytest
from ase import Atoms

from ..embedding_geometry import (
    HostSite, EmbeddingSymmetry, embedding_add_sites, embedding_host_indices,
    embedding_remove_sites, embedding_sites_within_radius,
    embedding_sites_within_shells,
)
from ..sprkkr_atoms import SPRKKRAtoms
from ...potentials.potentials import PeriodicPotential


def cubic():
    return Atoms("Cu", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)


def test_host_site_translation_validation_and_position():
    host = cubic()
    site = HostSite(0, (1, -2, 3))
    np.testing.assert_allclose(site.position(host), [2, -4, 6])
    assert HostSite(0, (1, -2, 3, 0, 0)).canonical(host) == site
    with pytest.raises(TypeError):
        HostSite(0, (0, 0.5, 0))
    with pytest.raises(ValueError, match="Expected 3"):
        HostSite(0, (1, 0)).position(host)
    with pytest.raises(IndexError):
        HostSite(1, (0, 0, 0)).position(host)


def test_cubic_orbit_and_unique_bulk_sites():
    symmetry = EmbeddingSymmetry(cubic(), HostSite(0, (0, 0, 0)))
    expected = {
        HostSite(0, (1, 0, 0)), HostSite(0, (-1, 0, 0)),
        HostSite(0, (0, 1, 0)), HostSite(0, (0, -1, 0)),
        HostSite(0, (0, 0, 1)), HostSite(0, (0, 0, -1)),
    }
    assert set(symmetry.equivalent_sites(HostSite(0, (1, 0, 0)))) == expected
    assert set(symmetry.expand_symmetry([HostSite(0, (1, 0, 0)),
                                        HostSite(0, (0, 1, 0))])) == expected
    assert symmetry.equivalent_sites(HostSite(0, (0, 0, 0))) == (HostSite(0, (0, 0, 0)),)


def test_inversion_can_change_both_host_basis_site_and_cell():
    host = Atoms("Cu2", scaled_positions=[[0.25, 0, 0], [0.75, 0, 0]],
                 cell=[2, 3, 4], pbc=True)
    symmetry = EmbeddingSymmetry(host, (0., 0., 0.))
    inversion = next(op for op in symmetry.operations
                     if np.allclose(op.rotation, -np.eye(3)))
    assert symmetry.map_site(HostSite(0, (0, 0, 0)), inversion) == HostSite(1, (-1, 0, 0))


def test_cesium_chloride_has_nontrivial_cell_image_even_without_site_swap():
    host = Atoms("CsCl", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                 cell=[2, 2, 2], pbc=True)
    symmetry = EmbeddingSymmetry(host, HostSite(0, (0, 0, 0)))
    inversion = next(op for op in symmetry.operations
                     if np.allclose(op.rotation, -np.eye(3)))
    assert symmetry.map_site(HostSite(1, (1, 0, 0)), inversion) == HostSite(1, (-2, -1, -1))


def test_cartesian_centre_must_be_fixed_in_real_space():
    host = cubic()
    # At a generic centre the point stabilizer is only the identity.
    symmetry = EmbeddingSymmetry(host, np.array([0.173, 0.239, 0.391]))
    assert len(symmetry.operations) == 1
    site = HostSite(0, (1, 0, 0))
    assert symmetry.equivalent_sites(site) == (site,)


def test_displacement_rotates_and_is_not_made_periodic():
    symmetry = EmbeddingSymmetry(cubic(), HostSite(0, (0, 0, 0)))
    rotated = next(op for op in symmetry.operations
                   if not np.allclose(op.rotation, np.eye(3)))
    shift = np.array([0.04, 0.02, 0.01])
    result = symmetry.transform_displacement(shift, rotated)
    np.testing.assert_allclose(result, shift @ rotated.rotation)
    assert np.isclose(np.linalg.norm(result), np.linalg.norm(shift))
    with pytest.raises(ValueError, match="does not belong"):
        symmetry.transform_displacement(shift, object())


def test_2d_symmetry_only_repeats_in_plane_and_preserves_normal():
    host = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2]],
                 cell=[2, 2, 10], pbc=(True, True, False))
    symmetry = EmbeddingSymmetry(host, HostSite(0, (0, 0)))
    assert len(symmetry.operations) >= 4
    assert all(np.allclose([0, 0, 1] @ op.rotation, [0, 0, 1])
               for op in symmetry.operations)
    orbit = set(symmetry.equivalent_sites(HostSite(0, (1, 0))))
    assert orbit == {
        HostSite(0, (1, 0)), HostSite(0, (-1, 0)),
        HostSite(0, (0, 1)), HostSite(0, (0, -1)),
    }
    # Unusual native N5VEC coefficients are not silently dropped.
    native = HostSite(0, (1, 0, 0, 0, 2))
    images = symmetry.equivalent_sites(native)
    assert len(images) == 4
    assert all(s.translations(host)[4] == 2 for s in images)


def test_2d_flat_cell_with_zero_third_lattice_vector():
    host = Atoms("Cu", positions=[[0, 0, 0]],
                 cell=[[2, 0, 0], [0, 2, 0], [0, 0, 0]], pbc=(True, True, False))
    symmetry = EmbeddingSymmetry(host, HostSite(0, (0, 0)))
    assert len(symmetry.equivalent_sites(HostSite(0, (1, 0)))) == 4


def test_radius_and_coordination_shells_for_cubic_host():
    host = cubic()
    center = HostSite(0, (0, 0, 0))
    assert embedding_sites_within_radius(host, center, 1.99) == (center,)
    assert len(embedding_sites_within_radius(host, center, 2.01)) == 7
    assert embedding_sites_within_shells(host, center, 0) == (center,)
    assert len(embedding_sites_within_shells(host, center, 1)) == 7
    assert len(embedding_sites_within_shells(host, center, 2)) == 19
    # Multiple centres are the union, not a sphere around their centroid.
    both = embedding_sites_within_radius(host, [center, HostSite(0, (2, 0, 0))], 0.01)
    assert set(both) == {center, HostSite(0, (2, 0, 0))}


def test_2d_radius_never_enumerates_third_dimension():
    host = Atoms("Cu", positions=[[0, 0, 0]],
                 cell=[2, 2, 10], pbc=(True, True, False))
    result = embedding_sites_within_radius(host, (0., 0., 0.), 2.01)
    assert len(result) == 5
    assert all(len(s.cell_index) == 2 for s in result)


def test_empty_or_invalid_region_and_duplicates():
    host = cubic()
    with pytest.raises(ValueError, match="radius"):
        embedding_sites_within_radius(host, (0, 0, 0), -1)
    with pytest.raises(ValueError, match="shells"):
        embedding_sites_within_shells(host, (0, 0, 0), -1)
    with pytest.raises(ValueError, match="too large"):
        embedding_sites_within_radius(host, (0, 0, 0), 20, max_candidates=10)
    with pytest.raises(ValueError, match="center"):
        embedding_sites_within_radius(host, [], 2)


def test_inherit_fractional_host_occupation_without_sharing_mutable_sites():
    host = SPRKKRAtoms("Cu", positions=[[0, 0, 0]], cell=[2, 2, 2])
    host.sites[0].occupation = {"Cu": 0.7, "Ni": 0.3}
    cluster = SPRKKRAtoms.from_embedding(host, [
        (0, (0, 0, 0), "Fe"),
        (0, (1, 0, 0), None),
        (0, (-1, 0, 0), None),
    ])
    assert cluster.sites[1].occupation.to_dict(True) == pytest.approx({"Cu": 0.7, "Ni": 0.3})
    assert cluster.sites[1].site_type is not host.sites[0].site_type
    assert cluster.sites[1].site_type is not cluster.sites[2].site_type
    cluster.sites[1].occupation = {"Co": 1.0}
    assert host.sites[0].occupation.to_dict(True) == pytest.approx({"Cu": 0.7, "Ni": 0.3})
    with pytest.raises(ValueError, match="only occur once"):
        SPRKKRAtoms.from_embedding(host, [
            (0, (0, 0, 0), "Fe"),
            (0, (0, 0, 0, 0, 0), "Co"),
        ])


def test_add_remove_cluster_sites_are_nonmutating():
    host = cubic()
    initial = SPRKKRAtoms.from_embedding(host, [(0, (0, 0, 0), "Fe")])
    added = embedding_add_sites(initial, [HostSite(0, (1, 0, 0))])
    assert len(initial) == 1
    assert len(added) == 2
    assert added.embedding_host is host
    assert added.sites[1].occupation.to_dict(True) == {"Cu": 1.0}
    with pytest.raises(ValueError, match="only once"):
        embedding_add_sites(added, [HostSite(0, (1, 0, 0, 0, 0))])
    removed = embedding_remove_sites(added, [HostSite(0, (0, 0, 0))])
    assert len(removed) == 1
    assert removed.sites[0].embedding.cell_index == (1, 0, 0)
    assert len(added) == 2
    with pytest.raises(ValueError, match="absent"):
        embedding_remove_sites(initial, [HostSite(0, (2, 0, 0))])


def test_2d_potential_lead_indices_are_excluded():
    atoms = Atoms("Cu3", positions=[[0, 0, -2], [0, 0, 0], [0, 0, 2]],
                  cell=[2, 2, 8], pbc=(True, True, False))
    host = PeriodicPotential(atoms, set_from_atoms=False)
    host.LATTICE.SYSDIM = "2D"
    host.LATTICE.SYSTYPE = "LIV"
    host.LATTICE.NQ_L = 1
    host.LATTICE.NQ_R = 1
    assert embedding_host_indices(host) == (1,)
    region = embedding_sites_within_radius(host, (0, 0, 0), 3)
    assert all(s.original_site == 1 for s in region)


def test_tilted_2d_lead_vectors_restrict_host_symmetry():
    from ase.units import Bohr

    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[2, 2, 10],
                  pbc=(True, True, False))
    host = PeriodicPotential(atoms, set_from_atoms=False)
    host.LATTICE.SYSDIM = "2D"
    host.LATTICE.SYSTYPE = "LIV"
    host.LATTICE.NQ_L = 0
    host.LATTICE.NQ_R = 0
    host.LATTICE.ALAT = 1.0 / Bohr
    host.LATTICE.A_L3 = (1, 0, 5)
    host.LATTICE.A_R3 = (1, 0, 5)
    symmetry = EmbeddingSymmetry(host, HostSite(0, (0, 0)))
    # A quarter-turn would rotate the tilted lead vector into a direction
    # which is NOT equivalent by an integer in-plane translation.
    images = set(symmetry.equivalent_sites(HostSite(0, (1, 0))))
    assert images == {HostSite(0, (1, 0)), HostSite(0, (-1, 0))}
    native = HostSite(0, (1, 0, 0, 0, 1))
    assert set(symmetry.equivalent_sites(native)) == {
        HostSite(0, (1, 0, 0, 0, 1)),
        HostSite(0, (-2, 0, 0, 0, 1)),
    }


def test_host_site_exposes_occupation_and_2d_region_metadata():
    host = Atoms("CuNi", positions=[[0, 0, 0], [0, 0, 2]],
                 cell=[2, 2, 8], pbc=(True, True, False))
    site = HostSite(1, (0, 0))
    assert site.occupation(host) == {"Ni": 1.0}
    assert site.region(host) is None
    bulk = cubic()
    assert HostSite(0, (0, 0, 0)).region(bulk) is None
