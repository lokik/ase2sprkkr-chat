"""Regression coverage for host scaling and negative supercell translations."""
import numpy as np
import pytest
from ase import Atoms
from ase2sprkkr.sprkkr.sprkkr_atoms import SPRKKRAtoms
from ase2sprkkr.sprkkr.embedding_transforms import (
    rebase_embedded_host, repeat_embedded_host, embedded_host_transform_capability
)


def cluster():
    h = Atoms("CuNi", positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 4], pbc=True)
    c = SPRKKRAtoms.from_embedding(
        h, [(0, (-1, 2, 0), "Fe"), (1, (2, -1, 0), "Co")]
    )
    c.sites[0].embedding.displacement = [0.2, -0.1, 0.3]
    return c


def test_change_host_lattice_preserves_embedding_displacement_and_source():
    c = cluster()
    h = c.embedding_host.copy()
    h.set_cell([4, 2, 4], scale_atoms=True)
    result = rebase_embedded_host(c, h)
    assert result is not c
    np.testing.assert_allclose(result.sites[0].embedding.displacement, [0.4, -0.1, 0.3])
    np.testing.assert_array_equal(result.arrays["sprkkr_embedding"],
                                  c.arrays["sprkkr_embedding"])
    np.testing.assert_allclose(c.embedding_host.cell, np.diag([2, 2, 4]))
    assert result.info["sprkkr_embedding_scf_status"] == "START"


def test_repeat_host_remaps_negative_cell_indices_without_replicating_cluster():
    c = cluster()
    result = repeat_embedded_host(c, (2, 3, 1))
    assert len(result) == 2
    assert len(result.embedding_host) == 12
    np.testing.assert_allclose(result.positions, c.positions)
    np.testing.assert_array_equal(result.arrays["sprkkr_embedding"][:, 1:4],
                                  [[-1, 0, 0], [1, -1, 0]])
    np.testing.assert_allclose(result.sites[0].embedding.displacement,
                               c.sites[0].embedding.displacement)


def test_reject_wrong_host_count_without_mutation():
    c = cluster()
    old_positions = c.positions.copy()
    with pytest.raises(ValueError, match="site count"):
        rebase_embedded_host(c, Atoms("Cu", cell=[2, 2, 4], pbc=True))
    np.testing.assert_array_equal(c.positions, old_positions)


def test_disable_miller_rotation_until_site_mapping_exists():
    allowed, reason = embedded_host_transform_capability(cluster(), "rotate")
    assert not allowed and "mapping" in reason
