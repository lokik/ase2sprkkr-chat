"""Public embedding API, Site identity, and ASE operation regression tests."""

import numpy as np
import pytest
from ase import Atoms

from ..embedding import EMBEDDING_ARRAY, EMBEDDING_HOST
from ..sprkkr_atoms import SPRKKRAtoms


def host(pbc=(True, True, True)):
    return Atoms("CuNi", positions=[[0, 0, 0], [1, 1, 1]],
                 cell=[2, 2, 4], pbc=pbc)


def test_site_index_is_instance_specific_with_shared_site_type():
    atoms = SPRKKRAtoms("Cu2", positions=[[0, 0, 0], [1, 0, 0]],
                        cell=[2, 2, 2], symmetry=False)
    sites = atoms.sites
    sites[1].site_type = sites[0].site_type
    assert sites[0].index == 0
    assert sites[1].index == 1
    assert sites[0].site_type is sites[1].site_type


def test_embedding_displacement_and_change_of_reference():
    h = host()
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (1, 0, 0), "Fe"),
                                           (1, (0, 1, 0), {"Ni": 1.0})])
    first = atoms.sites[0]
    assert first.index == 0
    assert first.embedding.original_site == 0
    assert first.embedding.cell_index == (1, 0, 0)
    np.testing.assert_allclose(first.embedding.reference_position, [2, 0, 0])
    np.testing.assert_allclose(first.embedding.displacement, [0, 0, 0])

    first.embedding.displacement = [0.01, 0.02, 0.03]
    np.testing.assert_allclose(atoms.positions[0], [2.01, 0.02, 0.03])
    atoms.positions[0, 2] += 0.07
    np.testing.assert_allclose(first.embedding.displacement, [0.01, 0.02, 0.10])

    first.embedding.cell_index = [2, -1, 0]
    assert first.embedding.cell_index == (2, -1, 0)
    np.testing.assert_allclose(first.embedding.displacement, [0.01, 0.02, 0.10])
    first.embedding.original_site = 1
    assert first.embedding.original_site == 1
    np.testing.assert_allclose(first.embedding.displacement, [0.01, 0.02, 0.10])
    assert atoms.arrays[EMBEDDING_ARRAY].dtype.kind in "iu"
    assert atoms.arrays[EMBEDDING_ARRAY].shape == (2, 6)


def test_embedding_requires_host_and_rejects_invalid_indices():
    h = host()
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (0, 0, 0), "Fe")])
    atoms.info.pop(EMBEDDING_HOST)
    with pytest.raises(ValueError, match="sprkkr_embedding_host"):
        atoms.sites[0].embedding.cell_index = (1, 0, 0)
    with pytest.raises(ValueError, match="sprkkr_embedding_host"):
        atoms.sites[0].embedding.displacement = (0, 0, 0)
    atoms.info[EMBEDDING_HOST] = h
    with pytest.raises(TypeError):
        atoms.sites[0].embedding.cell_index = (0.5, 0, 0)
    with pytest.raises(IndexError):
        atoms.sites[0].embedding.original_site = 10


def test_2d_embedding_cell_indices_and_native_translations():
    h = host(pbc=(True, True, False))
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (1, 2), "Fe")])
    site = atoms.sites[0]
    assert site.embedding.cell_index == (1, 2)
    site.embedding.cell_index = (1, 2, 0, 0, 0)
    assert site.embedding.translation_vector == (1, 2, 0, 0, 0)
    with pytest.raises(ValueError, match="Expected 2 or 5"):
        site.embedding.cell_index = (1, 2, 3)
    with pytest.raises(ValueError):
        site.embedding.cell_index = [1]


def test_copy_slice_preserve_host_identity_but_copy_site_types():
    h = host()
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (0, 0, 0), "Fe"),
                                           (1, (1, 0, 0), "Ni")])
    original_type = atoms.sites[0].site_type
    copied = atoms.copy()
    sliced = atoms[1:]
    assert copied.info[EMBEDDING_HOST] is h
    assert sliced.info[EMBEDDING_HOST] is h
    assert copied.sites[0] is not atoms.sites[0]
    assert copied.sites[0].site_type is not original_type
    assert sliced.sites[0] is not atoms.sites[1]
    assert sliced.sites[0].atoms is sliced
    assert sliced.sites[0].index == 0
    np.testing.assert_array_equal(copied.arrays[EMBEDDING_ARRAY], atoms.arrays[EMBEDDING_ARRAY])
    np.testing.assert_array_equal(sliced.arrays[EMBEDDING_ARRAY], atoms.arrays[EMBEDDING_ARRAY][1:])


def test_repeat_creates_distinct_sites_and_preserves_displacement():
    h = host()
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (0, 0, 0), "Fe"),
                                           (1, (0, 0, 0), "Ni")])
    atoms.sites[0].embedding.displacement = (0.08, 0, 0)
    result = atoms.repeat((2, 1, 1))
    assert result.info[EMBEDDING_HOST] is h
    assert len({id(s) for s in result.sites}) == 4
    assert all(site.atoms is result for site in result.sites)
    np.testing.assert_array_equal(result.arrays[EMBEDDING_ARRAY][:, 1:4],
                                  [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(result.sites[2].embedding.displacement, [0.08, 0, 0])
    np.testing.assert_allclose(result.sites[2].embedding.reference_position, [2, 0, 0])
    assert result.sites[2] is not result.sites[0]
    assert result.sites[2].site_type is result.sites[0].site_type


def test_incommensurate_repeat_raises_before_mutation():
    h = host()
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (0, 0, 0), "Fe")])
    atoms.cell[0, 0] = 2.3
    before = atoms.positions.copy()
    with pytest.raises(ValueError, match="commensurate"):
        atoms *= (2, 1, 1)
    assert len(atoms) == 1
    np.testing.assert_array_equal(atoms.positions, before)


def test_extend_refuses_mixed_embedding():
    h = host()
    atoms = SPRKKRAtoms.from_embedding(h, [(0, (0, 0, 0), "Fe")])
    with pytest.raises(ValueError, match="non-embedded"):
        atoms.extend(SPRKKRAtoms("Cu"))
