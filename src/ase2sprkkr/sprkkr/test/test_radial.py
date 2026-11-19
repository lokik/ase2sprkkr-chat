from types import SimpleNamespace

import numpy as np
import pytest

from ase2sprkkr.sprkkr.radial import RadialPotential


def test_julich_representation_round_trip():
    mesh = SimpleNamespace(coors=np.asarray([0.5, 1.0, 2.0, 4.0]))
    z = 3.0
    munchen = np.asarray(
        [
            [10, 11, 12, 13],
            [1, 2, 3, 4],
        ]
    )

    potential = RadialPotential.from_munchen(munchen.copy(), mesh, z)
    zr = z / mesh.coors
    expected_up = munchen[0] + munchen[1] + 2 * zr
    expected_down = munchen[0] - munchen[1] + 2 * zr

    np.testing.assert_allclose(potential.zr, zr)
    np.testing.assert_allclose(potential.up, expected_up)
    np.testing.assert_allclose(potential.down, expected_down)
    np.testing.assert_allclose(
        potential.julich,
        np.asarray([expected_up, expected_down]),
    )

    reconstructed = RadialPotential.from_julich(potential.julich, mesh, z)
    np.testing.assert_allclose(reconstructed.munchen, munchen)


def test_from_julich_keeps_fractional_values_for_integer_input():
    mesh = SimpleNamespace(coors=np.asarray([1.0, 2.0, 4.0]))
    julich = np.asarray(
        [
            [11, 12, 13],
            [4, 5, 6],
        ]
    )

    potential = RadialPotential.from_julich(julich, mesh, 1.0)

    np.testing.assert_allclose(
        potential.munchen,
        [
            [5.5, 7.5, 9.0],
            [3.5, 3.5, 3.5],
        ],
    )
    assert np.issubdtype(potential.munchen.dtype, np.floating)
    np.testing.assert_allclose(potential.julich, julich)


def test_zr_tracks_public_z_value():
    mesh = SimpleNamespace(coors=np.asarray([1.0, 2.0, 4.0]))
    potential = RadialPotential(np.zeros((2, 3)), mesh, 2.0)

    np.testing.assert_allclose(potential.zr, [2.0, 1.0, 0.5])
    potential.z = 4.0
    np.testing.assert_allclose(potential.zr, [4.0, 2.0, 1.0])



@pytest.mark.parametrize("shallow", [False, True])
def test_fullpot_payload_survives_equivalent_mesh_copy_and_refuses_remesh(shallow):
    from ase2sprkkr.sprkkr.radial import RadialCharge
    from ase2sprkkr.sprkkr.radial_meshes import ExponentialMesh

    mesh = ExponentialMesh(1e-6, 0.02, 2, 1.0, 3, 1.5)
    mesh.set_fullpot((1, 2, 1, [3]))
    payload = "NFP\n  1  2  3\n  4.0  5.0"
    radial_potential = RadialPotential(np.ones((2, 3)), mesh, 26, fullpot=payload)
    radial_charge = RadialCharge(np.ones((2, 3)), mesh, fullpot=payload)
    for radial in (radial_potential, radial_charge):
        clone = mesh.copy()
        clone.set_fullpot((1, 2, 1, [3]))
        copied = radial.copy(clone, shallow=shallow)
        assert copied.fullpot_data == payload
        assert copied.mesh is clone
        np.testing.assert_array_equal(copied.raw_value, radial.raw_value)
        assert np.shares_memory(copied.raw_value, radial.raw_value) == shallow

        different = ExponentialMesh(1e-6, 0.03, 2, 1.0, 3, 1.5)
        different.set_fullpot((1, 2, 1, [3]))
        with pytest.raises(ValueError, match="cannot be remeshed"):
            radial.copy(different, shallow=shallow)
        assert radial.mesh is mesh
        assert radial.fullpot_data == payload
