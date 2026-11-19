"""Regressions from a real CLUSCF output potential and its periodic host."""

from pathlib import Path

import numpy as np
import pytest

from ..cluster_potentials import EmbeddedPotential
from ..potential_base import Potential
from ...sprkkr.calculator import SPRKKR


EXAMPLES = Path(__file__).parent.parent / "examples"
CLUSTER = EXAMPLES / "Fe3CoFe3.clu.iterating.pot"
HOST = EXAMPLES / "Fe3CoFe3.host.pot"


@pytest.mark.parametrize("reader", [Potential, EmbeddedPotential])
def test_read_computed_cluster_potential(reader):
    cluster = reader.from_file(CLUSTER)

    assert isinstance(cluster, EmbeddedPotential)
    assert cluster.SCF_INFO.SCFSTATUS() == "ITERATING"
    assert cluster.SCF_INFO.SCF_ITER() == 34
    np.testing.assert_array_equal(cluster.SCF_INFO.NE(), [32, 0])
    assert cluster.LATTICE.SYSDIM() == "3D"
    assert cluster.LATTICE.SYSTYPE() == "EMBEDDED-CLUSTER"
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NQCLU() == 7
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NTCLU() == 3
    assert cluster.GLOBAL_SYSTEM_PARAMETER.NMCLU() == 0
    for name in ("POTENTIAL", "CHARGE", "MOMENTS"):
        assert [block.TYPE() for block in getattr(cluster, name).values()] == [3, 4, 5]


def test_computed_cluster_atoms_and_radial_round_trip(tmp_path):
    host = Potential.from_file(HOST)
    cluster = Potential.from_file(CLUSTER, host=host)
    atoms = cluster.atoms

    assert len(atoms) == 7
    assert str(atoms.symbols) == "Fe3CoFe3"
    assert atoms.embedding_host is host
    for index, site in enumerate(atoms.sites):
        local_type = int(cluster.OCCUPATION.DATA()[index]["ITCLU CONC"][0][0])
        atomic_type = next(iter(site.occupation))
        np.testing.assert_allclose(
            atomic_type.potential.raw_value, cluster.POTENTIAL[local_type - 1].DATA()
        )
        np.testing.assert_allclose(
            atomic_type.charge.raw_value, cluster.CHARGE[local_type - 1].DATA()
        )

    output = tmp_path / "roundtrip.clupot"
    cluster.save_to_file(output)
    restored = Potential.from_file(output, host=host)
    np.testing.assert_array_equal(restored.SCF_INFO.NE(), [32, 0])
    np.testing.assert_allclose(restored.atoms.positions, atoms.positions)
    for name in ("POTENTIAL", "CHARGE", "MOMENTS"):
        for original, saved in zip(getattr(cluster, name).values(), getattr(restored, name).values()):
            assert saved.TYPE() == original.TYPE()
            np.testing.assert_array_equal(saved.DATA(), original.DATA())


@pytest.mark.parametrize("shallow", [False, True])
def test_computed_cluster_copy_preserves_native_restart_state(tmp_path, shallow):
    host = Potential.from_file(HOST)
    source = Potential.from_file(CLUSTER, host=host)
    atoms = source.atoms
    copied = atoms.copy(shallow=shallow)

    assert copied is not atoms
    assert copied.embedding_host is host
    assert copied.has_potential()
    assert copied.potential is not source
    assert copied.potential.atoms is copied
    assert copied.potential.host is host
    assert Potential.from_atoms(copied) is copied.potential
    np.testing.assert_array_equal(copied.potential.OCCUPATION.DATA(), source.OCCUPATION.DATA())
    assert copied.sites[0].site_type is not copied.sites[1].site_type
    assert copied.sites[0].atomic_type(0) is copied.sites[1].atomic_type(0)
    assert copied.sites[0].atomic_type(0) is not atoms.sites[0].atomic_type(0)
    assert copied.sites[0].site_type.mesh is copied.sites[1].site_type.mesh
    assert copied.sites[0].site_type.mesh is not atoms.sites[0].site_type.mesh

    prepared = SPRKKR(atoms=copied).prepare(
        directory=tmp_path, input_file="restart.inp", potential_file="host.pot",
        output_file=False, empty_spheres=False,
    )
    try:
        written = Potential.from_file(prepared.cluster_potential_file, host=host)
        assert written.GLOBAL_SYSTEM_PARAMETER.NTCLU() == 3
        np.testing.assert_array_equal(written.OCCUPATION.DATA(), source.OCCUPATION.DATA())
        for name in ("POTENTIAL", "CHARGE", "MOMENTS"):
            for original, saved in zip(getattr(source, name).values(), getattr(written, name).values()):
                assert saved.TYPE() == original.TYPE()
                np.testing.assert_array_equal(saved.DATA(), original.DATA())
    finally:
        prepared.close()

    copied.sites[0].atomic_type(0).potential.raw_value[0, 0] = 123.0
    assert (atoms.sites[0].atomic_type(0).potential.raw_value[0, 0] == 123.0) == shallow
    copied.potential.POTENTIAL[0].DATA()[0, 0] = 456.0
    assert (source.POTENTIAL[0].DATA()[0, 0] == 456.0) == shallow
