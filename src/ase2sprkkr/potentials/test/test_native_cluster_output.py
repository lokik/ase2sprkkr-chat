"""Regressions from a real CLUSCF output potential and its periodic host."""

from pathlib import Path

import numpy as np
import pytest

from ..cluster_potentials import EmbeddedPotential
from ..potential_base import Potential


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
