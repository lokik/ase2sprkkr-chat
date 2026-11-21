"""Regression tests for ASE I/O of POTFIL and CLUPOT."""

from io import StringIO

import numpy as np
import pytest
from ase.build import bulk
from ase.io import read as ase_read, write as ase_write

from ..io import read_sprkkr, write_sprkkr
from ...potentials.cluster_potentials import ClusterPotential
from ...potentials.potentials import Potential
from ...sprkkr.sprkkr_atoms import SPRKKRAtoms


@pytest.fixture
def embedded():
    host = Potential.from_atoms(bulk("Cu", "fcc", a=3.61))
    atoms = SPRKKRAtoms.from_embedding(host, [(0, (1, 0, 0), "Fe")])
    return host, atoms


def test_writer_accepts_atoms_or_single_image_list(embedded):
    host, atoms = embedded
    direct = StringIO()
    listed = StringIO()
    write_sprkkr(direct, atoms, validate=False)
    write_sprkkr(listed, [atoms], validate=False)
    assert Potential.is_cluster_file(StringIO(direct.getvalue()))
    assert Potential.is_cluster_file(StringIO(listed.getvalue()))
    assert ClusterPotential.from_string(direct.getvalue()).SITES.DATA()[0]["IQ"] == 1

    with pytest.raises(ValueError, match="exactly one"):
        write_sprkkr(StringIO(), [atoms, atoms], validate=False)


def test_reader_requires_host_and_supports_potential_object(embedded, tmp_path):
    host, atoms = embedded
    path = tmp_path / "embedded.pot"
    write_sprkkr(path, atoms, validate=False)
    with pytest.raises(ValueError, match="host potential"):
        read_sprkkr(path)

    restored = read_sprkkr(path, host=host)
    assert restored.info["sprkkr_embedding_host"] is host
    assert isinstance(restored.potential, ClusterPotential)
    assert restored.sites[0].embedding.cell_index == (1, 0, 0)
    np.testing.assert_allclose(restored.positions, atoms.positions)


def test_reader_accepts_host_file_and_nonseekable_stream(embedded, tmp_path):
    host, atoms = embedded
    host_path = tmp_path / "host.pot"
    host.save_to_file(host_path, validate=False)
    output = StringIO()
    write_sprkkr(output, atoms, validate=False)

    class NonSeekable:
        def __init__(self, text):
            self.text = text

        def read(self):
            return self.text

    restored = read_sprkkr(NonSeekable(output.getvalue()), host=host_path)
    assert isinstance(restored.potential, ClusterPotential)
    assert restored.embedding_host is not host
    assert restored.embedding_host.LATTICE.SYSDIM() == "3D"
    np.testing.assert_allclose(restored.positions, atoms.positions)


def test_writer_uses_edited_cluster_atoms_not_stale_potential(embedded):
    host, atoms = embedded
    source = StringIO()
    write_sprkkr(source, atoms, validate=False)
    source.seek(0)
    loaded = read_sprkkr(source, host=host)
    old_position = loaded.positions[0].copy()
    loaded.sites[0].embedding.displacement = (0.2, 0, 0)

    updated = StringIO()
    write_sprkkr(updated, loaded, validate=False)
    updated.seek(0)
    reread = read_sprkkr(updated, host=host)
    np.testing.assert_allclose(reread.positions, loaded.positions)
    np.testing.assert_allclose(reread.positions[0] - old_position, [0.2, 0, 0])
    np.testing.assert_allclose(reread.sites[0].embedding.displacement, [0.2, 0, 0])


def test_regular_potential_still_uses_periodic_format(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.61)
    path = tmp_path / "bulk.pot"
    write_sprkkr(path, atoms, validate=False)
    assert not Potential.is_cluster_file(path)
    restored = read_sprkkr(path)
    assert restored.pbc.all()
    assert len(restored) == len(atoms)


def test_ase_read_write_plugin_supports_clupot(embedded, tmp_path):
    host, atoms = embedded
    path = tmp_path / "cluster.clupot"
    ase_write(path, atoms, format="SPRKKR")
    loaded = ase_read(path, format="SPRKKR", host=host)
    assert isinstance(loaded.potential, ClusterPotential)
    np.testing.assert_allclose(loaded.positions, atoms.positions)

    # The .pot extension is shared, so detection must use file contents.
    alternative = tmp_path / "cluster.pot"
    ase_write(alternative, loaded, format="SPRKKR")
    loaded_again = ase_read(alternative, format="SPRKKR", host=host)
    np.testing.assert_allclose(loaded_again.positions, atoms.positions)
