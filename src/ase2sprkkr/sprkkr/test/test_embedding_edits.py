import numpy as np
import pytest
from ase.build import bulk

from ase2sprkkr import SPRKKRAtoms
from ..embedding_edits import (
    EmbeddedSiteParameters,
    embedded_edit_discards_radial_data,
    rebuild_embedded_cluster,
)
from ..embedding_geometry import HostSite


@pytest.mark.parametrize("shallow", [False, True])
def test_rebuild_reuses_types_on_equivalent_fullpot_meshes(shallow):
    host = bulk("Fe")
    original = SPRKKRAtoms.from_embedding(
        host, [(0, (0, 0, 0), "Fe"), (0, (1, 0, 0), "Co")]
    )
    original_types = {site.atomic_type(0).symbol: site.atomic_type(0) for site in original.sites}
    assert original_types["Fe"].mesh is not original_types["Co"].mesh
    for atomic_type in original_types.values():
        atomic_type.mesh.set_fullpot((2, 4, 2, np.array([4, 8])))
        atomic_type.potential = np.ones((2, len(atomic_type.mesh)))
        atomic_type.charge = np.ones(len(atomic_type.mesh))
        atomic_type.moments = (0., 0., 0., 0., 0.)

    entries = {HostSite(0, (0, 0, 0)): EmbeddedSiteParameters({"Fe": .5, "Co": .5})}
    assert not embedded_edit_discards_radial_data(original, entries)
    edited = rebuild_embedded_cluster(host, entries, original=original, shallow=shallow)
    assert edited.info["sprkkr_embedding_scf_status"] == "ITERATING"
    for atomic_type in edited.sites[0].occupation:
        source = original_types[atomic_type.symbol]
        assert atomic_type is not source
        assert atomic_type.mesh is edited.sites[0].site_type.mesh
        assert np.shares_memory(atomic_type.potential.raw_value, source.potential.raw_value) == shallow
