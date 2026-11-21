"""The fast numeric path preserves native potential radial arrays."""

from pathlib import Path

import numpy as np
import pytest

from ..potential_base import Potential
from ..definitions.sections.potential import PotentialSectionDefinition
from ..definitions.sections.charge import ChargeSectionDefinition
from ..definitions.sections.moments import MomentsSectionDefinition


EXAMPLES = Path(__file__).parent.parent / "examples"


def test_radial_sections_use_fast_numpy_conversion():
    for cls in (
        PotentialSectionDefinition,
        ChargeSectionDefinition,
        MomentsSectionDefinition,
    ):
        section = cls()
        data = section["DATA"].type
        assert data.trivial_parsing is True
        assert data.dtypes == [float]


@pytest.mark.smoke
@pytest.mark.parametrize("filename", (
    "FePt.new.pot",
    "Fe3CoFe3.clu.iterating.pot",
))
def test_native_numeric_potential_blocks_are_floating_arrays(filename):
    potential = Potential.from_file(EXAMPLES / filename, atoms=False)
    seen = set()

    for section_name in ("POTENTIAL", "CHARGE", "MOMENTS"):
        sections = getattr(potential, section_name)
        for section in sections:
            array = section.DATA()
            assert isinstance(array, np.ndarray)
            assert np.issubdtype(array.dtype, np.floating)
            if section_name in ("POTENTIAL", "CHARGE"):
                assert array.ndim == 2 and array.shape[0] == 2
            else:
                assert array.shape == (5,)
            seen.add(section_name)

    assert {"POTENTIAL", "CHARGE"} <= seen
