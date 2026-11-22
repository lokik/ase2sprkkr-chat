import os

import numpy as np
import pytest
from ase.atoms import Atoms

from ..calculator import SPRKKR


@pytest.mark.slow
def test_arpes_run(calc_args):
    if os.environ.get("DO_NOT_RUN_SPRKKR", ""):
        pytest.skip("SPR-KKR execution disabled by DO_NOT_RUN_SPRKKR")

    atoms = Atoms(
        symbols="Cu",
        cell=np.array([(1.0, 0, 0), (0, 1, 0), (0, 0, 1)]),
        pbc=True,
    )
    calculator = SPRKKR(atoms=atoms, **calc_args())

    # A single inexpensive SCF iteration is sufficient to generate the
    # potential for this ARPES integration smoke test.
    scf_output = calculator.calculate(
        **calc_args(options={"NITER": 1, "NE": 8, "NKTAB": 4, "NL": 2})
    )
    assert len(scf_output.iterations) == 1

    arpes_calculator = scf_output.calculator
    arpes_calculator.input_parameters = "arpes"

    # The production ARPES defaults (50 layers, 57 reciprocal vectors and
    # layer doubling [10, 10]) make a smoke test unnecessarily expensive.
    # Keep a small but nontrivial layer construction and energy grid, while
    # still running the native ARPES executable and reading its SPC output.
    arpes_output = arpes_calculator.calculate(
        **calc_args(
            options={
                "NE": 5,
                "NKTAB": 4,
                "NL": 2,
                "SPEC_STR.N_LAYER": 6,
                "SPEC_STR.NLAT_G_VEC": 9,
                "SPEC_STR.N_LAYDBL": [3, 3],
            }
        )
    )
    energies = arpes_output.spectra[0].ENERGY()
    assert isinstance(energies, np.ndarray)
    assert energies.size > 0
