"""Cold-import boundaries and public-root API compatibility."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_root_import_does_not_eagerly_load_calculator_or_ase4_plugins():
    code = """
import sys
import ase2sprkkr
assert "ase2sprkkr.sprkkr.calculator" not in sys.modules
assert "ase2sprkkr.outputs.task_result" not in sys.modules
assert "ase._4.plugins" not in sys.modules
assert "SPRKKR" in dir(ase2sprkkr)
assert "TaskResult" in dir(ase2sprkkr)
"""
    env = dict(os.environ, ASE2SPRKKR_NO_USER_PROFILE="1")
    process = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_public_classes_and_helpers_resolve_to_original_objects():
    import ase2sprkkr
    from ase2sprkkr.input_parameters.input_parameters import InputParameters
    from ase2sprkkr.outputs.task_result import TaskResult
    from ase2sprkkr.potentials.potential_base import Potential
    from ase2sprkkr.potentials.potentials import PeriodicPotential
    from ase2sprkkr.potentials.cluster_potentials import EmbeddedPotential, ClusterPotential
    from ase2sprkkr.result_files import ResultFile
    from ase2sprkkr.sprkkr.calculator import SPRKKR
    from ase2sprkkr.sprkkr.sprkkr_atoms import SPRKKRAtoms

    for key, expected in (
        ("SPRKKR", SPRKKR),
        ("InputParameters", InputParameters),
        ("Potential", Potential),
        ("PeriodicPotential", PeriodicPotential),
        ("EmbeddedPotential", EmbeddedPotential),
        ("ClusterPotential", ClusterPotential),
        ("ResultFile", ResultFile),
        ("TaskResult", TaskResult),
        ("SPRKKRAtoms", SPRKKRAtoms),
    ):
        assert getattr(ase2sprkkr, key) is expected
        assert getattr(ase2sprkkr, key) is expected

    assert ase2sprkkr.promote_ase_atoms == SPRKKRAtoms.promote_ase_atoms
    with pytest.warns(DeprecationWarning, match="OutputFile"):
        assert ase2sprkkr.OutputFile is ResultFile
