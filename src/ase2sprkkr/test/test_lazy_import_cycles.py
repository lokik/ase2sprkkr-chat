"""Import-order regressions: user hooks may import root API while it initializes."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("requested", [
    "InputParameters",
    "Potential",
    "SPRKKRAtoms",
    "SPRKKR",
])
def test_public_export_is_importable_in_isolated_process(requested):
    program = (
        "from ase2sprkkr import " + requested + "\n"
        "assert " + requested + " is not None\n"
    )
    env = dict(os.environ, ASE2SPRKKR_NO_USER_PROFILE="1")
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_user_hook_can_import_lazy_root_classes(tmp_path):
    # Platformdirs reads XDG_CONFIG_HOME on Linux. The subprocess ensures an
    # actual cold import and prevents pytest module caching from hiding cycles.
    config_dir = tmp_path / "ase2sprkkr"
    config_dir.mkdir()
    (config_dir / "__init__.py").write_text(
        "import builtins\n"
        "from ase2sprkkr import InputParameters, Potential, SPRKKRAtoms\n"
        "builtins._ase2sprkkr_hook_types = (InputParameters, Potential, SPRKKRAtoms)\n",
        encoding="utf-8",
    )
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path))
    env.pop("ASE2SPRKKR_NO_USER_PROFILE", None)
    program = """
import builtins
import ase2sprkkr
assert builtins._ase2sprkkr_hook_types == (
    ase2sprkkr.InputParameters, ase2sprkkr.Potential, ase2sprkkr.SPRKKRAtoms
)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_sprkkr_atoms_import_does_not_import_potentials_until_requested():
    program = """
import sys
from ase2sprkkr.sprkkr.sprkkr_atoms import SPRKKRAtoms
assert SPRKKRAtoms is not None
assert "ase2sprkkr.potentials.potentials" not in sys.modules
"""
    env = dict(os.environ, ASE2SPRKKR_NO_USER_PROFILE="1")
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
