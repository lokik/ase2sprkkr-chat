"""Lazy IPD module discovery and single materialization contract."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run_cold(script, tmp_path):
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(tmp_path / "fresh-configuration")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


@pytest.mark.smoke
def test_all_builtin_definitions_export_lightweight_classes(tmp_path):
    _run_cold(
        """
from ase2sprkkr.input_parameters.input_parameters import InputParameters
from ase2sprkkr.input_parameters.input_parameters_definitions import _task_selector_map

registry = InputParameters.input_parameters_registry
for name in registry.names(include_aliases=False):
    cls = registry.definition_class(name)
    assert cls is not None, name
    assert callable(cls.create_definition), name
assert not registry._values, sorted(registry._values)

selectors = _task_selector_map()
assert not registry._values, sorted(registry._values)
assert 'DOS' in selectors
assert 'DMI' in selectors
assert 'JXC' in selectors
assert selectors['DMI'] == selectors['JXC']
assert set(registry._values) == {'JXC'}
assert 'DOS' not in registry._values
""", tmp_path,
    )


@pytest.mark.smoke
def test_dos_autodetection_avoids_all_other_definition_modules(tmp_path):
    path = EXAMPLES / "dos.in"
    _run_cold(
        f"""
from ase2sprkkr.input_parameters.input_parameters import InputParameters
from ase2sprkkr.input_parameters.input_parameters_definitions import (
    InputParametersDefinition, _task_selector_map,
)

reg = InputParameters.input_parameters_registry
assert not reg._values
parameters = InputParametersDefinition.from_file({str(path)!r})
assert parameters.definition_name == 'DOS'
assert _task_selector_map.cache_info().currsize == 1
assert 'DOS' in reg._values
assert set(reg._values) == {{'DOS'}}
assert 'JXC' not in reg._modules
assert 'ARPES' not in reg._modules
assert 'JXC' not in reg._values
assert 'ARPES' not in reg._values
""", tmp_path,
    )


@pytest.mark.smoke
def test_registry_calls_create_definition_once(tmp_path):
    _run_cold(
        """
from ase2sprkkr.input_parameters.input_parameters import InputParameters

reg = InputParameters.input_parameters_registry
cls = reg.definition_class('DOS')
original = cls.create_definition.__func__
calls = []

def counted(definition_type):
    calls.append(definition_type)
    return original(definition_type)

cls.create_definition = classmethod(counted)
one = reg['DOS']
two = reg['DOS']
assert one is two
assert calls == [cls], calls
""", tmp_path,
    )


@pytest.mark.smoke
def test_builtin_selector_aliases_agree_with_declared_classes(tmp_path):
    _run_cold(
        """
from ase2sprkkr.input_parameters.input_parameters import InputParameters
from ase2sprkkr.input_parameters.input_parameters_definitions import (
    _builtin_ipd_names, _BUILTIN_TASK_SELECTOR_ALIASES, _normalized_task_selectors,
)

reg = InputParameters.input_parameters_registry
aliases = _BUILTIN_TASK_SELECTOR_ALIASES
assert aliases == {'DMI': 'JXC'}
for name in _builtin_ipd_names():
    cls = reg.definition_class(name)
    expected = {name, *(alias for alias, target in aliases.items() if target == name)}
    declared = set(_normalized_task_selectors(cls.task_selectors, name))
    assert declared == expected, (name, declared, expected)
assert not reg._values
""", tmp_path,
    )
