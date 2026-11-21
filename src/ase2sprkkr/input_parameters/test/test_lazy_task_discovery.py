"""Task enumeration reads class metadata without constructing native IPDs."""

import os
import subprocess
import sys
import textwrap

import pytest


def _cold_run(tmp_path, code):
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(tmp_path / "isolated-config")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.smoke
def test_task_names_are_cached_without_materializing_ipds(tmp_path):
    _cold_run(tmp_path, """
        from ase2sprkkr.input_parameters.input_parameters import InputParameters

        schemas = InputParameters.input_parameters_registry
        tasks = InputParameters.task_registry
        assert not schemas._values
        assert not tasks.extra_tasks._values

        names = tasks.names()
        assert tasks.names() is names
        assert len(names) == len(set(names))
        assert names == tuple(sorted(names))
        assert {"SCF", "DOS", "XAS", "BSFEK", "BSFKK", "JXC", "DMI"} <= set(names)
        assert "BSF" not in names
        assert "ARPES" in names  # Declared separately in tasks/arpes.py.
        assert schemas._values == {}, sorted(schemas._values)

        assert tasks["DOS"] is schemas.definition_class("DOS")._task_definition
        assert tasks["SCF"] is schemas.definition_class("SCF")._task_definition
        assert tasks["BSFEK"].input_parameters_definition_name == "BSF"
        assert tasks["BSFKK"].input_parameters_definition_name == "BSF"
        assert tasks["DMI"].input_parameters_definition_name == "JXC"
        assert schemas._values == {}, sorted(schemas._values)

        assert tasks.names(name_contains="DO") == ("DOS",)
        assert schemas._values == {}
        assert tasks.names.cache_info().hits >= 1
    """)


@pytest.mark.smoke
def test_class_task_metadata_is_same_object_as_materialized_task(tmp_path):
    _cold_run(tmp_path, """
        from ase2sprkkr.input_parameters.input_parameters import InputParameters

        schemas = InputParameters.input_parameters_registry
        tasks = InputParameters.task_registry
        all_names = tasks.names()
        assert not schemas._values

        declared = set()
        for name in schemas.names(include_aliases=False):
            cls = schemas.definition_class(name)
            if cls is None:
                continue
            task = getattr(cls, "_task_definition", None)
            if task is None:
                continue
            assert task.name == name, (name, task.name)
            assert tasks[name] is task
            assert name in all_names
            declared.add(name)
        assert declared
        assert not schemas._values

        # AIPES and the PLOT helper create a copy/base-class instance rather
        # than an instance of the exported definition class.
        for name in ("DOS", "SCF", "AIPES", "BAND", "SPLEED",
                     "PLOTPOT", "PLOT2DPOT", "PLOT2DRHO", "GILBERT"):
            expected = tasks[name]
            schema = schemas[name]
            assert schema._task_definition is expected, name
            assert schemas[name] is schema
        assert "BSF" not in tasks.names()
        assert schemas["BSF"]._task_definition is None
        assert schemas["JXC"]._task_definition is None
    """)


@pytest.mark.smoke
def test_extras_validate_schema_reference_without_materialization(tmp_path):
    _cold_run(tmp_path, """
        from ase2sprkkr.input_parameters.input_parameters import InputParameters

        schemas = InputParameters.input_parameters_registry
        tasks = InputParameters.task_registry
        for name in ("BSFEK", "BSFKK", "DMI", "JXC"):
            task = tasks[name]
            assert task.input_parameters_definition_name in schemas
        assert schemas._values == {}
    """)


@pytest.mark.smoke
def test_custom_task_is_declared_on_class_without_building_schema(tmp_path):
    config_root = tmp_path / "xdg"
    directory = config_root / "ase2sprkkr" / "input_parameters"
    directory.mkdir(parents=True)
    (directory / "custom.py").write_text(textwrap.dedent("""
        from ase2sprkkr.input_parameters.task_definition import TaskDefinition
        from ase2sprkkr.input_parameters.input_parameters_definitions import (
            InputParametersDefinition,
        )

        class UserDefinition(InputParametersDefinition):
            _task_definition = TaskDefinition("CUSTOM", executable="kkrgen")

            @classmethod
            def create_definition(cls):
                raise AssertionError("Listing tasks must not construct this IPD")

        definition = UserDefinition
    """))

    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(config_root)
    code = textwrap.dedent("""
        from ase2sprkkr.input_parameters.input_parameters import InputParameters

        schemas = InputParameters.input_parameters_registry
        tasks = InputParameters.task_registry
        assert "CUSTOM" in tasks.names()
        assert tasks["CUSTOM"] is schemas.definition_class("CUSTOM")._task_definition
        assert schemas._values == {}
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
