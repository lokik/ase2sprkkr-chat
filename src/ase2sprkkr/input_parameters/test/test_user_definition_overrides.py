"""User IPD overrides are selected without importing all built-in schemas."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _run_with_modules(tmp_path, modules, script):
    # definitions.__path__ includes platformdirs.user_config_dir("ase2sprkkr", ...).
    # An isolated XDG_CONFIG_HOME makes registry discovery deterministic.
    config_root = tmp_path / "xdg"
    directory = config_root / "ase2sprkkr" / "input_parameters"
    directory.mkdir(parents=True)
    for filename, code in modules.items():
        (directory / filename).write_text(textwrap.dedent(code))

    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(config_root)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def _dos_override(class_name, schema_name, *, preset=False):
    return f'''
        from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition

        class {class_name}(InputParametersDefinition):
            task_selectors = ("DOS",)

            @classmethod
            def create_definition(cls):
                from ase2sprkkr.input_parameters.input_parameters import InputParameters
                base = InputParameters.input_parameters_registry["DOS"]
                return base.copy(name="{schema_name}", new_schema={not preset!r})

        definition = {class_name}
    '''


@pytest.mark.smoke
def test_user_ipd_overrides_native_dos_without_loading_other_schemas(tmp_path):
    _run_with_modules(
        tmp_path,
        {
            "mydos.py": _dos_override("MyDOS", "CUSTOM_DOS"),
            "other.py": '''
                from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition

                class OtherDefinition(InputParametersDefinition):
                    task_selectors = ("SOMETHING_ELSE",)

                    @classmethod
                    def create_definition(cls):
                        raise AssertionError("Unrelated user IPD must not be built")

                definition = OtherDefinition
            ''',
        },
        f'''
        from ase2sprkkr.input_parameters.input_parameters import InputParameters
        from ase2sprkkr.input_parameters.input_parameters_definitions import (
            InputParametersDefinition, _task_selector_map,
        )

        registry = InputParameters.input_parameters_registry
        selector_map = _task_selector_map()
        assert not registry._values
        assert 'MYDOS' in registry._modules
        assert 'OTHER' in registry._modules
        assert 'JXC' not in registry._modules
        parsed = InputParametersDefinition.from_file({str(EXAMPLES / "dos.in")!r})
        assert parsed.definition_name == "CUSTOM_DOS"
        assert registry["DOS"].name.upper() == "DOS"
        assert registry["MYDOS"].name.upper() == "CUSTOM_DOS"
        assert set(registry._values) == {{"DOS", "MYDOS"}}
        ''',
    )


@pytest.mark.smoke
def test_two_user_ipds_claiming_same_selector_are_error(tmp_path):
    _run_with_modules(
        tmp_path,
        {
            "dos_custom_a.py": _dos_override("CustomA", "CUSTOM_A"),
            "dos_custom_b.py": _dos_override("CustomB", "CUSTOM_B"),
        },
        f'''
        from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition
        try:
            InputParametersDefinition.from_file({str(EXAMPLES / "dos.in")!r})
        except ValueError as exc:
            message = str(exc)
            assert "Multiple user-defined IPDs" in message, message
            assert "DOS" in message and "CUSTOM_A" in message and "CUSTOM_B" in message
        else:
            raise AssertionError("Two conflicting user overrides were silently accepted")
        ''',
    )


@pytest.mark.smoke
def test_user_preset_does_not_override_native_ipd(tmp_path):
    _run_with_modules(
        tmp_path,
        {"my_preset.py": _dos_override("MyDOSPreset", "MY_PRESET", preset=True)},
        f'''
        from ase2sprkkr.input_parameters.input_parameters import InputParameters
        from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition

        parsed = InputParametersDefinition.from_file({str(EXAMPLES / "dos.in")!r})
        assert parsed.definition_name == "DOS"
        registry = InputParameters.input_parameters_registry
        assert registry["MY_PRESET"].is_derived_definition()
        ''',
    )


@pytest.mark.smoke
def test_legacy_user_factory_can_override_native_ipd(tmp_path):
    _run_with_modules(
        tmp_path,
        {
            "legacydos.py": '''
                from ase2sprkkr.input_parameters.input_parameters import InputParameters

                def input_parameters_definition():
                    result = InputParameters.input_parameters_registry["DOS"].copy(
                        name="LEGACY_DOS", new_schema=True,
                    )
                    result.task_selectors = ("DOS",)
                    return result
            ''',
        },
        f'''
        from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition

        parsed = InputParametersDefinition.from_file({str(EXAMPLES / "dos.in")!r})
        assert parsed.definition_name == "LEGACY_DOS"
        ''',
    )


@pytest.mark.smoke
def test_user_parse_error_never_falls_back_to_native_dos(tmp_path):
    _run_with_modules(
        tmp_path,
        {"mydos.py": _dos_override("MyDOS", "CUSTOM_DOS")},
        f'''
        from ase2sprkkr.input_parameters.input_parameters import InputParameters
        from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition

        selected = InputParameters.input_parameters_registry["MYDOS"]

        def fail(*args, **kwargs):
            raise RuntimeError("user parser failure")

        selected.read_from_file = fail
        try:
            InputParametersDefinition.from_file({str(EXAMPLES / "dos.in")!r})
        except RuntimeError as exc:
            assert str(exc) == "user parser failure"
        else:
            raise AssertionError("Native DOS unexpectedly took over")
        ''',
    )


@pytest.mark.smoke
def test_user_override_of_both_jxc_and_dmi_is_deduplicated(tmp_path):
    _run_with_modules(
        tmp_path,
        {
            "myjxc.py": '''
                from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition

                class UserJXC(InputParametersDefinition):
                    task_selectors = ("JXC", "DMI")

                    @classmethod
                    def create_definition(cls):
                        from ase2sprkkr.input_parameters.input_parameters import InputParameters
                        base = InputParameters.input_parameters_registry["JXC"]
                        return base.copy(new_schema=True, name="CUSTOM_JXC")

                definition = UserJXC
            ''',
        },
        f'''
        from ase2sprkkr.input_parameters.input_parameters_definitions import InputParametersDefinition
        from ase2sprkkr.input_parameters.input_parameters import InputParameters

        parsed = InputParametersDefinition.from_file({str(EXAMPLES / "jxc.in")!r})
        assert parsed.definition_name == "CUSTOM_JXC"
        assert parsed.TASK.DMI()
        assert InputParameters.input_parameters_registry["MYJXC"] is parsed._definition
        ''',
    )
