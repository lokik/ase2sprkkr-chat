from dataclasses import FrozenInstanceError

import pytest

from ase2sprkkr.sprkkr.executables import executables as public_executables
from ...common.module_registry import RegistryDefinitionError, RegistryEntryNotFound
from ..executables import ExecutableDefinition, _ExecutableRegistry, executables, resolve_task_executable


def test_executable_definition_normalizes_reader_name_only():
    definition = ExecutableDefinition(
        "CustomExecutable",
        default_output_reader=" spec ",
        description="Custom executable",
    )

    assert definition.name == "CustomExecutable"
    assert definition.default_output_reader == "SPEC"
    assert definition.description == "Custom executable"


def test_executable_definition_keeps_executable_name_exact():
    definition = ExecutableDefinition(" custom ")

    assert definition.name == " custom "


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"name": None}, TypeError),
        ({"name": "   "}, ValueError),
        ({"name": "custom", "default_output_reader": None}, TypeError),
        ({"name": "custom", "default_output_reader": "   "}, ValueError),
        ({"name": "custom", "description": None}, TypeError),
    ),
)
def test_executable_definition_rejects_invalid_metadata(kwargs, error):
    with pytest.raises(error):
        ExecutableDefinition(**kwargs)


def test_executable_registry_accepts_custom_definitions_at_construction():
    definition = ExecutableDefinition("custom", default_output_reader="SPEC")
    registry = _ExecutableRegistry((definition,))

    assert registry["custom"] is definition
    assert not hasattr(registry, "register")


def test_executable_registry_rejects_non_definition_items():
    with pytest.raises(TypeError, match="ExecutableDefinition"):
        _ExecutableRegistry(("custom",))


def test_executable_registry_rejects_duplicate_definitions():
    with pytest.raises(RegistryDefinitionError, match="Duplicate executable"):
        _ExecutableRegistry(
            (ExecutableDefinition("custom"), ExecutableDefinition("custom"))
        )


def test_executable_registry_reports_unknown_executable():
    with pytest.raises(RegistryEntryNotFound, match="unknown"):
        executables["unknown"]


def test_builtin_executable_reader_defaults():
    assert executables["kkrscf"].default_output_reader == "DEFAULT"
    assert executables["kkrgen"].default_output_reader == "DEFAULT"
    assert executables["kkrspec"].default_output_reader == "SPEC"
    assert executables["kkrchi"].default_output_reader == "DEFAULT"


def test_executables_is_public_read_only_metadata():
    assert public_executables is executables
    assert not hasattr(public_executables, "register")

    with pytest.raises(TypeError):
        public_executables["custom"] = ExecutableDefinition("custom")

    with pytest.raises(FrozenInstanceError):
        public_executables["kkrscf"].description = "Changed"


def test_resolve_native_executable_from_input_parameters():
    from types import SimpleNamespace

    def params(mode):
        return SimpleNamespace(CONTROL=SimpleNamespace(CLUTYPE=lambda: mode))

    for name, cluster_name in (("kkrscf", "embscf"), ("kkrgen", "embgen")):
        definition = executables[name]
        assert definition.resolve(params(None)) == name
        assert definition.resolve(params("embedded")) == cluster_name

    assert executables["kkrspec"].resolve(params(None)) == "kkrspec"
    with pytest.raises(ValueError, match="does not support embedded"):
        executables["kkrspec"].resolve(params("embedded"))
    with pytest.raises(ValueError, match="Unsupported SPR-KKR CLUTYPE"):
        executables["kkrscf"].resolve(params("other"))

    legacy = SimpleNamespace(CONTROL=SimpleNamespace())
    assert executables["kkrscf"].resolve(legacy) == "kkrscf"


@pytest.mark.parametrize("variant, error", [(False, TypeError), (" ", ValueError)])
def test_invalid_cluster_variant_metadata(variant, error):
    with pytest.raises(error):
        ExecutableDefinition("custom", cluster_variant=variant)


def test_unknown_custom_executable_still_works_outside_embedding():
    from types import SimpleNamespace

    normal = SimpleNamespace(CONTROL=SimpleNamespace(CLUTYPE=lambda: None))
    embedded = SimpleNamespace(CONTROL=SimpleNamespace(CLUTYPE=lambda: "embedded"))
    assert executables.resolve("my-custom-binary", normal) == "my-custom-binary"
    with pytest.raises(ValueError, match="does not support embedded"):
        executables.resolve("my-custom-binary", embedded)


@pytest.mark.parametrize(
    ("task_name", "native_name"),
    (
        ("SCF", "embscf"),
        ("DOS", "embgen"),
        ("JXC", "embgen"),
        ("DMI", "embgen"),
    ),
)
def test_only_explicitly_enabled_tasks_resolve_embedded_executable(task_name, native_name):
    from ase2sprkkr.input_parameters.input_parameters import InputParameters

    parameters = InputParameters.create(task_name)
    parameters.CONTROL.CLUTYPE = "embedded"
    assert "embedded" in parameters.task_definition.supports
    assert resolve_task_executable(parameters.task_definition, parameters) == native_name


@pytest.mark.parametrize(
    "task_name",
    (
        "PHAGEN", "XAS", "GILBERT", "ARPES", "TORQUE",
        # These tasks must not inherit embedding from their executable:
        # POSANITAU has only conditional native support, while STONER is
        # confirmed broken (SUSC_GAMMA processes host atom types).
        "APS", "COMPTON", "POSANI", "POSANIPREP", "POSANITAU", "STONER", "VBPES",
    ),
)
def test_non_enabled_tasks_do_not_inherit_embedding_from_executable(task_name):
    from ase2sprkkr.input_parameters.input_parameters import InputParameters

    parameters = InputParameters.create(task_name)
    parameters.CONTROL.CLUTYPE = "embedded"
    assert "embedded" not in parameters.task_definition.supports
    with pytest.raises(ValueError, match="not enabled for embedded-cluster"):
        resolve_task_executable(parameters.task_definition, parameters)


def test_normal_task_resolution_does_not_use_embedding_gate():
    from ase2sprkkr.input_parameters.input_parameters import InputParameters

    parameters = InputParameters.create("TORQUE")
    assert "embedded" not in parameters.task_definition.supports
    assert resolve_task_executable(parameters.task_definition, parameters) == "kkrgen"


@pytest.mark.parametrize("source", ["potential", "path", "input"])
def test_create_run_rejects_unsupported_2d_before_opening_files(source, tmp_path):
    from pathlib import Path
    from ase2sprkkr import InputParameters, Potential, SPRKKR

    source_path = Path(__file__).parents[2] / "potentials" / "examples" / "GeTe.pot"
    parameters = InputParameters.for_task("CHILANDAU")
    if source == "potential":
        potential = Potential.from_file(source_path)
    elif source == "path":
        potential = source_path
    else:
        parameters.CONTROL.POTFIL = str(source_path)
        potential = False

    existing_input = tmp_path / "run.inp"
    existing_input.write_text("keep this file\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CHILANDAU.*does not support 2D"):
        SPRKKR().create_run(
            potential=potential, input_parameters=parameters, directory=tmp_path,
            input_file="run.inp", output_file="run.out", empty_spheres=False,
        )
    assert existing_input.read_text(encoding="utf-8") == "keep this file\n"
    assert not (tmp_path / "run.out").exists()


def test_file_only_potential_uses_explicit_atoms_for_task_validation(tmp_path):
    from ase import Atoms
    from ase2sprkkr import SPRKKR

    params = InputParameters.for_task("DOS")
    path = tmp_path / "native.pot"
    path.write_text("native file is intentionally not parsed")
    params.CONTROL.POTFIL = str(path)

    with pytest.raises(ValueError, match="free clusters"):
        SPRKKR().prepare(
            atoms=Atoms("Fe", cell=[2, 2, 2], pbc=False),
            potential=False, input_parameters=params,
            input_file="dos.inp", directory=tmp_path, empty_spheres=False,
        )
    assert not (tmp_path / "dos.inp").exists()
