import importlib
import sys

import pytest

from ...common.module_registry import RegistryDefinitionError
from ..reader_registry import OutputReaderRegistry, reader_registry
from ..readers.default import DefaultOutputReader


def _reader_package(tmp_path, monkeypatch, name, modules):
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    for module_name, source in modules.items():
        (root / f"{module_name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(name)


def test_missing_task_reader_uses_default_reader():
    assert (
        reader_registry.class_for_task_name("definitely_missing_reader")
        is DefaultOutputReader
    )


def test_reader_registry_loads_expected_reader_class():
    reader = reader_registry["SCF"]

    assert reader.__name__ == "ScfOutputReader"
    assert reader.__module__ == "ase2sprkkr.outputs.readers.scf"


def test_reader_registry_detection_stays_lazy_until_class_lookup(
    tmp_path, monkeypatch,
):
    package = _reader_package(
        tmp_path,
        monkeypatch,
        "_ase2sprkkr_reader_registry_lazy",
        {
            "alpha": (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader\n"
                "class AlphaOutputReader(KkrOutputReader):\n"
                "    pass\n"
            )
        },
    )
    registry = OutputReaderRegistry(package)
    module_name = f"{package.__name__}.alpha"
    output = tmp_path / "explicit.out"
    output.write_bytes(b" header\n TASK = ALPHA\n")

    assert "ALPHA" in registry
    assert module_name not in sys.modules

    # Reading the explicit task selector must not import the reader module.
    assert registry.detect(output) == "ALPHA"
    assert module_name not in sys.modules

    reader = registry["ALPHA"]
    assert module_name in sys.modules
    assert reader.__name__ == "AlphaOutputReader"


def test_legacy_output_detection_is_owned_by_reader(tmp_path, monkeypatch):
    package = _reader_package(
        tmp_path,
        monkeypatch,
        "_ase2sprkkr_reader_registry_detect_legacy",
        {
            "alpha": (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader\n"
                "class AlphaOutputReader(KkrOutputReader):\n"
                "    @classmethod\n"
                "    def matches_output(cls, output):\n"
                "        return b'legacy alpha' in output\n"
            ),
            "beta": (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader\n"
                "class BetaOutputReader(KkrOutputReader):\n"
                "    pass\n"
            ),
        },
    )
    registry = OutputReaderRegistry(package)
    output = tmp_path / "legacy.out"
    output.write_bytes(b" TASK = NONE\nlegacy alpha\n")

    assert registry.detect(output) == "ALPHA"


def test_ambiguous_legacy_output_detection_is_rejected(tmp_path, monkeypatch):
    package = _reader_package(
        tmp_path,
        monkeypatch,
        "_ase2sprkkr_reader_registry_detect_ambiguous",
        {
            name: (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader\n"
                f"class {name.title()}OutputReader(KkrOutputReader):\n"
                "    @classmethod\n"
                "    def matches_output(cls, output):\n"
                "        return True\n"
            )
            for name in ("alpha", "beta")
        },
    )
    registry = OutputReaderRegistry(package)
    output = tmp_path / "ambiguous.out"
    output.write_bytes(b"legacy output\n")

    with pytest.raises(RegistryDefinitionError, match="multiple task-specific readers"):
        registry.detect(output)


def test_possible_legacy_match_yields_to_definite_match(tmp_path, monkeypatch):
    package = _reader_package(
        tmp_path, monkeypatch, "_ase2sprkkr_reader_confidence",
        {
            "alpha": (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader, OutputMatch\n"
                "class AlphaOutputReader(KkrOutputReader):\n"
                "    @classmethod\n"
                "    def matches_output(cls, output):\n"
                "        return OutputMatch.MAYBE if b'weak' in output else False\n"
            ),
            "beta": (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader\n"
                "class BetaOutputReader(KkrOutputReader):\n"
                "    @classmethod\n"
                "    def matches_output(cls, output):\n"
                "        return b'strong' in output\n"
            ),
        },
    )
    registry = OutputReaderRegistry(package)

    output = tmp_path / "legacy.out"
    output.write_bytes(b"TASK = NONE\nweak\n")
    assert registry.detect(output) == "ALPHA"

    output.write_bytes(b"TASK = NONE\nweak\nstrong\n")
    assert registry.detect(output) == "BETA"

    output.write_bytes(b"TASK = BETA\nweak\n")
    assert registry.detect(output) == "BETA"


def test_ambiguous_possible_legacy_matches_are_rejected(tmp_path, monkeypatch):
    package = _reader_package(
        tmp_path, monkeypatch, "_ase2sprkkr_reader_weak_ambiguity",
        {
            name: (
                "from ase2sprkkr.outputs.task_result import KkrOutputReader, OutputMatch\n"
                f"class {name.title()}OutputReader(KkrOutputReader):\n"
                "    @classmethod\n"
                "    def matches_output(cls, output):\n"
                "        return OutputMatch.MAYBE\n"
            )
            for name in ("alpha", "beta")
        },
    )
    registry = OutputReaderRegistry(package)
    output = tmp_path / "legacy.out"
    output.write_bytes(b"TASK = NONE\n")

    with pytest.raises(RegistryDefinitionError, match="multiple task-specific readers"):
        registry.detect(output)


def test_tentative_dos_can_be_overridden_by_matching_native_inp(tmp_path):
    """Reading a reliable sibling INP is stronger than DOS output artifacts."""
    from ase2sprkkr import InputParameters

    output = tmp_path / "Fe2_SPLEED.out"
    inp = tmp_path / "Fe2_SPLEED.inp"
    inp.write_text(InputParameters.create("SPLEED").to_string(validate=False))
    output.write_bytes(
        b"TASK = NONE\nFILES:\n\n potential : Fe2.pot\n"
        b" DOS : Fe2_SPLEED_SPLEED.dos\n\n"
    )

    assert reader_registry.detect(output) == "SPLEED"

    # A definite native fingerprint wins even if the paired INP disagrees.
    output.write_bytes(
        b"TASK = NONE\nFILES:\n\n DOS : x.dos\n\n"
        b"Components of Dzyaloshinski-Moriya vector Dij\n"
    )
    assert reader_registry.detect(output) == "JXC"
    output.write_bytes(
        b"TASK = DOS\nFILES:\n\n DOS : x.dos\n\n"
    )
    assert reader_registry.detect(output) == "DOS"


def test_only_explicitly_task_marked_matching_input_is_considered(tmp_path):
    from ase2sprkkr import InputParameters

    output = tmp_path / "run.out"
    output.write_bytes(b"TASK = NONE\nFILES:\n\n DOS : x.dos\n\n")
    unrelated = tmp_path / "other.inp"
    unrelated.write_text(InputParameters.create("SPLEED").to_string(validate=False))
    assert reader_registry.detect(output) == "DOS"

    # Invalid or task-less INP cannot silently become an SCF match.
    matched = tmp_path / "run.inp"
    matched.write_text("CONTROL\n  DATASET = example\n")
    assert reader_registry.detect(output) == "DOS"
    matched.write_text("not a valid SPR-KKR input")
    assert reader_registry.detect(output) == "DOS"

    # Both suffixes existing is ambiguous even when one looks plausible.
    matched.write_text(InputParameters.create("SPLEED").to_string(validate=False))
    (tmp_path / "run.in").write_text(InputParameters.create("SCF").to_string(validate=False))
    assert reader_registry.detect(output) == "DOS"


def test_task_from_paired_input_when_stdout_has_no_fingerprint(tmp_path):
    from ase2sprkkr import InputParameters

    output = tmp_path / "spectroscopy.out"
    inp = tmp_path / "spectroscopy.inp"
    inp.write_text(InputParameters.create("SPLEED").to_string(validate=False))
    output.write_bytes(b"TASK = NONE\nunrecognized legacy output\n")
    assert reader_registry.detect(output) == "SPLEED"


def test_builtin_legacy_output_fingerprints(tmp_path):
    jxc = tmp_path / "jxc.out"
    jxc.write_bytes(
        b" TASK = NONE\nComponents of Dzyaloshinski-Moriya vector Dij\n"
    )
    assert reader_registry.detect(jxc) == "JXC"

    dos = tmp_path / "dos.out"
    dos.write_bytes(b" TASK = NONE\nFILES:\n\n  DOS: dataset_DOS.dos\n")
    assert reader_registry.detect(dos) == "DOS"

    # Native SPLEED also lists a DOS artifact in FILES, but its task-specific
    # section is decisive even when the generic TASK marker says NONE.
    spleed = tmp_path / "spleed.out"
    spleed.write_bytes(
        b" * KKRSPEC VERSION 9.7 *\nTASK = NONE\n"
        b"FILES:\n potential : Fe2.pot\n DOS : Fe2_SPLEED_SPLEED.dos\n\n"
        b"********************\nSpin polarised LEED\n********************\n"
    )
    assert reader_registry.detect(spleed) == "SPLEED"

    unrelated = tmp_path / "unrelated.out"
    unrelated.write_bytes(b"DOS: not part of a FILES section\n")
    assert reader_registry.detect(unrelated) is None


@pytest.mark.parametrize(
    ("case", "module_source", "diagnostic"),
    (
        ("missing", "class SomethingElse:\n    pass\n", "BrokenOutputReader"),
        ("wrong_type", "BrokenOutputReader = object()\n", "KkrOutputReader subclass"),
    ),
)
def test_reader_registry_rejects_invalid_module_exports(
    tmp_path, monkeypatch, case, module_source, diagnostic,
):
    package = _reader_package(
        tmp_path,
        monkeypatch,
        f"_ase2sprkkr_reader_registry_invalid_{case}",
        {"broken": module_source},
    )
    registry = OutputReaderRegistry(package)

    with pytest.raises(RegistryDefinitionError, match=diagnostic):
        registry.class_for_task_name("BROKEN")


def test_reader_dependency_errors_remain_visible(tmp_path, monkeypatch):
    missing_dependency = "_ase2sprkkr_missing_reader_dependency"
    package = _reader_package(
        tmp_path,
        monkeypatch,
        "_ase2sprkkr_reader_registry_dependency",
        {"broken": f"import {missing_dependency}\n"},
    )
    registry = OutputReaderRegistry(package)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        registry.class_for_task_name("BROKEN")

    assert exc_info.value.name == missing_dependency


@pytest.mark.parametrize("task", ("PLOT2DPOT", "PLOT2DRHO", "PLOTSFN", "SCF"))
def test_explicit_output_detection_accepts_native_numeric_selectors(tmp_path, task):
    """TASK identifiers may contain digits; output detection must retain them."""
    output = tmp_path / "native.out"
    output.write_bytes(f" header\n TASK = {task}\n".encode("ascii"))
    assert reader_registry.detect(output) == task
