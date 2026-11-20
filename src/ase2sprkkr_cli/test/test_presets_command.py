"""CLI parser and workflow tests for YAML presets."""
import argparse

import pytest

from ase2sprkkr import InputParameters
from ase2sprkkr_cli.commands import presets


def _args(*args):
    parser = argparse.ArgumentParser()
    presets.parser(parser)
    return parser.parse_args(args)


def test_preset_cli_roundtrip(tmp_path, monkeypatch, capsys):
    import ase2sprkkr.input_parameters.presets as module
    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    source = InputParameters.for_task("DOS")
    source.ENERGY.NE.set(417)
    input_file = tmp_path / "source.inp"
    source.save_to_file(input_file, validate=False)

    presets.run(_args("create", "dos.my_mesh", str(input_file)), None)
    assert (tmp_path / "presets" / "input_parameters" / "dos" / "my_mesh.yaml").is_file()

    presets.run(_args("--list", "dos"), None)
    assert "my_mesh" in capsys.readouterr().out
    presets.run(_args("list"), None)
    assert "input_parameters:dos.my_mesh" in capsys.readouterr().out
    presets.run(_args("show", "my_mesh"), None)
    assert "417" in capsys.readouterr().out
    presets.run(_args("delete", "dos.my_mesh"), None)
    assert not (tmp_path / "presets" / "input_parameters" / "dos" / "my_mesh.yaml").exists()


def test_cli_create_from_yaml(tmp_path, monkeypatch, capsys):
    import ase2sprkkr.input_parameters.presets as module

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    source = tmp_path / "symmetry.yaml"
    source.write_text("version: 1\nparameters: {CONTROL: {NOSYM: false}}\n")
    presets.run(_args("create", "dos.symmetry", str(source)), None)
    target = InputParameters.for_task("DOS")
    target.CONTROL.NOSYM = True
    target.load_preset("symmetry", clear_first=False)
    assert target.CONTROL.NOSYM() is False
    capsys.readouterr()
    presets.run(_args("show", "symmetry"), None)
    assert "NOSYM: false" in capsys.readouterr().out


def test_run_parser_accepts_preset():
    from ase2sprkkr_cli.commands import run
    parser = argparse.ArgumentParser()
    run.parser(parser)
    args = parser.parse_args(["sample.pot", "--task", "dos", "--preset", "my_mesh"])
    assert args.preset == "my_mesh"


def test_run_preset_is_passed_before_cli_options(tmp_path, monkeypatch):
    import ase2sprkkr as package
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.commands import run

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    preset = InputParameters.for_task("DOS")
    preset.ENERGY.NE.set(415)
    preset.save_preset("dos.fine")

    observed = {}
    class FakeCalculator:
        def __init__(self, **kwargs):
            self._input_parameters = InputParameters.for_task("SCF")

        @property
        def input_parameters(self):
            return self._input_parameters

        @input_parameters.setter
        def input_parameters(self, value):
            self._input_parameters = InputParameters.create(value)

        def calculate(self, *, potential, input_parameters, options, **kwargs):
            observed["input_parameters"] = input_parameters
            observed["options"] = options
            observed["potential"] = potential
            return "success"

    sentinel = object()
    monkeypatch.setattr(package, "SPRKKR", FakeCalculator)
    monkeypatch.setattr(package.Potential, "from_file",
                        lambda *_args, **_kwargs: sentinel)

    parser = argparse.ArgumentParser()
    run.parser(parser)
    # The CLI dispatcher uses parse_known_args and appends its remainder to
    # run.options: argparse.parse_args alone cannot handle positional options
    # following optional flags in this order.
    args, unknown = parser.parse_known_args([
        "sample.pot", "--task", "dos", "--preset", "fine", "NE=521",
    ])
    args.options.extend(unknown)
    assert run.run(args, None) == "success"
    assert observed["potential"] is sentinel
    assert int(observed["input_parameters"].ENERGY.NE()[0]) == 415
    assert observed["options"] == {"NE": 521}


def test_run_missing_preset_is_reported_as_cli_usage_error(tmp_path, monkeypatch):
    import ase2sprkkr as package
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.commands import run
    from ase2sprkkr_cli.errors import CLIUsageError

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    monkeypatch.setattr(package.Potential, "from_file",
                        lambda *_args, **_kwargs: object())
    parser = argparse.ArgumentParser()
    run.parser(parser)
    args = parser.parse_args(["sample.pot", "--task", "dos", "--preset", "missing"])
    with pytest.raises(CLIUsageError, match="Cannot load preset"):
        run.run(args, None)


def test_create_invalid_source_is_cli_usage_error(tmp_path, monkeypatch):
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.errors import CLIUsageError

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    bad = tmp_path / "bad.inp"
    bad.write_text("ENERGY\n\tNE=NOT_A_NUMBER\n")
    with pytest.raises(CLIUsageError, match="Preset .* failed"):
        presets.run(_args("create", "dos.bad", str(bad)), None)


def test_missing_show_is_cli_usage_error(tmp_path, monkeypatch):
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.errors import CLIUsageError

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    with pytest.raises(CLIUsageError, match="Preset .* failed"):
        presets.run(_args("show", "dos.no_such_preset"), None)



def test_cli_task_scoped_create_and_list(tmp_path, monkeypatch, capsys):
    import ase2sprkkr.input_parameters.presets as module

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    source = InputParameters.for_task("BSFKK")
    source.ENERGY.NE.set([1])
    native = tmp_path / "kk.inp"
    source.save_to_file(native, validate=False)
    presets.run(_args("create", "task:bsfkk.fermi", str(native)), None)
    destination = tmp_path / "presets" / "task" / "bsfkk" / "fermi.yaml"
    assert destination.is_file()

    presets.run(_args("list", "input_parameters:bsf"), None)
    assert "task:bsfkk.fermi" in capsys.readouterr().out
    presets.run(_args("list", "task:bsfek"), None)
    assert "task:bsfkk.fermi" not in capsys.readouterr().out


def test_run_without_task_creates_task_specific_input(tmp_path, monkeypatch):
    import ase2sprkkr as package
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.commands import run

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    source = InputParameters.for_task("BSFKK")
    source.ENERGY.NE.set([1])
    source.save_preset("task:bsfkk.fermi")

    observed = {}
    class FakeCalculator:
        def __init__(self, **kwargs):
            self._input_parameters = InputParameters.for_task("SCF")

        @property
        def input_parameters(self):
            return self._input_parameters

        @input_parameters.setter
        def input_parameters(self, value):
            self._input_parameters = InputParameters.create(value)

        def calculate(self, *, input_parameters, **kwargs):
            observed["input"] = input_parameters
            return "ok"

    monkeypatch.setattr(package, "SPRKKR", FakeCalculator)
    monkeypatch.setattr(package.Potential, "from_file",
                        lambda *_args, **_kwargs: object())
    parser = argparse.ArgumentParser()
    run.parser(parser)
    args = parser.parse_args([
        "sample.pot", "--preset", "task:bsfkk.fermi",
    ])
    assert run.run(args, None) == "ok"
    assert observed["input"].task_name == "BSFKK"
    assert observed["input"].requested_task_name == "BSFKK"


def test_cli_explicit_task_conflict_with_task_preset(tmp_path, monkeypatch):
    import ase2sprkkr as package
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.commands import run
    from ase2sprkkr_cli.errors import CLIUsageError

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    InputParameters.for_task("BSFKK").save_preset("task:bsfkk.fermi")
    monkeypatch.setattr(package.Potential, "from_file",
                        lambda *_args, **_kwargs: object())
    parser = argparse.ArgumentParser()
    run.parser(parser)
    args = parser.parse_args([
        "sample.pot", "--task", "BSFEK", "--preset", "task:bsfkk.fermi",
    ])
    with pytest.raises(CLIUsageError, match="explicitly requested"):
        run.run(args, None)



def test_run_requires_task_for_task_neutral_shared_schema(tmp_path, monkeypatch):
    import ase2sprkkr as package
    import ase2sprkkr.input_parameters.presets as module
    from ase2sprkkr_cli.commands import run
    from ase2sprkkr_cli.errors import CLIUsageError

    monkeypatch.setattr(module, "user_preferences_directory", lambda: str(tmp_path))
    common = InputParameters.for_task("BSFEK")
    common.TAU.NKTAB.set(523)
    common.save_preset("input_parameters:bsf.common")
    monkeypatch.setattr(
        package.Potential, "from_file",
        lambda *_args, **_kwargs: object(),
    )
    parser = argparse.ArgumentParser()
    run.parser(parser)
    args = parser.parse_args([
        "sample.pot", "--preset", "input_parameters:bsf.common",
    ])
    with pytest.raises(CLIUsageError, match="select --task"):
        run.run(args, None)
