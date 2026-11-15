"""CLI regression tests for optional profile and persistent project runs."""

from argparse import ArgumentParser
from unittest.mock import MagicMock

from ase2sprkkr_cli.commands import run as command


def command_parser():
    parser = ArgumentParser()
    command.parser(parser)
    return parser


def test_profile_and_project_arguments_are_optional():
    args = command_parser().parse_args(["input.pot"])
    assert args.execution_profile is None
    assert args.project_id is None


def test_profile_and_project_flags():
    args = command_parser().parse_args([
        "input.pot", "--profile", "cluster",
        "--project-id", "46ece81e81e34269a607a0434e869a99"
    ])
    assert args.execution_profile == "cluster"
    assert args.project_id == "46ece81e81e34269a607a0434e869a99"


def test_persistent_calculation_passes_project_and_profile(monkeypatch, capsys):
    import ase2sprkkr
    from kkrunner import RunState

    calculator = MagicMock()
    calculator.create_run.return_value.id = "persisted-run"
    calculator.create_run.return_value.state = RunState.FINISHED
    monkeypatch.setattr(ase2sprkkr, "SPRKKR", lambda: calculator)
    monkeypatch.setattr(ase2sprkkr.Potential, "from_file", lambda name: object())
    args = command_parser().parse_args([
        "input.pot", "--profile", "cluster",
        "--project-id", "46ece81e81e34269a607a0434e869a99"
    ])

    command.run(args, None)

    kwargs = calculator.create_run.call_args.kwargs
    assert kwargs["profile"] == "cluster"
    assert kwargs["project"] == args.project_id
    calculator.create_run.return_value.run.assert_called_once_with()
    calculator.calculate.assert_not_called()
    assert "persisted-run" in capsys.readouterr().out


def test_untracked_calculation_uses_calculate(monkeypatch):
    import ase2sprkkr

    calculator = MagicMock()
    monkeypatch.setattr(ase2sprkkr, "SPRKKR", lambda: calculator)
    monkeypatch.setattr(ase2sprkkr.Potential, "from_file", lambda name: object())

    command.run(command_parser().parse_args(["input.pot"]), None)

    calculator.calculate.assert_called_once()
    assert calculator.calculate.call_args.kwargs["profile"] is None
    calculator.create_run.assert_not_called()



def test_execution_profile_does_not_enable_global_python_profiler(monkeypatch):
    from ase2sprkkr_cli import main

    captured = []

    def record(args, global_args):
        captured.append((args.execution_profile, global_args["profile"]))

    monkeypatch.setattr(command, "run", record)
    main.run(["run", "input.pot", "--profile", "cluster"])
    assert captured == [("cluster", False)]
