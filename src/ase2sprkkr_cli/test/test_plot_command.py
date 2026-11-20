import argparse
from functools import cache

import pytest

from ase2sprkkr_cli.commands import plot
from ase2sprkkr.result_files import ResultFile
from ase2sprkkr.result_files.definitions.sfn import SFNResultFile


@cache
def _parser():
    parser = argparse.ArgumentParser()
    plot.parser(parser)
    return parser


def parse_args(*arguments):
    parser = _parser()
    return parser, parser.parse_args(arguments)


def test_sfn_mode_flags_are_documented():
    parser, args = parse_args("file.sfn", "--shape")

    assert args.what == "shape"
    help_text = parser.format_help()
    for mode in plot.SFN_PLOT_MODES:
        assert f"--{mode}" in help_text


def test_sfn_plot_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args("file.sfn", "--shape", "--radial")


def test_sfn_mode_flag_is_rejected_for_non_sfn(monkeypatch):
    class OtherOutput:
        pass

    monkeypatch.setattr(
        ResultFile,
        "from_file",
        staticmethod(lambda *args, **kwargs: OtherOutput()),
    )
    _, args = parse_args("file.dos", "--shape")

    for debug in (True, False):
        with pytest.raises(ValueError, match="only valid for SFN files"):
            plot.run(args, {"debug": debug})


def test_sfn_mode_cannot_be_set_through_generic_option():
    _, args = parse_args("file.sfn", "--mesh", "-S", "what=shape")

    with pytest.raises(ValueError, match="cannot be set using"):
        plot.run(args, {"debug": False})


@pytest.mark.parametrize(
    "sites, expected",
    [
        ("2", 1),
        ("2,4", slice(1, 4)),
    ],
)
def test_sfn_mode_flag_and_deferred_sites_are_passed_to_plot(monkeypatch, sites, expected):
    class FakeSFN(SFNResultFile):
        def __init__(self):
            object.__setattr__(self, "plot_kwargs", None)

        def plot(self, **kwargs):
            object.__setattr__(self, "plot_kwargs", kwargs)

    output = FakeSFN()
    monkeypatch.setattr(
        ResultFile,
        "from_file",
        staticmethod(lambda *args, **kwargs: output),
    )
    _, args = parse_args("file.sfn", "--shape", "--sites", sites)

    assert args.sites == sites  # The parser preserves text for deferred conversion.
    assert plot.run(args, {"debug": True}) is None
    assert output.plot_kwargs["what"] == "shape"
    assert output.plot_kwargs["sites"] == expected



def test_grace_plot_cli_and_dataset_selection(tmp_path, capsys):
    from ase2sprkkr.result_files import GraceFile

    source = tmp_path / "spectrum.agr"
    source.write_text(
        '@with g0\n@s0 legend "peak"\n@target G0.S0\n@type xy\n0 1\n1 2\n&\n'
        '@target G0.S1\n@type xy\n0 3\n1 4\n&\n',
        encoding="utf-8",
    )
    _, args = parse_args(str(source), "-V")
    assert plot.run(args, {"debug": False}) is None
    assert "G0.S0" in capsys.readouterr().out

    png = tmp_path / "selected.png"
    _, args = parse_args(str(source), "--value", "g0.s1", "-o", str(png))
    assert plot.run(args, {"debug": False}) is None
    assert png.exists() and png.stat().st_size > 0
    parsed = GraceFile.from_file(source)
    assert set(parsed.datasets) == {"G0.S0", "G0.S1"}


def test_grace_plot_cli_rejects_unknown_set(tmp_path):
    source = tmp_path / "empty.agr"
    source.write_text("@target G0.S0\n@type xy\n0 1\n&\n")
    _, args = parse_args(str(source), "-v", "G1.S0", "-o", str(tmp_path / "bad.png"))
    with pytest.raises(ValueError, match="Unknown Grace dataset"):
        plot.run(args, {"debug": False})



def test_plot_windows_identify_each_source_file_and_show_together(monkeypatch):
    import matplotlib.pyplot as plt
    from ase2sprkkr import result_files

    plt.close("all")
    rendered = []
    shown = []

    class DummyFile:
        def __init__(self, path):
            self.path = path

        def plot(self, *, show=None, **kwargs):
            assert show is False
            fig = plt.figure()
            rendered.append((self.path, fig.number))
            if self.path.endswith("second.dos"):
                # A split plot already has a per-value window label.
                plt.figure().canvas.manager.set_window_title("SPIN UP")

    monkeypatch.setattr(
        result_files, "read_plottable_file",
        lambda path, **kwargs: DummyFile(path),
    )
    monkeypatch.setattr(
        plt, "show",
        lambda: shown.append([
            (number, plt.figure(number).canvas.manager.get_window_title())
            for number in plt.get_fignums()
        ]),
    )

    try:
        _, args = parse_args("first.dos", "second.dos")
        plot.run(args, {"debug": False})

        assert [path for path, _ in rendered] == ["first.dos", "second.dos"]
        assert len(shown) == 1
        assert [title for _, title in shown[0]] == [
            "first.dos", "second.dos", "second.dos — SPIN UP",
        ]
    finally:
        plt.close("all")


def test_plot_windows_disambiguate_identical_basenames(monkeypatch):
    import matplotlib.pyplot as plt
    from ase2sprkkr import result_files

    plt.close("all")
    titles = []

    class DummyFile:
        def plot(self, *, show=None, **kwargs):
            assert show is False
            plt.figure()

    monkeypatch.setattr(
        result_files, "read_plottable_file",
        lambda path, **kwargs: DummyFile(),
    )
    monkeypatch.setattr(
        plt, "show",
        lambda: titles.extend(
            plt.figure(number).canvas.manager.get_window_title()
            for number in plt.get_fignums()
        ),
    )
    try:
        _, args = parse_args("one/Fe.dos", "two/Fe.dos")
        plot.run(args, {"debug": False})
        assert titles == ["one/Fe.dos", "two/Fe.dos"]
    finally:
        plt.close("all")


def test_plot_output_only_does_not_show_windows(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt
    from ase2sprkkr import result_files

    plt.close("all")
    calls = []

    class DummyFile:
        def plot(self, *, filename=None, show=None, **kwargs):
            calls.append((filename, show))
            plt.figure()

    monkeypatch.setattr(
        result_files, "read_plottable_file",
        lambda path, **kwargs: DummyFile(),
    )
    monkeypatch.setattr(
        plt, "show",
        lambda: pytest.fail("Saving output should not open plot windows"),
    )
    try:
        _, args = parse_args("Fe.dos", "-o", str(tmp_path / "image.png"))
        plot.run(args, {"debug": False})
        assert calls == [(str(tmp_path / "image.png"), False)]
    finally:
        plt.close("all")
