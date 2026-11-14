"""Matplotlib rendering of Grace projects, including real SPR-KKR XAS files."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from ase2sprkkr.result_files import GraceFile, read_agr
from ase2sprkkr.result_files.grace_plot import grace_text


def test_grace_text_formatting():
    assert grace_text(r"L\s2,3\N-XAS") == "L$_{2,3}$-XAS"
    assert grace_text(r"\xD\xm\f{}") == "Δμ"


@pytest.mark.parametrize("filename", ["xas_Cu_L23_x.agr", "xas_Cu_L23_sumrule.agr"])
def test_real_xas_plot_preserves_panels(tmp_path, filename):
    from pathlib import Path

    source = Path(__file__).with_name(filename)
    output = tmp_path / (source.stem + ".png")
    document = read_agr(source)
    fig = document.plot(filename=output, dpi=75)
    try:
        assert output.stat().st_size > 0
        assert len(fig.axes) == 2
        assert all(len(axis.lines) == 1 for axis in fig.axes)
        assert fig.axes[0].get_xlabel() == "energy (eV)"
        assert fig.axes[0].get_position().bounds[1] == pytest.approx(0.15)
        assert fig.axes[1].get_position().bounds[1] == pytest.approx(0.50)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("kind,values", [
    ("xy", "0 1"),
    ("xydx", "0 1 .2"),
    ("xydy", "0 1 .2"),
    ("xydxdx", "0 1 .2 .3"),
    ("xydydy", "0 1 .2 .3"),
    ("xydxdy", "0 1 .2 .3"),
    ("xydxdxdydy", "0 1 .2 .3 .4 .5"),
    ("xyz", "0 1 3"),
    ("xycolor", "0 1 3"),
    ("xysize", "0 1 15"),
    ("xyvmap", "0 1 2 3"),
    ("bar", "0 1"),
    ("bardy", "0 1 .2"),
    ("bardydy", "0 1 .2 .3"),
])
def test_dataset_plot_types(kind, values):
    document = GraceFile.from_string(f"@target G0.S0\n@type {kind}\n{values}\n&\n")
    fig, ax = plt.subplots()
    try:
        assert document.graphs[0].datasets[0].plot(ax) is not None
    finally:
        plt.close(fig)


def test_plot_dataset_filter_and_errors(tmp_path):
    document = GraceFile.from_string(
        '@target G0.S0\n@type xy\n0 1\n&\n'
        '@target G0.S1\n@type xy\n0 2\n&\n'
        '@target G1.S0\n@type xy\n0 3\n&\n'
    )
    fig = document.plot(datasets=["G0.S1"], filename=tmp_path / "one.png")
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].lines) == 1
        assert fig.axes[0].lines[0].get_ydata()[0] == 2
        assert (tmp_path / "one.png").exists()
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="Unknown Grace dataset"):
        document.plot(datasets=["G6.S9"])
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="one selected Grace graph"):
            document.plot(axis=ax)
    finally:
        plt.close(fig)
