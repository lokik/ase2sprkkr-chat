"""Grace parser regressions, including layouts emitted by SPR-KKR."""

import io

import numpy as np
import pytest

from ase2sprkkr.result_files.grace import GraceFile, GraceParseError, read_agr


def test_sprkkr_multiple_graphs_and_style():
    source = r'''# Grace project file
@version 50005
@page size 600, 600
@g0 on
@with g0
@  world xmin -0.50000E+01
@  world xmax 0.50000E+02
@  world ymin -0.60000E-07
@  world ymax 0.39940E-04
@  xaxis label "energy (eV)"
@  yaxis label "\xD\xm\f{}\m{1}\SCu\M{1}\N (Mb)"
@g1 on
@with g1
@  view xmin 0.15
@  view xmax 0.85
@  view ymin 0.50
@  view ymax 0.85
@  title "L\s2,3\N-XAS spectra of Cu"
@  s0 legend "XAS spin up"
@target G1.S0
@type xy
 -0.439040E+01  0.714835E-01
 -0.439040E+01  0.771264E-01
&
@target G0.S0
@type xy
 1.0D+00 2.0D+00
&
'''
    obj = GraceFile.from_string(source)
    assert set(obj.datasets) == {"G0.S0", "G1.S0"}
    assert obj.settings["version"] == "50005"
    assert obj.settings["page size"] == "600, 600"
    assert obj.graphs[0].world == (-5.0, 50.0, -6e-8, 3.994e-5)
    assert obj.graphs[0].x_label == "energy (eV)"
    assert obj.graphs[0].y_label == r"\xD\xm\f{}\m{1}\SCu\M{1}\N (Mb)"
    assert obj.graphs[1].view == (0.15, 0.85, 0.5, 0.85)
    assert obj.graphs[1].title == r"L\s2,3\N-XAS spectra of Cu"
    assert obj.graphs[1].datasets[0].legend == "XAS spin up"
    np.testing.assert_allclose(obj.data["G1.S0"], [[-4.3904, 0.0714835], [-4.3904, 0.0771264]])
    np.testing.assert_allclose(obj.data["G0.S0"], [[1., 2.]])
    assert obj.datasets["G1.S0"].columns == ("x", "y")
    assert "target G1.S0" in obj.commands


def test_vmd_style_without_target_and_open_stream(tmp_path):
    source = '''# comment
@s0 legend "Total Energy"
@type xy
0 1
1 2
&
@type xy
0 3
1 4
'''
    obj = read_agr(io.StringIO(source))
    assert list(obj.datasets) == ["G0.S0", "G0.S1"]
    assert obj.datasets["G0.S0"].legend == "Total Energy"
    filename = tmp_path / "example.agr"
    filename.write_text(source, encoding="utf-8")
    assert read_agr(filename).data["G0.S1"].shape == (2, 2)


@pytest.mark.parametrize("kind,values", [
    ("xydy", "1 2 0.1"),
    ("xydxdy", "1 2 0.3 0.4"),
    ("xydxdxdydy", "1 2 0.1 0.2 0.3 0.4"),
    ("xyz", "1 2 3"),
    ("xyvmap", "1 2 3 4"),
    ("bardydy", "1 2 3 4"),
])
def test_dataset_types(kind, values):
    obj = GraceFile.from_string(f"@target G2.S5\n@type {kind}\n{values}\n&\n")
    dataset = obj.datasets["G2.S5"]
    assert dataset.type == kind
    assert dataset.data.shape == (1, len(values.split()))
    np.testing.assert_allclose(dataset.data[0], list(map(float, values.split())))


def test_interleaved_graph_and_dataset_settings():
    obj = GraceFile.from_string('''@g2 on
@with g2
@s3 legend "spin down"
@s3 line color 4
@target G2.S3
@type xy
0 0
&
@G2.S3 line linewidth 3
''')
    assert obj.datasets["G2.S3"].legend == "spin down"
    assert obj.datasets["G2.S3"].settings["line color"] == "4"
    assert obj.datasets["G2.S3"].settings["line linewidth"] == "3"
    assert obj.graphs[2].settings["enabled"] == "on"


def test_empty_block_and_missing_terminal_ampersand():
    obj = GraceFile.from_string("@target G0.S0\n@type xy\n&\n@target G0.S1\n@type xydy\n1 2 3\n")
    assert obj.datasets["G0.S0"].data.shape == (0, 2)
    assert obj.datasets["G0.S1"].data.shape == (1, 3)


@pytest.mark.parametrize("text,fragment", [
    ("0 1\n", "without @target or @type"),
    ("@type xynotreal\n0 1\n&\n", "unsupported Grace dataset type"),
    ("@target G0.S0\n@type xy\n0 1 2\n&\n", "requires 2 columns"),
    ("@target G0.S0\n@type xy\n0 bogus\n&\n", "invalid numeric"),
    ("@target G0.S0\n@type xy\n0 1\n@target G0.S1\n", "missing '&'"),
    ("@target G0.S0\n@type xy\n0 1\n&\n@target G0.S0\n", "duplicate dataset"),
    ("&", "unexpected '&'"),
])
def test_invalid_input_has_line_number(text, fragment):
    with pytest.raises(GraceParseError) as exc:
        GraceFile.from_string(text)
    assert fragment in str(exc.value)
    assert "Line " in str(exc.value)


@pytest.mark.parametrize("filename", ["xas_Cu_L23_x.agr", "xas_Cu_L23_sumrule.agr"])
def test_real_sprkkr_xas_reference_files(filename):
    from pathlib import Path

    path = Path(__file__).with_name(filename)
    result = read_agr(path)
    assert set(result.datasets) == {"G0.S0", "G1.S0"}
    assert result.graphs[0].x_label == "energy (eV)"
    assert "Cu" in result.graphs[1].title
    for values in result.data.values():
        assert values.ndim == 2 and values.shape[1] == 2
        assert len(values) > 10
        assert np.isfinite(values).all()
