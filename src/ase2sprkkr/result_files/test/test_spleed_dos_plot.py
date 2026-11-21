"""Regression cases for SPR-KKR SPLEED output and header-only DOS files.

The record formats and header reproduce the Fe2 SPLEED files supplied for
this regression (NE=200, NT=NP=1, seven SPC columns, empty DOS payload).
"""

import numpy as np
import pytest

from .. import ResultFile
from ..definitions.spleed_spc import SPLEEDSPCResultFile


def native_header(keyword):
    return (
        f"KEYWORD   {keyword}\n"
        "TITLE     SPR-KKR-ASA calculation for System:\n"
        "SYSTEM    System:\n"
        "NQ_eff             2\n"
        "NT_eff             2\n"
        "          \n"
        "NE               200\n"
        "IREL               3\n"
        "EFERMI       0.73544\n"
        "INFO      SPR-KKR-ASA\n"
        "          \n"
        "   IQ  NLQ \n"
        "    1    4\n"
        "    2    4\n"
        "   IT       TXT_T      CONC  NAT IQAT\n"
        "    1 Fe_1             1.00000    1    1\n"
        "    2 Fe_2             1.00000    1    2\n"
    )


def spleed_content():
    # Use the supplied first native data row, and 199 records of identical
    # format to exercise the full 200-energy-point output.
    first = (
        " 0.11000E+02   0.45000E+02   0.70057E-02   0.00000E+00"
        "  -0.22607E-01  -0.64191E-03   0.56930E-01"
    )
    rows = [first]
    for i in range(1, 200):
        energy = 11 + 49 * i / 199
        intensity = .007 + .08 * (i / 199)
        rows.append(
            f" {energy:.5E}  4.50000E+01  {intensity:.5E}  0.00000E+00"
            "  1.00000E-02  -2.00000E-02  3.00000E-02"
        )
    return (
        native_header("SPLEED")
        + "NT                 1\nNP                 1\n"
        + "#Decription of the columns\n"
        + "#1.-Energy, 2.-Theta\n"
        + "#3.-RINTG(G), 4.ASYMM(G)\n"
        + "#5.-P(1,G), 6.P(2,G)\n"
        + "#7.-P(3,G)\n"
        + "\n".join(rows) + "\n"
    )


@pytest.mark.parametrize("hint", [None, "spc", "spleed_spc"])
def test_spleed_parsed_by_keyword_with_shared_spc_extension(tmp_path, hint):
    source = tmp_path / "Fe2_SPLEED_SPLEED_data.spc"
    source.write_text(spleed_content())
    result = ResultFile.from_file(source, try_only=hint, unknown=False)
    assert isinstance(result, SPLEEDSPCResultFile)
    assert result.KEYWORD() == "SPLEED"
    assert result.RAW_DATA().shape == (200, 7)
    np.testing.assert_allclose(result.THETA(), 45.)
    assert result.INTENSITY()[0] == pytest.approx(0.0070057)
    assert result.P_Z()[0] == pytest.approx(0.05693)

    target = tmp_path / "spleed.png"
    figure = result.plot(filename=str(target), show=False, dpi=72)
    try:
        assert target.stat().st_size > 0
        assert len(figure.axes) == 3
        assert len(figure.axes[2].lines) == 3
    finally:
        from matplotlib import pyplot as plt
        plt.close(figure)


def test_header_only_dos_has_clear_plot_error(tmp_path):
    source = tmp_path / "Fe2_SPLEED_SPLEED.dos"
    source.write_text(native_header("DOS") + "DOS-FMT:  OLD-SPRKKR\n")
    result = ResultFile.from_file(source, try_only="dos", unknown=False)
    assert result.KEYWORD() == "DOS"
    assert result.NE() == 200
    assert np.asarray(result.RAW_DATA()).size == 0

    with pytest.raises(ValueError, match="no numerical DOS data"):
        result.plot(filename=str(tmp_path / "invalid_dos.png"), show=False)
