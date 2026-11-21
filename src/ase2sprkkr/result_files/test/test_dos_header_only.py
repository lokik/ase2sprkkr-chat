"""Regression for header-only DOS produced alongside SPLEED runs."""

import numpy as np
import pytest

from .. import ResultFile


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


def test_header_only_dos_has_clear_plot_error(tmp_path):
    source = tmp_path / "Fe2_SPLEED_SPLEED.dos"
    source.write_text(native_header("DOS") + "DOS-FMT:  OLD-SPRKKR\n")
    result = ResultFile.from_file(source, try_only="dos", unknown=False)
    assert result.KEYWORD() == "DOS"
    assert result.NE() == 200
    assert np.asarray(result.RAW_DATA()).size == 0

    with pytest.raises(ValueError, match="no numerical DOS data"):
        result.plot(filename=str(tmp_path / "invalid_dos.png"), show=False)

