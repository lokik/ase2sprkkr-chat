"""Regression for optional VECKA in the native 2026 SPR-KKR CONST-E BSF header."""

from pathlib import Path

import numpy as np
import pytest

from .. import ResultFile


FIXTURE = Path(__file__).parent / "data" / "KK_BLOCHSF_VECKA_small.bsf"


@pytest.mark.smoke
@pytest.mark.parametrize("with_vecka", (True, False))
def test_const_e_bsf_accepts_optional_vecka(tmp_path, with_vecka):
    # Reduced from the supplied 60x60 Fe BSFKK file, preserving the native
    # header layout and endpoint samples of every spin / site data block.
    text = FIXTURE.read_text()
    if not with_vecka:
        text = "".join(
            line for line in text.splitlines(keepends=True)
            if not line.startswith("VECKA ")
        )

    path = tmp_path / "KK_BLOCHSF.bsf"
    path.write_text(text)
    result = ResultFile.from_file(path, unknown=False)

    assert result.KEYWORD() == "BSF"
    assert result.MODE() == "CONST-E"
    assert (result.NK1(), result.NK2(), result.NQ_EFF()) == (2, 2, 2)
    np.testing.assert_allclose(result.VECK1(), (1, 0, 0))
    np.testing.assert_allclose(result.VECK2(), (0, 1, 0))
    if with_vecka:
        np.testing.assert_allclose(result.VECKA(), (0, 0, 0))
    else:
        assert result.VECKA() is None

    assert result.RAW_DATA().size == 24
    assert result.I().shape == (2, 2, 2)
    assert result.I_UP().shape == (2, 2, 2)
    assert result.I_DOWN().shape == (2, 2, 2)
    assert np.isfinite(result.RAW_DATA()).all()
    assert np.count_nonzero(result.RAW_DATA()) == 1


@pytest.mark.smoke
def test_const_e_bsf_with_vecka_plots(tmp_path):
    pytest.importorskip("matplotlib")
    result = ResultFile.from_file(FIXTURE, unknown=False)
    destination = tmp_path / "bsf.png"
    result.plot(filename=str(destination), show=False, dpi=72)
    assert destination.exists() and destination.stat().st_size > 0


def test_const_e_bsf_vecka_is_part_of_grid_identity(tmp_path):
    source = FIXTURE.read_text()
    other = tmp_path / "shifted.bsf"
    other.write_text(source.replace(
        "VECKA         0.0000    0.0000    0.0000",
        "VECKA         0.2500    0.0000    0.0000",
    ))
    original = ResultFile.from_file(FIXTURE, unknown=False)
    shifted = ResultFile.from_file(other, unknown=False)
    np.testing.assert_allclose(shifted.VECKA(), (0.25, 0, 0))
    with pytest.raises(ValueError, match="VECKA"):
        _ = original + shifted
