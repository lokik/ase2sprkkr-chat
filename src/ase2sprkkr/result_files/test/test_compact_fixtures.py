"""Compact real-data-derived smoke fixtures, independent of full-size examples.

The BSF files retain native column block ordering and a selected subgrid.
The SPC energy slice retains native rows; kx_ky uses a small orthogonal grid.
Full-resolution datasets remain in result_files/examples for numerical tests.
"""
from pathlib import Path

import numpy as np
import pytest

from .. import ResultFile

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("name", "keyword", "shape", "raw_size"),
    (
        ("SnTe_BLOCHSF_small.bsf", "BSF", (4, 4, 5), 240),
        ("SnTe_BLOCHSF_spol_small.bsf", "BSF-SPOL", (4, 4, 5), 320),
    ),
)
def test_compact_bsf_parses_all_native_blocks(name, keyword, shape, raw_size):
    result = ResultFile.from_file(DATA / name, unknown=False)
    assert result.KEYWORD() == keyword
    assert result.MODE() == "EK-REL"
    assert result.NE() == 4
    assert result.NK() == 5
    assert result.INDKDIR()[-1] == 5
    assert result.RAW_DATA().size == raw_size
    assert result.I().shape == shape
    channels = ("I_UP", "I_DOWN") if keyword == "BSF" else ("I_X", "I_Y", "I_Z")
    for channel in channels:
        assert result[channel]().shape == shape
        assert np.isfinite(result[channel]()).all()


@pytest.mark.parametrize(
    ("name", "mode", "np_points", "nt_points", "ne_points", "columns"),
    (
        ("Mo2Se4_ARPES_small.spc", "energy", 1, 1, 7, 8),
        ("Mo2Se4_ARPES_kx_ky_small.spc", "kx_ky", 5, 7, 1, 7),
    ),
)
def test_compact_arpes_parses_consistent_native_grid(
    name, mode, np_points, nt_points, ne_points, columns,
):
    result = ResultFile.from_file(DATA / name, unknown=False)
    assert result.KEYWORD() == "ARPES"
    assert result.MODE() == mode
    assert result.NP() == np_points
    assert result.NT() == nt_points
    assert result.NE() == ne_points
    assert result.RAW_DATA().shape == (np_points * nt_points * ne_points, columns)
    assert np.isfinite(result.TOTAL()).all()
    assert np.isfinite(result.POLARIZATION()).all()
