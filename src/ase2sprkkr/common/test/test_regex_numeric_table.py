"""Checks the bulk Table parser against the standard pyparsing grammar."""

import numpy as np
import pyparsing as pp
import pytest

from ase2sprkkr.common.grammar_types import Table
from ase2sprkkr.common.grammar import generate_grammar


CODES = {"ADD": {"(+)": 1, "(-)": -1, "(z)": 0, "(x)": 2, "(y)": 3}}
COLUMNS = {"ABSRTA": complex, "ABSRTB": complex, "ABSRATE": complex, "ADD": str}

# Taken from the native xrayspec.f FORMAT 99015 pattern: 2I2,
# 2E15.8 + 2E15.8 + 2E15.8 and a parenthesized polarization label.
SPHERICAL = (
    " 1 1  0.71394718E+00-0.10470321E-03  0.29517503E-05-0.43288600E-09   0.71395013E+00-0.10470365E-03(+)\n"
    " 1 2  0.24362109E+00-0.35728009E-04  0.82071741E-05-0.12036149E-08   0.24362930E+00-0.35729213E-04(-)\n"
    " 1 3  0.47878414E+00-0.70215611E-04  0.00000000E+00 0.00000000E+00   0.47878414E+00-0.70215611E-04(z)"
)
CARTESIAN = (
    " 1 1  0.71394718D+00-0.10470321D-03  0.29517503D-05-0.43288600D-09   0.71395013D+00-0.10470365D-03(x)\n"
    " 1 2  0.24362109E+00-0.35728009E-04  0.82071741E-05-0.12036149E-08   0.24362930E+00-0.35729213E-04(y)\n"
    " 1 3  0.47878414E+00-0.70215611E-04  0.00000000E+00 0.00000000E+00   0.47878414E+00-0.70215611E-04(z)"
)


def _table(*, fast):
    return Table(
        COLUMNS,
        grouping=True,
        numbering=True,
        header=False,
        groups_as_list=False,
        fast_numeric=fast,
        value_codes=CODES if fast else None,
    )


@pytest.mark.parametrize("sample", [SPHERICAL, CARTESIAN.replace("D", "E")])
def test_fast_table_matches_standard_parser(sample):
    with generate_grammar():
        expected = _table(fast=False).parse(sample)
        actual = _table(fast=True).parse(sample)
    assert actual.shape == expected.shape == (1, 3)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_fast_table_accepts_fortran_d_exponents():
    with generate_grammar():
        actual = _table(fast=True).parse(CARTESIAN)
    assert actual["ADD"].tolist() == [["(x)", "(y)", "(z)"]]
    assert np.isclose(actual["ABSRTA"][0, 0].real, 0.71394718)
    assert np.isclose(actual["ABSRTA"][0, 0].imag, -0.10470321e-3)


def test_fast_table_preserves_group_shape_and_separator():
    sample = SPHERICAL + "\n" + "\n".join(
        line.replace(" 1 ", " 2 ", 1) for line in SPHERICAL.splitlines()
    )
    with generate_grammar():
        expected = _table(fast=False).parse(sample)
        fast = _table(fast=True)
        actual = fast.parse(sample)
        # The blank line after the final row belongs to the next grammar.
        match_end, _ = fast._grammar().parseImpl(sample + "\n \nNEXT", 0)
    assert actual.shape == expected.shape == (2, 3)
    np.testing.assert_array_equal(actual, expected)
    assert match_end == len(sample)


@pytest.mark.parametrize("sample", [
    SPHERICAL.replace(" 1 2 ", " 1 3 ", 1),
    SPHERICAL.replace(" 1 2 ", " 3 2 ", 1),
    SPHERICAL.replace("0.24362109E+00", "INVALID", 1),
    SPHERICAL.replace("(+)", "(q)", 1),
    SPHERICAL.rsplit("\n", 1)[0] + "\n 1 3  0.47878414E+00 BROKEN (z)",
])
def test_fast_table_rejects_malformed_rows(sample):
    with generate_grammar():
        with pytest.raises((pp.ParseBaseException, ValueError)):
            _table(fast=True).parse(sample)


def test_fast_table_rejects_unsupported_schema():
    with generate_grammar():
        table = Table({"COUNT": int}, repeat_sparse=["COUNT"], fast_numeric=True)
        with pytest.raises(ValueError, match="regular named"):
            table.grammar()


def test_fast_table_preserves_homogeneous_numeric_arrays():
    sample = "1 1 2.0 3.0\n1 2 -1.25 4.5\n2 1 0.0 1.0\n2 2 2.5 6.0"
    options = dict(grouping=True, numbering=True, header=False, groups_as_list=False)
    with generate_grammar():
        expected = Table({"A": float, "B": float}, **options).parse(sample)
        actual = Table({"A": float, "B": float}, fast_numeric=True, **options).parse(sample)
    assert actual.shape == expected.shape == (2, 2, 2)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def test_native_rat_fixture_is_parsed_with_bulk_table():
    from ase2sprkkr.result_files.definitions.rat import RATResultFile
    from pathlib import Path

    filename = Path(__file__).parents[2] / "result_files" / "examples" / "Cu_XAS_Cu_L23.rat"
    result = RATResultFile.from_file(filename)
    assert result.GROUPS[0].DATA[0].DATA().shape == (6, 3)
    assert result.GROUPS[0].DATA[0].DATA()["ADD"][0].tolist() == ["(+)", "(-)", "(z)"]
