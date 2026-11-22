"""Opt-in bulk regex/NumPy parser for regular SPR-KKR numeric tables.

The grammar recognizes an entire block once, rather than creating a
pyparsing ParseResults for every scalar.  Numeric conversion is delegated to
NumPy; typed fields and grouping are reconstructed afterward.

Only regular, unheaded tables with primitive numeric columns and optional
enumerated trailing text are supported. Other Table configurations retain
their existing pyparsing implementation.
"""

import re

import numpy as np
import pyparsing as pp


_REAL = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?"
_SIGN_BOUNDARY = re.compile(r"(?<=[0-9])(?=[+-](?:[0-9]|\.[0-9]))")
_D_EXPONENT = re.compile(r"[dD](?=[+-]?[0-9])")


class RegexNumericTable(pp.Token):
    """A single pyparsing token containing a complete regular numeric table."""

    def __init__(self, table):
        super().__init__()

        from .arrays import Complex
        from .basic import Integer, Real, String

        if (
            table.header or table.free_header or table.repeat_sparse
            or table.group_size or table.groups_as_list or table.flatten
            or table.free_numbering or table.sequence.condition is not None
            or table.sequence.default_values or not table.names
        ):
            raise ValueError("fast_numeric requires a regular named, unheaded table")

        # Custom column grammars and custom index formats belong to the
        # existing, fully general Table parser.
        for special in (table.grouping, table.numbering):
            if special and (
                not isinstance(special.column, Integer)
                or special.column.prefix or special.column.postfix
            ):
                raise ValueError("fast_numeric requires plain integer group/row numbers")

        self.table = table
        self.fields = []
        row_columns = []
        for name, item in zip(table.names, table.sequence.types):
            if item.prefix or item.postfix or item.condition:
                raise ValueError(f"fast_numeric requires plain column tokens: {name}")
            if isinstance(item, (Integer, Real)) and (
                item.min is not None or item.max is not None
            ):
                raise ValueError(f"fast_numeric does not support numeric constraints: {name}")
            if isinstance(item, Real) and item.nan is not None:
                raise ValueError(f"fast_numeric does not support alternate NaN tokens: {name}")
            if isinstance(item, Complex):
                kind, pattern = "complex", _REAL + r"[ \t]*" + _REAL
            elif isinstance(item, Integer):
                kind, pattern = "int", r"[+-]?[0-9]+"
            elif isinstance(item, Real):
                kind, pattern = "float", _REAL
            elif isinstance(item, String) and name in table.value_codes:
                mapping = table.value_codes[name]
                if not mapping or len(set(mapping.values())) != len(mapping):
                    raise ValueError("fast_numeric value codes must be unique")
                kind = "label"
                pattern = "(?:" + "|".join(
                    re.escape(value) for value in sorted(mapping, key=len, reverse=True)
                ) + ")"
            else:
                raise ValueError(f"fast_numeric cannot parse column {name!r}: {item}")

            self.fields.append((name, kind))
            row_columns.append((kind, pattern))

        if sum(kind == "label" for _, kind in self.fields) > 1:
            raise ValueError("fast_numeric supports at most one coded text column")
        if any(kind == "label" for _, kind in self.fields[:-1]):
            raise ValueError("fast_numeric requires coded text to be the last column")

        parts = []
        if table.grouping:
            parts.append(("index", r"[0-9]+"))
        if table.numbering:
            parts.append(("index", r"[0-9]+"))
        parts.extend(row_columns)

        row = r"[ \t]*" + parts[0][1]
        for kind, pattern in parts[1:]:
            # The Fortran format writes the label immediately after the last
            # floating-point field, e.g. 0.123E-04(+).
            separator = r"[ \t]*" if kind == "label" else r"[ \t]+"
            row += separator + pattern
        row += r"[ \t]*"

        self._table_regex = re.compile(rf"(?:{row}\r?\n)*{row}(?=\r?\n|$)")
        self._number_columns = (
            int(bool(table.grouping)) + int(bool(table.numbering))
            + sum(2 if kind == "complex" else 1 for _, kind in self.fields)
        )
        codes = {}
        for name, kind in self.fields:
            if kind == "label":
                codes = table.value_codes[name]
        self._codes = codes
        if codes:
            self._codes_regex = re.compile(
                "(?:" + "|".join(re.escape(k) for k in sorted(codes, key=len, reverse=True))
                + r")(?=[ \t]*(?:\r?\n|$))"
            )
        else:
            self._codes_regex = None

        self.set_name("<fast numeric table>")

    def parseImpl(self, text, loc, doActions=True):
        match = self._table_regex.match(text, loc)
        if match is None:
            raise pp.ParseException(text, loc, "Expected numeric table", self)

        block = match.group()
        if self._codes_regex is not None:
            block = self._codes_regex.sub(lambda m: " " + str(self._codes[m.group()]), block)
        block = _D_EXPONENT.sub("E", block)
        block = _SIGN_BOUNDARY.sub(" ", block)

        nrows = match.group().count("\n") + 1
        try:
            raw = np.fromstring(block, dtype=float, sep=" ")
            if raw.size != nrows * self._number_columns:
                raise ValueError("Incorrect number of numeric fields")
            raw = raw.reshape(nrows, self._number_columns)
        except ValueError as exc:
            raise pp.ParseException(text, loc, str(exc), self) from exc

        offset = 0
        groups = None
        numbers = None
        if self.table.grouping:
            groups = raw[:, offset].astype(np.int64)
            offset += 1
        if self.table.numbering:
            numbers = raw[:, offset].astype(np.int64)
            offset += 1

        starts = np.array([0], dtype=np.int64)
        if groups is not None:
            starts = np.r_[starts, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
            if not np.array_equal(groups[starts], np.arange(1, len(starts) + 1)):
                raise pp.ParseException(text, loc, "Invalid group numbering", self)

        counts = np.diff(np.r_[starts, nrows])
        if numbers is not None:
            expected = np.arange(nrows) - np.repeat(starts, counts) + 1
            if not np.array_equal(numbers, expected):
                raise pp.ParseException(text, loc, "Invalid row numbering", self)

        dtype_spec = self.table._numpy_type
        structured = isinstance(dtype_spec, list)
        result = np.empty(
            nrows if structured else (nrows, len(self.fields)), dtype=dtype_spec
        )
        inverse = {code: label for label, code in self._codes.items()}
        for column, (name, kind) in enumerate(self.fields):
            if kind == "complex":
                value = raw[:, offset] + 1j * raw[:, offset + 1]
                offset += 2
            elif kind == "label":
                codes = raw[:, offset].astype(np.int64)
                if any(int(code) not in inverse for code in codes):
                    raise pp.ParseException(text, loc, "Invalid text column", self)
                value = [inverse[int(code)] for code in codes]
                offset += 1
            else:
                value = raw[:, offset]
                offset += 1
            if structured:
                result[name] = value
            else:
                result[:, column] = value

        if self.table.grouping:
            if np.any(counts != counts[0]):
                raise pp.ParseException(text, loc, "Unequal group lengths", self)
            result = result.reshape(len(counts), int(counts[0]))
        if self.table.length is not None and len(result) != self.table.length:
            raise pp.ParseException(text, loc, "Incorrect table length", self)
        return match.end(), pp.ParseResults([result])
