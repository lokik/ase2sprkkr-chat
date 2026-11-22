"""Structured INFO for native SPR-KKR 9.7 XAS (XRAYSPEC).

The ordinary result/file handling is retained; this parser observes exact
Fortran WRITE(6,99012) energy-loop records, not the separate .rat table.
XMO/XRS use other branches and intentionally do not share this reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os
import re

from .default import DefaultResult, DefaultOutputParser
from ..task_result import KkrOutputReader


def xas_total_points_from_parameters(parameters):
    """Read the single XAS energy contour; never guess from a split path."""
    try:
        values = parameters.ENERGY.NE()
        if isinstance(values, Integral):
            values = (values,)
        else:
            values = tuple(values)
    except (AttributeError, TypeError, ValueError):
        return None
    if len(values) != 1 or not isinstance(values[0], Integral):
        return None
    return int(values[0]) if values[0] > 0 else None


@dataclass(frozen=True)
class XasCoreInfo:
    """Core quantum numbers from XRAYSPEC's FORMAT 99022."""
    atom_type: int
    n: int
    l: int


@dataclass(frozen=True)
class XasEnergyInfo:
    """Energy being entered; the current energy is NOT yet completed."""
    index: int
    completed: int
    total: int | None


class _XasHeaderStream:
    """Forward header lines to INFO before read_commons consumes them."""

    def __init__(self, source, on_line):
        self.source = source
        self.on_line = on_line

    async def readline(self):
        line = await self.source.readline()
        if line:
            self.on_line(line.decode("utf-8", errors="replace"))
        return line

    def __fspath__(self):
        return os.fspath(self.source)

    def __getattr__(self, key):
        return getattr(self.source, key)


class XasOutputParser(DefaultOutputParser):
    """Detect milestones from native XRAYSPEC stdout, including early exits.

    In XAS, NELOOP=1. The 3-pass branch belongs to XMO/XRS, not XAS.
    FORMAT 99012: ' LOOP',I2,I5,' E=',2F6.3,'  IQ=',I2,'  IT=',I2.
    Each record is written at the *beginning* of an energy calculation.
    Only FORMAT 99013 confirms that the complete spectrum was calculated.
    """

    _rate_file = re.compile(
        r"^\s*RATE-FILE\s*:\s*\(\s*7\s*\)\s*(?P<filename>\S+)",
        re.IGNORECASE,
    )
    _core = re.compile(
        r"\bCORE\s+QUANTUM-NUMBERS\s+FOR\s+IT\s*=\s*(\d+)\s*:"
        r"\s*N\s*=\s*(\d+)\s+L\s*=\s*(\d+)",
        re.IGNORECASE,
    )
    # The two integer fields can touch when I5 is filled, so read widths
    # rather than assuming whitespace between IELOOP and IENERG.
    _energy = re.compile(
        r"^\s*LOOP(?P<pass>.{2})(?P<index>.{5})\s+E\s*=",
        re.IGNORECASE,
    )
    _complete = re.compile(
        r"\bX-RAY\s+ABSORPTION-RATE\s+CALCULATED\s+FOR\s+(\d+)"
        r"\s+ENERGY-POINTS\s+>>>\s*FEIERABEND\b", re.IGNORECASE,
    )
    _finished = re.compile(
        r"\bfinished\s+in\s+subroutine\s*<XRAYSPEC>", re.IGNORECASE,
    )

    def __init__(self, print_info=False, on_info=None, total_points=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self.total_points = total_points
        self._started = False
        self._last_index = None
        self._complete_seen = False
        self._finished_seen = False
        self.rate_filename = None
        self.core = None
        self.completed_points = None

    def format_info(self, kind, data):
        if kind == "xas_started":
            return "Calculating X-ray absorption spectrum..."
        if kind == "xas_rate_file":
            return f"XAS spectrum file: {data}"
        if kind == "xas_core":
            return f"Core shell: n={data.n}, l={data.l}, atom type {data.atom_type}"
        if kind == "xas_energy":
            if data.total is None:
                return f"Calculating XAS energy point {data.index}..."
            return f"Calculating XAS energy {data.index}/{data.total}..."
        if kind == "xas_complete":
            return f"XAS spectrum calculated: {data} energy points"
        if kind == "xas_finished":
            return "XAS calculation finished"
        return None

    def _start(self):
        if not self._started:
            self._started = True
            self.emit_info("xas_started", None)

    def _consume_line(self, line):
        if match := self._rate_file.search(line):
            self._start()
            filename = match.group("filename")
            if self.rate_filename != filename:
                self.rate_filename = filename
                self.emit_info("xas_rate_file", filename)
            return

        if match := self._core.search(line):
            self._start()
            core = XasCoreInfo(*map(int, match.groups()))
            if self.core != core:
                self.core = core
                self.emit_info("xas_core", core)
            return

        if match := self._energy.match(line):
            try:
                energy_pass = int(match.group("pass").strip())
                index = int(match.group("index").strip())
            except ValueError:
                return
            if energy_pass != 1 or index < 1:
                return
            self._start()
            if self._last_index == index:
                return
            self._last_index = index
            if self.total_points is None or index <= self.total_points:
                self.emit_info(
                    "xas_energy",
                    XasEnergyInfo(index, index - 1, self.total_points),
                )
            return

        if match := self._complete.search(line):
            count = int(match.group(1))
            if count > 0 and not self._complete_seen:
                self._start()
                self._complete_seen = True
                self.completed_points = count
                self.emit_info("xas_complete", count)
            return

        # A generic STOP_REGULAR from GEN is NOT an XAS completion:
        # the MPI branch exits before calling XRAYSPEC.
        if self._complete_seen and not self._finished_seen and self._finished.search(line):
            self._finished_seen = True
            self.emit_info("xas_finished", None)

    async def read_output(self, stdout, result):
        await self.read_commons(_XasHeaderStream(stdout, self._consume_line), result)
        if self.total_points is None:
            self.total_points = xas_total_points_from_parameters(
                getattr(result, "input_parameters", None)
            )
        result.output_lines = []
        async for raw_line in stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            result.output_lines.append(line)
            self._consume_line(line)
        result.xas_rate_filename = self.rate_filename
        result.xas_core = self.core
        result.xas_completed_points = self.completed_points
        return result


class XasProgressParser(XasOutputParser):
    """Incremental reader for GUI stdout chunks and attached-run replays."""

    def __init__(self, on_info=None, print_info=False, total_points=None):
        super().__init__(print_info=print_info, on_info=on_info, total_points=total_points)
        self._partial = ""

    def feed(self, text):
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._consume_line(line)

    def finish(self):
        if self._partial:
            self._consume_line(self._partial)
            self._partial = ""


class XasOutputReader(KkrOutputReader):
    """Keep the existing generic XAS results, add task-specific INFO."""
    result_class = DefaultResult
    parser_class = XasOutputParser
