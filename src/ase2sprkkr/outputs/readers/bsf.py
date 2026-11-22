"""The Bloch spectral functions (BSF) reader and result."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from math import ceil

from ..task_result import TaskResult, KkrOutputReader
from .default import DefaultOutputParser
from ...common.decorators import cached_property


class BsfResult(TaskResult):
    """BSF result provides access to the computed Bloch spectral functions in the
    BSF output file using the :py:attr:`~bsf` property."""

    @cached_property
    def bsf_filename(self):
        """Bloch spectral functions file name"""
        return self.path_to("Bloch-SF")

    @cached_property
    def bsf(self):
        """The computed Bloch spectral functions."""
        return self._bsf_file.parsed_file()

    @cached_property
    def _bsf_file(self):
        try:
            return self.files.set_file_type("Bloch-SF", "bsf")
        except KeyError as exc:
            raise FileNotFoundError("BSF result file is not listed in the SPR-KKR output.") from exc

    @property
    def output_values(self):
        # Missing BSF outputs should not suppress the other parsed results.
        if "Bloch-SF" in self.files:
            self._bsf_file
        return super().output_values



@dataclass(frozen=True)
class BsfGridInfo:
    """BSF sampling dimensions reported by SPR-KKR."""

    mode: str
    energies: int | None = None
    k_points: int | None = None
    nk1: int | None = None
    nk2: int | None = None


@dataclass(frozen=True)
class BsfEnergyInfo:
    """Energy whose k-loop is beginning, NOT a completed energy point."""

    index: int
    real_ry: float
    imaginary_ry: float
    total: int | None = None


@dataclass(frozen=True)
class BsfFinishedInfo:
    """Native BLOCHSF completion and diagnostic warning count."""

    warnings: int


@dataclass(frozen=True)
class BsfRankProgressInfo:
    """Energy loops entered by the reporting rank, not global MPI completion."""

    entered: int
    total: int
    first_global_index: int
    current_global_index: int

    @property
    def completed(self):
        # The current loop has only *started*, not finished.
        return max(0, self.entered - 1)


class _BsfHeaderStream:
    """Preserve MPI metadata from lines skipped by the standard FILES reader."""

    def __init__(self, source, on_line):
        self.source = source
        self.on_line = on_line

    async def readline(self):
        line = await self.source.readline()
        if line:
            self.on_line(line.decode("utf8", errors="replace"))
        return line

    def __fspath__(self):
        return os.fspath(self.source)

    def __getattr__(self, name):
        return getattr(self.source, name)


class BsfOutputParser(DefaultOutputParser):
    """Observe native BSF milestones in both E-k and constant-energy k-k modes.

    The E-k energy indices are not progress percentages: MPI output may contain
    only one rank's subset (e.g. IE=151..200 out of 200). In k-k mode there
    is only one energy and no k-point counters in native stdout.
    """

    _nprocs = re.compile(r"\bMPI calculation with NPROCS\s*=\s*(\d+)", re.IGNORECASE)
    _energy_window = re.compile(r"^\s*energy loop\s+(\d+)\s*$", re.IGNORECASE)
    _outer_krow = re.compile(r"^\s*outer k-loop:\s*(\d+)\s+k-points done\b", re.IGNORECASE)
    _ek_mode = re.compile(r"\bE\(k\)-relation mode\b", re.IGNORECASE)
    _kk_mode = re.compile(r"\bconstant E mode\b", re.IGNORECASE)
    _ek_grid = re.compile(r"\bwith\s+NE\s*=\s*(\d+)\s+NK\s*=\s*(\d+)", re.IGNORECASE)
    _kk_nk1 = re.compile(r"\bNK1\s*=\s*(\d+)", re.IGNORECASE)
    _kk_nk2 = re.compile(r"\bNK2\s*=\s*(\d+)", re.IGNORECASE)
    _fixed_energy = re.compile(r"^\s*fixed energy\s+(\S+)", re.IGNORECASE)
    _energy = re.compile(
        r"^\s*IE\s*=\s*(\d+)\s+E\s*=\s*(\S+)\s+(\S+)"
        r"\s+all matrices set up\s*-\s*enter k-loop\b", re.IGNORECASE
    )
    _negative = re.compile(r"WARNING\s*<BLOCHBSF>:\s*Spectral function negative", re.IGNORECASE)
    _finished = re.compile(r"\bfinished in subroutine\s*<BLOCHSF>", re.IGNORECASE)

    def __init__(self, print_info=False, on_info=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self.mode = None
        self.grid = None
        self._nk1 = None
        self._nk2 = None
        self.negative_warnings = 0
        self._finished_seen = False
        self._last_energy = None
        self.nprocs = None
        self._rank_start = None

    @staticmethod
    def format_info(kind, data):
        """User-facing INFO strings are owned by ase2sprkkr, not the GUI."""
        if kind == "bsf_energy_window":
            return f"BSF photon-energy loop {data}"
        if kind == "bsf_outer_krow":
            return f"Entering k-plane row {data} (row not yet complete)"
        if kind == "bsf_mode":
            return "Calculating BSF in E–k mode" if data == "EK" else "Calculating BSF in k–k mode"
        if kind == "bsf_grid":
            if data.mode == "EK":
                return f"Energy/k-path grid: {data.energies} energies, {data.k_points} k-points"
            return f"Constant-energy k-plane: {data.nk1} × {data.nk2} k-points"
        if kind == "bsf_fixed_energy":
            return f"Fixed energy: {data:g} Ry"
        if kind == "bsf_energy":
            index = f"{data.index} (NE={data.total})" if data.total else str(data.index)
            return (
                f"Entering k-loop for energy index {index}: "
                f"E = {data.real_ry:g} + {data.imaginary_ry:g}i Ry"
            )
        if kind == "bsf_rank_progress":
            return (
                f"Reporting MPI rank: energy loop {data.entered}/{data.total} "
                f"entered (global IE={data.current_global_index})"
            )
        if kind == "bsf_negative_warnings":
            return f"Negative spectral-function warnings: {data:,}"
        if kind == "bsf_finished":
            suffix = f" ({data.warnings:,} negative spectral-function warnings)" if data.warnings else ""
            return f"BSF calculation finished{suffix}"
        return None

    def _set_mode(self, mode):
        if self.mode != mode:
            self.mode = mode
            self.emit_info("bsf_mode", mode)

    def _set_grid(self, grid):
        if self.grid != grid:
            self.grid = grid
            self.emit_info("bsf_grid", grid)

    def _consume_line(self, line):
        if match := self._energy_window.match(line):
            # EPHOT triggers a second native energy loop (blochsf.f).
            # Start fresh local-rank accounting; this is not a completion.
            self._rank_start = None
            self._last_energy = None
            self.emit_info("bsf_energy_window", int(match[1]))
        if self.mode == "KK" and (match := self._outer_krow.match(line)):
            # Native wording says "done" but WRITE precedes the row's k-loop.
            self.emit_info("bsf_outer_krow", int(match[1]))
        if match := self._nprocs.search(line):
            self.nprocs = max(1, int(match[1]))
        if self._ek_mode.search(line):
            self._set_mode("EK")
        elif self._kk_mode.search(line):
            self._set_mode("KK")

        if match := self._ek_grid.search(line):
            self._set_mode("EK")
            self._set_grid(BsfGridInfo("EK", int(match[1]), int(match[2])))
        elif self.mode == "KK":
            if match := self._kk_nk1.search(line):
                self._nk1 = int(match[1])
            if match := self._kk_nk2.search(line):
                self._nk2 = int(match[1])
            if self._nk1 is not None and self._nk2 is not None:
                self._set_grid(BsfGridInfo("KK", nk1=self._nk1, nk2=self._nk2))

        if self.mode == "KK" and (match := self._fixed_energy.match(line)):
            try:
                self.emit_info("bsf_fixed_energy", self._to_float(match[1]))
            except ValueError:
                pass

        if match := self._energy.match(line):
            try:
                point = BsfEnergyInfo(
                    int(match[1]),
                    self._to_float(match[2]),
                    self._to_float(match[3]),
                    self.grid.energies if self.grid and self.grid.mode == "EK" else None,
                )
            except ValueError:
                pass
            else:
                if point != self._last_energy:
                    self._last_energy = point
                    self.emit_info("bsf_energy", point)
                    if self.mode == "EK" and self.grid is not None and self.nprocs:
                        if self._rank_start is None or point.index < self._rank_start:
                            self._rank_start = point.index
                        # MPI divides NE energy points across ranks. Rank 0
                        # reports 151..200 for NE=200, NPROCS=4 in this log.
                        # This is NOT a global completion counter.
                        total = min(
                            ceil(self.grid.energies / self.nprocs),
                            self.grid.energies - self._rank_start + 1,
                        )
                        entered = point.index - self._rank_start + 1
                        if total > 0 and 0 < entered <= total:
                            self.emit_info(
                                "bsf_rank_progress",
                                BsfRankProgressInfo(
                                    entered, total, self._rank_start, point.index
                                ),
                            )

        if self._negative.search(line):
            self.negative_warnings += 1
            # Throttle identical warnings in live GUI output, yet retain the
            # exact final count. The raw output is unaffected.
            if self.negative_warnings in (1, 10, 100, 1000, 10000):
                self.emit_info("bsf_negative_warnings", self.negative_warnings)

        if not self._finished_seen and self._finished.search(line):
            self._finished_seen = True
            self.emit_info("bsf_finished", BsfFinishedInfo(self.negative_warnings))

    @staticmethod
    def _to_float(value):
        return float(value.replace("D", "E").replace("d", "e"))

    async def read_output(self, stdout, result):
        await self.read_commons(
            _BsfHeaderStream(stdout, self._consume_line), result
        )
        result.output_lines = []
        async for raw_line in stdout:
            line = raw_line.decode("utf8", errors="replace").rstrip()
            result.output_lines.append(line)
            self._consume_line(line)
        result.bsf_negative_warnings = self.negative_warnings
        result.bsf_mode = self.mode
        return result


class BsfProgressParser(BsfOutputParser):
    """Incremental INFO events from a growing or replayed native BSF stdout."""

    def __init__(self, on_info=None, print_info=False):
        super().__init__(print_info=print_info, on_info=on_info)
        self._partial = ""

    def feed(self, text):
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._consume_line(line)

    def finish(self):
        if self._partial:
            self._consume_line(self._partial)
            self._partial = ""

class BsfOutputReader(KkrOutputReader):
    """Expose BSF results and task-specific structured INFO milestones."""

    result_class = BsfResult
    parser_class = BsfOutputParser
