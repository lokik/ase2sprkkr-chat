"""Density of states (DOS) reader and result."""

import os
import re
from dataclasses import dataclass
from numbers import Integral

from ..task_result import TaskResult, KkrOutputReader
from .default import DefaultOutputParser
from ...common.decorators import cached_property
from ...input_parameters.definitions._filenames import dataset_adsi


def dos_total_points_from_parameters(parameters):
    """Use only an unambiguous one-segment ENERGY.NE value."""
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
    total = int(values[0])
    return total if total > 0 else None


@dataclass(frozen=True)
class DosProgressInfo:
    """Completed DOS energy points; total is optional when unknown."""

    completed: int
    total: int | None


class DosOutputParser(DefaultOutputParser):
    """Parse native DOS progress lines without estimating missing milestones.

    Progress uses native energy-block transitions or explicit completed counts.
    The same detection is shared with the incremental GUI parser.
    """

    # Native kkrgen prints one such header for every energy AND atomic
    # component IT. Repeated IT headers must not advance the progress.
    _energy_header = re.compile(
        r"^\s*(?P<point>[1-9]\d*)\s+E\s*=\s*\S+.*?\s+IT\s*=\s*\d+\b",
        re.IGNORECASE,
    )
    _completed_points = re.compile(
        r"(?<!\d)(?P<completed>\d+)\s*(?:/|\bof\b)\s*"
        r"(?P<total>\d+)\s*(?:energy\s+points?\s*)?"
        r"(?:done|completed|processed)\b",
        re.IGNORECASE,
    )

    def __init__(self, print_info=False, on_info=None, total_points=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self.total_points = total_points
        self.last_progress = None
        self._current_point = None
        self._started = False

    def _note_started(self):
        if not self._started:
            self._started = True
            self.emit_info("dos_started", None)

    def format_info(self, kind, data):
        if kind == "dos_started":
            return "Calculating density of states..."
        if kind == "dos_progress":
            return (f"{data.completed}/{data.total} done." if data.total is not None
                    else f"{data.completed} energy points done.")
        return None

    def _emit_progress(self, completed, total):
        if completed <= 0 or (total is not None and completed > total):
            return
        progress = DosProgressInfo(completed, total)
        if progress != self.last_progress:
            self.last_progress = progress
            self.emit_info("dos_progress", progress)

    def _consume_line(self, line):
        if line.strip():
            self._note_started()
        # An explicitly completed count is authoritative when present.
        if match := self._completed_points.search(line):
            self._emit_progress(
                int(match.group("completed")), int(match.group("total"))
            )
            return
        # Native DOS prints '34 E= ... IT= 1 Ti' for each atomic component.
        # When the index changes to a higher energy, only the previous
        # observed energy is known to have finished. Never count IT blocks.
        match = self._energy_header.match(line)
        if match is None:
            return
        point = int(match.group("point"))
        previous = self._current_point
        if previous is not None and point > previous:
            self._emit_progress(previous, self.total_points)
        if previous is None or point != previous:
            self._current_point = point

    async def read_output(self, stdout, result):
        await self.read_commons(stdout, result)
        if self.total_points is None:
            self.total_points = dos_total_points_from_parameters(getattr(result, "input_parameters", None))
        self._note_started()
        result.output_lines = []
        async for raw_line in stdout:
            line = raw_line.decode("utf8", errors="replace").rstrip()
            result.output_lines.append(line)
            self._consume_line(line)
        return result


class DosProgressParser(DosOutputParser):
    """Incrementally parse DOS output chunks (including attached-job replays)."""

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


class DosResult(TaskResult):
    """Density of states (DOS) results provides access to the computed
    density of states in the DOS output file, using the :py:attr:dos property."""

    @cached_property
    def dos_filename(self):
        """Density of states file name"""
        fname = dataset_adsi(self.input_parameters) + ".dos"
        if self.directory:
            fname = os.path.join(self.directory, fname)
        return fname

    @cached_property
    def dos(self):
        """The computed density of states."""
        return self._dos_file.parsed_file()

    @cached_property
    def _dos_file(self):
        try:
            return self.files.set_file_type("DOS", "dos")
        except KeyError as exc:
            raise FileNotFoundError("DOS result file is not listed in the SPR-KKR output.") from exc

    @property
    def output_values(self):
        # The file may be absent in an incomplete or imported calculation.
        # Do not hide the whole result summary when the DOS file is missing.
        if "DOS" in self.files:
            self._dos_file
        return super().output_values


class DosOutputReader(KkrOutputReader):
    """Read DOS output and provide the parsed density-of-states result."""

    result_class = DosResult
    parser_class = DosOutputParser

    @classmethod
    def matches_output(cls, output):
        """Recognize legacy DOS output from its ``FILES`` section."""
        lines = output.splitlines()
        for index, line in enumerate(lines):
            if line.strip() != b"FILES:":
                continue
            started = False
            for file_line in lines[index + 1:]:
                file_line = file_line.strip()
                if not file_line:
                    if started:
                        return False
                    continue
                started = True
                name, separator, _ = file_line.partition(b":")
                if separator and name.strip() == b"DOS":
                    return True
            return False
        return False
