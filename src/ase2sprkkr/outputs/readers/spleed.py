"""SPLEED INFO parser for native KKRSPEC two-column diagrams.

The numerical spectrum remains owned by SpecResult; this parser reports
observed milestones only and never guesses a percentage from ENERGY.NE.
"""

from __future__ import annotations

from dataclasses import dataclass

from .spec import SpecOutputParser, SpecOutputReader


@dataclass(frozen=True)
class SpleedPointInfo:
    """A parsed, complete record of the first two-column SPLOUT diagram."""

    count: int
    energy_ev: float
    intensity: float
    diagram: str


@dataclass(frozen=True)
class SpleedDiagramInfo:
    """Type and number of two-column records observed in the native output."""

    diagram: str
    points: int


class SpleedOutputParser(SpecOutputParser):
    """Observe the first native two-column diagram, including common lines.

    SPLOUT uses two 12-character Fortran D12.5 fields separated by two spaces.
    Other tables, including seven-column diagrams, are not mistaken for
    numerical data. The existing SpecResult parser retains the complete output.
    """

    def __init__(self, print_info=False, on_info=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self._started = False
        self._in_diagram = False
        self._diagram_seen = False
        self._diagram_complete = False
        self._points = 0
        self._diagram_name = None
        self._finished = False

    @staticmethod
    def format_info(kind, data):
        if kind == "spleed_started":
            return "Preparing SPLEED calculation..."
        if kind == "spleed_diagram":
            return f"SPLEED diagram: {data}"
        if kind == "spleed_point":
            coordinate = f"E = {data.energy_ev:g} eV"
            quantity = "point" if data.count == 1 else "points"
            return (f"SPLEED {data.diagram}: {data.count} {quantity} read "
                    f"({coordinate}, I = {data.intensity:g})")
        if kind == "spleed_diagram_complete":
            return f"SPLEED {data.diagram}: {data.points} points read"
        if kind == "spleed_finished":
            return "SPLEED calculation finished"
        return None

    def _complete_diagram(self):
        if not self._diagram_complete and self._points:
            self._diagram_complete = True
            self._in_diagram = False
            self.emit_info("spleed_diagram_complete",
                           SpleedDiagramInfo(self._diagram_name, self._points))

    def observe_line(self, line):
        text = line.rstrip("\r\n")
        if not self._started and text.strip():
            self._started = True
            self.emit_info("spleed_started", None)

        # Completion is an explicit native marker, not inferred from EOF.
        if "program stopped regularly via STOP_REGULAR" in text:
            self._complete_diagram()
            if not self._finished:
                self._finished = True
                self.emit_info("spleed_finished", None)
            return

        if self._diagram_complete:
            return

        if self._DIAGRAM.fullmatch(text.strip()):
            if self._points:
                self._complete_diagram()
                return
            self._in_diagram = True
            self._diagram_name = text.strip()
            if not self._diagram_seen:
                self._diagram_seen = True
                self.emit_info("spleed_diagram", self._diagram_name)
            return

        row = self.parse_spleed_diagram_row(text)
        if not self._in_diagram:
            if row is None:
                return
            # Native WRITE(*,99004/99005) has no preceding diagram header;
            # the header is written to NOUT1 = IFILSPECOU3 (SPEC.out).
            self._in_diagram = True
            self._diagram_name = "spectral output"
            self._diagram_seen = True
            self.emit_info("spleed_diagram", self._diagram_name)

        if row is not None:
            self._points += 1
            # Milestones rather than one GUI event per numerical record.
            if self._points == 1 or self._points % 25 == 0:
                self.emit_info("spleed_point",
                               SpleedPointInfo(self._points, *row, self._diagram_name))
        elif text.strip():
            self._complete_diagram()

    async def read_output(self, stdout, result):
        result = await super().read_output(stdout, result)
        result.spleed_info_points = self._points
        return result


class SpleedProgressParser(SpleedOutputParser):
    """Consume arbitrary stdout chunks, including attached-run replays."""

    def __init__(self, on_info=None, print_info=False):
        super().__init__(print_info=print_info, on_info=on_info)
        self._partial = ""

    def feed(self, text):
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self.observe_line(line)

    def finish(self):
        if self._partial:
            self.observe_line(self._partial)
            self._partial = ""


class SpleedOutputReader(SpecOutputReader):
    """Keep the SPEC result and dedicated SPLEED INFO parsing together."""

    parser_class = SpleedOutputParser
