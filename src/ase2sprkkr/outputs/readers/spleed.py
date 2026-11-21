"""SPLEED INFO parser for native KKRSPEC I(E) diagrams.

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


class SpleedOutputParser(SpecOutputParser):
    """Observe the first native I(E) diagram, including streamed common lines.

    SPLOUT uses two 12-character Fortran D12.5 fields separated by two spaces.
    Other tables, including seven-column diagrams, are not mistaken for
    I(E) data. The existing SpecResult parser retains the complete spectrum.
    """

    def __init__(self, print_info=False, on_info=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self._started = False
        self._in_diagram = False
        self._diagram_seen = False
        self._diagram_complete = False
        self._points = 0
        self._last_point = None
        self._last_reported = 0
        self._finished = False

    @staticmethod
    def format_info(kind, data):
        if kind == "spleed_started":
            return "Preparing SPLEED calculation..."
        if kind == "spleed_diagram":
            return f"SPLEED diagram: {data}"
        if kind == "spleed_point":
            return f"SPLEED I(E): {data.count} points read (E = {data.energy_ev:g} eV)"
        if kind == "spleed_diagram_complete":
            return f"SPLEED I(E) diagram: {data} points read"
        if kind == "spleed_finished":
            return "SPLEED calculation finished"
        return None

    def _complete_diagram(self):
        if not self._diagram_complete and self._points:
            self._diagram_complete = True
            self._in_diagram = False
            self.emit_info("spleed_diagram_complete", self._points)

    @classmethod
    def _parse_row(cls, text):
        # Use the exact native SPLOUT record width, same as SpecResult.
        if len(text) != 26 or text[12:14] != "  ":
            return None
        first, second = text[:12], text[14:]
        if not (cls._D_NUMBER.fullmatch(first) and cls._D_NUMBER.fullmatch(second)):
            return None
        return (float(first.replace("D", "E").replace("d", "e")),
                float(second.replace("D", "E").replace("d", "e")))

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
            if not self._diagram_seen:
                self._diagram_seen = True
                self.emit_info("spleed_diagram", text.strip())
            return

        if not self._in_diagram:
            return

        row = self._parse_row(text)
        if row is not None:
            self._points += 1
            self._last_point = SpleedPointInfo(self._points, *row)
            # Milestones rather than one GUI event per numerical record.
            if self._points == 1 or self._points % 25 == 0:
                self._last_reported = self._points
                self.emit_info("spleed_point", self._last_point)
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
