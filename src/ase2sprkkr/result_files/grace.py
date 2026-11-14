"""Read Grace/XMGrace .agr projects without a plotting dependency.

The parser preserves graph/set identity, numeric data, common captions and
layout parameters, and the original configuration commands. Grace's text
markup is deliberately kept intact; it is not Matplotlib/TeX markup.
"""

from dataclasses import dataclass, field
from os import PathLike
import re
from typing import Dict, List

import numpy as np


# Number and order of numeric columns for standard Grace dataset types.
_TYPE_COLUMNS = {
    "xy": ("x", "y"),
    "xydx": ("x", "y", "dx"),
    "xydy": ("x", "y", "dy"),
    "xydxdx": ("x", "y", "dx_left", "dx_right"),
    "xydydy": ("x", "y", "dy_lower", "dy_upper"),
    "xydxdy": ("x", "y", "dx", "dy"),
    "xydxdxdydy": ("x", "y", "dx_left", "dx_right", "dy_lower", "dy_upper"),
    "xyz": ("x", "y", "z"),
    "xysize": ("x", "y", "size"),
    "xycolor": ("x", "y", "color"),
    "xyvmap": ("x", "y", "dx", "dy"),
    "bar": ("x", "y"),
    "bardy": ("x", "y", "dy"),
    "bardydy": ("x", "y", "dy_lower", "dy_upper"),
}
_TARGET = re.compile(r"target\s+G(\d+)\.S(\d+)\s*$", re.IGNORECASE)
_WITH = re.compile(r"with\s+g(\d+)\s*$", re.IGNORECASE)
_GRAPH = re.compile(r"g(\d+)\s+(on|off)\s*$", re.IGNORECASE)
_SET = re.compile(r"(?:g(\d+)\.)?s(\d+)\s+(.+)$", re.IGNORECASE)
_QUOTED = re.compile(r'^(.*?)\s+("(?:[^"\\]|\\.)*")\s*$')


class GraceParseError(ValueError):
    """Malformed or unsupported content in a Grace project."""


def _value(text):
    """Remove Grace's outer quotes, preserving its formatting escapes."""
    text = text.strip()
    return text[1:-1].replace(r'\"', '"') if len(text) >= 2 and text[0] == text[-1] == '"' else text


def _store_setting(settings, command):
    """Retain simple Grace attributes; full directives are also preserved."""
    quoted = _QUOTED.fullmatch(command)
    if quoted:
        key, value = quoted.groups()
        value = _value(value)
    elif command.lower().startswith("page size "):
        key, value = "page size", command[len("page size "):]
    else:
        key, _, value = command.rpartition(" ")
    if key:
        settings[" ".join(key.lower().split())] = value.strip()


@dataclass
class GraceDataset:
    """One Grace set (e.g. G1.S0), with data in file order."""

    graph_index: int
    index: int
    type: str = "xy"
    data: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    settings: Dict[str, str] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)

    @property
    def name(self):
        return f"G{self.graph_index}.S{self.index}"

    @property
    def columns(self):
        return _TYPE_COLUMNS[self.type]

    @property
    def legend(self):
        return self.settings.get("legend")

    def plot(self, axis, **kwargs):
        """Draw this set on an existing Matplotlib axis."""
        from .grace_plot import plot_grace_dataset
        return plot_grace_dataset(self, axis, **kwargs)


@dataclass
class GraceGraph:
    """A Grace graph, containing numbered datasets and plot metadata."""

    index: int
    datasets: Dict[int, GraceDataset] = field(default_factory=dict)
    settings: Dict[str, str] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)

    @property
    def title(self):
        return self.settings.get("title")

    @property
    def subtitle(self):
        return self.settings.get("subtitle")

    @property
    def x_label(self):
        return self.settings.get("xaxis label")

    @property
    def y_label(self):
        return self.settings.get("yaxis label")

    def _rectangle(self, prefix):
        keys = (f"{prefix} {part}" for part in ("xmin", "xmax", "ymin", "ymax"))
        values = [self.settings.get(key) for key in keys]
        return tuple(map(float, values)) if all(value is not None for value in values) else None

    @property
    def world(self):
        """(xmin, xmax, ymin, ymax) or None if unspecified."""
        return self._rectangle("world")

    @property
    def view(self):
        """Normalized viewport rectangle, if specified."""
        return self._rectangle("view")

    def plot(self, axis, **kwargs):
        """Draw this graph on an existing Matplotlib axis."""
        from .grace_plot import plot_grace_graph
        return plot_grace_graph(self, axis, **kwargs)


@dataclass
class GraceFile:
    """Parsed Grace file; no matplotlib import or dependency.

    Use GraceFile.from_file(path) or GraceFile.from_string(text).
    graphs[1].datasets[0].data retrieves the data for G1.S0.
    datasets is a mapping from G*.S* identifiers to datasets.
    """

    graphs: Dict[int, GraceGraph] = field(default_factory=dict)
    settings: Dict[str, str] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)

    @property
    def datasets(self):
        return {
            dataset.name: dataset
            for graph in self.graphs.values()
            for dataset in graph.datasets.values()
        }

    @property
    def data(self):
        """Mapping G*.S* to the underlying NumPy arrays."""
        return {name: dataset.data for name, dataset in self.datasets.items()}

    def plot(self, **kwargs):
        """Render the Grace project with ASE2SPRKKR's plotting helpers."""
        from .grace_plot import plot_grace
        return plot_grace(self, **kwargs)

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as stream:
            return cls._parse(stream)

    @classmethod
    def from_string(cls, source):
        from io import StringIO
        return cls._parse(StringIO(source))

    @classmethod
    def _parse(cls, stream):
        out = cls()
        current_graph = None
        active = None
        rows = []
        populated = set()

        def graph(index):
            if index not in out.graphs:
                out.graphs[index] = GraceGraph(index)
            return out.graphs[index]

        def dataset(g, s):
            sets = graph(g).datasets
            if s not in sets:
                sets[s] = GraceDataset(g, s)
            return sets[s]

        def finish():
            nonlocal active, rows
            if active is None:
                return
            cols = len(active.columns)
            active.data = np.asarray(rows, dtype=float).reshape((-1, cols))
            populated.add(active.name)
            active = None
            rows = []

        for line_no, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "&":
                if active is None:
                    raise GraceParseError(f"Line {line_no}: unexpected '&' without a dataset")
                finish()
                continue
            if line.startswith("@"):
                command = line[1:].strip()
                out.commands.append(command)
                match = _TARGET.fullmatch(command)
                if match:
                    if active is not None:
                        if rows:
                            raise GraceParseError(f"Line {line_no}: missing '&' before {command}")
                        finish()
                    g, s = map(int, match.groups())
                    active = dataset(g, s)
                    if active.name in populated:
                        raise GraceParseError(f"Line {line_no}: duplicate dataset {active.name}")
                    continue
                if command.lower().startswith("target"):
                    raise GraceParseError(f"Line {line_no}: invalid Grace target {command!r}")
                if command.lower().startswith("type "):
                    kind = command[5:].strip().lower()
                    if kind not in _TYPE_COLUMNS:
                        raise GraceParseError(f"Line {line_no}: unsupported Grace dataset type {kind!r}")
                    if active is None:
                        g = current_graph if current_graph is not None else 0
                        index = 0
                        while dataset(g, index).name in populated:
                            index += 1
                        active = dataset(g, index)
                    if rows:
                        raise GraceParseError(f"Line {line_no}: dataset type follows its data")
                    active.type = kind
                    continue
                if active is not None and rows:
                    raise GraceParseError(f"Line {line_no}: missing '&' before @{command}")
                match = _WITH.fullmatch(command)
                if match:
                    current_graph = int(match.group(1))
                    graph(current_graph)
                    continue
                match = _GRAPH.fullmatch(command)
                if match:
                    item = graph(int(match.group(1)))
                    item.settings["enabled"] = match.group(2).lower()
                    item.commands.append(command)
                    continue
                match = _SET.fullmatch(command)
                if match:
                    g = int(match.group(1)) if match.group(1) is not None else (
                        current_graph if current_graph is not None else 0
                    )
                    item = dataset(g, int(match.group(2)))
                    item.commands.append(command)
                    _store_setting(item.settings, match.group(3))
                    continue
                if current_graph is not None:
                    item = graph(current_graph)
                    item.commands.append(command)
                    _store_setting(item.settings, command)
                else:
                    _store_setting(out.settings, command)
                continue
            if active is None:
                raise GraceParseError(f"Line {line_no}: numeric data without @target or @type")
            try:
                values = [float(value.replace("D", "E").replace("d", "e")) for value in line.split()]
            except ValueError as exc:
                raise GraceParseError(f"Line {line_no}: invalid numeric data: {line!r}") from exc
            if len(values) != len(active.columns):
                raise GraceParseError(
                    f"Line {line_no}: {active.name} ({active.type}) requires "
                    f"{len(active.columns)} columns, found {len(values)}"
                )
            rows.append(values)

        finish()  # EOF without a final '&' is common and unambiguous.
        return out


def read_agr(source):
    """Read a Grace project from a path or an open text stream."""
    if isinstance(source, (str, PathLike)):
        return GraceFile.from_file(source)
    return GraceFile._parse(source)
