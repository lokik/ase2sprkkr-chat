"""Matplotlib rendering of Grace projects through ASE2SPRKKR plotting helpers.

Numeric parsing remains independent of the plotting backend.
"""
from __future__ import annotations

import re

import numpy as np


# Standard XMGrace color palette (zero-based indices).
_COLORS = (
    "white", "black", "red", "green", "blue", "yellow", "brown", "grey",
    "violet", "cyan", "magenta", "orange", "indigo", "maroon", "turquoise", "darkgreen",
)
_GREEK = {"D": "Δ", "m": "μ", "a": "α", "b": "β", "g": "γ", "d": "δ", "e": "ε",
          "t": "τ", "p": "π", "s": "σ", "w": "ω", "G": "Γ", "S": "Σ", "O": "Ω"}
_LINESTYLES = {0: "-", 1: "-", 2: "--", 3: ":", 4: "-.", 5: "--"}
_SYMBOLS = {1: "o", 2: "s", 3: "D", 4: "^", 5: "v", 6: "<", 7: ">", 8: "x", 9: "+"}


def grace_text(value):
    """Convert common Grace label escapes into Matplotlib-readable labels.

    Complex font, positioning, and special-symbol commands are not intended
    to reproduce every aspect of XMGrace's typography.
    """
    if not value:
        return ""
    value = re.sub(r"\\x([A-Za-z])", lambda m: _GREEK.get(m[1], m[1]), value)
    value = re.sub(r"\\[fcmMFC]\{[^}]*\}", "", value)
    result, opened, index = [], False, 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value):
            command = value[index + 1]
            if command in ("s", "S"):
                if opened:
                    result.append("}$")
                result.append("$_{" if command == "s" else "$^{")
                opened = True
                index += 2
                continue
            if command == "N":
                if opened:
                    result.append("}$")
                    opened = False
                index += 2
                continue
            if command == "\\":
                result.append("\\")
                index += 2
                continue
            if command in ("m", "M", "f", "F", "c", "C", "u", "U", "b", "B"):
                index += 2
                continue
        result.append(value[index])
        index += 1
    if opened:
        result.append("}$")
    return "".join(result)


def _color(settings, prefix="line"):
    value = settings.get(f"{prefix} color")
    if value is None:
        return None
    try:
        return _COLORS[int(value)]
    except (ValueError, IndexError):
        return value


def _errors(dataset):
    """Return x and y errors for symmetric/asymmetric error-bar datasets."""
    columns, data = dataset.columns, dataset.data

    def get(name):
        return data[:, columns.index(name)] if name in columns else None

    xerr, yerr = get("dx"), get("dy")
    if "dx_left" in columns:
        xerr = np.array([get("dx_left"), get("dx_right")])
    if "dy_lower" in columns:
        yerr = np.array([get("dy_lower"), get("dy_upper")])
    return xerr, yerr


def plot_grace_dataset(dataset, axis, *, colormap=None, norm=None, **kwargs):
    """Render one Grace dataset onto an existing Matplotlib axis."""
    from matplotlib.colors import LogNorm

    if dataset.data.size == 0 or dataset.settings.get("hidden", "false").lower() in ("true", "on"):
        return None
    data = dataset.data
    x, y = data[:, 0], data[:, 1]
    style = {}
    color = _color(dataset.settings)
    if color is not None:
        style["color"] = color
    if "line linewidth" in dataset.settings:
        style["linewidth"] = float(dataset.settings["line linewidth"])
    if "line linestyle" in dataset.settings:
        style["linestyle"] = _LINESTYLES.get(int(dataset.settings["line linestyle"]), "-")
    marker = dataset.settings.get("symbol")
    if marker is not None:
        style["marker"] = _SYMBOLS.get(int(marker))
    style.update(kwargs)
    if dataset.legend:
        style.setdefault("label", grace_text(dataset.legend))

    xerr, yerr = _errors(dataset)
    if dataset.type.startswith("bar"):
        style.pop("linestyle", None)
        style.pop("marker", None)
        return axis.bar(x, y, yerr=yerr, **style)
    if dataset.type == "xyvmap":
        for key in ("linestyle", "marker", "linewidth"):
            style.pop(key, None)
        return axis.quiver(x, y, data[:, 2], data[:, 3],
                           angles="xy", scale_units="xy", scale=1, **style)
    if dataset.type in ("xyz", "xycolor", "xysize"):
        for key in ("linestyle", "marker", "linewidth"):
            style.pop(key, None)
        if norm == "log":
            norm = LogNorm()
        elif norm == "lin":
            norm = None
        if dataset.type == "xysize":
            style["s"] = data[:, 2]
        else:
            style["c"] = data[:, 2]
            if colormap is not None:
                style["cmap"] = colormap
            if norm is not None:
                style["norm"] = norm
            style.pop("color", None)
        return axis.scatter(x, y, **style)
    if xerr is not None or yerr is not None:
        return axis.errorbar(x, y, xerr=xerr, yerr=yerr, **style)
    return axis.plot(x, y, **style)


def plot_grace_graph(graph, axis, *, datasets=None, colormap=None, norm=None,
                     title=None, xlabel=None, ylabel=None, **kwargs):
    """Render one Grace graph, including its labels, bounds, and legends."""
    selected = set(datasets) if datasets is not None else None
    for dataset in graph.datasets.values():
        if selected is None or dataset.name in selected:
            plot_grace_dataset(dataset, axis, colormap=colormap, norm=norm, **kwargs)
    if graph.title or title:
        axis.set_title(grace_text(graph.title) if title is None else title)
    if graph.x_label or xlabel:
        axis.set_xlabel(grace_text(graph.x_label) if xlabel is None else xlabel)
    if graph.y_label or ylabel:
        axis.set_ylabel(grace_text(graph.y_label) if ylabel is None else ylabel)
    if graph.world:
        xmin, xmax, ymin, ymax = graph.world
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
    if graph.settings.get("xaxis ticklabel") == "off":
        axis.tick_params(axis="x", labelbottom=False)
    if graph.settings.get("yaxis ticklabel") == "off":
        axis.tick_params(axis="y", labelleft=False)
    _, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend()
    return axis


def plot_grace(document, *, filename=None, show=None, figsize=(6, 4), dpi=600,
               latex=None, axis=None, datasets=None, separate_plots=False, **kwargs):
    """Plot a Grace project with ASE2SPRKKR's existing plot lifecycle.

    datasets can select G0.S0-style identifiers. The original normalized
    viewports are used when each selected graph specifies one; otherwise the
    regular Multiplot layout is used.
    """
    from ase2sprkkr.gui.plot import Multiplot, create_rc_context, finish_plot
    import matplotlib.pyplot as plt

    selected = set(datasets) if datasets is not None else None
    if selected is not None:
        unknown = selected.difference(document.datasets)
        if unknown:
            raise ValueError(f"Unknown Grace dataset(s): {', '.join(sorted(unknown))}")
    graphs = [
        graph for graph in document.graphs.values()
        if graph.settings.get("enabled", "on") != "off"
        and any((selected is None or item.name in selected) and item.data.size
                for item in graph.datasets.values())
    ]
    if not graphs:
        raise ValueError("No plottable datasets in Grace file")

    if axis is not None:
        if len(graphs) != 1:
            raise ValueError("An existing Matplotlib axis requires one selected Grace graph")
        plot_grace_graph(graphs[0], axis, datasets=selected, **kwargs)
        return axis.figure

    if not separate_plots and all(graph.view is not None for graph in graphs):
        with create_rc_context(latex):
            figure = plt.figure(figsize=figsize)
            for graph in graphs:
                xmin, xmax, ymin, ymax = graph.view
                if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                    raise ValueError(f"Invalid Grace viewport for G{graph.index}: {graph.view}")
                ax = figure.add_axes([xmin, ymin, xmax - xmin, ymax - ymin])
                plot_grace_graph(graph, ax, datasets=selected, **kwargs)
            finish_plot(figure, filename=filename, show=show, dpi=dpi)
            return figure

    with Multiplot(number_of_plots=len(graphs), layout=min(len(graphs), 2),
                   filename=filename, show=show, figsize=figsize, dpi=dpi,
                   latex=latex, separate_plots=separate_plots) as mp:
        for graph in graphs:
            with mp.new_axis(name=f"G{graph.index}") as (ax, unused):
                plot_grace_graph(graph, ax, datasets=selected, **kwargs)
    return mp.figure
