"""Structured seven-column relativistic SPLEED data from KKRSPEC.

The columns are taken from SPLOUT in SPR-KKR 9.7 (spec_rslabsub.f):
incident energy, angle, diffraction intensity, normalized spin asymmetry,
and three components of the outgoing spin polarization.

SPLOUT's two-column diagnostics go to NOUT1 (stdout), not this SPC file.
"""

import numpy as np

from .._plot_support import Multiplot
from ..result_files_definitions import (
    ResultFileDefinition,
    ResultFileValueDefinition as V,
    create_result_file_definition,
)
from ..result_files import CommonResultFile
from ...common.grammar_types import NumpyArray, Prefixed
from ...common.generated_configuration_definitions import NumpyViewDefinition as NV


class SPLEEDSPCResultFile(CommonResultFile):
    """Relativistic SPLEED diffraction intensities and spin polarization."""

    def plot(
        self, *, filename=None, show=None, axis=None, figsize=(8, 8),
        dpi=600, latex=None, **kwargs,
    ):
        """Plot native SPLOUT intensity, asymmetry and outgoing polarization.

        Fixed-angle SPLEED varies incident energy. If instead energy is
        constant, use theta as the abscissa. An explicitly supplied axis
        receives the primary intensity curve.
        """
        data = np.asarray(self.RAW_DATA())
        if data.ndim != 2 or data.shape[1] != 7 or len(data) == 0:
            raise ValueError("SPLEED SPC has no valid seven-column numerical data")

        energy, theta = data[:, 0], data[:, 1]
        if np.ptp(energy) > 0:
            x, xlabel = energy, "Incident energy (eV)"
        elif np.ptp(theta) > 0:
            x, xlabel = theta, "Angle (degrees)"
        else:
            x, xlabel = np.arange(len(data)), "Sample index"

        if axis is not None:
            axis.plot(x, data[:, 2], label="RINTG(G)")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Diffraction intensity")
            return axis

        panels = (
            ("Diffraction intensity", (("RINTG(G)", 2),)),
            ("Spin asymmetry", (("ASYMM(G)", 3),)),
            ("Outgoing spin polarization", (
                ("P(1,G)", 4), ("P(2,G)", 5), ("P(3,G)", 6),
            )),
        )
        with Multiplot(
            layout=(3, 1), number_of_plots=len(panels), figsize=figsize,
            filename=filename, show=show, dpi=dpi, latex=latex, **kwargs,
        ) as mp:
            for title, columns in panels:
                with mp.new_axis(name=title) as (ax, _):
                    for label, index in columns:
                        ax.plot(x, data[:, index], label=label)
                    ax.set_title(title)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(title)
                    if len(columns) > 1:
                        ax.legend(loc="best")
        return mp.figure


class SPLEEDSPCDefinition(ResultFileDefinition):
    result_class = SPLEEDSPCResultFile

    @property
    def extension(self):
        return "spc"


def column(index):
    return slice(None), index


definition = create_result_file_definition(
    "SPLEED",
    [
        V("NT", int),
        V("NP", int),
        V("COMMENT", Prefixed("#"), name_in_grammar=False),
        V(
            "RAW_DATA",
            NumpyArray(
                float,
                shape=(-1, 7),
                written_shape=(-1, 7),
                trivial_parsing=True,
            ),
            name_in_grammar=False,
        ),
        NV("ENERGY", "RAW_DATA", column(0), info="Incident electron kinetic energy (eV)"),
        NV("THETA", "RAW_DATA", column(1), info="Electron angle (degrees)"),
        NV("INTENSITY", "RAW_DATA", column(2), info="Diffracted beam intensity (RINTG)"),
        NV(
            "ASYMMETRY", "RAW_DATA", column(3),
            info="Spin asymmetry (I+ - I-)/(I+ + I-) divided by |P|",
        ),
        NV("P_X", "RAW_DATA", column(4), info="Outgoing spin polarization x"),
        NV("P_Y", "RAW_DATA", column(5), info="Outgoing spin polarization y"),
        NV("P_Z", "RAW_DATA", column(6), info="Outgoing spin polarization z"),
    ],
    name="SPLEED_SPC",
    cls=SPLEEDSPCDefinition,
    info="Seven-column relativistic KKRSPEC SPLEED diffraction output (SPLOUT).",
)
