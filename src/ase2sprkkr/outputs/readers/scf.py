"""SCF (selfconsisten cycle) reader and result."""

import os
import re
from dataclasses import dataclass
from math import isfinite
import pyparsing as pp
import unyt

from ..output_definitions import (
    OutputSectionDefinition as Section,
    OutputValueDefinition as V,
    OutputValueEqualDefinition as VE,
    OutputNonameValueDefinition as VN,
)
from ...common.grammar_types import Table, integer, string, Real, RealWithUnits, String, Sequence, Array
from ..task_result import TaskResult
from ..result_options import DataOutputOption, create_output_section, generated_output
from ...common.process_output_reader import readline, readline_until
from ..sprkkr_output_reader import SprKkrOutputParser
from ...common.decorators import cached_property
from ...sprkkr.calculator import SPRKKR
from ...common.formats import fortran_format
from ...common.grammar import replace_whitechars
from ..task_result import KkrOutputReader


def plot_iterations(
    iterations, what=["error", ("energy", "ETOT"), ("energy", "EMIN")], filename=None, logscale=None, **kwargs
):

    import matplotlib.pyplot as pyplot

    if logscale is None:
        logscale = {"error"}
    if not isinstance(what, list):
        what = [what]

    fig, axs = pyplot.subplots(len(what), sharex=True, squeeze=False)
    iterations_ids = iterations.values_of("iteration")
    for name, ax in zip(what, axs.ravel()):
        data = iterations.values_of(name)
        if name in logscale:
            ax.set_yscale("log")
        kwargs.update({"ls": None, "mfc": None, "mew": 2, "ms": 6})
        ax.plot(iterations_ids, data, "+", **kwargs)
        ax.set_ylabel(iterations[0][name]._definition.name)
    axs[-1, 0].set_xlabel("Iteration")

    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    else:
        pyplot.show()


class RealOrStars(Real):
    """A real value, where ``****`` means ``NaN``"""

    _grammar = Real._grammar | replace_whitechars(pp.Word("*")).set_parse_action(lambda x: float("NaN"))


class ScfResult(TaskResult):
    """Objects of this class holds the results of a self-consistent cycle and its iterations.
    It also allows to plot the convergence of values during iterations.
    """

    produces_potential = True

    def _with_output_scf_status(self, potential):
        """Project the final SCF convergence onto this result's embedded state.

        Native CLUPOT can retain SCFSTATUS=ITERATING after its OUT converges.
        The CLUPOT and host may be shared with other results by ResultLoader;
        neither the cached native potential nor its atoms may be modified.
        This interpretation belongs only to a successful, parsed SCF result.
        """
        if (
            not self.is_embedded
            or getattr(self, "return_code", 0) != 0
            or not getattr(self, "iterations", None)
            or not self.converged
            or str(potential.SCF_INFO.SCFSTATUS()).strip().upper() != "ITERATING"
        ):
            return potential
        atoms = potential.atoms
        if atoms is None:
            return potential

        from ...sprkkr.sprkkr_atoms import SPRKKRAtoms

        # Detach option containers and site metadata without duplicating the
        # large radial arrays. The periodic embedding host remains shared.
        selected = (
            atoms.copy(shallow=True)
            if isinstance(atoms, SPRKKRAtoms)
            else atoms.copy()
        )
        selected.potential.SCF_INFO.SCFSTATUS = "CONVERGED"
        selected.info["sprkkr_embedding_scf_status"] = "CONVERGED"
        return selected.potential

    def _load_result_potential(self, loader=None):
        """Return the SCF potential with result-specific convergence applied."""
        return self._with_output_scf_status(super()._load_result_potential(loader))

    @cached_property
    def calculator(self):
        """The calculator that has the new (output) potential assigned - i.e. that
        can be used for further calculations with the (hopefully) converged wavefunctions etc."""
        potential = self.potential if self.is_embedded else self.potential_filename
        if self._calculator:
            return self._calculator.copy_with_potential(potential)
        return SPRKKR(potential=potential)

    @cached_property
    def sfn_filename(self):
        """Shape-function file reported by SPR-KKR, if any."""
        return self.path_to("SFN") if "SFN" in self.files else None

    @cached_property
    def sfn(self):
        """Shape functions used by a full-potential calculation."""
        sfn_file = self._sfn_file
        return sfn_file.parsed_file() if sfn_file is not None else None

    @cached_property
    def _sfn_file(self):
        return self.files.set_file_type("SFN", "sfn") if "SFN" in self.files else None

    @property
    def potential_filename(self):
        # The parsed output's explicit converged file takes precedence, even
        # when the original input is no longer available.
        if "converged" in self.files:
            return self.path_to("converged")
        return super().potential_filename + "_new"

    @property
    def start_potential_filename(self):
        return super().potential_filename

    @cached_property
    def start_potential(self):
        """Potential used as input for the SCF calculation."""

        self.require_current_file(self.start_potential_filename)
        return self._read_potential(self.start_potential_filename)

    @property
    def any_potential(self):
        """Return the parsed result potential, falling back to the input one."""

        for attribute in ("potential", "start_potential"):
            try:
                return getattr(self, attribute)
            except (AttributeError, KeyError, OSError, ValueError):
                pass
        return None

    @property
    def energy(self):
        """Total energy of the last iteration"""
        try:
            return self.last_iteration.energy.ETOT()
        except Exception as e:
            raise AttributeError("No iteration has been finished") from e

    @property
    def converged(self):
        """Whether the last completed SCF iteration reached convergence."""
        # Failed or incomplete runs may have no finished iterations.
        # Repeated sections are truthy even when empty, so check the count.
        return len(getattr(self, "iterations", ())) > 0 and self.last_iteration["converged"]()

    def iteration_values(self, name):
        """Return the array of values of given name from the
        iterations (i.e. "the column of the table of the values")

        E.g. result.iteration_values(('energy', 'ETOT')) return list of
        floats - the total energies computed in each iteration.

        Parameters
        ----------
        name: str
          Name of the parameter

        Return
        ------
        values: list
          The values.
        """
        if name not in self.iterations[0]:
            raise KeyError(f"No such iteration value: {name}")

        return self.iterations.values_of(name)

    @property
    def last_iteration(self):
        """Return the data of the last iteration"""
        if len(getattr(self, "iterations", ())) == 0:
            raise AttributeError("No iteration has been finished")
        return self.iterations[-1]

    @property
    def energies(self):
        return self.last_iteration.energy

    @cached_property
    def _scf_output_values(self):
        return create_output_section(
            "OUTPUT_VALUES",
            [
                generated_output(
                    "Converged",
                    lambda _section: str(self.converged),
                    info="Whether the final SCF iteration reached convergence.",
                ),
                generated_output(
                    "Number of iterations",
                    lambda _section: len(self.iterations),
                    info="Number of completed SCF iterations.",
                ),
                generated_output(
                    "Fermi energy",
                    lambda _section: self.last_iteration.energy.EF(),
                    plot=lambda _option, **kwargs: self.plot(("energy", "EF"), **kwargs),
                    info="Fermi energy at the final SCF iteration.",
                ),
                generated_output(
                    "Error",
                    lambda _section: self.last_iteration.error(),
                    plot=lambda _option, **kwargs: self.plot("error", **kwargs),
                    info="Convergence error at the final SCF iteration.",
                    display_name="Convergence error",
                ),
                generated_output(
                    "Data",
                    lambda _section: self,
                    result_class=DataOutputOption,
                    info="Complete parsed SCF result data.",
                    display_name="SCF result data",
                ),
            ],
        )

    @property
    def output_values(self):
        self._sfn_file
        output_values = super().output_values
        output_values.update(self._scf_output_values.items())
        return output_values

    def plot(self, what=["error", ("energy", "ETOT"), ("energy", "EMIN")], filename=None, logscale=None, **kwargs):
        """Plot the development of the given value(s) during iterations.

        Parameters
        ----------
        what: list[str]
          Names to be plotted

        filename: str or None
          If filename is given, the plot is rendered to the file, it is show on the screen
          otherwise.

        logscale: collections.abc.Set[str]
          Which values should be rendered in logscale

        **kwargs: dict
          All other arguments are passed to the matplotlib Axes.plot function
        """
        return plot_iterations(self.iterations, what, filename, logscale, **kwargs)


"""
This definition parses the part of the output file that contains the datas
about atoms and computed data associated with them.
"""
atomic_types_definition = Section(
    "atoms",
    [
        VN("IECURR", integer),
        VE("E", RealOrStars()),
        VN("L", float),
        VE("IT", integer),
        VN("symbol", string),
        VN(
            "orbitals",
            Table(
                {
                    "l": string,
                    "DOS": float,
                    "NOS": float,
                    "P_spin": float,
                    "m_spin": float,
                    "P_orb": float,
                    "m_orb": float,
                    "B_val": float,
                    "B_val_desc": String(default_value=""),
                    "B_core": Real(default_value=float("NaN"), nan=r"\*+"),
                },
                free_header=True,
                default_values=True,
            ),
        ),
        V(
            "magnetic_dipole",
            Table({"l": string, "P": float, "T_z": float}, free_header=True),
            written_name="P(T_z)", is_required=False,
        ),
        V("E_band", RealWithUnits(units={"[Ry]": unyt.Ry}), is_required=False),
        V("dipole moment", Sequence(int, Array(float, length=3)), is_required=False),
    ],
)


class PlottableValue(V):
    def enrich(self, option):
        def plot(options, **kwargs):
            c = option._container._container
            plot_iterations(option._container._container, self.get_path[len(c.get_path) :], **kwargs)

        option.plot = lambda **kwargs: plot(option, **kwargs)
        super().enrich(option)


PV = PlottableValue

"""
This definition is not used for parsing, but just for the data returned
from the reader.
"""
scf_section = Section(
    "iteration",
    [
        V("system_name", str),
        V("iteration", int, info="Number of the iteration."),
        PV("error", float),
        V("b_error", float, is_required=False, info="RMS error of the exchange-correlation B-field at this iteration."),
        V("duration", float, is_required=False, info="Execution time of this iteration in seconds."),
        V("converged", bool, info="True, if the SCF cycle converged this iteration."),
        Section("moment", [PV("spin", float), PV("orbital", float)]),
        Section(
            "energy",
            [
                PV(("EF", "fermi"), float, info="Fermi energy"),
                PV(("ETOT", "total"), float, info="Total energy"),
                PV(
                    ("EMIN", "band_states_min"),
                    float,
                    info="Bottom of energy contour for band states",
                    is_required=False,
                ),
                PV(("ESCBOT", "semi_core_min"), float, info="Lower limit for semi-core states", is_required=False),
                PV(("ECTOP", "core_max"), float, info="Upper limit for core states", is_required=False),
            ],
        ),
        Section("atomic_types", atomic_types_definition.members(), is_repeated=True),
    ],
    is_repeated=True,
)

del PV


@dataclass(frozen=True)
class ScfToleranceInfo:
    """Convergence threshold reported by native SPR-KKR."""

    tolerance: float


@dataclass(frozen=True)
class ScfIterationInfo:
    """Completed SCF cycle (fermi_energy in Ry, total_energy in eV)."""

    iteration: int
    error: float
    b_error: float
    fermi_energy: float
    spin_moment: float
    orbital_moment: float
    total_energy: float
    converged: bool

    @property
    def convergence_error(self):
        return max(abs(self.error), abs(self.b_error))

    @classmethod
    def from_output(cls, value):
        return cls(
            iteration=value["iteration"],
            error=value["error"],
            b_error=value["b_error"],
            fermi_energy=value["energy"]["EF"],
            spin_moment=value["moment"]["spin"],
            orbital_moment=value["moment"]["orbital"],
            total_energy=value["energy"]["ETOT"],
            converged=value["converged"],
        )


class _ScfInfoStream:
    """Observe lines discarded by the full SCF parser's section readers."""

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


class ScfOutputParser(SprKkrOutputParser):
    """
    This class reads and parses the output of the SCF task of the SPR-KKR.
    """

    _tolerance_line = re.compile(r"^\s*SCF-tol\s*:\s*(\S+)", re.IGNORECASE)

    def __init__(self, print_info=False, on_info=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self.tolerance = None

    def _note_tolerance(self, line, result=None):
        match = self._tolerance_line.match(line)
        if match is None:
            return False
        try:
            value = float(match.group(1).replace("D", "E").replace("d", "e"))
        except ValueError:
            return False
        if not isfinite(value) or value <= 0:
            return False
        if result is not None:
            result.scf_tolerance = value
        if self.tolerance != value:
            self.tolerance = value
            self.emit_info("scf_tolerance", ScfToleranceInfo(value))
        return True

    @classmethod
    def format_info(cls, kind, data):
        """Keep all INFO text formatting in ase2sprkkr."""
        if kind == "scf_tolerance":
            return f"SCF tolerance: {data.tolerance:g}"
        if kind == "scf_started":
            return f"SPRKKR-run for: {data}"
        if kind == "scf_iteration":
            return cls.iteration_info(data)
        if kind == "scf_finished":
            if data is None:
                return ("ERROR: No iteration has been finished. There is probably an "
                        "error in the input files, please, examine the SPR-KKR output file.")
            return ("OK: The computation converged." if data
                    else "WARNING: The computation does not converged!!!")
        return None

    @staticmethod
    def iteration_header(line):
        """Extract numerical values shared by full and incremental SCF parsing."""
        items = line.split()
        return {
            "iteration": int(items[0]),
            "error": float(items[2].replace("D", "E").replace("d", "e")),
            "b_error": float(items[3].replace("D", "E").replace("d", "e")),
            "energy": {"EF": float(items[5].replace("D", "E").replace("d", "e"))},
            "moment": {
                "spin": float(items[10].replace("D", "E").replace("d", "e")),
                "orbital": float(items[11].replace("D", "E").replace("d", "e")),
            },
        }

    @staticmethod
    def iteration_footer(line):
        """Extract total energy and convergence from the ETOT/SCF line."""
        items = line.split()
        return float((float(items[1].replace("D", "E")) * unyt.Ry).to(unyt.eV)), items[5] == "converged"

    @staticmethod
    def iteration_info(out):
        """The compact summary produced by the complete SCF parser."""
        v_error = fortran_format(out.error, ":>12e")
        b_error = fortran_format(out.b_error, ":>12e")
        return (
            f"Iteration {out.iteration:>5} RMS(V) {v_error} RMS(B) {b_error} "
            f"spin moment: {out.spin_moment:>13.6e} "
            f"orbital moment: {out.orbital_moment:>13.6e} "
        )


    _iteration_event_line = re.compile(
        rb"EMIN   = |^\s*from file:|shape function file not available:|"
        rb"SPRKKR-run for:"
    )

    async def read_output(self, stdout, result):
        result.scf_tolerance = None
        stdout = _ScfInfoStream(stdout, lambda line: self._note_tolerance(line, result))
        await self.read_commons(stdout, result)
        # For embedded calculations the converged file belongs to
        # POTFIL_CLU, not the host POTFIL reported in the FILES header.
        result.files.add_file("converged", result.potential_filename)
        iterations = scf_section.create_object()
        try:
            first = True
            _err_or_time = re.compile(rb"execution time for last iteration| ERR.*EF|SPRKKR-run for:")
            pending_line = None
            pending_energy = {}

            while True:
                out = {"energy": pending_energy}
                pending_energy = {}
                line = pending_line
                pending_line = None
                if line is None:
                    line = await readline_until(stdout, self._iteration_event_line.search)
                if not line:
                    break

                while "SPRKKR-run for:" not in line:
                    if "EMIN   = " in line:
                        out["energy"]["EMIN"] = float(line.split("=")[1])
                        line = await stdout.readline()
                        if b"ESCBOT" in line:
                            out["energy"]["ESCBOT"] = float(line.split(b"=")[1])
                            line = await stdout.readline()
                        if b"ECTOP" in line:
                            out["energy"]["ECTOP"] = float(line.split(b"=")[1])
                    elif match := self._missing_sfn_file.match(line):
                        self._register_sfn(result, match.group(1), True)
                    elif match := self._sfn_file.match(line):
                        self._register_sfn(result, match.group(1), False)

                    line = await readline_until(
                        stdout,
                        self._iteration_event_line.search,
                        can_end=False,
                    )
                line = line.strip()
                if first:
                    self.emit_info("scf_started", line.replace("SPRKKR-run for:", "").strip())
                    first = False
                run = line.replace("SPRKKR-run for:", "")
                out["system_name"] = run.strip()

                line = await readline_until(stdout, lambda line: b" E=" in line)
                if not line:
                    break
                atoms = []
                while True:
                    atoms.append(
                        await atomic_types_definition.parse_from_stream(
                            stdout,
                            up_to=b"\n -------------------------------------------------------------------------------",
                            start=line,
                        )
                    )
                    line = await readline_until(stdout, lambda line: line != b"\n")
                    if not line or "E=" not in line:
                        break
                out["atomic_types"] = atoms
                duration = None
                while line:
                    if _err_or_time.search(line.encode("utf8")):
                        if "execution time for last iteration" not in line:
                            break
                        try:
                            duration = float(line.split()[-2])
                        except (ValueError, IndexError):
                            pass
                    line = await readline_until(stdout, _err_or_time.search)
                if not line:
                    break
                if "SPRKKR-run for:" in line:
                    # CLUSCF can first print a host <GEN> summary without an
                    # SCF footer. Do not attach its atomic types to the later
                    # cluster iteration whose ERR/ETOT footer follows.
                    pending_line = line
                    pending_energy = out["energy"]
                    continue
                summary = self.iteration_header(line)
                out.update({key: value for key, value in summary.items() if key != "energy"})
                out["energy"].update(summary["energy"])
                if duration is not None:
                    out["duration"] = duration
                out["energy"]["ETOT"], out["converged"] = self.iteration_footer(await readline(stdout))
                section = iterations.add()
                section.set(out)

                self.emit_info("scf_iteration", ScfIterationInfo.from_output(out))

            self.emit_info(
                "scf_finished",
                iterations[-1]["converged"]() if iterations else None,
            )

            result.iterations = iterations

        except EOFError:
            raise Exception("The output ends unexpectedly")


class ScfProgressParser(ScfOutputParser):
    """Incrementally read *completed* iterations from a growing SCF output.

    This lightweight descendant uses the complete parser's numerical extraction
    and formatting without requiring atomic tables or a final EOF marker.
    """

    _iteration_summary = re.compile(r"^\s*\d+\s+ERR\s+")

    def __init__(self, on_info=None, print_info=False):
        super().__init__(print_info=print_info, on_info=on_info)
        self.last_iteration = None
        self._started = False
        self._pending = None
        self._line_buffer = ""

    def feed(self, text):
        self._line_buffer += text
        *lines, self._line_buffer = self._line_buffer.split("\n")
        for line in lines:
            self._consume(line)

    def finish(self):
        """Consume an unterminated final line when the source reaches EOF."""
        if self._line_buffer:
            self._consume(self._line_buffer)
            self._line_buffer = ""

    def _consume(self, line):
        if self._note_tolerance(line):
            return
        if not self._started and "SPRKKR-run for:" in line:
            self._started = True
            self.emit_info("scf_started", line.partition("SPRKKR-run for:")[2].strip())
        if self._iteration_summary.match(line):
            try:
                self._pending = self.iteration_header(line)
            except (ValueError, IndexError):
                self._pending = None
        elif self._pending is not None and re.match(r"^\s*ETOT\s+", line):
            try:
                energy, converged = self.iteration_footer(line)
            except (ValueError, IndexError):
                return
            self._pending["energy"]["ETOT"] = energy
            self._pending["converged"] = converged
            self.last_iteration = self._pending
            self._pending = None
            self.emit_info("scf_iteration", ScfIterationInfo.from_output(self.last_iteration))

    @classmethod
    def latest_from_file(cls, path):
        """Find the last complete cycle in a possibly growing output file.

        Read backwards in chunks, stopping once a completed cycle is found.
        A partial current iteration is never mistaken for a completed cycle.
        """
        with open(path, "rb") as source:
            source.seek(0, os.SEEK_END)
            offset = source.tell()
            collected = b""
            while offset:
                count = min(offset, 65536)
                offset -= count
                source.seek(offset)
                collected = source.read(count) + collected
                parser = cls()
                parser.feed(collected.decode("utf8", errors="replace"))
                if offset == 0:
                    parser.finish()
                if parser.last_iteration is not None:
                    return parser.last_iteration
        return None


class ScfOutputReader(KkrOutputReader):
    result_class = ScfResult
    parser_class = ScfOutputParser
