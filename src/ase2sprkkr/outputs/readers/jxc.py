"""The reader for JXC task, providing exchange couplings, exchange tensors, and Dzyaloshinski-Moriya vectors."""

import warnings
import re
from dataclasses import dataclass

from ..task_result import TaskResult, KkrOutputReader
from ..sprkkr_output_reader import SprKkrOutputParser
from ...common.decorators import cached_property
from ...bindings.uppasd import write_pos_file, write_mom_file, write_inpsd_file
from ...common.process_output_reader import readline_until


@dataclass(frozen=True)
class JxcPairCountInfo:
    pairs: int


@dataclass(frozen=True)
class JxcOutputFileInfo:
    file_type: str
    filename: str


@dataclass(frozen=True)
class JxcCurieTemperatureInfo:
    temperature_kelvin: float


class JxcResult(TaskResult):
    """JXC task provides access to the generated JXC output file.
    It provides access to exchange couplings, exchange tensors, and Dzyaloshinski-Moriya vectors,
    using the :py:attr:`~jxc`, :py:attr:`~dij` and :py:attr:`~dmi` properties.
    Not all of them may be present in the output, depending on the settings of the JXC task.
    """

    @cached_property
    def dmi_filename(self):
        """Dzyaloshinski-Moriya vectors file name"""
        try:
            return self.path_to("dmi")
        except KeyError:
            raise AttributeError("Dzyaloshinski-Moriya vectors file not found in output")

    @cached_property
    def dmi(self):
        """Dzyaloshinski-Moriya vectors."""
        return self.output_values["dmi"].parsed_file()

    @cached_property
    def dij_filename(self):
        """Exchange tensors file name"""
        try:
            return self.path_to("dij")
        except KeyError:
            raise AttributeError("Exchange tensors file not found in output")

    @cached_property
    def dij(self):
        """Exchange tensors."""
        return self.output_values["dij"].parsed_file()

    @cached_property
    def jxc_filename(self):
        """Exchange couplings file name"""
        try:
            return self.path_to("jxc")
        except KeyError:
            raise AttributeError("Exchange couplings file not found in output")

    @cached_property
    def jxc(self):
        """Exchange couplings."""
        return self.output_values["jxc"].parsed_file()

    @property
    def output_values(self):
        files = {"jxc": "Exchange couplings", "dij": "Exchange tensors", "dmi": "Dzyaloshinski-Moriya vectors"}
        for key in files:
            if key in self.files:
                self.files.set_file_type(key, "jxc")
        return super().output_values

    def write_uppasd_files(
        self,
        directory=None,
        jxc_file="jfile.dat",
        dmi_file="dmfile.dat",
        pos_file="posfile.dat",
        mom_file="momfile.dat",
        inpsd_file="inpsd.dat",
        **kwargs,
    ):
        """Write the computed exchange couplings, exchange tensors and Dzyaloshinski-Moriya vectors
        to UppASD input files in the specified directory."""
        output_file = None
        args = {"dmi": "biqdm", "jxc": "exchange"}
        for key in ["jxc", "dmi"]:
            if key in self.files:
                file_name = locals()[key + "_file"]
                if file_name is False:
                    continue
                output_file = self.output_values[key].parsed_file()
                output_file.write_uppasd_file(file_name=file_name, directory=directory)
                argname = args[key]
                if argname in kwargs:
                    warnings.warn(
                        f"Option `{argname}` already present in the kwargs. I have no place to wrute the name of {key} file into inpsd.dat"
                    )
                else:
                    kwargs[argname] = file_name

        if pos_file is not False:
            write_pos_file(self.atoms, pos_file, jxc_ouput_file=output_file, directory=directory)
        if mom_file is not False:
            write_mom_file(self.atoms, mom_file, jxc_ouput_file=output_file, directory=directory)
        if inpsd_file is not False:
            write_inpsd_file(self.atoms, posfile=pos_file, momfile=mom_file, **kwargs)

    def run_uppasd(
        self,
        executable="uppasd",
        directory=None,
        jxc_file="jfile.dat",
        dmi_file="dmfile.dat",
        pos_file="posfile.dat",
        mom_file="momfile.dat",
        inpsd_file="inpsd.dat",
        **kwargs,
    ):
        self.write_uppasd_files(
            **{k: v for k, v in locals().items() if k not in ("self", "kwargs", "executable")}, **kwargs
        )
        import subprocess
        from ...common.file_utils import Directory

        with Directory(directory).chdir():
            subprocess.run([executable])


class JxcOutputParser(SprKkrOutputParser):
    """Emit observed JXC milestones, without inventing fractional progress."""

    # SPR-KKR 9.7: xcpljij.f, xcpltensor.f, cluxcpljij.f,
    # cluxcpltensor.f, init_mod_tauij_{star,cluster}.f.
    _file_line = re.compile(
        rb"^\s*results written to file:\s*(?P<file>\S+)\s*$",
        re.IGNORECASE,
    )
    _plot_line = re.compile(
        rb"^\s*(?:exchange coupling parameters|exchange tensor elements)"
        rb"\s+written to file\s+(?P<file>\S+)\s*$",
        re.IGNORECASE,
    )
    _curie_line = re.compile(
        rb"Curie temperature within mean field\s+approximation"
        rb"\s+T_C\s*=\s*(?P<temperature>[+-]?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)\s*K",
        re.IGNORECASE,
    )
    _pair_count_line = re.compile(
        rb"number of TAU\(I,J\)'s to be calculated:\s*(?P<count>\d+)",
        re.IGNORECASE,
    )

    @staticmethod
    def _file_kind(filename):
        """Classify only the native filenames actually written by these modes."""
        name = filename.lower()
        if name.endswith((b"_j_ij.dat", b"_jjij.dat", b"_xcplten_jij.dat")):
            return "jxc"
        if name.endswith(b"_xcplten_dij.dat"):
            return "dij"
        if name.endswith(b"_dmivec_dij.dat"):
            return "dmi"
        for component in ("x", "y", "z"):
            if name.endswith(("_dij_" + component + ".dat").encode()):
                return "dmi_" + component
        return None

    def format_info(self, kind, data):
        if kind == "jxc_pair_count":
            return f"Calculating {data.pairs} TAU(I,J) pairs"
        if kind == "jxc_result_file":
            return f"{data.file_type.upper()} results written to {data.filename}"
        if kind == "jxc_plot_file":
            return f"JXC plot written to {data}"
        if kind == "jxc_curie_temperature":
            return f"Mean-field Curie temperature: {data.temperature_kelvin:g} K"
        return None

    def _consume_line(self, line, result=None):
        if match := self._pair_count_line.search(line):
            pairs = int(match.group("count"))
            if result is not None:
                result.tau_pair_count = pairs
            self.emit_info("jxc_pair_count", JxcPairCountInfo(pairs))
            return

        if match := self._curie_line.search(line):
            try:
                temperature = float(match.group("temperature").replace(b"D", b"E").replace(b"d", b"e"))
            except ValueError:
                return
            if result is not None:
                result.mean_field_curie_temperature = temperature
            self.emit_info("jxc_curie_temperature", JxcCurieTemperatureInfo(temperature))
            return

        if match := self._file_line.match(line):
            filename = match.group("file").decode("utf8", errors="replace")
            file_type = self._file_kind(match.group("file"))
            if file_type is None:
                # A different supported JXC variant may write other files.
                # Unrecognized names must not be treated as parser errors.
                return
            if result is not None:
                result.files.add_file(file_type, filename, "jxc")
            self.emit_info("jxc_result_file", JxcOutputFileInfo(file_type, filename))
            return

        if match := self._plot_line.match(line):
            self.emit_info(
                "jxc_plot_file",
                match.group("file").decode("utf8", errors="replace"),
            )

    async def read_output(self, stdout, result):
        await self.read_commons(stdout, result)
        result.mean_field_curie_temperature = None
        result.tau_pair_count = None
        async for line in stdout:
            self._consume_line(line, result)
        # A missing T_C is legal: e.g. CLUXCPLJIJ omits it and XCPLTENSOR
        # prints it only when DGEEV succeeds. Do not warn on valid output.


class JxcProgressParser(JxcOutputParser):
    """Incremental JXC INFO reader for raw output chunks and remote replays."""

    def __init__(self, on_info=None, print_info=False):
        super().__init__(print_info=print_info, on_info=on_info)
        self._partial = ""

    def feed(self, text):
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            self._consume_line(line.encode("utf8"))

    def finish(self):
        if self._partial:
            self._consume_line(self._partial.encode("utf8"))
            self._partial = ""


class JxcOutputReader(KkrOutputReader):
    result_class = JxcResult
    parser_class = JxcOutputParser

    @classmethod
    def matches_output(cls, output):
        """Recognize legacy JXC output that predates a useful task marker."""
        return b"Components of Dzyaloshinski-Moriya vector Dij" in output
