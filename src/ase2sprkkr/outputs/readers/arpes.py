"""The ARPES reader and result."""

from ..task_result import TaskResult, KkrOutputReader
from .default import DefaultOutputParser
from ...common.decorators import cached_property
import os
import warnings
import re
from dataclasses import dataclass


class ArpesResult(TaskResult):
    """ARPES result provides acces to the generated SPC output file
    containing the results of Angle resolved photoelectron electroscopy,
    using the :py:attr:`~spc` property."""

    @cached_property
    def spc_filenames(self):
        """Native SPC outputs, declared by the task or discovered alongside stdout."""
        parameters = self.input_parameters
        if parameters is not None:
            return tuple(
                os.fspath(path) for path in parameters.output_files(self.directory)
                if os.fspath(path).endswith(".spc") and os.path.isfile(path)
            )
        if self.output_file is None:
            return ()
        output = os.path.basename(os.fspath(self.output_file))
        if not output.endswith("_SPEC.out"):
            return ()
        dataset = output[:-len("_SPEC.out")]
        candidates = (
            os.path.join(self.directory, dataset + suffix)
            for suffix in ("_data.spc", "_data_pol1.spc", "_data_pol2.spc")
        )
        found = tuple(path for path in candidates if os.path.isfile(path))
        return found

    @cached_property
    def spc_filename(self):
        """First SPC output, retained as a convenience for older callers."""
        return next(iter(self.spc_filenames), None)

    @cached_property
    def spectra(self):
        """Tuple of parsed spectral files, including all photon polarizations.

        A single-polarization calculation still returns a one-element tuple.
        """
        return tuple(file.parsed_file() for file in self._register_spc_files())

    @property
    def spc(self):
        """Deprecated single-spectrum accessor; prefer :attr:`spectra`."""
        warnings.warn(
            "ArpesResult.spc is deprecated because a calculation can produce "
            "multiple polarization spectra. Use result.spectra (always a tuple) "
            "and select a spectrum explicitly, e.g. result.spectra[0].",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.spc_filenames:
            raise FileNotFoundError("No SPC output file could be identified")
        return self.spectra[0]

    @property
    def arpes(self):
        """Legacy alias for the first spectral file."""
        return self.spc

    @property
    def output_values(self):
        self._register_spc_files()
        return super().output_values

    def _register_spc_files(self):
        filenames = self.spc_filenames
        if not filenames:
            return ()
        entries = []
        task = self._task_name or (self.input_parameters.task_name if self.input_parameters is not None else None)
        file_type = {"band": "band_spc", "spleed": "spleed_spc"}.get(
            str(task).lower(), "spc"
        )
        for index, filename in enumerate(filenames):
            key = "SPC" if index == 0 else f"SPC_{index + 1}"
            if key not in self.files:
                entries.append(self.files.add_file(key, filename, file_type))
            else:
                entries.append(self.files.set_file_type(key, file_type))
        return tuple(entries)

    @cached_property
    def _spc_file(self):
        files = self._register_spc_files()
        if not files:
            raise FileNotFoundError("No SPC output file could be identified")
        return files[0]



@dataclass(frozen=True)
class ArpesBarrierInfo:
    """Potential-barrier parameter reported in native SPEC.out."""

    name: str
    value: float


@dataclass(frozen=True)
class ArpesPhotonInfo:
    """Photon wavevector in native SPEC.out units."""

    wavevector: tuple[float, float, float]


@dataclass(frozen=True)
class ArpesStokesInfo:
    """Stokes normalization and three polarization percentages."""

    s0: float
    components_percent: tuple[float, float, float]


class _ArpesHeaderStream:
    """Observe setup lines which the ordinary FILES reader consumes."""

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

    def __getattr__(self, name):
        return getattr(self.source, name)


class ArpesOutputParser(DefaultOutputParser):
    """INFO events for actual KKRSPEC setup, without fabricated progress.

    ARPES spectra are written to SPC files. A requested number of energy or
    k points, or incidental printed spectral rows, is not a completion count.
    """

    _NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
    _barrier = re.compile(
        rf"^\s*#\s*(?P<name>ibar|epsx)\s*:\s*(?P<value>{_NUMBER})\s*$",
        re.IGNORECASE,
    )
    _lattice = re.compile(
        rf"^\s*lattice constant\s*:\s*(?P<value>{_NUMBER})(?:\s|$)",
        re.IGNORECASE,
    )
    _photon = re.compile(
        rf"^\s*vacuum-wavevector of the photonfield\s*:\s*"
        rf"(?P<x>{_NUMBER})\s+(?P<y>{_NUMBER})\s+(?P<z>{_NUMBER})(?:\s|$)",
        re.IGNORECASE,
    )
    _stokes = re.compile(
        rf"^\s*stookes-vector\s+of the photonfield\s*:?\s*"
        rf"(?P<s0>{_NUMBER})\s+"
        rf"(?P<s1>{_NUMBER})%\s+(?P<s2>{_NUMBER})%\s+"
        rf"(?P<s3>{_NUMBER})%(?:\s|$)",
        re.IGNORECASE,
    )
    _finished = re.compile(
        r"\bprogram stopped regularly via STOP_REGULAR\b", re.IGNORECASE
    )

    def __init__(self, print_info=False, on_info=None):
        super().__init__(print_info=print_info, on_info=on_info)
        self._started = False
        self._finished_seen = False
        self._last_values = {}

    @staticmethod
    def _float(value):
        return float(value.replace("D", "E").replace("d", "e"))

    @staticmethod
    def format_info(kind, data):
        if kind == "arpes_started":
            return "Preparing ARPES calculation..."
        if kind == "arpes_barrier":
            return f"Potential barrier {data.name.upper()}: {data.value:g}"
        if kind == "arpes_lattice":
            return f"Surface lattice constant: {data:g}"
        if kind == "arpes_photon":
            return "Photon wavevector: (" + ", ".join(
                f"{value:g}" for value in data.wavevector
            ) + ")"
        if kind == "arpes_stokes":
            components = ", ".join(f"{value:g}%" for value in data.components_percent)
            return f"Photon Stokes vector: S0={data.s0:g}, ({components})"
        if kind == "arpes_finished":
            return "ARPES calculation finished"
        return None

    def _emit_changed(self, kind, data):
        if kind not in self._last_values or self._last_values[kind] != data:
            self._last_values[kind] = data
            self.emit_info(kind, data)

    def observe_line(self, line):
        text = line.rstrip("\r\n")
        if not text.strip():
            return
        if not self._started:
            self._started = True
            self.emit_info("arpes_started", None)

        if match := self._barrier.match(text):
            try:
                self._emit_changed("arpes_barrier", ArpesBarrierInfo(
                    match["name"].lower(), self._float(match["value"])
                ))
            except ValueError:
                pass
        if match := self._lattice.match(text):
            try:
                self._emit_changed("arpes_lattice", self._float(match["value"]))
            except ValueError:
                pass
        if match := self._photon.match(text):
            try:
                vector = tuple(self._float(match[name]) for name in ("x", "y", "z"))
                self._emit_changed("arpes_photon", ArpesPhotonInfo(vector))
            except ValueError:
                pass
        if match := self._stokes.match(text):
            try:
                info = ArpesStokesInfo(
                    self._float(match["s0"]),
                    tuple(self._float(match[name]) for name in ("s1", "s2", "s3")),
                )
                self._emit_changed("arpes_stokes", info)
            except ValueError:
                pass

        # Only the native regular-stop message proves completion.
        if not self._finished_seen and self._finished.search(text):
            self._finished_seen = True
            self.emit_info("arpes_finished", None)

    async def read_output(self, stdout, result):
        await self.read_commons(_ArpesHeaderStream(stdout, self.observe_line), result)
        result.output_lines = []
        async for raw_line in stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            result.output_lines.append(line)
            self.observe_line(line)
        return result


class ArpesProgressParser(ArpesOutputParser):
    """Incremental INFO parsing for live stdout and stored-job replays."""

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


class ArpesOutputReader(KkrOutputReader):
    """ARPES result reader with structured native INFO and SPC files."""

    result_class = ArpesResult
    parser_class = ArpesOutputParser
