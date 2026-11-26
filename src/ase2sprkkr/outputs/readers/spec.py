"""
Parser for spec.out files containing spectral function calculations.
"""

import io
import os
import re
import numpy as np
from typing import Dict

from ..task_result import KkrOutputReader
from .arpes import ArpesResult
from .default import DefaultOutputParser
from ...common.decorators import cached_property


class SpleedIntensity:
    """Numerical I(E) data embedded in a SPLEED calculation's SPEC.out log.

    This is not a separate file. It preserves the familiar callable column
    accessors without routing embedded output through ResultFile.
    """

    def __init__(self, rows):
        self._data = np.asarray(rows, dtype=float)

    def RAW_DATA(self):
        return self._data

    def ENERGY(self):
        return self._data[:, 0]

    def INTENSITY(self):
        return self._data[:, 1]


class SpecResult(ArpesResult):
    """
    Class representing the results of a spectral function calculation.
    Provides access to the parsed data from spec.out files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._file_content = None

    @property
    def lattice_constants(self) -> Dict[str, float]:
        """Extract lattice constants from the output."""
        return self._lattice_info[0]

    @property
    def basis_vectors(self) -> Dict[str, np.ndarray]:
        """Extract basis vectors from the output."""
        return self._lattice_info[1]

    @property
    def spleed_intensity(self):
        """Current two-column SPLEED diagram from the main text output, if any.

        Do not cache: callers may inspect a result while SPEC.out is growing.
        """
        if self.task_name != "spleed":
            return None
        # A parsed SPEC stream is authoritative even when its output file is
        # unavailable or has different contents.
        if self._file_content is not None:
            return SpecOutputParser.parse_spleed_intensity(
                self._file_content.splitlines()
            )
        if self.output_file is None:
            return None
        filename = self.path_to("output")
        if not os.path.isfile(filename):
            return None
        self.require_current_file(filename)
        with open(filename, encoding="utf-8", errors="replace") as stream:
            return SpecOutputParser.parse_spleed_intensity(stream)

    @property
    def spc_filenames(self):
        """SPLEED's header-only SPC is not a separate spectrum."""
        if self.task_name == "spleed":
            parameters = self.input_parameters
            if parameters is not None and str(parameters.SPEC_EL.TYP()) == "0":
                return ()
            if self.spleed_intensity is not None:
                return ()
        return super().spc_filenames

    @property
    def spectral_data(self) -> np.ndarray:
        """Convenience array from the unique spectrum or legacy text log."""
        if self.output_file is None:
            if self.task_name == "spleed":
                spectrum = self.spleed_intensity
                if spectrum is not None:
                    return spectrum.RAW_DATA()
            return SpecOutputParser.parse_inline_spectral_data(
                self._get_file_content()
            )
        if self.task_name == "spleed":
            spectrum = self.spleed_intensity
            if spectrum is not None:
                return spectrum.RAW_DATA()
        spectra = self.spectra
        if len(spectra) > 1:
            raise ValueError("Multiple spectral files; select one from result.spectra")
        if spectra:
            return spectra[0].RAW_DATA()
        if os.path.isfile(self.path_to("output")):
            return SpecOutputParser.parse_inline_spectral_data(
                self._get_file_content()
            )
        return np.array([])

    @cached_property
    def potential_barrier(self) -> Dict[str, float]:
        """Parse the potential barrier section of the output."""
        values = {}
        content = self._get_file_content()

        # Parse potential barrier parameters
        ibar_match = re.search(r"#\s*ibar:\s*(\d+)", content)
        if ibar_match:
            values["ibar"] = int(ibar_match.group(1))

        epsx_match = re.search(r"#\s*epsx:\s*([\d.E+-]+)", content)
        if epsx_match:
            values["epsx"] = float(epsx_match.group(1))

        # Parse zparup and zpardn (these appear to be arrays)
        zpar_match = re.search(r"#\s*zparup:\s*([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)", content)
        if zpar_match:
            values["zparup"] = [float(x) for x in zpar_match.groups()]

        zpardn_match = re.search(r"#\s*zpardn:\s*([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)", content)
        if zpardn_match:
            values["zpardn"] = [float(x) for x in zpardn_match.groups()]

        # Parse bparp
        bparp_match = re.search(r"#\s*bparp:\s*([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.E+-]+)", content)
        if bparp_match:
            values["bparp"] = [float(x) for x in bparp_match.groups()]

        return values

    @cached_property
    def _lattice_info(self):
        """Parse lattice constants and basis vectors from the output."""
        content = self._get_file_content()
        constants = {}

        # Parse lattice constant
        lat_match = re.search(r"lattice constant\s*:\s*([\d.]+)", content)
        if lat_match:
            constants["a"] = float(lat_match.group(1))

        vectors = {"real": np.zeros((2, 2)), "reciprocal": np.zeros((2, 2))}
        # Parse real basis vectors
        matches = re.finditer(r"(real|reciprocal) basis\s+(\d+)\s*:\s*([\d.-]+)\s+([\d.-]+)", content)
        for match in matches:
            vectors[match.group(1)][int(match.group(2)) - 1] = [match.group(3), match.group(4)]

        return constants, vectors

    def _get_file_content(self) -> str:
        """Reuse captured SPEC text; fall back to disk for unparsed results."""
        if self._file_content is not None:
            return self._file_content
        if self.output_file is None:
            return ""
        filename = self.path_to("output")
        self.require_current_file(filename)
        with open(filename, encoding="utf-8", errors="replace") as stream:
            return stream.read()


class _CapturedSpecStream:
    """Tee readline-based common-header parsing and the remaining async lines."""

    def __init__(self, source, on_line=None):
        self.source = source
        self.on_line = on_line
        # Accumulate decoded text directly rather than holding a second list
        # of raw byte lines alongside DefaultOutputParser.output_lines.
        self._text = io.StringIO()

    async def readline(self):
        line = await self.source.readline()
        if line:
            decoded = line.decode("utf-8", errors="replace")
            self._text.write(decoded)
            if self.on_line is not None:
                self.on_line(decoded)
        return line

    def captured_text(self):
        return self._text.getvalue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        line = await self.readline()
        if not line:
            raise StopAsyncIteration
        return line

    def __fspath__(self):
        return os.fspath(self.source)

    def __getattr__(self, name):
        return getattr(self.source, name)


class SpecOutputParser(DefaultOutputParser):
    """Reader for the main text output produced by KKRSPEC."""

    def observe_line(self, line):
        """Hook for task-specific INFO parsing, including the common header."""

    async def read_output(self, stdout, result):
        # read_commons consumes the FILES header before DefaultOutputParser
        # collects output_lines. Capture both parts from the same stream.
        captured = _CapturedSpecStream(stdout, self.observe_line)
        parsed = await super().read_output(captured, result)
        result._file_content = captured.captured_text()
        return parsed

    _DIAGRAM = re.compile(
        r"^\s*(?:i\s*\(e\)\s*-?diagram|rotation diagram|"
        r"scattering-angle diagram,?)\s*$",
        re.IGNORECASE,
    )
    _D_NUMBER = re.compile(r"\s*[+-]?\d\.\d{5}[dD][+-]\d{2}\s*\Z")

    _NATIVE_SPLOUT_NUMBER = re.compile(
        r"\s*[+-]?\d\.\d{5}[dDeE][+-]\d{2}\s*\Z"
    )

    @classmethod
    def parse_spleed_diagram_row(cls, record):
        """Normalize the native SPLOUT intensity formats to (energy, intensity).

        SPR-KKR 9.7 spec_rslabsub.f SPLOUT formats 99005/99004:
        TYP=0 or NREL=0: two D12.5 values (energy, intensity).
        TYP=1/2 and NREL=1: six E12.5 values on stdout, or seven
        in SPEC.out (the additional second column is theta).
        """
        record = record.rstrip("\r\n")
        width = len(record)
        if width == 26:
            count = 2
        elif width == 82:
            count = 6
        elif width == 96:
            count = 7
        else:
            return None
        parts = []
        for index in range(count):
            offset = 14 * index
            value = record[offset:offset + 12]
            if not cls._NATIVE_SPLOUT_NUMBER.fullmatch(value):
                return None
            parts.append(float(value.replace("D", "E").replace("d", "e")))
            if index + 1 < count and record[offset + 12:offset + 14] != "  ":
                return None
        return (parts[0], parts[2] if count == 7 else parts[1])

    @classmethod
    def parse_spleed_intensity(cls, lines):
        """Read native SPLOUT from SPEC.out or standard output.

        The NOUT1 diagram header is written to SPEC.out, not necessarily
        stdout (spec_rslab.f sets NOUT1 = IFILSPECOU3). Accept strict
        formatted numerical rows with or without the header.
        """
        in_diagram = False
        rows = []
        for line in lines:
            if cls._DIAGRAM.fullmatch(line.strip()):
                if rows:
                    break
                in_diagram = True
                continue
            row = cls.parse_spleed_diagram_row(line)
            if row is not None:
                in_diagram = True
                rows.append(row)
                continue
            if in_diagram and line.strip():
                in_diagram = False
                if rows:
                    break
        return SpleedIntensity(rows) if rows else None

    @staticmethod
    def parse_inline_spectral_data(content):
        """Read legacy photoemission diagnostic rows from a text-only log."""
        marker = re.search(r"stookes-vector\s+of the photonfield", content)
        if marker is None:
            return np.array([])
        rows = []
        for line in content[marker.end():].splitlines():
            try:
                values = [
                    float(item.replace("D", "E").replace("d", "e"))
                    for item in line.split()
                ]
            except ValueError:
                if rows and line.strip():
                    break
                continue
            if values:
                if rows and len(values) != len(rows[0]):
                    break
                rows.append(values)
        return np.asarray(rows) if rows else np.array([])

class SpecOutputReader(KkrOutputReader):
    """Process class for spectral function calculations."""

    result_class = SpecResult
    parser_class = SpecOutputParser
