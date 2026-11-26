"""Option and section types used to expose task results."""

import os
import platform
import shutil
import subprocess
import functools
from enum import Enum, auto
from typing import Any, Tuple

from kkrunner.files import FileState

from ..common.container_definitions import SectionDefinition
from ..common.generated_configuration_definitions import GeneratedValueDefinition
from ..common.options import Option
from ..common.configuration_containers import Section
from ..common.file_utils import FilePath
from ..common.value_definitions import ValueDefinition
from ..common.warnings import ValidationResult
from ..result_files import ResultFile, read_plottable_file


def _open_system_path(path):
    """Open a local path with the OS default application, without blocking."""
    path = os.fspath(path)
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    else:
        command = ["open" if system == "Darwin" else "xdg-open", path]
        subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            # Keep stderr connected to the shell: xdg-open/open may report
            # asynchronous failures after Popen has returned.
            start_new_session=True,
        )


class OutputOption(Option):
    @property
    def display_name(self):
        return getattr(self._definition, "display_name", self.name)

    def value_label(self):
        return self()

    def actions(self):
        return ("plot",) if callable(getattr(self, "plot", None)) else ()


class DataOutputOption(OutputOption):
    def value_label(self):
        return "<data...>"

    def actions(self):
        return ("data",)

    def data(self):
        return self()


class FileAttribution(Enum):
    """How confidently these bytes belong to this particular calculation.

    Capturing a current file during import is not evidence of its original
    contents. This relation is owned by each result-file item.
    """

    UNKNOWN = auto()
    NOT_THIS_RESULT = auto()
    LIKELY = auto()
    VERIFIED = auto()


class OutputFileOption(OutputOption):
    def __init__(self, definition, container=None, value=None):
        super().__init__(definition, container, value)
        self.clear_attribution()

    _FILE_DISPLAY_NAMES = {
        "output": "Output file",
        "input": "Input file",
        "potential": "Potential file",
        "converged": "Converged potential file",
        "sfn": "Shape functions (SFN)",
        "dos": "Density of states (DOS)",
        "bloch-sf": "Bloch spectral function (BSF)",
        "spc": "Photoemission spectrum",
        "jxc": "Exchange couplings",
        "dij": "Exchange tensors",
        "dmi": "Dzyaloshinskii–Moriya vectors",
    }

    _TYPE_DISPLAY_NAMES = {
        "dos": "Density of states (DOS)",
        "bsf": "Bloch spectral function (BSF)",
        "spc": "Photoemission spectrum",
        "jxc": "Exchange-interaction data",
        "rat": "X-ray spectroscopy data",
        "sfn": "Shape functions (SFN)",
        "agr": "Grace plot (AGR)",
    }

    _FILE_INFO = {
        "output": "Main text output produced by SPR-KKR.",
        "input": "Input parameters used for this calculation.",
        "potential": "Potential used as input for this calculation.",
        "converged": "Potential produced by the converged SCF calculation.",
        "sfn": "Shape-function data used by a full-potential calculation.",
        "agr": "Grace plot project containing one or more datasets.",
    }

    _TYPE_INFO = {
        "dos": "Density-of-states data produced by the calculation.",
        "bsf": "Bloch spectral-function data produced by the calculation.",
        "spc": "Photoemission spectrum data produced by the calculation.",
        "jxc": "Exchange-interaction data produced by the calculation.",
        "rat": "X-ray spectroscopy data produced by the calculation.",
        "sfn": "Shape-function data used by a full-potential calculation.",
    }

    @property
    def info(self):
        return (
            self._TYPE_INFO.get(self.file_type)
            or self._FILE_INFO.get(self.name.casefold())
            or f"File produced or used by the calculation: {self.name}."
        )

    @property
    def display_name(self):
        return (
            self._FILE_DISPLAY_NAMES.get(self.name.casefold())
            or self._TYPE_DISPLAY_NAMES.get(self.file_type)
            or super().display_name
        )

    def set(self, value, **kwargs):
        """Clear attribution when the underlying file is changed directly.

        Direct `result.files[name].set(...)` must have the same semantics as
        `result.files.add_file(name, ...)`; provenance belongs to the file
        identity, not to the option's name.
        """
        try:
            previous = self()
        except (AttributeError, ValueError, TypeError):
            previous = None
        super().set(value, **kwargs)
        current = self()
        if (previous is None) != (current is None) or (
            previous is not None and current is not None
            and os.fspath(previous) != os.fspath(current)
        ):
            self.clear_attribution()

    def _assess_file(self):
        """Return (attribution, evidence, usable) from one currentness check.

        Only the version/attribution evidence held by this file option decides
        whether its bytes may be used. TaskResult does not make a second
        decision for registered paths, and no filesystem state is retained.
        """
        original = self._file_attribution
        evidence = self._file_attribution_evidence
        recorded = self.recorded_version
        observed = self.observed_version
        owner = self._container.result

        # Historical outputs may be used without a ResultLoader. Find their
        # recorded version without adopting today's contents as new history.
        historical = False
        if owner is not None:
            from .file_provenance import result_file_manifest, result_file_version
            historical = (result_file_manifest(owner) is not None
                          or owner.run_record is not None)
            if historical and recorded is None:
                recorded = result_file_version(owner, self.path())

        if recorded is not None:
            if recorded.kind != "file":
                return FileAttribution.UNKNOWN, "not_regular_file", False
            state = recorded.state()
            if recorded.available:
                return {
                    FileState.CURRENT: (FileAttribution.VERIFIED, "run_manifest_digest", True),
                    FileState.OVERWRITTEN: (FileAttribution.NOT_THIS_RESULT,
                                            "recorded_version_changed", False),
                    FileState.MISSING: (FileAttribution.UNKNOWN, "recorded_file_missing", False),
                    FileState.UNAVAILABLE: (FileAttribution.UNKNOWN,
                                            "recorded_file_unavailable", False),
                }[state]
            # Metadata-only history has weaker evidence, but a changed or
            # missing file must still be refused.
            if state is not FileState.CURRENT:
                return FileAttribution.UNKNOWN, "run_manifest_metadata_only", False
            if original is FileAttribution.NOT_THIS_RESULT:
                return original, evidence, False
            if evidence == "scf_mtime_tie_or_unreadable":
                return FileAttribution.UNKNOWN, evidence, False
            return FileAttribution.LIKELY, evidence or "run_manifest_metadata_only", True

        if historical:
            return FileAttribution.UNKNOWN, "not_in_manifest", False
        if original is FileAttribution.NOT_THIS_RESULT:
            return original, evidence, False
        if evidence == "scf_mtime_tie_or_unreadable":
            return FileAttribution.UNKNOWN, evidence, False

        if observed is not None and observed.kind != "file":
            return FileAttribution.UNKNOWN, "not_regular_file", False
        if observed is not None and observed.state() is not FileState.CURRENT:
            return FileAttribution.UNKNOWN, "observed_version_changed", False
        if observed is not None:
            # Observation alone is not a run-time digest-based proof.
            return (FileAttribution.LIKELY if original is FileAttribution.VERIFIED else original,
                    evidence, True)
        if not os.path.isfile(self.path()):
            return FileAttribution.UNKNOWN, "missing", False
        # A manually supplied label without an identity cannot prove bytes.
        return (FileAttribution.LIKELY if original is FileAttribution.VERIFIED else original,
                evidence, True)

    @property
    def attribution(self):
        """Current relationship between this pathname's bytes and the result."""
        return self._assess_file()[0]

    @property
    def attribution_evidence(self):
        return self._assess_file()[1]

    def attribution_details(self):
        """Return attribution and evidence from one filesystem observation."""
        attribution, evidence, _ = self._assess_file()
        return attribution, evidence

    @property
    def attributed_to(self):
        """Other SCF output identified as the likely source of these bytes."""
        return self._file_attributed_to

    @property
    def recorded_version(self):
        """Version recorded by the calculation, if any."""
        return self._file_recorded_version

    @property
    def observed_version(self):
        """Version captured at import, not proof of original-run identity."""
        return self._file_observed_version

    def set_attribution(
        self, attribution, *, evidence=None, recorded_version=None,
        observed_version=None, attributed_to=None,
    ):
        if not isinstance(attribution, FileAttribution):
            raise TypeError("attribution must be a FileAttribution")
        self._file_attribution = attribution
        self._file_attribution_evidence = evidence
        self._file_recorded_version = recorded_version
        self._file_observed_version = observed_version
        self._file_attributed_to = attributed_to
        return self

    def clear_attribution(self):
        return self.set_attribution(FileAttribution.UNKNOWN)

    def exists(self):
        """Whether the registered pathname exists, regardless of attribution."""
        return os.path.exists(self.path())

    def is_current(self):
        """Can this file's *current bytes* be used by its owning result?"""
        return self._assess_file()[2]

    def require_current(self):
        """Validate this file once, raising for missing or wrong bytes."""
        if not self._assess_file()[2]:
            from kkrunner.runs import HistoricalFileMismatchError
            raise HistoricalFileMismatchError(
                f"Result file is missing, overwritten, or not attributed: {self.path()}"
            )
        return self.path()

    def path(self):
        owner = self._container.result
        return owner.path_to(self.name) if owner is not None else os.fspath(self())

    @property
    def file_type(self):
        extension = os.path.splitext(self())[1].lstrip(".").lower()
        # Grace project format is authoritative even for a named output option
        # with an older task-specific file type hint.
        if extension == "agr":
            return "agr"
        if self._definition.file_type:
            return self._definition.file_type
        return extension if extension in ResultFile.definitions else None

    def definition(self):
        return ResultFile.definitions[self.file_type]

    def value_label(self):
        # The user-facing value is the actual filename, not a format placeholder.
        return os.path.basename(self.path())

    def actions(self):
        # Navigating to a calculation's directory remains useful even when
        # its historical file is missing or has been overwritten.
        path = self.path()
        directory_action = (
            ["open_directory"]
            if os.path.isdir(os.path.dirname(os.path.abspath(path)))
            else []
        )
        # Opening, copying and scientific processing must only use verified
        # current bytes; directory navigation does not read those bytes.
        if not os.path.isfile(path) or not self.is_current():
            return tuple(directory_action)
        out = ["open", *directory_action, "copy_to"]
        if self.file_type == "agr":
            out.append("plot")
        elif self.file_type:
            result_class = self.definition().result_class
            if result_class.can_be_plotted():
                out.append("plot")
            for action in result_class.additional_actions:
                # Several unrelated formats use .spc. When the filename only
                # supplies that ambiguous extension, do not advertise ARPES
                # conversion for SPLEED/BAND/AIPES calculations.
                if action == "export_nexus" and self.file_type == "spc":
                    owner = self._container.result
                    try:
                        task = owner.task_name if owner is not None else None
                    except (AttributeError, ValueError):
                        task = None
                    if str(task).upper() != "ARPES":
                        continue
                out.append(action)
        return tuple(out)

    def __getattr__(self, name):
        if not name.startswith("_") and self.file_type:
            result_class = self.definition().result_class
            if name in result_class.additional_actions:
                return functools.partial(self._run_file_action, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def _run_file_action(self, action, *args, **kwargs):
        return getattr(self.parsed_file(), action)(*args, **kwargs)

    def parsed_file(self, potential=None):
        """Parse the file, using its task result as context when available."""
        owner = self._container.result
        if owner is not None:
            return owner.parse_result_file(self, potential=potential)
        self.require_current()
        return read_plottable_file(
            self.path(),
            try_only=self.file_type,
            potential=potential,
        )

    def export_nexus(self, destination, **kwargs):
        """Export an ARPES spectrum to a caller-selected NeXus file.

        The parsed result owns the scientific writer; callers (including GUIs)
        choose the destination and handle presentation of errors.
        """
        spectrum = self.parsed_file()
        exporter = getattr(spectrum, "to_nexus", None)
        if not callable(exporter):
            raise ValueError(f"NeXus export is unavailable for {self.path()}")
        owner = self._container.result
        if owner is not None:
            # An associated TaskResult is authoritative: never rediscover a
            # possibly unrelated neighboring .inp from the SPC filename.
            kwargs.setdefault("discover_input", False)
            if "input_parameters" not in kwargs and "input_file" not in kwargs:
                known_parameters = owner._input_parameters
                if known_parameters is None:
                    # The lazy input accessor may already have parsed a
                    # verified source before it was subsequently removed.
                    known_parameters = vars(owner).get("input_parameters")
                if known_parameters is not None:
                    kwargs["input_parameters"] = known_parameters
                else:
                    recorded_input = owner.input_parameters_file
                    if recorded_input is not None:
                        kwargs["input_file"] = recorded_input
        return exporter(destination, **kwargs)

    def plot(self, *args, **kwargs):
        """Plot scientific data, forwarding options to the native plotter."""
        return self.parsed_file().plot(*args, **kwargs)

    def copy_to(self, destination):
        """Copy an existing result file after verifying its historical identity."""
        return shutil.copy2(self.require_current(), destination)

    def open(self):
        """Open this verified result file in its default external application."""
        _open_system_path(self.require_current())

    def open_directory(self):
        """Open the containing folder, even if the historical file has changed."""
        directory = os.path.dirname(os.path.abspath(self.path()))
        if not os.path.isdir(directory):
            raise FileNotFoundError(directory)
        _open_system_path(directory)


class ResultFileDefinition(ValueDefinition):
    result_class = OutputFileOption
    name_in_grammar = False

    def __init__(self, name, file_type=None):
        self.file_type = file_type
        super().__init__(name, str, is_optional=True)

    def convert_value(self, option: Option, value: Any, item: bool = False) -> FilePath:
        """Resolve ``value`` relative to the result owning ``option``.

        ``item`` is accepted for the common value-definition conversion API.
        """
        owner = option._container.result
        directory = owner.directory if owner is not None else None
        return FilePath(value, directory)

    def validate_converted(
        self, option: Option, value: Any, why: str = "set", item: bool = False
    ) -> Tuple[ValidationResult, ...]:
        """Accept converted ``value``; remaining arguments match the common API."""
        return ()


class OutputFilesSection(Section):
    def add_file(self, name, filename, file_type=None):
        if name in self:
            option = self[name]
            option.set(filename)
            if file_type is not None:
                option._definition.file_type = file_type
            return option
        definition = ResultFileDefinition(name, file_type)
        definition.added_to_container(self._definition)
        self._definition._members[name] = definition
        option = definition.create_object(self)
        option.set(filename)
        self._add(option)
        return option

    def set_file_type(self, name, file_type):
        """Set the parser type for an already registered output file."""
        self[name]._definition.file_type = file_type
        return self[name]


def create_files_section(result=None):
    definition = SectionDefinition("FILES", [], result_class=OutputFilesSection)
    out = definition.create_object()
    object.__setattr__(out, "result", result)
    return out


def generated_output(
    name,
    getter,
    *,
    result_class=OutputOption,
    plot=None,
    info=None,
    display_name=None,
):
    definition = GeneratedValueDefinition(
        name,
        getter,
        result_class=result_class,
        plot=plot,
        info=info,
    )
    definition.display_name = display_name or name
    return definition


def create_output_section(name, members):
    return SectionDefinition(name, members).create_object()
