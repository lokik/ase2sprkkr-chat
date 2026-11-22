"""Discovery and selection of task-specific SPR-KKR output readers."""

import os
import re
from pathlib import Path

from ..common.module_registry import (
    ModuleRegistry,
    RegistryDefinitionError,
    RegistryEntryNotFound,
)
from ..sprkkr.executables import executables as default_executables
from . import readers
from .task_result import OutputMatch


_TASK_PATTERN = re.compile(rb"(?m)^\s*TASK\s*=\s*([A-Z][A-Z0-9_]*)\s*$")
_TASK_HEADER_BYTES = 256 * 1024


class OutputReaderRegistry(ModuleRegistry):
    """Load, select and detect ``<Task>OutputReader`` classes."""

    def __init__(self, package=readers, executables=default_executables):
        super().__init__(package)
        self._executables = executables

    def value_from_module(self, name, module):
        class_name = name.title() + "OutputReader"
        try:
            return vars(module)[class_name]
        except KeyError as exc:
            raise RegistryDefinitionError(
                f"Reader module {module.__name__!r} does not define expected "
                f"class {class_name!r}"
            ) from exc

    def prepare_value(self, name, module, value):
        from .task_result import KkrOutputReader

        if not isinstance(value, type) or not issubclass(value, KkrOutputReader):
            raise RegistryDefinitionError(
                f"{module.__name__}.{name.title()}OutputReader must be a "
                f"KkrOutputReader subclass, got {value!r}"
            )
        return value

    def _class_for_reader_name(self, name):
        """Resolve a reader identifier, with the generic reader as fallback."""
        from .readers.default import DefaultOutputReader

        if not name:
            return DefaultOutputReader
        try:
            return self[name]
        except RegistryEntryNotFound:
            return DefaultOutputReader

    def class_for_task_name(self, task_name):
        """Return the reader for a registered task name or guess by reader name."""
        if not task_name:
            return self._class_for_reader_name(task_name)

        # Registered tasks use declarative task/executable metadata. Unknown
        # task names retain the generic standalone/legacy reader fallback.
        from ..input_parameters.input_parameters import InputParameters

        try:
            task = InputParameters.task_registry[task_name]
        except RegistryEntryNotFound:
            return self._class_for_reader_name(task_name)
        return self.class_for_task(task)

    def class_for_task(self, task):
        """Return the reader explicitly declared by the task or its executable."""
        from ..input_parameters.task_definition import TaskDefinition

        if not isinstance(task, TaskDefinition):
            raise TypeError("task must be a TaskDefinition")
        if task.output_reader_name is not None:
            reader_name = task.output_reader_name
        else:
            executable = self._executables[task.executable]
            reader_name = executable.default_output_reader
        return self[reader_name]

    @staticmethod
    def _validate_input_parameters(task, input_parameters):
        if input_parameters is None:
            return
        parameter_task = input_parameters.task_definition
        if parameter_task is not None and parameter_task != task:
            raise ValueError(
                f"Input parameters represent task {parameter_task.name!r}, "
                f"not requested task {task.name!r}"
            )

    @staticmethod
    def _validate_directory(directory):
        if directory is None:
            return
        try:
            os.fspath(directory)
        except TypeError as exc:
            raise TypeError("directory must be a path-like object or None") from exc

    def reader_for_task(
        self,
        task,
        *,
        input_parameters=None,
        calculator=None,
        print_info=False,
        directory=None,
    ):
        """Construct the registered output reader for *task*."""
        self._validate_input_parameters(task, input_parameters)
        if directory is None and calculator is not None:
            directory = getattr(calculator, "directory", None)
        self._validate_directory(directory)
        reader = self.class_for_task(task)(
            input_parameters,
            calculator,
            directory,
            print_info,
        )
        if reader.task_name is None:
            reader.task_name = task.name
        return reader

    def read_for_task(
        self,
        task,
        filename,
        *,
        input_parameters=None,
        calculator=None,
        directory=None,
    ):
        """Read a completed output file using the registered reader for *task*."""
        if directory is None and isinstance(filename, (str, os.PathLike)):
            directory = os.path.dirname(os.fspath(filename))
        return self.reader_for_task(
            task,
            input_parameters=input_parameters,
            calculator=calculator,
            directory=directory,
        ).read_from_file(filename)

    @staticmethod
    def _task_from_paired_input(file):
        """Use only one unambiguous, explicitly task-marked sibling INP/IN.

        A same-stem input is a useful hint for a legacy output, but an
        arbitrary input from the directory must not be associated with it.
        No result or input-parameter cache is retained by this detection.
        """
        from ..input_parameters.input_parameters import InputParameters
        from ..input_parameters.input_parameters_definitions import (
            _read_input_text, _scan_task_selectors, _snapshot_stream,
            _task_selector_map,
        )

        path = Path(os.fspath(file))
        try:
            candidates = [
                candidate for suffix in (".inp", ".in")
                if (candidate := path.with_suffix(suffix)).is_file()
            ]
            if len(candidates) != 1:
                return None
            candidate = candidates[0]
            snapshot = candidate.stat()
            text = _read_input_text(candidate)
            # Native INP without TASK defaults to SCF in the input parser.
            # That default is not evidence that this stdout belongs to SCF.
            if not _scan_task_selectors(text, _task_selector_map()):
                return None
            parameters = InputParameters.from_file(
                _snapshot_stream(text, str(candidate))
            )
            # A rewritten INP cannot be trusted as evidence of stdout type.
            current = candidate.stat()
            if (snapshot.st_size, snapshot.st_mtime_ns, snapshot.st_ctime_ns) != (
                    current.st_size, current.st_mtime_ns, current.st_ctime_ns):
                return None
            return parameters.task_name
        except Exception:
            # An absent, unreadable or invalid optional INP cannot prevent
            # detection using the stdout's own weaker fingerprints.
            return None

    def detect(self, file):
        """Detect the task represented by a standalone SPR-KKR output file.

        Modern outputs identify their task explicitly. Older outputs that do
        not do so are offered to task-specific reader fingerprints, keeping
        legacy format knowledge beside the reader that understands it.
        Definite fingerprints take priority over tentative matches. If no
        definite match exists, an unambiguous same-stem input file with an
        explicit native TASK selector outranks a tentative stdout fingerprint.
        Multiple equally strong remaining matches are treated as ambiguous.
        """
        with open(os.fspath(file), "rb") as output:
            # Native TASK is emitted near the top. Avoid rereading a large
            # stdout only to select its reader before parsing it once more.
            prefix = output.read(_TASK_HEADER_BYTES)
            match = _TASK_PATTERN.search(prefix)
            if match is not None and match.group(1) != b"NONE":
                return match.group(1).decode("ascii")

            # Legacy output readers still require the complete stream.
            raw_output = prefix + output.read()

        match = _TASK_PATTERN.search(raw_output)
        if match is not None:
            task = match.group(1).decode("ascii")
            if task != "NONE":
                return task

        certain, possible = [], []
        for task in self.names(include_aliases=False):
            match = self[task].matches_output(raw_output)
            if match is True or match is OutputMatch.YES:
                certain.append(task)
            elif match is OutputMatch.MAYBE:
                possible.append(task)
            elif match is False or match is None or match is OutputMatch.NO:
                continue
            else:
                raise RegistryDefinitionError(
                    f"Invalid output match confidence {match!r} from {task}"
                )

        # An authoritative stdout match always outranks sibling inputs.
        if len(certain) == 1:
            return certain[0]
        if len(certain) > 1:
            raise RegistryDefinitionError(
                "Output matches multiple task-specific readers: "
                + ", ".join(certain)
            )

        # Input inference is allowed only below authoritative stdout evidence.
        # It may override a weak DOS hint, or fill in an otherwise unknown task.
        input_task = self._task_from_paired_input(file)
        if input_task is not None:
            return input_task

        # Ties between the remaining possible matches are still ambiguous.
        detected = possible
        if len(detected) > 1:
            raise RegistryDefinitionError(
                "Output matches multiple task-specific readers: "
                + ", ".join(detected)
            )
        return detected[0] if detected else None


reader_registry = OutputReaderRegistry()
