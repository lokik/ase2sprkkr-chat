"""This module contains classes used by parsers of SPR-KKR output files."""

import asyncio
import os
from ..common.decorators import cached_property, maybeclassmethod
from ..common.file_utils import FilePath, filename_from_file
from ..potentials.potentials import Potential
from ..input_parameters import input_parameters as input_parameters
from .result_options import create_files_section


class TaskResult:
    """A base class for a result of a runned task (kkrscf executable)"""

    produces_potential = False
    """Whether this result produces a new potential usable by later tasks."""

    sfn_generated = None
    """Whether this calculation generated rather than reused its SFN file."""

    def __init__(
        self,
        input_parameters,
        calculator,
        directory,
        output_file=None,
        input_file=None,
        task_name=None,
    ):
        self._input_parameters = input_parameters
        self._calculator = calculator
        self._task_name = (
            str(task_name).strip().lower() if task_name is not None else None
        )
        self.files = create_files_section(self)
        self._directory = filename_from_file(directory, None, realpath=True)

        output_filename = filename_from_file(output_file, None)
        self.output_file = None
        if output_filename is not None:
            if self._directory is None and os.path.isabs(output_filename):
                self._directory = os.path.dirname(output_filename)
            self.output_file = FilePath(output_filename, self.directory)
            self.files.add_file("output", self.output_file)

        input_filename = filename_from_file(input_file, None)
        self.input_file = FilePath(input_filename, self.directory) if input_filename is not None else None
        if self.input_file is not None:
            self.files.add_file("input", self.input_file)

    @cached_property
    def directory(self):
        return self._directory or os.getcwd()

    def path_to(self, file):
        """return full path to a given file

        ..doctest::
        >>> t = TaskResult(None, None, "/example")
        >>> t.files.add_file("input", "input.txt")
        <Option input of type String with value = input.txt>
        >>> t.path_to("input")
        '/example/input.txt'
        """
        return os.fspath(self.files[file]())

    @property
    def task_name(self):
        if self._task_name is not None:
            return self._task_name
        parameters = self.input_parameters
        task = parameters.task_definition if parameters is not None else None
        if task is None:
            raise ValueError("Cannot determine the task of this result from its input")
        return str(task.name).strip().lower()

    @cached_property
    def input_parameters(self):
        if self._input_parameters is not None:
            return self._input_parameters
        source = self.input_parameters_file
        if source is None:
            return None
        # Parsing a registered input is independent of import caches.
        # Explicit input errors stay observable to the caller.
        return input_parameters.InputParameters.from_file(source)

    @property
    def input_parameters_file(self):
        """Registered input path. Native-input discovery belongs to ResultLoader."""
        if "input" not in self.files:
            return None
        filename = self.path_to("input")
        return filename if self.file_is_current(filename) else None

    @property
    def is_embedded(self):
        """Whether the run uses a separate cluster potential."""
        parameters = self.input_parameters
        if parameters is None:
            return False
        try:
            return parameters.CONTROL.CLUTYPE() == "embedded"
        except (AttributeError, KeyError):
            return False

    @cached_property
    def host_potential_filename(self):
        """Host POTFIL referenced by a cluster calculation."""
        parameters = self.input_parameters
        filename = parameters.CONTROL.POTFIL() if parameters is not None else None
        if not filename:
            raise ValueError("An embedded calculation needs a host POTFIL")
        return os.path.join(self.directory, os.fspath(filename))

    @cached_property
    def host_potential(self):
        """Parse the shared host once for input and converged CLUPOT files."""
        self.require_current_file(self.host_potential_filename)
        return Potential.from_file(self.host_potential_filename)

    def _read_potential(self, filename):
        if not self.is_embedded:
            return Potential.from_file(filename)
        from ..potentials.cluster_potentials import ClusterPotential

        return ClusterPotential.from_file(filename, host=self.host_potential)

    def _input_potential_filename(self):
        """Native POTFIL used as input, independent of any produced file."""
        parameters = self.input_parameters
        potfil = parameters.CONTROL.POTFIL() if parameters is not None else None
        if self.is_embedded:
            potfil = parameters.CONTROL.POTFIL_CLU()
            if not potfil:
                raise ValueError("Embedded results require CONTROL.POTFIL_CLU")
        if not potfil:
            potfil = self.files["potential"]() if "potential" in self.files else None
        if not potfil:
            raise ValueError("Please set CONTROL.POTFIL of the input_parameters to read the potential")
        if self.directory:
            potfil = os.path.join(self.directory, potfil)
        return potfil

    @cached_property
    def potential_filename(self):
        """Potential representing this result, not necessarily its input."""
        if self.produces_potential and "converged" in self.files:
            return self.path_to("converged")
        return self._input_potential_filename()

    @property
    def potential_dependencies(self):
        """Other source potentials required to reconstruct this result.

        The primary potential is :attr:`potential_filename`. An embedded
        calculation also depends on its separate periodic host.
        """
        return (self.host_potential_filename,) if self.is_embedded else ()

    def _load_result_potential(self, loader=None):
        """Read the potential belonging to this result, optionally using a cache.

        The result is responsible for choosing its own native files and
        verifying their provenance. A ResultLoader merely shares unmodified
        parsed file data; it does not interpret the calculation.
        """
        filename = self.potential_filename
        self.require_current_file(filename)
        if loader is None:
            return self._read_potential(filename)

        if self.is_embedded:
            host = self.host_potential_filename
            self.require_current_file(host)
            potential = loader.potential(filename, host=host)
            self.require_current_file(host)
        else:
            potential = loader.potential(filename)
        self.require_current_file(filename)
        return potential

    @cached_property
    def potential(self):
        """Scientifically interpreted potential belonging to this result."""
        return self._load_result_potential()

    def get_atoms(self, *, loader=None):
        """Return this result's atoms, optionally sharing raw parses via loader.

        Passing a loader changes performance, never scientific interpretation.
        The loader is not retained on the result.
        """
        if loader is None:
            return self.potential.atoms
        return self._load_result_potential(loader).atoms

    @property
    def any_potential(self):
        """Return the parsed potential (SCF then try output and then input) """
        try:
            return self.potential
        except (AttributeError, KeyError, OSError, ValueError):
            return None

    def _file_options(self, path):
        """All registered roles of one path; each retains its own attribution."""
        requested = os.path.abspath(os.fspath(path))
        return tuple(
            option for option in self.files.items().values()
            if os.path.abspath(os.fspath(option.path())) == requested
        )

    def file_is_current(self, path):
        """Whether the current contents can be used by this result.

        A registered file delegates its attribution and version check to
        OutputFileOption. Only unregistered dependencies (e.g. a cluster's
        host potential) use the manifest fallback here.
        """
        options = self._file_options(path)
        if options:
            # A pathname alone does not identify a role.  Refuse it if any
            # registered role identifies today's bytes as another result.
            return all(option.is_current() for option in options)

        from .file_provenance import result_file_manifest, result_file_version
        if result_file_manifest(self) is None and getattr(self, "run_record", None) is None:
            return os.path.isfile(path)
        version = result_file_version(self, path)
        return version is not None and version.is_current()

    def require_current_file(self, path):
        """Validate a registered file once, otherwise check its run manifest."""
        options = self._file_options(path)
        if options:
            for option in options:
                option.require_current()
            return
        if not self.file_is_current(path):
            from kkrunner.runs import HistoricalFileMismatchError
            raise HistoricalFileMismatchError(f"Result file is unavailable or changed: {path}")

    def parse_result_file(self, file, potential=None):
        """Parse a registered structured result file in this calculation context."""

        from ..result_files import read_plottable_file

        if getattr(getattr(file, "_container", None), "result", None) is not self:
            raise ValueError("A parsed result file must belong to this TaskResult")
        file.require_current()
        # Grace plots do not depend on the calculation potential.
        if potential is None and file.file_type != "agr":
            potential = self.any_potential
        return read_plottable_file(
            file.path(),
            try_only=file.file_type,
            potential=potential,
        )

    def new_task(self, task):
        """Create a fresh calculator for *task* using values from this result."""
        from ..input_parameters.task_definition import TaskDefinition
        from ..sprkkr.calculator import SPRKKR

        task = (
            task
            if isinstance(task, TaskDefinition)
            else input_parameters.InputParameters.task_registry[task]
        )
        parameters = (
            input_parameters.InputParameters.input_parameters_registry.create_from_result(
                task, self
            )
        )
        return SPRKKR(
            potential=self.potential if self.is_embedded else self.potential_filename,
            input_parameters=parameters,
        )

    def complete(self, error, return_code):
        self.error = error
        self.return_code = return_code
        # Completed results are parsed from locally available output files.
        # Absent declared outputs are not results; diagnose required ones only
        # for calculations that report successful completion.
        self.enrich_declared_output_files(warn_missing=return_code == 0)

    def enrich_declared_output_files(self, *, warn_missing=True, file_operations=None):
        """Expose existing task-declared outputs, preserving parsed FILES entries."""
        from .file_provenance import register_declared_result_files

        return register_declared_result_files(
            self, warn_missing=warn_missing, file_operations=file_operations
        )

    @property
    def atoms(self):
        return self.get_atoms()

    @property
    def output_values(self):
        """Files and other values exposed as the public result summary."""
        return self._ordered_output_files()

    @property
    def result_values(self):
        """Scientific outputs only, selected by the task's result declarations.

        Auxiliary outputs (such as SCF's *_new potential), input/output logs
        and caches stay available through output_values and the file manifest,
        but are not standalone end-of-run result tabs.
        """
        from fnmatch import fnmatchcase
        from .file_provenance import task_file_declarations

        _params, declarations = task_file_declarations(self, kinds=("result",))
        patterns = tuple(
            os.path.normcase(os.path.abspath(os.fspath(declaration)))
            for _kind, declaration in declarations if declaration.collect
        )
        if not patterns:
            return {}
        # Several parser keys can refer to the same physical output. A
        # scientific result represents a file, not an arbitrary alias key.
        results = {}
        seen_paths = set()
        for name, value in self.output_values.items():
            try:
                path = os.path.normcase(os.path.abspath(os.fspath(value.path())))
            except (AttributeError, TypeError, ValueError, OSError):
                continue
            if path in seen_paths or not any(fnmatchcase(path, pattern) for pattern in patterns):
                continue
            seen_paths.add(path)
            results[name] = value
        return results

    def _ordered_output_files(self):
        """Put actual result files first; omit internal, untyped task caches."""
        from fnmatch import fnmatchcase

        from .file_provenance import task_file_declarations

        _parameters, declarations = task_file_declarations(self, kinds=("cache",))
        cache_patterns = tuple(
            os.fspath(value) for kind, value in declarations if kind == "cache"
        )
        files = self.files.items()
        visible = {}
        for name, value in files.items():
            if value.file_type:
                visible[name] = value
                continue
            if name.casefold() in {"taudir", "tau_dir"}:
                # Native TAUDIR is a calculation cache, not a result file.
                continue
            if any(fnmatchcase(value.path(), pattern) for pattern in cache_patterns):
                continue
            visible[name] = value
        task_files = {name: value for name, value in visible.items() if value.file_type}
        task_files.update({name: value for name, value in visible.items() if not value.file_type})
        return task_files

    def to_file_manifest(
        self,
        *,
        output_path=None,
        potential_path=None,
        result_files=None,
        output_version=None,
        file_operations=None,
    ):
        """Capture the files used or produced by this result as a KKRunner manifest."""
        from .file_provenance import capture_result_file_manifest

        return capture_result_file_manifest(
            self,
            output_path=output_path,
            potential_path=potential_path,
            result_files=result_files,
            output_version=output_version,
            file_operations=file_operations,
        )

    @classmethod
    def from_file(cls, file):
        """Load a standalone output through a fresh import context."""
        from .result_loader import ResultLoader

        return ResultLoader().load_file(file)

    @classmethod
    def _parse_file(cls, file, *, on_files_parsed=None):
        """Parse stdout, optionally resolving inputs before finalization."""
        from .reader_registry import reader_registry

        filename = os.fspath(file)
        task = reader_registry.detect(filename)
        reader_class = reader_registry.class_for_task_name(task)
        reader = reader_class(
            None,
            None,
            os.path.dirname(filename),
            task_name=task,
        )
        return reader.read_from_file(filename, on_files_parsed=on_files_parsed)

    @classmethod
    def from_run(cls, run, manager=None, *, allow_unparsed=False):
        """Load a persisted run using an execution manager.

        With no manager, use the shared default project-run store. The
        manager retrieves missing files and checks their saved fingerprints.
        """
        if manager is None:
            from ase2sprkkr.sprkkr.projects import project_execution_manager

            manager = project_execution_manager()
        from .project_results import load_project_result
        return load_project_result(manager, run, allow_unparsed=allow_unparsed)

    @classmethod
    def from_run_record(
        cls, record, *, output_file=None, error_file=None, return_code=None
    ):
        """Parse a locally available run using its persisted task metadata.

        File retrieval and fingerprint verification are handled by
        :meth:`from_run` through the project run manager, not by this method.
        """
        files = record.files
        if files is None:
            raise FileNotFoundError(f"Run {record.id} has no recorded output files")
        output = files.output_file if output_file is None else output_file
        if output is None:
            raise FileNotFoundError(f"Run {record.id} has no recorded output file")

        from .project_results import _reader_for_record
        reader = _reader_for_record(None, record)
        available_input = files.input_file if getattr(reader, "input_parameters", None) is not None else None

        result = reader.read_from_file(
            output,
            error_file,
            return_code=(
                record.return_code if return_code is None else return_code
            ) or 0,
            input_file=available_input,
            run_record=record,
        )
        result.run_record = record
        from .result_loader import ResultLoader
        return ResultLoader().associate_result(result)


class KkrOutputReader:
    """Parse SPR-KKR output streams into a TaskResult.

    Execution is handled by :mod:`kkrunner`; this class owns only
    task-specific parsing and result construction.
    """

    def __init__(
        self,
        input_parameters,
        calculator,
        directory,
        print_info=False,
        task_name=None,
        on_info=None,
    ):
        self.input_parameters = input_parameters
        self.calculator = calculator
        self.directory = directory
        task = input_parameters.task_definition if input_parameters is not None else None
        self.task_name = (
            task_name
            if task_name is not None
            else task.name
            if task is not None
            else None
        )
        self.parser = self.parser_class(print_info=print_info, on_info=on_info)

    @classmethod
    def matches_output(cls, output):
        """Return whether legacy output without a task marker belongs to this reader."""
        return False

    def create_result(self, output_file, input_file=None):
        """Create the TaskResult before parsing starts."""
        return self.result_class(
            self.input_parameters,
            self.calculator,
            self.directory,
            output_file=output_file,
            input_file=input_file,
            task_name=self.task_name,
        )

    async def parse_streams(self, stdout, stderr, result):
        """Parse simultaneously growing stdout and stderr streams.

        Parser failures are returned to the execution backend rather than
        raised immediately. The backend must first observe the process exit
        status: a truncated output stream is often only a symptom of the
        executable having failed, and that failure is the more useful error.
        """
        return await asyncio.gather(
            self.parser.read_output(stdout, result),
            self.parser.read_error(stderr, result),
            return_exceptions=True,
        )

    def complete_result(self, result, output, error, return_code):
        """Finalize a result after the execution backend completes."""
        if return_code == 0:
            if isinstance(output, BaseException):
                raise output
            if isinstance(error, BaseException):
                raise error

        # A nonzero exit takes precedence over parser/stream errors. Ask the
        # parser to classify the exit once, then rethrow an earlier stream
        # failure only if it accepted the exit status.
        _output, parsed_error, parsed_return_code = self.parser.result(
            None if isinstance(output, BaseException) else output,
            None if isinstance(error, BaseException) else error,
            return_code,
        )
        if isinstance(output, BaseException):
            raise output
        if isinstance(error, BaseException):
            raise error

        error, return_code = parsed_error, parsed_return_code
        result.complete(error, return_code)
        return result

    @maybeclassmethod
    def read_from_file(self, cls, output, error=None, return_code=0, input_file=None, directory=None, *, run_record=None, on_files_parsed=None):
        """Parse output of an already finished SPR-KKR calculation."""
        if self is None:
            self = cls(None, None, directory)
        standalone_loader = None
        if (on_files_parsed is None and run_record is None
                and self.input_parameters is None
                and isinstance(output, (str, os.PathLike))):
            from .result_loader import ResultLoader

            standalone_loader = ResultLoader()
            on_files_parsed = standalone_loader._prepare_input
        result = self.create_result(output, input_file)
        if run_record is not None:
            result.run_record = run_record
        output_source = output if hasattr(output, "read") else result.path_to("output")

        if on_files_parsed is not None:
            self.parser._on_files_parsed = on_files_parsed
        try:
            _parsed_output, parsed_error, parsed_return_code = self.parser.read_from_file(
                output_source, error, [result], return_code
            )
        finally:
            if on_files_parsed is not None:
                del self.parser._on_files_parsed
        result.complete(parsed_error, parsed_return_code)
        if standalone_loader is not None:
            return standalone_loader.associate_result(result)
        return result
