"""Batch discovery and association of SPR-KKR output files.

The loader owns scientific discovery, parsed-input/potential caches and SCF matching.
KKRunner FileOperations owns short-lived filesystem observations; results
retain the resulting per-file attribution evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
import os
import stat as stat_module
from pathlib import Path
from typing import Iterable

from kkrunner.files import FileFingerprint, FileOperations, FileVersion

from ..input_parameters import input_parameters as input_parameters
from ..input_parameters.input_parameters import NATIVE_INPUT_ERRORS
from .file_provenance import result_file_manifest, result_file_version
from .output_detection import looks_like_scf_output
from .result_options import FileAttribution
from .task_result import TaskResult, MissingPotentialError
from ..common.process_output_reader import ProcessExitError


class _FileChangedDuringReadError(RuntimeError):
    """External bytes changed while the loader was reading or associating them."""


_INPUT_DISCOVERY_ERRORS = NATIVE_INPUT_ERRORS + (_FileChangedDuringReadError,)
_POTENTIAL_DISCOVERY_ERRORS = NATIVE_INPUT_ERRORS + (MissingPotentialError,)
_COMPANION_READ_ERRORS = _INPUT_DISCOVERY_ERRORS + (EOFError, ProcessExitError)


def _file_batch(method):
    """Reuse one coherent observation cache across nested loader calls."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._filesystem_batch():
            return method(self, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class _DirectoryIndex:
    version: FileFingerprint
    outputs: tuple[Path, ...]
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class _ResultSources:
    """Small observation snapshot, retained without retaining the loader."""
    files: tuple[FileVersion, ...]
    input_candidates: tuple | None = None


@dataclass(frozen=True)
class LoadFailure:
    path: Path
    error: Exception


@dataclass(frozen=True)
class ResultBatch:
    entries: tuple[tuple[Path, TaskResult], ...]
    failures: tuple[LoadFailure, ...]

    @property
    def results(self):
        return tuple(result for _, result in self.entries)


class _NativeInputCache:
    """Parsed native inputs scoped to one ResultLoader (never to TaskResult)."""

    def __init__(self, file_operations: FileOperations):
        self._file_operations = file_operations
        self._parameters = {}

    def parse(self, filename):
        path = os.path.realpath(os.fspath(filename))
        identity = self._file_operations.fingerprint(path)
        entry = self._parameters.get(path)
        if entry is None or entry[0] != identity:
            try:
                parameters = input_parameters.InputParameters.from_file(path)
                # A native input is shared between results.  Do not cache a
                # partial or mixed parse if its contents changed in the middle
                # of reading, even if a later output load is otherwise valid.
                if self._file_operations.fingerprint(path, refresh=True) != identity:
                    raise _FileChangedDuringReadError(f"Native input changed during parsing: {path}")
            except _INPUT_DISCOVERY_ERRORS as exc:
                self._parameters[path] = (identity, exc)
                raise
            self._parameters[path] = (identity, parameters)
        else:
            parameters = entry[1]
            if isinstance(parameters, Exception):
                raise parameters
        # Parameters are mutable; every result receives its own detached copy.
        if isinstance(parameters, input_parameters.InputParameters):
            parameters = parameters.copy(copy_values=True)
        else:
            parameters = deepcopy(parameters)
        return parameters, identity


class ResultLoader:
    """Short-lived, per-instance cache for filesystem result reconstruction.

    Not thread-safe; use one instance per import worker. Recorded versions
    (from a run manifest) and versions first seen during import are distinct.
    """

    def __init__(self, *, file_operations: FileOperations | None = None):
        self.file_operations = file_operations if file_operations is not None else FileOperations()
        self.inputs = _NativeInputCache(self.file_operations)
        self._filesystem_batch_depth = 0
        self._directories = {}
        self._outputs = {}
        self._scf_probes = {}
        self._scf_matches = {}
        self._potentials = {}

    @contextmanager
    def _filesystem_batch(self):
        """Share a derived directory index within one import operation.

        FileOperations owns physical observations, including their lifetime:
        a caller who passes an instance explicitly controls when it is
        invalidated. A new ResultLoader creates its own fresh observations.
        """
        if not self._filesystem_batch_depth:
            # Clear only the derived domain index, never the caller's cache.
            # Rebuilding it uses FileOperations' cached physical observations.
            self._directories.clear()
        self._filesystem_batch_depth += 1
        try:
            yield
        finally:
            self._filesystem_batch_depth -= 1

    @classmethod
    def load_directory(cls, directory):
        loader = cls()
        with loader._filesystem_batch():
            return loader.load_files(loader.find_outputs(directory))

    def _directory(self, folder):
        folder = folder.expanduser().resolve()
        cached = self._directories.get(folder)
        if cached is not None:
            return cached
        version = self.file_operations.fingerprint(folder)
        if not stat_module.S_ISDIR(version.mode):
            raise NotADirectoryError(folder)
        # Only candidates need a stat: large run directories may contain
        # many unrelated binary outputs and caches.
        paths = tuple(
            folder / name
            for name in self.file_operations.listdir(folder)
            if Path(name).suffix.casefold() in {".out", ".inp", ".in"}
            and self.file_operations.is_file(folder / name)
        )
        cached = _DirectoryIndex(
            version,
            tuple(sorted((p for p in paths if p.suffix.casefold() == ".out"),
                         key=lambda p: (p.name.casefold(), str(p)))),
            tuple(sorted(str(p) for p in paths
                         if p.suffix.casefold() in {".inp", ".in"})),
        )
        self._directories[folder] = cached
        return cached

    @_file_batch
    def find_outputs(self, directory):
        """Return only .out paths, in a reproducible order."""
        return self._directory(Path(directory)).outputs

    @_file_batch
    def scf_claims(self, directory, potential):
        """Find SCF .out files that explicitly name this output POTFIL.

        Includes outputs not selected for import, e.g. when loading recents.
        """
        expected = Path(potential).expanduser().resolve()
        found = []
        for output in self.find_outputs(directory):
            try:
                version = self.file_operations.fingerprint(output)
            except OSError:
                continue
            key = output, version
            if key not in self._scf_probes:
                self._scf_probes[key] = looks_like_scf_output(output)
            if not self._scf_probes[key]:
                continue
            match_key = output, version, expected
            if match_key not in self._scf_matches:
                self._scf_matches[match_key] = looks_like_scf_output(
                    output, potential_filename=expected
                )
            if self._scf_matches[match_key]:
                found.append(output)
        return tuple(found)

    def _newest(self, paths):
        try:
            versions = [(self.file_operations.fingerprint(p).mtime_ns, p) for p in set(paths)]
        except OSError:
            return None
        if not versions:
            return None
        newest = max(t for t, _ in versions)
        matches = [p for t, p in versions if t == newest]
        return matches[0] if len(matches) == 1 else None

    def _discover_input(self, result):
        """Attach the native input and its own parsed data to a result.

        Directory traversal and parsed-input reuse belong to this loader.
        The result retains only a registered file and independent parameters,
        never a reference back to the loader or its caches.
        """
        manifest = result_file_manifest(result)
        if manifest is not None or result.run_record is not None:
            # A tracked run has exactly one authoritative native input.
            # Do not search the directory when it is missing or overwritten.
            filename = manifest.input_file if manifest is not None else None
            if filename is None:
                return
            if "input" not in result.files or result.path_to("input") != os.fspath(filename):
                result.files.add_file("input", filename)
        if "input" in result.files:
            # A file named by the stdout or by the run is authoritative.
            filename = result.path_to("input")
            if (result._input_parameters is None
                    and result.__dict__.get("input_parameters") is None
                    and result.file_is_current(filename)):
                try:
                    parameters, identity = self.inputs.parse(filename)
                except _INPUT_DISCOVERY_ERRORS:
                    # An invalid explicitly named input remains registered so
                    # a later request can report its actual parse error.
                    pass
                else:
                    result.input_parameters = parameters
                    result.files["input"].set_attribution(
                        FileAttribution.LIKELY, observed_version=FileVersion(filename, identity),
                    )
                    return True
            return
        if result._input_parameters is not None:
            return
        try:
            candidates = self._directory(Path(result.directory)).inputs
        except (OSError, ValueError):
            return
        if result.output_file is not None:
            stem = Path(os.fspath(result.output_file)).stem
            paired = None
            for suffix in (".inp", ".in"):
                name = str(Path(result.directory) / (stem + suffix))
                if name in candidates:
                    paired = name
                    break
            if paired:
                candidates = (paired,)
            elif len(candidates) != 1:
                candidates = ()
        elif len(candidates) != 1:
            candidates = ()
        for filename in candidates:
            if not result.file_is_current(filename):
                continue
            try:
                parameters, identity = self.inputs.parse(filename)
            except _INPUT_DISCOVERY_ERRORS:
                continue
            if (result._task_name is not None
                    and parameters.task_name != result._task_name.upper()):
                continue
            result.input_parameters = parameters
            result.files.add_file("input", filename)
            result.files["input"].set_attribution(
                FileAttribution.LIKELY, observed_version=FileVersion(filename, identity),
            )
            return True

    def _prepare_input(self, result):
        """Resolve native inputs once the stdout FILES table has been parsed."""
        self._discover_input(result)

    def _expected_files(self, result):
        """Add named inputs/outputs even if a physical file is missing."""
        from .output_only_result import OutputOnlyResult

        # An unparseable historical stdout deliberately exposes no inferred
        # scientific inputs; preserve that conservative fallback contract.
        if isinstance(result, OutputOnlyResult):
            return
        if self._discover_input(result):
            # The output parser may have completed before the native input
            # was discovered. Add declarations now that it is available.
            result.enrich_declared_output_files(
                warn_missing=result.return_code == 0,
                file_operations=self.file_operations,
            )
        try:
            potential = result.potential_filename
        except _POTENTIAL_DISCOVERY_ERRORS:
            # No POTFIL or an invalid external input does not invalidate OUT.
            return
        if not potential:
            return
        if result.produces_potential:
            name = "converged"
        elif result.is_embedded:
            name = "cluster_potential"
        else:
            name = "potential"
        if name not in result.files:
            result.files.add_file(name, potential)

    def _associate(self, result, output):
        self._expected_files(result)
        historical = (result_file_manifest(result) is not None
                      or result.run_record is not None)
        sources = {}
        for name, item in result.files.items().items():
            try:
                path = Path(item.path()).expanduser().absolute()
            except (OSError, ValueError):
                continue
            recorded = result_file_version(result, path) if historical else None
            if name == "input" and item.observed_version is not None:
                # The input parser has already established the identity of
                # the actual bytes used to construct the parameters.
                observed = item.observed_version
                if not observed.is_current():
                    raise _FileChangedDuringReadError(f"Native input changed after parsing: {path}")
            else:
                observed = self._observe_source(path)
            sources[path] = observed
            if observed.fingerprint is None:
                observed = None
            if recorded is not None:
                # Record evidence only; OutputFileOption evaluates its state
                # once at the point where it is needed.
                status = FileAttribution.UNKNOWN
                evidence = "run_manifest_digest" if recorded.available else "run_manifest_metadata_only"
            elif historical:
                status, evidence = FileAttribution.UNKNOWN, "not_in_manifest"
            elif observed is not None:
                status, evidence = FileAttribution.LIKELY, "output_or_input_reference"
            else:
                status, evidence = FileAttribution.UNKNOWN, "missing"
            item.set_attribution(status, recorded_version=recorded,
                                 observed_version=observed, evidence=evidence)

        manifest = result_file_manifest(result)
        if manifest is not None:
            # A missing historical input/output may not have been registered
            # by the parser. Its restoration matters, but execution caches
            # alone are not result sources. Explicit result files (including
            # e.g. SCF's SFN) are already observed above.
            for version in (*manifest.input_versions, *manifest.output_versions):
                if version.kind == "file" and version.path not in sources:
                    sources[version.path] = self._observe_source(version.path)
        input_candidates = None
        if not historical and "input" not in result.files and result._input_parameters is None:
            input_candidates = self._input_candidate_versions(Path(result.directory).resolve())
        recorded_output = result.files["output"].recorded_version
        # The recorded digest may require a full read after restoration.
        # Establish historical identity here, never on a cache hit.
        result._source_snapshot = (
            _ResultSources(tuple(sources.values()), input_candidates)
            if recorded_output is None or recorded_output.is_current() else None
        )
        if not historical:
            attribution = self._scf_attribution(result, output)
            if attribution is not None:
                status, evidence, owner = attribution
                item = result.files["converged"]
                item.set_attribution(status, evidence=evidence, attributed_to=owner,
                                     observed_version=item.observed_version)

    def _scf_attribution(self, result, output):
        """Infer standalone SCF ownership without changing a parsed result."""
        if not result.produces_potential or "converged" not in result.files:
            return None
        item = result.files["converged"]
        if item.observed_version is None:
            return None
        target = Path(item.path()).resolve()
        candidates = set(self.scf_claims(output.parent, target))
        candidates.add(output)
        if len(candidates) <= 1:
            return FileAttribution.LIKELY, "output_or_input_reference", None
        newest = self._newest(candidates)
        if newest is None:
            return FileAttribution.UNKNOWN, "scf_mtime_tie_or_unreadable", None
        if newest != output:
            return FileAttribution.NOT_THIS_RESULT, "newer_scf_stdout", newest
        return FileAttribution.LIKELY, "newest_scf_stdout", None

    def _observe_source(self, path):
        version = self.file_operations.capture_version(path, digest=False)
        if version.fingerprint is not None and not stat_module.S_ISREG(version.fingerprint.mode):
            return FileVersion(path, None)
        return version

    @_file_batch
    def sources_are_current(self, result: TaskResult) -> bool:
        """Whether the source observations of a parsed result still hold.

        An unassociated/live result has no source snapshot. Missing optional
        files are valid observations, but their later appearance invalidates
        the parse just as a rewrite of an existing file does.
        Discovery of additional outputs and SCF ownership runs when importing
        a new result, not when checking already parsed sources.
        """
        snapshot = result._source_snapshot
        if snapshot is None or "output" not in result.files:
            return False
        if result.files["output"].observed_version is None:
            return False
        try:
            for version in snapshot.files:
                if version.fingerprint is None:
                    # Use the same best-effort observation as association.
                    # An optional inaccessible file must not invalidate OUT.
                    if self._observe_source(version.path).fingerprint is not None:
                        return False
                elif not version.is_current():
                    return False
            if (snapshot.input_candidates is not None
                    and self._input_candidate_versions(Path(result.directory).resolve()) != snapshot.input_candidates):
                return False
            return True
        except OSError:
            return False

    @_file_batch
    def associate_result(self, result: TaskResult) -> TaskResult:
        """Annotate an existing result (including one from a persisted run)."""
        path = Path(result.path_to("output")).expanduser().resolve()
        self._associate(result, path)
        return result

    def _input_candidate_versions(self, directory):
        """Fingerprint unresolved input candidates without parsing them."""
        versions = []
        index = self._directories.get(directory)
        names = (index.inputs if index is not None else (
            str(directory / name) for name in self.file_operations.listdir(directory)
            if Path(name).suffix.casefold() in {".inp", ".in"}
            and self.file_operations.is_file(directory / name)
        ))
        for name in sorted(names):
            try:
                versions.append((name, self.file_operations.fingerprint(name)))
            except OSError:
                versions.append((name, None))
        return tuple(versions)

    @_file_batch
    def load_file(self, file):
        path = Path(file).expanduser().resolve()
        version = self.file_operations.fingerprint(path)
        cached = self._outputs.get(path)
        if cached is not None and self.sources_are_current(cached):
            return cached

        result = TaskResult._parse_file(path, on_files_parsed=self._prepare_input)
        if self.file_operations.fingerprint(path, refresh=True) != version:
            raise _FileChangedDuringReadError(f"Output changed during parsing: {path}")
        self._associate(result, path)
        if self.file_operations.fingerprint(path, refresh=True) != version:
            raise _FileChangedDuringReadError(f"Output changed during association: {path}")
        self._outputs[path] = result
        return result

    @_file_batch
    def load_files(self, paths: Iterable[str | os.PathLike]):
        entries, failures = [], []
        for value in paths:
            path = Path(value).expanduser().resolve()
            try:
                entries.append((path, self.load_file(path)))
            except Exception as exc:
                failures.append(LoadFailure(path, exc))
        return ResultBatch(tuple(entries), tuple(failures))

    @_file_batch
    def companion_scf(self, result):
        """Find the latest parseable SCF actually producing this input POTFIL.

        The cheap header probe is only a prefilter: a parsed SCF may disagree
        with it, or fail to parse. Such a candidate must not be returned, and
        must not prevent trying the next older unambiguous candidate.
        """
        if result.produces_potential or result.output_file is None:
            return None
        try:
            source = Path(result.path_to("output")).resolve()
            potential = Path(result.potential_filename).resolve()
        except _POTENTIAL_DISCOVERY_ERRORS:
            return None
        candidates = set(self.scf_claims(source.parent, potential))
        while candidates:
            match = self._newest(candidates)
            if match is None:
                return None  # tied mtimes are insufficient provenance
            candidates.remove(match)
            try:
                companion = self.load_file(match)
            except _COMPANION_READ_ERRORS:
                # An unreadable external candidate cannot block older ones.
                continue
            if not companion.produces_potential:
                continue
            try:
                declared = Path(companion.potential_filename).resolve()
            except _POTENTIAL_DISCOVERY_ERRORS:
                continue
            if declared != potential:
                continue
            if "converged" in companion.files:
                registered = Path(companion.path_to("converged")).resolve()
                if registered != potential:
                    continue
            return companion
        return None

    @_file_batch
    def potential(self, filename, *, host=None):
        """Cache raw potentials without mutating shared SCF convergence state."""
        from ..potentials.potentials import Potential

        path = Path(filename).expanduser().resolve()
        version = self.file_operations.fingerprint(path)
        host_path = Path(host).expanduser().resolve() if host is not None else None
        host_version = self.file_operations.fingerprint(host_path) if host_path is not None else None
        key = path, version, host_path, host_version
        if key not in self._potentials:
            host_pot = self.potential(host_path) if host_path else None
            potential = Potential.from_file(
                path, atoms=False, host=host_pot, check_file_unchanged=True
            )
            if self.file_operations.fingerprint(path, refresh=True) != version or (
                host_path is not None and self.file_operations.fingerprint(host_path, refresh=True) != host_version
            ):
                raise RuntimeError(f"Potential changed during parsing: {path}")
            self._potentials[key] = potential
        return self._potentials[key]
