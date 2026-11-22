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

import pyparsing as pp

from kkrunner.files import FileFingerprint, FileOperations

from ..common.warnings import DataValidityError
from ..input_parameters import input_parameters as input_parameters
from .file_provenance import result_file_manifest, result_file_version
from .output_detection import looks_like_scf_output
from .result_options import FileAttribution
from .task_result import TaskResult


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
                    raise RuntimeError(f"Native input changed during parsing: {path}")
            except Exception as exc:
                self._parameters[path] = (identity, exc)
                raise
            self._parameters[path] = (identity, parameters)
        else:
            parameters = entry[1]
            if isinstance(parameters, Exception):
                raise parameters
        # Parameters are mutable; every result receives its own detached copy.
        if isinstance(parameters, input_parameters.InputParameters):
            return parameters.copy(copy_values=True)
        return deepcopy(parameters)


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
        if "input" in result.files:
            # A file named by the stdout or by the run is authoritative.
            filename = result.path_to("input")
            if (result._input_parameters is None
                    and result.__dict__.get("input_parameters") is None
                    and result.file_is_current(filename)):
                try:
                    result.input_parameters = self.inputs.parse(filename)
                    return True
                except Exception:
                    # The explicit file remains registered. A later request
                    # for input_parameters must still report its parse error.
                    pass
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
                parameters = self.inputs.parse(filename)
            except (OSError, UnicodeError, ValueError, pp.ParseBaseException, DataValidityError):
                continue
            if (result._task_name is not None
                    and parameters.task_name != result._task_name.upper()):
                continue
            result.input_parameters = parameters
            result.files.add_file("input", filename)
            return True

    def _prepare_input(self, result):
        """Resolve native inputs once the stdout FILES table has been parsed."""
        try:
            self._discover_input(result)
        except Exception:
            # Unrelated or invalid INP files must not break OUT parsing.
            pass

    def _expected_files(self, result):
        """Add named inputs/outputs even if a physical file is missing."""
        from .output_only_result import OutputOnlyResult

        # An unparseable historical stdout deliberately exposes no inferred
        # scientific inputs; preserve that conservative fallback contract.
        if isinstance(result, OutputOnlyResult):
            return
        try:
            if self._discover_input(result):
                # The output parser may have completed before the native input
                # was discovered. Add declarations now that it is available.
                result.enrich_declared_output_files(
                    warn_missing=getattr(result, "return_code", None) == 0,
                    file_operations=self.file_operations,
                )
        except Exception:
            # Optional input discovery must not make a parsed OUT unusable.
            pass
        try:
            potential = result.potential_filename
        except Exception:
            return
        if not potential or not hasattr(result.files, "add_file"):
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
                      or getattr(result, "run_record", None) is not None)
        file_entries = result.files.items()
        if hasattr(file_entries, "items"):
            file_entries = file_entries.items()
        for name, item in file_entries:
            if not hasattr(item, "set_attribution"):
                continue
            try:
                path = Path(item.path()).expanduser().resolve()
            except (OSError, ValueError, TypeError):
                continue
            recorded = result_file_version(result, path) if historical else None
            observed = None
            try:
                version = self.file_operations.capture_version(path, digest=False)
                if version.fingerprint is not None and stat_module.S_ISREG(version.fingerprint.mode):
                    observed = version
            except OSError:
                pass
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

        if not result.produces_potential or "converged" not in result.files:
            return
        item = result.files["converged"]
        if not hasattr(item, "association"):
            return
        if historical or item.observed_version is None:
            # Historical run evidence is authoritative; standalone SCF
            # heuristics need an observed physical potential. Do not compute
            # a redundant FileVersion.state() to make this decision.
            return
        target = Path(item.path()).resolve()
        candidates = set(self.scf_claims(output.parent, target))
        candidates.add(output)
        if len(candidates) <= 1:
            return
        newest = self._newest(candidates)
        if newest is None:
            item.set_attribution(FileAttribution.UNKNOWN,
                                 observed_version=item.observed_version,
                                 evidence="scf_mtime_tie_or_unreadable")
        elif newest != output:
            item.set_attribution(FileAttribution.NOT_THIS_RESULT,
                                 observed_version=item.observed_version,
                                 evidence="newer_scf_stdout", attributed_to=newest)
        else:
            item.set_attribution(FileAttribution.LIKELY,
                                 observed_version=item.observed_version,
                                 evidence="newest_scf_stdout")

    @_file_batch
    def associate_result(self, result: TaskResult) -> TaskResult:
        """Annotate an existing result (including one from a persisted run)."""
        path = Path(result.path_to("output")).expanduser().resolve()
        self._associate(result, path)
        return result

    def _input_candidate_versions(self, directory):
        """Fingerprint unresolved input candidates without parsing them."""
        versions = []
        for name in self._directory(directory).inputs:
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
        directory_version = self._directory(path.parent).version
        same_sources = False
        if cached is not None and cached[0] == version and cached[2] == directory_version:
            result = cached[1]
            input_file = result.files["input"] if "input" in result.files else None
            observed = getattr(input_file, "observed_version", None)
            if observed is not None:
                same_sources = observed.is_current()
            elif input_file is None:
                # Failed or ambiguous input discovery can become resolvable
                # without changing the directory mtime (in-place INP edit).
                same_sources = cached[3] == self._input_candidate_versions(path.parent)
            if same_sources:
                # Not just INPUT: a parsed result may already cache its
                # POTFIL or auxiliary data. If those bytes changed, reuse of
                # the TaskResult object would return stale cached properties.
                for item in result.files.items().values():
                    observed_file = getattr(item, "observed_version", None)
                    if (observed_file is not None
                            and not observed_file.is_current()):
                        same_sources = False
                        break

        if not same_sources:
            result = TaskResult._parse_file(path, on_files_parsed=self._prepare_input)
            if self.file_operations.fingerprint(path, refresh=True) != version:
                raise RuntimeError(f"Output changed during parsing: {path}")
        self._associate(result, path)
        if self.file_operations.fingerprint(path, refresh=True) != version:
            raise RuntimeError(f"Output changed during association: {path}")
        if not same_sources:
            input_signature = (
                self._input_candidate_versions(path.parent)
                if "input" not in result.files else None
            )
            # Commit the parsed result to the cache only after association.
            self._outputs[path] = (version, result, directory_version, input_signature)
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
        except (OSError, ValueError, KeyError, AttributeError):
            return None
        candidates = set(self.scf_claims(source.parent, potential))
        while candidates:
            match = self._newest(candidates)
            if match is None:
                return None  # tied mtimes are insufficient provenance
            candidates.remove(match)
            try:
                companion = self.load_file(match)
                if not companion.produces_potential:
                    continue
                declared = Path(companion.potential_filename).resolve()
                if declared != potential:
                    continue
                if "converged" in companion.files:
                    registered = Path(companion.path_to("converged")).resolve()
                    if registered != potential:
                        continue
                return companion
            except Exception:
                # An unreadable or inconsistent candidate cannot block others.
                continue
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
