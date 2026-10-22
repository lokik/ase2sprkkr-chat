"""Persistent, non-redundant identity and file layout of one execution.

Submission intent and lifecycle are properties of a RunRecord, not backend
references.  This module contains only serializable value objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

from ..common.file_versions import FileState, FileVersion

from .endpoint import ExecutorEndpoint
from .file_identity import same_local_file


@dataclass(frozen=True)
class WorkingDirectory:
    """Local working directory and its optional target-side counterpart."""

    local: str
    remote: Optional[str] = None

    def __post_init__(self):
        import os

        object.__setattr__(self, "local", os.fspath(self.local))
        if self.remote is not None:
            object.__setattr__(self, "remote", os.fspath(self.remote))

    @property
    def target(self):
        return self.remote if self.remote is not None else self.local


@dataclass(frozen=True)
class Execution:
    """Backend identity plus the directory in which the job executes."""

    endpoint: ExecutorEndpoint
    directory: WorkingDirectory
    backend_id: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.endpoint, dict):
            object.__setattr__(self, "endpoint", ExecutorEndpoint(**self.endpoint))
        if isinstance(self.directory, dict):
            object.__setattr__(self, "directory", WorkingDirectory(**self.directory))
        if self.backend_id is not None:
            identifier = str(self.backend_id)
            if not identifier:
                raise ValueError("backend_id cannot be empty")
            object.__setattr__(self, "backend_id", identifier)


    def submitted(self, backend_id: str):
        """Return the execution bound to its immutable backend identity."""
        submitted = replace(self, backend_id=backend_id)
        if self.backend_id is not None and self.backend_id != submitted.backend_id:
            raise ValueError("Cannot replace an assigned backend identity")
        return submitted



class FileAvailabilityState(str, Enum):
    """Availability of a declared local file."""

    PENDING = "pending"
    AVAILABLE = "available"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class FileEntry:
    """Location of one file, with no assumptions about its content."""

    local_path: Optional[str] = None
    remote_path: Optional[str] = None
    availability_state: Optional[str] = None
    availability_message: Optional[str] = None
    source: Optional[str] = None
    pre_run_signature: Optional[tuple] = None
    optional: bool = False

    def __post_init__(self):
        import os
        if self.source is not None:
            object.__setattr__(self, "source", os.fspath(self.source))

        if self.local_path is not None:
            object.__setattr__(self, "local_path", os.fspath(self.local_path))
        if self.remote_path is not None:
            object.__setattr__(self, "remote_path", os.fspath(self.remote_path))
        if self.pre_run_signature is not None:
            object.__setattr__(self, "pre_run_signature", tuple(self.pre_run_signature))
        if type(self.optional) is not bool:
            raise TypeError("FileEntry.optional must be a bool")
        if self.local_path is None and self.remote_path is None:
            raise ValueError("A file entry must have at least one location")
        if self.availability_state is not None:
            try:
                availability = FileAvailabilityState(self.availability_state)
            except ValueError as exc:
                raise ValueError("Invalid per-file availability state") from exc
            # Persist JSON-compatible strings, not implementation-specific enums.
            object.__setattr__(self, "availability_state", availability.value)
        if (
            self.availability_message is not None
            and self.availability_state != FileAvailabilityState.FAILED
        ):
            raise ValueError("An availability error belongs to a failed file")


class FileSyncPolicy(str, Enum):
    """Policy controlling transfers between local and remote workspaces."""

    CHANGED = "changed"
    ALWAYS = "always"
    MISSING = "missing"
    NEVER = "never"

    def should_transfer(self, source_digest, destination_digest):
        """Apply the direction-independent transfer rule to file digests.

        The caller handles missing mandatory files separately; a missing
        source is never transferable, including with ALWAYS.
        """
        if self == FileSyncPolicy.NEVER or source_digest is None:
            return False
        if self == FileSyncPolicy.ALWAYS:
            return True
        if self == FileSyncPolicy.MISSING:
            return destination_digest is None
        return source_digest != destination_digest


@dataclass(frozen=True)
class FileMapping:
    """Resolved mapping and transfer policy of this run's file manifest."""

    local_root: str
    remote_root: str
    input_sync: str = FileSyncPolicy.CHANGED.value
    output_sync: str = FileSyncPolicy.CHANGED.value

    def __post_init__(self):
        import os

        object.__setattr__(self, "local_root", os.fspath(self.local_root))
        object.__setattr__(self, "remote_root", os.fspath(self.remote_root))
        try:
            input_sync = FileSyncPolicy(self.input_sync)
            output_sync = FileSyncPolicy(self.output_sync)
        except ValueError as exc:
            raise ValueError("Invalid file synchronization policy") from exc
        object.__setattr__(self, "input_sync", input_sync.value)
        object.__setattr__(self, "output_sync", output_sync.value)

    def map_path(self, path):
        """Map an existing local path without querying the target service."""
        import os
        import posixpath
        from pathlib import Path
        root = Path(self.local_root).expanduser().resolve()
        local = Path(path).expanduser().resolve()
        try:
            relative = local.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Local path {local} is outside mapped root {root}") from exc
        return posixpath.join(self.remote_root, *relative.parts)

    def unmap_path(self, path):
        """Map target path back, rejecting lexical and symlink escapes."""
        import os
        import posixpath
        from pathlib import Path, PurePosixPath
        remote = posixpath.normpath(os.fspath(path))
        if not posixpath.isabs(remote):
            raise ValueError(f"Remote path must be absolute: {remote}")
        try:
            inside = posixpath.commonpath((self.remote_root, remote)) == self.remote_root
        except ValueError:
            inside = False
        if not inside:
            raise ValueError(f"Remote path {remote} is outside mapped root {self.remote_root}")
        relative = posixpath.relpath(remote, self.remote_root)
        root = Path(self.local_root).expanduser().resolve()
        local = root if relative == "." else root.joinpath(*PurePosixPath(relative).parts)
        try:
            local.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Local path {local} is outside mapped root {root}") from exc
        return local


def _validate_file_declarations(entries, *, primary_input=None):
    """Reject duplicate file identities, except intentionally merged streams.

    primary_input is a reference to an inputs entry, not an extra declaration.
    Remote-only cache baselines are metadata, not file declarations.
    """
    import os
    import posixpath
    import fnmatch
    from .transport import has_file_magic

    declared = []
    for role, entry in entries:
        if entry is None:
            continue
        if role == "cache" and entry.local_path is None:
            continue
        for previous_role, previous in declared:
            same_local = same_local_file(entry.local_path, previous.local_path)
            same_remote = (
                entry.remote_path is not None and previous.remote_path is not None
                and posixpath.normpath(entry.remote_path)
                == posixpath.normpath(previous.remote_path)
            )
            roles = {role, previous_role}
            if roles in ({"input", "stdin"}, {"stdout", "output"}, {"stderr", "output"}):
                for literal, pattern in ((entry.local_path, previous.local_path),
                                         (previous.local_path, entry.local_path)):
                    if (literal is not None and pattern is not None
                            and not has_file_magic(literal)
                            and has_file_magic(pattern)
                            and fnmatch.fnmatchcase(os.path.normpath(literal),
                                                   os.path.normpath(pattern))):
                        same_local = True
            if roles in ({"stdout", "output"}, {"stderr", "output"}):
                for literal, pattern in ((entry.remote_path, previous.remote_path),
                                         (previous.remote_path, entry.remote_path)):
                    if (literal is not None and pattern is not None
                            and not has_file_magic(literal)
                            and has_file_magic(pattern)
                            and fnmatch.fnmatchcase(posixpath.normpath(literal),
                                                   posixpath.normpath(pattern))):
                        same_remote = True
            if not same_local and not same_remote:
                continue
            if (roles == {"input", "stdin"} and primary_input is not None
                    and same_local
                    and same_local_file(entry.local_path, primary_input.local_path)
                    and same_local_file(previous.local_path, primary_input.local_path)):
                continue
            # An expanded wildcard may repeat an exact output/cache member,
            # but the same local destination must not have different sources.
            # Cross-role output/cache collisions are never permitted.
            if (same_local
                    and (entry.source is not None or previous.source is not None)
                    and role == previous_role and role in ("output", "cache")
                    and (entry.remote_path is None or previous.remote_path is None
                         or same_remote)):
                continue
            # Multiple local destinations may deliberately retrieve the same
            # remote result. Only a shared *destination* is a collision.
            if (same_remote and not same_local
                    and role == previous_role and role in ("output", "cache")):
                continue
            if {role, previous_role} == {"stdout", "stderr"}:
                if (entry.local_path is None or previous.local_path is None
                        or not same_local
                        or (entry.remote_path is not None
                            and previous.remote_path is not None
                            and not same_remote)):
                    raise ValueError("Merged stdout/stderr must share one file")
                continue
            raise ValueError(
                f"Duplicate file declaration: {previous_role} and {role} "
                f"refer to the same file"
            )
        declared.append((role, entry))


@dataclass(frozen=True)
class FileManifest:
    """All paths and cache baselines required for transfer and reconnect."""

    inputs: tuple[FileEntry, ...] = ()
    outputs: tuple[FileEntry, ...] = ()
    caches: tuple[FileEntry, ...] = ()
    stdout: Optional[FileEntry] = None
    stderr: Optional[FileEntry] = None
    mapping: Optional[FileMapping] = None
    input_versions: tuple[FileVersion, ...] = ()
    output_versions: tuple[FileVersion, ...] = ()
    captured_at: Optional[str] = None
    primary_input: Optional[FileEntry] = None
    stdin: Optional[FileEntry] = None

    @classmethod
    def from_spec(cls, spec):
        """Build the local file manifest for an execution specification."""
        import os

        from .base import file_path

        directory = os.path.abspath(os.fspath(spec.directory))

        def entries(*values):
            return tuple(
                FileEntry(local_path=path, optional=getattr(value, "optional", False))
                for value in values
                if (path := file_path(value, directory)) is not None
            )

        primary = file_path(spec.input_file, directory)
        # Every independent input declaration is validated, never coalesced.
        # primary_input is only a reference to the first entry, and stdin may
        # explicitly reference that same primary file.
        inputs = entries(spec.input_file, *spec.input_files)
        stdin = file_path(spec.stdin, directory)
        stdout = file_path(spec.output_file, directory)
        stderr = file_path(spec.error_file, directory)
        outputs = entries(*spec.output_files)
        caches = entries(*spec.cache_files)

        return cls(
            inputs=inputs,
            primary_input=None if primary is None else FileEntry(local_path=primary),
            stdin=None if stdin is None else FileEntry(local_path=stdin),
            outputs=outputs,
            caches=caches,
            stdout=None if stdout is None else FileEntry(local_path=stdout),
            stderr=None if stderr is None else FileEntry(local_path=stderr),
        ).initialize_availability()

    def __post_init__(self):
        for name in ("inputs", "outputs", "caches"):
            object.__setattr__(
                self, name,
                tuple(
                    item if isinstance(item, FileEntry) else FileEntry(**item)
                    for item in getattr(self, name)
                ),
            )
        for name in ("stdout", "stderr", "primary_input", "stdin"):
            item = getattr(self, name)
            if isinstance(item, dict):
                object.__setattr__(self, name, FileEntry(**item))
        if isinstance(self.mapping, dict):
            object.__setattr__(self, "mapping", FileMapping(**self.mapping))
        for name in ("input_versions", "output_versions"):
            object.__setattr__(
                self, name,
                tuple(
                    version if isinstance(version, FileVersion) else FileVersion(**version)
                    for version in getattr(self, name)
                ),
            )
        _validate_file_declarations((
            *(("input", entry) for entry in self.inputs),
            ("stdin", self.stdin),
            ("stdout", self.stdout),
            ("stderr", self.stderr),
            *(("output", entry) for entry in self.outputs),
            *(("cache", entry) for entry in self.caches),
        ), primary_input=self.primary_input)

    def _merge_availability(self, received, preserve):
        """Reconcile matching files without altering the received membership.

        FileManifest already rejects output/cache identity collisions.
        """
        # Remote-only caches are included even though FileManifest.files
        # deliberately omits them. Merged streams share one physical identity.
        recorded = {}
        priority = {"failed": 3, "available": 2, "skipped": 1, "pending": 0, None: 0}
        for entry in (self.stdout, self.stderr,
                      *self.outputs, *self.caches):
            if entry is None:
                continue
            identity = (entry.local_path, entry.remote_path)
            prior = recorded.get(identity)
            if (prior is None or
                    priority[entry.availability_state] > priority[prior.availability_state]):
                recorded[identity] = entry

        def merged(entry):
            if entry is None:
                return None
            prior = recorded.get((entry.local_path, entry.remote_path))
            if prior is None or not preserve(prior, entry):
                return entry
            if (entry.availability_state == prior.availability_state
                    and entry.availability_message == prior.availability_message):
                return entry
            return replace(
                entry, availability_state=prior.availability_state,
                availability_message=prior.availability_message,
            )

        return received._map_files(merged)

    def discard_capture(self):
        """Invalidate speculative captured versions, preserving file statuses."""
        return replace(self, captured_at=None, input_versions=(), output_versions=())

    def merge(self, requested):
        """Do not let stale observations erase independently persisted delivery."""
        if isinstance(requested, dict):
            requested = FileManifest(**requested)
        if self.captured_at is not None:
            if requested != self:
                raise ValueError("Cannot replace a captured file manifest")
            return self

        def preserve(prior, entry):
            if prior.availability_state not in ("available", "failed", "skipped"):
                return False
            if entry.availability_state == "pending":
                return True
            # A concurrent failed transfer invalidates speculative capture.
            return (
                requested.captured_at is not None
                and entry.availability_state in ("available", "skipped")
                and prior.availability_state == "failed"
            )

        return self._merge_availability(requested, preserve)

    def merge_terminal(self, received):
        """Retain only still-present local deliveries in terminal facts."""
        def preserve(prior, entry):
            if entry.availability_state in ("failed", "skipped"):
                return False
            if prior.availability_state == "skipped":
                return True
            return (
                prior.availability_state == "available"
                and entry.local_path is not None
                and os.path.isfile(entry.local_path)
            )

        return self._merge_availability(received, preserve)

    @property
    def files(self):
        """Distinct downloadable file states; shared streams count only once.

        Output/cache destination collisions are rejected at declaration time.
        Differing availability states must still remain separately visible.
        """
        unique = {}
        for entry in (self.stdout, self.stderr, *self.outputs, *self.caches):
            if entry is None or entry.local_path is None:
                continue
            key = (
                entry.local_path, entry.remote_path,
                entry.availability_state, entry.availability_message,
            )
            unique.setdefault(key, entry)
        return tuple(unique.values())

    @property
    def failed_files(self):
        return tuple(x for x in self.files if x.availability_state == "failed")

    @property
    def availability_state(self):
        """Derived summary for callers; never stored independently."""
        statuses = tuple(x.availability_state for x in self.files)
        if "failed" in statuses:
            return "failed"
        if not statuses or None in statuses or "pending" in statuses:
            return "pending"
        if all(x == "skipped" for x in statuses):
            return "skipped"
        return "available"

    @property
    def availability_message(self):
        failures = self.failed_files
        if failures:
            return failures[0].availability_message
        return None

    def with_mapping(self, mapping: FileMapping):
        """Resolve all roles once and reject any change to a reserved target.

        FileManifest owns placement. Reservation and staging must use this
        identical validation before any file transfer takes place.
        """
        if self.mapping is not None and self.mapping != mapping:
            raise ValueError("Submission mapping differs from reserved mapping")

        def mapped(entry):
            if entry is None or entry.local_path is None:
                return entry
            remote = mapping.map_path(entry.local_path)
            if entry.remote_path is not None and entry.remote_path != remote:
                raise ValueError("Submission path differs from reserved path")
            return entry if entry.remote_path == remote else replace(
                entry, remote_path=remote,
            )

        return replace(
            self,
            inputs=tuple(mapped(entry) for entry in self.inputs),
            primary_input=mapped(self.primary_input),
            stdin=mapped(self.stdin),
            outputs=tuple(mapped(entry) for entry in self.outputs),
            caches=tuple(mapped(entry) for entry in self.caches),
            stdout=mapped(self.stdout),
            stderr=mapped(self.stderr),
            mapping=mapping,
        )

    def with_remote_stream_paths(self, *, stdout=None, stderr=None):
        """Update recovered stream locations without changing file identity.

        Missing or empty recovered paths do not replace persisted locations.
        A stream that was not recorded remains absent.
        """
        changes = {}
        for name, path in (("stdout", stdout), ("stderr", stderr)):
            entry = getattr(self, name)
            if not path or entry is None or path == entry.remote_path:
                continue
            if self.mapping is not None and entry.local_path is not None:
                expected = self.mapping.map_path(entry.local_path)
                if path != expected:
                    raise ValueError(
                        f"Recovered {name} path {path} does not match "
                        f"the file mapping: {expected}"
                    )
            changes[name] = replace(entry, remote_path=path)
        return replace(self, **changes) if changes else self

    def initialize_availability(self):
        """Mark declared downloadable files pending without marking inputs."""
        return self._map_files(
            lambda item: replace(item, availability_state="pending")
            if item.availability_state is None else item
        )

    def _map_files(self, transform, *, kind=None):
        """Transform delivery state for selected file roles without copying unchanged facts."""
        if kind not in (None, "output", "cache"):
            raise ValueError(f"Unknown file kind: {kind!r}")
        output_files = kind in (None, "output")
        cache_files = kind in (None, "cache")
        stdout = (
            None if self.stdout is None else transform(self.stdout)
        ) if output_files else self.stdout
        stderr = (
            None if self.stderr is None else transform(self.stderr)
        ) if output_files else self.stderr
        outputs = tuple(transform(entry) for entry in self.outputs) if output_files else self.outputs
        caches = tuple(transform(entry) for entry in self.caches) if cache_files else self.caches
        if (stdout == self.stdout and stderr == self.stderr
                and outputs == self.outputs and caches == self.caches):
            return self
        return replace(
            self, stdout=stdout, stderr=stderr,
            outputs=outputs, caches=caches,
        )

    def retry_failed_downloads(self):
        """Reopen only failed deliveries, never input declarations.

        A successful earlier download stays available across retry. The
        original backend identity and all file membership remain unchanged.
        """
        return self._map_files(
            lambda item: replace(
                item, availability_state="pending", availability_message=None,
            ) if item.availability_state == "failed" else item
        )

    def with_file_availability(
        self, local_path, state, message=None, *,
        remote_path=None, kind=None,
    ):
        """Update one declared file identity; never register new members.

        A shared stdout/stderr destination is one output identity when both
        entries name the same remote file. Ambiguous output/cache destinations
        or distinct remote sources require an explicit kind or remote path.
        """
        import os

        if kind not in (None, "output", "cache"):
            raise ValueError(f"Unknown file kind: {kind!r}")
        path = os.fspath(local_path)
        groups = (
            ("output", (self.stdout, self.stderr, *self.outputs)),
            ("cache", self.caches),
        )
        identities = {
            (category, entry.remote_path)
            for category, entries in groups if kind is None or kind == category
            for entry in entries
            if entry is not None and entry.local_path == path
            and (remote_path is None or entry.remote_path == remote_path)
        }
        if len(identities) > 1:
            raise ValueError(
                f"Ambiguous file destination {path}: specify kind or remote_path"
            )
        if identities:
            selected_kind, selected_remote = next(iter(identities))

            def update(entry):
                if (entry is not None and entry.local_path == path
                        and entry.remote_path == selected_remote):
                    return replace(
                        entry, availability_state=state,
                        availability_message=message,
                    )
                return entry

            return self._map_files(update, kind=selected_kind)

        raise KeyError(f"File is not declared in manifest: {path}")

    def has_wildcards(self):
        """Whether any declared output/cache path still needs materializing."""
        from .transport import has_file_magic
        return any(
            entry is not None and
            has_file_magic(entry.remote_path or entry.local_path)
            for entry in (*self.outputs, *self.caches)
            if entry.local_path is not None
        )

    @property
    def input_file(self):
        return None if self.primary_input is None else self.primary_input.local_path

    @property
    def input_entries(self):
        """All read inputs, including stdin without making it primary."""
        return (*self.inputs, self.stdin) if self.stdin is not None else self.inputs

    @property
    def output_file(self):
        return self.stdout.local_path if self.stdout is not None else None

    @property
    def input_files(self):
        return tuple(dict.fromkeys(entry.local_path for entry in self.input_entries
                                   if entry.local_path is not None))

    @property
    def output_files(self):
        streams = (self.stdout, self.stderr)
        entries = tuple(entry for entry in streams if entry is not None)
        return tuple(dict.fromkeys(
            entry.local_path
            for entry in (*entries, *self.outputs)
            if entry.local_path is not None
        ))

    @property
    def manifests(self):
        return tuple(dict.fromkeys((*self.input_files, *self.output_files)))

    @property
    def all_versions(self):
        by_path = {
            str(item.path): item
            for item in (*self.input_versions, *self.output_versions)
        }
        return tuple(by_path[key] for key in sorted(by_path))

    @property
    def state(self):
        if self.captured_at is None:
            return FileState.CURRENT
        states = tuple(version.state() for version in self.output_versions)
        if all(value is FileState.CURRENT for value in states):
            return FileState.CURRENT
        if FileState.OVERWRITTEN in states:
            return FileState.OVERWRITTEN
        if FileState.UNAVAILABLE in states:
            return FileState.UNAVAILABLE
        return FileState.MISSING
