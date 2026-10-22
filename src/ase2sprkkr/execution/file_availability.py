"""Local file discovery shared by process and PSI/J runners.

No transport or execution state lives here. The caller decides whether the
producer's output streams are safe to consider published.
"""

from __future__ import annotations

import os

from .transport import expand_local_candidates, expand_local_files, has_file_magic
from .file_identity import same_local_file


def prepare_local_files(spec, *, output_file=None, error_file=None,
                        preserve_open_streams=False):
    """Protect inputs and caches, then clear declared outputs before launch."""
    from .base import file_path
    from .run_model import FileManifest

    files = FileManifest.from_spec(spec)
    # Use the same canonical declarations as staging and run persistence.
    protected = (
        *files.input_files,
        *(entry.local_path for entry in files.caches),
    )
    output = files.output_file if output_file is None else file_path(output_file, spec.directory)
    error = (
        None if files.stderr is None else files.stderr.local_path
    ) if error_file is None else file_path(error_file, spec.directory)
    streams = tuple(path for path in (output, error) if path is not None)
    declared = tuple(entry.local_path for entry in files.outputs)
    validate_output_collisions((*declared, *streams), protected=protected)
    # Never unlink streams already open: that would orphan their descriptors.
    outputs = declared if preserve_open_streams else (*declared, *streams)
    protected_paths = (*protected, *streams) if preserve_open_streams else protected
    clear_declared_outputs(outputs, protected=protected_paths)



def _pattern_root(pattern):
    """Find an existing literal ancestor for a local wildcard declaration."""
    directory = os.path.dirname(os.path.abspath(os.fspath(pattern)))
    while has_file_magic(directory):
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return directory


def materialize_manifest(manifest, *, remote=False, include_streams=True):
    """Resolve local filesystem declarations and return (manifest, first_error).

    This is the only wildcard evaluator for job outputs. Missing mandatory
    remote output patterns are failures; optional output/cache patterns are
    skipped. Exceptional errors preserve successful matches of other patterns.
    The caller owns lifecycle and persistence.
    """
    from dataclasses import replace
    from .run_model import FileManifest

    if not isinstance(manifest, FileManifest):
        manifest = FileManifest(**manifest)
    first_error = None

    def resolve(entry, *, mandatory=False):
        nonlocal first_error
        if entry is None:
            return ()
        if entry.local_path is None:
            return (entry,)
        if entry.availability_state not in (None, "pending"):
            return (entry,)
        pattern = entry.remote_path if remote else entry.local_path
        if pattern is None:
            return (entry,)
        if not has_file_magic(pattern):
            if remote:
                return (entry,)
            return (replace(entry, availability_state=(
                "available" if os.path.isfile(pattern) else "skipped"
            )),)
        if remote:
            if manifest.mapping is None:
                if first_error is None:
                    first_error = ValueError(
                        "Remote wildcard materialization requires a file mapping"
                    )
                return (entry,)
            # expand_local_files returns canonical paths below its resolved root.
            root = os.path.realpath(os.path.expanduser(manifest.mapping.remote_root))
        else:
            root = _pattern_root(pattern)
        try:
            matches = expand_local_files(pattern, root)
        except (OSError, ValueError) as exc:
            if first_error is None:
                first_error = exc
            # A failed filesystem scan is not an empty wildcard. Preserve
            # the unresolved declaration in the frozen receipt so the client
            # can distinguish it from a genuinely unmatched optional output.
            return (entry,)
        if remote and not matches:
            if not mandatory:
                return (replace(entry, availability_state="skipped"),)
            exc = FileNotFoundError(
                f"Declared remote output pattern matched no files: {pattern}"
            )
            if first_error is None:
                first_error = exc
            return (replace(
                entry, availability_state="failed",
                availability_message=str(exc),
            ),)
        found = []
        for path in matches:
            if remote:
                relative = os.path.relpath(path, root)
                local_path = os.path.join(manifest.mapping.local_root, relative)
                remote_path = path
                state = "pending"
            else:
                local_path = path
                remote_path = entry.remote_path
                state = "available"
            found.append(replace(
                entry, local_path=local_path, remote_path=remote_path,
                source=entry.local_path, availability_state=state,
            ))
        return tuple(found)

    outputs = tuple(concrete for entry in manifest.outputs
                    for concrete in resolve(entry, mandatory=remote and not entry.optional))
    # Remote-only cache entries carry pre-run signatures, not files.
    # Apply those baselines to the frozen concrete entries and drop the
    # placeholders once membership is known.
    baselines = {
        cache.remote_path: cache.pre_run_signature
        for cache in manifest.caches if cache.local_path is None
    } if remote else {}
    caches = tuple(
        replace(concrete,
                pre_run_signature=baselines.get(
                    concrete.remote_path, cache.pre_run_signature))
        for cache in manifest.caches
        if not remote or cache.local_path is not None
        for concrete in resolve(cache)
    )
    stdout, stderr = manifest.stdout, manifest.stderr
    if not remote and include_streams:
        stdout = next(iter(resolve(stdout)), None)
        stderr = next(iter(resolve(stderr)), None)
    return replace(manifest, outputs=outputs, caches=caches,
                   stdout=stdout, stderr=stderr), first_error


def discover_local_files(files, *, include_streams=True):
    """Read local files using the common materializer."""
    materialized, error = materialize_manifest(files, include_streams=include_streams)
    if error is not None:
        raise error
    return materialized


def _expanded_protected_paths(protected):
    """Expand protected path declarations once, retaining their source names."""
    for value in protected:
        if value is None:
            continue
        declaration = os.path.abspath(os.fspath(value))
        for candidate in expand_local_candidates(declaration):
            yield declaration, os.path.abspath(candidate)


def validate_output_collisions(streams, *, protected=()):
    """Reject explicit output paths overlapping declared inputs or restart caches.

    Broad output patterns retain their existing behavior: matching protected
    files are skipped by clear_declared_outputs rather than removed.
    """
    protected_paths = tuple(_expanded_protected_paths(protected))
    for stream in streams:
        if stream is None:
            continue
        target = os.path.abspath(os.fspath(stream))
        if has_file_magic(target):
            continue
        for declaration, candidate in protected_paths:
            if same_local_file(target, candidate):
                raise ValueError(
                    f"Output {target} overlaps protected input/cache {declaration}"
                )


async def probe_submission_outputs(manifest, transport, submitted_at, *, remote=False):
    """Read-only tri-state evidence from declared outputs, including streams.

    True means all required outputs are newer than dispatch, False means no
    outputs exist, and None means partial/old results or inconclusive evidence.
    """
    from datetime import datetime, timezone
    from .transport import has_file_magic

    when = datetime.fromisoformat(submitted_at)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    since_ns = int(when.timestamp() * 1_000_000_000)
    entries = (manifest.stdout, manifest.stderr, *manifest.outputs)
    required = tuple(entry for entry in entries if entry is not None and not entry.optional)
    if not required:
        return None

    inspected = set()
    fresh = missing = old = 0
    for entry in required:
        path = entry.remote_path if remote else entry.local_path
        if path is None:
            return None
        if has_file_magic(path):
            root = manifest.mapping.remote_root if remote and manifest.mapping else (
                None if remote else _pattern_root(path)
            )
            if root is None:
                return None
            matches = await transport.expand_files(path, root)
            if not matches:
                missing += 1
                continue
        else:
            matches = (path,)
        for match in matches:
            if match in inspected:
                continue
            inspected.add(match)
            signature = await transport.file_signature(match)
            if signature is None:
                missing += 1
            elif signature[1] >= since_ns:
                fresh += 1
            else:
                old += 1
    if fresh and not (missing or old):
        return True
    if not (fresh or old):
        return False
    return None


def clear_declared_outputs(paths, *, protected=()):
    """Remove old declared results without destroying inputs or restart caches.

    Called on the execution host before starting a new calculation, never on
    reattach. A wildcard removes only its concrete files.
    """
    protected_targets = {
        os.path.realpath(candidate)
        for _, candidate in _expanded_protected_paths(protected)
    }
    for declared in dict.fromkeys(os.fspath(p) for p in paths if p is not None):
        for candidate in expand_local_candidates(declared):
            path = os.path.abspath(candidate)
            if os.path.realpath(path) in protected_targets or not os.path.isfile(path):
                continue
            os.unlink(path)
