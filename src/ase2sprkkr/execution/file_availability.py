"""Local file discovery shared by process and PSI/J runners.

No transport or execution state lives here. The caller decides whether the
producer's output streams are safe to consider published.
"""

from __future__ import annotations

import os

from .transport import expand_local_candidates, expand_local_files, has_file_magic


def prepare_local_files(spec, *, output_file=None, error_file=None,
                        preserve_open_streams=False):
    """Protect inputs and caches, then clear declared outputs before launch."""
    from .base import file_path
    directory = spec.directory

    def paths(values):
        return tuple(
            path for value in values
            if (path := file_path(value, directory)) is not None
        )

    protected = paths((
        spec.input_file, spec.stdin, *spec.input_files, *spec.cache_files,
    ))
    streams = paths((
        spec.output_file if output_file is None else output_file,
        spec.error_file if error_file is None else error_file,
    ))
    declared = paths(spec.output_files)
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
        root = manifest.mapping.remote_root if remote else _pattern_root(pattern)
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
                relative = os.path.relpath(path, manifest.mapping.remote_root)
                if relative == ".." or relative.startswith(".." + os.sep):
                    if first_error is None:
                        first_error = ValueError("Materialized file is outside remote root")
                    continue
                found.append(replace(
                    entry, local_path=os.path.join(manifest.mapping.local_root, relative),
                    remote_path=path, source=entry.local_path,
                    availability_state="pending",
                ))
            else:
                found.append(replace(
                    entry, local_path=path, source=entry.local_path,
                    availability_state="available",
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
    protected_paths = tuple(
        (declaration, candidate, os.path.realpath(candidate))
        for declaration, candidate in _expanded_protected_paths(protected)
    )
    for stream in streams:
        if stream is None:
            continue
        target = os.path.abspath(os.fspath(stream))
        if has_file_magic(target):
            continue
        resolved_target = os.path.realpath(target)
        for declaration, candidate, resolved_candidate in protected_paths:
            collision = resolved_target == resolved_candidate
            if not collision:
                # Existing distinct hardlinks may refer to the same inode.
                try:
                    collision = os.path.samefile(target, candidate)
                except (FileNotFoundError, NotADirectoryError):
                    pass
            if collision:
                raise ValueError(
                    f"Output {target} overlaps protected input/cache {declaration}"
                )


def clear_declared_outputs(paths, *, protected=()):
    """Remove old declared results without destroying inputs or restart caches.

    Called on the execution host before starting a new calculation, never on
    reattach. A wildcard removes only its concrete files.
    """
    protected_paths = {candidate for _, candidate in _expanded_protected_paths(protected)}
    protected_targets = {os.path.realpath(path) for path in protected_paths}
    for declared in dict.fromkeys(os.fspath(p) for p in paths if p is not None):
        for candidate in expand_local_candidates(declared):
            path = os.path.abspath(candidate)
            if (path in protected_paths
                    or os.path.realpath(path) in protected_targets
                    or not os.path.isfile(path)):
                continue
            os.unlink(path)
