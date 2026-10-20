"""Local artifact discovery shared by process and PSI/J runners.

No transport or execution state lives here. The caller decides whether the
producer's output streams are safe to consider published.
"""

from __future__ import annotations

import os

from .transport import expand_local_candidates, expand_local_files, has_file_magic


def materialize_target_patterns(directory, *, outputs=(), caches=()):
    """Evaluate each target-side wildcard once, preserving partial results.

    Every source pattern has its own concrete match list and optional error;
    later consumers never need to reapply glob matching rules.
    """
    result = {"outputs": [], "caches": []}
    for kind, patterns in (("outputs", outputs), ("caches", caches)):
        for pattern in patterns:
            matches = ()
            error = None
            try:
                matches = tuple(dict.fromkeys(
                    os.path.abspath(os.fspath(match))
                    for match in expand_local_files(pattern, directory)
                    if os.path.isfile(match)
                ))
            except Exception as exc:
                error = str(exc)
            result[kind].append({
                "source": os.fspath(pattern),
                "matches": matches,
                "error": error,
            })
    return result


def discover_local_files(files, *, include_streams=True):
    """Resolve pending local declarations and retain individual wildcard matches."""
    # Local process and PSI/J jobs materialize their wildcard declarations at
    # the first terminal observation. Concrete members replace declarations;
    # nothing will be expanded again from an already finalized manifest.
    from dataclasses import replace

    def resolve(entry, *, is_cache):
        if entry is None or entry.local_path is None:
            return ()
        if entry.availability_state not in (None, "pending"):
            return (entry,)
        path = entry.local_path
        if not has_file_magic(path):
            return (replace(entry, availability_state=(
                "available" if os.path.isfile(path) else "skipped"
            )),)
        return tuple(
            replace(entry, local_path=candidate, availability_state="available",
                    source=path)
            for candidate in expand_local_candidates(path)
            if os.path.isfile(candidate)
        )

    outputs = tuple(x for entry in files.outputs
                    for x in resolve(entry, is_cache=False))
    caches = tuple(replace(cache, file=x)
                   for cache in files.caches
                   for x in resolve(cache.file, is_cache=True))
    stdout = files.stdout
    stderr = files.stderr
    if include_streams:
        stdout = next(iter(resolve(files.stdout, is_cache=False)), None)
        stderr = next(iter(resolve(files.stderr, is_cache=False)), None)
    return replace(files, outputs=outputs, caches=caches,
                   stdout=stdout, stderr=stderr)


def validate_output_collisions(streams, *, protected=()):
    """Reject explicit output paths overlapping declared inputs or restart caches.

    Broad output patterns retain their existing behavior: matching protected
    files are skipped by clear_declared_outputs rather than removed.
    """
    protected_paths = tuple(
        os.path.abspath(os.fspath(p)) for p in protected if p is not None
    )
    for stream in streams:
        if stream is None:
            continue
        target = os.path.abspath(os.fspath(stream))
        if has_file_magic(target):
            continue
        for declaration in protected_paths:
            for candidate in expand_local_candidates(declaration):
                same_path = os.path.realpath(target) == os.path.realpath(candidate)
                same_inode = (
                    os.path.exists(target) and os.path.exists(candidate)
                    and os.path.samefile(target, candidate)
                )
                if same_path or same_inode:
                    raise ValueError(
                        f"Output {target} overlaps protected input/cache {declaration}"
                    )


def clear_declared_outputs(paths, *, protected=()):
    """Remove old declared results without destroying inputs or restart caches.

    Called on the execution host before starting a new calculation, never on
    reattach. A wildcard removes only its concrete files.
    """
    protected_paths = (os.path.abspath(os.fspath(p)) for p in protected if p is not None)
    protected = {
        os.path.abspath(candidate)
        for declaration in protected_paths
        for candidate in expand_local_candidates(declaration)
    }
    for declared in dict.fromkeys(os.fspath(p) for p in paths if p is not None):
        for candidate in expand_local_candidates(declared):
            path = os.path.abspath(candidate)
            if path in protected or not os.path.isfile(path):
                continue
            os.unlink(path)
