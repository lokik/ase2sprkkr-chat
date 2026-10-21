"""Run file synchronization strategies.

This endpoint-owned handler contains no per-run state. All paths, cache
baselines, wildcard memberships and availability live in FileManifest.
The low-level Transport owns physical filesystem operations.
"""
from __future__ import annotations

import os
from dataclasses import replace

from ..common.file_versions import file_digest
from .base import file_path
from .run_model import CacheEntry, FileEntry, FileManifest, FileMapping, FileSyncPolicy
from .transport import expand_local_files, has_file_magic



def _local_path(spec, value):
    if value is None:
        return None
    try:
        return file_path(value, spec.directory, flush=True, required=True)
    except ValueError as exc:
        raise ValueError("Remote execution requires named files") from exc


def _unique_declared_paths(spec, values, *, excluded=()):
    """Normalize and deduplicate declarations, preserving declaration order."""
    seen = set(excluded)
    for value in values:
        local = _local_path(spec, value)
        if local is not None and local not in seen:
            seen.add(local)
            yield local, has_file_magic(local)


class RemoteFileHandler:
    """Reusable remote file strategy, owned by an endpoint, not a run."""

    def __init__(self, transport):
        self.transport = transport

    async def _stage_file(self, context, spec, value, used, *,
                          uploads=None, required=True):
        source = _local_path(spec, value)
        if source is None:
            return None
        target = await context.map_path(source)
        previous = used.get(target)
        if previous is not None and previous != source:
            raise ValueError(f"Remote input target collision: {target}")
        if previous is not None:
            return target
        policy = context.input_files
        if policy == FileSyncPolicy.NEVER:
            used[target] = source
            return target
        local_digest = file_digest(source, missing_ok=True)
        remote_digest = await self.transport.file_digest(target)
        if required and local_digest is None:
            if policy == FileSyncPolicy.ALWAYS:
                raise FileNotFoundError(f"Local input file does not exist: {source}")
            if remote_digest is None:
                raise FileNotFoundError(
                    f"Input file exists neither locally nor remotely: {source}"
                )
        if policy.should_transfer(local_digest, remote_digest):
            if uploads is None:
                await self.transport.put_file(source, target)
            else:
                uploads.append(source)
        used[target] = source
        return target

    async def _stage_input_pattern(self, context, spec, value, used, *,
                                   required=True):
        source = _local_path(spec, value)
        matches = expand_local_files(source, context.local_root)
        if not matches and required:
            raise FileNotFoundError(
                f"Local input pattern did not match any files: {source}"
            )
        uploads = []
        for match in matches:
            await self._stage_file(
                context, spec, match, used, uploads=uploads, required=required,
            )
        if uploads:
            await self.transport.put_files(
                uploads, context.local_root, await context._resolved_remote_root(),
            )

    async def _prepare_cache_files(self, context, spec, used):
        """Return per-call cache plan; do not retain any of it on the handler."""
        targets, baselines = {}, {}
        root = await context._resolved_remote_root()
        for local, is_pattern in _unique_declared_paths(spec, spec.cache_files):
            if is_pattern:
                remote_pattern = await context.map_path(local)
                await self._stage_input_pattern(
                    context, spec, local, used, required=False,
                )
                if context.output_files != FileSyncPolicy.NEVER:
                    for remote in await self.transport.expand_files(remote_pattern, root):
                        baselines[remote] = await self.transport.file_signature(remote)
                continue
            remote = await self._stage_file(
                context, spec, local, used, required=False,
            )
            before = (None if context.output_files == FileSyncPolicy.NEVER
                      else await self.transport.file_signature(remote))
            targets[local] = (remote, before)
        return targets, baselines

    async def prepare(self, context, spec):
        """Stage files and return their mapping and baseline without submission data."""
        used = {}
        for value in spec.input_files:
            source = _local_path(spec, value)
            if source is not None and has_file_magic(source):
                await self._stage_input_pattern(context, spec, value, used)
            else:
                await self._stage_file(context, spec, value, used)
        await self._stage_file(context, spec, spec.input_file, used)
        if spec.stdin is not None:
            await self._stage_file(context, spec, spec.stdin, used)
        # Cache staging shares the collision map, not input file identity.
        input_sources = tuple(dict.fromkeys(used.values()))
        cache_targets, cache_before = (
            await self._prepare_cache_files(context, spec, used)
        )
        manifest = FileManifest.from_spec(spec)

        async def mapped(entry):
            return replace(entry, remote_path=await context.map_path(entry.local_path))

        caches = []
        for entry in manifest.caches:
            mapped_entry = await mapped(entry.file)
            baseline = cache_targets.get(entry.file.local_path)
            caches.append(CacheEntry(mapped_entry, None if baseline is None else baseline[1]))
        for remote, before in cache_before.items():
            caches.append(CacheEntry(FileEntry(remote_path=remote), before))

        # Wildcard inputs are concretized during staging, not at completion.
        inputs = []
        for local in dict.fromkeys((*manifest.input_files, *input_sources)):
            if has_file_magic(local):
                continue
            inputs.append(FileEntry(local_path=local, remote_path=await context.map_path(local)))

        return replace(
            manifest,
            inputs=tuple(inputs),
            outputs=tuple([await mapped(e) for e in manifest.outputs]),
            caches=tuple(caches),
            stdout=None if manifest.stdout is None else await mapped(manifest.stdout),
            stderr=None if manifest.stderr is None else await mapped(manifest.stderr),
            mapping=FileMapping(
                local_root=os.fspath(context.local_root),
                remote_root=await context._resolved_remote_root(),
                input_sync=context.input_files.value,
                output_sync=context.output_files.value,
            ),
        )

    async def _sync_file(self, context, entry, *, is_cache, before=None, required=False):
        """One transfer mechanism; cache and output policies differ only in decision."""
        remote, local = entry.remote_path, entry.local_path
        if is_cache:
            after = await self.transport.file_signature(remote)
            if (after is None or after == before
                    or (context.output_files == FileSyncPolicy.MISSING
                        and file_digest(local, missing_ok=True) is not None)):
                return "skipped"
        else:
            digest = await self.transport.file_digest(remote)
            if digest is None:
                if required:
                    raise FileNotFoundError(f"Declared remote output was not created: {remote}")
                return "skipped"
            if not context.output_files.should_transfer(
                digest, file_digest(local, missing_ok=True),
            ):
                return "available"
        await self.transport.get_file(remote, local)
        return "available"


    @staticmethod
    def _validate_download_targets(manifest):
        """Preflight destination collisions before transferring any artifact.

        Output/cache declarations that only overlap after wildcard expansion
        must not be allowed to overwrite each other. Exact duplicates of one
        source are harmless; different output sources for one path are not.
        """
        destinations = {}
        entries = (
            *(("input", entry) for entry in manifest.inputs),
            *(("output", entry) for entry in
              (manifest.stdout, manifest.stderr, *manifest.outputs)),
            *(("cache", cache.file) for cache in manifest.caches),
        )
        for kind, entry in entries:
            if entry is None or entry.local_path is None:
                continue
            target = os.path.realpath(os.path.abspath(entry.local_path))
            previous = destinations.get(target)
            if previous is None:
                destinations[target] = (kind, entry.remote_path)
                continue
            old_kind, old_source = previous
            if (kind == "output") != (old_kind == "output"):
                raise ValueError(
                    f"Conflicting {old_kind}/{kind} file destination: {target}"
                )
            if (
                kind == old_kind
                and kind in ("output", "cache")
                and entry.remote_path != old_source
            ):
                raise ValueError(
                    f"Conflicting {kind} sources for destination: {target}"
                )

    async def collect(self, context, manifest, result, *, required=False):
        """Return partial manifest and failure; never silently lose successful files."""
        if manifest is None:
            return None, None
        self._validate_download_targets(manifest)
        primary = {p for p in (result.output_file, result.error_file) if p}
        baselines = {
            c.file.remote_path: c.before for c in manifest.caches
            if c.file.local_path is None
        }
        candidates = (
            *((False, entry, None) for entry in manifest.outputs),
            *((True, cache.file, baselines.get(cache.file.remote_path, cache.before))
              for cache in manifest.caches),
        )
        seen = set()
        first_error = None
        for is_cache, entry, before in candidates:
            if entry.local_path is None or entry.remote_path is None:
                continue
            kind = "cache" if is_cache else "output"
            if has_file_magic(entry.remote_path):
                exc = RuntimeError("Cannot collect unresolved wildcard")
                manifest = manifest.with_file_availability(
                    entry.local_path, "failed", str(exc), kind=kind,
                )
                if first_error is None:
                    first_error = exc
                continue
            identity = (kind, entry.remote_path)
            if identity in seen:
                continue
            seen.add(identity)
            if not is_cache and entry.local_path in primary:
                continue
            try:
                state = ("skipped" if context.output_files == FileSyncPolicy.NEVER
                         else await self._sync_file(
                             context, entry, is_cache=is_cache,
                             before=before, required=required,
                         ))
            except Exception as exc:
                manifest = manifest.with_file_availability(
                    entry.local_path, "failed", str(exc), kind=kind,
                )
                if first_error is None:
                    first_error = exc
                continue
            manifest = manifest.with_file_availability(
                entry.local_path, state, kind=kind,
            )
        return manifest, first_error

    async def restore(self, run, paths, context, *, destinations=None):
        """Restore only persisted concrete files; caller checks digests."""
        destinations = {} if destinations is None else {
            os.path.abspath(os.fspath(p)): os.fspath(target)
            for p, target in destinations.items()
        }
        entries = (
            *run.files.inputs, *run.files.outputs,
            *(cache.file for cache in run.files.caches),
            run.files.stdout, run.files.stderr,
        )
        for value in paths:
            local = os.path.abspath(os.fspath(value))
            # A wildcard request selects only already-materialized members.
            # Never rediscover historical membership on the target.
            if has_file_magic(local):
                selected = tuple(
                    entry for entry in entries if entry is not None
                    and entry.source == local
                )
            else:
                selected = tuple(
                    entry for entry in entries if entry is not None
                    and entry.local_path == local
                )
            if not selected:
                raise ValueError(f"File has no persisted remote source: {local}")
            for entry in selected:
                if entry.remote_path is None or has_file_magic(entry.remote_path):
                    raise ValueError(
                        f"Cannot restore unresolved remote file: {entry.local_path}"
                    )
                target = destinations.get(entry.local_path, entry.local_path)
                if not os.path.isfile(target):
                    await self.transport.get_file(entry.remote_path, target)
