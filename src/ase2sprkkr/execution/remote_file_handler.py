"""Run file synchronization strategies.

This endpoint-owned handler contains no per-run state. All paths, cache
baselines, wildcard memberships and availability live in FileManifest.
The low-level Transport owns physical filesystem operations.
"""
from __future__ import annotations

import fnmatch
import os
import posixpath
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
        targets, patterns, baselines = {}, [], {}
        root = await context._resolved_remote_root()
        for local, is_pattern in _unique_declared_paths(spec, spec.cache_files):
            if is_pattern:
                remote_pattern = await context.map_path(local)
                patterns.append(remote_pattern)
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
        return targets, tuple(patterns), baselines

    async def _prepare_output_files(self, context, spec, result):
        primary = {
            p for p in (result.output_file, result.error_file) if p is not None
        }
        targets, patterns = {}, []
        for local, is_pattern in _unique_declared_paths(
            spec, spec.output_files, excluded=primary,
        ):
            remote = await context.map_path(local)
            if is_pattern:
                patterns.append(remote)
            else:
                targets[local] = remote
        # Target-side launch owns cleanup, never staging or attach.
        return targets, tuple(patterns)

    @staticmethod
    def _remote_argv(spec, argv, staged):
        result = []
        for arg in argv:
            value = os.fspath(arg)
            replacement = staged.get(file_path(value, spec.directory))
            result.append(replacement or value)
        return result

    async def prepare(self, context, spec, result, run_id, remote_directory):
        """Stage input/cache files and return (request, immutable manifest)."""
        used = {}
        for value in spec.input_files:
            source = _local_path(spec, value)
            if source is not None and has_file_magic(source):
                await self._stage_input_pattern(context, spec, value, used)
            else:
                await self._stage_file(context, spec, value, used)
        input_target = await self._stage_file(context, spec, spec.input_file, used)
        stdin_target = await self._stage_file(context, spec, spec.stdin, used)

        cache_targets, cache_patterns, cache_before = (
            await self._prepare_cache_files(context, spec, used)
        )
        staged = {source: target for target, source in used.items()}
        output_targets, output_patterns = (
            await self._prepare_output_files(context, spec, result)
        )
        output_target = (await context.map_path(result.output_file)
                         if result.output_file else None)
        error_target = (await context.map_path(result.error_file)
                        if result.error_file else None)
        request = {
            "id": run_id,
            "executor": context.backend,
            "executor_options": context.backend_options,
            "directory": remote_directory,
            **spec.command_request(
                lambda args: self._remote_argv(spec, args, staged)
            ),
            "input_file": input_target,
            "stdin": stdin_target,
            "output_file": output_target,
            "error_file": error_target,
            "client_sync": {
                "output_file": output_target if spec.output_file is not None else None,
                "error_file": error_target if spec.error_file is not None else None,
                "output_files": list(output_targets.values()),
                "cache_files": [
                    {"path": remote, "before": before}
                    for remote, before in cache_targets.values()
                ],
                "output_patterns": list(output_patterns),
                "cache_patterns": list(cache_patterns),
                "cache_before": [
                    {"path": remote, "signature": before}
                    for remote, before in cache_before.items()
                ],
            },
            "input_files": list(used.keys()),
        }
        return request, await self.submitted_manifest(context, spec, request)

    async def submitted_manifest(self, context, spec, request):
        """Persist mapping and all cache baselines from the prepared request."""
        local = FileManifest.from_spec(spec)
        async def mapped(entry):
            return replace(entry, remote_path=await context.map_path(entry.local_path))
        sync = request.get("client_sync") or {}
        cache_targets = {
            item["path"]: item.get("before")
            for item in sync.get("cache_files", ())
        }
        caches = []
        for item in local.caches:
            entry = await mapped(item.file)
            caches.append(CacheEntry(entry, cache_targets.get(entry.remote_path)))
        for item in sync.get("cache_before", ()):
            caches.append(CacheEntry(
                FileEntry(remote_path=item["path"]), item.get("signature"),
            ))
        return replace(
            local,
            inputs=tuple([await mapped(e) for e in local.inputs]),
            outputs=tuple([await mapped(e) for e in local.outputs]),
            caches=tuple(caches),
            stdout=None if local.stdout is None else replace(
                local.stdout, remote_path=sync.get("output_file"),
            ),
            stderr=None if local.stderr is None else replace(
                local.stderr, remote_path=sync.get("error_file"),
            ),
            mapping=FileMapping(
                local_root=os.fspath(context.local_root),
                remote_root=await context._resolved_remote_root(),
                input_sync=context.input_files.value,
                output_sync=context.output_files.value,
            ),
        )

    @staticmethod
    def freeze_from_outcome(manifest, matched):
        """Freeze the target's terminal wildcard set in the manifest itself."""
        if manifest is None or manifest.wildcards_frozen:
            return manifest
        patterns = (
            tuple(e for e in manifest.outputs
                  if e.source_pattern is None and e.remote_path
                  and has_file_magic(e.remote_path)),
            tuple(c.file for c in manifest.caches
                  if c.file.source_pattern is None and c.file.local_path
                  and c.file.remote_path and has_file_magic(c.file.remote_path)),
        )
        if matched is None:
            if any(patterns):
                raise RuntimeError("Terminal backend omitted wildcard membership")
            return replace(manifest, wildcards_frozen=True)
        if manifest.mapping is None:
            raise ValueError("Remote wildcard resolution needs persisted mapping")
        outputs, caches = [], []
        for kind, declarations, target in (
            ("outputs", patterns[0], outputs),
            ("caches", patterns[1], caches),
        ):
            for entry in declarations:
                for remote in matched[kind]:
                    if not fnmatch.fnmatchcase(remote, entry.remote_path):
                        continue
                    relative = posixpath.relpath(
                        remote, manifest.mapping.remote_root,
                    )
                    if (relative == ".." or relative.startswith("../")
                            or posixpath.isabs(relative)):
                        raise ValueError(
                            f"Remote wildcard match escaped root: {remote}"
                        )
                    target.append((
                        os.path.join(
                            manifest.mapping.local_root, *relative.split("/"),
                        ),
                        remote, entry.local_path,
                    ))
        return manifest.freeze_wildcards(outputs=outputs, caches=caches)

    async def _sync_one_output(self, context, local, remote, required):
        digest = await self.transport.file_digest(remote)
        if digest is None:
            if required:
                raise FileNotFoundError(
                    f"Declared remote output file was not created: {remote}"
                )
            return False
        local_digest = file_digest(local, missing_ok=True)
        if context.output_files.should_transfer(digest, local_digest):
            await self.transport.get_file(remote, local)
        return True

    async def _sync_one_cache(self, context, local, remote, before):
        after = await self.transport.file_signature(remote)
        if (after is None or after == before
                or (context.output_files == FileSyncPolicy.MISSING
                    and file_digest(local, missing_ok=True) is not None)):
            return "skipped"
        await self.transport.get_file(remote, local)
        return "available"

    async def collect(self, context, manifest, result, *, required=False):
        """Return (updated manifest, error); never lose partial file results.

        Successful files remain recorded if a subsequent transfer fails.
        The runner interprets any returned error as a DOWNLOAD failure while
        keeping the original exception and the updated file availability.
        """
        if manifest is None:
            return None, None

        def marked(files, entry, state, message=None, *, kind="output"):
            return files.with_file_availability(
                entry.local_path, state, message, kind=kind,
            )

        primary = {p for p in (result.output_file, result.error_file) if p}
        if context.output_files == FileSyncPolicy.NEVER:
            for entry in manifest.artifacts:
                if entry.availability_state == "pending":
                    manifest = marked(
                        manifest, entry,
                        "available" if entry.local_path in primary else "skipped",
                    )
            return manifest, None

        seen_remote = set()
        # Collect declared outputs, including only the children materialized
        # at terminal completion; no new wildcard expansion is performed.
        for entry in tuple(manifest.outputs):
            if entry.source_pattern is not None:
                continue
            remote = entry.remote_path
            if remote is None:
                continue
            if has_file_magic(remote):
                if not manifest.wildcards_frozen:
                    error = RuntimeError("Cannot collect an unresolved wildcard")
                    return marked(manifest, entry, "failed", str(error)), error
                children = tuple(
                    child for child in manifest.outputs
                    if child.source_pattern == entry.local_path
                )
                if not children:
                    error = (FileNotFoundError(
                        f"Declared remote output pattern matched no files: {remote}"
                    ) if required else None)
                    manifest = marked(
                        manifest, entry,
                        "failed" if error else "skipped",
                        None if error is None else str(error),
                    )
                    if error:
                        return manifest, error
                    continue
            else:
                children = (entry,)

            for child in children:
                if child.remote_path in seen_remote or child.local_path in primary:
                    continue
                seen_remote.add(child.remote_path)
                try:
                    available = await self._sync_one_output(
                        context, child.local_path, child.remote_path, required,
                    )
                except Exception as exc:
                    return marked(manifest, child, "failed", str(exc)), exc
                manifest = marked(
                    manifest, child, "available" if available else "skipped",
                )
            if children != (entry,):
                manifest = marked(manifest, entry, "available")

        # Remote-only caches are the stored *pre-run* baselines for wildcard
        # children, never a second download target.
        baselines = {
            c.file.remote_path: c.before
            for c in manifest.caches
            if c.file.local_path is None
        }
        seen_remote = set()
        for cache in tuple(manifest.caches):
            entry = cache.file
            if entry.local_path is None or entry.source_pattern is not None:
                continue
            remote = entry.remote_path
            if remote is None:
                continue
            if has_file_magic(remote):
                if not manifest.wildcards_frozen:
                    error = RuntimeError("Cannot collect unresolved cache wildcard")
                    return marked(manifest, entry, "failed", str(error), kind="cache"), error
                children = tuple(
                    c.file for c in manifest.caches
                    if c.file.source_pattern == entry.local_path
                )
                if not children:
                    manifest = marked(manifest, entry, "skipped", kind="cache")
                    continue
            else:
                children = (entry,)
            for child in children:
                if child.remote_path in seen_remote:
                    continue
                seen_remote.add(child.remote_path)
                before = baselines.get(child.remote_path, cache.before)
                try:
                    state = await self._sync_one_cache(
                        context, child.local_path, child.remote_path, before,
                    )
                except Exception as exc:
                    return marked(manifest, child, "failed", str(exc), kind="cache"), exc
                manifest = marked(manifest, child, state, kind="cache")
            if children != (entry,):
                manifest = marked(manifest, entry, "available", kind="cache")
        return manifest, None

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
            entry = next(
                (e for e in entries if e is not None
                 and e.local_path == local), None,
            )
            if entry is None or entry.remote_path is None:
                raise ValueError(f"File has no persisted remote source: {local}")
            if has_file_magic(entry.remote_path):
                if not run.files.wildcards_frozen:
                    raise ValueError("Cannot restore unresolved wildcard declaration")
                continue
            target = destinations.get(local, local)
            if not os.path.isfile(target):
                await self.transport.get_file(entry.remote_path, target)
