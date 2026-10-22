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
from .run_model import FileEntry, FileManifest, FileMapping, FileSyncPolicy
from .transport import expand_local_files, has_file_magic



def _local_path(spec, value):
    if value is None:
        return None
    try:
        return file_path(value, spec.directory, flush=True, required=True)
    except ValueError as exc:
        raise ValueError("Remote execution requires named files") from exc


class RemoteFileHandler:
    """Reusable remote file strategy, owned by an endpoint, not a run."""

    def __init__(self, transport):
        self.transport = transport

    async def _stage_file(self, mapping, spec, value, used, *,
                          uploads=None, required=True):
        source = _local_path(spec, value)
        if source is None:
            return None
        target = mapping.map_path(source)
        previous = used.get(target)
        if previous is not None and previous != source:
            raise ValueError(f"Remote input target collision: {target}")
        if previous is not None:
            return target
        policy = FileSyncPolicy(mapping.input_sync)
        if policy == FileSyncPolicy.NEVER:
            used[target] = source
            return target
        local_exists = os.path.isfile(source)
        if policy == FileSyncPolicy.ALWAYS:
            if required and not local_exists:
                raise FileNotFoundError(f"Local input file does not exist: {source}")
            should_upload = local_exists
        elif not local_exists:
            # A missing local cache is valid; a mandatory input must exist
            # on at least one host. Neither case requires a content digest.
            if required and await self.transport.file_signature(target) is None:
                raise FileNotFoundError(
                    f"Input file exists neither locally nor remotely: {source}"
                )
            should_upload = False
        elif policy == FileSyncPolicy.MISSING:
            should_upload = await self.transport.file_signature(target) is None
        else:
            remote_digest = await self.transport.file_digest(target)
            # No need to hash the local file when the target is absent.
            should_upload = (
                remote_digest is None or
                policy.should_transfer(
                    file_digest(source, missing_ok=True), remote_digest,
                )
            )
        if should_upload:
            if uploads is None:
                await self.transport.put_file(source, target)
            else:
                uploads.append(source)
        used[target] = source
        return target

    async def _stage_input_pattern(self, mapping, spec, value, used, *,
                                   required=True):
        source = _local_path(spec, value)
        matches = expand_local_files(source, mapping.local_root)
        if not matches and required:
            raise FileNotFoundError(
                f"Local input pattern did not match any files: {source}"
            )
        uploads = []
        for match in matches:
            await self._stage_file(
                mapping, spec, match, used, uploads=uploads, required=required,
            )
        if uploads:
            await self.transport.put_files(
                uploads, mapping.local_root, mapping.remote_root,
            )
        return matches

    async def _prepare_cache_files(self, mapping, spec, used, cache_entries):
        """Stage the manifest's cache declarations and capture target baselines."""
        targets, baselines = {}, {}
        root = mapping.remote_root
        # FileManifest already normalizes and validates declarations.
        for entry in cache_entries:
            local = entry.local_path
            if local is None:
                continue
            if has_file_magic(local):
                remote_pattern = mapping.map_path(local)
                await self._stage_input_pattern(
                    mapping, spec, local, used, required=False,
                )
                if FileSyncPolicy(mapping.output_sync) != FileSyncPolicy.NEVER:
                    for remote in await self.transport.expand_files(remote_pattern, root):
                        baselines[remote] = await self.transport.file_signature(remote)
                continue
            remote = await self._stage_file(
                mapping, spec, local, used, required=False,
            )
            before = (None if FileSyncPolicy(mapping.output_sync) == FileSyncPolicy.NEVER
                      else await self.transport.file_signature(remote))
            targets[local] = before
        return targets, baselines

    async def prepare(self, mapping, spec, manifest=None):
        """Stage the authoritative manifest; never re-declare files from spec."""
        manifest = FileManifest.from_spec(spec) if manifest is None else manifest
        # Validate every reserved target before touching a remote file.
        manifest = manifest.with_mapping(mapping)

        # Named source handles may hold buffered input; flush without
        # rebuilding the authoritative file declarations from the spec.
        for value in (
            spec.input_file, spec.stdin, *spec.input_files, *spec.cache_files,
        ):
            _local_path(spec, value)

        used = {}
        # Track optionality of materialized wildcard members as well as
        # exact declarations. Required aliases always take precedence.
        input_optional = {
            entry.local_path: entry.optional for entry in manifest.inputs
            if entry.local_path is not None
        }
        primary = (None if manifest.primary_input is None
                   else manifest.primary_input.local_path)
        # Primary input and stdin are mandatory references. Stage them first,
        # so an optional auxiliary alias cannot weaken their requirements.
        if primary is not None:
            await self._stage_file(mapping, spec, primary, used)
        if manifest.stdin is not None and manifest.stdin.local_path is not None:
            await self._stage_file(mapping, spec, manifest.stdin.local_path, used)
        for entry in manifest.inputs:
            source = entry.local_path
            if source is None or source == primary:
                continue
            if has_file_magic(source):
                matches = await self._stage_input_pattern(
                    mapping, spec, source, used, required=not entry.optional,
                )
                for match in matches:
                    input_optional[match] = (
                        input_optional.get(match, True) and entry.optional
                    )
            else:
                await self._stage_file(
                    mapping, spec, source, used, required=not entry.optional,
                )

        # Primary input and stdin are mandatory even when matched by an
        # optional wildcard declaration.
        for reference in (manifest.primary_input, manifest.stdin):
            if reference is not None and reference.local_path is not None:
                input_optional[reference.local_path] = False

        # Cache staging shares the collision map, not input file identity.
        input_sources = tuple(dict.fromkeys(used.values()))
        cache_targets, cache_before = await self._prepare_cache_files(
            mapping, spec, used, manifest.caches,
        )

        caches = [
            replace(entry, pre_run_signature=cache_targets.get(entry.local_path))
            for entry in manifest.caches
        ]
        for remote, before in cache_before.items():
            caches.append(FileEntry(remote_path=remote, pre_run_signature=before))

        # Wildcard inputs are concretized during staging, not at completion.
        inputs = []
        declared = {entry.local_path: entry for entry in manifest.inputs}
        for local in dict.fromkeys((*declared, *input_sources)):
            if local is None or has_file_magic(local):
                continue
            if (manifest.stdin is not None and local == manifest.stdin.local_path
                    and local not in declared):
                continue
            inputs.append(FileEntry(
                local_path=local, remote_path=mapping.map_path(local),
                optional=input_optional.get(local, False),
            ))

        return replace(
            manifest,
            inputs=tuple(inputs),
            caches=tuple(caches),
        )

    async def _sync_file(self, mapping, entry, *, is_cache, before=None, required=False):
        """One transfer mechanism; cache and output policies differ only in decision."""
        remote, local = entry.remote_path, entry.local_path
        policy = FileSyncPolicy(mapping.output_sync)
        local_exists = os.path.isfile(local)
        if is_cache:
            # Cache membership was frozen at termination; compare to the
            # target's pre-run signature, not the client's current contents.
            after = await self.transport.file_signature(remote)
            if after is None or after == before:
                return "skipped"
        elif policy == FileSyncPolicy.CHANGED and local_exists:
            # Hash only when both sides have a file and contents matter.
            remote_digest = await self.transport.file_digest(remote)
            if remote_digest is None:
                if required:
                    raise FileNotFoundError(f"Declared remote output was not created: {remote}")
                return "skipped"
            if remote_digest == file_digest(local, missing_ok=True):
                return "available"
        elif await self.transport.file_signature(remote) is None:
            if required:
                raise FileNotFoundError(f"Declared remote output was not created: {remote}")
            return "skipped"

        # The MISSING policy applies to both kinds, but an unchanged cache is
        # "skipped", while a locally present output is already "available".
        if policy == FileSyncPolicy.MISSING and local_exists:
            return "skipped" if is_cache else "available"
        await self.transport.get_file(remote, local)
        return "available"


    async def collect(self, mapping, manifest, result, *, required=False):
        """Return partial manifest and failure; never silently lose successful files."""
        if manifest is None:
            return None, None
        primary = {p for p in (result.output_file, result.error_file) if p}
        baselines = {
            c.remote_path: c.pre_run_signature for c in manifest.caches
            if c.local_path is None
        }
        candidates = (
            *((False, entry, None) for entry in manifest.outputs),
            *((True, cache, baselines.get(cache.remote_path, cache.pre_run_signature))
              for cache in manifest.caches),
        )
        seen = set()
        # Overlapping output globs can materialize the same physical file.
        # The first declaration must not make a second mandatory declaration
        # optional: resolve the effective requirement across all aliases.
        mandatory_outputs = {
            (entry.local_path, entry.remote_path)
            for entry in manifest.outputs if not entry.optional
        }
        first_error = None
        for is_cache, entry, before in candidates:
            if entry.local_path is None or entry.remote_path is None:
                continue
            # Terminal receipts describe a frozen file set. An already
            # delivered local file (or an intentionally skipped declaration)
            # must not be re-evaluated on attach. Retry only incomplete
            # deliveries, including available files subsequently removed.
            if entry.availability_state == "skipped":
                continue
            if entry.availability_state == "available" and os.path.isfile(entry.local_path):
                continue
            kind = "cache" if is_cache else "output"
            if has_file_magic(entry.remote_path):
                exc = RuntimeError("Cannot collect unresolved wildcard")
                manifest = manifest.with_file_availability(
                    entry.local_path, "failed", str(exc),
                    kind=kind, remote_path=entry.remote_path,
                )
                if first_error is None:
                    first_error = exc
                continue
            # One source may legitimately be requested at several local paths.
            # Suppress only repeated declarations of the same transfer.
            identity = (kind, entry.remote_path, entry.local_path)
            if identity in seen:
                continue
            seen.add(identity)
            if not is_cache and entry.local_path in primary:
                continue
            try:
                state = ("skipped" if FileSyncPolicy(mapping.output_sync) == FileSyncPolicy.NEVER
                         else await self._sync_file(
                             mapping, entry, is_cache=is_cache,
                             before=before,
                             required=required and (
                                 not entry.optional if is_cache else
                                 (entry.local_path, entry.remote_path)
                                 in mandatory_outputs
                             ),
                         ))
            except Exception as exc:
                manifest = manifest.with_file_availability(
                    entry.local_path, "failed", str(exc),
                    kind=kind, remote_path=entry.remote_path,
                )
                if first_error is None:
                    first_error = exc
                continue
            manifest = manifest.with_file_availability(
                entry.local_path, state, kind=kind,
                remote_path=entry.remote_path,
            )
        return manifest, first_error

    async def verify_stream_availability(self, manifest, spec):
        """Verify each unresolved FOLLOW identity and record its own result."""
        if manifest is None:
            return None, None
        first_error = None
        checked = set()
        frozen = {
            (entry.local_path, entry.remote_path)
            for entry in (manifest.stdout, manifest.stderr)
            if entry is not None
            and entry.availability_state in ("failed", "skipped")
        }
        for kind, entry in (("stdout", manifest.stdout), ("stderr", manifest.stderr)):
            if entry is None or entry.local_path is None:
                continue
            identity = (entry.local_path, entry.remote_path)
            if identity in checked or identity in frozen:
                continue
            checked.add(identity)
            try:
                await self._verify_stream(entry, kind, spec)
            except Exception as exc:
                manifest = manifest.with_file_availability(
                    entry.local_path, "failed", str(exc),
                    remote_path=entry.remote_path,
                )
                if first_error is None:
                    first_error = exc
            else:
                manifest = manifest.with_file_availability(
                    entry.local_path, "available", remote_path=entry.remote_path,
                )
        return manifest, first_error

    async def _verify_stream(self, entry, kind, spec):
        """Verify one streamed file against its remote authoritative contents."""
        from ..common.file_versions import file_digest

        if entry.remote_path is None:
            raise ValueError(f"No remote path for {kind}")
        signature = await self.transport.file_signature(entry.remote_path)
        if signature is None:
            raise FileNotFoundError(
                f"Remote {kind} output is missing: {entry.remote_path}"
            )
        for stream in (spec.output_file, spec.error_file):
            if stream is not None and hasattr(stream, "flush") and not stream.closed:
                stream.flush()
        stat = os.stat(entry.local_path)
        if stat.st_size != signature[0]:
            raise RuntimeError(
                f"Streamed local {kind} size does not match the authoritative "
                f"remote output: {entry.local_path}"
            )
        digest = await self.transport.file_digest(entry.remote_path)
        if digest is None or file_digest(entry.local_path) != digest:
            raise RuntimeError(
                f"Streamed local {kind} contents differ from remote output: "
                f"{entry.local_path}"
            )
        if await self.transport.file_signature(entry.remote_path) != signature:
            raise RuntimeError(f"Remote {kind} output changed during verification")
        os.utime(entry.local_path, ns=(stat.st_atime_ns, signature[1]))

    async def restore(self, run, paths, *, destinations=None):
        """Restore persisted concrete files, preflighting the entire request.

        The manifest is the sole source of historical file membership and
        target paths. Validation completes before any transfer takes place.
        """
        destinations = {} if destinations is None else {
            os.path.abspath(os.fspath(p)): os.fspath(target)
            for p, target in destinations.items()
        }
        manifest = run.files
        mapping = manifest.mapping
        if mapping is None:
            raise ValueError("Historical remote files require a persisted mapping")
        entries = (
            *manifest.inputs, *manifest.outputs, *manifest.caches,
            manifest.stdout, manifest.stderr,
        )
        planned = {}
        for value in paths:
            local = os.path.abspath(os.fspath(value))
            # Validate the requested canonical location before selecting a
            # frozen wildcard member. This also rejects symlink escapes when
            # no historical member is present.
            mapping.map_path(local)
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
                if (entry.local_path is None or entry.remote_path is None
                        or has_file_magic(entry.local_path)
                        or has_file_magic(entry.remote_path)):
                    raise ValueError(
                        f"Cannot restore unresolved remote file: {entry.local_path}"
                    )
                # Validate even if an alternate staging destination was
                # requested. A symlink introduced after submission must not
                # allow the historical canonical path to escape the root.
                expected = mapping.map_path(entry.local_path)
                if entry.remote_path != expected:
                    raise ValueError(
                        f"Persisted remote source differs from mapped path: {entry.local_path}"
                    )
                target = os.path.abspath(os.fspath(
                    destinations.get(os.path.abspath(entry.local_path), entry.local_path)
                ))
                key = os.path.realpath(target)
                previous = planned.get(key)
                if previous is not None and previous[0] != entry.remote_path:
                    raise ValueError(f"Conflicting restore sources for destination: {target}")
                planned[key] = (entry.remote_path, target)

        for remote, target in planned.values():
            if not os.path.isfile(target):
                await self.transport.get_file(remote, target)
