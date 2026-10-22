"""Run file synchronization strategies.

This endpoint-owned handler contains no per-run state. All paths, cache
baselines, wildcard memberships and availability live in FileManifest.
The low-level Transport owns physical filesystem operations.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import replace

from ..common.file_fingerprints import file_fingerprint
from ..common.file_versions import FileVersion, file_digest
from .base import file_path
from .run_model import FileEntry, FileManifest, FileSyncPolicy
from .transport import FILE_DELIVERY_ERRORS, expand_local_files, has_file_magic




class RemoteFileHandler:
    """Reusable remote file strategy, owned by an endpoint, not a run."""

    def __init__(self, transport):
        self.transport = transport

    async def _stage_file(self, mapping, source, used, *,
                          uploads=None, required=True):
        # The manifest, not spec, defines which absolute source is staged.
        if not os.path.isabs(source):
            raise ValueError("Manifest input paths must be absolute")
        target = mapping.map_path(source)
        previous = used.get(target)
        if previous is not None and previous != source:
            raise ValueError(f"Remote input target collision: {target}")
        policy = FileSyncPolicy(mapping.input_sync)
        if policy == FileSyncPolicy.NEVER:
            # NEVER disables uploading, not mandatory input validation.
            # A previous optional alias must not suppress this check.
            if required and await self.transport.file_signature(target) is None:
                raise FileNotFoundError(
                    f"Required remote input file does not exist: {target}"
                )
            used[target] = source
            return target
        if previous is not None:
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
                    await asyncio.to_thread(file_digest, source, missing_ok=True),
                    remote_digest,
                )
            )
        if should_upload:
            if uploads is None:
                await self.transport.put_file(source, target)
            else:
                uploads.append(source)
        used[target] = source
        return target

    async def _stage_input_pattern(self, mapping, source, used, *,
                                   required=True):
        if not os.path.isabs(source):
            raise ValueError("Manifest input patterns must be absolute")
        if FileSyncPolicy(mapping.input_sync) == FileSyncPolicy.NEVER:
            # Remote inputs may exist even when no local wildcard member does.
            remote_pattern = mapping.map_path(source)
            remotes = await self.transport.expand_files(
                remote_pattern, mapping.remote_root,
            )
            matches = tuple(
                os.fspath(mapping.unmap_path(remote)) for remote in remotes
            )
            if not matches and required:
                raise FileNotFoundError(
                    f"Required remote input pattern did not match any files: {remote_pattern}"
                )
        else:
            matches = expand_local_files(source, mapping.local_root)
            if not matches and required:
                raise FileNotFoundError(
                    f"Local input pattern did not match any files: {source}"
                )
        uploads = []
        for match in matches:
            await self._stage_file(
                mapping, match, used, uploads=uploads, required=required,
            )
        if uploads:
            await self.transport.put_files(
                uploads, mapping.local_root, mapping.remote_root,
            )
        return matches

    async def _prepare_caches(self, mapping, used, cache_entries):
        """Stage cache inputs and retain pre-run signatures for exact and glob caches."""
        declared, expanded = [], []
        policy = FileSyncPolicy(mapping.output_sync)
        for entry in cache_entries:
            local = entry.local_path
            if local is None:
                declared.append(entry)
                continue
            if has_file_magic(local):
                remote_pattern = mapping.map_path(local)
                await self._stage_input_pattern(
                    mapping, local, used, required=False,
                )
                declared.append(entry)
                if policy != FileSyncPolicy.NEVER:
                    for remote in await self.transport.expand_files(
                        remote_pattern, mapping.remote_root,
                    ):
                        expanded.append(FileEntry(
                            remote_path=remote,
                            source=local,
                            pre_run_signature=await self.transport.file_signature(remote),
                        ))
            else:
                remote = await self._stage_file(
                    mapping, local, used, required=False,
                )
                before = (None if policy == FileSyncPolicy.NEVER
                          else await self.transport.file_signature(remote))
                declared.append(replace(entry, pre_run_signature=before))
        # Keep frozen declarations first; discovered wildcard members follow.
        return tuple((*declared, *expanded))

    async def _prepare_inputs(self, mapping, manifest, used):
        """Materialize reserved inputs once; return only concrete input members."""
        optional = {
            entry.local_path: entry.optional
            for entry in manifest.inputs if entry.local_path is not None
        }
        # stdin is the first input, never another declaration. Stage every
        # member, then coalesce wildcard/exact overlaps without changing order.
        for index, entry in enumerate(manifest.inputs):
            source = entry.local_path
            if source is None:
                continue
            required = not entry.optional or (index == 0 and manifest.stdin)
            if has_file_magic(source):
                matches = await self._stage_input_pattern(
                    mapping, source, used, required=required,
                )
                for match in matches:
                    optional[match] = optional.get(match, True) and entry.optional
            else:
                await self._stage_file(mapping, source, used, required=required)
                if index == 0 and manifest.stdin:
                    optional[source] = False

        declared = {entry.local_path: entry for entry in manifest.inputs}
        inputs = []
        for local in dict.fromkeys((*declared, *used.values())):
            if local is None or has_file_magic(local):
                continue
            inputs.append(FileEntry(
                local_path=local, remote_path=mapping.map_path(local),
                optional=optional.get(local, False),
            ))
        return tuple(inputs)

    async def prepare(self, mapping, spec, manifest=None):
        """Stage frozen declarations and record concrete input/cache membership."""
        manifest = FileManifest.from_spec(spec) if manifest is None else manifest
        # Validate every reserved path, including outputs, before first upload.
        manifest = manifest.with_mapping(mapping)

        # Source handles may hold buffered bytes. Flush only handles whose
        # names are members of the frozen manifest, never new spec declarations.
        declared = {
            entry.local_path for entry in (
                *manifest.inputs, *manifest.caches,
            )
            if entry is not None and entry.local_path is not None
        }
        for value in (spec.input_file, spec.stdin, *spec.input_files, *spec.cache_files):
            if value is None or value is False:
                continue
            try:
                path = file_path(value, spec.directory, required=True)
            except ValueError as exc:
                raise ValueError("Remote execution requires named files") from exc
            if path in declared and hasattr(value, "flush"):
                value.flush()

        used = {}
        inputs = await self._prepare_inputs(mapping, manifest, used)
        caches = await self._prepare_caches(mapping, used, manifest.caches)
        return replace(manifest, inputs=inputs, caches=caches)


    async def _sync_file(self, mapping, entry, *, is_cache, before=None, required=False):
        """Apply one output policy; return state and its validated local version."""
        policy = FileSyncPolicy(mapping.output_sync)
        if policy == FileSyncPolicy.NEVER:
            return "skipped", None

        remote, local = entry.remote_path, entry.local_path
        local_exists = os.path.isfile(local)

        # MISSING never overwrites or blesses an existing local copy. Still
        # require an actual remote output for mandatory declarations.
        if not is_cache and policy == FileSyncPolicy.MISSING and local_exists:
            if required and await self.transport.file_signature(remote) is None:
                raise FileNotFoundError(
                    f"Declared remote output was not created: {remote}"
                )
            return "skipped", None

        if is_cache:
            after = await self.transport.file_signature(remote)
            if after is None or after == before:
                return "skipped", None
            if policy == FileSyncPolicy.MISSING and local_exists:
                return "skipped", None
        elif policy == FileSyncPolicy.CHANGED and local_exists:
            remote_digest = await self.transport.file_digest(remote)
            if remote_digest is None:
                if required:
                    raise FileNotFoundError(
                        f"Declared remote output was not created: {remote}"
                    )
                return "skipped", None
            try:
                version = await asyncio.to_thread(
                    FileVersion.capture, local, require_digest=True,
                )
            except FileNotFoundError:
                # The existing local copy vanished before it could be hashed.
                pass
            else:
                if version.digest == remote_digest:
                    return "available", version
        elif await self.transport.file_signature(remote) is None:
            if required:
                raise FileNotFoundError(
                    f"Declared remote output was not created: {remote}"
                )
            return "skipped", None

        await self.transport.get_file(remote, local)
        return "available", await asyncio.to_thread(
            FileVersion.capture, local, require_digest=True,
        )

    async def collect(self, mapping, manifest, result, *, required=False):
        """Collect independent file results; retain successes after a failure."""
        if manifest is None:
            return None, None
        primary = {p for p in (result.output_file, result.error_file) if p}
        baselines = {
            cache.remote_path: cache.pre_run_signature for cache in manifest.caches
            if cache.local_path is None
        }
        # An optional wildcard and a mandatory declaration can refer to one
        # concrete output. The strictest requirement governs both aliases.
        mandatory_outputs = {
            (entry.local_path, entry.remote_path)
            for entry in manifest.outputs if not entry.optional
        }

        seen = set()
        first_error = None
        for kind, entries in (
            ("output", manifest.outputs),
            ("cache", manifest.caches),
        ):
            is_cache = kind == "cache"
            for entry in entries:
                if entry.local_path is None or entry.remote_path is None:
                    continue
                if (entry.availability_state == "skipped"
                        and not (mapping.output_sync == FileSyncPolicy.MISSING
                                 and not os.path.isfile(entry.local_path))):
                    continue
                if await asyncio.to_thread(entry.verified_delivery_is_current):
                    continue

                identity = (kind, entry.remote_path, entry.local_path)
                if identity in seen:
                    continue
                seen.add(identity)
                if not is_cache and entry.local_path in primary:
                    continue

                try:
                    if has_file_magic(entry.remote_path):
                        raise RuntimeError("Cannot collect unresolved wildcard")
                    before = (
                        baselines.get(entry.remote_path, entry.pre_run_signature)
                        if is_cache else None
                    )
                    mandatory = required and (
                        not entry.optional if is_cache else
                        (entry.local_path, entry.remote_path) in mandatory_outputs
                    )
                    state, version = await self._sync_file(
                        mapping, entry, is_cache=is_cache, before=before,
                        required=mandatory,
                    )
                except FILE_DELIVERY_ERRORS as exc:
                    manifest = manifest.with_file_availability(
                        entry.local_path, "failed", str(exc),
                        kind=kind, remote_path=entry.remote_path,
                    )
                    if first_error is None:
                        first_error = exc
                else:
                    manifest = manifest.with_file_availability(
                        entry.local_path, state, kind=kind,
                        remote_path=entry.remote_path,
                        verified_local_version=version,
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
                version = await self._verify_stream(entry, kind, spec)
            except FILE_DELIVERY_ERRORS as exc:
                manifest = manifest.with_file_availability(
                    entry.local_path, "failed", str(exc),
                    remote_path=entry.remote_path,
                )
                if first_error is None:
                    first_error = exc
            else:
                manifest = manifest.with_file_availability(
                    entry.local_path, "available", remote_path=entry.remote_path,
                    verified_local_version=version,
                )
        return manifest, first_error

    async def _verify_stream(self, entry, kind, spec):
        """Verify one streamed file against its remote authoritative contents."""
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
        # One stable content capture both verifies the stream and supplies its
        # durable SHA-256 identity; do not read the entire file a second time.
        version = await asyncio.to_thread(
            FileVersion.capture, entry.local_path, require_digest=True,
        )
        digest = await self.transport.file_digest(entry.remote_path)
        if digest is None or version.digest != digest:
            raise RuntimeError(
                f"Streamed local {kind} contents differ from remote output: "
                f"{entry.local_path}"
            )
        if await self.transport.file_signature(entry.remote_path) != signature:
            raise RuntimeError(f"Remote {kind} output changed during verification")
        before = file_fingerprint(entry.local_path)
        if before != version.fingerprint:
            raise RuntimeError(f"Local {kind} output changed during verification")
        os.utime(entry.local_path, ns=(stat.st_atime_ns, signature[1]))
        after = file_fingerprint(entry.local_path)
        if ((after.device, after.inode, after.mode, after.size)
                != (before.device, before.inode, before.mode, before.size)
                or after.mtime_ns != signature[1]):
            raise RuntimeError(f"Local {kind} output changed during timestamp update")
        return replace(version, fingerprint=after)

    @staticmethod
    def plan_restore(run, paths, *, destinations=None):
        """Validate the complete historical request without filesystem writes.

        The manifest owns membership and canonical source identities. The
        returned plan is also used by restore(), so preflight and transfer
        enforce precisely the same rules.
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

        staged_targets = {
            os.path.abspath(os.fspath(target)) for target in destinations.values()
        }
        # Explicit staging targets are owned by JobFiles: an existing target
        # is an error, never evidence of a completed historical download.
        # Canonical direct fetches retain their documented keep-existing
        # behavior for callers that do not request staging.
        for _, target in planned.values():
            if target in staged_targets and os.path.lexists(target):
                raise FileExistsError(
                    f"Historical staging destination already exists: {target}"
                )
        return tuple(planned.values()), staged_targets

    async def restore(self, run, paths, *, destinations=None):
        """Restore the preflighted batch, without changing its file identities."""
        planned, staged_targets = self.plan_restore(
            run, paths, destinations=destinations,
        )
        for remote, target in planned:
            if target in staged_targets or not os.path.isfile(target):
                await self.transport.get_file(remote, target)
