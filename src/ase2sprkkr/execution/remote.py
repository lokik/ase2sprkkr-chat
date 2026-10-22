"""Client-side remote execution staging, streaming and file synchronization."""

from __future__ import annotations

import asyncio
import fcntl
import os
from dataclasses import asdict

from .base import FailurePhase, RunState, SubmissionNotStarted, file_path, start_monitor
from ..common.async_utils import run_coro_sync as _run_sync
from .remote_file_handler import FileSyncPolicy
from .job_files import JobFiles
from .managed_runner import ManagedRunner
from .supervisor_monitor import backend_status_from_facts, monitor_supervisor




class RemoteRunner(ManagedRunner):
    """Client-side raw execution identified by an opaque backend ID.

    The remote target owns only backend facts. This runner owns staging,
    streaming and the explicit ``DOWNLOADING`` phase in which declared files
    become locally available. It performs no task-specific parsing.
    """

    def __init__(
        self,
        executor,
        spec,
        callback=None,
        *,
        output_listener=None,
        start=False,
        backend_id=None,
        remote_directory=None,
        files=None,
        state=RunState.NEW,
        run_id=None,
    ):
        super().__init__(
            executor, spec, callback, output_listener=output_listener,
            backend_id=backend_id, state=state, files=files,
            strict_paths=True,
        )
        self._run_id = run_id
        self._prepared = None
        self._remote_directory = remote_directory
        self._stream_positions = {}
        self.file_handler = executor.file_handler
        if start:
            self.start()

    @property
    def state(self):
        # Once monitoring has started, only the client monitor may advance from
        # backend execution into DOWNLOADING. A direct backend query must never
        # expose FINISHED before files are locally available.
        if self._thread is not None:
            return self._state
        if (self._backend_id is None or self._state in (RunState.NEW, RunState.DOWNLOADING)
                or self._state.is_terminal):
            return self._state
        status = _run_sync(
            self.executor.connection.status(self._backend_id, self._job_files.manifest)
        )
        return self._apply_status(
            backend_status_from_facts(
                status, self._backend_id, needs_delivery=True,
            )
        )

    def _local_path(self, value):
        return file_path(value, self.spec.directory, flush=True, required=value is not None)

    def _write_output(self, data, kind):
        """Persist FOLLOW bytes at their logical offset, not blindly append.

        Another client may attach to the same working directory while this
        client is still following. Lock the destination inode while comparing
        overlapping bytes and extending only the missing suffix.
        """
        target = self.spec.error_file if kind == "err" else self.spec.output_file
        if target is None or not data:
            return
        offset = self._stream_positions[kind]
        path = self._local_path(target)
        with open(path, "r+b", buffering=0) as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                if size < offset:
                    raise RuntimeError(f"FOLLOW stream was truncated: {path}")
                overlap = min(size - offset, len(data))
                if overlap:
                    stream.seek(offset)
                    if stream.read(overlap) != data[:overlap]:
                        raise RuntimeError(f"Conflicting FOLLOW bytes at offset {offset}: {path}")
                if overlap < len(data):
                    stream.seek(offset + overlap)
                    stream.write(data[overlap:])
                    stream.flush()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self._stream_positions[kind] = offset + len(data)

    def _on_follow_chunk(self, kind, data):
        """Delivery policy is client-owned, not part of the SSH transport."""
        self._write_output(data, kind)
        self._output_observers.emit_bytes(data, kind)

    async def _monitor(self):
        return await monitor_supervisor(
            self, self._follow_once, self._backend_status,
            self._complete_terminal,
        )

    async def _follow_once(self):
        offsets = {}
        for kind, target in (("out", self.spec.output_file),
                             ("err", self.spec.error_file)):
            if target is None:
                offsets[kind] = None
            elif kind == "err" and target is self.spec.output_file:
                offsets[kind] = offsets["out"]
            else:
                offsets[kind] = os.path.getsize(self._local_path(target))
        self._stream_positions = dict(offsets)
        return await self.executor.connection.follow(
            self._backend_id, self._job_files.manifest, offsets=offsets,
            on_chunk=self._on_follow_chunk,
        )

    async def _backend_status(self):
        return await self.executor.connection.status(
            self._backend_id, self._job_files.manifest,
        )

    async def _complete_terminal(self, status, backend_state):
        if backend_state == RunState.FAILED and self._failure_phase in (
            FailurePhase.SUBMISSION, FailurePhase.LAUNCH,
        ):
            self._state = RunState.FAILED
            return

        self._state = RunState.DOWNLOADING
        try:
            from .run_model import FileManifest
            frozen = (
                FileManifest(**status["files"])
                if status.get("files") is not None else self._job_files.manifest
            )
            observed, transfer_error = await JobFiles.collect_remote(
                self._job_files.manifest, frozen, self.file_handler, self.result, self.spec,
                required=backend_state == RunState.FINISHED,
                materialization_error=status.get("materialization_error"),
            )
            self._job_files.merge(observed)
            if transfer_error is not None:
                raise transfer_error
        except Exception as exc:
            if backend_state == RunState.FINISHED:
                self._state = RunState.FAILED
                self._failure_phase = FailurePhase.DOWNLOAD
                self._failure_message = str(exc)
                raise
            self._state = backend_state
            if backend_state == RunState.FAILED and self._failure_phase is None:
                self._failure_phase = FailurePhase.EXECUTION
            return

        self._state = backend_state
        if self._state == RunState.FAILED and self._failure_phase is None:
            self._failure_phase = FailurePhase.EXECUTION

    async def _prepare_async(self):
        try:
            files = self._job_files.manifest
            mapping = files.mapping
            if mapping is None:
                mapping = await self.executor.resolve_mapping(self.spec.launch)
            remote_directory = mapping.map_path(self.spec.directory)
            await self.executor.transport.run(("mkdir", "-p", "--", remote_directory))
            manifest = await self.file_handler.prepare(mapping, self.spec, files)
            # The runner owns backend identity, command construction and the
            # submission protocol. FileHandler knows only file locations.
            mapped = {entry.local_path: entry.remote_path
                      for entry in (*manifest.inputs, *manifest.caches, manifest.stdin)
                      if entry is not None
                      if entry.local_path is not None}
            def argv(args):
                return [mapped.get(file_path(os.fspath(arg), self.spec.directory),
                                   os.fspath(arg)) for arg in args]
            request = {
                "id": self._run_id,
                "executor": self.executor.backend,
                "executor_options": self.executor.backend_options,
                "directory": remote_directory,
                **self.spec.command_request(argv),
                "input_file": (None if manifest.primary_input is None
                               else manifest.primary_input.remote_path),
                "stdin": (None if manifest.stdin is None
                          else manifest.stdin.remote_path),
                "output_file": (None if manifest.stdout is None
                                else manifest.stdout.remote_path),
                "error_file": (None if manifest.stderr is None
                               else manifest.stderr.remote_path),
                "files": asdict(manifest),
            }
            return request, manifest
        except Exception as exc:
            # No calculation has been launched yet. Persisted preflight and
            # direct submission must classify this known failure identically.
            self._failure_phase = FailurePhase.SUBMISSION
            self._failure_message = str(exc)
            raise SubmissionNotStarted(str(exc)) from exc

    def prepare_submission(self):
        """Complete all staging before the client commits SUBMITTING metadata."""
        with self._lock:
            if self._state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._state.value}")
            if self._prepared is not None:
                return
            request, manifest = _run_sync(self._prepare_async())
            self._prepared = (request, manifest)
            self._remote_directory = request["directory"]
            self._job_files.merge(manifest)

    async def _submit_async(self):
        if self._prepared is None:
            request, manifest = await self._prepare_async()
        else:
            request, manifest = self._prepared

        request["detached"] = self._run_id is not None
        status = await self.executor.connection.submit(request)
        remote_id = status.get("remote_id")
        if not remote_id:
            raise RuntimeError("Remote submit returned no remote_id")
        if status.get("files") is not None:
            from .run_model import FileManifest
            manifest = FileManifest(**status["files"])
        return remote_id, manifest, status, request

    def submit(self):
        with self._lock:
            if self._state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._state.value}")
            remote_id, manifest, status, request = _run_sync(self._submit_async())
            self._state = RunState(status["state"])
            self._return_code = status.get("return_code")
            self.result.return_code = self._return_code
            phase = status.get("failure_phase")
            self._failure_phase = None if phase is None else FailurePhase(phase)
            self._failure_message = status.get("error")
            self._backend_id = str(remote_id)
            self._remote_directory = request["directory"]
            self._job_files.merge(manifest)
        start_monitor(self, missing_backend="Run has no backend reference")

    def _cancel_backend(self):
        return _run_sync(
            self.executor.connection.cancel(self._backend_id)
        )
