"""Client-side remote execution staging, streaming and artifact synchronization."""

from __future__ import annotations

import asyncio
import fcntl
import os
import posixpath
from threading import Event, Lock, Thread

from .base import (
    ExecutionOutcome,
    FailurePhase,
    OutputObservers,
    close_output_files,
    file_path,
    finish_monitor,
    wait_for_monitor,
    RunState,
    request_cancellation,
    start_monitor,
)
from ..common.async_utils import run_coro_sync as _run_sync
from .remote_file_handler import FileSyncPolicy, RemoteFileHandler




class RemoteRunner:
    """Client-side raw execution identified by an opaque backend ID.

    The remote target owns only backend facts. This runner owns staging,
    streaming and the explicit ``DOWNLOADING`` phase in which declared artifacts
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
    ):
        self.executor = executor
        self.spec = spec
        self.callback = callback
        self._state = RunState(state)
        self._backend_id = backend_id
        self._remote_directory = remote_directory
        self._output_observers = OutputObservers(output_listener)
        self._lock = Lock()
        self._done = Event()
        self._thread = None
        self._exception = None
        self._callback_exception = None
        self._return_code = None
        self._failure_phase = None
        self._failure_message = None
        self._follow_process = None
        self._stream_positions = {}
        self._cancel_future = None
        self.result = ExecutionOutcome(
            output_file=file_path(spec.output_file, spec.directory, flush=True, required=spec.output_file is not None),
            input_file=file_path(spec.input_file, spec.directory, flush=True, required=spec.input_file is not None),
            error_file=file_path(spec.error_file, spec.directory, flush=True, required=spec.error_file is not None),
        )

        self.file_handler = getattr(
            getattr(executor, "endpoint", executor),
            "file_handler", None,
        ) or RemoteFileHandler(getattr(executor, "transport", None))
        self._files = files

        if start:
            self.start()

    @property
    def backend_id(self):
        return self._backend_id

    @property
    def files(self):
        return self._files

    @property
    def execution(self):
        if self._backend_id is None and self._remote_directory is None:
            return None
        from .endpoint import endpoint_for_executor
        from .run_model import Execution, WorkingDirectory
        return Execution(
            endpoint_for_executor(self.executor.endpoint),
            WorkingDirectory(os.path.abspath(self.spec.directory), self._remote_directory),
            backend_id=self._backend_id,
        )

    @property
    def state(self):
        # Once monitoring has started, only the client monitor may advance from
        # backend execution into DOWNLOADING. A direct backend query must never
        # expose FINISHED before artifacts are locally available.
        if self._thread is not None:
            return self._state
        if self._backend_id is None or self._state in (
            RunState.NEW,
            RunState.DOWNLOADING,
            RunState.FINISHED,
            RunState.FAILED,
            RunState.CANCELLED,
        ):
            return self._state
        status = _run_sync(
            self.executor._status(
                self._remote_directory,
                self._backend_id,
                self._files,
            )
        )
        self._state = RunState(status.get("state", "unknown"))
        if status.get("return_code") is not None:
            self._return_code = status["return_code"]
            self.result.return_code = self._return_code
        phase = status.get("failure_phase")
        self._failure_phase = None if phase is None else FailurePhase(phase)
        self._failure_message = status.get("error")
        return self._state

    @property
    def exception(self):
        return self._exception

    @property
    def return_code(self):
        return self._return_code

    @property
    def failure_phase(self):
        return self._failure_phase

    @property
    def failure_message(self):
        return self._failure_message

    def add_output_listener(self, callback):
        self._output_observers.add(callback)

    def add_output_error_listener(self, callback):
        self._output_observers.add_error_listener(callback)

    def _mark_pending_files_failed(self, message):
        if self._files is None:
            return
        for entry in self._files.artifacts:
            if entry.availability_state in (None, "pending"):
                self._files = self._files.with_file_availability(
                    entry.local_path, "failed", message,
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
        target.flush()
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

    async def _pump(self, source, kind):
        while True:
            data = await source.read(65536)
            if not data:
                return
            self._write_output(data, kind)
            self._output_observers.emit_bytes(data, kind)

    async def _monitor(self):
        stdout_task = stderr_task = None
        try:
            offsets = {}
            for kind, target in (("out", self.spec.output_file),
                                 ("err", self.spec.error_file)):
                if target is None:
                    offsets[kind] = None
                elif kind == "err" and target is self.spec.output_file:
                    offsets[kind] = offsets["out"]
                else:
                    target.flush()
                    offsets[kind] = os.path.getsize(self._local_path(target))
            self._stream_positions = dict(offsets)
            try:
                self._follow_process = await self.executor._follow(
                    self._remote_directory, self._backend_id, self._files,
                    offsets=offsets,
                )
            except (OSError, ConnectionError) as exc:
                # An unreachable transport is not a failed calculation.
                self._state = RunState.UNKNOWN
                self._failure_phase = None
                self._failure_message = str(exc)
                return self.result
            stdout_task = asyncio.create_task(
                self._pump(self._follow_process.stdout, "out")
            )
            stderr_task = asyncio.create_task(
                self._pump(self._follow_process.stderr, "err")
            )
            stream_error = None
            try:
                await asyncio.gather(
                    self._follow_process.wait(), stdout_task, stderr_task
                )
            except Exception as exc:
                stream_error = exc

            # The backend outcome is authoritative. A broken observational
            # stream cannot turn a successful calculation into FAILED.
            status = self._follow_process.terminal
            if status is None:
                try:
                    status = await self.executor._status(
                        self._remote_directory,
                        self._backend_id,
                        self._files,
                    )
                except (OSError, ConnectionError) as exc:
                    self._state = RunState.UNKNOWN
                    self._failure_phase = None
                    self._failure_message = str(stream_error or exc)
                    return self.result
            backend_state = RunState(status.get("state", "unknown"))
            self._return_code = status.get("return_code")
            self.result.return_code = self._return_code
            phase = status.get("failure_phase")
            self._failure_phase = None if phase is None else FailurePhase(phase)
            self._failure_message = status.get("error")

            if stream_error is not None and backend_state not in (
                RunState.FINISHED, RunState.FAILED, RunState.CANCELLED,
            ):
                self._state = RunState.UNKNOWN
                self._failure_message = str(stream_error)
                return self.result

            if backend_state == RunState.FAILED and self._failure_phase in (
                FailurePhase.SUBMISSION,
                FailurePhase.LAUNCH,
            ):
                self._state = RunState.FAILED
                return self.result

            if backend_state not in (
                RunState.FINISHED,
                RunState.FAILED,
                RunState.CANCELLED,
            ):
                # The stream helper may stop because a backend identity was lost
                # (for example a direct-process hard crash before its durable
                # outcome barrier). Received bytes remain on disk, but no
                # successful execution may be reported.
                self._state = RunState.UNKNOWN
                self._failure_phase = None
                if self._failure_message is None:
                    self._failure_message = (
                        "Remote output stream ended without a terminal backend "
                        f"state ({backend_state.value})"
                    )
                return self.result

            self._state = RunState.DOWNLOADING
            try:
                self._files = self.file_handler.freeze_from_outcome(
                    self._files, status.get("matched_files"),
                )
                self._files, transfer_error = await self.file_handler.collect(
                    self.executor, self._files, self.result,
                    required=backend_state == RunState.FINISHED,
                )
                if transfer_error is not None:
                    raise transfer_error
            except Exception as exc:
                self._mark_pending_files_failed(str(exc))
                if backend_state == RunState.FINISHED:
                    # A successful calculation with unavailable artifacts is a
                    # failed run, even though its backend return code is zero.
                    self._state = RunState.FAILED
                    self._failure_phase = FailurePhase.DOWNLOAD
                    self._failure_message = str(exc)
                    raise
                # Diagnostic download failure cannot replace the original
                # failed/cancelled backend outcome or its failure reason.
                self._state = backend_state
                if backend_state == RunState.FAILED and self._failure_phase is None:
                    self._failure_phase = FailurePhase.EXECUTION
                return self.result

            if self._files is not None:
                for entry in self._files.artifacts:
                    if entry.availability_state == "pending":
                        self._files = self._files.with_file_availability(entry.local_path, "available")
            self._state = backend_state
            if self._state == RunState.FAILED and self._failure_phase is None:
                self._failure_phase = FailurePhase.EXECUTION
            return self.result
        except BaseException as exc:
            if self._failure_phase is None:
                self._failure_phase = FailurePhase.DOWNLOAD
            if self._failure_message is None:
                self._failure_message = str(exc)
            if self._state != RunState.CANCELLED:
                self._state = RunState.FAILED
            raise
        finally:
            tasks = tuple(
                task for task in (stdout_task, stderr_task) if task is not None
            )
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            # Files already contain every received byte, including incomplete
            # sessions. Closing flushes them without any publish/discard step.
            close_output_files(self.spec.output_file, self.spec.error_file)

    def _thread_main(self):
        try:
            asyncio.run(self._monitor())
        except BaseException as exc:
            self._exception = exc
        finally:
            finish_monitor(self)

    def start(self):
        if self._backend_id is None:
            return self.submit()
        return start_monitor(self, missing_backend="Run has no backend reference")

    def submit(self):
        raise NotImplementedError(
            "Remote submission is provided by the endpoint-specific runner"
        )

    def wait(self):
        return wait_for_monitor(self)

    def run(self):
        if self._thread is None:
            self.start()
        return self.wait()

    async def run_async(self):
        return await asyncio.to_thread(self.run)

    def cancel(self):
        if self._backend_id is None:
            raise RuntimeError("Run has not been submitted")
        return request_cancellation(
            self,
            lambda: _run_sync(
                self.executor._cancel(self._remote_directory, self._backend_id)
            ),
        )
