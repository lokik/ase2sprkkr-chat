"""Local subprocess execution backend."""

from __future__ import annotations

import asyncio
import os
import stat
import sys
from threading import Event, Lock, Thread

from .run_files import RunFiles
from .run_model import FileManifest
from .base import (
    ExecutionOutcome,
    ExecutionSpec,
    Executor,
    FailurePhase,
    OutputObservers,
    close_output_files,
    finish_monitor,
    file_path,
    wait_for_monitor,
    RunState,
    SubmissionNotStarted,
)
from .file_availability import prepare_local_files
from .local_process import (
    materialize_local_spec,
    pump_stream,
    start_local_process,
    stop_local_process,
)


class LocalRunner:
    """One local subprocess execution.

    The runner owns only process lifecycle, output-file writing and raw output
    observation. It deliberately performs no task-specific parsing.
    """

    def __init__(self, spec: ExecutionSpec, callback=None):
        self.spec = spec
        self.run_files = RunFiles(FileManifest.from_spec(spec))
        self.callback = callback
        self.result = ExecutionOutcome.from_spec(spec)

        self._state = RunState.NEW
        self._output_observers = OutputObservers()
        self._lock = Lock()
        self._cancel_requested = Event()
        self._started = Event()
        self._done = Event()

        self._thread = None
        self._loop = None
        self._process = None
        self._pid = None
        self._stop_task = None
        self._exception = None
        self._callback_exception = None
        self._failure_phase = None
        self._failure_message = None
        self._stream_handles = {}
        self._owned_streams = []
        self._owned_stdin = None
        self._cleanup_exception = None

    def prepare_submission(self):
        """Local execution has no separate staging phase."""

    def stop_observation(self):
        raise NotImplementedError("Local execution cannot detach observation")

    @property
    def state(self):
        return self._state

    @property
    def exception(self):
        return self._exception

    @property
    def backend_id(self):
        return None if self._pid is None else str(self._pid)

    @property
    def failure_phase(self):
        return self._failure_phase

    @property
    def failure_message(self):
        return self._failure_message

    @property
    def cleanup_exception(self):
        return self._cleanup_exception

    @property
    def return_code(self):
        return self.result.return_code

    def add_output_listener(self, callback):
        self._output_observers.add(callback)

    def add_output_error_listener(self, callback):
        self._output_observers.add_error_listener(callback)

    def _output_target(self, kind):
        # A missing stderr target means live observation only.
        return self._stream_handles.get(kind)

    @staticmethod
    def _stream_inode(value):
        """Identify a regular file, including aliases and open descriptors."""
        try:
            if isinstance(value, (str, bytes, os.PathLike)):
                info = os.stat(value)
            else:
                info = os.fstat(value.fileno())
        except (AttributeError, OSError, TypeError, ValueError):
            return None
        if not stat.S_ISREG(info.st_mode):
            return None
        return (info.st_dev, info.st_ino)

    def _open_streams(self, *, stdin):
        targets = (("out", self.spec.output_file),
                   ("err", self.spec.error_file))
        by_inode = {}
        by_path = {}

        # Reuse pre-opened file objects before opening any named target:
        # opening a second alias with "wb" would truncate their contents.
        for kind, value in targets:
            if not isinstance(value, (str, bytes, os.PathLike)):
                inode = self._stream_inode(value) if value is not None else None
                self._stream_handles[kind] = (
                    by_inode.setdefault(inode, value) if inode is not None else value
                )

        for kind, value in targets:
            if not isinstance(value, (str, bytes, os.PathLike)):
                continue
            path = file_path(value, self.spec.directory)
            inode = self._stream_inode(path)
            if path in by_path:
                handle = by_path[path]
            elif inode is not None and inode in by_inode:
                handle = by_inode[inode]
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                handle = open(path, "wb")
                self._owned_streams.append(handle)
                inode = self._stream_inode(handle)
                if inode is not None:
                    by_inode[inode] = handle
            by_path[path] = handle
            self._stream_handles[kind] = handle

        if isinstance(stdin, (str, bytes, os.PathLike)):
            self._owned_stdin = open(file_path(stdin, self.spec.directory), "rb")
            return self._owned_stdin
        return stdin

    def _forward_output(self, data, kind):
        target = self._output_target(kind)
        if target is not None:
            target.write(data)
            target.flush()
        self._output_observers.emit_bytes(data, kind)

    def _ensure_stop_task(self):
        if self._process is None or self._process.returncode is not None:
            return None
        if self._stop_task is None or self._stop_task.done():
            self._stop_task = asyncio.create_task(stop_local_process(self._process))
        return self._stop_task

    def cancel(self):
        self._cancel_requested.set()
        loop = self._loop
        if loop is None:
            if self._state == RunState.NEW:
                self._state = RunState.CANCELLED
            return
        try:
            loop.call_soon_threadsafe(self._ensure_stop_task)
        except RuntimeError:
            pass

    async def _execute(self):
        self._loop = asyncio.get_running_loop()
        try:
            if self._cancel_requested.is_set():
                self._state = RunState.CANCELLED
                self._started.set()
                return self.result
            prepare_local_files(
                self.spec, preserve_open_streams=True,
            )
            try:
                runtime_spec = materialize_local_spec(self.spec)
                stdin = self._open_streams(stdin=runtime_spec.stdin)
                self._process = await start_local_process(runtime_spec, stdin=stdin)
            except BaseException as exc:
                self._failure_phase = FailurePhase.LAUNCH
                self._failure_message = str(exc)
                self._state = RunState.FAILED
                raise

            self._pid = self._process.pid
            self._state = RunState.RUNNING
            self._started.set()

            if self._cancel_requested.is_set():
                stop_task = self._ensure_stop_task()
                if stop_task is not None:
                    await stop_task

            wait_task = asyncio.create_task(self._process.wait())
            stdout_task = asyncio.create_task(
                pump_stream(
                    self._process.stdout,
                    lambda data: self._forward_output(data, "out"),
                )
            )
            stderr_task = asyncio.create_task(
                pump_stream(
                    self._process.stderr,
                    lambda data: self._forward_output(data, "err"),
                )
            )
            tasks = (wait_task, stdout_task, stderr_task)
            try:
                return_code, _, _ = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            self.result.return_code = return_code
            if self._cancel_requested.is_set():
                self._state = RunState.CANCELLED
            elif return_code == 0:
                self._state = RunState.FINISHED
            else:
                self._state = RunState.FAILED
                self._failure_phase = FailurePhase.EXECUTION
                self._failure_message = f"Process exited with status {return_code}"
            return self.result
        except BaseException:
            self._started.set()
            if self._state not in (RunState.CANCELLED, RunState.FAILED):
                self._state = RunState.FAILED
                self._failure_phase = FailurePhase.EXECUTION
            if self._process is not None and self._process.returncode is None:
                await stop_local_process(self._process)
            raise
        finally:
            primary_error = sys.exc_info()[1]
            cleanup_error = None
            try:
                if self._stop_task is not None:
                    await asyncio.gather(self._stop_task, return_exceptions=True)
                self._output_observers.finish_bytes()
            except BaseException as exc:
                cleanup_error = exc
            try:
                close_output_files(
                    *self._owned_streams, self._owned_stdin,
                    self.spec.output_file, self.spec.error_file,
                )
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            finally:
                self._process = None
                self._loop = None
            self._cleanup_exception = cleanup_error
            if cleanup_error is not None and primary_error is None:
                if self._state == RunState.FAILED:
                    # A nonzero exit remains the authoritative failure.
                    # The secondary close error is available separately.
                    return self.result
                self._state = RunState.FAILED
                self._failure_phase = FailurePhase.FILES
                self._failure_message = str(cleanup_error)
                raise cleanup_error

    def _thread_main(self):
        try:
            asyncio.run(self._execute())
        except BaseException as exc:
            self._exception = exc
            if self._failure_message is None:
                self._failure_message = str(exc)
        finally:
            finish_monitor(self)

    def submit(self):
        with self._lock:
            if self._state == RunState.CANCELLED and self._thread is None:
                raise RuntimeError("Run was cancelled before submission")
            if self._state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._state.value}")
            self._state = RunState.SUBMITTING
            self._thread = Thread(target=self._thread_main, daemon=True)
            self._thread.start()

        self._started.wait()
        if self._pid is None and self._state == RunState.FAILED:
            self._done.wait()
            error = self._exception or RuntimeError(self._failure_message or "Launch failed")
            raise SubmissionNotStarted(str(error)) from self._exception

    def wait(self):
        if self._state == RunState.NEW:
            raise RuntimeError("Run has not been submitted")
        if self._state == RunState.CANCELLED and self._thread is None:
            return self.result
        return wait_for_monitor(self)

    def run(self):
        if self._state == RunState.NEW:
            self.submit()
        return self.wait()

    async def run_async(self):
        return await asyncio.to_thread(self.run)


class LocalExecutor(Executor):
    """Create raw runners executed as local subprocesses."""

    name = "local"

    def __init__(self):
        from .target_connection import LocalConnection
        self.connection = LocalConnection()

    def abandoned_preparation_is_safe(self):
        return True

    def create_run(self, spec: ExecutionSpec, callback=None):
        return LocalRunner(self.resolve_spec(spec), callback=callback)

    def create_project_run(
        self, spec: ExecutionSpec, *, run_id: str, callback=None
    ):
        from .local_project import LocalProjectRunner

        return LocalProjectRunner(
            self,
            self.resolve_spec(spec),
            run_id=run_id,
            callback=callback,
        )

    def recover_submission(self, run_id, submission, *, submitted_at=None):
        from ..common.async_utils import run_coro_sync as _run_sync
        from .local_project import _local_status_from_facts
        reply = _run_sync(self.connection.recover(
            run_id, submission.execution.directory.local, submitted_at=submitted_at,
        ))
        if reply is None:
            # The absence of a ticket is not proof that submission never
            # happened. Never authorize an automatic duplicate launch.
            raise NotImplementedError("Submission identity is not recoverable")
        return _local_status_from_facts(
            reply["remote_id"], reply, submission.files,
        )

    def attach(self, run, callback=None, output_listener=None, start=True):
        from .local_project import attach_local
        return attach_local(
            self, run, callback=callback,
            output_listener=output_listener, start=start,
        )

    def acknowledge_submission(self, run):
        from ..common.async_utils import run_coro_sync as _run_sync
        _run_sync(self.connection.acknowledge_submission(run.id))

    def acknowledge(self, run):
        from ..common.async_utils import run_coro_sync as _run_sync
        _run_sync(self.connection.acknowledge(run.execution.backend_id))

    def status(self, *runs):
        from .local_project import local_status
        statuses = []
        for run in runs:
            if run.execution.endpoint.executor != "local":
                raise ValueError("Run does not belong to the local backend")
            statuses.append(local_status(run.execution.backend_id, run.files))
        return tuple(statuses)
