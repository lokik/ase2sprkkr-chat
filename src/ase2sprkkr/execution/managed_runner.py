"""Lifecycle of a reconnectable calculation, independent of its transport.

Local project and SSH executions differ in staging, FOLLOW delivery, file
synchronization and backend cancellation, not in client state ownership,
observer/callback dispatch or cancellation coordination.
Non-reconnectable LocalRunner intentionally does not inherit this class.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Event, Lock

from .job_files import JobFiles
from .run_model import FileManifest
from .base import (
    ExecutionOutcome, FailurePhase, ObservationDetached, OutputObservers, RunState,
    finish_monitor, request_cancellation, start_monitor,
)


class ManagedRunner:
    """Shared client-side runner; target-specific operations are hooks."""

    _missing_backend_message = "Run has no backend reference"

    def __init__(
        self, executor, spec, callback=None, *, output_listener=None,
        backend_id=None, state=RunState.NEW, files=None, strict_paths=False,
    ):
        self.executor = executor
        self.spec = spec
        self.callback = callback
        self._backend_id = backend_id
        self._state = RunState(state)
        self._return_code = None
        self._failure_phase = None
        self._failure_message = None
        self.job_files = JobFiles(files if files is not None else FileManifest.from_spec(spec))
        self._output_observers = OutputObservers(output_listener)
        self._lock = Lock()
        self._done = Event()
        self._thread = None
        self._exception = None
        self._callback_exception = None
        self._cancel_future = None
        self._completion = Future()
        self._observation_stopped = Event()
        self._monitor_loop = None
        self._monitor_task = None
        self._finalizing = False
        self.observation_detached_callback = None
        self.result = ExecutionOutcome.from_spec(
            spec, flush=True, strict_paths=strict_paths,
        )

    @property
    def backend_id(self):
        return self._backend_id

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

    def _apply_status(self, status):
        """Update client observation without duplicating backend interpretation."""
        self._state = status.state
        self._failure_phase = status.failure_phase
        self._failure_message = status.message
        if status.return_code is not None or self._return_code is None:
            self._return_code = status.return_code
            self.result.return_code = status.return_code
        if status.files is not None:
            self.job_files.merge(status.files)
        return self._state

    def add_output_listener(self, callback):
        self._output_observers.add(callback)

    def add_output_error_listener(self, callback):
        self._output_observers.add_error_listener(callback)

    async def _monitor(self):
        """Backend-specific observation; subclasses choose the mechanism.

        The shared lifecycle deliberately knows nothing about supervisor
        FOLLOW, PSI/J callbacks, or how files become available.
        """
        raise NotImplementedError("Managed runner must implement observation")

    def _monitor_failure(self, exc):
        self._exception = exc
        if self._failure_message is None:
            self._failure_message = str(exc)
        if self._failure_phase is None:
            self._failure_phase = FailurePhase.EXECUTION
        if self._state != RunState.CANCELLED:
            self._state = RunState.FAILED

    def _finish_monitoring(self):
        """Flush incomplete UTF-8 data before notifying completion callbacks."""
        self._output_observers.finish_bytes()

    async def _observe(self):
        """Make the single monitor task cancellable from another thread."""
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(self._monitor())
        with self._lock:
            self._monitor_loop, self._monitor_task = loop, task
            detached = self._observation_stopped.is_set()
        if detached:
            task.cancel()
        try:
            await task
        finally:
            with self._lock:
                self._monitor_loop = None
                self._monitor_task = None

    def _publish_detachment(self):
        self._state = RunState.DETACHED
        try:
            if self.observation_detached_callback is not None:
                self.observation_detached_callback()
        finally:
            self._done.set()
            if not self._completion.done():
                self._completion.set_result(self.result)

    def _thread_main(self):
        """Finalize once: either a completed observation or a detached client."""
        try:
            asyncio.run(self._observe())
        except asyncio.CancelledError as exc:
            if not self._observation_stopped.is_set():
                self._monitor_failure(exc)
        except BaseException as exc:
            if not self._observation_stopped.is_set():
                self._monitor_failure(exc)
        finally:
            with self._lock:
                detached = self._observation_stopped.is_set()
                if not detached:
                    self._finalizing = True
            if detached:
                self._publish_detachment()
            else:
                try:
                    self._finish_monitoring()
                except BaseException as exc:
                    if self._exception is None:
                        self._monitor_failure(exc)
                try:
                    finish_monitor(self)
                finally:
                    self._completion.set_result(self.result)

    def prepare_submission(self):
        """Backends without separate staging use their submit operation."""

    def stop_observation(self):
        """Stop just this client's observation without touching the backend."""
        with self._lock:
            if (self._finalizing or self._observation_stopped.is_set()
                    or self._completion.done()):
                return False
            if self._thread is None and self.backend_id is None:
                raise RuntimeError("Run has not been submitted")
            self._observation_stopped.set()
            loop, task = self._monitor_loop, self._monitor_task
            no_monitor = self._thread is None
        if no_monitor:
            self._publish_detachment()
        elif loop is not None and task is not None:
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # The monitor thread is already shutting down.
                pass
        return True

    def start(self):
        if self._observation_stopped.is_set():
            raise ObservationDetached("Cannot restart a detached observer; attach anew")
        if self.backend_id is None:
            return self.submit()
        return start_monitor(self)

    def _wait_result(self):
        if self._state == RunState.DETACHED:
            raise ObservationDetached("Run is detached; attach a new observer")
        if self._exception is not None:
            raise self._exception
        if self._callback_exception is not None:
            raise self._callback_exception
        return self.result

    def wait(self):
        if self._thread is None and not self._completion.done():
            raise RuntimeError(
                "Run has not been submitted" if self._state == RunState.NEW
                else "Attached run has not been started"
            )
        self._completion.result()
        return self._wait_result()

    async def wait_async(self):
        if self._thread is None and not self._completion.done():
            raise RuntimeError(
                "Run has not been submitted" if self._state == RunState.NEW
                else "Attached run has not been started"
            )
        await asyncio.shield(asyncio.wrap_future(self._completion))
        return self._wait_result()

    def run(self):
        if self._thread is None:
            self.start()
        return self.wait()

    async def run_async(self):
        if self._thread is None:
            await asyncio.to_thread(self.start)
        return await self.wait_async()

    def _can_cancel(self):
        return self.backend_id is not None

    def cancel(self):
        if not self._can_cancel():
            raise RuntimeError("Run has not been submitted")
        return request_cancellation(self, self._cancel_backend)
