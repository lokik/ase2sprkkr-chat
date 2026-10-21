"""Lifecycle of a reconnectable calculation, independent of its transport.

Local project and SSH executions differ in staging, FOLLOW delivery, artifact
synchronization and backend cancellation, not in client state ownership,
monitoring, observer/callback dispatch, or cancellation coordination.
Non-reconnectable LocalRunner intentionally does not inherit this class.
"""
from __future__ import annotations

import asyncio
from threading import Event, Lock

from .base import (
    ExecutionOutcome, FailurePhase, OutputObservers, RunState, file_path,
    finish_monitor, request_cancellation, start_monitor, wait_for_monitor,
)
from .supervisor_monitor import monitor_supervisor


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
        self._files = files
        self._output_observers = OutputObservers(output_listener)
        self._lock = Lock()
        self._done = Event()
        self._thread = None
        self._exception = None
        self._callback_exception = None
        self._cancel_future = None
        self.result = ExecutionOutcome(
            output_file=file_path(
                spec.output_file, spec.directory, flush=True,
                required=strict_paths and spec.output_file is not None,
            ),
            input_file=file_path(
                spec.input_file, spec.directory, flush=True,
                required=strict_paths and spec.input_file is not None,
            ),
            error_file=file_path(
                spec.error_file, spec.directory, flush=True,
                required=strict_paths and spec.error_file is not None,
            ),
        )

    @property
    def backend_id(self):
        return self._backend_id

    @property
    def files(self):
        return self._files

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
            self._files = status.files
        return self._state

    def add_output_listener(self, callback):
        self._output_observers.add(callback)

    def add_output_error_listener(self, callback):
        self._output_observers.add_error_listener(callback)

    async def _monitor(self):
        return await monitor_supervisor(
            self, self._follow_once, self._backend_status,
            self._complete_terminal,
        )

    def _monitor_failure(self, exc):
        self._exception = exc
        if self._failure_message is None:
            self._failure_message = str(exc)
        if self._failure_phase is None:
            self._failure_phase = FailurePhase.EXECUTION
        if self._state != RunState.CANCELLED:
            self._state = RunState.FAILED

    def _finish_monitoring(self):
        """Optional backend-specific observation cleanup before callbacks."""

    def _thread_main(self):
        """One exception, cleanup, callback and completion boundary for all monitors."""
        try:
            asyncio.run(self._monitor())
        except BaseException as exc:
            self._monitor_failure(exc)
        finally:
            try:
                self._finish_monitoring()
            except BaseException as exc:
                if self._exception is None:
                    self._monitor_failure(exc)
            finish_monitor(self)

    def start(self):
        if self.backend_id is None:
            return self.submit()
        return start_monitor(self, missing_backend=self._missing_backend_message)

    def wait(self):
        return wait_for_monitor(self)

    def run(self):
        if self._thread is None:
            self.start()
        return self.wait()

    async def run_async(self):
        return await asyncio.to_thread(self.run)

    def _can_cancel(self):
        return self.backend_id is not None

    def cancel(self):
        if not self._can_cancel():
            raise RuntimeError("Run has not been submitted")
        return request_cancellation(self, self._cancel_backend)
