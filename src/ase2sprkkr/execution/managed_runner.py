"""Lifecycle of a reconnectable calculation, independent of its transport.

Local project and SSH executions differ in staging, FOLLOW delivery, file
synchronization and backend cancellation, not in client state ownership,
observer/callback dispatch or cancellation coordination.
Non-reconnectable LocalRunner intentionally does not inherit this class.
"""
from __future__ import annotations

import asyncio
from threading import Event, Lock

from .base import (
    ExecutionOutcome, FailurePhase, OutputObservers, RunState,
    finish_monitor, request_cancellation, start_monitor, wait_for_monitor,
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
        self._files = files
        self._output_observers = OutputObservers(output_listener)
        self._lock = Lock()
        self._done = Event()
        self._thread = None
        self._exception = None
        self._callback_exception = None
        self._cancel_future = None
        self.result = ExecutionOutcome.from_spec(
            spec, flush=True, strict_paths=strict_paths,
        )

    @property
    def execution(self):
        """Execution metadata is provided by backends that carry it."""
        return None

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

    def prepare_submission(self):
        """Backends without separate staging use their submit operation."""

    def stop_observation(self):
        raise NotImplementedError("This execution backend cannot detach observation")

    def start(self):
        if self.backend_id is None:
            return self.submit()
        return start_monitor(self)

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
