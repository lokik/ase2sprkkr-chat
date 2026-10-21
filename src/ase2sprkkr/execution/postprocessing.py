"""One public calculation lifecycle over an executor-owned backend monitor.

The backend's completion callback runs on its existing monitor thread, after
stream delivery and cleanup. Task postprocessing and durable project completion
therefore need no additional waiter thread. This class does not query the
backend to decide whether execution has finished.
"""
from __future__ import annotations

import asyncio
from threading import Event, Lock

from .base import FailurePhase, RunState


class CalculationRun:
    """Interpret one backend observation and publish the caller-visible result.

    The raw runner owns the backend execution and its monitoring thread. This
    runner owns the postprocessing result, its public completion event and
    optional resource ownership. Its callback must persist project completion
    before returning; callbacks are delivered before waiters are unblocked.
    """

    def __init__(self, runner, process_result, callback=None, *, resources=None):
        self._runner = runner
        self._process_result = process_result
        self.callback = callback
        self._resources = resources
        self.result = None
        self._state = RunState.NEW
        self._failure_phase = None
        self._failure_message = None
        self._exception = None
        self._callback_exception = None
        self._done = Event()
        self._finish_lock = Lock()
        self._start_lock = Lock()
        self._started = False
        self._starting = False
        self._processing = False
        self._settled = False

        # A backend publishes its observation once, after it has finalized
        # stdout/stderr and artifact delivery. Never call raw.wait() here:
        # finish_monitor() has not yet signalled the backend's own event.
        previous = getattr(runner, "callback", None)
        if previous is not None:
            raise ValueError("Raw runner already has a completion callback")
        runner.callback = self._backend_completed

    @property
    def spec(self):
        return self._runner.spec

    @property
    def executor(self):
        return self._runner.executor

    @property
    def backend_id(self):
        return self._runner.backend_id

    @property
    def execution(self):
        return getattr(self._runner, "execution", None)

    @property
    def files(self):
        return getattr(self._runner, "files", None)

    @property
    def exception(self):
        return self._exception or self._runner.exception

    @property
    def failure_phase(self):
        return self._failure_phase or getattr(self._runner, "failure_phase", None)

    @property
    def failure_message(self):
        return self._failure_message or getattr(self._runner, "failure_message", None)

    @property
    def return_code(self):
        value = getattr(self.result, "return_code", None)
        if value is not None:
            return value
        return getattr(self._runner, "return_code", None)

    @property
    def state(self):
        if self._settled:
            return self._state
        if self._processing:
            return RunState.POSTPROCESSING
        raw_state = self._runner.state
        if raw_state == RunState.FINISHED and (self._started or self._starting):
            return RunState.POSTPROCESSING
        return raw_state

    def add_output_listener(self, callback):
        return self._runner.add_output_listener(callback)

    def add_output_error_listener(self, callback):
        return self._runner.add_output_error_listener(callback)

    def _close_resources(self):
        if self._resources is not None:
            self._resources.close(suppress_errors=True)

    def _backend_completed(self, outcome):
        """The backend monitor's completion hook; must never wait on itself."""
        self._finish_observation(outcome)

    def _finish_observation(self, outcome, error=None):
        # A caller's wait() may race with the monitor callback. Neither
        # completion processing nor the durable callback may execute twice.
        with self._finish_lock:
            if self._settled:
                return
            try:
                if error is not None:
                    raise error
                raw_state = self._runner.state
                self.result = outcome
                if raw_state == RunState.CANCELLED:
                    self._state = RunState.CANCELLED
                elif raw_state == RunState.FAILED:
                    self._state = RunState.FAILED
                    self._failure_phase = (
                        getattr(self._runner, "failure_phase", None)
                        or FailurePhase.EXECUTION
                    )
                    self._failure_message = getattr(
                        self._runner, "failure_message", None
                    )
                    self._exception = self._runner.exception
                elif raw_state == RunState.UNKNOWN:
                    # UNKNOWN finishes only the current observation, not the
                    # backend job: no terminal project commit may be written.
                    self._state = RunState.UNKNOWN
                    self._failure_message = getattr(
                        self._runner, "failure_message", None
                    )
                elif raw_state == RunState.FINISHED:
                    self._processing = True
                    try:
                        self.result = self._process_result(outcome)
                    except BaseException as exc:
                        self._exception = exc
                        self._failure_phase = FailurePhase.POSTPROCESSING
                        self._failure_message = str(exc)
                        self._state = RunState.FAILED
                    else:
                        self._state = RunState.FINISHED
                    finally:
                        self._processing = False
                else:
                    raise RuntimeError(
                        f"Backend observation ended in unexpected state {raw_state.value}"
                    )
            except BaseException as exc:
                self._exception = exc
                self._failure_phase = (
                    getattr(self._runner, "failure_phase", None)
                    or FailurePhase.EXECUTION
                )
                self._failure_message = (
                    getattr(self._runner, "failure_message", None) or str(exc)
                )
                try:
                    cancelled = self._runner.state == RunState.CANCELLED
                except Exception:
                    cancelled = False
                self._state = RunState.CANCELLED if cancelled else RunState.FAILED
            finally:
                # The persistent store is updated by callback before the run
                # is observable as completed by wait(). Callback failures must
                # not change the already-established execution result.
                self._settled = True
                try:
                    if self.callback is not None:
                        self.callback(self.result)
                except BaseException as exc:
                    self._callback_exception = exc
                finally:
                    try:
                        self._close_resources()
                    finally:
                        self._done.set()

    def prepare_submission(self):
        prepare = getattr(self._runner, "prepare_submission", None)
        if prepare is None:
            return False
        prepare()
        return True

    def _start(self, *, attached=False):
        with self._start_lock:
            if self._started or self._starting or self._settled:
                raise RuntimeError(f"Run is already {self.state.value}")
            self._starting = True
        try:
            if attached:
                start = getattr(self._runner, "start", None)
                if start is not None:
                    start()
            else:
                self._runner.submit()
        except BaseException:
            with self._start_lock:
                self._starting = False
            raise
        else:
            with self._start_lock:
                self._started = True
                self._starting = False

    def submit(self):
        if self._runner.state != RunState.NEW:
            raise RuntimeError(f"Run is already {self.state.value}")
        self._start()

    def start(self):
        if self._started:
            return
        self._start(attached=self._runner.state != RunState.NEW)

    def wait(self):
        if not self._started and not self._settled:
            raise RuntimeError("Run has not been started")
        # The backend event fires after its completion callback. For small
        # external runners without callback delivery, waiting may still
        # supply the observation; the same idempotent finalizer handles it.
        try:
            outcome = self._runner.wait()
        except BaseException as exc:
            # A completed callback may already own the canonical outcome.
            # Never replace it with an error from raw.wait().
            if not self._done.is_set():
                self._finish_observation(
                    getattr(self._runner, "result", None), error=exc,
                )
        else:
            self._finish_observation(outcome)
        self._done.wait()
        if self._exception is not None:
            raise self._exception
        if self._callback_exception is not None:
            raise self._callback_exception
        return self.result

    def run(self):
        if not self._started and not self._settled:
            self.start()
        return self.wait()

    async def run_async(self):
        return await asyncio.to_thread(self.run)

    def stop_observation(self):
        stop = getattr(self._runner, "stop_observation", None)
        if stop is None:
            raise NotImplementedError(
                "This execution backend cannot detach its observation"
            )
        return stop()

    def cancel(self):
        return self._runner.cancel()


# Retain the published import name while moving the implementation to the
# single-observer CalculationRun contract. No second monitor thread exists.
PostprocessingRunner = CalculationRun
