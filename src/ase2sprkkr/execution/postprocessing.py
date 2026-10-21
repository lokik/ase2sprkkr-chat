"""Generic client-side result processing layered over raw execution."""

from __future__ import annotations

import asyncio
from threading import Event, Lock, Thread

from .base import (FailurePhase, RunState, SubmissionNotStarted,
                   finish_monitor, wait_for_monitor)


class PostprocessingRunner:
    """Convert a raw execution outcome into the caller-visible result.

    The wrapped runner owns submission, execution and artifact availability.
    This layer owns only the ``POSTPROCESSING`` phase and invokes one supplied
    task-agnostic callable after successful raw execution.
    """

    def __init__(self, runner, process_result, callback=None, *, resources=None):
        self._runner = runner
        self._process_result = process_result
        self.callback = callback
        self.result = None
        self._resources = resources
        self._cancel_requested = Event()
        self._cancel_dispatched = False
        self._submission_uncertain = False
        self._submitting = False
        self._preparing = False
        self._cancel_after_prepare = False

        self._state = RunState.NEW
        self._failure_phase = None
        self._failure_message = None
        self._exception = None
        self._callback_exception = None
        self._started = False
        self._processing = False
        self._settled = False
        self._done = Event()
        self._lock = Lock()
        self._thread = None

    @property
    def spec(self):
        """Expose the immutable launch specification before submission."""
        return self._runner.spec

    @property
    def executor(self):
        """Expose the raw executor without leaking task processing into it."""
        return self._runner.executor

    @property
    def backend_id(self):
        return self._runner.backend_id

    @property
    def execution(self):
        return getattr(self._runner, 'execution', None)

    @property
    def files(self):
        return getattr(self._runner, 'files', None)

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
        state = self._runner.state
        if state == RunState.FINISHED and self._started:
            # Raw execution is complete, but the public lifecycle is not.
            return RunState.POSTPROCESSING
        return state

    def add_output_listener(self, callback):
        return self._runner.add_output_listener(callback)

    def add_output_error_listener(self, callback):
        return self._runner.add_output_error_listener(callback)

    def _finish(self):
        try:
            outcome = self._runner.wait()
            raw_state = self._runner.state
            if raw_state == RunState.CANCELLED:
                self.result = outcome
                self._state = RunState.CANCELLED
                return
            if raw_state == RunState.FAILED:
                self.result = outcome
                self._state = RunState.FAILED
                self._failure_phase = (
                    getattr(self._runner, "failure_phase", None)
                    or FailurePhase.EXECUTION
                )
                self._failure_message = getattr(
                    self._runner, "failure_message", None
                )
                return
            if raw_state == RunState.UNKNOWN:
                self.result = outcome
                self._state = RunState.UNKNOWN
                # An unresolved backend may still be executing. It is
                # recoverable and must not finalize as a failed project run.
                self._failure_message = getattr(
                    self._runner, "failure_message", None
                )
                return
            if raw_state != RunState.FINISHED:
                raise RuntimeError(
                    f"Raw runner completed wait() in unexpected state {raw_state.value}"
                )

            self._processing = True
            try:
                self.result = self._process_result(outcome)
            except BaseException as exc:
                self._exception = exc
                self._failure_phase = FailurePhase.POSTPROCESSING
                self._failure_message = str(exc)
                self._state = RunState.FAILED
                return
            finally:
                self._processing = False
            self._state = RunState.FINISHED
        except BaseException as exc:
            self._exception = exc
            self._failure_phase = (
                getattr(self._runner, "failure_phase", None)
                or FailurePhase.EXECUTION
            )
            self._failure_message = (
                getattr(self._runner, "failure_message", None) or str(exc)
            )
            # A failed wait() can itself mean that the backend is unreachable.
            # Never replace the original error by a second status-query error.
            try:
                cancelled = self._runner.state == RunState.CANCELLED
            except Exception:
                cancelled = False
            self._state = RunState.CANCELLED if cancelled else RunState.FAILED
        finally:
            # Make terminal status visible during the persistence callback.
            self._settled = True
            finish_monitor(self, cleanup=self._close_resources)

    def _close_resources(self):
        if self._resources is not None:
            self._resources.close(suppress_errors=True)

    def _start_completion(self):
        self._started = True
        self._thread = Thread(target=self._finish, daemon=True)
        self._thread.start()

    def request_cancel(self):
        """Latch cancellation without touching resources or blocking the submitter.

        ProjectRunner uses this while reserving/staging; there is no backend
        identity yet and the prepared inputs still belong to that operation.
        """
        self._cancel_requested.set()

    def prepare_submission(self):
        """Run preflight without holding the lifecycle lock across I/O."""
        with self._lock:
            if self._cancel_requested.is_set():
                raise asyncio.CancelledError("Run was cancelled before submission")
            if self._preparing or self._submitting or self._started or self._settled or self._submission_uncertain:
                raise RuntimeError(f"Run is already {self.state.value}")
            prepare = getattr(self._runner, "prepare_submission", None)
            if prepare is None:
                return False
            self._preparing = True
        try:
            try:
                prepare()
            except Exception as exc:
                failure = (exc if isinstance(exc, SubmissionNotStarted)
                           else SubmissionNotStarted(str(exc)))
                with self._lock:
                    self._exception = failure
                    self._failure_phase = FailurePhase.SUBMISSION
                    self._failure_message = str(exc)
                    self._state = RunState.FAILED
                    self._settled = True
                self._close_resources()
                if failure is exc:
                    raise
                raise failure from exc
            return True
        finally:
            with self._lock:
                self._preparing = False
                cancel_after = self._cancel_after_prepare
            if cancel_after:
                self.cancel()

    def _submit_raw(self, *, attached=False):
        if attached:
            start = getattr(self._runner, "start", None)
            if start is not None:
                start()
        else:
            self._runner.submit()

    def _start_submission(self, *, attached=False):
        with self._lock:
            if self._submission_uncertain:
                raise RuntimeError("Submission outcome is uncertain; recover the run before retrying")
            if self._cancel_requested.is_set():
                self._state = RunState.CANCELLED
                self._settled = True
                self._done.set()
                cancelled = True
            else:
                cancelled = False
                if self._preparing or self._submitting or self._settled or self._started:
                    raise RuntimeError(f"Run is already {self.state.value}")
                if not attached and self._runner.state != RunState.NEW:
                    raise RuntimeError(f"Run is already {self.state.value}")
                self._submitting = True
        if cancelled:
            # No backend I/O was started. This is a definite pre-submit abort.
            self._close_resources()
            raise asyncio.CancelledError("Run was cancelled before submission")

        try:
            self._submit_raw(attached=attached)
        except SubmissionNotStarted as exc:
            with self._lock:
                self._exception = exc
                self._failure_phase = FailurePhase.SUBMISSION
                self._failure_message = str(exc)
                self._state = RunState.FAILED
                self._settled = True
                self._submitting = False
            self._close_resources()
            raise
        except BaseException:
            # A backend may have accepted the job even if its reply was lost.
            # Never close prepared resources or allow an automatic resubmit.
            with self._lock:
                self._submission_uncertain = True
                self._submitting = False
            raise

        with self._lock:
            self._start_completion()
            self._submitting = False
            dispatch = (self._cancel_requested.is_set()
                        and not self._settled
                        and self._runner.state not in (
                            RunState.FINISHED, RunState.FAILED, RunState.CANCELLED
                        ) and not self._cancel_dispatched)
            if dispatch:
                self._cancel_dispatched = True
        if dispatch:
            return self._dispatch_cancel()
        return None

    def submit(self):
        return self._start_submission()

    def start(self):
        with self._lock:
            if self._started:
                return
            attached = self._runner.state != RunState.NEW
        return self._start_submission(attached=attached)

    def _dispatch_cancel(self):
        try:
            return self._runner.cancel()
        except BaseException:
            with self._lock:
                self._cancel_dispatched = False
            raise

    def wait(self):
        if not self._started:
            raise RuntimeError("Run has not been started")
        return wait_for_monitor(self)

    def close(self):
        """Release an unstarted calculation without launching a backend."""
        with self._lock:
            if self._resources is None or self._settled:
                return
            if (self._preparing or self._submitting or self._started
                    or self._submission_uncertain or self._runner.state != RunState.NEW):
                raise RuntimeError("Cannot close resources of a submitted or uncertain run")
            self._resources.close()
            self._state = RunState.CANCELLED
            self._settled = True
            self._done.set()

    def run(self):
        if self._submission_uncertain:
            raise RuntimeError("Submission outcome is uncertain; recover the run before retrying")
        if not self._started:
            self.start()
        return self.wait()

    async def run_async(self):
        # Submission can block on filesystem or scheduler I/O.
        return await asyncio.to_thread(self.run)

    def stop_observation(self):
        """Stop this client's monitor; leave the backend calculation intact."""
        stop = getattr(self._runner, "stop_observation", None)
        if stop is None:
            raise NotImplementedError(
                "This execution backend cannot detach its observation"
            )
        return stop()

    def cancel(self):
        self.request_cancel()
        with self._lock:
            if self._settled or self._runner.state in (
                RunState.FINISHED, RunState.FAILED, RunState.CANCELLED
            ):
                return None
            if self._preparing:
                self._cancel_after_prepare = True
                return None
            if self._submitting:
                # Submission owns the hand-off. It will dispatch cancellation
                # once the backend accepts the job and exposes its identity.
                return None
            if (not self._started and not self._submission_uncertain
                    and self._runner.state == RunState.NEW):
                self._state = RunState.CANCELLED
                self._settled = True
                self._done.set()
                release = True
            else:
                release = False
                if self._submission_uncertain and self.backend_id is None:
                    # Outcome may be recoverable, but no handle can be cancelled.
                    return None
                if self._cancel_dispatched:
                    return None
                self._cancel_dispatched = True
        if release:
            self._close_resources()
            return None
        return self._dispatch_cancel()
