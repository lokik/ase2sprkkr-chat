"""Generic client-side result processing layered over raw execution."""

from __future__ import annotations

import asyncio
from threading import Event, Lock

from .base import (FailurePhase, RunState, SubmissionNotStarted,
                   finish_monitor)
from .run_hooks import PersistenceError


class CalculationRun:
    """Convert a raw execution outcome into the caller-visible result.

    The wrapped runner owns submission, execution and file availability.
    This layer owns only the ``POSTPROCESSING`` phase and invokes one supplied
    task-agnostic callable after successful raw execution.
    """

    def __init__(self, runner, process_result, callback=None, *, resources=None,
                 hooks=None):
        self._runner = runner
        self._process_result = process_result
        self.callback = callback
        self.hooks = hooks
        self.persistence_error = None
        self.result = None
        self._resources = resources
        self._cancel_requested = Event()
        self._cancel_dispatched = False
        self._submission_uncertain = False
        self._backend_submit_attempted = False
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
        self._completion_lock = Lock()

        # The existing backend monitor owns observation and invokes this
        # callback only after file delivery and stream finalization.
        if getattr(runner, "callback", None) is not None:
            raise ValueError("Raw runner already has a completion callback")
        runner.callback = self._backend_completed

    @property
    def id(self):
        if self.hooks is None:
            raise AttributeError("Untracked calculation has no persistent ID")
        return self.hooks.id

    @property
    def record(self):
        if self.hooks is None:
            raise AttributeError("Untracked calculation has no RunRecord")
        return self.hooks.record

    def refresh(self):
        if self.hooks is None:
            raise RuntimeError("Untracked calculation has no persistent record")
        return self.hooks.refresh()

    @property
    def cancellation_requested(self):
        """Single source of truth for cancellation across project and processing."""
        return self._cancel_requested.is_set()

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
        if state == RunState.FINISHED and (self._started or self._submitting):
            # Raw execution is complete, but the public lifecycle is not.
            return RunState.POSTPROCESSING
        return state

    def set_manifest(self, manifest):
        setter = getattr(self._runner, "set_manifest", None)
        if setter is not None:
            setter(manifest)

    def add_output_listener(self, callback):
        return self._runner.add_output_listener(callback)

    def add_output_error_listener(self, callback):
        return self._runner.add_output_error_listener(callback)

    def _backend_completed(self, outcome):
        """Called on the backend monitor thread, before its waiters wake up."""
        self._complete_observation(outcome)

    def _complete_observation(self, outcome, error=None):
        """Publish one observation, including durable callback and cleanup.

        The backend may notify before submit() returns, and a caller's wait()
        can race the callback. Neither parsing nor persistence may run twice.
        """
        with self._completion_lock:
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
                    # An observation ended, not the job. Project persistence
                    # must not create a terminal timestamp or acknowledge it.
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
                        "Backend observation ended in unexpected state "
                        f"{raw_state.value}"
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
                # Persist before releasing resources or waking public waiters.
                # A callback failure must not alter a completed backend state.
                self._settled = True
                self._publish_completion()

    def _publish_completion(self):
        """Persistence must succeed before completion is published to observers."""
        try:
            if self.hooks is not None:
                persisted_state = self.hooks.completed(self)
                if persisted_state is not None:
                    self._state = RunState(persisted_state)
        except BaseException as exc:
            self.persistence_error = (
                exc if isinstance(exc, PersistenceError)
                else PersistenceError("Could not persist calculation completion")
            )
            if self.persistence_error is not exc:
                self.persistence_error.__cause__ = exc
            # A completed backend is never reclassified as a failed calculation.
            # No completion callback or backend acknowledgment is emitted here.
            try:
                self._close_resources()
            finally:
                self._done.set()
        else:
            finish_monitor(self, cleanup=self._close_resources)

    def retry_persistence(self):
        """Retry the durable completion without executing or parsing the job again."""
        with self._completion_lock:
            if not self._settled:
                raise RuntimeError("Calculation has not completed")
            if self.persistence_error is None:
                return
            if self.hooks is None:
                raise RuntimeError("No persistence hook is configured")
            try:
                persisted_state = self.hooks.completed(self)
                if persisted_state is not None:
                    self._state = RunState(persisted_state)
            except BaseException as exc:
                self.persistence_error = (
                    exc if isinstance(exc, PersistenceError)
                    else PersistenceError("Could not persist calculation completion")
                )
                if self.persistence_error is not exc:
                    self.persistence_error.__cause__ = exc
                raise self.persistence_error
            self.persistence_error = None
            if self.callback is not None:
                try:
                    self.callback(self.result)
                except BaseException as exc:
                    self._callback_exception = exc

    def _close_resources(self):
        if self._resources is not None:
            self._resources.close(suppress_errors=True)

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
            return
        if self.hooks is None:
            self._runner.submit()
            return

        # All persistent transaction barriers are sequenced by this one
        # lifecycle owner. The project layer never submits the backend itself.
        backend_attempted = False
        self._backend_submit_attempted = False
        try:
            self.hooks.before_submit(self)
            preflight = getattr(self._runner, "prepare_submission", None)
            if preflight is not None:
                try:
                    preflight()
                except Exception as exc:
                    if isinstance(exc, SubmissionNotStarted):
                        raise
                    raise SubmissionNotStarted(str(exc)) from exc
                self.hooks.prepared(self)
            if self.cancellation_requested:
                raise asyncio.CancelledError("Run was cancelled before backend submission")
            backend_attempted = True
            self._backend_submit_attempted = True
            self._runner.submit()
            self.hooks.submitted(self)
        except BaseException as exc:
            try:
                self.hooks.submission_failed(
                    self, exc, uncertain=backend_attempted,
                )
            except BaseException as hook_exc:
                # A persistence failure must not erase the original
                # submission error, especially after an uncertain launch.
                if hasattr(exc, "add_note"):
                    exc.add_note(f"Submission persistence also failed: {hook_exc!r}")
            raise

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
        except BaseException as exc:
            # Only backend I/O can make submission ambiguous. Reservation,
            # staging and pre-submit cancellation are definitely non-launches.
            uncertain = (self.hooks is None or self._backend_submit_attempted)
            with self._lock:
                self._submission_uncertain = uncertain
                self._submitting = False
                if not uncertain:
                    self._state = (RunState.CANCELLED
                                   if isinstance(exc, asyncio.CancelledError)
                                   else RunState.FAILED)
                    self._exception = None if isinstance(exc, asyncio.CancelledError) else exc
                    self._failure_phase = (
                        None if isinstance(exc, asyncio.CancelledError)
                        else FailurePhase.SUBMISSION
                    )
                    self._failure_message = (
                        None if isinstance(exc, asyncio.CancelledError)
                        else str(exc)
                    )
                    self._settled = True
                    self._done.set()
            if not uncertain:
                self._close_resources()
            raise

        with self._lock:
            self._started = True
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

    def start(self, *, attached=None):
        with self._lock:
            if self._started:
                return
            if attached is None:
                attached = self._runner.state != RunState.NEW
        result = self._start_submission(attached=attached)
        if attached and self.hooks is not None:
            self.hooks.attached(self)
        return result

    def _dispatch_cancel(self):
        try:
            return self._runner.cancel()
        except BaseException:
            with self._lock:
                self._cancel_dispatched = False
            raise

    def wait(self):
        if not self._started and not self._settled:
            raise RuntimeError("Run has not been started")
        # Production runners notify before raw.wait() returns. A lightweight
        # external runner may only report its terminal state from wait().
        # Preserve a callback's canonical completion if raw.wait() later fails.
        try:
            outcome = self._runner.wait()
        except BaseException as exc:
            if not self._done.is_set():
                self._complete_observation(
                    getattr(self._runner, "result", None), error=exc,
                )
        else:
            if not self._done.is_set():
                self._complete_observation(outcome)
        self._done.wait()
        if self.persistence_error is not None:
            raise self.persistence_error
        if self._exception is not None:
            raise self._exception
        if self._callback_exception is not None:
            raise self._callback_exception
        return self.result

    def close(self):
        """Release an unstarted calculation without launching a backend."""
        with self._lock:
            if self._settled:
                return
            if (self._preparing or self._submitting or self._started
                    or self._submission_uncertain or self._runner.state != RunState.NEW):
                raise RuntimeError("Cannot close resources of a submitted or uncertain run")
            if self._resources is not None:
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
        if self.hooks is not None and not self.hooks.can_cancel(self):
            return None
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


# Keep the published import while consolidating the lifecycle.
PostprocessingRunner = CalculationRun
