"""Generic client-side result processing layered over raw execution."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Callable, Protocol

from ..common.file_versions import FileState
from .base import (ExecutionOutcome, FailurePhase, RunState,
                   SubmissionNotStarted, finish_monitor)
from .file_history import file_is_current
from .run_hooks import PersistenceError, RunPersistenceHooks


class ProcessedResult(Protocol):
    """Result-processing output exposes the process return code explicitly."""

    return_code: int | None


class CalculationRun:
    """Convert a raw execution outcome into the caller-visible result.

    The wrapped runner owns submission, execution and file availability.
    This layer owns only the ``POSTPROCESSING`` phase and invokes one supplied
    task-agnostic callable after successful raw execution.
    """

    def __init__(
        self, runner, process_result: Callable[[ExecutionOutcome], ProcessedResult],
        callback=None, *, resources=None,
                 persistence: RunPersistenceHooks | None = None):
        self._runner = runner
        self.job_files = runner.job_files
        self._process_result = process_result
        self.callback = callback
        self.persistence = persistence
        self.persistence_error = None
        self.result: ProcessedResult | None = None
        self._resources = resources
        self._cancel_requested = Event()
        self._cancel_dispatched = False
        self._submission_uncertain = False
        self._backend_submit_attempted = False
        self._submitting = False
        self._completion_deferred = False

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
        if runner.callback is not None:
            raise ValueError("Raw runner already has a completion callback")
        runner.callback = self._backend_completed

    @property
    def id(self):
        if self.persistence is None:
            raise AttributeError("Untracked calculation has no persistent ID")
        return self.persistence.id

    @property
    def record(self):
        if self.persistence is None:
            raise AttributeError("Untracked calculation has no RunRecord")
        return self.persistence.record

    def refresh(self):
        if self.persistence is None:
            raise RuntimeError("Untracked calculation has no persistent record")
        return self.persistence.refresh()

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
    def exception(self):
        return self._exception or self._runner.exception

    @property
    def failure_phase(self):
        return self._failure_phase or self._runner.failure_phase

    @property
    def failure_message(self):
        return self._failure_message or self._runner.failure_message

    @property
    def return_code(self):
        # Processing may return any object (including bytes), so the
        # backend's execution status is the sole source for its exit code.
        return self._runner.return_code

    @property
    def state(self):
        if self._settled:
            return self._state
        # Reattached persistent runs may already be terminal, even if their
        # backend is gone. Do not query a stale or released backend.
        record = self.persistence.record if self.persistence is not None else None
        if record is not None and record.state.is_terminal:
            return record.state
        if self._processing:
            return RunState.POSTPROCESSING
        state = self._runner.state
        if state == RunState.FINISHED and (self._started or self._submitting):
            # Raw execution is complete, but the public lifecycle is not.
            return RunState.POSTPROCESSING
        return state

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
                        self._runner.failure_phase
                        or FailurePhase.EXECUTION
                    )
                    self._failure_message = self._runner.failure_message
                    self._exception = self._runner.exception
                elif raw_state == RunState.UNKNOWN:
                    # An observation ended, not the job. Project persistence
                    # must not create a terminal timestamp or acknowledge it.
                    self._state = RunState.UNKNOWN
                    self._failure_message = self._runner.failure_message
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
                    self._runner.failure_phase
                    or FailurePhase.EXECUTION
                )
                self._failure_message = (
                    self._runner.failure_message or str(exc)
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
                # A backend may invoke its callback synchronously inside
                # submit(). Its identity must be durably stored by submitted()
                # before the completion hook can write a terminal record.
                with self._lock:
                    if (self._submitting and self._backend_submit_attempted
                            and self.persistence is not None):
                        self._completion_deferred = True
                        publish = False
                    else:
                        publish = True
                if publish:
                    self._publish_completion()

    def _publish_completion(self):
        """Persistence must succeed before completion is published to observers."""
        try:
            if self.persistence is not None:
                persisted_state = self.persistence.completed(self)
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
            if self.persistence is None:
                raise RuntimeError("No persistence hook is configured")
            try:
                persisted_state = self.persistence.completed(self)
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

        Cancellation may arrive during reservation/staging, before a backend
        identity exists. The submitting operation still owns the inputs.
        """
        self._cancel_requested.set()

    def _submit_raw(self, *, attached=False):
        if attached:
            self._runner.start()
            return

        # There is one preflight, submission and error boundary for both
        # project-backed and untracked calculations. A project additionally
        # reserves the run before staging and persists the staged manifest
        # before any backend can start.
        self._backend_submit_attempted = False
        try:
            if self.persistence is not None:
                self.persistence.before_submit(self)

            try:
                self._runner.prepare_submission()
            except Exception as exc:
                if isinstance(exc, SubmissionNotStarted):
                    raise
                raise SubmissionNotStarted(str(exc)) from exc

            if self.persistence is not None:
                self.persistence.prepared(self)
            if self.cancellation_requested:
                raise asyncio.CancelledError(
                    "Run was cancelled before backend submission"
                )

            self._backend_submit_attempted = True
            self._runner.submit()
            if self.persistence is not None:
                self.persistence.submitted(self)
        except BaseException as exc:
            if self.persistence is not None:
                try:
                    self.persistence.submission_failed(
                        self, exc, uncertain=(
                            self._backend_submit_attempted
                            and not isinstance(exc, SubmissionNotStarted)
                        ),
                    )
                except BaseException as hook_exc:
                    # Do not hide the original submission failure, especially
                    # when the backend may already have accepted the job.
                    if hasattr(exc, "add_note"):
                        exc.add_note(
                            f"Submission persistence also failed: {hook_exc!r}"
                        )
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
                if self._submitting or self._settled or self._started:
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
            try:
                self._close_resources()
            finally:
                # No backend exists to signal completion; wake all waiters.
                self._done.set()
            raise
        except BaseException as exc:
            # Only backend I/O can make submission ambiguous. Reservation,
            # staging and pre-submit cancellation are definitely non-launches.
            uncertain = self._backend_submit_attempted
            with self._lock:
                self._submission_uncertain = uncertain
                self._submitting = False
                deferred = self._completion_deferred
                self._completion_deferred = False
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
            if deferred:
                # The backend finished, but the submission transaction did
                # not complete. Do not publish a terminal record without
                # first recovering its durable backend identity.
                failure = PersistenceError(
                    "Backend finished before submission identity was persisted"
                )
                failure.__cause__ = exc
                self.persistence_error = failure
                try:
                    self._close_resources()
                finally:
                    self._done.set()
            elif not uncertain:
                self._close_resources()
            raise

        with self._lock:
            self._started = True
            self._submitting = False
            deferred = self._completion_deferred
            self._completion_deferred = False
            dispatch = (self._cancel_requested.is_set()
                        and not self._settled
                        and self._runner.state not in (
                            RunState.FINISHED, RunState.FAILED, RunState.CANCELLED
                        ) and not self._cancel_dispatched)
            if dispatch:
                self._cancel_dispatched = True
        if deferred:
            self._publish_completion()
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
        if attached and self.persistence is not None:
            self.persistence.attached(self)
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
        # A definitely rejected or closed run never started a backend; its
        # terminal exception/state is already available locally. An uncertain
        # submission, however, may still have a live backend to observe.
        if self._started or self._submission_uncertain:
            # Production runners notify before raw.wait() returns. A lightweight
            # external runner may only report its terminal state from wait().
            # Preserve a callback's canonical completion if raw.wait() fails.
            try:
                outcome = self._runner.wait()
            except BaseException as exc:
                if not self._done.is_set():
                    self._complete_observation(
                        self._runner.result, error=exc,
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
            if (self._submitting or self._started
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
        return self._runner.stop_observation()

    def cancel(self):
        if self.persistence is not None and not self.persistence.can_cancel(self):
            return None
        self.request_cancel()
        with self._lock:
            if self._settled or self._runner.state in (
                RunState.FINISHED, RunState.FAILED, RunState.CANCELLED
            ):
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




def claim_historical_postprocessing(latest):
    """Atomically claim a retriable historical parse, validating its snapshot."""
    files = latest.files
    if (latest.state != RunState.FAILED
            or latest.failure_phase != FailurePhase.POSTPROCESSING
            or latest.return_code != 0
            or latest.execution.backend_id is None):
        raise ValueError("Only postprocessing failures of successful backend runs can be retried")
    if files.captured_at is None or files.output_file is None:
        raise ValueError("Postprocessing retry requires captured output files")
    if files.failed_files:
        raise ValueError("Postprocessing retry requires locally available files")
    if (not files.output_versions
            or any(version.state() is not FileState.CURRENT
                   for version in files.output_versions)
            or not file_is_current(files, files.output_file, files.output_versions)):
        raise ValueError("Postprocessing retry requires unchanged historical files")
    return replace(latest, state=RunState.POSTPROCESSING, finished_at=None,
                   failure_phase=None, message=None,
                   updated_at=datetime.now(timezone.utc).isoformat())


def parse_historical_result(reopened, process_result):
    """Parse captured output without a backend, checking for concurrent rewrites."""
    files = reopened.files
    outcome = ExecutionOutcome(
        input_file=(files.input_file if file_is_current(
            files, files.input_file, files.input_versions) else None),
        output_file=files.output_file,
        error_file=None if files.stderr is None else files.stderr.local_path,
        return_code=reopened.return_code,
    )
    result = process_result(outcome)
    if any(version.state() is not FileState.CURRENT
           for version in files.output_versions):
        raise RuntimeError("Historical files changed during postprocessing retry")
    return result
