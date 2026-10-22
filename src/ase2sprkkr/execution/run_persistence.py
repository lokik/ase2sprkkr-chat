"""Persistence hooks joining CalculationRun to durable lifecycle orchestration."""

from __future__ import annotations

import asyncio
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from threading import Lock

from .base import FailurePhase, RunState
from .postprocessing import CalculationRun
from .run_record import _failure_phase


_NO_FILES_FAILURES = {
    FailurePhase.SUBMISSION,
    FailurePhase.LAUNCH,
    FailurePhase.FILES,
    FailurePhase.DOWNLOAD,
}

class RunPersistence:
    """Durable transaction participant for one CalculationRun.

    CalculationRun sequences execution and public completion. This object
    serializes durable commits. The completion lock protects refresh and
    finalization; submission uniqueness is owned by CalculationRun and the
    atomically reserving RunStore, not by a second lifecycle state machine.
    """

    def __init__(self, manager, executor, record, *, spec=None):
        self._spec = spec
        self.manager = manager
        self.executor = executor
        self._record = record
        self._completion_lock = Lock()
        # Network ACK is best-effort and intentionally separate from the
        # durable terminal state: a new instance retries it after a crash.
        self._terminal_acknowledged = False
        self._monitor_record_version = record.updated_at

    @property
    def id(self):
        return self._record.id

    @property
    def record(self):
        return self._record

    @staticmethod
    def _execution_for(run, record):
        """Preserve reserved placement; only backend identity comes from the runner."""
        if run.backend_id is None:
            return None
        return record.execution.submitted(run.backend_id)

    @contextmanager
    def submission_owner(self):
        """Hold the submission owner only on the actual submit call stack."""
        with self.manager.store.submission_owner(self._record) as acquired:
            if not acquired:
                raise RuntimeError(f"Submission {self._record.id} already has an owner")
            yield

    def before_submit(self, run: CalculationRun):
        """Reserve atomically before any backend file staging."""
        if self._record.state != RunState.NEW:
            raise RuntimeError(f"Run is already {self._record.state.value}")
        if self._spec is not None:
            files, remote_directory = self.executor.reservation_context(
                self._spec, self._record.files,
            )
            execution = self._record.execution
            if remote_directory is not None:
                execution = replace(
                    execution,
                    directory=replace(execution.directory, remote=remote_directory),
                )
            self._record = replace(self._record, files=files, execution=execution)
        self._record = self.manager.reserve(self._record)
        run.job_files.replace(self._record.files)
        self._monitor_record_version = self._record.updated_at

    def prepared(self, run: CalculationRun):
        """Commit the exact staged layout before the backend is launched."""
        self._record = self.manager.transition(
            self._record,
            execution=self._execution_for(run, self._record),
            files=run.job_files.manifest,
        )
        self._monitor_record_version = self._record.updated_at

    def dispatching(self):
        """Persist the uncertain-launch boundary before contacting the backend."""
        self._record = self.manager.store.begin_dispatch(self._record)
        self._monitor_record_version = self._record.updated_at

    def submitted(self, run: CalculationRun):
        """Persist backend identity before clearing its temporary ticket."""
        if run.backend_id is None:
            raise RuntimeError("Executor submit returned without a backend ID")
        with self._completion_lock:
            if not self._record.state.is_terminal:
                observed = run.state
                if observed.is_terminal:
                    # The backend may finish synchronously in submit().
                    # Its completion callback waits for durable identity.
                    observed = RunState.SUBMITTING
                self._record = self.manager.transition(
                    self._record, state=observed,
                    execution=self._execution_for(run, self._record),
                    failure_phase=(None if observed == RunState.SUBMITTING
                                   else _failure_phase(run.failure_phase)),
                    files=run.job_files.manifest,
                )
                self._monitor_record_version = self._record.updated_at
        self.manager._acknowledge_submission(self.executor, self._record)

    def submission_failed(
        self, run: CalculationRun, error: BaseException, *, uncertain: bool,
    ):
        """Record definite non-launches and preserve uncertain submissions."""
        if not uncertain:
            if self._record.state == RunState.NEW:
                return  # Reservation was refused; nothing was launched.
            if isinstance(error, asyncio.CancelledError):
                self._record = self.manager.complete(
                    self._record, state=RunState.CANCELLED,
                    files_available=False,
                )
            else:
                phase = _failure_phase(run.failure_phase) or FailurePhase.SUBMISSION
                message = run.failure_message or str(error)
                self._record = self.manager.complete(
                    self._record, state=RunState.FAILED, message=message,
                    failure_phase=phase, files_available=False,
                )
        elif run.backend_id is not None:
            with self._completion_lock:
                if not self._record.state.is_terminal:
                    self._record = self.manager.transition(
                        self._record, state=run.state,
                        execution=self._execution_for(run, self._record),
                        failure_phase=_failure_phase(run.failure_phase),
                    )

    def attached(self, run: CalculationRun):
        """Synchronize an attached observation without resubmitting its job."""
        with self._completion_lock:
            state = run.state
            if (not self._record.state.is_terminal and run.backend_id is not None
                    and state != RunState.NEW):
                self._record = self.manager.transition(
                    self._record, state=state,
                    execution=self._execution_for(run, self._record), failure_phase=_failure_phase(run.failure_phase),
                )
                self._monitor_record_version = self._record.updated_at

    def can_cancel(self, run: CalculationRun):
        """A durable terminal record cannot be cancelled by a stale observer."""
        return not self._record.state.is_terminal

    def _acknowledge_terminal(self, run):
        """ACK only a durable terminal result; retry after a lost ACK reply."""
        if (self._terminal_acknowledged or not self._record.state.is_terminal
                or (self._record.execution.backend_id is None
                    and run.backend_id is None)):
            return
        try:
            self.executor.acknowledge(self._record)
        except (OSError, RuntimeError, subprocess.CalledProcessError):
            # Backend receipts remain safely repeatable after this failure.
            return
        self._terminal_acknowledged = True

    def completed(self, run: CalculationRun):
        """Persist one observation, taking idempotence from the durable record."""
        with self._completion_lock:
            if self._record.state.is_terminal:
                self._acknowledge_terminal(run)
                return self._record.state
            state = run.state
            phase = _failure_phase(run.failure_phase)
            message = run.failure_message
            error = run.exception
            if message is None and error is not None:
                message = str(error)
            if state == RunState.UNKNOWN:
                # Repeated UNKNOWN is not a new backend observation. A fresh
                # reattachment has its own RunPersistence and can recover it.
                if self._record.state == RunState.UNKNOWN:
                    return self._record.state
                # A stale observer cannot overwrite a newer refresh.
                self._record = self.manager.transition(
                    self._record, state=state, execution=self._execution_for(run, self._record),
                    message=message,
                    expected_updated_at=self._monitor_record_version,
                )
                return self._record.state
            if not state.is_terminal:
                return
            if state == RunState.FAILED and phase is None:
                phase = FailurePhase.EXECUTION

            files_available = phase not in _NO_FILES_FAILURES
            self._record = self.manager.complete(
                self._record,
                state=state,
                execution=self._execution_for(run, self._record),
                return_code=run.return_code,
                message=message,
                failure_phase=phase,
                files_available=files_available,
                files=run.job_files.manifest,
            )
            # ACK follows the crash-durable transition. A new RunPersistence
            # retries a lost ACK, without repeating terminal persistence.
            self._acknowledge_terminal(run)
            return self._record.state

    def refresh(self):
        with self._completion_lock:
            self._record = self.manager.refresh((self._record,))[0]
            return self._record
