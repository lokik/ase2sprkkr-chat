"""Persistence hooks joining CalculationRun to durable lifecycle orchestration."""

from __future__ import annotations

from threading import Lock

from .base import RunState
from .postprocessing import CalculationRun
from .run_record import _failure_phase


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

    def before_submit(self, run: CalculationRun):
        """Reserve atomically before any backend file staging."""
        if self._record.state != RunState.NEW:
            raise RuntimeError(f"Run is already {self._record.state.value}")
        self._record = self.manager.reserve(
            self._record, spec=self._spec, executor=self.executor,
        )
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

    def submitted(self, run: CalculationRun):
        """Persist backend identity before clearing its temporary ticket."""
        with self._completion_lock:
            if not self._record.state.is_terminal:
                self._record = self.manager.record_submitted(
                    self._record, backend_id=run.backend_id, state=run.state,
                    failure_phase=run.failure_phase, files=run.job_files.manifest,
                )
                self._monitor_record_version = self._record.updated_at
        self.manager._acknowledge_submission(self.executor, self._record)

    def submission_failed(
        self, run: CalculationRun, error: BaseException, *, uncertain: bool,
    ):
        """Record definite non-launches and preserve uncertain submissions."""
        if uncertain and run.backend_id is not None:
            with self._completion_lock:
                self._record = self.manager.record_submission_failure(
                    self._record, error=error, uncertain=True,
                    state=run.state, backend_id=run.backend_id,
                    failure_phase=run.failure_phase,
                    message=run.failure_message,
                )
        elif not uncertain:
            self._record = self.manager.record_submission_failure(
                self._record, error=error, uncertain=False,
                state=run.state, backend_id=run.backend_id,
                failure_phase=run.failure_phase,
                message=run.failure_message,
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
        if not self._terminal_acknowledged:
            self._terminal_acknowledged = self.manager.acknowledge_terminal(
                self.executor, self._record,
            )

    def completed(self, run: CalculationRun):
        """Persist one observation, taking idempotence from the durable record."""
        with self._completion_lock:
            if self._record.state.is_terminal:
                self._acknowledge_terminal(run)
                return self._record.state
            message = run.failure_message
            if message is None and run.exception is not None:
                message = str(run.exception)
            updated = self.manager.record_observation(
                self._record, state=run.state,
                execution=self._execution_for(run, self._record),
                return_code=run.return_code, message=message,
                failure_phase=run.failure_phase,
                files=run.job_files.manifest,
                expected_updated_at=self._monitor_record_version,
            )
            if updated is None:
                return
            self._record = updated
            if updated.state.is_terminal:
                # The backend receipt may be retried by a new observer.
                self._acknowledge_terminal(run)
            return self._record.state

    def refresh(self):
        with self._completion_lock:
            self._record = self.manager.refresh((self._record,))[0]
            return self._record
