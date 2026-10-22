"""Persistence hooks joining CalculationRun to durable lifecycle orchestration."""

from __future__ import annotations

from threading import Lock
from contextlib import contextmanager

from .base import RunState
from .run_model import FileManifest
from .postprocessing import CalculationRun


class RunPersistence:
    """Adapter from one CalculationRun's events to durable manager operations.

    The manager owns lifecycle policy and RunStore owns atomic persistence.
    This adapter serializes changes to its local RunRecord snapshot. Backend
    acknowledgements run separately so slow transport cannot block refresh.
    """

    def __init__(self, manager, executor, record, *, spec=None):
        self._spec = spec
        self.manager = manager
        self.executor = executor
        self._record = record
        self._record_lock = Lock()
        self._ack_lock = Lock()
        self._terminal_acknowledged = False
        # CAS baseline for the running monitor, not the most recent refresh.
        # Refresh must not allow an older UNKNOWN observation to overwrite
        # fresher scheduler state.
        self._observation_base_version = record.updated_at

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
        with self.manager.store.submission_owner(self._record) as acquired:
            if not acquired:
                raise RuntimeError(f"Submission {self._record.id} already has an owner")
            yield

    def dispatching(self):
        with self._record_lock:
            self._record = self.manager.store.begin_dispatch(self._record)
            self._observation_base_version = self._record.updated_at

    def before_submit(self, run: CalculationRun) -> FileManifest:
        """Reserve atomically and return the committed layout for runner staging."""
        with self._record_lock:
            if self._record.state != RunState.NEW:
                raise RuntimeError(f"Run is already {self._record.state.value}")
            self._record = self.manager.reserve(
                self._record, spec=self._spec, executor=self.executor,
            )
            self._observation_base_version = self._record.updated_at
            return self._record.files

    def prepared(self, run: CalculationRun):
        """Commit the staged layout before the backend can launch."""
        with self._record_lock:
            self._record = self.manager.transition(
                self._record,
                execution=self._execution_for(run, self._record),
                files=run.job_files.manifest,
            )
            self._observation_base_version = self._record.updated_at

    def submitted(self, run: CalculationRun):
        """Persist backend identity before acknowledging its temporary ticket."""
        with self._record_lock:
            if not self._record.state.is_terminal:
                self._record = self.manager.record_submitted(
                    self._record, backend_id=run.backend_id, state=run.state,
                    failure_phase=run.failure_phase, files=run.job_files.manifest,
                )
                self._observation_base_version = self._record.updated_at
            record = self._record
        self.manager._acknowledge_submission(self.executor, record)

    def submission_failed(
        self, run: CalculationRun, error: BaseException, *, uncertain: bool,
    ):
        """Record definite non-launches; preserve unknown backend acceptance."""
        if uncertain and run.backend_id is None:
            return
        with self._record_lock:
            self._record = self.manager.record_submission_failure(
                self._record, error=error, uncertain=uncertain,
                state=run.state, backend_id=run.backend_id,
                failure_phase=run.failure_phase, message=run.failure_message,
            )

    def attached(self, run: CalculationRun):
        """Sync nonterminal attached state; completion has its own transaction."""
        state = run.state
        if run.backend_id is None or state == RunState.NEW or state.is_terminal:
            return
        with self._record_lock:
            if self._record.state.is_terminal:
                return
            self._record = self.manager.transition(
                self._record, state=state,
                execution=self._execution_for(run, self._record),
                failure_phase=run.failure_phase,
                expected_updated_at=self._record.updated_at,
            )
            self._observation_base_version = self._record.updated_at

    def can_cancel(self, run: CalculationRun):
        """A confirmed terminal run cannot be cancelled; never wait for I/O.

        Each RunRecord is immutable. Reading its published reference is safe
        while reservation or staging holds the persistence transaction lock.
        Cancellation is latched by CalculationRun and reconciled after submit.
        """
        return not self._record.state.is_terminal

    def _acknowledge_terminal(self, record):
        # A separate lock avoids blocking record reads or refresh on remote I/O,
        # while ensuring concurrent completion observers only ACK once.
        with self._ack_lock:
            if not self._terminal_acknowledged:
                self._terminal_acknowledged = self.manager.acknowledge_terminal(
                    self.executor, record,
                )

    def completed(self, run: CalculationRun):
        """Persist one observation and ACK only a durable terminal snapshot."""
        with self._record_lock:
            if not self._record.state.is_terminal:
                message = run.failure_message
                if message is None and run.exception is not None:
                    message = str(run.exception)
                updated = self.manager.record_observation(
                    self._record, state=run.state,
                    execution=self._execution_for(run, self._record),
                    return_code=run.return_code, message=message,
                    failure_phase=run.failure_phase,
                    files=run.job_files.manifest,
                    expected_updated_at=self._observation_base_version,
                )
                if updated is None:
                    return
                self._record = updated
            record = self._record
        if record.state.is_terminal:
            self._acknowledge_terminal(record)
        return record.state

    def refresh(self):
        with self._record_lock:
            self._record = self.manager.refresh((self._record,))[0]
            # Do not advance _observation_base_version here: scheduler status
            # is not a new observation from the still-running file monitor.
            return self._record
