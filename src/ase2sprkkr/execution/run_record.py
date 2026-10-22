"""Immutable durable run identity and validated state transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional

from .base import FailurePhase, RunState
from .job_files import JobFiles
from .run_model import Execution, FileManifest

_RUN_RECORD_SCHEMA_VERSION = 3
_STATE_RANK = {
    RunState.NEW: 0,
    RunState.SUBMITTING: 1,
    RunState.QUEUED: 2,
    RunState.RUNNING: 3,
    RunState.DOWNLOADING: 4,
    RunState.POSTPROCESSING: 5,
    RunState.FINISHED: 6,
    RunState.FAILED: 6,
    RunState.CANCELLED: 6,
}
_UNCHANGED = object()

def _now():
    return datetime.now(timezone.utc).isoformat()


def _failure_phase(value):
    if value is None or isinstance(value, FailurePhase):
        return value
    return FailurePhase(value)


@dataclass(frozen=True)
class RunRecord:
    """Single durable identity, file manifest and lifecycle of one calculation."""

    id: str
    state: RunState
    created_at: str
    updated_at: str
    task: str
    execution: Execution
    files: FileManifest
    finished_at: Optional[str] = None
    dataset: Optional[str] = None
    return_code: Optional[int] = None
    message: Optional[str] = None
    input_definition: Optional[str] = None
    failure_phase: Optional[FailurePhase] = None
    schema_version: int = _RUN_RECORD_SCHEMA_VERSION
    project_id: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "task", str(self.task).upper())
        if self.dataset is not None:
            object.__setattr__(self, "dataset", str(self.dataset))
        if self.project_id is not None:
            object.__setattr__(self, "project_id", str(self.project_id))
        if self.input_definition is not None:
            object.__setattr__(self, "input_definition", str(self.input_definition).upper())
        if isinstance(self.execution, dict):
            object.__setattr__(self, "execution", Execution(**self.execution))
        if isinstance(self.files, dict):
            object.__setattr__(self, "files", FileManifest(**self.files))
        if not isinstance(self.state, RunState):
            object.__setattr__(self, "state", RunState(self.state))
        if self.failure_phase is not None and not isinstance(self.failure_phase, FailurePhase):
            object.__setattr__(self, "failure_phase", FailurePhase(self.failure_phase))
        if self.schema_version != _RUN_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {self.schema_version}; "
                f"expected {_RUN_RECORD_SCHEMA_VERSION}"
            )
    def _merged_state(self, requested):
        if requested is None:
            return self.state
        requested = RunState(requested)
        if self.state.is_terminal:
            return self.state
        if requested == RunState.UNKNOWN:
            # Uncertainty cannot retract confirmed execution completion.
            if self.state in (RunState.POSTPROCESSING, RunState.DOWNLOADING):
                return self.state
            return requested
        if self.state == RunState.UNKNOWN:
            return requested
        if _STATE_RANK[requested] < _STATE_RANK[self.state]:
            return self.state
        return requested

    def with_transition(
        self,
        *,
        state=None,
        execution=_UNCHANGED,
        return_code=_UNCHANGED,
        message=_UNCHANGED,
        failure_phase=_UNCHANGED,
        files=_UNCHANGED,
        expected_updated_at=_UNCHANGED,
        expected_record=None,
        skip_if_terminal=False,
        validate_completed_files=False,
    ):
        """Return a validated immutable snapshot of a run-state observation.

        No disk I/O or backend calls happen here; RunStore owns atomicity.
        """
        if expected_record is not None and self != expected_record:
            raise RuntimeError("Run changed during postprocessing retry")
        if skip_if_terminal and self.state.is_terminal:
            return self
        if (expected_updated_at is not _UNCHANGED
                and self.updated_at != expected_updated_at):
            # An independent refresh has superseded this observation.
            return self
        final_locked = self.state.is_terminal
        now = _now()
        merged_state = self._merged_state(state)
        if (state is not None and RunState(state) == RunState.UNKNOWN
                and merged_state != RunState.UNKNOWN):
            # Ignore the entire stale observation, including its message
            # and file metadata, not merely its proposed state.
            return self
        changes = {"state": merged_state, "updated_at": now}
        if not self.state.is_terminal and merged_state.is_terminal:
            changes["finished_at"] = now
        if not final_locked and files is not _UNCHANGED and files is not None:
            merged_files = JobFiles.merge_snapshot(self.files, files)
            if merged_files is not self.files:
                changes["files"] = merged_files
        if execution is not _UNCHANGED and execution is not None:
            if execution.endpoint != self.execution.endpoint:
                raise ValueError("Execution endpoint cannot change")
            if execution.directory.local != self.execution.directory.local:
                raise ValueError("Execution local directory cannot change")
            if self.execution.backend_id is not None:
                if execution.backend_id != self.execution.backend_id:
                    raise ValueError("Backend identity cannot change")
                if execution.directory != self.execution.directory:
                    raise ValueError("Submitted execution directory cannot change")
            elif final_locked and execution != self.execution:
                raise ValueError("Final execution identity cannot change")
            changes["execution"] = execution
        if return_code is not _UNCHANGED and return_code is not None:
            if not final_locked or self.return_code is None:
                changes["return_code"] = return_code
        if message is not _UNCHANGED:
            if message is None and self.state == RunState.UNKNOWN and merged_state != RunState.UNKNOWN:
                # A later scheduler observation supersedes the stale
                # connection-error diagnostic.
                changes["message"] = None
            elif message is not None and (not final_locked or self.message is None):
                changes["message"] = message
        if failure_phase is not _UNCHANGED:
            normalized = _failure_phase(failure_phase)
            if not final_locked or self.failure_phase is None:
                changes["failure_phase"] = normalized
        if validate_completed_files and changes["state"] == RunState.FINISHED:
            # Completion must use the manifest merged under the RunStore
            # lock. A concurrent download failure may have been persisted
            # after the caller captured its candidate manifest.
            completed_files = changes.get("files", self.files)
            if completed_files.failed_files:
                changes["state"] = RunState.FAILED
                changes["failure_phase"] = FailurePhase.DOWNLOAD
                changes["message"] = completed_files.availability_message or message
                if (completed_files.captured_at is not None
                        and self.files.captured_at is None):
                    changes["files"] = JobFiles.discard_capture(completed_files)
        return replace(self, **changes)
