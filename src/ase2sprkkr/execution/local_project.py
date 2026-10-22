"""Reconnectable local projects use the same supervisor as SSH execution.

A project RunRecord belongs to the client. The supervisor owns only process
facts, a temporary submission ticket and a terminal outcome receipt.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, replace

from .base import (
    ExecutionSpec, FailurePhase, RunState, RunStatus, start_monitor, file_path,
)
from .run_model import FileManifest
from .managed_runner import ManagedRunner
from .supervisor_monitor import backend_status_from_facts, monitor_supervisor
from .remote_service import _decode_handle
from .target_connection import LocalConnection
from ..common.async_utils import run_coro_sync as _run_sync



def local_manifest(files, status):
    """Interpret the supervisor's frozen stream availability; never scan globs.

    The supervisor resolves all declared files exactly once, before publishing
    its terminal receipt. Reconnect must not add files created afterwards.
    """
    published = (
        status.state in (RunState.FINISHED, RunState.CANCELLED) or
        (status.state == RunState.FAILED and status.failure_phase == FailurePhase.EXECUTION)
    )
    for entry in (files.stdout, files.stderr):
        if entry is None or entry.local_path is None:
            continue
        if status.failure_phase == FailurePhase.FILES:
            state, message = "failed", status.message or "Output publication failed"
        elif published and entry.availability_state in (None, "pending", "skipped"):
            # Exact stream availability may be inspected locally, but never
            # expand output/cache globs after the supervisor's terminal freeze.
            state, message = (
                ("available", None) if os.path.isfile(entry.local_path)
                else ("failed", "Published output file is missing")
            )
        elif not published and entry.availability_state in (None, "pending"):
            state, message = "skipped", None
        else:
            continue
        files = files.with_file_availability(entry.local_path, state, message)
    return files


def _local_status_from_facts(backend_id, facts, files=None):
    """Project local supervisor facts into a status without querying it again.

    Recovery already has an authoritative reply; asking the supervisor for
    status a second time risks replacing that snapshot with newer/unknown facts.
    """
    status = backend_status_from_facts(facts, backend_id)
    if status.state.is_terminal:
        # The terminal receipt, not a late filesystem scan, owns membership.
        frozen = facts.get("files")
        if frozen is not None:
            files = FileManifest(**frozen)
        if files is not None:
            status = replace(status, files=local_manifest(files, status))
        if facts.get("materialization_error") and status.state == RunState.FINISHED:
            status = replace(
                status, state=RunState.FAILED,
                failure_phase=FailurePhase.FILES,
                message=facts["materialization_error"],
            )
    return status


_LOCAL_CONNECTION = LocalConnection()


def local_status(backend_id, files=None):
    return _local_status_from_facts(
        backend_id, _run_sync(_LOCAL_CONNECTION.status(backend_id)), files,
    )


class LocalProjectRunner(ManagedRunner):
    _missing_backend_message = "Local run has no backend identity"
    """Project-associated process with supervisor-based follow and reattach."""

    def __init__(
        self, executor, spec: ExecutionSpec, *, run_id=None, callback=None,
        output_listener=None, start=False, backend_id=None, state=RunState.NEW,
        failure_phase=None, failure_message=None,
    ):
        super().__init__(
            executor, spec, callback, output_listener=output_listener,
            backend_id=backend_id, state=state,
        )
        # Direct runners construct a manifest once; project persistence
        # replaces it with the authoritative reserved manifest.
        self._run_id = run_id
        self._prepared_request = None
        self._failure_phase = failure_phase
        self._failure_message = failure_message
        if start:
            self.start()

    def _refresh(self):
        # A terminal result is immutable. After acknowledgment the receipt
        # may disappear while the supervisor process is still shutting down.
        if (self._backend_id is None or self._state.is_terminal
                or self._thread is not None):
            return self._state
        status = local_status(self._backend_id, self._job_files.manifest)
        return self._apply_status(status)

    @property
    def state(self):
        return self._refresh() if self._backend_id is not None else self._state

    async def _monitor(self):
        return await monitor_supervisor(
            self, self._follow_once, self._backend_status,
            self._complete_terminal,
        )

    async def _follow_once(self):
        return await self.executor.connection.follow(
            self._backend_id, self._job_files.manifest,
            on_chunk=lambda kind, data: self._output_observers.emit_bytes(data, kind),
        )

    async def _backend_status(self):
        return await self.executor.connection.status(self._backend_id, self._job_files.manifest)

    async def _complete_terminal(self, facts, state):
        status = _local_status_from_facts(self._backend_id, facts, self._job_files.manifest)
        self._apply_status(status)

    def _request(self):
        # Flush supplementary streams without deriving file membership again.
        for value in (
            self.spec.stdin, *self.spec.input_files,
            *self.spec.output_files, *self.spec.cache_files,
        ):
            file_path(value, self.spec.directory, flush=True)
        files = self._job_files.snapshot()
        return {
            "executor": "local",
            "id": self._run_id,
            "detached": True,
            "directory": os.fspath(self.spec.directory),
            **self.spec.command_request(),
            "input_file": files.input_file,
            "stdin": (None if files.stdin is None
                      else files.stdin.local_path),
            "output_file": files.output_file,
            "error_file": (None if files.stderr is None
                           else files.stderr.local_path),
            "files": asdict(files),
        }

    def prepare_submission(self):
        """Prepare and expose the durable manifest before backend submission."""
        if self._state != RunState.NEW:
            raise RuntimeError(f"Run is already {self._state.value}")
        if not self._run_id:
            raise RuntimeError("Project run has no run id")
        if self._prepared_request is None:
            from .base import SubmissionNotStarted
            try:
                self._prepared_request = self._request()
            except (OSError, ValueError) as exc:
                self._failure_phase = FailurePhase.SUBMISSION
                self._failure_message = str(exc)
                raise SubmissionNotStarted(str(exc)) from exc
        return True

    def submit(self):
        self.prepare_submission()
        from .base import SubmissionNotStarted
        try:
            reply = _run_sync(self.executor.connection.submit(self._prepared_request))
        except SubmissionNotStarted as exc:
            self._state = RunState.FAILED
            self._failure_phase = FailurePhase.SUBMISSION
            self._failure_message = str(exc)
            raise
        self._backend_id = reply["remote_id"]
        if reply.get("files") is not None:
            self._job_files.merge(FileManifest(**reply["files"]))
        self._state = RunState(reply["state"])
        start_monitor(self, missing_backend="Local run has no backend identity")

    def _cancel_backend(self):
        return _run_sync(self.executor.connection.cancel(self._backend_id))


def attach_local(executor, run, *, callback=None, output_listener=None, start=True):
    if run.execution.endpoint.executor != "local":
        raise ValueError("Run does not belong to the local backend")
    backend_id = run.execution.backend_id
    if _decode_handle(backend_id).get("kind") != "process":
        raise ValueError("Local project does not have a supervisor handle")
    files = run.files
    spec = ExecutionSpec(
        argv=(), input_file=files.input_file,
        output_file=files.output_file,
        error_file=None if files.stderr is None else files.stderr.local_path,
        directory=run.execution.directory.local,
    )
    status = local_status(backend_id, files)
    runner = LocalProjectRunner(
        executor, spec, callback=callback, output_listener=output_listener,
        backend_id=backend_id, run_id=run.id,
        state=status.state, failure_phase=status.failure_phase,
        failure_message=status.message,
    )
    runner.replace_file_manifest(status.files or files)
    runner._return_code = status.return_code
    runner.result.return_code = status.return_code
    if start:
        runner.start()
    return runner
