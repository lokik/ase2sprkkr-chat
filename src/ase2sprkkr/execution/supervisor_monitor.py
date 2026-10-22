"""Shared client-side supervisor observation for local and SSH.

FOLLOW is only observational. Backend status and the frozen manifest are
authoritative; file delivery is supplied by each runner.
"""
from __future__ import annotations

import asyncio

from .base import FailurePhase, RunState, RunStatus, close_output_files

_TERMINAL = {RunState.FINISHED, RunState.FAILED, RunState.CANCELLED}
_FOLLOW_ERRORS = (OSError, ConnectionError, RuntimeError, ValueError)


def backend_status_from_facts(facts, backend_id, *, needs_delivery=False,
                              execution=None, files=None):
    """Interpret raw backend facts once for local and SSH clients.

    The backend cannot declare a remote calculation FINISHED to the client
    until its files have been delivered. A local project's results are
    already on its filesystem, so it never enters DOWNLOADING.
    """
    state = RunState(facts.get("state", RunState.UNKNOWN.value))
    phase = facts.get("failure_phase")
    phase = None if phase is None else FailurePhase(phase)
    if needs_delivery and (
        state == RunState.FINISHED or
        (state == RunState.FAILED and phase not in
         (FailurePhase.SUBMISSION, FailurePhase.LAUNCH))
    ):
        state = RunState.DOWNLOADING
    return RunStatus(
        state, backend_id=backend_id, execution=execution, files=files,
        return_code=facts.get("return_code"), message=facts.get("error"),
        failure_phase=phase,
    )


async def observe_supervisor(follow, status):
    """Return (facts, follow_error, status_error) without guessing completion."""
    follow_error = None
    facts = None
    try:
        facts = await follow()
    except _FOLLOW_ERRORS as exc:
        follow_error = exc

    if facts is None or (
        facts.get("state") in {state.value for state in _TERMINAL}
        and facts.get("files") is None
    ):
        try:
            facts = await status()
        except (OSError, ConnectionError) as exc:
            return None, follow_error, exc
    return facts, follow_error, None


async def monitor_supervisor(runner, follow, status, complete):
    """Shared FOLLOW -> facts -> terminal file completion state machine."""
    try:
        facts, follow_error, status_error = await observe_supervisor(follow, status)
        if facts is None:
            runner._state = RunState.UNKNOWN
            runner._failure_phase = None
            runner._failure_message = str(
                follow_error or status_error or "Supervisor status unavailable"
            )
            return runner.result

        state = RunState(facts.get("state", "unknown"))
        runner._return_code = facts.get("return_code")
        runner.result.return_code = runner._return_code
        phase = facts.get("failure_phase")
        runner._failure_phase = None if phase is None else FailurePhase(phase)
        runner._failure_message = facts.get("error")
        if state not in _TERMINAL:
            runner._state = RunState.UNKNOWN
            runner._failure_phase = None
            runner._failure_message = str(
                follow_error or runner._failure_message or
                f"Supervisor FOLLOW ended without terminal backend state ({state.value})"
            )
            return runner.result

        await complete(facts, state)
        return runner.result
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        if runner._failure_phase is None:
            runner._failure_phase = FailurePhase.EXECUTION
        if runner._failure_message is None:
            runner._failure_message = str(exc)
        if runner._state != RunState.CANCELLED:
            runner._state = RunState.FAILED
        raise
    finally:
        # The shared ManagedRunner lifecycle finalizes output observers
        # before callbacks. The supervisor only releases stream resources.
        # A close failure must not mask a primary monitor exception.
        import sys
        primary = sys.exc_info()[1]
        try:
            close_output_files(runner.spec.output_file, runner.spec.error_file)
        except BaseException:
            if primary is None:
                raise
