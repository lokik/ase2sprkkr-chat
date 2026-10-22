"""Parity tests for local and SSH supervisor observation."""
import asyncio
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import RunState
from ase2sprkkr.execution.base import ExecutionSpec, OutputObservers
from ase2sprkkr.execution.supervisor_monitor import monitor_supervisor, observe_supervisor


@pytest.mark.parametrize("mode", ("local", "ssh"))
def test_follow_disconnection_preserves_durable_terminal_outcome(mode, tmp_path):
    output = (tmp_path / (mode + ".out")).open("wb")
    runner = SimpleNamespace(
        _state=RunState.RUNNING, _return_code=None, _failure_phase=None,
        _failure_message=None, result=SimpleNamespace(return_code=None),
        spec=ExecutionSpec(argv=(), directory=str(tmp_path),
                           input_file=None, output_file=output),
        _output_observers=OutputObservers(),
    )
    completed = []

    async def follow():
        raise ConnectionError("FOLLOW disappeared")

    async def status():
        return {"state": "finished", "return_code": 0, "files": {"outputs": []}}

    async def complete(facts, state):
        completed.append((facts["return_code"], state))
        runner._state = state

    asyncio.run(monitor_supervisor(runner, follow, status, complete))
    assert completed == [(0, RunState.FINISHED)]
    assert runner.result.return_code == 0
    assert output.closed


@pytest.mark.parametrize("mode", ("local", "ssh"))
def test_disconnected_status_keeps_observation_unknown(mode, tmp_path):
    runner = SimpleNamespace(
        _state=RunState.RUNNING, _return_code=None, _failure_phase=None,
        _failure_message=None, result=SimpleNamespace(return_code=None),
        spec=ExecutionSpec(argv=(), directory=str(tmp_path),
                           input_file=None, output_file=None),
        _output_observers=OutputObservers(),
    )

    async def follow():
        return {"state": "finished", "return_code": 0, "files": None}

    async def status():
        raise ConnectionError("STATUS lost")

    async def complete(facts, state):
        pytest.fail("A lost terminal status cannot publish completion")

    asyncio.run(monitor_supervisor(runner, follow, status, complete))
    assert runner._state == RunState.UNKNOWN
    assert runner._failure_phase is None
    assert runner._failure_message == "STATUS lost"


def test_terminal_follow_without_manifest_queries_status_once():
    actions = []

    async def follow():
        actions.append("follow")
        return {"state": "finished", "return_code": 0, "files": None}

    async def status():
        actions.append("status")
        return {"state": "finished", "return_code": 0,
                "files": {"outputs": []}}

    facts, follow_error, status_error = asyncio.run(
        observe_supervisor(follow, status)
    )
    assert actions == ["follow", "status"]
    assert follow_error is None and status_error is None
    assert facts["files"] == {"outputs": []}


@pytest.mark.parametrize("state,phase,local,remote", [
    ("running", None, RunState.RUNNING, RunState.RUNNING),
    ("finished", None, RunState.FINISHED, RunState.DOWNLOADING),
    ("failed", "execution", RunState.FAILED, RunState.DOWNLOADING),
    ("failed", "launch", RunState.FAILED, RunState.FAILED),
    ("failed", "submission", RunState.FAILED, RunState.FAILED),
    ("cancelled", None, RunState.CANCELLED, RunState.CANCELLED),
    ("unknown", None, RunState.UNKNOWN, RunState.UNKNOWN),
])
def test_raw_backend_facts_have_shared_interpretation(
    state, phase, local, remote,
):
    from ase2sprkkr.execution.supervisor_monitor import backend_status_from_facts

    facts = {"state": state, "failure_phase": phase, "return_code": 7}
    local_status = backend_status_from_facts(facts, "job")
    remote_status = backend_status_from_facts(
        facts, "job", needs_delivery=True,
    )
    assert local_status.state == local
    assert remote_status.state == remote
    assert local_status.return_code == remote_status.return_code == 7
    assert local_status.failure_phase == remote_status.failure_phase
