"""Shared lifecycle contract of reconnectable local and SSH runners."""
import asyncio
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import ExecutionSpec, RunState
from ase2sprkkr.execution.managed_runner import ManagedRunner
from ase2sprkkr.execution.local_project import LocalProjectRunner
from ase2sprkkr.execution.remote import RemoteRunner
from ase2sprkkr.execution.psij import PsijRunner


class _ManagedStub(ManagedRunner):
    def __init__(self, spec):
        super().__init__(SimpleNamespace(), spec, backend_id="job")
        self.cancel_calls = 0

    async def _follow_once(self):
        return {"state": "finished", "return_code": 0, "files": {"outputs": []}}

    async def _backend_status(self):
        raise AssertionError("terminal follow must not query status again")

    async def _complete_terminal(self, facts, state):
        self._state = state

    def _cancel_backend(self):
        self.cancel_calls += 1
        return True


def test_both_reconnectable_modes_share_one_runner_lifecycle():
    assert issubclass(LocalProjectRunner, ManagedRunner)
    assert issubclass(RemoteRunner, ManagedRunner)
    for name in ("_monitor", "_thread_main", "start", "wait", "run",
                 "run_async", "cancel", "add_output_listener",
                 "add_output_error_listener"):
        assert getattr(LocalProjectRunner, name) is getattr(ManagedRunner, name)
        assert getattr(RemoteRunner, name) is getattr(ManagedRunner, name)


def test_psij_inherits_common_lifecycle_without_inheriting_supervisor_monitor():
    assert issubclass(PsijRunner, ManagedRunner)
    for name in ("_thread_main", "start", "wait", "run", "run_async",
                 "cancel", "add_output_listener", "add_output_error_listener"):
        assert getattr(PsijRunner, name) is getattr(ManagedRunner, name)
    assert PsijRunner._monitor is not ManagedRunner._monitor
    assert PsijRunner._finish_monitoring is ManagedRunner._finish_monitoring


def test_managed_runner_uses_terminal_backend_facts(tmp_path):
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    runner = _ManagedStub(spec)
    result = asyncio.run(runner._monitor())
    assert result.return_code == 0
    assert runner._state is RunState.FINISHED


def test_managed_runner_flushes_partial_utf8_before_callback(tmp_path):
    """SSH/local-project lifecycle closes decoders before publishing completion."""
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    runner = _ManagedStub(spec)
    received = []
    runner.add_output_listener(
        lambda text, kind: received.append((kind, text))
    )
    runner.callback = lambda result: received.append(("callback", "done"))

    async def follow():
        runner._output_observers.emit_bytes(bytes((0xe2,)), "out")
        return {"state": "finished", "return_code": 0, "files": {"outputs": []}}

    runner._follow_once = follow
    runner._thread_main()

    assert received == [("out", "\ufffd"), ("callback", "done")]
    assert runner._state is RunState.FINISHED
    assert runner._done.is_set()


def test_managed_runner_cancellation_is_dispatched_once(tmp_path):
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    runner = _ManagedStub(spec)
    future = runner.cancel()
    assert future is runner.cancel()
    assert future.result(timeout=5) is True
    assert runner.cancel_calls == 1
