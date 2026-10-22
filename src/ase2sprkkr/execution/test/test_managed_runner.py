"""Shared lifecycle contract of reconnectable local and SSH runners."""
import asyncio
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import ExecutionSpec, RunState
from ase2sprkkr.execution.base import ObservationDetached
from ase2sprkkr.execution.postprocessing import CalculationRun
from ase2sprkkr.execution.managed_runner import ManagedRunner
from ase2sprkkr.execution.local_project import LocalProjectRunner
from ase2sprkkr.execution.remote import RemoteRunner
from ase2sprkkr.execution.psij import PsijRunner
from ase2sprkkr.execution.supervisor_monitor import monitor_supervisor


class _ManagedStub(ManagedRunner):
    def __init__(self, spec):
        super().__init__(SimpleNamespace(), spec, backend_id="job")
        self.cancel_calls = 0

    async def _monitor(self):
        return await monitor_supervisor(
            self, self._follow_once, self._backend_status,
            self._complete_terminal,
        )

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
    for name in ("_thread_main", "start", "wait", "run", "run_async",
                 "cancel", "add_output_listener", "add_output_error_listener"):
        assert getattr(LocalProjectRunner, name) is getattr(ManagedRunner, name)
        assert getattr(RemoteRunner, name) is getattr(ManagedRunner, name)

    # Each concrete backend chooses observation while lifecycle stays shared.
    assert LocalProjectRunner._monitor is not ManagedRunner._monitor
    assert RemoteRunner._monitor is not ManagedRunner._monitor


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


def test_managed_runner_closes_decoders_exactly_once(tmp_path):
    """Supervisor and lifecycle must not duplicate output finalization."""
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    runner = _ManagedStub(spec)
    finalizations = []
    finish = runner._output_observers.finish_bytes

    def finish_once():
        finalizations.append(True)
        finish()

    runner._output_observers.finish_bytes = finish_once
    runner._thread_main()
    assert runner._state is RunState.FINISHED
    assert finalizations == [True]


def test_managed_runner_cancellation_is_dispatched_once(tmp_path):
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    runner = _ManagedStub(spec)
    future = runner.cancel()
    assert future is runner.cancel()
    assert future.result(timeout=5) is True
    assert runner.cancel_calls == 1


def test_managed_runner_without_observation_strategy_fails_explicitly(tmp_path):
    spec = ExecutionSpec(
        argv=(), directory=str(tmp_path), input_file=None, output_file=None,
    )
    runner = ManagedRunner(SimpleNamespace(), spec, backend_id="job")
    with pytest.raises(NotImplementedError, match="implement observation"):
        asyncio.run(runner._monitor())


@pytest.mark.parametrize(
    "runner_class, module_name",
    (
        (LocalProjectRunner, "ase2sprkkr.execution.local_project"),
        (RemoteRunner, "ase2sprkkr.execution.remote"),
    ),
)
def test_supervisor_runner_delegates_to_shared_monitor(
    monkeypatch, runner_class, module_name,
):
    observed = []

    async def observe(runner, follow, status, complete):
        observed.append((runner, follow, status, complete))
        return "observed"

    monkeypatch.setattr(f"{module_name}.monitor_supervisor", observe)
    runner = object.__new__(runner_class)
    runner._follow_once = lambda: None
    runner._backend_status = lambda: None
    runner._complete_terminal = lambda *_: None

    assert asyncio.run(runner._monitor()) == "observed"
    assert observed == [(
        runner, runner._follow_once, runner._backend_status,
        runner._complete_terminal,
    )]



@pytest.mark.parametrize("runner_class", (LocalProjectRunner, RemoteRunner))
def test_detach_unblocks_shared_waiters_without_completing_backend(
    tmp_path, runner_class,
):
    spec = ExecutionSpec(argv=(), directory=str(tmp_path),
                         input_file=None, output_file=None)
    runner = runner_class(
        SimpleNamespace(file_handler=None), spec, backend_id="job",
        run_id="persistent", state=RunState.RUNNING,
    )
    listening = Event()
    callbacks = []

    async def following():
        listening.set()
        await asyncio.Event().wait()

    runner._follow_once = following
    runner.callback = callbacks.append
    runner.start()
    assert listening.wait(5)
    observed = []

    def waiter():
        try:
            runner.wait()
        except ObservationDetached:
            observed.append("detached")

    threads = [Thread(target=waiter) for _ in range(4)]
    for thread in threads:
        thread.start()
    assert runner.stop_observation()
    runner._thread.join(5)
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert not runner._thread.is_alive()
    assert observed == ["detached"] * 4
    assert runner.detached
    assert runner._state == RunState.RUNNING
    assert callbacks == []
    assert not runner.stop_observation()
    with pytest.raises(ObservationDetached):
        runner.wait()


def test_async_waiter_cancellation_does_not_cancel_shared_observation(tmp_path):
    spec = ExecutionSpec(argv=(), directory=str(tmp_path),
                         input_file=None, output_file=None)
    runner = _ManagedStub(spec)
    observing = Event()
    release = Event()

    async def following():
        observing.set()
        await asyncio.to_thread(release.wait)
        return {"state": "finished", "return_code": 0, "files": {"outputs": []}}

    runner._follow_once = following
    runner.start()
    assert observing.wait(5)

    async def check():
        one = asyncio.create_task(runner.wait_async())
        two = asyncio.create_task(runner.wait_async())
        await asyncio.sleep(0)
        one.cancel()
        with pytest.raises(asyncio.CancelledError):
            await one
        release.set()
        assert await asyncio.wait_for(two, 5) is runner.result

    asyncio.run(check())
    assert runner._state == RunState.FINISHED


def test_calculation_detach_does_not_postprocess(tmp_path):
    spec = ExecutionSpec(argv=(), directory=str(tmp_path),
                         input_file=None, output_file=None)
    raw = _ManagedStub(spec)
    raw._state = RunState.RUNNING
    entered = Event()
    parsed = []
    callbacks = []

    async def following():
        entered.set()
        await asyncio.Event().wait()

    raw._follow_once = following
    run = CalculationRun(raw, lambda x: parsed.append(x) or x,
                         callback=callbacks.append)
    run.start(attached=True)
    assert entered.wait(5)
    assert run.stop_observation()
    raw._thread.join(5)
    assert run.detached
    assert run.state == RunState.RUNNING
    assert not run._settled
    with pytest.raises(ObservationDetached):
        run.wait()
    assert parsed == []
    assert callbacks == []


def test_second_stop_does_not_publish_detach_twice_before_monitor_start(tmp_path):
    spec = ExecutionSpec(argv=(), directory=str(tmp_path),
                         input_file=None, output_file=None)
    runner = _ManagedStub(spec)
    calls = []
    runner.observation_detached_callback = lambda: calls.append(True)
    assert runner.stop_observation()
    assert runner.stop_observation() is False
    assert calls == [True]
    assert runner.detached
    assert runner._state == RunState.NEW
    with pytest.raises(ObservationDetached):
        runner.start()
