from types import SimpleNamespace
from threading import Event

from ase2sprkkr.execution import RunState
from ase2sprkkr.execution.base import OutputObservers
from ase2sprkkr.execution.psij import PsijRunner


def test_psij_finalization_failure_overrides_completed_backend():
    runner = object.__new__(PsijRunner)
    runner._job = SimpleNamespace(
        status=SimpleNamespace(state=SimpleNamespace(name="COMPLETED"))
    )
    runner._exception = None
    runner._state = RunState.NEW
    runner._failure_phase = None
    runner._failure_message = None
    runner._return_code = None
    runner._output_observers = OutputObservers()
    runner._done = Event()
    runner.callback = None
    runner.result = object()

    async def fail():
        raise ValueError("result parsing failed")

    runner._monitor = fail
    runner._thread_main()

    assert isinstance(runner._exception, ValueError)
    assert runner.state == RunState.FAILED
    assert runner.failure_phase.value == "execution"


def test_psij_observation_cleanup_failure_is_reported_before_callback():
    runner = object.__new__(PsijRunner)
    runner._job = SimpleNamespace(
        status=SimpleNamespace(state=SimpleNamespace(name="COMPLETED"))
    )
    runner._exception = None
    runner._state = RunState.NEW
    runner._failure_phase = None
    runner._failure_message = None
    runner._return_code = None
    runner._observed_state = None
    runner._output_observers = OutputObservers()
    runner._done = Event()
    seen = []
    runner.callback = lambda result: seen.append(result)
    runner.result = object()

    async def completed():
        return runner.result

    def fail_flush():
        raise OSError("observer decoder flush failed")

    runner._monitor = completed
    runner._output_observers.finish_bytes = fail_flush
    runner._thread_main()

    assert isinstance(runner.exception, OSError)
    assert runner.state is RunState.FAILED
    assert runner.failure_phase.value == "execution"
    assert runner.failure_message == "observer decoder flush failed"
    assert runner._done.is_set()
    assert seen == [runner.result]


def test_psij_terminal_callback_wakes_file_follower(monkeypatch, tmp_path):
    """A silent output file must not delay a scheduler terminal callback."""
    import asyncio
    import ase2sprkkr.execution.psij as psij_module

    async def check():
        runner = object.__new__(PsijRunner)
        runner._job = SimpleNamespace(
            status=SimpleNamespace(state=SimpleNamespace(name="ACTIVE"))
        )
        runner._stdout_path = str(tmp_path / "silent.out")
        runner._stderr_path = None
        runner._observation_stopped = Event()
        from threading import Lock
        runner._observation_signal_lock = Lock()
        runner._observation_wakeup = None
        runner._output_observers = OutputObservers()
        runner._failure_phase = None
        runner._failure_message = None
        runner._observed_state = None
        runner._return_code = None
        runner.result = SimpleNamespace(return_code=None)
        runner._update_files = lambda: None
        terminal_status = SimpleNamespace(
            state=SimpleNamespace(name="COMPLETED"), exit_code=0,
        )

        def observe(job, callback):
            asyncio.get_running_loop().call_soon(callback, terminal_status)
            return lambda: None

        runner.executor = SimpleNamespace(_observe_job=observe, poll_interval=None)

        async def follow(paths, on_chunk, terminal, *, wakeup=None, **kwargs):
            assert wakeup is not None
            await asyncio.wait_for(wakeup, 1)
            return await terminal()

        monkeypatch.setattr(psij_module, "follow_growing_files", follow)
        await asyncio.wait_for(runner._monitor(), 1)
        assert runner._observed_state is RunState.FINISHED
        assert runner.result.return_code == 0

    asyncio.run(check())
