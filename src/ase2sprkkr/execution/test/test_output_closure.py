"""Output handle lifecycle shared by local, persistent and remote runners."""

import asyncio

import pytest

from ase2sprkkr.execution.base import ExecutionSpec, RunState, close_output_files
from ase2sprkkr.execution.local import LocalRunner
from ase2sprkkr.execution.local_project import LocalProjectRunner
from ase2sprkkr.execution.remote import RemoteRunner


class _TrackedOutput:
    def __init__(self, name=None):
        self.name = name
        self.close_calls = 0

    def close(self):
        self.close_calls += 1

    def flush(self):
        pass


@pytest.mark.parametrize("backend", ("local", "local_project"))
@pytest.mark.parametrize("merged", (False, True))
def test_local_runners_close_unique_output_handles(monkeypatch, tmp_path, backend, merged):
    stdout = _TrackedOutput(str(tmp_path / "stdout.out"))
    stderr = stdout if merged else _TrackedOutput(str(tmp_path / "stderr.err"))
    spec = ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file=None, output_file=stdout, error_file=stderr,
    )

    if backend == "local":
        async def fail_to_launch(_spec, *, stdin):
            assert stdin is False
            raise RuntimeError("launch failed")

        monkeypatch.setattr("ase2sprkkr.execution.local.start_local_process", fail_to_launch)
        runner = LocalRunner(spec)
        with pytest.raises(RuntimeError, match="launch failed"):
            asyncio.run(runner._execute())
        assert runner.state == RunState.FAILED
    else:
        runner = LocalProjectRunner(None, spec)
        async def follow():
            return {"state": "finished", "return_code": 0, "files": {}}

        async def complete(_facts, state):
            runner._state = state

        # Exercise the actual supervisor cleanup path rather than replacing
        # _monitor(), which owns stream closure.
        monkeypatch.setattr(runner, "_follow_once", follow)
        monkeypatch.setattr(runner, "_complete_terminal", complete)
        runner._thread_main()
        assert runner._done.is_set()
        assert runner.exception is None

    assert stdout.close_calls == 1
    assert stderr.close_calls == 1


@pytest.mark.parametrize(
    "monitor_fails,finish_fails,close_fails",
    [(True, False, False), (False, True, False), (False, False, True),
     (True, True, True), (False, True, True)],
)
def test_local_project_cleanup_attempts_all_stages(
    tmp_path, monkeypatch, monitor_fails, finish_fails, close_fails,
):
    from ase2sprkkr.execution.base import FailurePhase
    calls = []
    errors = (AssertionError("monitor failed"), OSError("finish failed"),
              OSError("close failed"))

    class Output:
        name = str(tmp_path / "out.dat")
        def flush(self):
            pass
        def close(self):
            calls.append("close")
            if close_fails:
                raise errors[2]

    runner = LocalProjectRunner(None, ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file=None, output_file=Output(),
    ))
    async def follow():
        calls.append("monitor")
        if monitor_fails:
            raise errors[0]
        return {"state": "finished", "return_code": 0, "files": {}}
    async def status():
        return {"state": "finished", "return_code": 0, "files": {}}
    async def complete(facts, state):
        runner._state = state
    def finish_bytes():
        calls.append("finish")
        if finish_fails:
            raise errors[1]

    monkeypatch.setattr(runner, "_follow_once", follow)
    monkeypatch.setattr(runner, "_backend_status", status)
    monkeypatch.setattr(runner, "_complete_terminal", complete)
    monkeypatch.setattr(runner._output_observers, "finish_bytes", finish_bytes)
    runner._thread_main()

    assert calls == ["monitor", "close", "finish"]
    expected = next(error for failed, error in zip(
        (monitor_fails, close_fails, finish_fails),
        (errors[0], errors[2], errors[1])
    ) if failed)
    assert runner.exception is expected
    assert runner.failure_message == str(expected)
    assert runner.failure_phase is FailurePhase.EXECUTION
    assert runner.state is RunState.FAILED
    assert runner._done.is_set()
