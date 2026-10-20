"""An asynchronous runner must not block its caller during submission."""

import asyncio
from threading import Event

import pytest

from ..base import ExecutionResources, ResourceOwnedRunner, RunState
from ..local import LocalRunner
from ..postprocessing import PostprocessingRunner


def _run_while_submission_blocked(runner, entered, release):
    async def exercise():
        running = asyncio.create_task(runner.run_async())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4)
            # The event loop must be able to resume while submit/start
            # is still waiting on the release event.
            assert not running.done()
        finally:
            release.set()
        return await asyncio.wait_for(running, 4)

    return asyncio.run(exercise())


@pytest.mark.parametrize("runner_type", (LocalRunner, PostprocessingRunner))
def test_raw_and_postprocessing_async_runs_offload_blocking_start(runner_type):
    entered, release = Event(), Event()
    answer = object()
    runner = object.__new__(runner_type)

    def blocking_start():
        entered.set()
        if not release.wait(3):
            raise TimeoutError("submission blocked the event loop")
        if runner_type is LocalRunner:
            runner._state = RunState.RUNNING
        else:
            runner._started = True

    if runner_type is LocalRunner:
        runner._state = RunState.NEW
        runner.submit = blocking_start
    else:
        runner._started = False
        runner.start = blocking_start

    runner.wait = lambda: answer
    assert _run_while_submission_blocked(runner, entered, release) is answer


def test_resource_owned_async_run_offloads_submission_and_closes_resources():
    entered, release = Event(), Event()
    answer = object()

    class RawRunner:
        def __init__(self):
            self.state = RunState.NEW
            self.result = answer

        def submit(self):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("resource-owned submission blocked the event loop")
            self.state = RunState.RUNNING

        def wait(self):
            return answer

    resources = ExecutionResources()
    runner = ResourceOwnedRunner(RawRunner(), resources)
    assert _run_while_submission_blocked(runner, entered, release) is answer
    assert resources.closed


def test_resource_owned_async_submission_failure_closes_once():
    failure = RuntimeError("rejected by backend")
    attempts = []
    closed = []

    class RawRunner:
        state = RunState.NEW

        def submit(self):
            attempts.append("submit")
            raise failure

        def wait(self):
            raise AssertionError("wait must not run after submission failure")

    resources = ExecutionResources(
        SimpleClosable(lambda: closed.append("closed")),
    )
    runner = ResourceOwnedRunner(RawRunner(), resources)

    with pytest.raises(RuntimeError, match="rejected by backend") as error:
        asyncio.run(runner.run_async())

    assert error.value is failure
    assert attempts == ["submit"]
    assert closed == ["closed"]
    assert resources.closed


def test_resource_owned_async_run_respects_cancel_before_submit():
    submissions = []

    class RawRunner:
        state = RunState.NEW

        def submit(self):
            submissions.append("submit")

        def wait(self):
            raise AssertionError("cancelled run must not wait")

    resources = ExecutionResources()
    runner = ResourceOwnedRunner(RawRunner(), resources)
    runner.cancel()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.run_async())
    assert not submissions
    assert resources.closed


class SimpleClosable:
    def __init__(self, callback):
        self.callback = callback

    def close(self):
        self.callback()
