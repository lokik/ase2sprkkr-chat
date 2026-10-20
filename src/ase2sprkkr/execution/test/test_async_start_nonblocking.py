"""Async raw local submission must not block the caller's event loop."""

import asyncio
from threading import Event

from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.local import LocalRunner


def test_local_run_async_offloads_blocking_submission():
    runner = object.__new__(LocalRunner)
    runner._state = RunState.NEW
    entered, release = Event(), Event()
    expected = object()

    def submit():
        entered.set()
        if not release.wait(3):
            raise TimeoutError("Local submission blocked the event loop")
        runner._state = RunState.RUNNING

    runner.submit = submit
    runner.wait = lambda: expected

    async def exercise():
        running = asyncio.create_task(runner.run_async())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4)
            assert not running.done()
        finally:
            release.set()
        return await asyncio.wait_for(running, 4)

    assert asyncio.run(exercise()) is expected

def test_postprocessing_run_async_offloads_submission_and_closes_resources():
    """A blocking backend submit must not starve asyncio or leak owned inputs."""
    import io

    from ase2sprkkr.execution.base import ExecutionResources
    from ase2sprkkr.execution.postprocessing import PostprocessingRunner

    entered, release = Event(), Event()
    input_stream = io.BytesIO(b"input")
    resources = ExecutionResources(input_stream)
    answer = object()

    class BlockingBackend:
        state = RunState.NEW
        exception = None

        def submit(self):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Postprocessing submission blocked the event loop")
            self.state = RunState.FINISHED

        def wait(self):
            return answer

    runner = PostprocessingRunner(
        BlockingBackend(), lambda outcome: outcome, resources=resources,
    )

    async def exercise():
        running = asyncio.create_task(runner.run_async())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4)
            assert not running.done()
            assert not resources.closed
        finally:
            release.set()
        return await asyncio.wait_for(running, 5)

    assert asyncio.run(exercise()) is answer
    assert runner.state is RunState.FINISHED
    assert resources.closed
    assert input_stream.closed
