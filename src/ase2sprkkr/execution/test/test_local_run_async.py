"""Regression tests for blocking synchronous work in async runner submissions."""

import asyncio
from threading import Event

from ..base import RunState
from ..local import LocalRunner


def test_local_run_async_does_not_block_event_loop_during_submit():
    entered, release = Event(), Event()
    answer = object()
    runner = object.__new__(LocalRunner)
    runner._state = RunState.NEW

    def blocking_submit():
        entered.set()
        if not release.wait(5):
            raise TimeoutError("Local submission blocked the event loop")
        runner._state = RunState.RUNNING

    runner.submit = blocking_submit
    runner.wait = lambda: answer

    async def exercise():
        running = asyncio.create_task(runner.run_async())
        try:
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 3), 4)
            assert not running.done(), "Local submission blocked the event loop"
        finally:
            release.set()
        return await asyncio.wait_for(running, 5)

    assert asyncio.run(exercise()) is answer
