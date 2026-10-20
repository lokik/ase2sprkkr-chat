"""FOLLOW channel plumbing shares the local process byte-stream pump."""

import asyncio
from types import SimpleNamespace

import pytest

from ..local_process import pump_stream
from ..remote import RemoteRunner


@pytest.mark.parametrize("kind", ("out", "err"))
def test_forward_follow_stream_preserves_binary_chunks_and_write_order(kind):
    actions = []
    runner = object.__new__(RemoteRunner)
    runner._write_output = lambda data, channel: actions.append(
        ("write", data, channel)
    )
    runner._output_observers = SimpleNamespace(
        emit_bytes=lambda data, channel: actions.append(
            ("notify", data, channel)
        ),
    )

    async def follow():
        source = asyncio.StreamReader()
        payload = b"A" * 65536 + bytes((0, 255))
        source.feed_data(payload)
        source.feed_eof()
        await pump_stream(
            source,
            lambda chunk: runner._forward_follow_chunk(chunk, kind),
        )

    asyncio.run(follow())
    assert actions == [
        ("write", b"A" * 65536, kind),
        ("notify", b"A" * 65536, kind),
        ("write", bytes((0, 255)), kind),
        ("notify", bytes((0, 255)), kind),
    ]


def test_follow_pump_propagates_output_write_failure_without_notify():
    notifications = []
    runner = object.__new__(RemoteRunner)

    def fail_write(data, kind):
        raise OSError("stream output failed")

    runner._write_output = fail_write
    runner._output_observers = SimpleNamespace(
        emit_bytes=lambda data, kind: notifications.append((data, kind))
    )

    async def follow():
        source = asyncio.StreamReader()
        source.feed_data(b"data")
        source.feed_eof()
        await pump_stream(
            source,
            lambda chunk: runner._forward_follow_chunk(chunk, "out"),
        )

    with pytest.raises(OSError, match="stream output failed"):
        asyncio.run(follow())
    assert notifications == []
