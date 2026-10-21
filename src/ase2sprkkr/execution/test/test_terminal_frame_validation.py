"""The local supervisor and SSH wire formats validate the same terminal facts."""

import asyncio
import struct

import pytest

from ..remote_supervisor_stream import MAGIC, receive
from ..stream_protocol import (
    STREAM_MAGIC,
    StreamProtocolError,
    read_stream_frames,
)


async def _decode(reader, wire):
    stream = asyncio.StreamReader()
    stream.feed_data(wire)
    stream.feed_eof()
    return [item async for item in reader(stream)]


def _wire_frames(payload):
    yield read_stream_frames, (
        STREAM_MAGIC + struct.pack("!cQ", b"T", len(payload)) + payload
    )
    yield receive, (
        MAGIC + struct.pack("!cQQ", b"T", 0, len(payload)) + payload
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b'{"state": "invalid-state"}',
        b'{"state": null}',
        b"\xff",
    ],
)
def test_both_terminals_reject_invalid_payload(payload):
    async def check():
        for reader, wire in _wire_frames(payload):
            with pytest.raises(StreamProtocolError):
                await _decode(reader, wire)

    asyncio.run(check())


@pytest.mark.parametrize("state", ["finished", "failed", "unknown"])
def test_both_terminals_preserve_valid_outcome(state):
    async def check():
        expected = {"state": state, "return_code": None}
        payload = ('{"state":"' + state + '","return_code":null}').encode()
        for reader, wire in _wire_frames(payload):
            frames = await _decode(reader, wire)
            if reader is receive:
                assert frames == [("terminal", 0, expected)]
            else:
                assert frames == [("terminal", expected)]

    asyncio.run(check())
