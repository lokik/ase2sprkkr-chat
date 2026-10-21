"""The two stream transports share framing while retaining replay semantics."""
import asyncio
import pytest

from ..frame_codec import (
    KIND_TO_CODE, MAX_DATA, MAX_TERMINAL, encode_frame, read_frame,
)
from ..remote_supervisor_stream import frame as supervisor_frame
from ..stream_protocol import encode_stream_frame, encode_terminal_frame


def _reader(data):
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


@pytest.mark.parametrize("kind", ("out", "err"))
@pytest.mark.parametrize("offset", (None, 0, 123456))
def test_shared_codec_preserves_optional_offset(kind, offset):
    payload = b"binary\x00payload"

    async def check():
        wire = encode_frame(KIND_TO_CODE[kind], payload, offset=offset)
        assert await read_frame(_reader(wire), offsets=offset is not None) == (
            KIND_TO_CODE[kind], offset, payload,
        )
        if offset is None:
            assert wire == encode_stream_frame(kind, payload)
        else:
            assert wire == supervisor_frame(kind, offset, payload)

    asyncio.run(check())


@pytest.mark.parametrize("offset", (None, 3))
@pytest.mark.parametrize("limit", (MAX_DATA, MAX_TERMINAL))
def test_shared_codec_rejects_oversized_frames_before_reading_payload(offset, limit):
    async def check():
        code = b"T" if limit == MAX_TERMINAL else b"O"
        wire = encode_frame(code, b"x" * (limit + 1), offset=offset)
        with pytest.raises(ValueError, match="frame is too large"):
            await read_frame(_reader(wire), offsets=offset is not None)

    asyncio.run(check())


@pytest.mark.parametrize("offset", (None, 4))
def test_shared_codec_never_accepts_truncated_frame(offset):
    async def check():
        wire = encode_frame(b"O", b"payload", offset=offset)
        with pytest.raises(asyncio.IncompleteReadError):
            await read_frame(_reader(wire[:-1]), offsets=offset is not None)
        assert await read_frame(_reader(b""), offsets=False, allow_eof=True) is None
        with pytest.raises(asyncio.IncompleteReadError):
            await read_frame(_reader(b""), offsets=offset is not None)

    asyncio.run(check())


def test_terminal_encoder_uses_shared_frame_layout():
    async def check():
        outcome = {"state": "finished", "return_code": 0}
        frame = encode_terminal_frame(outcome)
        code, offset, payload = await read_frame(_reader(frame))
        assert code == b"T" and offset is None
        assert payload == b'{"state":"finished","return_code":0}'

    asyncio.run(check())
