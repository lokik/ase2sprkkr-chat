"""Common byte framing for calculation streams and supervisor FOLLOW.

The transport format omits offsets; the supervisor format includes byte
offsets so followers can resume and validate replay.  They deliberately retain
distinct magic headers, while using the same channel codes, size limits, and
framing implementation.
"""
from __future__ import annotations

import asyncio
import struct


STREAM_HEADER = struct.Struct("!cQ")
OFFSET_HEADER = struct.Struct("!cQQ")
KIND_TO_CODE = {"out": b"O", "err": b"E"}
CODE_TO_KIND = {code: kind for kind, code in KIND_TO_CODE.items()}
MAX_DATA = 65536
MAX_TERMINAL = 4096


def encode_frame(code: bytes, data: bytes, *, offset: int | None = None) -> bytes:
    """Encode a channel/terminal payload, optionally with an absolute offset."""
    header = STREAM_HEADER if offset is None else OFFSET_HEADER
    if offset is None:
        return header.pack(code, len(data)) + data
    return header.pack(code, offset, len(data)) + data


async def read_frame(source, *, offsets=False, allow_eof=False):
    """Return (channel code, offset or None, payload) for one bounded frame.

    EOF between complete frames is allowed only for the outer SSH transport.
    Truncated headers and payloads always raise IncompleteReadError.
    Channel-specific validation and terminal JSON interpretation are delegated
    to the caller.
    """
    header = OFFSET_HEADER if offsets else STREAM_HEADER
    first = await source.read(1)
    if not first:
        if allow_eof:
            return None
        raise asyncio.IncompleteReadError(b"", 1)
    rest = await source.readexactly(header.size - 1)
    fields = header.unpack(first + rest)
    code, size = fields[0], fields[-1]
    limit = MAX_TERMINAL if code == b"T" else MAX_DATA
    if size > limit:
        kind = "Terminal" if code == b"T" else "Stream"
        raise ValueError(f"{kind} frame is too large")
    payload = await source.readexactly(size)
    return code, (fields[1] if offsets else None), payload
