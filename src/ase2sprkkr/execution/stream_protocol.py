"""Framing for calculation stdout/stderr over one transport data channel.

The protocol preserves stdout/stderr identity while the streams remain
separate. If a caller explicitly requests a merged sink (``error_file=True``
at the SPR-KKR API), the merge is intentionally lossy: all data is retained in
the output, but cross-stream ordering is not guaranteed to be reproducible and
the original channel identity cannot be recovered afterwards.
"""

from __future__ import annotations

import asyncio
import json
from .base import RunState
from .frame_codec import (
    STREAM_HEADER as _HEADER, KIND_TO_CODE as _KIND_TO_CODE,
    CODE_TO_KIND as _CODE_TO_KIND, encode_frame, read_frame,
)


STREAM_MAGIC = b"A2SSTRM1"


class StreamProtocolError(RuntimeError):
    """Malformed or unsupported remote stream protocol data."""


def encode_stream_frame(kind, data):
    try:
        code = _KIND_TO_CODE[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown stream kind: {kind!r}") from exc
    data = bytes(data)
    return encode_frame(code, data)


def encode_terminal_frame(outcome):
    data = json.dumps(outcome, separators=(",", ":")).encode("utf8")
    return encode_frame(b"T", data)


async def _read_exactly(source, size):
    try:
        return await source.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise StreamProtocolError(
            "Remote stream ended in the middle of a frame"
        ) from exc


async def read_stream_frames(source):
    """Yield ``(kind, data)`` records from a framed remote stream."""
    magic = await _read_exactly(source, len(STREAM_MAGIC))
    if magic != STREAM_MAGIC:
        raise StreamProtocolError("Remote stream has an unknown protocol header")

    while True:
        try:
            record = await read_frame(source, allow_eof=True)
        except asyncio.IncompleteReadError as exc:
            raise StreamProtocolError(
                "Remote stream ended in the middle of a frame"
            ) from exc
        except ValueError as exc:
            raise StreamProtocolError("Stream frame is too large") from exc
        if record is None:
            return
        code, _, payload = record
        if code == b"T":
            try:
                outcome = json.loads(payload)
            except (ValueError, UnicodeError) as exc:
                raise StreamProtocolError("Invalid terminal frame JSON") from exc
            if not isinstance(outcome, dict):
                raise StreamProtocolError("Invalid terminal frame outcome")
            try:
                RunState(outcome.get("state"))
            except (TypeError, ValueError) as exc:
                raise StreamProtocolError("Invalid terminal frame outcome") from exc
            yield "terminal", outcome
            return
        try:
            kind = _CODE_TO_KIND[code]
        except KeyError as exc:
            raise StreamProtocolError(
                f"Remote stream contains an unknown channel code: {code!r}"
            ) from exc
        yield kind, payload


class FramedRunningCommand:
    """Expose calculation streams from one framed transport command.

    The wrapped process' stdout is the framed data channel. Its stderr is a
    transport/control diagnostic channel and is deliberately never exposed as
    calculation stderr.
    """

    def __init__(self, process, *, diagnostic_limit=65536):
        self._process = process
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode = None
        self._diagnostic_limit = diagnostic_limit
        self._diagnostics = bytearray()
        self._protocol_error = None
        self.terminal = None
        self._frames_task = asyncio.create_task(self._demultiplex())
        self._diagnostics_task = asyncio.create_task(self._drain_diagnostics())

    @property
    def diagnostics(self):
        return bytes(self._diagnostics)

    async def _demultiplex(self):
        try:
            async for kind, data in read_stream_frames(self._process.stdout):
                if kind == "terminal":
                    self.terminal = data
                    continue
                (self.stderr if kind == "err" else self.stdout).feed_data(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._protocol_error = exc
            try:
                self._process.terminate()
            except (ProcessLookupError, RuntimeError):
                pass
        finally:
            self.stdout.feed_eof()
            self.stderr.feed_eof()

    async def _drain_diagnostics(self):
        while True:
            data = await self._process.stderr.read(65536)
            if not data:
                return
            remaining = self._diagnostic_limit - len(self._diagnostics)
            if remaining > 0:
                self._diagnostics.extend(data[:remaining])

    async def wait(self):
        self.returncode = await self._process.wait()
        await asyncio.gather(self._frames_task, self._diagnostics_task)
        diagnostic = self.diagnostics.decode("utf8", errors="replace").strip()
        if self._protocol_error is not None:
            message = str(self._protocol_error)
            if diagnostic:
                message += f"; remote transport diagnostic: {diagnostic}"
            raise StreamProtocolError(message) from self._protocol_error
        if self.returncode != 0:
            message = f"Remote stream helper exited with status {self.returncode}"
            if diagnostic:
                message += f"; remote transport diagnostic: {diagnostic}"
            raise StreamProtocolError(message)
        return self.returncode

    def terminate(self):
        return self._process.terminate()
