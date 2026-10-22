"""Access to the same target execution service over local calls or SSH.

A connection owns submission, status, FOLLOW and acknowledgment transport.
Neither adapter owns RunRecord, stream files, workspace mapping, or
artifact synchronization. Those remain with the managed runner and workspace.
"""
from __future__ import annotations

import asyncio
import json
import os
import posixpath
import warnings
from dataclasses import asdict

from .base import MpiNotFoundWarning
from .stream_protocol import FramedRunningCommand


class _ObserverStream:
    """The local supervisor accepts byte writers, not a transport process."""

    def __init__(self, on_chunk, kind):
        self.on_chunk = on_chunk
        self.kind = kind

    def write(self, data):
        self.on_chunk(self.kind, data)

    def flush(self):
        pass


class LocalConnection:
    """Direct calls to the very same service used by the remote CLI."""

    async def submit(self, request):
        from .remote_service import submit_remote
        return await asyncio.to_thread(submit_remote, request)

    async def status(self, backend_id, files=None):
        from .remote_service import status_remote
        return await asyncio.to_thread(status_remote, backend_id, files)

    async def cancel(self, backend_id):
        from .remote_service import cancel_remote
        return await asyncio.to_thread(cancel_remote, backend_id)

    async def follow(self, backend_id, files, *, offsets=None, on_chunk):
        from .remote_service import follow_remote_async
        offsets = offsets or {}
        return await follow_remote_async(
            backend_id,
            _ObserverStream(on_chunk, "out"),
            _ObserverStream(on_chunk, "err"),
            stdout_offset=offsets.get("out", 0),
            stderr_offset=offsets.get("err", 0),
            stdout_path=None if files is None or files.stdout is None
            else files.stdout.local_path,
            stderr_path=None if files is None or files.stderr is None
            else files.stderr.local_path,
        )

    async def recover(self, run_id, directory, *, submitted_at=None):
        from .remote_service import recover_remote
        return await asyncio.to_thread(
            recover_remote,
            run_id,
            {"executor": "local", "directory": directory},
            submitted_at=submitted_at,
        )

    async def acknowledge_submission(self, run_id):
        from .remote_service import acknowledge_submission
        return await asyncio.to_thread(acknowledge_submission, run_id)

    async def acknowledge(self, backend_id):
        from .remote_service import acknowledge_remote
        return await asyncio.to_thread(acknowledge_remote, backend_id)


class SSHConnection:
    """Remote CLI transport of the same target-side service contract."""

    def __init__(self, transport, *, command, backend, backend_options):
        self.transport = transport
        self.command = tuple(command)
        self.backend = backend
        self.backend_options = dict(backend_options)
        self._home = None

    async def _json_request(self, action, *args, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf8")
        result = await self.transport.run((*self.command, action, *args), input=data)
        return json.loads(result.stdout.decode("utf8"))

    async def remote_home(self):
        if self._home is None:
            result = await self.transport.run(("sh", "-c", 'printf "%s" "$HOME"'))
            home = result.stdout.decode("utf8").strip()
            if not home:
                raise RuntimeError("Could not determine remote home directory")
            self._home = posixpath.normpath(home)
        return self._home

    async def resolve_directory(self, directory):
        directory = posixpath.normpath(os.fspath(directory))
        if directory == "~":
            return await self.remote_home()
        if directory.startswith("~/"):
            return posixpath.join(await self.remote_home(), directory[2:])
        return directory

    async def submit(self, request):
        launch = request.get("launch") or {}
        mpi = launch.get("mpi") or {}
        if (
            self.backend == "local"
            and mpi.get("autodetect")
            and not mpi.get("required")
            and mpi.get("warning", True)
        ):
            resolved = await self._json_request("resolve-mpi", payload={"mpi": "auto"})
            if not resolved.get("mpi"):
                warnings.warn(
                    "MPI autodetection did not find an MPI launcher on the remote "
                    "host; the calculation will run without MPI.",
                    MpiNotFoundWarning,
                    stacklevel=3,
                )
        return await self._json_request(
            "submit", request["directory"], payload=request,
        )

    async def status(self, backend_id, files=None):
        return await self._json_request(
            "status", backend_id,
            payload=None if files is None else {"files": asdict(files)},
        )

    async def cancel(self, backend_id):
        return await self._json_request("cancel", backend_id)

    async def recover(self, run_id, directory, *, submitted_at=None):
        directory = await self.resolve_directory(directory)
        return await self._json_request(
            "recover", directory, run_id,
            payload={
                "executor": self.backend,
                "executor_options": self.backend_options,
                "submitted_at": submitted_at,
            },
        )

    async def acknowledge_submission(self, run_id):
        return await self._json_request("ack-submit", run_id)

    async def acknowledge(self, backend_id):
        return await self._json_request("ack", backend_id)

    async def follow(self, backend_id, files, *, offsets=None, on_chunk):
        """Stream decoded calculation bytes to the caller, like LocalConnection.

        The framed transport and its diagnostic stderr are implementation
        details of SSHConnection, never exposed to a managed runner.
        """
        options = []
        if files is not None:
            if files.stdout and files.stdout.remote_path:
                options.extend(("--stdout-path", files.stdout.remote_path))
            if files.stderr and files.stderr.remote_path:
                options.extend(("--stderr-path", files.stderr.remote_path))
        for kind, option in (("out", "--stdout-offset"),
                             ("err", "--stderr-offset")):
            offset = None if offsets is None else offsets.get(kind)
            if offset is not None:
                options.extend((option, str(offset)))
            elif offsets is not None:
                options.append("--live-" + ("stdout" if kind == "out" else "stderr"))
        process = await self.transport.start(
            (*self.command, "follow", str(backend_id), *options, "--framed"),
        )
        framed = FramedRunningCommand(process)

        async def pump(source, kind):
            while True:
                chunk = await source.read(65536)
                if not chunk:
                    return
                on_chunk(kind, chunk)

        stdout = asyncio.create_task(pump(framed.stdout, "out"))
        stderr = asyncio.create_task(pump(framed.stderr, "err"))
        try:
            await asyncio.gather(framed.wait(), stdout, stderr)
            return framed.terminal
        except BaseException:
            # A failed consumer must not abandon an SSH helper which could
            # otherwise remain blocked writing into an unread pipe.
            try:
                framed.terminate()
            except (ProcessLookupError, RuntimeError):
                pass
            raise
        finally:
            for task in (stdout, stderr):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdout, stderr, return_exceptions=True)
