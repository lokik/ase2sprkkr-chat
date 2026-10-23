"""Low-level local subprocess launch and stream plumbing."""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import warnings
from dataclasses import replace

from .base import ExecutionSpec, MpiNotFoundWarning, MpiSpec
from .mpi import resolve_mpi_command

_DEFAULT_STDIN = object()


def _mpi_command(launch):
    """Materialize portable MPI intent for this machine."""
    mpi = launch.mpi
    command = resolve_mpi_command(mpi)
    if not command:
        if mpi.allows_serial_fallback:
            if mpi.warning:
                warnings.warn(
                    "MPI autodetection did not find an MPI launcher; "
                    "the calculation will run without MPI.",
                    MpiNotFoundWarning,
                    stacklevel=3,
                )
            return None
        raise FileNotFoundError(
            errno.ENOENT,
            "MPI was requested but no MPI launcher was found",
        )
    return command


def select_mpi_spec(spec: ExecutionSpec, launcher=None, *, scheduler=False):
    """Select the actual command and stdin before file manifest creation.

    MPI-specific None means inherit the serial value; explicit False means
    no stdin. The resulting spec has no alternative variant to stage.
    """
    if not spec.launch and spec.mpi_argv is None and spec.mpi_stdin is None:
        return spec

    enabled = scheduler or launcher is not None
    command = spec.mpi_argv if spec.mpi_argv is not None else spec.argv
    if scheduler:
        argv = tuple(command)
    elif launcher is not None:
        argv = (*launcher, *command)
    else:
        argv = spec.argv

    stdin = (spec.stdin if not enabled or spec.mpi_stdin is None
             else spec.mpi_stdin)
    return replace(
        spec, argv=argv, stdin=stdin,
        mpi_argv=None, mpi_stdin=None,
        launch=(spec.launch if scheduler and enabled
                else replace(spec.launch, mpi=MpiSpec())),
    )


def materialize_local_spec(spec: ExecutionSpec) -> ExecutionSpec:
    """Resolve local MPI before declaring or reserving any input files."""
    if not spec.launch:
        return select_mpi_spec(spec)
    return select_mpi_spec(spec, _mpi_command(spec.launch))


async def start_local_process(
    spec: ExecutionSpec,
    *,
    stdin=_DEFAULT_STDIN,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    start_new_session=False,
):
    """Start a local process with caller-selected standard streams."""
    if stdin is _DEFAULT_STDIN:
        stdin = spec.stdin
    if stdin is False:
        stdin = asyncio.subprocess.DEVNULL
    return await asyncio.create_subprocess_exec(
        *spec.argv,
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        cwd=spec.directory,
        start_new_session=start_new_session,
    )


async def pump_stream(source, sink, *, eof=None):
    """Forward one subprocess byte stream to a synchronous chunk consumer."""
    try:
        while data := await source.read(65536):
            sink(data)
    finally:
        if eof is not None:
            eof()


async def stop_local_process(
    process, *, process_group=False, timeout=1.0, waiting=None,
):
    """Send SIGTERM, then SIGKILL on timeout, without cancelling process.wait().

    For a process group, signal even if its leader has already exited:
    descendants may still be alive and need the full SIGTERM grace period.
    ``waiting`` reuses an existing wait task.
    """
    if process is None or (process.returncode is not None and not process_group):
        return

    if waiting is None:
        waiting = asyncio.create_task(process.wait())

    try:
        if process_group:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        await waiting
        return

    deadline = asyncio.get_running_loop().time() + timeout
    try:
        await asyncio.wait_for(asyncio.shield(waiting), timeout)
    except asyncio.TimeoutError:
        if process.returncode is not None and not process_group:
            return
    else:
        if not process_group:
            return
        # The group can outlive its leader. Preserve the SIGTERM grace
        # period if any descendant still belongs to the group.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(max(0, deadline - asyncio.get_running_loop().time()))

    try:
        if process_group:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    await waiting
