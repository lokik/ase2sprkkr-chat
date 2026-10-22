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
    """Select exactly one command and stdin before creating a file manifest.

    Scheduler jobs keep launch metadata so PSI/J can configure resources.
    The unused command and stdin never enter the persistent execution state.
    """
    if spec.mpi_argv is None:
        return spec
    enabled = scheduler or launcher is not None
    if scheduler:
        argv = tuple(spec.mpi_argv)
    elif launcher is not None:
        argv = (*launcher, *spec.mpi_argv)
    else:
        argv = spec.argv
    return replace(
        spec, argv=argv, stdin=spec.mpi_stdin if enabled else spec.stdin,
        mpi_argv=None, mpi_stdin=None,
        launch=(spec.launch if enabled else replace(spec.launch, mpi=MpiSpec())),
    )


def materialize_local_spec(spec: ExecutionSpec) -> ExecutionSpec:
    """Choose serial/MPI before file reservation and local process creation."""
    if spec.mpi_argv is None:
        return spec
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
