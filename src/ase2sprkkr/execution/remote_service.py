"""Stateless target-side operations for remote execution.

The remote host is deliberately not a lifecycle authority. Every operation is
fully described by its request or by an opaque backend handle returned from
submission. No ASE2SPRKKR run record is written on the target.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import shutil
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from .growing_file_tail import follow_job_files
from .file_availability import prepare_local_files
from .file_availability import materialize_manifest
from .run_model import FileManifest
from dataclasses import asdict
from .base import ExecutionSpec, FailurePhase, LaunchSpec, RunState, SubmissionNotStarted
from .remote_process_worker import _SPEC_ENV, _START_FD_ENV
from .remote_mailbox import (
    publish_outcome, read_outcome, acknowledge_outcome,
    publish_submission, read_submission, drop_submission,
)




def _status_reply(state, return_code=None, *, failure_phase=None,
                  error=None, files=None, materialization_error=None):
    """Encode backend status facts for the common remote-service wire contract."""
    return {
        "state": state.value,
        "return_code": return_code,
        "error": error,
        "failure_phase": None if failure_phase is None else failure_phase.value,
        "files": files,
        "materialization_error": materialization_error,
    }


def acknowledge_submission(run_id):
    return {"acknowledged": drop_submission(run_id)}


def _encode_handle(data):
    payload = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_handle(value):
    if not isinstance(value, str):
        raise ValueError("Invalid remote execution id")
    padding = "=" * (-len(value) % 4)
    try:
        payload = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        data = json.loads(payload.decode("utf8"))
    except (binascii.Error, UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("Invalid remote execution id") from exc
    if not isinstance(data, dict):
        raise ValueError("Invalid remote execution id")
    kind = data.get("kind")
    if kind == "process":
        valid = (
            type(data.get("pid")) is int and data["pid"] > 0
            and isinstance(data.get("start_time"), str) and bool(data["start_time"])
            and isinstance(data.get("receipt_id"), str)
            and re.fullmatch(r"[0-9a-f]{32}", data["receipt_id"]) is not None
            and isinstance(data.get("socket_path"), str) and bool(data["socket_path"])
        )
    elif kind == "psij":
        valid = (
            isinstance(data.get("backend"), str) and bool(data["backend"])
            and (data.get("url") is None or isinstance(data["url"], str))
            and isinstance(data.get("native_id"), str) and bool(data["native_id"])
        )
    else:
        valid = False
    if not valid:
        raise ValueError("Invalid remote execution id")
    return data


def _absolute(directory, value):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(directory) / path
    return os.fspath(path)


def _process_info(pid):
    try:
        data = Path(f"/proc/{int(pid)}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    end = data.rfind(")")
    if end < 0:
        return None
    fields = data[end + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[19], fields[0]


def _process_running(handle):
    info = _process_info(handle["pid"])
    return (
        info is not None
        and info[0] == str(handle["start_time"])
        and info[1] != "Z"
    )


def _request_spec(request):
    mpi_argv = request.get("mpi_argv")
    supplied = request.get("files")
    manifest = None if supplied is None else FileManifest(**supplied)
    input_files = (
        tuple(request.get("input_files", ())) if manifest is None else
        tuple(entry.remote_path for entry in manifest.inputs if entry.remote_path)
    )
    output_files = (
        tuple(request.get("output_files", ())) if manifest is None else
        tuple(entry.remote_path for entry in manifest.outputs if entry.remote_path)
    )
    cache_files = (
        tuple(request.get("cache_files", ())) if manifest is None else
        tuple(entry.remote_path for entry in manifest.caches
              if entry.remote_path)
    )
    return ExecutionSpec(
        argv=tuple(request["argv"]),
        input_file=request.get("input_file"),
        output_file=request.get("output_file"),
        error_file=request.get("error_file"),
        directory=os.fspath(request["directory"]),
        stdin=request.get("stdin"),
        input_files=input_files,
        output_files=output_files,
        cache_files=cache_files,
        mpi_argv=None if mpi_argv is None else tuple(mpi_argv),
        launch=LaunchSpec(**(request.get("launch") or {})),
        mpi_uses_stdin=request.get("mpi_uses_stdin", True),
    )


def _local_submit(request):
    from .local import materialize_local_spec

    # Everything here happens before launching a supervisor. A failed local
    # filesystem setup proves that submission never started, unlike a lost
    # reply after Popen.
    try:
        directory = Path(request["directory"]).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        original_spec = _request_spec(dict(request, directory=os.fspath(directory)))
        spec = materialize_local_spec(original_spec)

        output_path = _absolute(directory, spec.output_file)
        error_path = _absolute(directory, spec.error_file)
        for path in (output_path, error_path):
            if path is not None:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        prepare_local_files(
            original_spec, output_file=output_path, error_file=error_path,
        )
    except (OSError, ValueError) as exc:
        raise SubmissionNotStarted(str(exc)) from exc

    run_id = str(request.get("id") or uuid4().hex)
    receipt_id = uuid4().hex
    socket_dir = None
    process = None
    published = False
    start_fd = release_fd = socket_read_fd = socket_write_fd = None
    try:
        # IPC creation belongs inside the launch cleanup boundary.
        socket_dir = tempfile.mkdtemp(prefix="ase2sprkkr-follow-")
        socket_path = os.path.join(socket_dir, "socket")
        worker_spec = {
            "argv": list(spec.argv),
            "directory": os.fspath(directory),
            "stdin": _absolute(directory, spec.stdin),
            "output_file": output_path,
            "error_file": error_path,
            "receipt_id": receipt_id,
            "socket_dir": socket_dir,
            "detached": bool(request.get("detached", True)),
            "files": request.get("files"),
        }
        env = os.environ.copy()
        env["ASE2SPRKKR_RUN_ID"] = run_id

        start_fd, release_fd = os.pipe()
        socket_read_fd, socket_write_fd = os.pipe()
        worker_spec["socket_fd"] = socket_write_fd
        env[_SPEC_ENV] = json.dumps(worker_spec, separators=(",", ":"))
        env[_START_FD_ENV] = str(start_fd)
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "ase2sprkkr.execution.remote_process_worker"],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                start_new_session=True,
                close_fds=True,
                pass_fds=(start_fd, socket_write_fd),
            )
        finally:
            # Close write ends before awaiting the supervisor handshake.
            if start_fd is not None:
                fd, start_fd = start_fd, None
                os.close(fd)
            if socket_write_fd is not None:
                fd, socket_write_fd = socket_write_fd, None
                os.close(fd)

        info = _process_info(process.pid)
        if info is None:
            raise RuntimeError("Started remote process disappeared before identification")
        start_time, _ = info
        remote_id = _encode_handle({
            "kind": "process",
            "pid": process.pid,
            "start_time": start_time,
            "receipt_id": receipt_id,
            "socket_path": socket_path,
        })

        # Durable identity precedes releasing the supervisor. Until this
        # receipt is safely stored, the child cannot launch a calculation.
        if request.get("detached"):
            publish_submission(run_id, {"remote_id": remote_id})
            published = True

        os.write(release_fd, b"1")
        with os.fdopen(socket_read_fd, "rb", closefd=True) as ready:
            socket_read_fd = None
            observed_path = ready.readline().decode("utf8").rstrip("\n")
        if observed_path != socket_path:
            raise RuntimeError("Remote supervisor did not create its follow socket")
    except BaseException as exc:
        if not published and process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not published and socket_dir is not None:
            shutil.rmtree(socket_dir, ignore_errors=True)
        if isinstance(exc, OSError) and not published:
            raise SubmissionNotStarted(str(exc)) from exc
        raise
    finally:
        for fd in (start_fd, release_fd, socket_read_fd, socket_write_fd):
            if fd is not None:
                os.close(fd)

    reply = {
        "remote_id": remote_id,
        "state": RunState.RUNNING.value,
        "directory": os.fspath(directory),
        "input_file": _absolute(directory, spec.input_file),
        "output_file": output_path,
        "error_file": error_path,
        "return_code": None,
        "failure_phase": None,
        "files": request.get("files"),
    }
    return reply


def _psij_executor(options):
    from .psij import PsijExecutor

    options = options or {}
    return PsijExecutor(options.get("backend") or "slurm", url=options.get("url"))


def _psij_receipt_id(run_id, backend, url, native_id, directory):
    """Stable project-job receipt across submit and native PSI/J recovery."""
    if not run_id:
        return uuid4().hex
    identity = json.dumps(
        ("ase2sprkkr-psij", str(run_id), backend, url, str(native_id),
         os.fspath(directory)),
        separators=(",", ":"),
    )
    return uuid5(NAMESPACE_URL, identity).hex


def _psij_submit(request):
    executor = _psij_executor(request.get("executor_options"))
    spec = _request_spec(request)
    runner = (
        executor.create_project_run(spec, run_id=request["id"])
        if request.get("id")
        else executor.create_run(spec)
    )
    # PSI/J's _prepare_io() owns cleanup of its complete declared manifest.
    # Running a second cleanup here would duplicate it and risk wildcard caches.
    runner.submit()
    backend_id = runner.backend_id
    if backend_id is None:
        raise RuntimeError("PSI/J submission did not return a backend id")
    remote_id = _encode_handle(
        {
            "kind": "psij",
            "backend": executor.backend,
            "url": executor.url,
            "native_id": str(backend_id),
            "receipt_id": _psij_receipt_id(
                request.get("id"), executor.backend, executor.url,
                backend_id, request["directory"],
            ),
            "directory": request["directory"],
        }
    )
    return {
        "remote_id": remote_id,
        "state": runner.state.value,
        "directory": request["directory"],
        "input_file": request.get("input_file"),
        "output_file": runner.result.output_file,
        "error_file": runner.result.error_file,
        "return_code": runner.return_code,
        "files": request.get("files"),
        "failure_phase": (
            None if runner.failure_phase is None else runner.failure_phase.value
        ),
    }


def submit_remote(request):
    executor = request.get("executor", "local")
    if executor == "local":
        return _local_submit(request)
    if executor == "psij":
        return _psij_submit(request)
    raise ValueError(f"Unsupported remote executor: {executor!r}")


def _process_outcome(receipt_id):
    data = read_outcome(receipt_id)
    if data is None:
        return None
    try:
        state = RunState(data["state"])
        return_code = data.get("return_code")
        raw_phase = data.get("failure_phase")
        failure_phase = None if raw_phase is None else FailurePhase(raw_phase)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Invalid remote process outcome artifact") from exc
    return state, return_code, failure_phase, data.get("error"), data.get("files"), data.get("materialization_error")


def status_remote(remote_id, files=None):
    """Return backend facts and a materialized manifest at completion."""
    handle = _decode_handle(remote_id)
    receipt_id = handle.get("receipt_id")
    outcome = _process_outcome(receipt_id) if receipt_id else None
    if outcome is not None:
        state, code, phase, error, frozen, resolution_error = outcome
        return _status_reply(
            state, code, failure_phase=phase, error=error,
            files=frozen, materialization_error=resolution_error,
        )
    if handle["kind"] == "process":
        state = RunState.RUNNING if _process_running(handle) else RunState.UNKNOWN
        return _status_reply(state)
    if handle["kind"] != "psij":
        raise ValueError("Unsupported remote execution id")
    if str(handle["backend"]).lower() == "slurm" and handle.get("url") is None:
        from .psij import _slurm_status_native
        status = _slurm_status_native(handle["native_id"])
    else:
        status = _psij_executor(handle).status_native(handle["native_id"])
    frozen = None
    resolution_error = None
    if status.state.is_terminal:
        supplied = files
        if supplied is not None:
            result, issue = materialize_manifest(supplied, remote=True)
            frozen = asdict(result)
            resolution_error = None if issue is None else str(issue)
        if receipt_id and supplied is not None:
            publish_outcome(
                receipt_id, status.state, status.return_code,
                status.failure_phase, error=status.message,
                files=frozen, materialization_error=resolution_error,
            )
            outcome = _process_outcome(receipt_id)
            if outcome is not None:
                state, code, phase, error, frozen, resolution_error = outcome
                return _status_reply(state, code, failure_phase=phase, error=error,
                                     files=frozen, materialization_error=resolution_error)
    return _status_reply(
        status.state, status.return_code,
        failure_phase=status.failure_phase, error=status.message,
        files=frozen, materialization_error=resolution_error,
    )


def cancel_remote(remote_id, timeout=1.0):
    handle = _decode_handle(remote_id)
    if handle["kind"] == "process":
        outcome = _process_outcome(handle["receipt_id"])
        if outcome is not None:
            return status_remote(remote_id)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(timeout)
                conn.connect(handle["socket_path"])
                conn.sendall(b'{"command":"cancel"}\n')
                reply = conn.recv(256)
                if b'"accepted":true' not in reply:
                    raise RuntimeError("Supervisor rejected cancellation")
        except (FileNotFoundError, ConnectionRefusedError):
            return status_remote(remote_id)
        # The request is not a terminal fact. Only the supervisor waits for and
        # publishes the final child outcome.
        return status_remote(remote_id)

    if handle["kind"] == "psij":
        executor = _psij_executor(handle)
        job = executor.attach_native_job(handle["native_id"])
        job.cancel()
        state = status_remote(remote_id)
        return state
    raise ValueError("Unsupported remote execution id")


def acknowledge_remote(remote_id):
    """Drop the terminal receipt only after the client persists its run state."""
    handle = _decode_handle(remote_id)
    if handle["kind"] not in ("process", "psij"):
        return {"acknowledged": False}
    receipt_id = handle.get("receipt_id")
    return {"acknowledged": bool(receipt_id and acknowledge_outcome(receipt_id))}


def recover_remote(run_id, request, *, submitted_at=None):
    """Recover an uncertain submit using backend-native identity only."""
    executor = request.get("executor", "local")
    if executor == "local":
        reply = read_submission(run_id)
        if reply is None:
            return None
        return dict(reply, directory=request["directory"], **status_remote(reply["remote_id"]))
    if executor != "psij":
        raise ValueError(f"Unsupported remote executor: {executor!r}")

    backend = _psij_executor(request.get("executor_options"))
    facts = backend.recover_native(
        run_id, request["directory"], submitted_at=submitted_at,
    )
    if facts is None:
        return None
    native_id = facts["native_id"]
    remote_id = _encode_handle({
        "kind": "psij", "backend": backend.backend,
        "url": backend.url, "native_id": str(native_id),
        "receipt_id": _psij_receipt_id(
            run_id, backend.backend, backend.url, native_id,
            request["directory"],
        ),
        "directory": request["directory"],
    })
    state = facts["state"]
    return {
        "remote_id": remote_id,
        "state": state.value,
        "return_code": facts["return_code"],
        "failure_phase": (
            FailurePhase.EXECUTION.value if state == RunState.FAILED else None
        ),
        "directory": facts["directory"],
        "output_file": facts["stdout"],
        "error_file": facts["stderr"],
    }


async def _follow_scheduler_async(remote_id, streams, paths, offsets, poll_interval):
    """Follow scheduler-owned files while PSI/J waits for terminal completion."""
    from .psij import _psij_state

    handle = _decode_handle(remote_id)
    executor = _psij_executor(handle)
    job = executor.attach_native_job(handle["native_id"])

    def emit(kind, data):
        streams[kind].write(data)
        streams[kind].flush()

    async def terminal(_native_status):
        status = await asyncio.to_thread(status_remote, remote_id)
        if RunState(status["state"]).is_terminal:
            return status
        native = job.status
        state = _psij_state(native)
        return _status_reply(
            state, getattr(native, "exit_code", None),
            failure_phase=(
                FailurePhase.EXECUTION if state == RunState.FAILED else None
            ),
            error=getattr(native, "message", None),
        )

    return await follow_job_files(
        paths, emit, job.wait, terminal=terminal, offsets=offsets,
        poll_interval=max(poll_interval, 1.0),
    )


def follow_remote(
    remote_id,
    stdout,
    stderr,
    *,
    stdout_offset=0,
    stderr_offset=0,
    poll_interval=0.1,
    stdout_path=None,
    stderr_path=None,
):
    """Follow direct supervisors or scheduler-owned streams until terminal state."""
    handle = _decode_handle(remote_id)
    offsets = {"out": stdout_offset, "err": stderr_offset}
    streams = {"out": stdout, "err": stderr}
    paths = {
        "out": stdout_path,
        "err": stderr_path,
    }

    if handle["kind"] == "psij":
        return asyncio.run(
            _follow_scheduler_async(remote_id, streams, paths, offsets, poll_interval)
        )
    if handle["kind"] != "process":
        raise ValueError("Unknown execution backend")
    from .remote_supervisor_stream import follow_socket
    path = handle.get("socket_path")
    if path is None:
        raise RuntimeError("Backend handle contains no follow socket")

    # Use the same frame decoder for the local socket and the remote CLI.
    # A merged stream has only one logical channel.
    def on_chunk(kind, offset, data):
        previous = offsets[kind]
        if previous is not None and offset != previous and paths[kind] is not None:
            raise RuntimeError("Supervisor FOLLOW offset mismatch")
        offsets[kind] = offset + len(data)
        streams[kind].write(data)
        streams[kind].flush()

    async def drain_from_socket():
        requested = dict(offsets)
        if stdout_path is not None and stdout_path == stderr_path:
            requested.pop("err")
        return await follow_socket(path, requested, on_chunk)

    try:
        return asyncio.run(drain_from_socket())
    except (FileNotFoundError, ConnectionRefusedError):
        # A completed supervisor has removed its transient socket. Replay
        # declared spools once, without the historical polling lifecycle.
        status = status_remote(remote_id)
        if not RunState(status["state"]).is_terminal:
            raise
        if not any(path is not None for path in paths.values()):
            raise ValueError("Output was live-only and cannot be reattached")
        seen = set()
        for kind in ("out", "err"):
            path = paths[kind]
            if path is None or path in seen:
                continue
            seen.add(path)
            with open(path, "rb") as source:
                source.seek(offsets[kind])
                while True:
                    data = source.read(65536)
                    if not data:
                        break
                    streams[kind].write(data)
                    streams[kind].flush()
        return status
