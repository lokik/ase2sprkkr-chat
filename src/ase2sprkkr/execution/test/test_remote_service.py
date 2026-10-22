import base64
import io
import json
from unittest.mock import MagicMock

import pytest

from ase2sprkkr.execution import FailurePhase, RunState, RunStatus
from ase2sprkkr.execution import psij, remote_process_worker, remote_service


def _process_handle(tmp_path):
    outcome = tmp_path / "remote-outcomes" / ("a" * 32 + ".json")
    handle = {
        "kind": "process",
        "pid": 123,
        "start_time": "456",
        "receipt_id": "a" * 32,
        "socket_path": "/tmp/test-supervisor.sock",
    }
    return handle, remote_service._encode_handle(handle), outcome


def _psij_handle(tmp_path):
    handle = {
        "kind": "psij",
        "backend": "slurm",
        "url": None,
        "native_id": "42",
    }
    return handle, remote_service._encode_handle(handle)


def _write_outcome(path, state, return_code, failure_phase=None):
    from ase2sprkkr.execution import remote_mailbox
    return remote_mailbox.publish_outcome("a" * 32, state, return_code, failure_phase)


def test_remote_id_is_opaque_backend_handle_without_lifecycle_state(tmp_path):
    handle, remote_id, _ = _process_handle(tmp_path)

    assert remote_id != str(handle)
    decoded = remote_service._decode_handle(remote_id)
    assert decoded["kind"] == "process"
    assert decoded["pid"] == 123
    assert "state" not in decoded
    assert "failure_phase" not in decoded
    assert set(decoded) == {"kind", "pid", "start_time", "receipt_id", "socket_path"}


@pytest.mark.parametrize(
    "state, return_code, phase, error",
    [
        (RunState.UNKNOWN, None, None, None),
        (RunState.RUNNING, None, None, None),
        (RunState.FINISHED, 0, None, None),
        (RunState.CANCELLED, -15, None, None),
        (RunState.FAILED, 127, FailurePhase.LAUNCH, "spawn failed"),
        (RunState.FAILED, 7, FailurePhase.EXECUTION, "job failed"),
    ],
)
def test_remote_status_wire_reply_is_shared_between_process_and_psij(
    monkeypatch, tmp_path, state, return_code, phase, error,
):
    _, process_id, _ = _process_handle(tmp_path)
    _, psij_id = _psij_handle(tmp_path)

    monkeypatch.setattr(
        remote_service, "_process_outcome",
        lambda receipt_id: (state, return_code, phase, error, None, None),
    )
    monkeypatch.setattr(
        psij, "_slurm_status_native",
        lambda native_id, **_options: RunStatus(
            state, backend_id=native_id, return_code=return_code,
            failure_phase=phase, message=error,
        ),
    )
    expected = {
        "state": state.value,
        "return_code": return_code,
        "error": error,
        "failure_phase": None if phase is None else phase.value,
        "files": None,
        "materialization_error": None,
    }

    assert remote_service.status_remote(process_id) == expected
    assert remote_service.status_remote(psij_id) == expected


@pytest.mark.parametrize("payload", [
    "!",
    "%%%invalid",
    base64.urlsafe_b64encode(b"not JSON").decode(),
    base64.urlsafe_b64encode(b"\xff").decode(),
    base64.urlsafe_b64encode(b"[]").decode(),
    base64.urlsafe_b64encode(b'{"kind":"process"}').decode(),
    base64.urlsafe_b64encode(b'{"kind":"psij","backend":"slurm"}').decode(),
    base64.urlsafe_b64encode(b'{"kind":"unknown"}').decode(),
])
def test_malformed_remote_id_is_rejected_at_decode_boundary(payload):
    with pytest.raises(ValueError, match="Invalid remote execution id"):
        remote_service._decode_handle(payload)


def test_remote_id_rejects_wrong_field_types():
    for handle in (
        {"kind": "process", "pid": True, "start_time": "10",
         "receipt_id": "a" * 32, "socket_path": "/tmp/socket"},
        {"kind": "process", "pid": 11, "start_time": "10",
         "receipt_id": "../bad", "socket_path": "/tmp/socket"},
        {"kind": "psij", "backend": "", "native_id": "10", "url": None},
        {"kind": "psij", "backend": "slurm", "native_id": None, "url": None},
    ):
        with pytest.raises(ValueError, match="Invalid remote execution id"):
            remote_service._decode_handle(remote_service._encode_handle(handle))


def test_mailbox_receipt_survives_until_client_ack(tmp_path, monkeypatch):
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    _, remote_id, outcome = _process_handle(tmp_path)
    monkeypatch.setattr(remote_service, "_process_running", lambda _handle: False)

    assert remote_service.status_remote(remote_id)["state"] == RunState.UNKNOWN.value
    _write_outcome(outcome, RunState.FINISHED, 0)
    assert remote_service.status_remote(remote_id)["state"] == RunState.FINISHED.value
    assert remote_service.status_remote(remote_id)["state"] == RunState.FINISHED.value
    assert remote_service.acknowledge_remote(remote_id) == {"acknowledged": True}
    assert remote_service.acknowledge_remote(remote_id) == {"acknowledged": True}
    assert not outcome.exists()


def test_status_distinguishes_launch_and_execution_failure(tmp_path, monkeypatch):
    _, remote_id, outcome = _process_handle(tmp_path)
    monkeypatch.setattr(remote_service, "_process_running", lambda _handle: False)

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    _write_outcome(outcome, RunState.FAILED, 127, FailurePhase.LAUNCH)
    launch = remote_service.status_remote(remote_id)
    assert launch["state"] == RunState.FAILED.value
    assert launch["failure_phase"] == FailurePhase.LAUNCH.value

    remote_service.acknowledge_remote(remote_id)
    _write_outcome(outcome, RunState.FAILED, 3, FailurePhase.EXECUTION)
    execution = remote_service.status_remote(remote_id)
    assert execution["state"] == RunState.FAILED.value
    assert execution["return_code"] == 3
    assert execution["failure_phase"] == FailurePhase.EXECUTION.value


def test_status_forwards_slurm_timestamp_metadata(tmp_path, monkeypatch):
    _, encoded = _psij_handle(tmp_path)
    handle = remote_service._decode_handle(encoded)
    handle.update(submitted_at="2026-10-22T12:00:00+00:00",
                  output_file="/scratch/job.out",
                  error_file="/scratch/job.err")
    calls = []

    def native(native_id, *, submitted_at=None, output_paths=()):
        calls.append((native_id, submitted_at, output_paths))
        return RunStatus(RunState.UNKNOWN, backend_id=native_id)

    monkeypatch.setattr(psij, "_slurm_status_native", native)
    status = remote_service.status_remote(remote_service._encode_handle(handle))
    assert status["state"] == RunState.UNKNOWN.value
    assert calls == [("42", "2026-10-22T12:00:00+00:00",
                      ("/scratch/job.out", "/scratch/job.err"))]


def test_slurm_status_is_native_and_does_not_use_psij_job_cache(tmp_path, monkeypatch):
    _, remote_id = _psij_handle(tmp_path)
    monkeypatch.setattr(
        remote_service, "_psij_executor",
        lambda opts: pytest.fail("local Slurm status must not initialize PSI/J"),
    )
    monkeypatch.setattr(
        psij,
        "_slurm_details",
        lambda native_id: {"JobState": "RUNNING", "ExitCode": "0:0"},
    )

    status = remote_service.status_remote(remote_id)

    assert status["state"] == RunState.RUNNING.value
    assert status["return_code"] is None
    assert status["failure_phase"] is None


@pytest.mark.parametrize(
    "accounting_state, code, expected_state, expected_phase",
    [
        ("COMPLETED", 0, RunState.FINISHED, None),
        ("FAILED", 7, RunState.FAILED, FailurePhase.EXECUTION),
        ("COMPLETED", 7, RunState.FAILED, FailurePhase.EXECUTION),
    ],
)
def test_slurm_accounting_status_maps_state_and_exit_code(
    tmp_path, monkeypatch, accounting_state, code, expected_state, expected_phase,
):
    _, remote_id = _psij_handle(tmp_path)
    monkeypatch.setattr(psij, "_slurm_details", lambda _native_id: {})
    monkeypatch.setattr(
        psij, "_scheduler_output",
        lambda argv: f"42|{accounting_state}|{code}:0|\n",
    )

    status = remote_service.status_remote(remote_id)

    assert status["state"] == expected_state.value
    assert status["return_code"] == code
    assert status["failure_phase"] == (
        None if expected_phase is None else expected_phase.value
    )


def test_psij_cancellation_does_not_invent_terminal_state(monkeypatch, tmp_path):
    _, remote_id = _psij_handle(tmp_path)
    calls = []

    class Job:
        def cancel(self):
            calls.append("cancel")

    class Backend:
        def attach_native_job(self, native_id):
            calls.append(native_id)
            return Job()

    monkeypatch.setattr(remote_service, "_psij_executor", lambda _: Backend())
    monkeypatch.setattr(remote_service, "status_remote", lambda _: {
        "state": RunState.RUNNING.value, "return_code": None,
        "failure_phase": None,
    })
    status = remote_service.cancel_remote(remote_id)
    assert status["state"] == RunState.RUNNING.value
    assert calls == ["42", "cancel"]


def _stub_terminal_psij_job(
    monkeypatch, state, return_code, observed, *, subscribe_error=None,
):
    from types import SimpleNamespace

    class Job:
        status = SimpleNamespace(
            state=SimpleNamespace(name=state),
            exit_code=return_code,
            message=None,
        )

        def wait(self):
            raise AssertionError("Remote FOLLOW must not create a job.wait thread")

    class Backend:
        def attach_native_job(self, native_id):
            observed.append(("attach", native_id))
            return Job()

        def _observe_job(self, job, callback):
            observed.append("observe")
            if subscribe_error is not None:
                raise subscribe_error
            callback(job.status)

            def detach():
                observed.append("detach")

            return detach

    monkeypatch.setattr(remote_service, "_psij_executor", lambda _: Backend())


def test_follow_waits_for_psij_terminal_without_repeated_status_probes(tmp_path, monkeypatch):
    _, remote_id = _psij_handle(tmp_path)
    observed = []
    _stub_terminal_psij_job(monkeypatch, "COMPLETED", 0, observed)
    checks = []

    def status(_remote_id):
        checks.append(_remote_id)
        return {"state": RunState.FINISHED.value, "return_code": 0}

    monkeypatch.setattr(remote_service, "status_remote", status)
    outcome = remote_service.follow_remote(remote_id, io.BytesIO(), io.BytesIO())

    assert outcome["state"] == RunState.FINISHED.value
    assert observed == [("attach", "42"), "observe", "detach"]
    assert checks == []


def test_follow_uses_native_terminal_when_status_query_is_transient_unknown(tmp_path, monkeypatch):
    _, remote_id = _psij_handle(tmp_path)
    observed = []
    _stub_terminal_psij_job(monkeypatch, "FAILED", 7, observed)
    checks = []
    def status(_remote_id):
        checks.append(_remote_id)
        return {"state": RunState.UNKNOWN.value}

    monkeypatch.setattr(remote_service, "status_remote", status)
    outcome = remote_service.follow_remote(remote_id, io.BytesIO(), io.BytesIO())

    assert outcome["state"] == RunState.FAILED.value
    assert outcome["return_code"] == 7
    assert outcome["failure_phase"] == FailurePhase.EXECUTION.value
    assert observed == [("attach", "42"), "observe", "detach"]
    assert checks == []


def test_follow_fallback_treats_completed_nonzero_psij_as_failed(
    tmp_path, monkeypatch,
):
    _, remote_id = _psij_handle(tmp_path)
    observed = []
    _stub_terminal_psij_job(monkeypatch, "COMPLETED", 7, observed)
    monkeypatch.setattr(
        remote_service, "status_remote",
        lambda _remote_id: {"state": RunState.UNKNOWN.value},
    )
    outcome = remote_service.follow_remote(
        remote_id, io.BytesIO(), io.BytesIO(),
    )
    assert outcome == {
        "state": RunState.FAILED.value,
        "return_code": 7,
        "failure_phase": FailurePhase.EXECUTION.value,
        "error": None,
        "files": None,
        "materialization_error": None,
    }
    assert observed == [("attach", "42"), "observe", "detach"]


def test_cancel_requests_supervisor_without_inventing_terminal_state(
    tmp_path, monkeypatch
):
    _, remote_id, outcome = _process_handle(tmp_path)
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(remote_service, "_process_running", lambda _handle: True)
    sent = []

    class Connection:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def settimeout(self, timeout):
            pass
        def connect(self, path):
            assert path == "/tmp/test-supervisor.sock"
        def sendall(self, data):
            sent.append(data)
        def makefile(self, mode):
            assert mode == "rb"
            return io.BytesIO(b'{"accepted":true}\n')

    monkeypatch.setattr(remote_service.socket, "socket", lambda *args: Connection())
    status = remote_service.cancel_remote(remote_id)
    assert sent == [b'{"command":"cancel"}\n']
    assert status["state"] == RunState.RUNNING.value
    assert not outcome.exists()


def test_cancel_after_natural_completion_keeps_backend_outcome(tmp_path, monkeypatch):
    _, remote_id, _ = _process_handle(tmp_path)
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    _write_outcome(None, RunState.FINISHED, 0)

    def forbidden(*args):
        raise AssertionError("Terminal process must not receive cancellation")
    monkeypatch.setattr(remote_service.socket, "socket", forbidden)

    status = remote_service.cancel_remote(remote_id)
    assert status["state"] == RunState.FINISHED.value
    assert status["return_code"] == 0


def test_supervisor_requires_parent_launch_release(monkeypatch):
    import os

    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    monkeypatch.setenv(remote_process_worker._START_FD_ENV, str(read_fd))

    with pytest.raises(RuntimeError, match="not released"):
        remote_process_worker._wait_for_parent_release()


def test_uncertain_submit_uses_only_temporary_identity_ticket(tmp_path, monkeypatch):
    from uuid import uuid4
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    run_id = uuid4().hex
    handle = remote_service._encode_handle({
        "kind": "process", "pid": 123,
        "start_time": "456", "receipt_id": "a" * 32,
        "socket_path": "/tmp/fake-socket",
    })
    reply = {"remote_id": handle, "directory": str(tmp_path), "state": "running"}
    from ase2sprkkr.execution import remote_mailbox
    remote_mailbox.publish_submission(run_id, reply)
    recovered = remote_service.recover_remote(run_id, {
        "executor": "local", "directory": str(tmp_path),
    })
    assert recovered["remote_id"] == handle
    assert remote_service.acknowledge_submission(run_id) == {"acknowledged": True}
    assert remote_service.recover_remote(run_id, {"executor": "local"}) is None
    assert remote_service.acknowledge_submission(run_id) == {"acknowledged": True}


def test_direct_follow_uses_shared_socket_protocol(tmp_path, monkeypatch):
    """The target command and the supervisor must agree on frames and offsets."""
    import asyncio
    import os
    import sys

    async def exercise():
        monkeypatch.setattr(
            remote_process_worker, "publish_outcome", lambda *args, **kwargs: None,
        )
        rd, wr = os.pipe()
        spec = {
            "argv": [sys.executable, "-c",
                     "import sys,time; print('stdout', flush=True); "
                     "print('stderr', file=sys.stderr, flush=True); time.sleep(.1)"],
            "directory": str(tmp_path),
            "output_file": str(tmp_path / "stdout"),
            "error_file": str(tmp_path / "stderr"),
            "receipt_id": "a" * 32,
            "socket_fd": wr,
            "detached": True,
        }
        worker = asyncio.create_task(remote_process_worker._run(spec))
        with os.fdopen(rd, "rb") as source:
            socket_path = (await asyncio.to_thread(source.readline)).decode().strip()
        remote_id = remote_service._encode_handle({
            "kind": "process", "socket_path": socket_path,
            "pid": os.getpid(), "start_time": "0", "receipt_id": "a" * 32,
        })
        out, err = io.BytesIO(), io.BytesIO()
        terminal = await asyncio.to_thread(
            remote_service.follow_remote, remote_id, out, err,
            stdout_path=spec["output_file"], stderr_path=spec["error_file"],
        )
        assert terminal["state"] == "finished"
        assert out.getvalue() == b"stdout\n"
        assert err.getvalue() == b"stderr\n"
        assert await worker == 0

    asyncio.run(exercise())


def test_direct_follow_without_stream_files_is_live_only(tmp_path, monkeypatch):
    import asyncio
    import os
    import sys

    async def exercise():
        monkeypatch.setattr(
            remote_process_worker, "publish_outcome", lambda *args, **kwargs: None,
        )
        rd, wr = os.pipe()
        spec = {
            "argv": [sys.executable, "-c",
                     "import time; time.sleep(.15); print('live', flush=True); time.sleep(.1)"],
            "directory": str(tmp_path),
            "output_file": None, "error_file": None,
            "receipt_id": "b" * 32,
            "socket_fd": wr, "detached": True,
        }
        worker = asyncio.create_task(remote_process_worker._run(spec))
        with os.fdopen(rd, "rb") as source:
            socket_path = (await asyncio.to_thread(source.readline)).decode().strip()
        remote_id = remote_service._encode_handle({
            "kind": "process", "socket_path": socket_path,
            "pid": os.getpid(), "start_time": "0", "receipt_id": "b" * 32,
        })
        out, err = io.BytesIO(), io.BytesIO()
        terminal = await asyncio.to_thread(
            remote_service.follow_remote, remote_id, out, err,
            stdout_offset=0, stderr_offset=0,
        )
        assert terminal["state"] == "finished"
        assert out.getvalue() == b"live\n"
        assert err.getvalue() == b""
        assert await worker == 0
        assert list(tmp_path.iterdir()) == []

    asyncio.run(exercise())


def test_ssh_psij_cleanup_removes_declared_outputs_before_submit(tmp_path, monkeypatch):
    from ase2sprkkr.execution import remote_service
    output = tmp_path / "previous.dat"
    kept = tmp_path / "cache.pot"
    output.write_bytes(b"stale")
    kept.write_bytes(b"restart")

    from ase2sprkkr.execution.psij import PsijRunner

    class Runner(PsijRunner):
        def __init__(self, spec):
            self.spec = spec
            self._stdout_path = None
            self._stderr_path = None

        backend_id = "native-123"
        state = RunState.RUNNING
        return_code = None
        failure_phase = None
        result = type("Result", (), {"output_file": None, "error_file": None})()
        def submit(self):
            self._prepare_io()
            assert not output.exists()
            assert kept.read_bytes() == b"restart"

    class Backend:
        backend = "slurm"
        url = None
        def create_project_run(self, spec, *, run_id, cleanup_side):
            assert cleanup_side == "local"
            return Runner(spec)
    monkeypatch.setattr(remote_service, "_psij_executor", lambda options: Backend())
    request = {
        "id": "d" * 32, "executor": "psij", "directory": str(tmp_path),
        "argv": ["true"], "output_files": [str(output)],
        "cache_files": [str(kept)], "input_file": None, "output_file": None,
    }
    response = remote_service.submit_remote(request)
    assert response["remote_id"]



def test_scheduler_request_protects_remote_cache_wildcards(tmp_path):
    """PSI/J cleanup must see cache globs, not just locally staged cache files."""
    request = {
        "directory": str(tmp_path),
        "argv": ["program"],
        "input_file": "case.inp",
        "output_file": "case.out",
        "files": {
            "inputs": [
                {"remote_path": "case.inp"},
                {"local_path": "local-only.inp"},
            ],
            "outputs": [
                {"remote_path": "*.pot"}, {"remote_path": "case.out"},
                {"local_path": "local-only.out"},
            ],
            "caches": [
                {"remote_path": "known-restart.pot"},
                {"remote_path": "restart-*.pot"},
                {"remote_path": "external-cache.pot"},
            ],
        },
    }
    spec = remote_service._request_spec(request)
    assert spec.input_files == ()
    assert spec.output_files == ("*.pot", "case.out")
    assert spec.cache_files == (
        "known-restart.pot", "restart-*.pot", "external-cache.pot",
    )



@pytest.mark.skipif(__import__("sys").platform != "linux", reason="PID identity needs Linux /proc")
def test_failed_submission_ticket_cannot_leave_calculation_running(tmp_path, monkeypatch):
    """A ticket write failure happens while the worker is still blocked."""
    from ase2sprkkr.execution.base import SubmissionNotStarted

    marker = tmp_path / "must-not-run"
    request = {
        "id": "a" * 32,
        "executor": "local",
        "detached": True,
        "directory": str(tmp_path),
        "argv": [
            __import__("sys").executable, "-c",
            "from pathlib import Path; Path('must-not-run').write_text('bad')",
        ],
        "input_file": None,
        "output_file": str(tmp_path / "run.out"),
        "error_file": None,
    }

    def reject(_run_id, _reply):
        raise OSError("ticket store unavailable")

    monkeypatch.setattr(remote_service, "publish_submission", reject)
    with pytest.raises(SubmissionNotStarted, match="ticket store unavailable"):
        remote_service.submit_remote(request)
    assert not marker.exists()


def test_remote_cleanup_preserves_cache_matches_while_removing_stale_outputs(tmp_path):
    from ase2sprkkr.execution.base import ExecutionSpec
    from ase2sprkkr.execution.file_availability import prepare_local_files

    # Output patterns may include protected inputs/caches, but must only
    # remove results, not restart state.
    work = tmp_path / "calculation"
    work.mkdir()
    cache = work / "restart.out"
    stale = work / "obsolete.out"
    stdout = work / "run.log"
    input_file = work / "main.inp"
    for file in (cache, stale, stdout, input_file):
        file.write_text(file.name)

    spec = ExecutionSpec(
        argv=(), directory=str(work), input_file="main.inp",
        output_file="run.log", output_files=("*.out",),
        cache_files=("restart.out",),
    )
    prepare_local_files(spec, output_file=str(stdout))

    assert cache.read_text() == "restart.out"
    assert input_file.read_text() == "main.inp"
    assert not stale.exists()
    assert not stdout.exists()


def test_remote_cleanup_checks_collisions_before_deleting_any_output(tmp_path):
    from ase2sprkkr.execution.base import ExecutionSpec
    from ase2sprkkr.execution.file_availability import prepare_local_files

    work = tmp_path / "calculation"
    work.mkdir()
    cache = work / "restart.dat"
    stale = work / "old.dat"
    cache.write_text("restart")
    stale.write_text("old")

    spec = ExecutionSpec(
        argv=(), directory=str(work), input_file=None, output_file=None,
        output_files=("old.dat", "restart.dat"),
        cache_files=("restart.dat",),
    )
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        prepare_local_files(spec)

    assert stale.read_text() == "old"
    assert cache.read_text() == "restart"


@pytest.mark.parametrize("failure", ("stream_start", "handshake", "startup_cleanup"))
def test_supervisor_prelaunch_failure_publishes_durable_receipt_and_closes_fd(
    tmp_path, monkeypatch, failure,
):
    import asyncio
    import os
    from ase2sprkkr.execution import remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    read_fd, write_fd = os.pipe()
    receipt = "c" * 32
    spec = {
        "argv": ["this-command-must-not-launch"],
        "directory": str(tmp_path),
        "output_file": None, "error_file": None,
        "receipt_id": receipt, "socket_fd": write_fd,
        "detached": True,
    }
    if failure == "stream_start":
        async def broken_start(_self):
            raise OSError("failed to initialize socket")
        monkeypatch.setattr(remote_process_worker.SupervisorStreams, "start", broken_start)
    elif failure == "startup_cleanup":
        original_finish = remote_process_worker.SupervisorStreams.finish

        async def fail_socket_bind(*args, **kwargs):
            raise OSError("socket bind failed")

        async def cleanup_then_fail(self):
            await original_finish(self)
            raise OSError("cleanup failed")

        monkeypatch.setattr(asyncio, "start_unix_server", fail_socket_bind)
        monkeypatch.setattr(
            remote_process_worker.SupervisorStreams, "finish", cleanup_then_fail,
        )
    else:
        os.close(read_fd)
        read_fd = None

    try:
        assert asyncio.run(remote_process_worker._run(spec)) == 127
        if read_fd is not None:
            with os.fdopen(read_fd, "rb") as reader:
                assert reader.read() == b""
            read_fd = None
        with pytest.raises(OSError):
            os.fstat(write_fd)
        outcome = remote_mailbox.read_outcome(receipt)
        assert outcome["state"] == RunState.FAILED.value
        assert outcome["failure_phase"] == FailurePhase.LAUNCH.value
        assert outcome["return_code"] == 127
        assert "socket" in outcome["error"].lower() or "pipe" in outcome["error"].lower()
        if failure == "startup_cleanup":
            assert "socket bind failed" in outcome["error"]
            assert "cleanup failed" in outcome["error"]
    finally:
        if read_fd is not None:
            os.close(read_fd)



@pytest.mark.parametrize("launch_failure", [False, True])
def test_supervisor_stdin_close_failure_preserves_terminal_receipt(
    tmp_path, monkeypatch, launch_failure,
):
    """A failing close() cannot suppress a previous launch error or receipt."""
    import asyncio
    import os
    import sys
    from ase2sprkkr.execution import remote_mailbox

    original_input = tmp_path / "input"
    original_input.write_bytes(b"input")
    from builtins import open as builtin_open

    class FailingInput:
        def __init__(self):
            self.stream = builtin_open(original_input, "rb")

        def fileno(self):
            return self.stream.fileno()

        def close(self):
            self.stream.close()
            raise OSError("stdin close failed")

    opened = []
    def open_input(path, mode):
        assert path == str(original_input)
        assert mode == "rb"
        obj = FailingInput()
        opened.append(obj)
        return obj

    monkeypatch.setattr(remote_process_worker, "open", open_input, raising=False)
    if launch_failure:
        async def broken_spawn(*args, **kwargs):
            raise FileNotFoundError("calculation executable missing")
        monkeypatch.setattr(
            remote_process_worker.asyncio, "create_subprocess_exec", broken_spawn,
        )

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    reader, writer = os.pipe()
    receipt_id = "d" * 32 if launch_failure else "e" * 32
    spec = {
        "argv": [sys.executable, "-c", "pass"],
        "directory": str(tmp_path),
        "stdin": str(original_input),
        "output_file": None, "error_file": None,
        "receipt_id": receipt_id, "socket_fd": writer,
        "detached": True,
    }
    try:
        return_code = asyncio.run(remote_process_worker._run(spec))
    finally:
        with os.fdopen(reader, "rb") as pipe:
            assert pipe.readline().endswith(b"/socket\n")
    receipt = remote_mailbox.read_outcome(receipt_id)
    assert opened and opened[0].stream.closed
    assert receipt["state"] == RunState.FAILED.value
    if launch_failure:
        assert return_code == 127
        assert receipt["failure_phase"] == FailurePhase.LAUNCH.value
        assert "calculation executable missing" in receipt["error"]
        assert "stdin close failed" not in receipt["error"]
    else:
        assert return_code == 0
        assert receipt["failure_phase"] == FailurePhase.EXECUTION.value
        assert "stdin close failed" in receipt["error"]


def test_psij_terminal_receives_and_freezes_client_manifest(tmp_path, monkeypatch):
    from dataclasses import asdict
    from uuid import uuid4

    from ase2sprkkr.execution import FileEntry, FileManifest, FileMapping, remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    local, target = tmp_path / "local", tmp_path / "target"
    local.mkdir()
    target.mkdir()
    (target / "one.dat").write_text("first")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.dat"), str(target / "*.dat"),
                           availability_state="pending"),),
        mapping=FileMapping(str(local), str(target)),
    )
    monkeypatch.setattr(psij, "_slurm_status_native", lambda _id, **_options: RunStatus(
        RunState.FINISHED, backend_id="42", return_code=0,
    ))
    receipt = uuid4().hex
    handle, _ = _psij_handle(tmp_path)
    handle.update(receipt_id=receipt, directory=str(target))
    remote_id = remote_service._encode_handle(handle)

    # A status probe without the manifest cannot freeze membership.
    assert remote_service.status_remote(remote_id)["files"] is None
    assert remote_mailbox.read_outcome(receipt) is None

    reply = remote_service.status_remote(remote_id, asdict(manifest))
    assert reply["state"] == "finished"
    assert reply["materialization_error"] is None
    assert len(reply["files"]["outputs"]) == 1
    assert reply["files"]["outputs"][0]["source"] == str(local / "*.dat")

    (target / "two.dat").write_text("later")
    repeated = remote_service.status_remote(remote_id, asdict(manifest))
    assert repeated["files"] == reply["files"]
    assert remote_mailbox.read_outcome(receipt)["files"] == reply["files"]


@pytest.mark.parametrize("failure_point", ("socket_directory", "first_pipe", "second_pipe"))
def test_remote_submit_ipc_setup_failure_closes_every_fd(
    tmp_path, monkeypatch, failure_point,
):
    """No supervisor can have started; all owned prelaunch IPC must be released."""
    import os
    from ase2sprkkr.execution.base import SubmissionNotStarted

    directory = tmp_path / "socket-directory"
    created = []
    real_pipe = os.pipe
    attempts = []

    def directory_factory(*, prefix):
        assert prefix.startswith("ase2sprkkr-follow-")
        if failure_point == "socket_directory":
            raise OSError("cannot allocate socket directory")
        directory.mkdir()
        return str(directory)

    def pipe():
        attempts.append(True)
        if failure_point == "first_pipe" or (
            failure_point == "second_pipe" and len(attempts) == 2
        ):
            raise OSError("cannot allocate pipe")
        pair = real_pipe()
        created.extend(pair)
        return pair

    monkeypatch.setattr(remote_service.tempfile, "mkdtemp", directory_factory)
    monkeypatch.setattr(remote_service.os, "pipe", pipe)
    monkeypatch.setattr(
        remote_service.subprocess, "Popen",
        lambda *args, **kwargs: pytest.fail("supervisor must not start"),
    )

    with pytest.raises(SubmissionNotStarted, match="cannot allocate"):
        remote_service._local_submit({
            "argv": ("never-executed",),
            "directory": str(tmp_path),
            "detached": True,
        })

    assert not directory.exists()
    assert len(created) == {
        "socket_directory": 0, "first_pipe": 0, "second_pipe": 2,
    }[failure_point]
    for fd in created:
        with pytest.raises(OSError):
            os.fstat(fd)


def _cancel_socket(reply=b'{"accepted":true}\n'):
    """Small socket double for the one-line cancellation protocol."""
    connection = MagicMock()
    connection.__enter__.return_value = connection
    stream = MagicMock()
    stream.__enter__.return_value = stream
    stream.readline.side_effect = lambda limit: reply[:limit]
    connection.makefile.return_value = stream
    return connection


@pytest.mark.parametrize(
    "reply, message",
    [
        (b'{"accepted":false}\n', "Supervisor rejected cancellation"),
        (b'{"accepted":true,"accepted":false}\n', "Invalid supervisor cancellation reply"),
        (b'{"accepted":false,"accepted":true}\n', "Invalid supervisor cancellation reply"),
        (b'{"accepted":1}\n', "Supervisor rejected cancellation"),
        (b'["accepted",true]\n', "Supervisor rejected cancellation"),
        (b'not json\n', "Invalid supervisor cancellation reply"),
        (b'{"accepted":true', "Incomplete supervisor cancellation reply"),
        (b'', "Incomplete supervisor cancellation reply"),
        (b" " * 256 + b"\n", "Incomplete supervisor cancellation reply"),
    ],
)
def test_cancel_rejects_invalid_supervisor_replies(
    tmp_path, monkeypatch, reply, message,
):
    _, remote_id, _ = _process_handle(tmp_path)
    monkeypatch.setattr(remote_service, "_process_running", lambda _: True)
    conn = _cancel_socket(reply)
    monkeypatch.setattr(remote_service.socket, "socket", lambda *args: conn)

    with pytest.raises(RuntimeError, match=message):
        remote_service.cancel_remote(remote_id)
    conn.sendall.assert_called_once_with(b'{"command":"cancel"}\n')


@pytest.mark.parametrize("failure", ("broken_pipe", "reset", "timeout", "eof"))
@pytest.mark.parametrize("terminal", (False, True))
def test_cancel_handles_supervisor_exit_during_ack(
    tmp_path, monkeypatch, failure, terminal,
):
    """A completed backend remains authoritative after the ACK transport dies."""
    _, remote_id, _ = _process_handle(tmp_path)
    receipts = []

    def receipt(_id):
        receipts.append(True)
        if terminal and len(receipts) >= 2:
            return (RunState.FINISHED, 0, None, None, None, None)
        return None

    monkeypatch.setattr(remote_service, "_process_outcome", receipt)
    monkeypatch.setattr(remote_service, "_process_running", lambda _: True)
    conn = _cancel_socket(b"")
    if failure == "broken_pipe":
        conn.sendall.side_effect = BrokenPipeError("supervisor exited")
    elif failure == "reset":
        conn.makefile.return_value.readline.side_effect = ConnectionResetError("reset")
    elif failure == "timeout":
        conn.makefile.return_value.readline.side_effect = TimeoutError("timed out")
    monkeypatch.setattr(remote_service.socket, "socket", lambda *args: conn)

    if terminal:
        result = remote_service.cancel_remote(remote_id)
        assert result["state"] == RunState.FINISHED.value
        assert result["return_code"] == 0
    else:
        expected = RuntimeError if failure == "eof" else {
            "broken_pipe": BrokenPipeError,
            "reset": ConnectionResetError,
            "timeout": TimeoutError,
        }[failure]
        with pytest.raises(expected):
            remote_service.cancel_remote(remote_id)
    assert len(receipts) == 2


def test_cancel_reads_split_ack_over_unix_socket(tmp_path, monkeypatch):
    """The receiver really consumes the prefix before the rest is transmitted."""
    import socket
    from threading import Event, Thread

    _, remote_id, _ = _process_handle(tmp_path)
    monkeypatch.setattr(remote_service, "_process_running", lambda _: True)
    client, server = socket.socketpair()
    prefix = b'{"accepted":'
    prefix_read = Event()
    received = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            client.close()

        def settimeout(self, timeout):
            client.settimeout(timeout)

        def connect(self, path):
            assert path == "/tmp/test-supervisor.sock"

        def sendall(self, data):
            client.sendall(data)

        def makefile(self, mode):
            stream = client.makefile(mode)

            class SplitReader:
                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    stream.close()

                def readline(self, limit):
                    first = stream.read(len(prefix))
                    assert first == prefix
                    prefix_read.set()
                    return first + stream.readline(limit - len(first))

            return SplitReader()

    def respond():
        with server:
            with server.makefile("rb") as request_stream:
                received.append(request_stream.readline())
            server.sendall(prefix)
            if prefix_read.wait(2):
                server.sendall(b'true}\n')

    server.settimeout(2)
    thread = Thread(target=respond, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(remote_service.socket, "socket", lambda *args: Connection())
        status = remote_service.cancel_remote(remote_id)
    finally:
        client.close()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert received == [b'{"command":"cancel"}\n']
    assert status["state"] == RunState.RUNNING.value


def test_cancel_accepts_spaced_json_acknowledgement(tmp_path, monkeypatch):
    _, remote_id, _ = _process_handle(tmp_path)
    monkeypatch.setattr(remote_service, "_process_running", lambda _: True)
    conn = _cancel_socket(b'{ "accepted" : true }\n')
    monkeypatch.setattr(remote_service.socket, "socket", lambda *args: conn)
    assert remote_service.cancel_remote(remote_id)["state"] == RunState.RUNNING.value


def test_cancel_terminal_receipt_is_read_exactly_once(tmp_path, monkeypatch):
    _, remote_id, _ = _process_handle(tmp_path)
    reads = []

    def terminal_outcome(receipt_id):
        reads.append(receipt_id)
        return RunState.FINISHED, 0, None, None, None, None

    monkeypatch.setattr(remote_service, "_process_outcome", terminal_outcome)
    monkeypatch.setattr(
        remote_service.socket, "socket",
        lambda *_args: pytest.fail("Finished supervisor must not be signalled"),
    )
    result = remote_service.cancel_remote(remote_id)
    assert result["state"] == RunState.FINISHED.value
    assert result["return_code"] == 0
    assert reads == ["a" * 32]


def test_cancel_rechecks_status_when_supervisor_exits_before_connect(
    tmp_path, monkeypatch,
):
    _, remote_id, _ = _process_handle(tmp_path)
    calls = []

    def racing_outcome(_id):
        calls.append(True)
        return None if len(calls) == 1 else (RunState.FINISHED, 0, None, None, None, None)

    monkeypatch.setattr(remote_service, "_process_outcome", racing_outcome)
    monkeypatch.setattr(remote_service, "_process_running", lambda _: True)
    conn = _cancel_socket()
    conn.connect.side_effect = ConnectionRefusedError("supervisor exited")
    monkeypatch.setattr(remote_service.socket, "socket", lambda *args: conn)
    assert remote_service.cancel_remote(remote_id)["state"] == RunState.FINISHED.value
    assert len(calls) == 2



@pytest.mark.parametrize("connect_error", [FileNotFoundError, ConnectionRefusedError])
@pytest.mark.parametrize("terminal", [False, True])
def test_cancel_missing_supervisor_socket_never_claims_delivery(
    tmp_path, monkeypatch, connect_error, terminal,
):
    _, remote_id, _ = _process_handle(tmp_path)
    reads = []

    def receipt(_id):
        reads.append(True)
        if terminal and len(reads) == 2:
            return (RunState.FINISHED, 0, None, None, None, None)
        return None

    monkeypatch.setattr(remote_service, "_process_outcome", receipt)
    monkeypatch.setattr(remote_service, "_process_running", lambda _: True)
    conn = _cancel_socket()
    conn.connect.side_effect = connect_error("supervisor unavailable")
    monkeypatch.setattr(remote_service.socket, "socket", lambda *args: conn)

    if terminal:
        result = remote_service.cancel_remote(remote_id)
        assert result["state"] == RunState.FINISHED.value
        assert result["return_code"] == 0
    else:
        with pytest.raises(connect_error, match="supervisor unavailable"):
            remote_service.cancel_remote(remote_id)
    assert len(reads) == 2
    conn.sendall.assert_not_called()


def test_remote_submit_json_setup_failure_cleans_pipes_and_directory(
    tmp_path, monkeypatch,
):
    """Even a non-I/O failure before Popen cannot strand raw pipe ends."""
    import os

    directory = tmp_path / "socket-directory"
    real_pipe = os.pipe
    created = []

    def directory_factory(*, prefix):
        directory.mkdir()
        return str(directory)

    def pipe():
        pair = real_pipe()
        created.extend(pair)
        return pair

    def broken_json(*args, **kwargs):
        raise TypeError("invalid worker specification")

    monkeypatch.setattr(remote_service.tempfile, "mkdtemp", directory_factory)
    monkeypatch.setattr(remote_service.os, "pipe", pipe)
    monkeypatch.setattr(remote_service.json, "dumps", broken_json)
    monkeypatch.setattr(
        remote_service.subprocess, "Popen",
        lambda *args, **kwargs: pytest.fail("supervisor must not start"),
    )

    with pytest.raises(TypeError, match="invalid worker specification"):
        remote_service._local_submit({
            "argv": ("never-executed",),
            "directory": str(tmp_path),
            "detached": True,
        })

    assert not directory.exists()
    assert len(created) == 4
    for fd in created:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_psij_follow_uses_callback_status_not_stale_job_status(monkeypatch, tmp_path):
    from types import SimpleNamespace
    _, remote_id = _psij_handle(tmp_path)
    result_status = SimpleNamespace(
        state=SimpleNamespace(name="FAILED"), exit_code=4, message="failed",
    )
    stale = SimpleNamespace(state=SimpleNamespace(name="ACTIVE"), exit_code=None)
    subscriptions = []

    class Job:
        status = stale
        def wait(self):
            raise AssertionError("Native wait must not be called")

    class Backend:
        def attach_native_job(self, native_id):
            assert native_id == "42"
            return Job()

        def _observe_job(self, job, callback):
            callback(result_status)
            subscriptions.append("subscribe")
            return lambda: subscriptions.append("detach")

    monkeypatch.setattr(remote_service, "_psij_executor", lambda _: Backend())
    monkeypatch.setattr(remote_service, "status_remote",
                        lambda _: pytest.fail("Redundant status probe"))
    result = remote_service.follow_remote(remote_id, io.BytesIO(), io.BytesIO())
    assert result == {
        "state": "failed", "return_code": 4,
        "failure_phase": "execution", "error": "failed",
        "files": None, "materialization_error": None,
    }
    assert subscriptions == ["subscribe", "detach"]


def test_psij_follow_uncertain_subscription_uses_one_slurm_probe(monkeypatch, tmp_path):
    _, remote_id = _psij_handle(tmp_path)
    events = []
    _stub_terminal_psij_job(
        monkeypatch, "ACTIVE", None, events,
        subscribe_error=OSError("PSI/J status subscription unavailable"),
    )
    probes = []
    def status(_):
        probes.append(True)
        return {"state": "unknown"}
    monkeypatch.setattr(remote_service, "status_remote", status)
    result = remote_service.follow_remote(remote_id, io.BytesIO(), io.BytesIO())
    assert result["state"] == "unknown"
    assert result["return_code"] is None
    assert len(probes) == 1


def test_psij_follow_uncertain_subscription_other_scheduler_never_reprobes(monkeypatch, tmp_path):
    handle, _ = _psij_handle(tmp_path)
    handle["backend"] = "pbs"
    handle["url"] = "ssh://remote"
    remote_id = remote_service._encode_handle(handle)
    events = []
    _stub_terminal_psij_job(
        monkeypatch, "ACTIVE", None, events,
        subscribe_error=OSError("PSI/J status subscription unavailable"),
    )
    monkeypatch.setattr(remote_service, "status_remote",
                        lambda _: pytest.fail("Unexpected PSI/J status probe"))
    result = remote_service.follow_remote(remote_id, io.BytesIO(), io.BytesIO())
    assert result["state"] == "unknown"



def test_psij_follow_receives_late_status_from_another_thread(tmp_path, monkeypatch):
    import asyncio
    import threading
    from types import SimpleNamespace

    _, remote_id = _psij_handle(tmp_path)
    # PSI/J attach initially reports NEW before any actual scheduler observation.
    pending = SimpleNamespace(state=SimpleNamespace(name="NEW"), exit_code=None)
    finished = SimpleNamespace(state=SimpleNamespace(name="COMPLETED"), exit_code=0,
                               message=None)
    subscribed = threading.Event()
    callback = []
    detached = []

    class Job:
        status = pending

        def wait(self):
            raise AssertionError("remote FOLLOW must never call job.wait()")

    class Backend:
        def attach_native_job(self, native_id):
            return Job()

        def _observe_job(self, job, receive):
            callback.append(receive)
            subscribed.set()
            return lambda: detached.append(True)

    monkeypatch.setattr(remote_service, "_psij_executor", lambda _: Backend())
    monkeypatch.setattr(remote_service, "status_remote",
                        lambda _: pytest.fail("terminal callback needs no probe"))

    def native_worker():
        assert subscribed.wait(2)
        callback[0](finished)

    worker = threading.Thread(target=native_worker)
    worker.start()
    try:
        outcome = remote_service.follow_remote(
            remote_id, io.BytesIO(), io.BytesIO(), poll_interval=0.01,
        )
    finally:
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert outcome["state"] == RunState.FINISHED.value
    assert outcome["return_code"] == 0
    assert detached == [True]


def test_psij_follow_cancellation_detaches_without_cancelling_job(
    tmp_path, monkeypatch,
):
    import asyncio
    from types import SimpleNamespace

    _, remote_id = _psij_handle(tmp_path)
    calls = []
    active = SimpleNamespace(state=SimpleNamespace(name="ACTIVE"), exit_code=None)

    class Job:
        status = active

        def wait(self):
            raise AssertionError("native wait is forbidden")

        def cancel(self):
            calls.append("job cancelled")

    async def check():
        # Bind the event to the running loop, including on Python 3.9.
        attached = asyncio.Event()

        class Backend:
            def attach_native_job(self, native_id):
                return Job()

            def _observe_job(self, job, callback):
                calls.append("observing")
                attached.set()
                return lambda: calls.append("detached")

        monkeypatch.setattr(remote_service, "_psij_executor", lambda _: Backend())
        task = asyncio.create_task(remote_service._follow_scheduler_async(
            remote_id, {"out": io.BytesIO(), "err": io.BytesIO()},
            {"out": None, "err": None}, {"out": 0, "err": 0}, 0.01,
        ))
        await asyncio.wait_for(attached.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == ["observing", "detached"]

    asyncio.run(check())



def test_psij_follow_unknown_callback_ends_observation_without_waiting(
    tmp_path, monkeypatch,
):
    import asyncio
    from types import SimpleNamespace

    handle, _ = _psij_handle(tmp_path)
    handle["backend"] = "pbs"
    handle["url"] = "ssh://remote"
    remote_id = remote_service._encode_handle(handle)
    pending = SimpleNamespace(state=SimpleNamespace(name="ACTIVE"), exit_code=None)
    unknown = SimpleNamespace(state=SimpleNamespace(name="UNKNOWN"), exit_code=None)
    received = []
    class Job:
        status = pending
        def wait(self):
            raise AssertionError("No blocking wait permitted")
        def cancel(self):
            raise AssertionError("UNKNOWN observation must not cancel job")
    class Backend:
        def attach_native_job(self, native_id):
            return Job()
        def _observe_job(self, job, callback):
            received.append(("observe", callback))
            return lambda: received.append(("detach", None))
    monkeypatch.setattr(remote_service, "_psij_executor", lambda _: Backend())
    monkeypatch.setattr(remote_service, "status_remote",
                        lambda _: pytest.fail("PBS must not invoke Slurm fallback"))

    async def check():
        task = asyncio.create_task(remote_service._follow_scheduler_async(
            remote_id, {"out": io.BytesIO(), "err": io.BytesIO()},
            {"out": None, "err": None}, {"out": 0, "err": 0}, 0.01,
        ))
        for _ in range(30):
            if received:
                break
            await asyncio.sleep(0)
        assert received and received[0][0] == "observe"
        received[0][1](unknown)
        return await asyncio.wait_for(task, 2)

    result = asyncio.run(check())
    assert result["state"] == RunState.UNKNOWN.value
    assert result["return_code"] is None
    assert [x[0] for x in received] == ["observe", "detach"]


def test_psij_callback_follow_preserves_manifest_receipt_boundary(tmp_path, monkeypatch):
    from dataclasses import asdict
    from uuid import uuid4
    from ase2sprkkr.execution import FileEntry, FileManifest, FileMapping, remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    local, target = tmp_path / "local", tmp_path / "target"
    local.mkdir()
    target.mkdir()
    (target / "one.dat").write_text("first")
    manifest = FileManifest(
        outputs=(FileEntry(str(local / "*.dat"), str(target / "*.dat"),
                           availability_state="pending"),),
        mapping=FileMapping(str(local), str(target)),
    )
    receipt = uuid4().hex
    handle, _ = _psij_handle(tmp_path)
    handle.update(receipt_id=receipt, directory=str(target))
    remote_id = remote_service._encode_handle(handle)
    observed = []
    _stub_terminal_psij_job(monkeypatch, "COMPLETED", 0, observed)
    monkeypatch.setattr(psij, "_slurm_status_native", lambda native_id, **_options: RunStatus(
        RunState.FINISHED, backend_id=native_id, return_code=0,
    ))

    followed = remote_service.follow_remote(remote_id, io.BytesIO(), io.BytesIO())
    assert followed["state"] == RunState.FINISHED.value
    assert followed["files"] is None
    assert followed["materialization_error"] is None
    assert remote_mailbox.read_outcome(receipt) is None

    frozen = remote_service.status_remote(remote_id, asdict(manifest))
    assert frozen["state"] == RunState.FINISHED.value
    assert len(frozen["files"]["outputs"]) == 1
    assert frozen["files"]["outputs"][0]["source"] == str(local / "*.dat")
    assert remote_mailbox.read_outcome(receipt)["files"] == frozen["files"]
    assert observed == [("attach", "42"), "observe", "detach"]


def test_remote_ack_retries_failed_unlink_barrier_without_losing_terminal_state(
    tmp_path, monkeypatch,
):
    """A lost durability barrier on ACK is recoverable through the service API."""
    from ase2sprkkr.execution import remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    _, remote_id, outcome = _process_handle(tmp_path)
    monkeypatch.setattr(remote_service, "_process_running", lambda _handle: False)
    assert remote_mailbox.publish_outcome("a" * 32, RunState.FINISHED, 0)
    assert remote_service.status_remote(remote_id)["state"] == "finished"

    original_sync = remote_mailbox.fsync_directory
    calls = 0

    def interrupt_ack_sync(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("directory barrier interrupted")
        return original_sync(directory)

    monkeypatch.setattr(remote_mailbox, "fsync_directory", interrupt_ack_sync)
    with pytest.raises(OSError, match="directory barrier interrupted"):
        remote_service.acknowledge_remote(remote_id)
    assert not outcome.exists()

    # Retrying the same protocol request durably commits the deletion.
    assert remote_service.acknowledge_remote(remote_id) == {"acknowledged": True}
    assert calls == 2
    assert remote_service.status_remote(remote_id)["state"] == "unknown"


def test_submission_ack_retries_failed_unlink_barrier_via_service(
    tmp_path, monkeypatch,
):
    from uuid import uuid4
    from ase2sprkkr.execution import remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    run_id = uuid4().hex
    remote_mailbox.publish_submission(run_id, {"remote_id": "native"})
    original_sync = remote_mailbox.fsync_directory
    calls = 0

    def interrupt_ack_sync(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("submission removal not synced")
        return original_sync(directory)

    monkeypatch.setattr(remote_mailbox, "fsync_directory", interrupt_ack_sync)
    with pytest.raises(OSError, match="submission removal not synced"):
        remote_service.acknowledge_submission(run_id)
    assert remote_mailbox.read_submission(run_id) is None
    assert remote_service.acknowledge_submission(run_id) == {"acknowledged": True}
    assert calls == 2


def test_submission_to_terminal_receipt_and_both_acks(tmp_path, monkeypatch):
    """A temporary submit ticket and terminal receipt have independent ACKs."""
    from uuid import uuid4
    from ase2sprkkr.execution import remote_mailbox

    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(remote_service, "_process_running", lambda _handle: False)
    run_id = uuid4().hex
    receipt_id = uuid4().hex
    remote_id = remote_service._encode_handle({
        "kind": "process", "pid": 123, "start_time": "456",
        "receipt_id": receipt_id, "socket_path": "/tmp/unneeded-supervisor.sock",
    })
    remote_mailbox.publish_submission(run_id, {"remote_id": remote_id})
    request = {"executor": "local", "directory": str(tmp_path)}

    # Recover the submission before the process has produced a result.
    recovered = remote_service.recover_remote(run_id, request)
    assert recovered["remote_id"] == remote_id
    assert recovered["state"] == RunState.UNKNOWN.value

    assert remote_mailbox.publish_outcome(receipt_id, RunState.FINISHED, 0)
    assert remote_service.recover_remote(run_id, request)["state"] == "finished"
    assert remote_service.acknowledge_submission(run_id) == {"acknowledged": True}
    assert remote_service.recover_remote(run_id, request) is None

    # Submission ACK must not discard the independently retained result.
    assert remote_service.status_remote(remote_id)["state"] == "finished"
    assert remote_service.acknowledge_remote(remote_id) == {"acknowledged": True}
    assert remote_service.acknowledge_remote(remote_id) == {"acknowledged": True}
    assert remote_service.status_remote(remote_id)["state"] == "unknown"


def test_request_spec_does_not_duplicate_primary_input():
    from dataclasses import asdict
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    primary = FileEntry("/local/main.in", "/remote/main.in")
    auxiliary = FileEntry("/local/extra.in", "/remote/extra.in")
    manifest = FileManifest(
        inputs=(primary, auxiliary), stdin=True,
    )
    spec = remote_service._request_spec({
        "argv": ("true",), "directory": "/remote",
        "input_file": "/remote/main.in", "stdin": "/remote/main.in",
        "files": asdict(manifest),
    })
    assert spec.input_file == spec.stdin == "/remote/main.in"
    assert spec.input_files == ("/remote/extra.in",)
    assert FileManifest.from_spec(spec).input_files == (
        "/remote/main.in", "/remote/extra.in",
    )


def test_disabled_stdin_is_not_materialized_as_filename(tmp_path):
    from ase2sprkkr.execution import ExecutionSpec

    assert remote_service._absolute(tmp_path, False) is None
    spec = ExecutionSpec(
        argv=("program",), directory=str(tmp_path),
        input_file=None, output_file=None,
    )
    request = {
        "directory": str(tmp_path),
        "argv": list(spec.argv),
        "stdin": None,
    }
    recovered = remote_service._request_spec(request)
    assert recovered.stdin is False
    assert remote_service._absolute(tmp_path, recovered.stdin) is None


def test_local_submit_prepares_resolved_spec_before_supervisor_launch(
    tmp_path, monkeypatch,
):
    """Prelaunch cleanup must receive the resolved request, not an undefined name."""
    from ase2sprkkr.execution.base import ExecutionSpec, SubmissionNotStarted

    prepared = []

    def record_preparation(spec, *, output_file, error_file, side):
        assert isinstance(spec, ExecutionSpec)
        prepared.append((spec, output_file, error_file, side))

    def reject_launch(*args, **kwargs):
        assert len(prepared) == 1, "Cleanup must precede supervisor creation"
        raise OSError("supervisor launch disabled for test")

    monkeypatch.setattr(remote_service, "prepare_local_files", record_preparation)
    monkeypatch.setattr(remote_service.subprocess, "Popen", reject_launch)

    with pytest.raises(SubmissionNotStarted, match="supervisor launch disabled"):
        remote_service._local_submit({
            "argv": ("no-process",),
            "directory": str(tmp_path),
            "input_file": "main.inp",
            "output_file": "run.out",
            "error_file": "run.err",
            "output_files": ("part.dat",),
            "cache_files": ("restart.dat",),
            "cleanup_side": "remote",
        })

    spec, output, error, side = prepared[0]
    assert spec.input_file == "main.inp"
    assert spec.output_files == ("part.dat",)
    assert spec.cache_files == ("restart.dat",)
    assert output == str(tmp_path / "run.out")
    assert error == str(tmp_path / "run.err")
    assert side == "remote"


def test_target_ipc_failure_does_not_delete_declared_outputs(tmp_path, monkeypatch):
    """Target cleanup belongs after IPC preparation, just before Popen."""
    from ase2sprkkr.execution.base import SubmissionNotStarted

    old = tmp_path / "old.dat"
    old.write_text("previous run")

    def broken_pipe():
        raise OSError("IPC unavailable")

    monkeypatch.setattr(remote_service.os, "pipe", broken_pipe)
    request = {
        "directory": str(tmp_path),
        "argv": ["echo", "new"],
        "input_file": None,
        "output_file": None,
        "output_files": ["old.dat"],
        "launch": {"cleanup_outputs": "both"},
    }
    with pytest.raises(SubmissionNotStarted, match="IPC unavailable"):
        remote_service._local_submit(request)
    assert old.read_text() == "previous run"
