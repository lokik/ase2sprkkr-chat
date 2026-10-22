import asyncio
from dataclasses import replace
import hashlib

import pytest

from ase2sprkkr.execution import ExecutionSpec, RunState, FileManifest, FileEntry, FileMapping
from ase2sprkkr.execution.remote import FileSyncPolicy, RemoteRunner


class _StreamTransport:
    async def file_signature(self, remote):
        size = 8 if remote.endswith("run.err") else len(b"streamed\n")
        return (size, 0)

    async def file_digest(self, remote):
        return hashlib.sha256(b"streamed\n").hexdigest()


class _Executor:
    output_files = FileSyncPolicy.NEVER

    def __init__(self, state):
        from ase2sprkkr.execution.remote_file_handler import RemoteFileHandler

        self.backend_state = RunState(state)
        self.sync_calls = 0
        self.connection = self
        self.transport = _StreamTransport()
        self.file_handler = RemoteFileHandler(self.transport)

    async def follow(self, run_id, files, *, offsets=None, on_chunk):
        on_chunk("out", b"streamed\n")
        return None

    async def status(self, run_id, files=None):
        return {
            "state": self.backend_state.value,
            "return_code": 0 if self.backend_state == RunState.FINISHED else None,
            "failure_phase": None,
        }

    async def cancel(self, run_id):
        return {"state": "cancelled"}


class _GateRunner(RemoteRunner):
    pass


def _run(tmp_path, state, *, original=b"old\n"):
    destination = tmp_path / "calculation.out"
    destination.write_bytes(original)
    output = destination.open("wb")
    executor = _Executor(state)
    runner = _GateRunner(
        executor,
        ExecutionSpec(
            argv=(),
            input_file=None,
            output_file=output,
            directory=str(tmp_path),
        ),
        backend_id="opaque-handle",
        remote_directory="/work/case",
        files=FileManifest(
            stdout=FileEntry(str(destination), "/work/case/calculation.out"),
            mapping=FileMapping(str(tmp_path), "/work/case"),
        ).initialize_availability(),
        state=RunState.RUNNING,
    )
    original = runner.file_handler.collect

    async def counting_collect(*args, **kwargs):
        executor.sync_calls += 1
        return await original(*args, **kwargs)

    runner.file_handler.collect = counting_collect
    return runner, executor, destination, output


def _fail_download(runner, tmp_path):
    """Fail a declared result transfer while still verifying FOLLOW streams."""
    file = FileEntry(
        str(tmp_path / "result.dat"), "/work/case/result.dat",
        availability_state="pending",
    )
    runner.job_files.replace(replace(runner.job_files.snapshot(), outputs=(file,)))

    async def fail_sync(*args, **kwargs):
        raise OSError("remote output could not be downloaded")

    runner.file_handler._sync_file = fail_sync


def test_terminal_remote_state_may_publish_streamed_output(tmp_path):
    runner, executor, destination, output = _run(tmp_path, RunState.FINISHED)

    result = asyncio.run(runner._monitor())

    assert runner.state == RunState.FINISHED
    assert result.return_code == 0
    assert destination.read_bytes() == b"streamed\n"
    assert output.closed is True
    assert executor.sync_calls == 1
    assert runner.job_files.snapshot().stdout.availability_state == "available"
    assert runner.job_files.snapshot().stdout.availability_message is None


def test_unknown_remote_state_does_not_publish_or_download_output(tmp_path):
    runner, executor, destination, output = _run(tmp_path, RunState.UNKNOWN)

    asyncio.run(runner._monitor())

    assert runner.state == RunState.UNKNOWN
    assert runner.failure_phase is None
    assert destination.read_bytes() == b"streamed\n"
    assert output.closed is True
    assert not list(tmp_path.glob(".*.tmp"))
    assert executor.sync_calls == 0


def test_successful_backend_with_failed_download_fails_run(tmp_path):
    runner, executor, destination, output = _run(tmp_path, RunState.FINISHED)
    _fail_download(runner, tmp_path)

    with pytest.raises(OSError, match="could not be downloaded"):
        asyncio.run(runner._monitor())

    assert runner.state == RunState.FAILED
    assert runner.failure_phase.value == "download"
    assert runner.return_code == 0
    failed = runner.job_files.snapshot().outputs[0]
    assert failed.availability_state == "failed"
    assert "could not be downloaded" in failed.availability_message
    assert destination.read_bytes() == b"streamed\n"
    assert output.closed is True


def test_failed_backend_keeps_execution_failure_when_diagnostics_cannot_download(tmp_path):
    runner, executor, destination, output = _run(tmp_path, RunState.FAILED)
    _fail_download(runner, tmp_path)

    async def failed_status(*_args):
        return {
            "state": "failed",
            "return_code": 17,
            "failure_phase": "execution",
            "error": "SPR-KKR exited with status 17",
        }

    executor.status = failed_status
    result = asyncio.run(runner._monitor())

    assert runner.state == RunState.FAILED
    assert runner.failure_phase.value == "execution"
    assert runner.failure_message == "SPR-KKR exited with status 17"
    assert result.return_code == 17
    failed = runner.job_files.snapshot().outputs[0]
    assert failed.availability_state == "failed"
    assert "could not be downloaded" in failed.availability_message
    assert destination.read_bytes() == b"streamed\n"
    assert output.closed is True


def test_cancelled_backend_survives_diagnostic_download_failure(tmp_path):
    runner, _, destination, _ = _run(tmp_path, RunState.CANCELLED)
    _fail_download(runner, tmp_path)
    asyncio.run(runner._monitor())
    assert runner.state == RunState.CANCELLED
    assert runner.failure_phase is None
    assert runner.job_files.snapshot().outputs[0].availability_state == "failed"
    assert destination.read_bytes() == b"streamed\n"



def test_remote_stderr_incomplete_after_success_is_download_failure(tmp_path):
    """FINISHED must not hide a truncated stderr FOLLOW stream."""
    from ase2sprkkr.execution.base import FailurePhase
    from ase2sprkkr.execution.remote import RemoteRunner

    stdout_path = tmp_path / "run.out"
    stderr_path = tmp_path / "run.err"
    output = stdout_path.open("wb")
    error = stderr_path.open("wb")

    executor = _Executor(RunState.FINISHED)
    runner = RemoteRunner(
        executor,
        ExecutionSpec(
            argv=(), input_file=None, output_file=output,
            error_file=error, directory=str(tmp_path),
        ),
        backend_id="opaque-handle", remote_directory="/work/case",
        files=FileManifest(
            stdout=FileEntry(str(stdout_path), "/work/case/run.out"),
            stderr=FileEntry(str(stderr_path), "/work/case/run.err"),
        ).initialize_availability(),
        state=RunState.RUNNING,
    )

    with pytest.raises(RuntimeError, match="stderr size does not match"):
        asyncio.run(runner._monitor())

    assert runner.state == RunState.FAILED
    assert runner.failure_phase == FailurePhase.DOWNLOAD
    assert runner.job_files.snapshot().stderr.availability_state == "failed"
    assert runner.job_files.snapshot().stdout.availability_state == "available"
    assert stdout_path.read_bytes() == b"streamed\n"
    assert stderr_path.read_bytes() == b""
    assert output.closed and error.closed



def test_follow_write_is_idempotent_across_attached_clients(tmp_path):
    """Two clients consuming the same chunks must not duplicate the output."""
    from ase2sprkkr.execution.remote import RemoteRunner
    destination = tmp_path / "output"
    destination.write_bytes(b"abc")
    first = destination.open("ab")
    second = destination.open("ab")
    try:
        spec1 = ExecutionSpec(argv=(), input_file=None, output_file=first, directory=str(tmp_path))
        spec2 = ExecutionSpec(argv=(), input_file=None, output_file=second, directory=str(tmp_path))
        a = RemoteRunner(_Executor(RunState.RUNNING), spec1)
        b = RemoteRunner(_Executor(RunState.RUNNING), spec2)
        a._stream_positions = {"out": 3}
        b._stream_positions = {"out": 3}
        a._write_output(b"def", "out")
        b._write_output(b"def", "out")
        assert destination.read_bytes() == b"abcdef"
        assert a._stream_positions["out"] == b._stream_positions["out"] == 6
        b._stream_positions["out"] = 3
        with pytest.raises(RuntimeError, match="Conflicting FOLLOW bytes"):
            b._write_output(b"wrong", "out")
        assert destination.read_bytes() == b"abcdef"
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    "stage, exception_type, message",
    [
        ("follow", ConnectionError, "SSH disconnected"),
        ("status", OSError, "SSH lost during status query"),
    ],
)
def test_remote_transport_failure_is_recoverable_unknown(
    tmp_path, stage, exception_type, message,
):
    runner, executor, destination, output = _run(tmp_path, RunState.RUNNING)

    async def unavailable(*_args, **_kwargs):
        raise exception_type(message)

    setattr(executor, stage, unavailable)
    result = asyncio.run(runner._monitor())
    assert result.return_code is None
    assert runner._state == RunState.UNKNOWN
    assert runner.failure_phase is None
    assert runner.failure_message == message
    assert output.closed
