"""Explicit stream persistence must not depend on the other stream."""

import os
import sys
from io import BytesIO

import pytest

from .. import local

from ..base import ExecutionSpec, RunState
from ..local import LocalExecutor


@pytest.mark.parametrize(
    "store_stdout, store_stderr",
    ((True, True), (True, False), (False, True), (False, False)),
)
def test_local_stream_targets_are_independent(tmp_path, store_stdout, store_stderr):
    stdout_path = tmp_path / "stdout.txt"
    stderr_path = tmp_path / "stderr.txt"
    stdout = stdout_path.open("wb") if store_stdout else None
    stderr = stderr_path.open("wb") if store_stderr else None

    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c",
              "import sys; print('from stdout'); print('from stderr', file=sys.stderr)"),
        input_file=None,
        output_file=stdout,
        error_file=stderr,
        directory=str(tmp_path),
    ))
    observed = {"out": [], "err": []}
    runner.add_output_listener(lambda data, kind: observed[kind].append(data))
    assert runner.run().return_code == 0
    assert runner.state == RunState.FINISHED

    assert "".join(observed["out"]) == "from stdout\n"
    assert "".join(observed["err"]) == "from stderr\n"
    assert stdout_path.exists() is store_stdout
    assert stderr_path.exists() is store_stderr
    if store_stdout:
        assert stdout_path.read_bytes() == b"from stdout\n"
        assert stdout.closed
    if store_stderr:
        assert stderr_path.read_bytes() == b"from stderr\n"
        assert stderr.closed


def test_repeated_cancel_schedules_a_single_stop(tmp_path, monkeypatch):
    stopped = []
    stop = local.stop_local_process

    async def record_stop(process):
        stopped.append(process.pid)
        await stop(process)

    monkeypatch.setattr(local, "stop_local_process", record_stop)
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "import time; time.sleep(30)"),
        directory=str(tmp_path), input_file=None, output_file=None,
    ))
    runner.submit()
    pid = int(runner.backend_id)
    runner.cancel()
    runner.cancel()
    runner.wait()

    assert runner.state == RunState.CANCELLED
    assert stopped == [pid]


def test_failed_output_pump_terminates_child_without_wrapper(tmp_path, monkeypatch):
    stopped = []
    stop = local.stop_local_process

    async def record_stop(process):
        stopped.append(process.pid)
        await stop(process)

    async def failing_pump(source, sink, *, eof=None):
        raise RuntimeError("output pump failed")

    monkeypatch.setattr(local, "stop_local_process", record_stop)
    monkeypatch.setattr(local, "pump_stream", failing_pump)
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "import time; time.sleep(30)"),
        directory=str(tmp_path), input_file=None, output_file=None,
    ))
    runner.submit()
    pid = int(runner.backend_id)
    with pytest.raises(RuntimeError, match="output pump failed"):
        runner.wait()

    assert runner.state == RunState.FAILED
    assert stopped == [pid]

def test_local_accepts_stream_paths_and_named_stdin(tmp_path):
    stdin = tmp_path / "input.dat"
    stdin.write_bytes(b"hello from stdin\n")
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(
            sys.executable, "-c",
            "import sys; data=sys.stdin.buffer.read(); "
            "sys.stdout.buffer.write(data); sys.stderr.buffer.write(b'warning')",
        ),
        input_file=None, directory=str(tmp_path), stdin="input.dat",
        output_file="stdout.dat", error_file="stderr.dat",
    ))
    result = runner.run()
    assert result.return_code == 0
    assert runner.state is RunState.FINISHED
    assert (tmp_path / "stdout.dat").read_bytes() == stdin.read_bytes()
    assert (tmp_path / "stderr.dat").read_bytes() == b"warning"
    assert runner._owned_stdin.closed
    assert all(stream.closed for stream in runner._owned_streams)


def test_local_same_path_is_shared_even_without_inode_info(tmp_path, monkeypatch):
    """Reuse identical paths even when their file type cannot be identified."""
    from ..base import close_output_files

    shared = tmp_path / "shared.dat"
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "pass"),
        directory=str(tmp_path), input_file=None,
        output_file=shared, error_file=str(shared),
    ))
    monkeypatch.setattr(runner, "_stream_inode", lambda value: None)
    try:
        runner._open_streams(stdin=None)
        assert runner._stream_handles["out"] is runner._stream_handles["err"]
        assert len(runner._owned_streams) == 1
        runner._forward_output(b"stdout\\n", "out")
        runner._forward_output(b"stderr\\n", "err")
    finally:
        close_output_files(*runner._owned_streams)
    assert shared.read_bytes() == b"stdout\\nstderr\\n"


def test_local_pathlike_stdout_creates_parent(tmp_path):
    stdout = tmp_path / "nested" / "output.dat"
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "print('ok')"),
        input_file=None, output_file=stdout, directory=str(tmp_path),
    ))
    assert runner.run().return_code == 0
    assert stdout.read_bytes() == b"ok\n"


def test_local_same_output_and_error_path_reuses_one_handle(tmp_path):
    shared = tmp_path / "both.txt"
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c",
              "import sys; print('out'); print('err', file=sys.stderr)"),
        input_file=None, output_file=shared, error_file=str(shared),
        directory=str(tmp_path),
    ))
    assert runner.run().return_code == 0
    contents = shared.read_text()
    assert "out\n" in contents
    assert "err\n" in contents

def test_failed_named_stdin_closes_partially_opened_outputs(tmp_path):
    from ..base import FailurePhase, SubmissionNotStarted

    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "print('not started')"),
        input_file=None, directory=str(tmp_path),
        stdin="missing.dat", output_file="stdout.dat", error_file="stderr.dat",
    ))
    with pytest.raises(SubmissionNotStarted, match="missing.dat"):
        runner.run()
    assert runner.state is RunState.FAILED
    assert runner.failure_phase is FailurePhase.LAUNCH
    assert len(runner._owned_streams) == 2
    assert all(handle.closed for handle in runner._owned_streams)
    assert runner._process is None
    assert runner._loop is None


def test_failed_spawn_preserves_primary_error_over_output_close(tmp_path):
    from io import BytesIO
    from ..base import FailurePhase, SubmissionNotStarted

    class BadClose(BytesIO):
        def close(self):
            super().close()
            raise OSError("close failed")

    output = BadClose()
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(str(tmp_path / "missing-executable"),),
        input_file=None, directory=str(tmp_path), output_file=output,
    ))
    with pytest.raises(SubmissionNotStarted, match="missing-executable"):
        runner.run()
    assert output.closed
    assert runner.failure_phase is FailurePhase.LAUNCH
    assert runner._process is None
    assert runner._loop is None


def test_output_cleanup_error_clears_runner_state(tmp_path):
    from io import BytesIO
    from ..base import FailurePhase

    class BadClose(BytesIO):
        def close(self):
            super().close()
            raise OSError("output close failed")

    output = BadClose()
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "print('finished')"),
        input_file=None, directory=str(tmp_path), output_file=output,
    ))
    with pytest.raises(OSError, match="output close failed"):
        runner.run()
    assert output.closed
    assert runner.state is RunState.FAILED
    assert runner.failure_phase is FailurePhase.FILES
    assert runner._process is None
    assert runner._loop is None



@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_local_stream_paths_share_inode_alias(tmp_path, alias_kind):
    out_path = tmp_path / "out.dat"
    err_path = tmp_path / "err.dat"
    out_path.write_bytes(b"old data")
    if alias_kind == "symlink":
        err_path.symlink_to(out_path)
    else:
        os.link(out_path, err_path)

    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "pass"), directory=str(tmp_path),
        input_file=None, output_file=out_path, error_file=err_path,
    ))
    try:
        runner._open_streams(stdin=None)
        assert runner._stream_handles["out"] is runner._stream_handles["err"]
        assert len(runner._owned_streams) == 1
        runner._forward_output(b"first\n", "out")
        runner._forward_output(b"second\n", "err")
    finally:
        from ..base import close_output_files
        close_output_files(*runner._owned_streams)

    assert out_path.read_bytes() == b"first\nsecond\n"
    assert err_path.read_bytes() == b"first\nsecond\n"


@pytest.mark.parametrize("alias_kind", ("symlink", "hardlink"))
def test_local_aliased_stdout_stderr_end_to_end(tmp_path, alias_kind):
    """A real subprocess must not overwrite either of two inode aliases."""
    combined = tmp_path / "combined.dat"
    combined.write_bytes(b"stale output")
    stderr_alias = tmp_path / "stderr-link.dat"
    if alias_kind == "symlink":
        stderr_alias.symlink_to(combined.name)
    else:
        os.link(combined, stderr_alias)

    stdout_data = b"from stdout\n"
    stderr_data = b"from stderr\n"
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(
            sys.executable, "-c",
            "import sys; "
            "sys.stdout.buffer.write(b'from stdout\\n'); sys.stdout.flush(); "
            "sys.stderr.buffer.write(b'from stderr\\n'); sys.stderr.flush()",
        ),
        directory=str(tmp_path), input_file=None,
        output_file=combined, error_file=stderr_alias,
    ))

    assert runner.run().return_code == 0
    assert runner.state is RunState.FINISHED
    actual = combined.read_bytes()
    assert len(actual) == len(stdout_data) + len(stderr_data)
    assert actual.count(stdout_data) == 1
    assert actual.count(stderr_data) == 1
    assert stderr_alias.read_bytes() == actual


@pytest.mark.parametrize("preopened_kind", ("out", "err"))
def test_local_named_stream_reuses_existing_file_handle(tmp_path, preopened_kind):
    path = tmp_path / "shared.dat"
    with path.open("wb") as stream:
        runner = LocalExecutor().create_run(ExecutionSpec(
            argv=(sys.executable, "-c", "pass"), directory=str(tmp_path),
            input_file=None,
            output_file=stream if preopened_kind == "out" else path,
            error_file=stream if preopened_kind == "err" else path,
        ))
        runner._open_streams(stdin=None)
        assert runner._stream_handles["out"] is stream
        assert runner._stream_handles["err"] is stream
        assert not runner._owned_streams
        runner._forward_output(b"first\n", "out")
        runner._forward_output(b"second\n", "err")

    assert path.read_bytes() == b"first\nsecond\n"


def test_local_prelaunch_cancel_cleans_up_streams_and_loop(tmp_path):
    stdout = (tmp_path / "stdout.dat").open("wb")
    stderr = (tmp_path / "stderr.dat").open("wb")
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "raise RuntimeError('must not launch')"),
        input_file=None, directory=str(tmp_path),
        output_file=stdout, error_file=stderr,
    ))
    runner._cancel_requested.set()
    runner.submit()
    assert runner.wait() is runner.result
    assert runner.state is RunState.CANCELLED
    assert stdout.closed and stderr.closed
    assert runner._loop is None
    assert runner._process is None


def test_nonzero_exit_retains_primary_failure_when_output_close_fails(tmp_path):
    from ..base import FailurePhase

    class BadClose(BytesIO):
        def close(self):
            super().close()
            raise OSError("secondary close failure")

    output = BadClose()
    runner = LocalExecutor().create_run(ExecutionSpec(
        argv=(sys.executable, "-c", "import sys; sys.exit(7)"),
        input_file=None, directory=str(tmp_path), output_file=output,
    ))
    result = runner.run()
    assert result.return_code == 7
    assert runner.state is RunState.FAILED
    assert runner.failure_phase is FailurePhase.EXECUTION
    assert "status 7" in runner.failure_message
    assert isinstance(runner.cleanup_exception, OSError)
    assert str(runner.cleanup_exception) == "secondary close failure"
    assert output.closed
    assert runner._process is None
    assert runner._loop is None
