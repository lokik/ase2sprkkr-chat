"""Real Local/SSH(LocalTransport) parity across submit, FOLLOW and reattach.

This exercises the *target service* and supervisor in subprocesses, not a
mocked monitor. SSH here means the SSH protocol/client path with LocalTransport
as its transport, so no login credentials or network service are required.
"""
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from ase2sprkkr.execution import (
    Execution, ExecutionSpec, FileManifest, LaunchSpec, LocalExecutor,
    RunRecord, RunState, WorkingDirectory,
)
from ase2sprkkr.execution.endpoint import endpoint_for_executor
from ase2sprkkr.execution.remote_endpoint import RemoteEndpointExecutor
from ase2sprkkr.execution.transport import LocalTransport


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="The supervisor requires Linux /proc process identity",
)


@pytest.fixture(autouse=True)
def isolated_supervisor_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ASE2SPRKKR_REMOTE_STATE_DIR", str(tmp_path / "supervisor"))


def _executor(mode):
    if mode == "local":
        return LocalExecutor()
    return RemoteEndpointExecutor(
        LocalTransport(),
        command=(sys.executable, "-m", "ase2sprkkr.tools.ase2sprkkr", "remote"),
    )


def _record(executor, runner, run_id, directory):
    execution = Execution(
        endpoint_for_executor(executor),
        WorkingDirectory(
            str(directory),
            (runner.job_files.manifest.mapping.map_path(str(directory))
             if isinstance(executor, RemoteEndpointExecutor)
             else None),
        ),
        runner.backend_id,
    )
    return RunRecord(
        id=run_id, task="SCF", state=RunState.RUNNING,
        created_at="2026-10-21T08:00:00+00:00",
        updated_at="2026-10-21T08:00:00+00:00",
        execution=execution, files=runner.job_files.manifest,
    )


@pytest.mark.parametrize("mode", ("local", "ssh"))
def test_project_round_trip_freezes_same_files(mode, tmp_path):
    directory = tmp_path / "local" / "calculation"
    directory.mkdir(parents=True)
    stdout, stderr = directory / "run.out", directory / "run.err"
    product = directory / "result-01.dat"
    executor = _executor(mode)
    run_id = uuid4().hex
    code = (
        "import pathlib,sys; "
        "pathlib.Path('result-01.dat').write_text('result'); "
        "print('stdout-value', flush=True); "
        "print('stderr-value', file=sys.stderr, flush=True)"
    )
    with stdout.open("w+b") as out, stderr.open("w+b") as err:
        spec = ExecutionSpec(
            argv=(sys.executable, "-c", code), input_file=None,
            output_file=out, error_file=err, directory=str(directory),
            output_files=(str(directory / "result-*.dat"),),
            launch=LaunchSpec(
                local_root=str(tmp_path / "local"),
                remote_root=str(tmp_path / "remote"),
            ),
        )
        runner = executor.create_project_run(spec, run_id=run_id)
        observed = []
        runner.add_output_listener(lambda text, kind: observed.append((kind, text)))
        runner.prepare_submission()
        assert runner.job_files.manifest is not None
        runner.submit()
        assert runner.wait().return_code == 0
        assert runner.state == RunState.FINISHED
        assert ("out", "stdout-value\n") in observed
        assert ("err", "stderr-value\n") in observed

    assert stdout.read_text() == "stdout-value\n"
    assert stderr.read_text() == "stderr-value\n"
    assert product.read_text() == "result"
    assert [Path(e.local_path) for e in runner.job_files.manifest.outputs] == [product]
    assert runner.job_files.manifest.stdout.availability_state == "available"
    assert runner.job_files.manifest.stderr.availability_state == "available"

    # A late file is not part of the supervisor's terminal receipt.
    (directory / "result-late.dat").write_text("not a result")
    record = _record(executor, runner, run_id, directory)
    (status,) = executor.status(record)
    assert status.backend_id == runner.backend_id
    assert status.return_code == 0
    assert len(runner.job_files.manifest.outputs) == 1

    attached = executor.attach(record, start=True)
    assert attached.wait().return_code == 0
    assert attached.state == RunState.FINISHED
    assert [Path(e.local_path) for e in attached.job_files.manifest.outputs] == [product]
    assert stdout.read_text() == "stdout-value\n"
    assert stderr.read_text() == "stderr-value\n"


@pytest.mark.parametrize("mode", ("local", "ssh"))
def test_project_cancel_keeps_terminal_receipt(mode, tmp_path):
    directory = tmp_path / "local" / "calculation"
    directory.mkdir(parents=True)
    executor = _executor(mode)
    run_id = uuid4().hex
    with (directory / "run.out").open("w+b") as out:
        spec = ExecutionSpec(
            argv=(sys.executable, "-c", "import time; time.sleep(30)"),
            input_file=None, output_file=out, directory=str(directory),
            launch=LaunchSpec(
                local_root=str(tmp_path / "local"),
                remote_root=str(tmp_path / "remote"),
            ),
        )
        runner = executor.create_project_run(spec, run_id=run_id)
        runner.submit()
        runner.cancel().result(timeout=10)
        runner.wait()
        assert runner.state == RunState.CANCELLED

    record = _record(executor, runner, run_id, directory)
    (status,) = executor.status(record)
    assert status.state == RunState.CANCELLED
    assert status.backend_id == runner.backend_id
    attached = executor.attach(record, start=False)
    assert attached.state == RunState.CANCELLED


@pytest.mark.parametrize("mode", ("local", "ssh"))
def test_failed_project_can_reconnect_without_changing_backend_result(mode, tmp_path):
    directory = tmp_path / "local" / "calculation"
    directory.mkdir(parents=True)
    executor = _executor(mode)
    run_id = uuid4().hex
    with (directory / "run.out").open("w+b") as out:
        spec = ExecutionSpec(
            argv=(sys.executable, "-c",
                  "import sys; print('before failure', flush=True); sys.exit(7)"),
            input_file=None, output_file=out, directory=str(directory),
            launch=LaunchSpec(
                local_root=str(tmp_path / "local"),
                remote_root=str(tmp_path / "remote"),
            ),
        )
        runner = executor.create_project_run(spec, run_id=run_id)
        runner.submit()
        assert runner.wait().return_code == 7
        assert runner.state == RunState.FAILED

    record = _record(executor, runner, run_id, directory)
    (status,) = executor.status(record)
    assert status.backend_id == runner.backend_id
    assert status.return_code == 7

    attached = executor.attach(record, start=True)
    assert attached.wait().return_code == 7
    assert attached.state == RunState.FAILED
    assert (directory / "run.out").read_text() == "before failure\n"
