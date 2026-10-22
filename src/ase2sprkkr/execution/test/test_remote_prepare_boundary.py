"""Crash boundary: remote staging metadata must be persisted before submit."""

import asyncio

import pytest

from ase2sprkkr.execution.remote import RemoteRunner
from ase2sprkkr.execution import (
    ExecutionSpec, FileManifest,
    LaunchSpec, RemoteEndpointExecutor, RunState, RunStore, LocalTransport,
)

from ase2sprkkr.execution.manager import ExecutionManager


def test_prepared_manifest_is_durable_before_remote_backend_submit(tmp_path, monkeypatch):
    executor = RemoteEndpointExecutor(LocalTransport())
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    directory = tmp_path / "local"
    directory.mkdir()
    output = directory / "run.out"
    spec = ExecutionSpec(
        argv=("ignored",),
        directory=str(directory),
        input_file=None,
        output_file=str(output),
        launch=LaunchSpec(
            local_root=str(tmp_path),
            remote_root=str(tmp_path / "target"),
        ),
    )
    run = manager.create_run(
        spec, endpoint=manager.executors.register(executor),
        task="DOS", process_result=lambda outcome: outcome,
    )
    # Match the mapping actually reserved by PersistentRun.before_submit().
    # The fake prepare step must not invent another remote file namespace.
    expected, remote_directory = executor.reservation_context(spec, run.record.files)

    async def prepare(_runner):
        return {"directory": remote_directory, "id": run.id}, expected

    submitted = []

    async def submit(_request):
        current = manager.store.get(run.id)
        submitted.append(current)
        return {"state": "running", "remote_id": "opaque-job"}

    monkeypatch.setattr(RemoteRunner, "_prepare_async", prepare)
    monkeypatch.setattr("ase2sprkkr.execution.remote.start_monitor", lambda *args, **kwargs: None)
    executor.connection.submit = submit

    async def status(_backend_id, _files=None):
        return {"state": "running"}

    # The test substitutes an opaque backend id and disables the monitor;
    # supply the corresponding fake status instead of invoking the CLI.
    executor.connection.status = status
    run.submit()

    assert len(submitted) == 1
    snapshot = submitted[0]
    assert snapshot.state == RunState.SUBMITTING
    from ase2sprkkr.execution.run_record import SubmissionPhase
    assert snapshot.submission_phase == SubmissionPhase.DISPATCHING
    assert snapshot.execution.backend_id is None
    assert snapshot.execution.directory.remote == remote_directory
    assert snapshot.files == expected
    assert manager.store.get(run.id).execution.backend_id == "opaque-job"


def test_remote_manifest_uses_same_declarations_as_local(tmp_path):
    directory = tmp_path / "local"
    directory.mkdir(exist_ok=True)
    remote = tmp_path / "target"
    spec = ExecutionSpec(
        argv=("ignored",), directory=str(directory),
        input_file="main.inp", input_files=("extra.inp",),
        output_file="main.out", error_file="main.err",
        output_files=("one.dat", "two.dat"),
        cache_files=("restart.pot",),
        launch=LaunchSpec(local_root=str(tmp_path), remote_root=str(remote)),
    )
    (directory / "main.inp").write_text("main")
    (directory / "extra.inp").write_text("extra")
    local = FileManifest.from_spec(spec)
    executor = RemoteEndpointExecutor(LocalTransport())
    runner = executor.create_run(spec)
    async def prepare_files():
        mapping = await executor.resolve_mapping(runner.spec.launch)
        return await runner.file_handler.prepare(mapping, runner.spec)

    manifest = asyncio.run(prepare_files())
    assert tuple(e.local_path for e in manifest.inputs) == (
        str(directory / "main.inp"), str(directory / "extra.inp"),
    )
    assert tuple(e.local_path for e in manifest.inputs) == local.input_files
    assert tuple(e.remote_path for e in manifest.inputs) == (
        str(remote / "local" / "main.inp"),
        str(remote / "local" / "extra.inp"),
    )
    assert tuple(e.local_path for e in manifest.outputs) == tuple(e.local_path for e in local.outputs)
    assert tuple(e.local_path for e in manifest.outputs) == (
        str(directory / "one.dat"), str(directory / "two.dat"),
    )
    assert manifest.output_files == local.output_files
    assert manifest.stderr.remote_path == str(remote / "local" / "main.err")
    assert manifest.stdout.remote_path == str(remote / "local" / "main.out")
    assert manifest.outputs[0].remote_path == str(remote / "local" / "one.dat")
    assert manifest.caches[0].remote_path == str(remote / "local" / "restart.pot")
    assert len(manifest.caches) == 1


def test_remote_file_preparation_rejects_duplicate_paths_before_mapping(tmp_path):
    directory = tmp_path / "work"
    directory.mkdir()
    spec = ExecutionSpec(
        argv=("ignored",), directory=str(directory),
        input_file=None, output_file="stdout.out",
        output_files=("one.dat", "./one.dat"),
        launch=LaunchSpec(local_root=str(tmp_path),
                          remote_root=str(tmp_path / "remote")),
    )
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest.from_spec(spec)


@pytest.mark.parametrize("method", ("submit", "start"))
@pytest.mark.parametrize("rejection_phase", ("prepare", "submit"))
def test_postprocessing_rejected_submit_preserves_error_and_does_not_monitor(
    method, rejection_phase,
):
    from ase2sprkkr.execution.base import FailurePhase, SubmissionNotStarted
    from ase2sprkkr.execution.postprocessing import CalculationRun

    failure = SubmissionNotStarted("preflight refused")

    class RejectedRunner:
        state = RunState.NEW
        callback = None

        def prepare_submission(self):
            if rejection_phase == "prepare":
                raise failure

        def submit(self):
            if rejection_phase == "submit":
                raise failure
            pytest.fail("Backend must not start after rejected preparation")

        def wait(self):
            pytest.fail("Rejected submission must not call backend wait")

    runner = CalculationRun(
        RejectedRunner(),
        lambda result: pytest.fail("Rejected submission must not be parsed"),
        callback=lambda result: pytest.fail("Rejected submission has no completion callback"),
    )
    with pytest.raises(SubmissionNotStarted, match="preflight refused") as caught:
        getattr(runner, method)()
    assert caught.value is failure
    assert runner._started is False
    assert runner._settled is True
    assert runner._done.is_set()
    assert runner.state is RunState.FAILED
    assert runner.failure_phase is FailurePhase.SUBMISSION
    with pytest.raises(SubmissionNotStarted, match="preflight refused") as waited:
        runner.wait()
    assert waited.value is failure
