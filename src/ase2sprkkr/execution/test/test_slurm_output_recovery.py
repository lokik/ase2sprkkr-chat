"""Recovery guesses are deliberately weaker than backend identity."""

import asyncio
import os
import time
from datetime import datetime, timezone

from ase2sprkkr.execution import FileEntry, FileManifest, RunState, RunStore
from ase2sprkkr.execution.base import SubmissionRecovery
from ase2sprkkr.execution.file_availability import probe_submission_outputs
from ase2sprkkr.execution.manager import ExecutionManager
from ase2sprkkr.execution.pool import Executors
from ase2sprkkr.execution.run_record import SubmissionPhase
from ase2sprkkr.execution.transport import LocalTransport


def _probe(manifest, submitted_at):
    return asyncio.run(probe_submission_outputs(manifest, LocalTransport(), submitted_at))


def test_stdout_only_counts_as_fresh_output(tmp_path):
    stdout = tmp_path / "only.out"
    manifest = FileManifest(stdout=FileEntry(local_path=str(stdout)))
    before = datetime.now(timezone.utc).isoformat()
    assert _probe(manifest, before) is False
    stdout.write_bytes(b"")
    assert _probe(manifest, before) is True


def test_old_and_partial_outputs_are_inconclusive(tmp_path):
    stdout = tmp_path / "only.out"
    other = tmp_path / "result.dat"
    manifest = FileManifest(
        stdout=FileEntry(local_path=str(stdout)),
        outputs=(FileEntry(local_path=str(other)),),
    )
    before = datetime.now(timezone.utc).isoformat()
    stdout.write_bytes(b"hello")
    assert _probe(manifest, before) is None
    other.write_bytes(b"data")
    assert _probe(manifest, before) is True
    os.utime(other, ns=(1, 1))
    assert _probe(manifest, before) is None


def test_untracked_streams_and_empty_manifest_are_not_evidence(tmp_path):
    before = datetime.now(timezone.utc).isoformat()
    assert _probe(FileManifest(), before) is None


class _NotFoundExecutor:
    name = "test"

    def __init__(self, evidence):
        self.evidence = evidence

    def recover_submission(self, *args, **kwargs):
        return SubmissionRecovery.NOT_FOUND

    def submission_output_evidence(self, *args, **kwargs):
        return self.evidence


def _dispatched(manager, executor, tmp_path):
    endpoint = manager.executors.register(executor)
    record = manager.new_record(task="SCF", directory=tmp_path, endpoint=endpoint)
    return manager.store.begin_dispatch(manager.reserve(record))


def test_new_outputs_keep_reservation_and_dispatch_timestamp(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), executors=Executors(),
                               submitting_grace=0)
    run = _dispatched(manager, _NotFoundExecutor(True), tmp_path)
    recovered = manager.recover_submitting(run)
    assert recovered.state == RunState.SUBMITTING
    assert recovered.submission_phase == SubmissionPhase.DISPATCHING
    assert recovered.updated_at == run.updated_at
    assert not recovered.confirmed
    assert "output" in recovered.error
    assert manager.recover_submitting(recovered) == recovered


def test_no_outputs_timeout_allows_retry_without_claiming_proof(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), executors=Executors(),
                               submitting_grace=0)
    run = _dispatched(manager, _NotFoundExecutor(False), tmp_path)
    recovered = manager.recover_submitting(run)
    assert recovered.state == RunState.FAILED
    assert recovered.execution.backend_id is None
    assert not recovered.confirmed
    assert "not proven" in recovered.message
    assert manager.reserve(manager.new_record(task="SCF", directory=tmp_path,
        endpoint=run.execution.endpoint)).state == RunState.SUBMITTING


def test_partial_outputs_never_release_reservation(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"), executors=Executors(),
                               submitting_grace=0)
    run = _dispatched(manager, _NotFoundExecutor(None), tmp_path)
    assert manager.recover_submitting(run) == run


def test_remote_probe_does_not_consult_local_files(tmp_path):
    from ase2sprkkr.execution.run_model import FileMapping
    class RemoteTransport:
        async def file_signature(self, path):
            assert path == "/cluster/out"
            return (0, time.time_ns() + 1_000_000_000)
    manifest = FileManifest(
        stdout=FileEntry(local_path=str(tmp_path / "out"), remote_path="/cluster/out"),
        mapping=FileMapping(str(tmp_path), "/cluster"),
    )
    before = datetime.now(timezone.utc).isoformat()
    assert asyncio.run(probe_submission_outputs(
        manifest, RemoteTransport(), before, remote=True,
    )) is True
