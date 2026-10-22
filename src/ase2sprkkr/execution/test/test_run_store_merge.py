from ase2sprkkr.execution import (
    
    ExecutorEndpoint,
    RunState,
    RunStore,
)

from ase2sprkkr.execution.manager import ExecutionManager
import pytest


def _record(manager, tmp_path):
    return manager.new_record(
        task="DOS",
        directory=tmp_path,
        endpoint=ExecutorEndpoint(),
    )


def test_stale_transition_cannot_regress_newer_lifecycle_state(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    submitting = manager.transition(original, state=RunState.SUBMITTING)
    running = manager.transition(
        submitting,
        state=RunState.RUNNING,
        execution=submitting.execution.submitted("123"),
    )

    merged = manager.transition(
        submitting,
        state=RunState.QUEUED,
    )

    assert merged.state == RunState.RUNNING
    assert merged.execution.backend_id == "123"
    assert manager.store.get(original.id) == merged
    assert running.execution.backend_id == "123"


def test_none_does_not_erase_known_completion_metadata(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    stale = manager.transition(original, state=RunState.RUNNING)
    finished = manager.transition(
        stale,
        state=RunState.FINISHED,
        return_code=0,
        message="complete",
    )

    merged = manager.transition(
        stale,
        state=RunState.RUNNING,
        return_code=None,
        message=None,
    )

    assert merged.state == RunState.FINISHED
    assert merged.return_code == 0
    assert merged.message == "complete"
    assert manager.store.get(original.id) == merged
    assert finished.state == RunState.FINISHED


def test_late_conflicting_final_state_does_not_replace_known_final_outcome(tmp_path):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    running = manager.transition(original, state=RunState.RUNNING)
    manager.transition(
        running,
        state=RunState.FINISHED,
        return_code=0,
    )

    merged = manager.transition(
        running,
        state=RunState.FAILED,
        return_code=1,
    )

    assert merged.state == RunState.FINISHED
    # A stale conflicting final update must not overwrite the precise outcome.
    assert merged.return_code == 0


def test_late_uncaptured_manifest_cannot_destroy_historical_snapshot(tmp_path):
    from ase2sprkkr.execution import FileEntry, FileManifest

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    output = tmp_path / "out"
    output.write_bytes(b"historical contents")
    original = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(stdout=FileEntry(local_path=str(output))),
    )
    running = manager.transition(original, state=RunState.RUNNING)
    completed = manager.complete(
        running, state=RunState.FINISHED, return_code=0,
        files_available=True,
    )
    assert completed.files.captured_at is not None
    assert len(completed.files.output_versions) == 1

    old_manifest = running.files
    late = manager.transition(
        running, state=RunState.RUNNING, files=old_manifest,
    )
    assert late.files == completed.files
    assert late.files.output_versions == completed.files.output_versions
    assert manager.store.get(running.id).files == completed.files


def test_submitted_identity_rejects_directory_and_endpoint_changes(tmp_path):
    from dataclasses import replace
    import pytest
    from ase2sprkkr.execution import WorkingDirectory

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    manager.store.save(original)
    staged = manager.transition(
        original,
        execution=replace(
            original.execution,
            directory=WorkingDirectory(str(tmp_path), "/remote/actual"),
        ),
    )
    running = manager.transition(
        staged, state=RunState.RUNNING,
        execution=staged.execution.submitted("job-1"),
    )
    assert running.execution.directory.remote == "/remote/actual"
    for changes in (
        {"directory": WorkingDirectory(str(tmp_path), "/remote/other")},
        {"directory": WorkingDirectory(str(tmp_path / "other"), "/remote/actual")},
        {"endpoint": ExecutorEndpoint(executor="psij", psij_backend="slurm")},
        {"backend_id": "job-2"},
    ):
        with pytest.raises(ValueError):
            manager.transition(
                running, execution=replace(running.execution, **changes),
            )
    assert manager.store.get(running.id).execution == running.execution


def test_terminal_refresh_cannot_replace_manifest_or_execution(tmp_path):
    from dataclasses import replace
    import pytest
    from ase2sprkkr.execution import FileEntry, FileManifest, WorkingDirectory

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(outputs=(FileEntry(local_path=str(tmp_path / "a")),)),
    )
    running = manager.transition(original, state=RunState.RUNNING)
    finished = manager.transition(running, state=RunState.FINISHED)
    stale_files = FileManifest(outputs=(
        FileEntry(local_path=str(tmp_path / "different")),
    ))
    unchanged = manager.transition(running, state=RunState.RUNNING, files=stale_files)
    assert unchanged.files == finished.files
    with pytest.raises(ValueError, match="directory"):
        manager.transition(
            running,
            execution=replace(
                finished.execution,
                directory=WorkingDirectory(str(tmp_path / "other")),
            ),
        )
    assert manager.store.get(original.id).execution == finished.execution


def test_terminal_persistent_run_state_does_not_query_backend(tmp_path):
    from ase2sprkkr.execution.run_persistence import RunPersistence
    from ase2sprkkr.execution.postprocessing import CalculationRun
    from ase2sprkkr.execution import JobFiles, FileManifest

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = _record(manager, tmp_path)
    finished = manager.transition(record, state=RunState.FINISHED)

    class Backend:
        callback = None
        job_files = JobFiles(FileManifest())

        @property
        def state(self):
            raise AssertionError("Terminal state must not query backend")

    run = CalculationRun(Backend(), lambda outcome: outcome)
    run.persistence = RunPersistence(manager, object(), finished)
    assert run.state is RunState.FINISHED


def test_explicit_availability_update_preserves_terminal_snapshot(tmp_path):
    """A terminal run rejects unsolicited manifests, not explicit file status."""
    from ase2sprkkr.execution import FileEntry, FileManifest

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    output = str(tmp_path / "optional.out")
    original = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(outputs=(FileEntry(local_path=output),)),
    )
    terminal = manager.transition(original, state=RunState.FINISHED)
    from dataclasses import replace

    # A completed run permits explicit status amendments through RunStore,
    # independently of the mutable working JobFiles object.
    updated = manager.store.update(
        terminal,
        lambda latest: replace(
            latest,
            files=latest.files.with_file_availability(output, "skipped"),
        ),
    )

    assert updated.state is RunState.FINISHED
    assert updated.finished_at == terminal.finished_at
    assert updated.files.outputs[0].local_path == output
    assert updated.files.outputs[0].availability_state == "skipped"
    assert manager.store.get(original.id).files == updated.files


def test_terminal_run_cannot_acquire_backend_identity_after_failure(tmp_path):
    """A failed submission with no backend ID cannot be reassigned later."""
    from dataclasses import replace
    import pytest

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    failed = manager.transition(
        _record(manager, tmp_path), state=RunState.FAILED,
    )
    with pytest.raises(ValueError, match="Final execution identity"):
        manager.transition(
            failed,
            execution=replace(failed.execution, backend_id="late-job"),
        )
    assert manager.store.get(failed.id).execution.backend_id is None


def test_endpoint_and_local_directory_are_immutable_before_submission(tmp_path):
    from dataclasses import replace
    import pytest
    from ase2sprkkr.execution import WorkingDirectory

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    manager.store.save(original)
    for changes in (
        {"endpoint": ExecutorEndpoint(executor="psij", psij_backend="slurm")},
        {"directory": WorkingDirectory(str(tmp_path / "other"))},
    ):
        with pytest.raises(ValueError):
            manager.transition(
                original, execution=replace(original.execution, **changes),
            )
    assert manager.store.get(original.id).execution == original.execution


def test_concurrent_completers_reuse_one_durable_terminal_snapshot(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = _record(manager, tmp_path)
    running = manager.transition(record, state=RunState.RUNNING)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(manager.complete, running, state=RunState.FINISHED,
                        return_code=0, files_available=False)
            for _ in range(2)
        ]
        first, second = (future.result() for future in futures)

    assert first == second == manager.store.get(record.id)
    assert first.state is RunState.FINISHED
    assert first.finished_at is not None


def test_terminal_retry_after_lost_fsync_confirmation(tmp_path, monkeypatch):

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = _record(manager, tmp_path)
    running = manager.transition(record, state=RunState.RUNNING)
    actual_write = manager.store._write_unlocked
    calls = []

    def lost_confirmation(value):
        calls.append("write")
        result = actual_write(value)
        if len(calls) == 1:
            raise OSError("lost fsync confirmation")
        return result

    monkeypatch.setattr(manager.store, "_write_unlocked", lost_confirmation)
    with pytest.raises(OSError, match="lost fsync confirmation"):
        manager.complete(running, state=RunState.FINISHED, return_code=0)
    confirmed = manager.complete(running, state=RunState.FINISHED, return_code=0)
    assert confirmed == manager.store.get(record.id)
    assert confirmed.state is RunState.FINISHED
    assert calls == ["write"]


def test_terminal_completion_preserves_first_file_snapshot(tmp_path):
    from dataclasses import replace
    from ase2sprkkr.execution import FileEntry, FileManifest

    output = tmp_path / "output.out"
    output.write_text("durable output")
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(stdout=FileEntry(local_path=str(output))),
    )
    running = manager.transition(original, state=RunState.RUNNING)
    first = manager.complete(
        running, state=RunState.FINISHED, return_code=0,
        files_available=True,
    )
    assert first.files.captured_at is not None
    assert first.files.output_versions

    # An observer with an older manifest and failed transfer diagnostics
    # must not invalidate the first durably committed result.
    stale = replace(
        running, files=FileManifest(stdout=FileEntry(
            local_path=str(output), availability_state="failed",
            availability_message="observer disconnected",
        )),
    )
    second = manager.complete(
        stale, state=RunState.FAILED, return_code=1, files_available=False,
        files=stale.files,
    )
    assert second == first == manager.store.get(original.id)
    assert second.files.output_versions == first.files.output_versions


def test_terminal_lookup_rejects_foreign_project_identity(tmp_path):
    from dataclasses import replace

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    final = manager.complete(original, state=RunState.FINISHED)
    with pytest.raises(ValueError, match="persistent project identity"):
        manager.complete(
            replace(original, project_id="different-project"),
            state=RunState.FINISHED,
        )
    assert manager.store.get(original.id) == final




def test_save_cannot_overwrite_terminal_history(tmp_path):
    from dataclasses import replace

    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    original = _record(manager, tmp_path)
    store.save(original)
    completed = manager.complete(original, state=RunState.FINISHED, return_code=0)

    for stale in (original, replace(completed, return_code=7)):
        with pytest.raises(ValueError, match="Cannot overwrite a terminal run"):
            store.save(stale)
        assert store.get(original.id) == completed

    assert store.save(completed) == completed


def test_non_project_run_identity_is_immutable_across_day_shards(tmp_path):
    from dataclasses import replace

    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    timestamp = "2030-01-01T10:00:00+00:00"
    original = replace(
        _record(manager, tmp_path), created_at=timestamp, updated_at=timestamp,
    )
    store.save(original)
    forged = replace(
        original, created_at="2030-01-02T10:00:00+00:00",
        updated_at="2030-01-02T10:00:00+00:00",
    )

    with pytest.raises(ValueError, match="persistent creation time"):
        store.save(forged)
    with pytest.raises(ValueError, match="persistent creation time"):
        store.update(forged, lambda latest: latest)
    assert store.get(original.id) == original
    assert store.runs() == (original,)


def test_non_project_store_rejects_duplicate_ids_across_day_shards(tmp_path):
    from dataclasses import replace

    store = RunStore(tmp_path / "runs")
    manager = ExecutionManager(store)
    original = replace(
        _record(manager, tmp_path), created_at="2030-01-01T10:00:00+00:00",
    )
    duplicate = replace(
        original, created_at="2030-01-02T10:00:00+00:00",
    )
    store.save(original)
    store._write_unlocked(duplicate)  # Simulate pre-existing corrupt history.

    for operation in (
        lambda: store.get(original.id),
        lambda: store.runs(),
        lambda: store.runs(limit=1),
        lambda: store.save(original),
        lambda: store.update(original, lambda latest: latest),
    ):
        with pytest.raises(ValueError, match="Duplicate run ID"):
            operation()


def test_file_revalidation_releases_store_lock_and_retries(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from ase2sprkkr.common.file_versions import FileVersion
    from ase2sprkkr.execution.run_model import FileEntry, FileManifest

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = _record(manager, tmp_path)
    output = tmp_path / "out.dat"
    output.write_bytes(b"good")
    version = FileVersion.capture(output, require_digest=True)
    current_files = FileManifest(outputs=(FileEntry(
        str(output), "/remote/out.dat", availability_state="available",
        verified_local_version=version,
    ),))
    running = manager.transition(record, state=RunState.RUNNING, files=current_files)
    failed = current_files.with_file_availability(
        str(output), "failed", "later failed transfer",
    )
    verifying = Event()
    release = Event()
    calls = []

    def slow_validation(self):
        calls.append(self)
        verifying.set()
        assert release.wait(timeout=10)
        return True

    monkeypatch.setattr(FileVersion, "is_current", slow_validation)
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(manager.transition, running, files=failed)
        try:
            assert verifying.wait(timeout=5)
            # This transition must not wait for a SHA-256 operation in the
            # other thread, even when updating the same persistent run.
            fast = pool.submit(
                manager.transition, running, state=RunState.RUNNING,
                files=current_files,
            )
            fast_result = fast.result(timeout=5)
        finally:
            release.set()
        slow_result = slow.result(timeout=10)

    assert len(calls) >= 1
    assert fast_result.state is RunState.RUNNING
    assert slow_result.files.outputs[0].availability_state == "available"
    assert manager.store.get(record.id).files.outputs[0].availability_state == "available"


def test_optimistic_noop_does_not_rewrite_terminal_record(tmp_path, monkeypatch):
    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    original = _record(manager, tmp_path)
    completed = manager.transition(original, state=RunState.FINISHED)
    writes = []

    def forbidden(record):
        writes.append(record)
        raise AssertionError("An ignored terminal observation must not be rewritten")

    monkeypatch.setattr(manager.store, "_write_unlocked", forbidden)
    observed = manager.transition(
        completed, state=RunState.RUNNING,
        files=completed.files, observed=True,
    )
    assert observed == completed
    assert not writes
