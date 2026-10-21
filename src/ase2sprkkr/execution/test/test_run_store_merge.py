from ase2sprkkr.execution import (
    
    ExecutorEndpoint,
    RunState,
    RunStore,
)

from ase2sprkkr.execution.manager import ExecutionManager


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
    from ase2sprkkr.execution.manager import PersistentRun
    from ase2sprkkr.execution.postprocessing import CalculationRun

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    record = _record(manager, tmp_path)
    finished = manager.transition(record, state=RunState.FINISHED)

    class Backend:
        callback = None

        @property
        def state(self):
            raise AssertionError("Terminal state must not query backend")

    run = CalculationRun(Backend(), lambda outcome: outcome)
    run.hooks = PersistentRun(manager, object(), finished)
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
    updated = manager.update_file_availability(terminal, output, "skipped")

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
