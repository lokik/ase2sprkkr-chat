from types import SimpleNamespace

import pytest

from ase2sprkkr.execution import (
    FileAvailabilityState,
    ExecutionSpec,
    Executor,
    FileState,
    FileManifest,
    RunState,
    RunStore,
)
from ase2sprkkr.execution.endpoint import endpoint_for_executor
from ase2sprkkr.execution.manager import ExecutionManager, HistoricalFileMismatchError
from ase2sprkkr.outputs import project_results
from ase2sprkkr.outputs.project_results import (
    ProjectResultUnavailableError,
    load_project_result,
)


class _HistoryExecutor(Executor):
    name = "history"

    def __init__(self):
        self.fetch_calls = []
        self.output_file = None
        self.restored_output = "original output"

    def fetch_result_files(self, run, files, *, destinations=None):
        files = tuple(files)
        self.fetch_calls.append((run.id, files))
        if self.output_file is None or self.output_file not in files:
            return
        target = self.output_file if destinations is None else destinations[self.output_file]
        with open(target, "w") as output:
            output.write(self.restored_output)


def _spec(tmp_path):
    input_file = tmp_path / "run.inp"
    output_file = tmp_path / "run.out"
    input_file.write_text("input")
    return ExecutionSpec(
        argv=(),
        directory=str(tmp_path),
        input_file=str(input_file),
        output_file=str(output_file),
        input_files=(str(input_file),),
    )


def _manager(store):
    return ExecutionManager(store)


def _terminal_run(tmp_path, *, state=RunState.FINISHED, return_code=0, task="DOS"):
    store = RunStore(tmp_path / "runs")
    manager = _manager(store)
    executor = _HistoryExecutor()
    spec = _spec(tmp_path)
    output_file = tmp_path / "run.out"
    output_file.write_text("original output")
    executor.output_file = str(output_file)

    endpoint = endpoint_for_executor(executor)
    manager.executors.register(executor, endpoint)
    record = manager.new_record(
        task=task,
        directory=tmp_path,
        endpoint=endpoint,
        files=FileManifest.from_spec(spec),
    )
    record = manager.store.save(record)
    record = manager.complete(
        record,
        state=state,
        execution=record.execution.submitted("job-1"),
        return_code=return_code,
        files_available=True,
    )
    return manager, executor, record


def _patch_reader(monkeypatch, parsed):
    class Reader:
        input_parameters = object()

        def read_from_file(self, output, **kwargs):
            parsed.append((output, kwargs))
            return SimpleNamespace(return_code=kwargs["return_code"])

    monkeypatch.setattr(
        project_results,
        "_reader_for_record",
        lambda manager, record: Reader(),
    )


def test_history_listing_is_lazy_but_loading_restores_missing_output(
    tmp_path, monkeypatch,
):
    manager, executor, record = _terminal_run(tmp_path)
    output_path = tmp_path / "run.out"
    output_path.unlink()

    # Merely enumerating stored history must not retrieve files.
    records = manager.store.runs()
    assert records[0].id == record.id
    assert executor.fetch_calls == []
    assert not output_path.exists()

    # Loading that very same record must retrieve and parse the exact output.
    parsed = []
    _patch_reader(monkeypatch, parsed)
    result = load_project_result(manager, record.id)

    assert executor.fetch_calls == [
        (record.id, (str(output_path),))
    ]
    assert output_path.read_text() == "original output"
    assert parsed[0][0] == str(output_path)
    assert result.return_code == 0
    assert result.run_record.state == RunState.FINISHED
    assert result.run_record.files.state == FileState.CURRENT
    assert (
        result.run_record.files.availability_state
        == FileAvailabilityState.AVAILABLE
    )


@pytest.mark.parametrize(
    "entrypoint, expected_error",
    [
        pytest.param("load", ProjectResultUnavailableError, id="load-project-result"),
        pytest.param("retrieve", HistoricalFileMismatchError, id="retrieve-files"),
    ],
)
def test_restored_changed_content_is_rejected_as_overwritten(
    tmp_path, monkeypatch, entrypoint, expected_error,
):
    manager, executor, record = _terminal_run(tmp_path)
    path = tmp_path / "run.out"
    path.unlink()
    executor.restored_output = "changed output with different contents"

    if entrypoint == "load":
        class Reader:
            input_parameters = object()

            def read_from_file(self, *args, **kwargs):
                raise AssertionError("overwritten historical output must not be parsed")

        monkeypatch.setattr(
            project_results, "_reader_for_record",
            lambda manager, record: Reader(),
        )

    with pytest.raises(expected_error, match="overwritten") as error:
        if entrypoint == "load":
            load_project_result(manager, record.id)
        else:
            manager.retrieve_result_files(record.id)

    stored = manager.store.get(record.id)
    assert stored.state == RunState.FINISHED
    assert not path.exists()
    assert not list(tmp_path.glob(".run.out.*.tmp"))
    if entrypoint == "load":
        assert error.value.record.files.availability_state == FileAvailabilityState.FAILED
        assert stored.files.state == FileState.MISSING
        assert stored.files.availability_state == FileAvailabilityState.FAILED


def test_retrieval_programming_error_is_not_persisted_as_file_failure(tmp_path):
    manager, executor, record = _terminal_run(tmp_path)
    (tmp_path / "run.out").unlink()

    def broken_fetch(reference, files, *, destinations=None):
        raise KeyError("programming error")

    executor.fetch_result_files = broken_fetch

    with pytest.raises(KeyError, match="programming error"):
        manager.retrieve_result_files(record.id)

    stored = manager.store.get(record.id)
    assert (
        stored.files.availability_state
        == FileAvailabilityState.AVAILABLE
    )


def test_finished_history_without_available_output_is_not_synthesized(tmp_path):
    store = RunStore(tmp_path / "runs")
    manager = _manager(store)
    spec = _spec(tmp_path)
    record = manager.new_record(
        task="DOS",
        directory=tmp_path,
        files=FileManifest.from_spec(spec),
    )
    record = manager.store.save(record)
    record = manager.complete(
        record,
        state=RunState.FINISHED,
        return_code=0,
        files_available=False,
    )

    with pytest.raises(ProjectResultUnavailableError, match="unavailable"):
        load_project_result(manager, record.id)


def test_failed_history_without_return_code_is_not_parsed_as_success(
    tmp_path, monkeypatch
):
    manager, _, record = _terminal_run(
        tmp_path,
        state=RunState.FAILED,
        return_code=None,
    )
    parsed = []
    _patch_reader(monkeypatch, parsed)

    result = load_project_result(manager, record.id)

    assert parsed[0][1]["return_code"] != 0
    assert result.return_code != 0
    assert result.run_record.state == RunState.FAILED


def test_history_load_uses_retrieval_record_without_extra_store_lookup(
    tmp_path, monkeypatch,
):
    manager, _, record = _terminal_run(tmp_path)
    parsed = []
    _patch_reader(monkeypatch, parsed)
    original_get = manager.store.get
    retrieved = []

    def get(run_id):
        retrieved.append(run_id)
        return original_get(run_id)

    monkeypatch.setattr(manager.store, "get", get)
    loaded = load_project_result(manager, record.id)

    assert loaded.run_record.state is RunState.FINISHED
    assert len(parsed) == 1
    # Read initial record, perform retrieval, verify output, verify input.
    # The removed check_result_files() used to load the same record again.
    assert retrieved == [record.id] * 4


def test_historical_missing_input_restores_without_output_availability_update(tmp_path):
    manager, executor, record = _terminal_run(tmp_path)
    input_path = tmp_path / "run.inp"
    input_path.unlink()
    original_fetch = executor.fetch_result_files

    def fetch(reference, files, *, destinations=None):
        if str(input_path) in files:
            target = (str(input_path) if destinations is None
                      else destinations[str(input_path)])
            from pathlib import Path
            Path(target).write_text("input")
        else:
            original_fetch(reference, files, destinations=destinations)

    executor.fetch_result_files = fetch
    restored = manager.retrieve_result_files(record.id)
    assert input_path.read_text() == "input"
    assert restored.id == record.id
    assert restored.state == record.state


def test_historical_missing_input_reports_unrestored_file(tmp_path):
    manager, executor, record = _terminal_run(tmp_path)
    input_path = tmp_path / "run.inp"
    input_path.unlink()
    executor.fetch_result_files = lambda reference, files, *, destinations=None: None

    with pytest.raises(FileNotFoundError, match="Historical file was not retrieved"):
        manager.retrieve_result_files(record.id)
    assert manager.store.get(record.id).state == record.state


def test_simultaneous_historical_restores_download_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    manager, executor, record = _terminal_run(tmp_path)
    output_path = tmp_path / "run.out"
    output_path.unlink()
    original_fetch = executor.fetch_result_files
    access = Lock()
    calls = []

    def fetch(run, files, *, destinations=None):
        with access:
            calls.append(tuple(files))
        return original_fetch(run, files, destinations=destinations)

    executor.fetch_result_files = fetch
    start = Barrier(2)

    def restore():
        start.wait()
        return manager.retrieve_result_files(record.id)

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: restore(), range(2)))

    assert len(calls) == 1
    assert output_path.read_text() == "original output"
    assert all(value.state is RunState.FINISHED for value in results)
    stored = manager.store.get(record.id)
    assert stored.files.availability_state == FileAvailabilityState.AVAILABLE


def test_uncaptured_missing_output_after_fetch_raises_and_marks_failed(tmp_path):
    """A no-op backend retrieval is never considered a successful restore."""
    from ase2sprkkr.execution.run_model import FileEntry

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    executor = _HistoryExecutor()
    endpoint = endpoint_for_executor(executor)
    manager.executors.register(executor, endpoint)
    output = tmp_path / "missing.out"
    record = manager.new_record(
        task="DOS", directory=tmp_path, endpoint=endpoint,
        files=FileManifest(outputs=(
            FileEntry(local_path=str(output), availability_state="pending"),
        )),
    )
    record = manager.transition(
        record, state=RunState.FINISHED,
        execution=record.execution.submitted("job-1"),
    )
    with pytest.raises(FileNotFoundError, match="Historical file was not retrieved"):
        manager.retrieve_result_files(record.id)
    saved = manager.store.get(record.id)
    assert saved.state is RunState.FINISHED
    assert saved.files.outputs[0].availability_state == "failed"
    assert executor.fetch_calls == [(record.id, (str(output),))]


def test_uncaptured_restore_interruption_never_publishes_partial_output(tmp_path):
    from ase2sprkkr.execution.run_model import FileEntry

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    executor = _HistoryExecutor()
    endpoint = endpoint_for_executor(executor)
    manager.executors.register(executor, endpoint)
    output = tmp_path / "missing.out"
    record = manager.new_record(
        task="DOS", directory=tmp_path, endpoint=endpoint,
        files=FileManifest(outputs=(
            FileEntry(local_path=str(output), availability_state="pending"),
        )),
    )
    record = manager.transition(
        record, state=RunState.FINISHED,
        execution=record.execution.submitted("job-1"),
    )

    def interrupted_fetch(reference, files, *, destinations=None):
        from pathlib import Path
        assert destinations is not None
        Path(destinations[str(output)]).write_bytes(b"incomplete data")
        raise OSError("connection interrupted")

    executor.fetch_result_files = interrupted_fetch
    with pytest.raises(OSError, match="connection interrupted"):
        manager.retrieve_result_files(record.id)
    assert not output.exists()
    assert list(tmp_path.glob(".missing.out.*.tmp")) == []
    assert manager.store.get(record.id).files.outputs[0].availability_state == "failed"
    assert manager.store.get(record.id).state is RunState.FINISHED


def test_uncaptured_restore_publishes_complete_file_atomically(tmp_path):
    from pathlib import Path
    from ase2sprkkr.execution.run_model import FileEntry

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    executor = _HistoryExecutor()
    endpoint = endpoint_for_executor(executor)
    manager.executors.register(executor, endpoint)
    output = tmp_path / "missing.out"
    record = manager.new_record(
        task="DOS", directory=tmp_path, endpoint=endpoint,
        files=FileManifest(outputs=(FileEntry(
            local_path=str(output), availability_state="pending",
        ),)),
    )
    record = manager.transition(
        record, state=RunState.FINISHED,
        execution=record.execution.submitted("job-1"),
    )

    def fetch(reference, files, *, destinations=None):
        assert destinations is not None
        Path(destinations[str(output)]).write_bytes(b"complete data")

    executor.fetch_result_files = fetch
    retrieved = manager.retrieve_result_files(record.id)
    assert output.read_bytes() == b"complete data"
    assert retrieved.files.outputs[0].availability_state == "available"
    assert list(tmp_path.glob(".missing.out.*.tmp")) == []


@pytest.mark.parametrize("matching", [False, True])
def test_external_writer_cannot_be_overwritten_during_historical_restore(
    tmp_path, monkeypatch, matching,
):
    import ase2sprkkr.execution.historical_restore as historical_restore

    manager, executor, record = _terminal_run(tmp_path)
    target = tmp_path / "run.out"
    target.unlink()
    publish = historical_restore.durable_publish_new

    def competing_publish(staged, destination):
        # Simulate a writer not participating in the restoration file lock.
        target.write_text("original output" if matching else "different contents")
        return publish(staged, destination)

    monkeypatch.setattr(historical_restore, "durable_publish_new", competing_publish)
    if matching:
        restored = manager.retrieve_result_files(record.id)
        assert restored.files.availability_state == FileAvailabilityState.AVAILABLE
        assert target.read_text() == "original output"
    else:
        with pytest.raises(HistoricalFileMismatchError, match="appeared during retrieval"):
            manager.retrieve_result_files(record.id)
        assert target.read_text() == "different contents"
    assert list(tmp_path.glob(".run.out.*.tmp")) == []


def test_uncaptured_external_file_is_never_overwritten(tmp_path, monkeypatch):
    import ase2sprkkr.execution.historical_restore as historical_restore
    from ase2sprkkr.execution.run_model import FileEntry

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    executor = _HistoryExecutor()
    endpoint = endpoint_for_executor(executor)
    manager.executors.register(executor, endpoint)
    target = tmp_path / "uncaptured.out"
    record = manager.new_record(
        task="DOS", directory=tmp_path, endpoint=endpoint,
        files=FileManifest(outputs=(FileEntry(
            local_path=str(target), availability_state="pending",
        ),)),
    )
    record = manager.transition(
        record, state=RunState.FINISHED,
        execution=record.execution.submitted("job-1"),
    )

    def fetch(reference, files, *, destinations=None):
        from pathlib import Path
        Path(destinations[str(target)]).write_text("downloaded")
    executor.fetch_result_files = fetch
    publish = historical_restore.durable_publish_new

    def competing_publish(staged, destination):
        target.write_text("external")
        return publish(staged, destination)

    monkeypatch.setattr(historical_restore, "durable_publish_new", competing_publish)
    with pytest.raises(HistoricalFileMismatchError, match="appeared during retrieval"):
        manager.retrieve_result_files(record.id)
    assert target.read_text() == "external"
    assert list(tmp_path.glob(".uncaptured.out.*.tmp")) == []


def test_uncaptured_wildcard_requires_concrete_historical_membership(tmp_path):
    """Historical retrieval must never materialize new wildcard members."""
    from ase2sprkkr.execution.run_model import FileEntry

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    executor = _HistoryExecutor()
    endpoint = endpoint_for_executor(executor)
    manager.executors.register(executor, endpoint)
    wildcard = tmp_path / "results-*" / "out.dat"
    record = manager.new_record(
        task="DOS", directory=tmp_path, endpoint=endpoint,
        files=FileManifest(outputs=(
            FileEntry(local_path=str(wildcard), availability_state="pending"),
        )),
    )
    record = manager.transition(
        record, state=RunState.FINISHED,
        execution=record.execution.submitted("job-1"),
    )

    with pytest.raises(ValueError, match="unresolved historical wildcard"):
        manager.retrieve_result_files(record.id)
    assert executor.fetch_calls == []
    assert not (tmp_path / "results-*").exists()
    assert record.files.captured_at is None

def test_restore_directory_sync_failure_does_not_erase_published_file(
    tmp_path, monkeypatch,
):
    import ase2sprkkr.execution._durability as durability
    from pathlib import Path

    manager, executor, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    output.unlink()
    real_sync_directory = durability.fsync_directory

    def fail_after_publish(directory):
        if Path(directory) == tmp_path and output.exists():
            raise OSError("directory fsync failed")
        return real_sync_directory(directory)

    monkeypatch.setattr(durability, "fsync_directory", fail_after_publish)
    with pytest.raises(OSError, match="directory fsync failed"):
        manager.retrieve_result_files(record.id)

    # Publication may have occurred before the durability barrier raised:
    # the content remains correct, and temporary staging is still cleaned up.
    assert output.read_text() == "original output"
    assert list(tmp_path.glob(".run.out.*.tmp")) == []
    assert manager.store.get(record.id).state is RunState.FINISHED


def test_restore_rejects_symlink_supplied_as_staged_file(tmp_path):
    manager, executor, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    output.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("original output")

    def fetch(reference, files, *, destinations=None):
        from pathlib import Path
        Path(destinations[str(output)]).symlink_to(outside)

    executor.fetch_result_files = fetch
    with pytest.raises(OSError):
        manager.retrieve_result_files(record.id)
    assert not output.exists()
    assert outside.read_text() == "original output"
    assert list(tmp_path.glob(".run.out.*.tmp")) == []


def test_restore_detects_content_changed_immediately_after_publication(
    tmp_path, monkeypatch,
):
    """An external writer cannot make corrupted history appear available."""
    import ase2sprkkr.execution.historical_restore as historical_restore

    manager, executor, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    output.unlink()
    publish = historical_restore.durable_publish_new

    def overwrite_after_publish(staged, destination):
        publish(staged, destination)
        output.write_text("different contents")

    monkeypatch.setattr(historical_restore, "durable_publish_new", overwrite_after_publish)
    with pytest.raises(HistoricalFileMismatchError, match="changed during retrieval"):
        manager.retrieve_result_files(record.id)

    saved = manager.store.get(record.id)
    assert saved.state is RunState.FINISHED
    assert saved.files.outputs[0].availability_state == "failed"
    assert output.read_text() == "different contents"


def test_historical_restore_retries_failed_transfer_without_resubmission(tmp_path):
    manager, executor, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    output.unlink()
    original_fetch = executor.fetch_result_files
    attempts = []

    def flaky_fetch(run, files, *, destinations=None):
        attempts.append(tuple(files))
        if len(attempts) == 1:
            raise OSError("transient transfer failure")
        return original_fetch(run, files, destinations=destinations)

    executor.fetch_result_files = flaky_fetch
    with pytest.raises(OSError, match="transient transfer failure"):
        manager.retrieve_result_files(record.id)
    failed = manager.store.get(record.id)
    assert failed.state is RunState.FINISHED
    assert failed.files.availability_state == FileAvailabilityState.FAILED
    assert not output.exists()

    recovered = manager.retrieve_result_files(record.id)
    assert len(attempts) == 2
    assert output.read_text() == "original output"
    assert recovered.state is RunState.FINISHED
    assert recovered.files.availability_state == FileAvailabilityState.AVAILABLE
    assert recovered.return_code == 0
    assert list(tmp_path.glob(".run.out.*.tmp")) == []


def test_history_reconciles_durable_file_after_failed_directory_fsync(tmp_path, monkeypatch):
    import ase2sprkkr.execution._durability as durability

    manager, executor, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    output.unlink()
    original = durability.fsync_directory

    def fail_publish(directory):
        if output.exists():
            raise OSError("publication fsync interrupted")
        return original(directory)

    monkeypatch.setattr(durability, "fsync_directory", fail_publish)
    with pytest.raises(OSError, match="publication fsync interrupted"):
        manager.retrieve_result_files(record.id)

    assert output.read_text() == "original output"
    assert manager.store.get(record.id).files.availability_state == "failed"
    monkeypatch.setattr(durability, "fsync_directory", original)

    restored = manager.retrieve_result_files(record.id)
    assert restored.files.availability_state == "available"
    assert executor.fetch_calls == [(record.id, (str(output),))]
    assert restored.state is RunState.FINISHED


def test_failed_published_history_reconciles_without_executor(tmp_path):
    from ase2sprkkr.execution.job_files import JobFiles

    manager, executor, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    failed = record.job_files.update_availability(
        manager.store, str(output), "failed", "directory fsync failed",
    )

    def unexpected_fetch(*args, **kwargs):
        raise AssertionError("Verified published bytes must not download")

    executor.fetch_result_files = unexpected_fetch
    recovered = JobFiles(failed).retrieve(manager.store, executor=None)
    assert recovered.files.availability_state == FileAvailabilityState.AVAILABLE
    assert recovered.state is RunState.FINISHED
    assert output.read_text() == "original output"


def test_failed_published_history_remains_failed_without_backend_if_missing(tmp_path):
    from ase2sprkkr.execution.job_files import JobFiles

    manager, _, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    failed = record.job_files.update_availability(
        manager.store, str(output), "failed", "download failed",
    )
    output.unlink()
    unchanged = JobFiles(failed).retrieve(manager.store, executor=None)
    assert unchanged.files.availability_state == FileAvailabilityState.FAILED
    assert not output.exists()


def test_historical_restore_rejects_symlink_to_captured_contents(tmp_path):
    manager, _, record = _terminal_run(tmp_path)
    output = tmp_path / "run.out"
    original = tmp_path / "original.out"
    output.rename(original)
    output.symlink_to(original)

    with pytest.raises(HistoricalFileMismatchError, match="symlink"):
        manager.retrieve_result_files(record.id)
    assert output.is_symlink()
    assert original.read_text() == "original output"
    assert manager.store.get(record.id).state is RunState.FINISHED
