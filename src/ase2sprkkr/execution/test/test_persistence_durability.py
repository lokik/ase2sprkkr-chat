import os
import stat
from pathlib import Path

import pytest

import ase2sprkkr.execution._durability as durability
import ase2sprkkr.execution.run_store as store_module
import ase2sprkkr.execution.file_history as file_history
from ase2sprkkr.execution.base import RunState
from ase2sprkkr.execution.endpoint import ExecutorEndpoint
from ase2sprkkr.execution.run_model import FileManifest, FileEntry
from ase2sprkkr.execution.run_manager import RunManager
from ase2sprkkr.execution.run_store import RunStore


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="strong directory fsync semantics are currently POSIX-only",
)


def test_durable_write_orders_file_fsync_before_replace_and_directory_fsync(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.json"
    events = []
    real_fsync = durability.os.fsync
    real_replace = durability.os.replace

    def tracked_fsync(descriptor):
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        return real_fsync(descriptor)

    def tracked_replace(source, destination):
        events.append(("replace", Path(destination).name))
        return real_replace(source, destination)

    monkeypatch.setattr(durability.os, "fsync", tracked_fsync)
    monkeypatch.setattr(durability.os, "replace", tracked_replace)

    durability.durable_write_json(path, {"state": "running"})

    assert events == [
        ("fsync", "file"),
        ("replace", "state.json"),
        ("fsync", "directory"),
    ]


def test_durable_write_propagates_directory_barrier_failure(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    path.write_text("old")

    def fail_directory_fsync(_directory):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(durability, "fsync_directory", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        durability.durable_write_text(path, "new")

    # The rename may already have happened; the important contract is that the
    # caller was not told the transition was durably committed.
    assert path.read_text() == "new"
    assert not tuple(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("already_exists", (False, True))
def test_durable_mkdir_syncs_parent_directory_entries(
    tmp_path, monkeypatch, already_exists,
):
    directory = tmp_path / "one"
    if already_exists:
        directory.mkdir()
        target = directory
        expected = [tmp_path]
    else:
        target = directory / "two"
        expected = [tmp_path, directory]
    synced = []
    monkeypatch.setattr(
        durability, "fsync_directory",
        lambda parent: synced.append(Path(parent)),
    )

    durability.durable_mkdir(target, parents=True, exist_ok=already_exists)

    assert synced == expected


def test_run_store_uses_durable_json_writer(tmp_path, monkeypatch):
    calls = []
    real_write = store_module.durable_write_json

    def tracked_write(path, data, **options):
        calls.append((Path(path), options))
        return real_write(path, data, **options)

    monkeypatch.setattr(store_module, "durable_write_json", tracked_write)
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    record = manager.new_record(task="SCF", directory=tmp_path)

    store.save(record)

    assert len(calls) == 1
    assert calls[0][0].name == f"{record.id}.json"
    assert calls[0][1]["mode"] == 0o600


def test_reserve_syncs_inputs_before_submitting_record(tmp_path, monkeypatch):
    input_file = tmp_path / "input.inp"
    input_file.write_text("input")
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    record = manager.new_record(
        task="SCF",
        directory=tmp_path,
        endpoint=ExecutorEndpoint(),
        files=FileManifest(inputs=(FileEntry(local_path=str(input_file)),)),
    )
    events = []

    monkeypatch.setattr(
        file_history,
        "sync_file",
        lambda path: events.append(("sync", Path(path).name)),
    )
    real_write = store_module.durable_write_json

    def tracked_write(path, data, **options):
        if data["state"] == RunState.SUBMITTING.value:
            events.append(("state", "submitting"))
        return real_write(path, data, **options)

    monkeypatch.setattr(store_module, "durable_write_json", tracked_write)

    reserved = manager.reserve(record)

    assert reserved.state == RunState.SUBMITTING
    assert events[:2] == [("sync", "input.inp"), ("state", "submitting")]



def test_input_sync_failure_never_commits_submission(tmp_path, monkeypatch):
    input_file = tmp_path / "input.inp"
    input_file.write_text("input")
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    record = manager.new_record(
        task="SCF", directory=tmp_path,
        files=FileManifest(inputs=(FileEntry(local_path=str(input_file)),)),
    )

    def fail_sync(path):
        raise OSError("input fsync failed")

    monkeypatch.setattr(file_history, "sync_file", fail_sync)
    with pytest.raises(OSError, match="input fsync failed"):
        manager.reserve(record)
    with pytest.raises(FileNotFoundError):
        store.get(record.id)



def test_file_publication_syncs_outputs_before_terminal_record(tmp_path, monkeypatch):
    output_file = tmp_path / "output.out"
    output_file.write_text("result")
    store = RunStore(tmp_path / "runs")
    manager = RunManager(store)
    record = manager.new_record(
        task="SCF",
        directory=tmp_path,
        files=FileManifest(stdout=FileEntry(local_path=str(output_file))),
    )
    store.save(record)
    record = manager.transition(record, state=RunState.POSTPROCESSING)
    events = []

    monkeypatch.setattr(
        file_history,
        "sync_file",
        lambda path: events.append(("sync", Path(path).name)),
    )
    real_transition = manager.transition

    def tracked_transition(current, **changes):
        if changes.get("state") == RunState.FINISHED:
            events.append(("state", "finished"))
        return real_transition(current, **changes)

    monkeypatch.setattr(manager, "transition", tracked_transition)

    finalized = manager.complete(record, state=RunState.FINISHED, return_code=0, files_available=True)

    assert finalized.state == RunState.FINISHED
    assert events[:2] == [("sync", "output.out"), ("state", "finished")]
    assert len(finalized.files.output_versions) == 1




def test_durable_write_preserves_write_error_over_temp_cleanup_error(
    tmp_path, monkeypatch,
):
    original_unlink = Path.unlink

    def fail_write(*args, **kwargs):
        raise OSError("original write failure")

    def fail_unlink(self, *args, **kwargs):
        if self.name.endswith(".tmp"):
            raise OSError("secondary cleanup failure")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(durability.os, "fsync", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(OSError, match="original write failure"):
        durability.durable_write_text(tmp_path / "state.json", "new")
