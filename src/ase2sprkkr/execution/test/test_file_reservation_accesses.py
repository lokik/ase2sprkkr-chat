"""Submission conflict detection is a pure comparison of run manifests."""

import os

import pytest

from ase2sprkkr.execution import (
    RunManager, FileEntry, FileManifest, FileMapping, RunStore,
)
from ase2sprkkr.execution.endpoint import ExecutorEndpoint
from ase2sprkkr.execution.file_reservations import reservation_conflict


def _record(manager, directory, task, files, *, dataset=None, endpoint=None,
            remote_directory=None):
    return manager.new_record(
        task=task, directory=directory, dataset=dataset, files=files,
        endpoint=endpoint, remote_directory=remote_directory,
    )


@pytest.mark.parametrize("first_role, second_role, conflict", (
    ("input", "input", False),
    ("input", "output", True),
    ("output", "input", True),
    ("output", "output", True),
    ("cache", "input", True),
    ("input", "cache", True),
    ("cache", "cache", True),
))
def test_local_access_conflict_is_symmetric(tmp_path, first_role, second_role, conflict):
    manager = RunManager(RunStore(tmp_path / "runs"))
    shared = str(tmp_path / "shared.dat")

    def manifest(role):
        entry = FileEntry(local_path=shared)
        return FileManifest(**{dict(input="inputs", output="outputs", cache="caches")[role]: (entry,)})

    left = _record(manager, tmp_path / "one", "DOS", manifest(first_role))
    right = _record(manager, tmp_path / "two", "BSF", manifest(second_role))
    assert reservation_conflict(left, right) is conflict
    assert reservation_conflict(right, left) is conflict


@pytest.mark.parametrize("sync, conflict", (
    ("never", False),
    ("missing", True),
    ("changed", True),
    ("always", True),
))
def test_remote_input_staging_is_conditional_write(tmp_path, sync, conflict):
    manager = RunManager(RunStore(tmp_path / "runs"))
    remote_root = "/scratch/common"
    remote_input = remote_root + "/stdin.in"
    endpoint = ExecutorEndpoint(executor="ssh", host="cluster")

    def record(local, task):
        mapping = FileMapping(str(local), remote_root, input_sync=sync)
        files = FileManifest(
            inputs=(FileEntry(local_path=str(local / "stdin.in")),),
            stdin=True,
            mapping=mapping,
        ).with_mapping(mapping)
        return _record(manager, local, task, files, endpoint=endpoint,
                       remote_directory=remote_root)

    first = record(tmp_path / "first", "DOS")
    second = record(tmp_path / "second", "BSF")
    assert reservation_conflict(first, second) is conflict
    assert reservation_conflict(second, first) is conflict


def test_reservation_uses_endpoint_namespace(tmp_path):
    manager = RunManager(RunStore(tmp_path / "runs"))
    files = FileManifest(stdout=FileEntry(local_path=str(tmp_path / "out.dat")))
    left = _record(manager, tmp_path, "SCF", files)
    right = _record(
        manager, tmp_path, "DOS", files,
        endpoint=ExecutorEndpoint(executor="ssh", host="cluster"),
    )
    assert not reservation_conflict(left, right)


def test_local_aliases_and_glob_members_are_writes(tmp_path):
    manager = RunManager(RunStore(tmp_path / "runs"))
    actual = tmp_path / "real.dat"
    alias = tmp_path / "linked.dat"
    actual.write_bytes(b"existing")
    os.link(actual, alias)
    left = _record(manager, tmp_path, "DOS", FileManifest(
        outputs=(FileEntry(local_path=str(actual)),),
    ))
    right = _record(manager, tmp_path / "other", "BSF", FileManifest(
        outputs=(FileEntry(local_path=str(alias)),),
    ))
    assert reservation_conflict(left, right)

    pattern = _record(manager, tmp_path, "DOS", FileManifest(
        outputs=(FileEntry(local_path=str(tmp_path / "result-*.dat")),),
    ))
    member = _record(manager, tmp_path, "BSF", FileManifest(
        stdout=FileEntry(local_path=str(tmp_path / "result-1.dat")),
    ))
    unrelated = _record(manager, tmp_path, "BSF", FileManifest(
        stdout=FileEntry(local_path=str(tmp_path / "other.dat")),
    ))
    assert reservation_conflict(pattern, member)
    assert not reservation_conflict(pattern, unrelated)


def test_empty_declarations_keep_task_dataset_fallback(tmp_path):
    manager = RunManager(RunStore(tmp_path / "runs"))
    first = _record(manager, tmp_path, "DOS", FileManifest(), dataset="same")
    second = _record(manager, tmp_path, "DOS", FileManifest(), dataset="same")
    different = _record(manager, tmp_path, "DOS", FileManifest(), dataset="other")
    assert reservation_conflict(first, second)
    assert not reservation_conflict(first, different)


def test_pure_conflict_check_does_not_mutate_records(tmp_path):
    manager = RunManager(RunStore(tmp_path / "runs"))
    files = FileManifest(outputs=(FileEntry(local_path=str(tmp_path / "out")),))
    a = _record(manager, tmp_path, "DOS", files)
    b = _record(manager, tmp_path, "BSF", files)
    before = (a, b)
    assert reservation_conflict(a, b)
    assert (a, b) == before
