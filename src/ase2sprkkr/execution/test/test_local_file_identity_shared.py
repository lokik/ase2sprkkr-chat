"""Declaration, launch cleanup and reservation share local file identity."""

import os

import pytest

from ase2sprkkr.execution import (
    ExecutionManager, FileEntry, FileManifest, RunStore,
)
from ase2sprkkr.execution.file_availability import validate_output_collisions
from ase2sprkkr.execution.file_identity import same_local_file
from ase2sprkkr.execution.file_reservations import reservation_conflict


def test_missing_path_lexical_alias_counts_as_same_file(tmp_path):
    a = tmp_path / "missing.dat"
    b = tmp_path / "subdir" / ".." / "missing.dat"
    assert same_local_file(a, b)
    assert not same_local_file(a, tmp_path / "other.dat")
    assert not same_local_file(None, a)


@pytest.mark.parametrize("alias_kind", ("symlink-file", "symlink-dir", "hardlink"))
def test_same_file_identity_is_shared_across_all_three_policies(tmp_path, alias_kind):
    file = tmp_path / "main.out"
    file.write_text("existing output")
    if alias_kind == "symlink-file":
        alias = tmp_path / "alias.out"
        alias.symlink_to(file)
    elif alias_kind == "symlink-dir":
        folder = tmp_path / "alias"
        folder.symlink_to(tmp_path, target_is_directory=True)
        alias = folder / "main.out"
    else:
        alias = tmp_path / "alias.out"
        os.link(file, alias)
    assert same_local_file(file, alias)

    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(
            stdout=FileEntry(local_path=str(file)),
            caches=(FileEntry(local_path=str(alias)),),
        )
    with pytest.raises(ValueError, match="overlaps protected"):
        validate_output_collisions((str(alias),), protected=(str(file),))

    manager = ExecutionManager(RunStore(tmp_path / "runs"))
    first = manager.new_record(
        task="DOS", directory=tmp_path,
        files=FileManifest(outputs=(FileEntry(local_path=str(file)),)),
    )
    second = manager.new_record(
        task="BSF", directory=tmp_path,
        files=FileManifest(stdout=FileEntry(local_path=str(alias))),
    )
    assert reservation_conflict(first, second)
    assert reservation_conflict(second, first)


def test_nonexistent_symlink_directory_alias_is_protected(tmp_path):
    root = tmp_path / "real"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    target = root / "future.out"
    linked = alias / "future.out"
    assert same_local_file(target, linked)
    with pytest.raises(ValueError, match="Duplicate file declaration"):
        FileManifest(outputs=(
            FileEntry(local_path=str(target)),
            FileEntry(local_path=str(linked)),
        ))
