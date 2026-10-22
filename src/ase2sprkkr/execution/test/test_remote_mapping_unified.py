"""Resolved file mappings own both provisional paths and remote staging."""

import asyncio

import pytest

from ase2sprkkr.execution import (
    ExecutionSpec, FileManifest, FileMapping, LaunchSpec,
    LocalTransport, RemoteEndpointExecutor,
)


@pytest.mark.parametrize("root, expected", (
    (None, "~/job"),
    ("~", "~/job"),
    ("~/projects", "~/projects/job"),
    ("projects", "~/projects/job"),
    ("/scratch/runs", "/scratch/runs/job"),
    ("/scratch/../work", "/work/job"),
))
def test_submission_directory_uses_mapping(tmp_path, root, expected):
    local = tmp_path / "workspace"
    job = local / "job"
    job.mkdir(parents=True)
    executor = RemoteEndpointExecutor(LocalTransport())
    spec = ExecutionSpec(
        argv=("true",), input_file=None, output_file=None,
        directory=str(job),
        launch=LaunchSpec(local_root=str(local), remote_root=root),
    )
    assert executor.submission_directory(spec) == expected
    assert executor.submission_directory(spec) == FileMapping(
        str(local), expected.rsplit("/job", 1)[0],
    ).map_path(job)


def test_submission_directory_rejects_local_symlink_escape(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    executor = RemoteEndpointExecutor(LocalTransport())
    spec = ExecutionSpec(
        argv=("true",), input_file=None, output_file=None,
        directory=str(root / "alias"),
        launch=LaunchSpec(local_root=root, remote_root="/scratch"),
    )
    with pytest.raises(ValueError, match="outside mapped root"):
        executor.submission_directory(spec)


def test_submission_directory_resolves_symlink_within_root(tmp_path):
    root = tmp_path / "root"
    (root / "actual").mkdir(parents=True)
    (root / "alias").symlink_to(root / "actual", target_is_directory=True)
    executor = RemoteEndpointExecutor(LocalTransport())
    spec = ExecutionSpec(
        argv=("true",), input_file=None, output_file=None,
        directory=str(root / "alias"),
        launch=LaunchSpec(local_root=root, remote_root="/scratch"),
    )
    assert executor.submission_directory(spec) == "/scratch/actual"


def test_stage_helper_rejects_noncanonical_relative_path(tmp_path):
    executor = RemoteEndpointExecutor(LocalTransport())
    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"))
    with pytest.raises(ValueError, match="must be absolute"):
        asyncio.run(executor.file_handler._stage_file(mapping, "relative.in", {}))
