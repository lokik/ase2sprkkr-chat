"""One manifest defines file placement from reservation through submission."""

import asyncio
from dataclasses import replace

import pytest

from ase2sprkkr.execution import (
    ExecutionSpec, FileManifest, FileMapping, LaunchSpec,
    LocalTransport, RemoteEndpointExecutor,
)
from ase2sprkkr.execution.local import LocalExecutor
from ase2sprkkr.execution.local_project import LocalProjectRunner


def _spec(tmp_path):
    return ExecutionSpec(
        argv=("true",), directory=str(tmp_path),
        input_file="main.in", stdin="stdin.in", input_files=("extra.in",),
        output_file="stdout.log", error_file="stderr.log",
        output_files=("results-*.dat",), cache_files=("restart.pot",),
        launch=LaunchSpec(
            local_root=str(tmp_path), remote_root=str(tmp_path / "remote"),
            input_sync="never",
        ),
    )


def test_mapping_is_idempotent_for_every_manifest_role(tmp_path):
    spec = _spec(tmp_path)
    local = FileManifest.from_spec(spec)
    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"))
    mapped = local.with_mapping(mapping)
    assert mapped.with_mapping(mapping) == mapped
    assert mapped.mapping == mapping
    for entry in (
        *mapped.inputs, mapped.primary_input, mapped.stdin,
        *mapped.outputs, *mapped.caches, mapped.stdout, mapped.stderr,
    ):
        assert entry.remote_path == mapping.map_path(entry.local_path)
    assert local.mapping is None
    assert all(entry.remote_path is None for entry in local.inputs)


@pytest.mark.parametrize("role", (
    "inputs", "primary_input", "stdin", "outputs", "caches", "stdout", "stderr",
))
def test_mapping_rejects_changed_reserved_path_for_all_roles(tmp_path, role):
    manifest = FileManifest.from_spec(_spec(tmp_path))
    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"))
    reserved = manifest.with_mapping(mapping)
    old = getattr(reserved, role)
    if isinstance(old, tuple):
        bad = (replace(old[0], remote_path="/elsewhere/bad"), *old[1:])
    else:
        bad = replace(old, remote_path="/elsewhere/bad")
    corrupted = replace(reserved, **{role: bad})
    with pytest.raises(ValueError, match="differs from reserved path"):
        corrupted.with_mapping(mapping)


def test_mapping_rejects_changed_policy_even_with_same_roots(tmp_path):
    initial = FileMapping(
        str(tmp_path), str(tmp_path / "remote"), input_sync="never",
    )
    changed = FileMapping(
        str(tmp_path), str(tmp_path / "remote"), input_sync="always",
    )
    files = FileManifest.from_spec(_spec(tmp_path)).with_mapping(initial)
    with pytest.raises(ValueError, match="differs from reserved mapping"):
        files.with_mapping(changed)


def test_remote_submission_uses_reserved_primary_input_and_stdin(tmp_path):
    spec = _spec(tmp_path)
    executor = RemoteEndpointExecutor(LocalTransport())
    runner = executor.create_run(spec)
    mapping = FileMapping(str(tmp_path), str(tmp_path / "remote"),
                          input_sync="never")
    # NEVER transfers nothing: required inputs must already exist on target.
    remote = tmp_path / "remote"
    remote.mkdir()
    for name in ("main.in", "stdin.in", "extra.in"):
        (remote / name).write_text(name)
    reserved = FileManifest.from_spec(spec).with_mapping(mapping)
    runner.job_files.replace(reserved)
    # With NEVER sync the mandatory inputs must already exist remotely.
    remote = tmp_path / "remote"
    remote.mkdir(exist_ok=True)
    for name in ("main.in", "stdin.in", "extra.in"):
        (remote / name).write_text(name)
    # A stale or independently modified spec cannot change file membership.
    runner.spec = replace(spec, input_file="wrong.in", stdin="wrong.stdin",
                          input_files=("unreserved.in",))
    request, prepared = asyncio.run(runner._prepare_async())
    assert prepared.primary_input == reserved.primary_input
    assert prepared.stdin == reserved.stdin
    assert request["input_file"] == reserved.primary_input.remote_path
    assert request["stdin"] == reserved.stdin.remote_path
    assert "wrong.in" not in str(request["files"])
    assert "unreserved.in" not in str(request["files"])
    assert {path.name for path in remote.iterdir()} == {
        "main.in", "stdin.in", "extra.in",
    }


def test_never_requires_existing_remote_primary_input(tmp_path):
    spec = _spec(tmp_path)
    # Local content does not satisfy input_sync=never.
    (tmp_path / "main.in").write_text("local input")
    remote = tmp_path / "remote"
    remote.mkdir()
    mapping = FileMapping(str(tmp_path), str(remote), input_sync="never")
    manifest = FileManifest.from_spec(spec).with_mapping(mapping)
    executor = RemoteEndpointExecutor(LocalTransport())

    with pytest.raises(FileNotFoundError, match="Required remote input file"):
        asyncio.run(executor.file_handler.prepare(mapping, spec, manifest))
    assert not (remote / "main.in").exists()



def test_local_submission_uses_reserved_file_declarations(tmp_path):
    spec = _spec(tmp_path)
    reserved = FileManifest.from_spec(spec)
    runner = LocalProjectRunner(LocalExecutor(), spec, run_id="reserved-job")
    runner.job_files.replace(reserved)
    runner.spec = replace(spec, input_file="wrong.in", stdin="wrong.stdin",
                          output_file="wrong.out", error_file="wrong.err")
    request = runner._request()
    assert request["input_file"] == reserved.input_file
    assert request["stdin"] == reserved.stdin.local_path
    assert request["output_file"] == reserved.output_file
    assert request["error_file"] == reserved.stderr.local_path
    assert request["files"]["primary_input"]["local_path"] == reserved.input_file
