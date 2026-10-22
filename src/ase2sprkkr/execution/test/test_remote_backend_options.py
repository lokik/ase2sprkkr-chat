import asyncio

from ase2sprkkr.execution import (
    ExecutionSpec,
    LaunchSpec,
    LocalTransport,
    MpiSpec,
    RemoteExecutor,
)


def test_remote_request_carries_target_executor_options(tmp_path):
    local_root = tmp_path / "local"
    directory = local_root / "calculation"
    directory.mkdir(parents=True)
    input_file = (directory / "run.inp").open("w+b")
    input_file.write(b"input")
    input_file.flush()
    potential = directory / "start.pot"
    potential.write_text("potential")
    output_file = (directory / "run.out").open("wb")

    executor = RemoteExecutor(
        LocalTransport(),
        backend="psij",
        backend_options={"backend": "slurm", "url": None},
    )
    launch = LaunchSpec(
        mpi=MpiSpec(required=True, autodetect=True, process_count=16),
        local_root=local_root,
        remote_root=tmp_path / "remote",
    )
    spec = ExecutionSpec(
        argv=("program",),
        mpi_argv=("programMPI", input_file.name),
        launch=launch,
        input_file=input_file,
        output_file=output_file,
        directory=str(directory),
        stdin=input_file,
        input_files=(potential,),
    )
    runner = executor.create_run(spec)
    request, _manifest = asyncio.run(runner._prepare_async())

    assert request["executor"] == "psij"
    assert request["executor_options"] == {"backend": "slurm", "url": None}
    assert request["argv"][0] == "programMPI"
    assert request["argv"][1].endswith("/calculation/run.inp")
    assert request["stdin"] is None
    assert request["mpi_argv"] is None
    assert request["launch"] == {
        "mpi": {
            "required": True,
            "autodetect": True,
            "launcher": (),
            "process_count": 16,
            "warning": True,
        },
        "local_root": str(local_root),
        "remote_root": str(tmp_path / "remote"),
        "input_sync": "changed",
        "output_sync": "changed",
    }
    remote_input = tmp_path / "remote" / "calculation" / "run.inp"
    remote_potential = tmp_path / "remote" / "calculation" / "start.pot"
    assert "output_files" not in request
    assert "cache_files" not in request
    assert "client_sync" not in request
    assert request["files"]["outputs"] == ()
    assert request["files"]["caches"] == ()
    assert request["input_file"] == str(remote_input)
    assert remote_input.read_bytes() == b"input"
    assert remote_potential.read_text() == "potential"

    input_file.close()
    output_file.close()

def test_nonproject_remote_request_does_not_publish_project_identity(tmp_path):
    root = tmp_path / "local"
    directory = root / "work"
    directory.mkdir(parents=True)
    executor = RemoteExecutor(LocalTransport(), backend="psij")
    runner = executor.create_run(ExecutionSpec(
        argv=("program",), directory=str(directory), input_file=None,
        output_file=None, launch=LaunchSpec(
            local_root=root, remote_root=tmp_path / "remote",
        ),
    ))
    request, _manifest = asyncio.run(runner._prepare_async())
    assert request["id"] is None


def test_remote_sync_declarations_have_one_wire_authority(tmp_path):
    from ase2sprkkr.execution.remote_service import _request_spec

    root = tmp_path / "local"
    directory = root / "work"
    directory.mkdir(parents=True)
    output = directory / "run.out"
    auxiliary = directory / "plot-*.dat"
    cache = directory / "restart.dat"
    cache.write_text("restart")
    executor = RemoteExecutor(LocalTransport())
    run = executor.create_run(ExecutionSpec(
        argv=("program",), directory=str(directory), input_file=None,
        output_file=str(output),
        output_files=(auxiliary,),
        cache_files=(cache,),
        launch=LaunchSpec(local_root=root, remote_root=tmp_path / "remote"),
    ))
    remote_dir = tmp_path / "remote" / "work"
    request, _manifest = asyncio.run(run._prepare_async())
    assert "output_files" not in request
    assert "cache_files" not in request
    restored = _request_spec(request)
    assert str(remote_dir / "plot-*.dat") in restored.output_files
    assert str(remote_dir / "restart.dat") in restored.cache_files


def test_remote_manifest_is_sole_authority_for_input_declarations(tmp_path):
    from ase2sprkkr.execution.remote_service import _request_spec

    root = tmp_path / "local"
    root.mkdir()
    inp = root / "data.in"
    inp.write_text("input")
    executor = RemoteExecutor(LocalTransport())
    run = executor.create_run(ExecutionSpec(
        argv=("program",), directory=str(root),
        input_file=str(inp), input_files=(),
        output_file=None, launch=LaunchSpec(
            local_root=str(root), remote_root=str(tmp_path / "remote"),
        ),
    ))
    request, manifest = asyncio.run(run._prepare_async())
    assert "input_files" not in request
    assert _request_spec(request).input_files == ()
    assert _request_spec(request).input_file == manifest.inputs[0].remote_path
    assert manifest.input_files == (str(inp),)


def test_remote_spec_ignores_remote_only_cache_baselines(tmp_path):
    """Cache signatures describe staging history, not another input file."""
    from dataclasses import asdict
    from ase2sprkkr.execution import FileEntry, FileManifest
    from ase2sprkkr.execution.remote_service import _request_spec

    directory = tmp_path / "remote" / "calculation"
    directory.mkdir(parents=True)
    exact = directory / "restart-a.dat"
    pattern = directory / "restart-*.dat"
    exact.write_text("initial")
    manifest = FileManifest(caches=(
        FileEntry(local_path=str(exact), remote_path=str(exact)),
        FileEntry(local_path=str(pattern), remote_path=str(pattern)),
        FileEntry(remote_path=str(exact), source=str(pattern), pre_run_signature=(7, 42)),
    ))
    assert len(manifest.caches) == 3
    spec = _request_spec({
        "directory": str(directory), "argv": ("program",),
        "files": asdict(manifest),
    })
    assert spec.cache_files == (str(exact), str(pattern))
    remote_only = FileManifest(caches=(FileEntry(remote_path=str(exact)),))
    assert _request_spec({
        "directory": str(directory), "argv": ("program",),
        "files": asdict(remote_only),
    }).cache_files == (str(exact),)
    # The original frozen manifest still carries the remote baseline.
    assert manifest.caches[2].pre_run_signature == (7, 42)
    assert FileManifest.from_spec(spec).caches == (
        FileEntry(local_path=str(exact), availability_state="pending"),
        FileEntry(local_path=str(pattern), availability_state="pending"),
    )


def test_remote_psij_selects_mpi_stdin_without_alternate_command(tmp_path):
    local_root = tmp_path / "local"
    directory = local_root / "calculation"
    directory.mkdir(parents=True)
    (directory / "mpi.in").write_text("mpi input")

    executor = RemoteExecutor(
        LocalTransport(), backend="psij",
        backend_options={"backend": "slurm", "url": None},
    )
    spec = ExecutionSpec(
        argv=("program",), mpi_argv=None,
        directory=str(directory),
        input_file=None, output_file=None,
        stdin=str(directory / "serial.in"),
        mpi_stdin=str(directory / "mpi.in"),
        launch=LaunchSpec(
            mpi=MpiSpec(required=True, autodetect=True, process_count=4),
            local_root=str(local_root), remote_root=str(tmp_path / "remote"),
        ),
    )
    runner = executor.create_run(spec)
    assert runner.spec.stdin == str(directory / "mpi.in")
    assert runner.spec.mpi_stdin is None
    request, manifest = asyncio.run(runner._prepare_async())
    assert request["argv"] == ["program"]
    assert request["stdin"].endswith("/calculation/mpi.in")
    assert request["mpi_argv"] is None
    assert manifest.input_files == (str(directory / "mpi.in"),)
    assert not any("serial.in" in str(path) for path in manifest.input_files)
