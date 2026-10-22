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
    assert request["argv"] == ["program"]
    assert request["mpi_uses_stdin"] is True
    assert request["mpi_argv"][0] == "programMPI"
    assert request["mpi_argv"][1].endswith("/calculation/run.inp")
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
        input_file=str(inp), input_files=(str(inp),),
        output_file=None, launch=LaunchSpec(
            local_root=str(root), remote_root=str(tmp_path / "remote"),
        ),
    ))
    request, manifest = asyncio.run(run._prepare_async())
    assert "input_files" not in request
    assert _request_spec(request).input_files == ()
    assert _request_spec(request).input_file == manifest.primary_input.remote_path
    assert manifest.input_files == (str(inp),)
