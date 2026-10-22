"""MPI selection is frozen before file membership and reservation."""
from dataclasses import replace

import pytest

from ase2sprkkr.execution import ExecutionSpec, FileManifest, LaunchSpec, LocalExecutor
from ase2sprkkr.execution.local_process import select_mpi_spec


def spec(tmp_path):
    return ExecutionSpec(
        argv=("serial",), mpi_argv=("parallel",),
        directory=str(tmp_path), input_file=None, output_file=None,
        stdin="only-serial.in", mpi_stdin="only-mpi.in",
        launch=LaunchSpec.from_mpi("auto"),
    )


def test_only_selected_mpi_stdin_is_in_manifest(tmp_path):
    selected = select_mpi_spec(spec(tmp_path), ("mpirun",))
    assert selected.argv == ("mpirun", "parallel")
    assert selected.stdin == "only-mpi.in"
    assert selected.mpi_argv is None
    assert selected.mpi_stdin is None
    assert [entry.local_path for entry in FileManifest.from_spec(selected).inputs] == [
        str(tmp_path / "only-mpi.in")
    ]


def test_serial_fallback_does_not_stage_mpi_input(tmp_path):
    selected = select_mpi_spec(spec(tmp_path))
    assert selected.argv == ("serial",)
    assert selected.stdin == "only-serial.in"
    assert not selected.launch.mpi.requested
    assert [entry.local_path for entry in FileManifest.from_spec(selected).inputs] == [
        str(tmp_path / "only-serial.in")
    ]


def test_mpi_only_stdin_is_supported(tmp_path):
    candidate = replace(spec(tmp_path), stdin=None)
    assert select_mpi_spec(candidate, ("mpirun",)).stdin == "only-mpi.in"
    assert select_mpi_spec(candidate).stdin is None


def test_local_executor_resolves_before_runner_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command",
        lambda _: ("srun", "-n", "4"),
    )
    runner = LocalExecutor().create_run(spec(tmp_path))
    assert runner.spec.argv == ("srun", "-n", "4", "parallel")
    assert runner.job_files.manifest.input_file == str(tmp_path / "only-mpi.in")


def test_required_mpi_preflight_failure_does_not_create_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command", lambda _: None,
    )
    required = replace(spec(tmp_path), launch=LaunchSpec.from_mpi(True))
    with pytest.raises(FileNotFoundError):
        LocalExecutor().create_run(required)


def test_local_mpi_without_alternate_command_wraps_ordinary_argv(tmp_path, monkeypatch):
    """Absent mpi_argv does not prohibit MPI; stdin remains independent."""
    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command",
        lambda _: ("mpirun", "-np", "2"),
    )
    candidate = replace(
        spec(tmp_path), mpi_argv=None, mpi_stdin=None, stdin="ordinary.in",
        launch=LaunchSpec.from_mpi(2),
    )
    selected = LocalExecutor().resolve_spec(candidate)
    assert selected.argv == ("mpirun", "-np", "2", "serial")
    assert selected.stdin == "ordinary.in"
    assert not selected.launch.mpi.requested
    assert selected.mpi_argv is None
    assert selected.mpi_stdin is None
    assert LocalExecutor().resolve_spec(selected) == selected
    assert FileManifest.from_spec(selected).input_files == (
        str(tmp_path / "ordinary.in"),
    )


def test_ssh_mpi_without_alternate_command_wraps_ordinary_argv(
    tmp_path, monkeypatch,
):
    from ase2sprkkr.execution import LocalTransport, RemoteExecutor

    executor = RemoteExecutor(LocalTransport())
    observed = []

    async def resolve_mpi(mpi):
        observed.append(mpi)
        return ("srun", "-n", "2")

    monkeypatch.setattr(executor.connection, "resolve_mpi", resolve_mpi)
    candidate = replace(
        spec(tmp_path), mpi_argv=None, mpi_stdin=None,
        launch=LaunchSpec.from_mpi(2),
    )
    selected = executor.resolve_spec(candidate)
    assert selected.argv == ("srun", "-n", "2", "serial")
    assert selected.stdin == "only-serial.in"
    assert selected.mpi_argv is None
    assert not selected.launch.mpi.requested
    assert executor.resolve_spec(selected) == selected
    assert len(observed) == 1


def test_no_alternate_command_with_mpi_disabled_remains_serial(tmp_path):
    candidate = replace(
        spec(tmp_path), mpi_argv=None, launch=LaunchSpec(), mpi_stdin=None,
    )
    selected = LocalExecutor().resolve_spec(candidate)
    assert selected is candidate
    assert selected.argv == ("serial",)


def test_mpi_stdin_with_ordinary_argv_overrides_serial_stdin(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command",
        lambda _: ("mpirun",),
    )
    candidate = replace(
        spec(tmp_path), mpi_argv=None, mpi_stdin="mpi-input.in",
        launch=LaunchSpec.from_mpi(True),
    )
    selected = LocalExecutor().resolve_spec(candidate)
    assert selected.argv == ("mpirun", "serial")
    assert selected.stdin == "mpi-input.in"
    assert LocalExecutor().resolve_spec(selected) == selected
    assert FileManifest.from_spec(selected).input_files == (
        str(tmp_path / "mpi-input.in"),
    )


def test_serial_fallback_with_default_argv_uses_serial_stdin(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command",
        lambda _: None,
    )
    candidate = replace(
        spec(tmp_path), mpi_argv=None, mpi_stdin="mpi-input.in",
        launch=LaunchSpec.from_mpi("auto"),
    )
    selected = LocalExecutor().resolve_spec(candidate)
    assert selected.argv == ("serial",)
    assert selected.stdin == "only-serial.in"
    assert selected.mpi_stdin is None
    assert not selected.launch.mpi.requested
    assert FileManifest.from_spec(selected).input_files == (
        str(tmp_path / "only-serial.in"),
    )


def test_ssh_mpi_stdin_with_ordinary_argv(tmp_path, monkeypatch):
    from ase2sprkkr.execution import LocalTransport, RemoteExecutor

    executor = RemoteExecutor(LocalTransport())

    async def resolve_mpi(mpi):
        return ("mpiexec",)

    monkeypatch.setattr(executor.connection, "resolve_mpi", resolve_mpi)
    candidate = replace(
        spec(tmp_path), mpi_argv=None, mpi_stdin="mpi-input.in",
        launch=LaunchSpec.from_mpi(True),
    )
    selected = executor.resolve_spec(candidate)
    assert selected.argv == ("mpiexec", "serial")
    assert selected.stdin == "mpi-input.in"
    assert selected.mpi_stdin is None
    assert FileManifest.from_spec(selected).input_files == (
        str(tmp_path / "mpi-input.in"),
    )
