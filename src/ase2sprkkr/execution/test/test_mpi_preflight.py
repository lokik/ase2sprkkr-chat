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
    candidate = replace(spec(tmp_path), stdin=False)
    assert select_mpi_spec(candidate, ("mpirun",)).stdin == "only-mpi.in"
    assert select_mpi_spec(candidate).stdin is False


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


def test_mpi_stdin_none_inherits_even_with_alternative_command(tmp_path):
    selected = select_mpi_spec(
        replace(spec(tmp_path), mpi_stdin=None), ("mpiexec",)
    )
    assert selected.argv == ("mpiexec", "parallel")
    assert selected.stdin == "only-serial.in"


def test_default_argv_and_mpi_stdin_are_independent(tmp_path):
    candidate = replace(spec(tmp_path), mpi_argv=None)
    selected = select_mpi_spec(candidate, ("mpiexec",))
    assert selected.argv == ("mpiexec", "serial")
    assert selected.stdin == "only-mpi.in"
    assert [entry.local_path for entry in FileManifest.from_spec(selected).inputs] == [
        str(tmp_path / "only-mpi.in")
    ]


def test_explicit_false_stdio_and_none_validation(tmp_path):
    candidate = replace(spec(tmp_path), stdin=False, mpi_stdin=None)
    assert select_mpi_spec(candidate, ("mpiexec",)).stdin is False
    assert not FileManifest.from_spec(select_mpi_spec(candidate, ("mpiexec",))).stdin
    with pytest.raises(ValueError, match="stdin=None is ambiguous"):
        replace(candidate, stdin=None)


def test_default_stdin_is_explicitly_disabled(tmp_path):
    candidate = ExecutionSpec(
        argv=("program",), directory=str(tmp_path),
        input_file=None, output_file=None,
    )
    assert candidate.stdin is False
    assert not FileManifest.from_spec(candidate).stdin


def test_auto_serial_fallback_preserves_serial_input(tmp_path):
    candidate = replace(spec(tmp_path), mpi_stdin=False)
    selected = select_mpi_spec(candidate)
    assert selected.argv == ("serial",)
    assert selected.stdin == "only-serial.in"
    assert selected.mpi_stdin is None


def test_direct_local_runner_does_not_resolve_again(tmp_path, monkeypatch):
    from ase2sprkkr.execution import LocalRunner

    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command",
        lambda _: ("mpiexec",),
    )
    candidate = LocalExecutor().resolve_spec(spec(tmp_path))
    runner = LocalRunner(candidate)
    assert runner.spec.argv == ("mpiexec", "parallel")
    assert runner.job_files.manifest.input_files == (
        str(tmp_path / "only-mpi.in"),
    )
    assert runner.job_files.manifest.stdin is True


def test_direct_local_project_runner_does_not_resolve_again(tmp_path, monkeypatch):
    from ase2sprkkr.execution.local_project import LocalProjectRunner

    monkeypatch.setattr(
        "ase2sprkkr.execution.local_process.resolve_mpi_command",
        lambda _: ("mpiexec",),
    )
    executor = LocalExecutor()
    candidate = executor.resolve_spec(spec(tmp_path))
    runner = LocalProjectRunner(executor, candidate, run_id="direct")
    assert runner.spec.argv == ("mpiexec", "parallel")
    assert runner.job_files.manifest.input_files == (
        str(tmp_path / "only-mpi.in"),
    )


def test_direct_remote_runner_does_not_resolve_again(tmp_path, monkeypatch):
    from ase2sprkkr.execution import LocalTransport, RemoteExecutor, RemoteRunner

    executor = RemoteExecutor(LocalTransport())

    async def resolve_mpi(mpi):
        return ("mpiexec",)

    monkeypatch.setattr(executor.connection, "resolve_mpi", resolve_mpi)
    candidate = executor.resolve_spec(spec(tmp_path))
    runner = RemoteRunner(executor, candidate)
    assert runner.spec.argv == ("mpiexec", "parallel")
    assert runner.job_files.manifest.input_files == (
        str(tmp_path / "only-mpi.in"),
    )


def test_factory_resolves_exactly_once(tmp_path, monkeypatch):
    from ase2sprkkr.execution import LocalTransport, RemoteExecutor

    local = LocalExecutor()
    remote = RemoteExecutor(LocalTransport())
    for executor in (local, remote):
        calls = []

        def resolve(candidate):
            calls.append(candidate)
            assert len(calls) == 1
            return select_mpi_spec(candidate, ("mpiexec",))

        monkeypatch.setattr(executor, "resolve_spec", resolve)
        runner = executor.create_run(spec(tmp_path))
        assert runner.spec.argv == ("mpiexec", "parallel")
        assert len(calls) == 1
