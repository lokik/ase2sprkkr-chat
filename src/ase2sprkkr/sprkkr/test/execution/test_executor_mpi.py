"""MPI is a per-run declaration and is not resolved by a client at creation."""

from kkrunner.execution import MpiSpec

from ase2sprkkr.sprkkr.profile import SPRKKRProfile


def test_auto_mpi_preserves_optional_fallback(tmp_path):
    path = tmp_path / "test.inp"
    path.write_text("data")
    spec = SPRKKRProfile(mpi="auto").create_execution(
        "kkrscf", path, tmp_path / "test.out", tmp_path, mpi_supported=True
    )
    assert spec.mpi == MpiSpec.from_value("auto")
    assert spec.mpi_argv == ("kkrscfMPI", path.name)
    assert spec.argv == ("kkrscf",)
    assert spec.mpi_uses_stdin is False


def test_explicit_mpi_process_count_and_serial_selection(tmp_path):
    path = tmp_path / "test.inp"
    path.write_text("data")
    mpi = SPRKKRProfile(mpi=8).create_execution(
        "kkrscf", path, tmp_path / "test.out", tmp_path, mpi_supported=True
    )
    serial = SPRKKRProfile(mpi=8).create_execution(
        "kkrscf", path, tmp_path / "test.out", tmp_path, mpi_supported=False
    )
    assert mpi.mpi.process_count == 8
    assert mpi.mpi.required
    assert serial.mpi == MpiSpec()
    assert serial.mpi_argv is None


def test_mpi_preserves_input_subdirectory_for_planned_paths(tmp_path):
    path = tmp_path / "subdir" / "test.inp"
    profile = SPRKKRProfile(mpi=2)
    for declaration in (path, str(path)):
        spec = profile.create_execution("kkrscf", declaration, None, tmp_path, mpi_supported=True)
        assert spec.mpi_argv == ("kkrscfMPI", "subdir/test.inp")
