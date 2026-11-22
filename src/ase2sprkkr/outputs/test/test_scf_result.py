import os
from pathlib import Path
from shutil import copyfile

import pytest

from ...input_parameters.input_parameters import InputParameters
from ..readers import scf as scf_reader
from ..readers.scf import ScfOutputParser, ScfResult
from ..task_result import TaskResult


POTENTIAL_EXAMPLES = Path(__file__).resolve().parents[2] / "potentials" / "examples"
CLUSTER_OUTPUT = Path(__file__).parent / "data" / "native_cluster_restart.txt"


@pytest.fixture
def embedded_scf_result(tmp_path):
    for example, filename in (
        ("Fe3CoFe3.host.pot", "host.pot"),
        ("Fe3CoFe3.clu.pot", "cluster.pot"),
        ("Fe3CoFe3.clu.iterating.pot", "cluster.pot_new"),
    ):
        copyfile(POTENTIAL_EXAMPLES / example, tmp_path / filename)

    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"
    parameters.CONTROL.CLUTYPE = "embedded"
    return ScfResult(parameters, None, tmp_path)


def test_native_magnetic_dipole_table_is_preserved():
    text = CLUSTER_OUTPUT.read_text()
    start = text.index("  33 E=")
    end = text.index("\n -------------------------------------------------------------------------------", start)
    atom = scf_reader.atomic_types_definition.parse(text[start:end])
    assert atom["IT"] == 1
    data = atom["magnetic_dipole"]
    assert list(data["l"]) == ["s", "p", "d", "sum"]
    assert data["P"][-1] == pytest.approx(0.0052)


@pytest.mark.parametrize("complete", [True, False])
def test_cluster_restart_uses_cluster_atomic_types_after_host_gen_summary(tmp_path, complete):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"
    parameters.CONTROL.CLUTYPE = "embedded"
    result = ScfResult(parameters, None, tmp_path)
    source = CLUSTER_OUTPUT
    if not complete:
        truncated = tmp_path / "truncated.out"
        truncated.write_text(source.read_text().split("   1 ERR", 1)[0])
        source = truncated
    ScfOutputParser().read_from_file(source, None, [result])
    if not complete:
        assert len(result.iterations) == 0
        assert result.converged is False
        return
    assert len(result.iterations) == 1
    assert result.converged is True
    assert [atom.IT() for atom in result.last_iteration.atomic_types.values()] == [3, 4, 5]
    assert result.last_iteration.energy.EMIN() == pytest.approx(-0.2)


def test_scf_failure_before_first_iteration_has_no_convergence_or_last_iteration(tmp_path):
    result = ScfResult(None, None, tmp_path)
    assert result.converged is False
    with pytest.raises(AttributeError, match="No iteration has been finished"):
        _ = result.last_iteration


def test_scf_output_without_input_file_preserves_result_semantics(tmp_path):
    source = Path(__file__).parent.parent / "examples" / "scf.out"
    output = tmp_path / "scf.out"
    output.write_bytes(source.read_bytes())

    result = TaskResult.from_file(output)

    assert isinstance(result, ScfResult)
    assert result.input_parameters is None
    assert result.task_name == "scf"
    assert result.produces_potential is True


def test_import_with_native_input_resolves_potential_before_scf_parser(tmp_path):
    """SCF needs POTFIL immediately after FILES, not only at completion."""
    source = Path(__file__).parent.parent / "examples" / "scf.out"
    output = tmp_path / "scf.out"
    output.write_bytes(source.read_bytes())

    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "from-native-input.pot"
    (tmp_path / "scf.inp").write_text(parameters.to_string(validate=False))

    result = TaskResult.from_file(output)
    assert isinstance(result, ScfResult)
    assert result.input_parameters.task_name == "SCF"
    assert result.start_potential_filename == str(tmp_path / "from-native-input.pot")
    assert result.files["converged"].path() == str(tmp_path / "from-native-input.pot_new")


def test_scf_result_calculator_without_original_calculator(monkeypatch, tmp_path):
    result = ScfResult(None, None, tmp_path)
    result.files.add_file("potential", "Fe.pot")
    expected_potential = os.path.join(os.fspath(tmp_path), "Fe.pot_new")
    marker = object()

    def make_calculator(*, potential):
        assert potential == expected_potential
        return marker

    monkeypatch.setattr(scf_reader, "SPRKKR", make_calculator)

    assert result.calculator is marker


@pytest.mark.parametrize("has_original_calculator", (False, True))
def test_new_task_continuation_is_independent_of_original_calculator(
    tmp_path, has_original_calculator,
):
    class OriginalCalculator:
        def copy_with_potential(self, potential):
            raise AssertionError("continuation must not copy the original calculator")

    original = OriginalCalculator() if has_original_calculator else None
    result = TaskResult(None, original, tmp_path)
    result.files.add_file("potential", "Fe.pot")

    calculator = result.new_task("DOS")

    assert calculator._potential == os.path.join(os.fspath(tmp_path), "Fe.pot")
    assert calculator.input_parameters.task_definition.name == "DOS"


def test_scf_info_output_includes_final_status(capsys):
    output = Path(__file__).parent.parent / "examples" / "scf.out"
    parser = ScfOutputParser(print_info=True)
    result = ScfResult(None, None, None)

    parser.read_from_file(output, read_args=[result])

    assert parser.print_info is True
    assert "OK: The computation converged." in capsys.readouterr().out


def test_embedded_scf_result_reads_clupot_and_continues_with_dos(tmp_path, embedded_scf_result):
    """SCF output must not replace CLUPOT by the unchanged periodic host."""
    from ...potentials.cluster_potentials import ClusterPotential
    from ...sprkkr.executables import executables

    result = embedded_scf_result
    ScfOutputParser().read_from_file(CLUSTER_OUTPUT, None, [result])
    start = tmp_path / "cluster.pot"
    converged = tmp_path / "cluster.pot_new"

    assert result.start_potential_filename == str(start)
    assert result.potential_filename == str(converged)
    assert result.is_embedded
    assert isinstance(result.start_potential, ClusterPotential)
    assert isinstance(result.potential, ClusterPotential)
    assert result.potential.host is result.start_potential.host
    assert result.start_potential.SCF_INFO.SCFSTATUS() == "START"
    assert result.potential.SCF_INFO.SCFSTATUS() == "CONVERGED"
    assert str(result.atoms.symbols) == "Fe3CoFe3"
    assert result.atoms.sites[0].embedding.cell_index == (-1, 0, 0)
    assert result.calculator.potential is result.potential

    output_paths = {
        os.fspath(item) for item in result.input_parameters.output_files(str(tmp_path))
    }
    assert str(converged) in output_paths
    assert str(tmp_path / "host.pot_new") not in output_paths

    dos = result.new_task("DOS")
    assert dos.potential is result.potential
    assert dos.input_parameters.task_name == "DOS"
    prepared = dos.prepare(
        directory=tmp_path / "dos",
        input_file="dos.inp",
        potential_file="host.pot",
        output_file=False,
        create_subdirs=True,
        empty_spheres=False,
    )
    try:
        assert prepared.cluster_potential_file is not None
        assert prepared.input_parameters.CONTROL.CLUTYPE() == "embedded"
        assert executables.resolve(
            prepared.input_parameters.task_definition_required().executable,
            prepared.input_parameters,
        ) == "embgen"
        assert os.path.isfile(prepared.cluster_potential_file)
        assert os.path.isfile(prepared.potential_file)
    finally:
        prepared.close()


def test_embedded_result_requires_clupot_and_validates_host_file(tmp_path, monkeypatch):
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.CLUTYPE = "embedded"
    result = ScfResult(parameters, None, tmp_path)
    with pytest.raises(ValueError, match="CONTROL.POTFIL_CLU"):
        _ = result.potential_filename

    parameters.CONTROL.POTFIL_CLU = "cluster.pot"
    result = ScfResult(parameters, None, tmp_path)
    verified = []
    monkeypatch.setattr(result, "require_current_file", verified.append)
    with pytest.raises(OSError):
        result._read_potential(str(tmp_path / "cluster.pot"))
    assert verified == [str(tmp_path / "host.pot")]


def test_input_parameters_load_existing_embedded_potentials(tmp_path, embedded_scf_result):
    parameters = InputParameters.for_task("SCF")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"
    from ...sprkkr._preparation import load_existing_potential_input

    with pytest.raises(ValueError, match="requires CONTROL.CLUTYPE=embedded"):
        load_existing_potential_input(parameters, tmp_path)

    parameters.CONTROL.CLUTYPE = "embedded"
    source = load_existing_potential_input(parameters, tmp_path)
    potential, snapshot = source.potential, source.parameters
    assert len(source.versions) == 2
    assert potential.atoms.embedding_host is potential.host
    assert snapshot.CONTROL.POTFIL() == str(tmp_path / "host.pot")
    assert snapshot.CONTROL.POTFIL_CLU() == str(tmp_path / "cluster.pot")
    assert snapshot.CONTROL.DATASET() == "CoFe6"
    assert parameters.CONTROL.POTFIL() == "host.pot"

    # Saving an embedded structure directly derives the same native settings
    # as calculator preparation.
    exported = parameters.copy()
    exported.CONTROL.CLUTYPE.clear()
    path = tmp_path / "embedded.inp"
    exported.save_to_file(path, atoms=potential.atoms)
    reread = InputParameters.from_file(path)
    assert reread.CONTROL.CLUTYPE() == "embedded"
    assert reread.TAU.NKTAB.is_set()

def test_embedded_scf_parser_registers_converged_cluster_not_host(tmp_path, embedded_scf_result):
    """The parsed FILES table must not redirect embedded SCF to host.pot_new."""
    parameters = embedded_scf_result.input_parameters

    original = Path(__file__).parent.parent / "examples" / "scf.out"
    output = tmp_path / "cluster_scf.out"
    output.write_bytes(original.read_bytes())

    reader = parameters.task_definition_required().output_reader(
        input_parameters=parameters, directory=str(tmp_path)
    )
    result = reader.read_from_file(output)

    assert result.is_embedded
    assert result.start_potential_filename == str(tmp_path / "cluster.pot")
    assert result.potential_filename == str(tmp_path / "cluster.pot_new")
    assert os.fspath(result.files["converged"]()) == str(tmp_path / "cluster.pot_new")


@pytest.mark.parametrize(
    ("converged", "return_code", "expected_status"),
    [
        (True, 0, "CONVERGED"),
        (False, 0, "ITERATING"),
        (None, 0, "ITERATING"),
        (True, 139, "ITERATING"),
    ],
)
def test_parsed_embedded_scf_result_updates_in_memory_convergence_only(
    tmp_path, monkeypatch, embedded_scf_result, converged, return_code, expected_status,
):
    """Native CLUPOT may retain ITERATING despite a converged SCF OUT.

    Only a positive final iteration is authoritative. The result reader must
    not touch the original native file (or the separate host potential).
    """
    result = embedded_scf_result
    result.return_code = return_code
    if converged is not None:
        result.iterations = [{"converged": lambda: converged}]

    read_potential = result._read_potential

    def read_with_atoms(filename):
        cluster = read_potential(filename)
        assert cluster.atoms.info["sprkkr_embedding_scf_status"] == "ITERATING"
        return cluster

    monkeypatch.setattr(result, "_read_potential", read_with_atoms)
    cluster = result.potential
    assert result.potential is cluster
    assert cluster.SCF_INFO.SCFSTATUS() == expected_status
    assert cluster.atoms.info["sprkkr_embedding_scf_status"] == expected_status
    assert cluster.host.SCF_INFO.SCFSTATUS() == "CONVERGED"
    assert (tmp_path / "cluster.pot_new").read_bytes() == (
        POTENTIAL_EXAMPLES / "Fe3CoFe3.clu.iterating.pot"
    ).read_bytes()
    assert (tmp_path / "host.pot").read_bytes() == (POTENTIAL_EXAMPLES / "Fe3CoFe3.host.pot").read_bytes()


def test_scf_result_does_not_reinterpret_periodic_status(tmp_path):
    """Convergence reconciliation is specific to embedded result potentials."""
    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "GeTeX5.pot"
    result = ScfResult(parameters, None, tmp_path)
    result.iterations = [{"converged": lambda: True}]
    copyfile(POTENTIAL_EXAMPLES / "GeTeX5.pot", result.potential_filename)
    assert result.potential.SCF_INFO.SCFSTATUS() == "ITR-BULK"
