import os
from pathlib import Path

import pytest

from ..readers import scf as scf_reader
from ..readers.scf import ScfOutputParser, ScfResult
from ..task_result import TaskResult


def test_scf_output_without_input_file_preserves_result_semantics(tmp_path):
    source = Path(__file__).parent.parent / "examples" / "scf.out"
    output = tmp_path / "scf.out"
    output.write_bytes(source.read_bytes())

    result = TaskResult.from_file(output)

    assert isinstance(result, ScfResult)
    assert result.input_parameters is None
    assert result.task_name == "scf"
    assert result.produces_potential is True


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



def test_embedded_scf_result_reads_clupot_and_continues_with_dos(tmp_path):
    """SCF output must not replace CLUPOT by the unchanged periodic host."""
    from ase.build import bulk
    from ...input_parameters.input_parameters import InputParameters
    from ...potentials.potentials import Potential
    from ...potentials.cluster_potentials import ClusterPotential
    from ...sprkkr.sprkkr_atoms import SPRKKRAtoms
    from ...sprkkr.executables import executables

    host = Potential.from_atoms(bulk("Cu"))
    host_file = tmp_path / "host.pot"
    host.save_to_file(host_file)

    atoms = SPRKKRAtoms.from_embedding(host, [(0, (1, 0, 0), "Fe")])
    cluster = ClusterPotential.from_atoms(atoms)
    start = tmp_path / "cluster.pot"
    converged = tmp_path / "cluster.pot_new"
    cluster.save_to_file(start)
    # Use a native CLUPOT fixture for both input and output. Numerical radial
    # convergence is covered by the independent CLUPOT radial round-trip tests.
    cluster.save_to_file(converged)

    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"
    parameters.CONTROL.CLUTYPE = "embedded"

    result = ScfResult(parameters, None, tmp_path)
    assert result.start_potential_filename == str(start)
    assert result.potential_filename == str(converged)
    assert result.is_embedded
    assert isinstance(result.start_potential, ClusterPotential)
    assert isinstance(result.potential, ClusterPotential)
    assert result.potential.host is result.start_potential.host
    assert result.atoms.sites[0].occupation.as_dict == {"Fe": 1.0}
    assert result.atoms.sites[0].embedding.cell_index == (1, 0, 0)
    assert result.calculator.potential is result.potential

    output_paths = {
        os.fspath(item) for item in parameters.output_files(str(tmp_path))
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
    from ...input_parameters.input_parameters import InputParameters

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



def test_embedded_scf_parser_registers_converged_cluster_not_host(tmp_path):
    """The parsed FILES table must not redirect embedded SCF to host.pot_new."""
    from ...input_parameters.input_parameters import InputParameters

    parameters = InputParameters.create("SCF")
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"
    parameters.CONTROL.CLUTYPE = "embedded"

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
    assert result.files["converged"]() == str(tmp_path / "cluster.pot_new")
