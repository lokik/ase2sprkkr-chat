"""SCF native algorithm and input-domain regressions."""
import io
import pickle
import pytest
from ..input_parameters import InputParameters
from ...common.warnings import DataValidityError, DataValidityErrors


def parameters():
    p = InputParameters.create("SCF")
    p.CONTROL.POTFIL.set("Fe.pot")
    return p


@pytest.mark.parametrize("algorithm", ["BROYDEN2", "ANDERSON", "TCHEBY"])
def test_algorithms_roundtrip(algorithm):
    p = parameters()
    p.SCF.ALG.set(algorithm)
    q = p._definition.read_from_file(io.StringIO(p.to_string()))
    assert q.SCF.ALG() == algorithm
    assert pickle.loads(pickle.dumps(q)).to_string() == q.to_string()
    assert p.task_definition.executable == "kkrscf" and p.task_definition.mpi is True


def test_defaults_preserved():
    p = parameters()
    assert p.SCF.NITER() == 200 and p.SCF.MIX() == 0.2
    assert p.SCF.TOL() == 1e-5 and p.SCF.ITDEPT() == 40
    assert p.SCF.ALG() == "BROYDEN2"
    assert p.ENERGY.GRID().tolist() == ["5"]


@pytest.mark.parametrize("option,value", [
    ("SCF.ALG", "PULAY"), ("SCF.NITER", 0),
    ("SCF.ISTBRY", 0), ("SCF.ITDEPT", 2), ("SCF.ITDEPT", 41),
    ("SCF.MIX", float("nan")), ("SCF.TOL", 0.),
    ("SCF.EFGUESS", float("inf")), ("ENERGY.NE", [0]),
])
def test_invalid(option, value):
    p = parameters()
    with pytest.raises((DataValidityError, DataValidityErrors, ValueError)):
        p.set({option: value})
        p.validate("save")


def test_single_point_rectangular_mesh():
    p = parameters()
    p.ENERGY.GRID.set([3])
    p.ENERGY.NE.set([1])
    with pytest.raises((DataValidityError, DataValidityErrors), match="NE>=2"):
        p.validate("save")


def test_tcheby_fullpot():
    p = parameters()
    p.SCF.ALG.set("TCHEBY")
    p.SCF.FULLPOT.set(True)
    with pytest.raises((DataValidityError, DataValidityErrors), match="TCHEBY"):
        p.validate("save")


@pytest.mark.parametrize("option, expected", [("IQCNTR", 2), ("ITCNTR", 3)])
def test_cluster_centre_object_references_are_written_as_native_numbers(option, expected):
    from ...sprkkr.sprkkr_atoms import SPRKKRAtoms

    atoms = SPRKKRAtoms(
        "FeNi", positions=[[0, 0, 0], [1, 1, 1]], cell=[3, 3, 3],
        pbc=True, symmetry=False,
    )
    atoms.sites[0].occupation = {"Fe": 0.5, "Co": 0.5}
    value = (atoms.sites[1] if option == "IQCNTR"
             else atoms.sites[1].occupation.atomic_type("Ni"))
    p = parameters()
    p.TAU[option].set(value)
    output = io.StringIO()
    p.save_to_file(output, atoms)
    parsed = p._definition.read_from_string(output.getvalue())
    assert parsed.TAU[option]() == expected
    # Replacing a reference with a numeric input must still work.
    p.TAU[option].set(1)
    output = io.StringIO()
    p.save_to_file(output, atoms)
    assert p._definition.read_from_string(output.getvalue()).TAU[option]() == 1


def test_cluster_site_reference_follows_2d_potfil_order_on_each_write():
    from ...sprkkr.atoms_region import AtomsRegion
    from ...sprkkr.sprkkr_atoms import SPRKKRAtoms

    atoms = SPRKKRAtoms(
        "Fe4", positions=[[0, 0, 1], [0, 0, 0], [0, 0, 2], [0, 0, 4]],
        cell=[2, 2, 6], pbc=(True, True, False), symmetry=False,
    )
    atoms.set_regions([
        AtomsRegion("left", slice(0, 2), [2, 2, 2]),
        AtomsRegion("central", slice(2, 3), [2, 2, 2], pbc=(True, True, False)),
        AtomsRegion("right", slice(3, 4), [2, 2, 2]),
    ])
    p = parameters()
    p.TAU.IQCNTR = atoms.sites[0]
    p.set_from_atoms(atoms)
    assert p.TAU.IQCNTR.result == 2

    atoms.positions[0, 2] = -1
    p.set_from_atoms(atoms)
    assert p.TAU.IQCNTR.result == 1
