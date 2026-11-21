"""Native BSF grid and K-K denominator regressions."""
import io
import pickle
import pytest
from ..input_parameters import InputParameters
from ...common.warnings import DataValidityError, DataValidityErrors


def parameters():
    p = InputParameters.create("BSFKK")
    p.CONTROL.POTFIL.set("Fe.pot")
    p.TASK.NK1.set(10)
    p.TASK.NK2.set(10)
    p.TASK.K1.set([1., 0., 0.])
    p.TASK.K2.set([0., 1., 0.])
    return p


def test_kk_grid_roundtrip():
    p = parameters()
    q = p._definition.read_from_file(io.StringIO(p.to_string()))
    assert q.ENERGY.GRID().tolist() == ["0"]
    assert q.ENERGY.NE().tolist() == [1]
    assert q.ENERGY.EMAX() == q.ENERGY.EMIN()
    assert pickle.loads(pickle.dumps(q)).to_string() == q.to_string()


def test_ek_defaults_preserved():
    p = InputParameters.create("BSFEK")
    assert p.ENERGY.GRID().tolist() == ["3"]
    assert p.ENERGY.EMAX() == -1.0


def test_nonzero_origin_endpoints():
    p = parameters()
    p.TASK.KA.set([[2., 2., 0.]])
    p.TASK.K1.set([3., 2., 0.])
    p.TASK.K2.set([2., 3., 0.])
    p.validate("save")


@pytest.mark.parametrize("option,value", [
    ("TASK.NK1", 1), ("TASK.NK2", 1),
    ("TASK.NK1", 0), ("TASK.NK2", -1),
    ("TASK.K1", [0., 0., 0.]),
    ("TASK.K2", [2., 0., 0.]),
    ("TASK.K2", [float("nan"), 1., 0.]),
    ("ENERGY.NE", [0]),
])
def test_invalid(option, value):
    p = parameters()
    with pytest.raises((DataValidityError, DataValidityErrors, ValueError)):
        p.set({option: value})
        p.validate("save")


@pytest.mark.parametrize("initial, points, effective", [
    ("BSFEK", 1, "BSFKK"),
    ("BSFKK", 200, "BSFEK"),
])
def test_switching_bsf_mode_does_not_warn_about_requested_alias(
    initial, points, effective
):
    import warnings

    p = InputParameters.create(initial)
    p.CONTROL.POTFIL.set("Fe.pot")
    p.ENERGY.NE.set([points])
    if effective == "BSFKK":
        p.TASK.NK1.set(10)
        p.TASK.NK2.set(10)
        p.TASK.K1.set([1., 0., 0.])
        p.TASK.K2.set([0., 1., 0.])
    else:
        p.TASK.KPATH.set(1)

    # The initial alias stays as provenance; only ENERGY.NE selects the task.
    assert p.requested_task_name == initial
    assert p.task_name == effective
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p.validate("set")
        p.validate("save")
        p.copy().validate("save")

    assert not any(
        "ENERGY.NE selects BSF-" in str(warning.message)
        for warning in caught
    )
