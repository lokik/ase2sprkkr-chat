from ase.build import bulk

from ..calculator import SPRKKR


def test_plan_preserves_explicit_false_file_sentinels(tmp_path):
    calculator = SPRKKR(
        atoms=bulk("Fe"),
        directory=tmp_path,
        input_file=True,
        potential_file=True,
        empty_spheres=False,
    )
    plan = calculator.plan(
        input_file=False,
        potential_file=False,
        output_file=False,
        allow_temporary_files=True,
    )
    try:
        assert plan.input_plan.temporary
        assert plan.potential_plan.temporary
        assert plan.output_plan is None
        assert not tuple(tmp_path.iterdir())
    finally:
        plan.close()
