from pathlib import Path
from types import SimpleNamespace

from ...potentials.potentials import Potential
from ..calculation import PreparedCalculation
from ..calculator import SPRKKR


EXAMPLES = Path(__file__).parents[2] / "potentials" / "examples"


def test_scf_status_serialization_follows_effective_task(tmp_path):
    # Parse FePt once. Check DOS conversion before any save can normalize
    # the potential, then exercise status resets between successive cases.
    potential = Potential.from_file(EXAMPLES / "FePt.new.pot")
    cases = (
        ("DOS", None, "START", "DOS", "CONVERGED"),
        ("DOS", None, "ITR-BULK", "DOS", "ITR-BULK"),
        ("SCF", None, "START", "SCF", "START"),
        ("SCF", "DOS", "START", "DOS", "CONVERGED"),
    )
    for index, (input_parameters, task, status, expected_task, expected_status) in enumerate(cases):
        directory = tmp_path / str(index)
        directory.mkdir()
        potential.SCF_INFO.SCFSTATUS = status
        calculator = SPRKKR()

        parameters, _, potential_file, _ = calculator.save_input(
            potential=potential,
            input_parameters=input_parameters,
            task=task,
            directory=directory,
            input_file="task.inp",
            potential_file="task.pot",
            output_file=False,
            empty_spheres=False,
        )

        saved = Potential.from_file(potential_file)
        case = f"{input_parameters=}, {task=}, {status=}"
        assert parameters.task_name.upper() == expected_task, case
        assert potential.SCF_INFO.SCFSTATUS() == status, case
        assert saved.SCF_INFO.SCFSTATUS() == expected_status, case


def test_calculate_serializes_status_before_mocked_execution(tmp_path, monkeypatch):
    potential = Potential.from_file(EXAMPLES / "FePt.new.pot")
    potential.SCF_INFO.SCFSTATUS = "START"
    result = SimpleNamespace(task_name="DOS")

    class Runner:
        ran = False

        def run(self):
            self.ran = True
            return result

    runner = Runner()

    def fake_create_run(prepared, **kwargs):
        assert Potential.from_file(prepared.potential_file).SCF_INFO.SCFSTATUS() == "CONVERGED"
        prepared.close()
        return runner

    monkeypatch.setattr(PreparedCalculation, "create_run", fake_create_run)

    calculator = SPRKKR(directory=tmp_path, mpi=False, empty_spheres=False)
    returned = calculator.calculate(
        potential=potential,
        input_parameters="SCF",
        task="DOS",
        input_file="task.inp",
        potential_file="task.pot",
        output_file=False,
    )

    assert returned is result
    assert runner.ran
    assert potential.SCF_INFO.SCFSTATUS() == "START"
    assert Potential.from_file(tmp_path / "task.pot").SCF_INFO.SCFSTATUS() == "CONVERGED"
