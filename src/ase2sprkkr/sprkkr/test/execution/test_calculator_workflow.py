from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace

import pytest
from ase.build import bulk
from ase.calculators.calculator import CalculationFailed, PropertyNotImplementedError
from kkrunner.runs import RunState

from ase2sprkkr.outputs.task_result import TaskResult
from ase2sprkkr.outputs.readers.scf import ScfResult
from ase2sprkkr.sprkkr.calculator import SPRKKR


class _Runner:

    def __init__(self, result=None):
        self.result = result
        self.ran = False
        self.state = RunState.FINISHED

    def run(self):
        self.ran = True
        return self.result


@pytest.mark.parametrize('case', ('standard', 'explicit-file-names', 'project'))
def test_create_run_only_prepares_runner(tmp_path, monkeypatch, case):
    project = object()
    if case == 'standard':
        calculator = SPRKKR(atoms=bulk('Al'), directory=tmp_path, mpi=False, empty_spheres=False)
        options = dict(input_file='custom.inp', potential_file='custom.pot', create_subdirs=True)
    elif case == 'explicit-file-names':
        calculator = SPRKKR(directory=False, empty_spheres=False)
        options = dict(input_file='custom.inp', potential_file='custom.pot', output_file='custom.out', directory=False)
    else:
        calculator = SPRKKR(directory=tmp_path)
        options = dict(project=project)
    runner = _Runner()
    seen = {}

    def create_run(**kwargs):
        seen['run'] = kwargs
        return runner
    prepared = SimpleNamespace(create_run=create_run)

    def prepare(**kwargs):
        seen['prepare'] = kwargs
        return prepared
    monkeypatch.setattr(calculator, 'plan', prepare)
    created = calculator.create_run(**options)
    assert created is runner
    assert not runner.ran
    if case == 'project':
        assert seen['run']['project'] is project
    else:
        for name, value in options.items():
            assert seen['prepare'][name] == value
    assert seen['run']['result_context'] is calculator


@pytest.fixture
def scf_result(tmp_path):
    package = Path(__file__).resolve().parents[3]
    output = tmp_path / "scf.out"
    copyfile(package / "outputs/examples/scf.out", output)
    result = TaskResult.from_file(output)
    potential = tmp_path / "FePt.pot_new"
    copyfile(package / "potentials/examples/FePt.pot", potential)
    result.files.add_file("converged", potential)
    assert isinstance(result, ScfResult)
    assert len(result.iterations) > 0
    return result


def test_calculate_runs_and_updates_state(monkeypatch, scf_result):
    calculator = SPRKKR(atoms=bulk('Al'), mpi=False, empty_spheres=False)
    result = scf_result
    new_potential = result.potential
    energy = result.energy
    runner = _Runner(result)
    callback_state = []
    monkeypatch.setattr(calculator, 'create_run', lambda **kwargs: runner)
    returned = calculator.calculate(callback=lambda out: callback_state.append((out, calculator.results['energy'], calculator._potential)))
    assert returned is result
    assert runner.ran
    assert calculator.results['energy'] == energy
    assert calculator._potential is new_potential
    assert callback_state == [(result, energy, new_potential)]


def test_calculate_does_not_adopt_potential_when_explicit_atoms_are_used(monkeypatch, scf_result):
    calculator = SPRKKR(atoms=bulk('Al'), mpi=False, empty_spheres=False)
    old_potential = calculator.potential
    result = scf_result
    runner = _Runner(result)
    monkeypatch.setattr(calculator, 'create_run', lambda **kwargs: runner)
    calculator.calculate(atoms=bulk('Al'))
    assert calculator._potential is old_potential


@pytest.mark.parametrize("task", ["SCF", "DOS"])
def test_calculate_without_completed_scf_iteration_has_no_energy(tmp_path, monkeypatch, task):
    calculator = SPRKKR(atoms=bulk("Al"), mpi=False, empty_spheres=False)
    result_class = ScfResult if task == "SCF" else TaskResult
    result = result_class(None, None, tmp_path, task_name=task)
    monkeypatch.setattr(calculator, "create_run", lambda **kwargs: _Runner(result))
    assert calculator.calculate(atoms=bulk("Al")) is result
    assert "energy" not in calculator.results


def test_calculate_reports_failed_process_without_updating_state(tmp_path):
    executable = tmp_path / 'kkrscf'
    executable.write_text('#!/bin/sh\nprintf "deliberate failure\\n" >&2\nexit 7\n')
    executable.chmod(0o755)
    calculator = SPRKKR(
        atoms=bulk('Al'), directory=tmp_path, mpi=False,
        empty_spheres=False, executable_dir=tmp_path, executable_suffix='',
    )
    calculator.results['energy'] = 1.25
    potential = calculator.potential
    callbacks = []

    with pytest.raises(CalculationFailed, match=r'failed.*return code 7'):
        calculator.calculate(callback=callbacks.append)

    assert calculator.results['energy'] == 1.25
    assert calculator._potential is potential
    assert callbacks == []
    assert 'deliberate failure' in next(tmp_path.glob('*.err')).read_text()


@pytest.mark.parametrize('name', ['forces', 'energy'])
def test_missing_property_raises_ase_error(monkeypatch, name):
    calculator = SPRKKR()
    monkeypatch.setattr(
        calculator, 'calculate', lambda **kwargs: SimpleNamespace(task_name='DOS'),
    )
    with pytest.raises(PropertyNotImplementedError, match=name):
        calculator.get_property(name)
