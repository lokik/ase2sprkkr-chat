"""Requirements used by the SSH/Slurm Docker integration image."""

import pytest
from packaging.requirements import Requirement

from . import conftest as execution_fixtures


def test_docker_kkrunner_requirement_follows_project_metadata(monkeypatch):
    def distribution_requirements(name):
        assert name == "ase2sprkkr"
        return [
            "packaging",
            "kkrunner>=0.3.0b1,<0.4",
            'kkrunner[psij]>=0.3.0b1,<0.4; extra == "psij"',
        ]

    monkeypatch.setattr(execution_fixtures, "distribution_requires", distribution_requirements)

    requirement = Requirement(execution_fixtures._declared_kkrunner_requirement())
    assert requirement.name == "kkrunner"
    assert requirement.extras == {"psij"}
    assert requirement.specifier == Requirement("kkrunner>=0.3.0b1,<0.4").specifier
    assert not requirement.specifier.contains("0.2.0b7")


def test_docker_kkrunner_requirement_rejects_missing_runtime_dependency(monkeypatch):
    monkeypatch.setattr(
        execution_fixtures,
        "distribution_requires",
        lambda _name: ['kkrunner[psij]>=0.3.0b1,<0.4; extra == "psij"'],
    )

    with pytest.raises(RuntimeError, match="unconditional KKRunner"):
        execution_fixtures._declared_kkrunner_requirement()
