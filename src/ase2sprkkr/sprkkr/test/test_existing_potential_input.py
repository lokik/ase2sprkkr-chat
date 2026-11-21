"""Existing POTFIL/CLUPOT sources must be stable and staged as one input."""

from pathlib import Path
from shutil import copyfile

import pytest

import ase2sprkkr
from ase2sprkkr import InputParameters
from ase2sprkkr.sprkkr.input_sources import (
    existing_potential_files,
    load_existing_potential_input,
)


@pytest.fixture
def embedded_potential_files(tmp_path):
    """Real independent periodic host and embedded cluster inputs."""
    examples = Path(ase2sprkkr.__file__).parent / "potentials" / "examples"
    host, cluster = tmp_path / "host.pot", tmp_path / "cluster.pot"
    copyfile(examples / "Fe3CoFe3.host.pot", host)
    copyfile(examples / "Fe3CoFe3.clu.pot", cluster)
    return host, cluster


def test_embedded_input_loads_and_declares_both_potentials(tmp_path, embedded_potential_files):
    params = InputParameters.for_task("DOS")
    params.CONTROL.CLUTYPE = "embedded"
    params.CONTROL.POTFIL = "host.pot"
    params.CONTROL.POTFIL_CLU = "cluster.pot"
    source = load_existing_potential_input(params, tmp_path)

    assert len(source.versions) == 2
    assert {version.path for version in source.versions} == {
        tmp_path / "host.pot", tmp_path / "cluster.pot",
    }
    assert all(version.is_current() for version in source.versions)
    assert source.parameters.CONTROL.POTFIL() == str(tmp_path / "host.pot")
    assert source.parameters.CONTROL.POTFIL_CLU() == str(tmp_path / "cluster.pot")
    assert params.CONTROL.POTFIL() == "host.pot"
    from kkrunner import TaskFile

    inputs = source.parameters.input_files(tmp_path)
    assert all(isinstance(item, TaskFile) for item in inputs)
    assert str(tmp_path / "cluster.pot") in tuple(map(str, inputs))
    assert source.potential.atoms.embedding_host is not None

    (tmp_path / "host.pot").write_text("overwritten")
    assert not source.versions[0].is_current()


def test_existing_input_requires_both_embedded_files(tmp_path):
    params = InputParameters.for_task("SCF")
    params.CONTROL.CLUTYPE = "embedded"
    params.CONTROL.POTFIL = "host.pot"
    params.CONTROL.POTFIL_CLU = "cluster.pot"
    (tmp_path / "host.pot").write_text("unimportant")
    with pytest.raises(ValueError, match="POTFIL_CLU"):
        existing_potential_files(params, tmp_path)


def test_cluster_mode_must_match_structure(tmp_path):
    from ase import Atoms

    params = InputParameters.for_task("SCF")
    params.CONTROL.CLUTYPE = "embedded"
    with pytest.raises(ValueError, match="embedded atoms"):
        params.set_from_atoms(Atoms("Fe", cell=[2, 2, 2], pbc=True))


def test_existing_source_loading_is_not_task_applicability_validation(tmp_path, embedded_potential_files):
    """Reading a valid host/CLUPOT is distinct from launching a supported task."""
    from ase2sprkkr import SPRKKR

    parameters = InputParameters.for_task("CHILANDAU")
    parameters.CONTROL.CLUTYPE = "embedded"
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"

    source = load_existing_potential_input(parameters, tmp_path)
    assert source.potential.atoms.embedding_host is not None
    with pytest.raises(ValueError, match="embedded"):
        SPRKKR().prepare(
            atoms=source.potential.atoms, potential=False,
            input_parameters=source.parameters, directory=tmp_path,
            input_file="unsupported.inp", empty_spheres=False,
        )
    assert not (tmp_path / "unsupported.inp").exists()


def test_source_loading_rejects_potential_changed_during_parse(
    tmp_path, embedded_potential_files, monkeypatch,
):
    """Loading must not attach structure parsed from an earlier file version."""
    from ase2sprkkr.potentials.cluster_potentials import EmbeddedPotential

    host, _cluster = embedded_potential_files
    parameters = InputParameters.for_task("DOS")
    parameters.CONTROL.CLUTYPE = "embedded"
    parameters.CONTROL.POTFIL = "host.pot"
    parameters.CONTROL.POTFIL_CLU = "cluster.pot"

    parse = EmbeddedPotential.from_file

    def parse_and_replace_host(*args, **kwargs):
        parsed = parse(*args, **kwargs)
        host.write_text("host overwritten while loading cluster")
        return parsed

    monkeypatch.setattr(EmbeddedPotential, "from_file", parse_and_replace_host)
    with pytest.raises(ValueError, match="changed while being loaded"):
        load_existing_potential_input(parameters, tmp_path)
