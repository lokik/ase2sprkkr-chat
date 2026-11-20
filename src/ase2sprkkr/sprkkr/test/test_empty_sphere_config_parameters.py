"""Shared empty-sphere defaults and explicit override precedence."""
from types import SimpleNamespace

import pytest

from ase2sprkkr.bindings.empty_spheres import configured_empty_spheres_params
from ase2sprkkr.configuration import definition
from ase2sprkkr.sprkkr import _preparation as preparation


def _search(monkeypatch, empty_spheres, *, config=None, params=None, default=None):
    calls = []
    atoms = object()
    monkeypatch.setattr(
        preparation, "add_empty_spheres",
        lambda candidate, **kwargs: calls.append((candidate, kwargs)),
    )
    preparation.resolve_empty_spheres(
        input_parameters=SimpleNamespace(task_name="SCF"),
        atoms=atoms,
        empty_spheres=empty_spheres,
        default_empty_spheres=default,
        empty_spheres_params=params,
        configured_empty_spheres_params=config or {},
    )
    return [kwargs for _atoms, kwargs in calls]


def test_configuration_stores_only_explicit_values():
    configuration = definition.create_object()
    assert configured_empty_spheres_params(configuration) == {}
    section = configuration.running.empty_spheres_params
    section.min_radius.set(0.8)
    section.mesh.set((20, 22, 24))
    assert configured_empty_spheres_params(configuration) == {
        "min_radius": 0.8, "mesh": (20, 22, 24),
    }
    section.min_radius.set(None)
    assert configured_empty_spheres_params(configuration) == {
        "mesh": (20, 22, 24),
    }


def test_config_applies_to_ordinary_enabled_search(monkeypatch):
    assert _search(
        monkeypatch, True, config={"min_radius": 0.8, "mesh": (20, 22, 24)}
    ) == [{"min_radius": 0.8, "mesh": (20, 22, 24), "method": "inhouse"}]


def test_legacy_mapping_and_tuple_merge_over_config(monkeypatch):
    config = {"min_radius": 0.8, "max_radius": 1.8, "mesh": (20, 22, 24)}
    assert _search(
        monkeypatch, {"max_radius": 2.1}, config=config
    ) == [{
        "min_radius": 0.8, "max_radius": 2.1, "mesh": (20, 22, 24),
        "method": "inhouse",
    }]
    assert _search(
        monkeypatch, (0.6, 2.2), config=config
    ) == [{
        "min_radius": 0.6, "max_radius": 2.2, "mesh": (20, 22, 24),
        "method": "inhouse",
    }]


def test_per_call_params_merge_with_configuration(monkeypatch):
    assert _search(
        monkeypatch, True,
        config={"min_radius": 0.8, "mesh": (20, 22, 24)},
        params={"max_radius": 1.7},
    ) == [{
        "min_radius": 0.8, "mesh": (20, 22, 24), "max_radius": 1.7,
        "method": "inhouse",
    }]


def test_library_sentinels_bypass_user_parameters(monkeypatch):
    config = {"min_radius": 0.8, "max_radius": 1.8}
    assert _search(monkeypatch, "library", config=config) == [{}]
    assert _search(monkeypatch, True, config=config, params="library") == [{}]


@pytest.mark.parametrize("legacy", [{"min_radius": 0.8}, (0.7, 1.8), "library"])
def test_conflicting_explicit_arguments_raise_value_error(monkeypatch, legacy):
    with pytest.raises(ValueError, match="either empty_spheres"):
        _search(monkeypatch, legacy, params={"max_radius": 2.0})


def test_bad_radius_tuple_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="tuple must contain"):
        _search(monkeypatch, (0.6,))


def test_auto_and_disabled_search_remain_unchanged(monkeypatch):
    assert _search(monkeypatch, False, config={"mesh": (20, 20, 20)}) == []

def test_auto_search_respects_configuration(monkeypatch):
    calls = []
    class AtomsWithStartingPotential:
        sites = ()

        def has_potential(self):
            return False

    monkeypatch.setattr(
        preparation, "add_empty_spheres",
        lambda atoms, **kwargs: calls.append(kwargs),
    )
    preparation.resolve_empty_spheres(
        input_parameters=SimpleNamespace(task_name="SCF"),
        atoms=AtomsWithStartingPotential(),
        empty_spheres="auto",
        default_empty_spheres="auto",
        configured_empty_spheres_params={"min_radius": 0.75},
    )
    assert calls == [{"min_radius": 0.75, "method": "inhouse"}]


def test_incompatible_finder_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="incompatible"):
        _search(
            monkeypatch, {"method": "es_finder"},
            config={"min_radius": 0.75},
        )


@pytest.mark.parametrize("legacy", [{"min_radius": 0.8}, (0.7, 1.8), "library"])
def test_calculator_constructor_rejects_conflicting_arguments(legacy):
    from ase2sprkkr import SPRKKR

    with pytest.raises(ValueError, match="either empty_spheres"):
        SPRKKR(
            empty_spheres=legacy,
            empty_spheres_params={"max_radius": 2.0},
        )


def test_calculator_defaults_do_not_conflict_with_explicit_per_call_policy(monkeypatch):
    """Per-call legacy parameters replace calculator defaults, not vice versa."""
    calls = []
    monkeypatch.setattr(
        preparation, "add_empty_spheres",
        lambda atoms, **kwargs: calls.append(kwargs),
    )
    preparation.resolve_empty_spheres(
        input_parameters=SimpleNamespace(task_name="SCF"),
        atoms=object(),
        empty_spheres={"min_radius": 0.9},
        default_empty_spheres=True,
        default_empty_spheres_params={"max_radius": 1.5},
        configured_empty_spheres_params={"mesh": (20, 20, 20)},
    )
    assert calls == [{
        "min_radius": 0.9, "mesh": (20, 20, 20), "method": "inhouse",
    }]
