from types import SimpleNamespace

import pytest

from ase2sprkkr.sprkkr.atomic_types import AtomicType


def make_atom():
    return AtomicType(
        "Fe",
        n_electrons=26,
        n_core=18,
        n_valence=8,
        n_semicore=0,
    )


def test_electron_counts_accept_consistent_values_and_reject_inconsistent_ones():
    """Exercise successive setters on one atom and check cross-field invariants."""
    atom = make_atom()
    expected = (26, 18, 8, 0)
    attributes = ("n_electrons", "n_core", "n_valence", "n_semicore")
    invalid = (25, 17, 7, 1)

    # Check constructor values before testing the setters.
    assert tuple(getattr(atom, name) for name in attributes) == expected

    for name, value, invalid_value in zip(attributes, expected, invalid):
        setattr(atom, name, value)
        # A successful assignment must not change any other electron count.
        assert tuple(getattr(atom, attr) for attr in attributes) == expected, f"accepted {name}"

        with pytest.raises(ValueError):
            setattr(atom, name, invalid_value)

        # A failed assignment must leave every electron count unchanged.
        assert tuple(getattr(atom, attr) for attr in attributes) == expected, f"rejected {name}"


def test_invalid_n_electrons_does_not_update_potential_z():
    atom = make_atom()
    atom._potential = SimpleNamespace(z=26)

    with pytest.raises(ValueError):
        atom.n_electrons = 25

    assert atom.n_electrons == 26
    assert atom._potential.z == 26
