"""Invariants for the task-independent progress payload."""
import pytest
from ase2sprkkr.outputs.progress import ProgressInfo


def test_unknown_total_is_valid_but_has_no_percentage():
    assert ProgressInfo(12, None).total is None


@pytest.mark.parametrize("current,total", [(-1, None), (1, 0), (3, 2)])
def test_progress_rejects_invalid_counts(current, total):
    with pytest.raises(ValueError):
        ProgressInfo(current, total)
