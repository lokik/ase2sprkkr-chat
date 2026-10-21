"""Normalized outcome paths are a snapshot of the launch specification."""

from pathlib import Path

import pytest

from ..base import ExecutionOutcome, ExecutionSpec


class NamedStream:
    def __init__(self, name):
        self.name = name
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1


def _spec(tmp_path, *, input_file=None, output_file=None, error_file=None):
    return ExecutionSpec(
        argv=("unused-command",),
        directory=str(tmp_path),
        input_file=input_file,
        output_file=output_file,
        error_file=error_file,
    )


def test_outcome_normalizes_paths_once_without_mutating_original_spec(tmp_path):
    output = NamedStream("run.out")
    spec = _spec(
        tmp_path,
        input_file=Path("run.in"),
        output_file=output,
        error_file=None,
    )

    outcome = ExecutionOutcome.from_spec(spec)

    assert outcome.input_file == str(tmp_path / "run.in")
    assert outcome.output_file == str(tmp_path / "run.out")
    assert outcome.error_file is None
    assert outcome.return_code is None
    assert output.flush_count == 0
    assert spec.output_file is output

    # Later changes to an open handle must not alter the published path.
    output.name = "different.out"
    assert outcome.output_file == str(tmp_path / "run.out")


def test_outcome_preserves_managed_flush_and_strict_path_semantics(tmp_path):
    output = NamedStream("out")
    error = NamedStream("err")
    spec = _spec(tmp_path, output_file=output, error_file=error)

    outcome = ExecutionOutcome.from_spec(spec, flush=True, strict_paths=True)

    assert (outcome.output_file, outcome.error_file) == (
        str(tmp_path / "out"), str(tmp_path / "err"),
    )
    assert outcome.input_file is None
    assert (output.flush_count, error.flush_count) == (1, 1)


def test_outcome_strictness_only_rejects_unnamed_present_streams(tmp_path):
    spec = _spec(tmp_path, output_file=object())
    assert ExecutionOutcome.from_spec(spec).output_file is None
    with pytest.raises(ValueError, match="named file or path"):
        ExecutionOutcome.from_spec(spec, flush=True, strict_paths=True)
