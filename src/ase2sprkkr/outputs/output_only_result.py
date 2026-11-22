"""Inspectable historical output that cannot be parsed as scientific data."""

from __future__ import annotations

from typing import Any

from .task_result import TaskResult


class OutputOnlyResult(TaskResult):
    """Expose a verified historical OUT without inventing scientific values.

    The run record stays attached so normal OutputFileOption operations verify
    the saved file version before reading it.
    """

    produces_potential = False

    def __init__(self, record: Any, parse_error: Exception) -> None:
        files = record.files
        if files is None or files.output_file is None:
            raise FileNotFoundError("The run has no recorded output file")
        metadata = record.metadata.get("ase2sprkkr", {})
        super().__init__(
            None,
            None,
            record.execution.directory.local,
            output_file=files.output_file,
            task_name=metadata.get("task") or "unknown",
        )
        self.run_record = record
        self.return_code = record.return_code
        self.parse_error = f"{type(parse_error).__name__}: {parse_error}"

    @property
    def input_parameters(self) -> None:
        return None

    @property
    def output_values(self) -> dict[str, Any]:
        return {"output": self.files["output"]}
