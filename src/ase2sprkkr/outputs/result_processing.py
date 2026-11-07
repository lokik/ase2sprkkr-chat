"""SPR-KKR adapter for the generic execution result-processing protocol."""

from __future__ import annotations

from .task_result import TaskResult


class SprkkrResultProcessor:
    """Reconstruct SPR-KKR output using the existing task reader registry."""

    def process(self, record, outcome):
        return TaskResult.from_run_record(
            record,
            output_file=outcome.output_file,
            error_file=outcome.error_file,
            return_code=outcome.return_code,
        )

    def record_updated(self, result, record):
        result.run_record = record


SPRKKR_RESULT_PROCESSOR = SprkkrResultProcessor()
