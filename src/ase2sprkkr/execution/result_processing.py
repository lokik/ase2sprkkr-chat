"""Task-agnostic protocol for durable calculation result reconstruction."""

from __future__ import annotations

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ExecutionOutcome
    from .manager import RunRecord


class ResultProcessor(Protocol):
    """Reconstruct results from recorded metadata and completed execution.

    Implementations live in their respective calculation domains. Execution
    keeps only their stable names in RunRecord.
    """

    def process(self, record: RunRecord, outcome: ExecutionOutcome) -> Any:
        """Parse one completed or reattached calculation."""

    def record_updated(self, result: Any, record: RunRecord) -> None:
        """Bind the latest durable record after processing completes."""
