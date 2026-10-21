"""Lifecycle persistence hooks for one calculation.

These callbacks participate in submission/finalization transactions. Unlike
best-effort output observers, failure must be reported to the caller.
"""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .postprocessing import CalculationRun


class PersistenceError(RuntimeError):
    """Durable persistence failed; the calculation result is unaffected."""


class CalculationRunHooks(Protocol):
    """Transaction boundaries invoked by CalculationRun for a single run."""

    def before_submit(self, run: "CalculationRun") -> None: ...
    def prepared(self, run: "CalculationRun") -> None: ...
    def submitted(self, run: "CalculationRun") -> None: ...
    def submission_failed(
        self, run: "CalculationRun", error: BaseException, *,
        uncertain: bool
    ) -> None: ...
    def completed(self, run: "CalculationRun") -> None: ...
