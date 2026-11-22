"""A common structured progress event, independent of the SPR-KKR task."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressInfo:
    """Count *completed* work, not an index of work just entered.

    Unknown total prevents percentage display; estimated is an extrapolation.
    Label identifies a local (e.g. MPI-rank) counter; unit is display wording.
    """

    current: int
    total: int | None
    estimated: bool = False
    label: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.current < 0:
            raise ValueError("completed progress must not be negative")
        if self.total is not None and (self.total <= 0 or self.current > self.total):
            raise ValueError("completed progress must be within the total")
