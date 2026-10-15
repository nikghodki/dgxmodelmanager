from __future__ import annotations

from dataclasses import dataclass

from dgxmm.memory.budget import Budget
from dgxmm.memory.estimate import MemoryEstimate


@dataclass(frozen=True)
class FitResult:
    fits: bool
    required: int
    free_for_new: int
    headroom: int
    estimate: MemoryEstimate
    budget: Budget

    def as_dict(self) -> dict:
        return {"fits": self.fits, "required": self.required, "free_for_new": self.free_for_new,
                "headroom": self.headroom, "estimate": self.estimate.as_dict()}


def check_fit(estimate: MemoryEstimate, budget: Budget, extra_free: int = 0) -> FitResult:
    """`extra_free`: memory that will be released first (e.g. the deployment being replaced)."""
    free = budget.free_for_new + extra_free
    return FitResult(fits=estimate.total <= free, required=estimate.total, free_for_new=free,
                     headroom=free - estimate.total, estimate=estimate, budget=budget)
