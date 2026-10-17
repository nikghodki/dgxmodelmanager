"""The Decider interface the smart features depend on (contract.md), plus a NullDecider that never answers.

The real implementation (`dgxmm/decider/service.py`) is wired as `ctx.decider`; features only use this Protocol.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dgxmm.decider.types import Decision, Option

# Default per-feature confidence thresholds; the real Decider reads overrides from app_settings. Calibrated on the
# real corpora (docs/decider-evaluation.md, d3-mini @97dd054): a model answer is used only at/above its threshold.
# The thresholds were chosen on the same data they are reported on (no separate calibration split).
# - diagnosis 0.8: 16 real start-up failures, precision 1.00 at every threshold (coverage 0.94 at 0.8); kept at the
#   spike's calibrated 0.8 (>= 0.8 -> 98 % on the golden set) because the real sample is small.
# - parser 0.8 (was 0.9): auto-apply needs precision >= 95 %. Model-only on 189 real repo questions: precision 1.00
#   at >= 0.8 (47 answers) vs 0.95 at 0.75; at 0.9 only 12 % of answers qualified. The point estimate meets the
#   target; the Wilson 95 % lower bound (0.92) does not yet, so this is re-checked against the feedback log.
# - plan 0.6: no real corpus beyond the live checks (spike value).
# - kind 0.8: 140 real repos, model precision 0.99 at >= 0.8 (126/127, Wilson lower bound 0.96; coverage 0.92).
# - workload 0.75 (was 0.7): 24 windows of real traffic (6 profiles): every model answer >= 0.4 was right and the
#   correct non-mixed labels all scored >= 0.82; but on 60-minute production-like windows spanning several
#   profiles (truth "mixed") the model said batch_throughput at up to 0.71. 0.75 keeps all correct labels and none
#   of those (the old state put them at 0.78-0.81, wrong).
DEFAULT_THRESHOLDS = {"diagnosis": 0.8, "parser": 0.8, "plan": 0.6, "kind": 0.8, "workload": 0.75}


@runtime_checkable
class DeciderProtocol(Protocol):
    async def available(self) -> bool: ...
    def threshold(self, feature: str) -> float: ...
    def feature_enabled(self, feature: str) -> bool: ...
    async def decide(self, *, feature: str, subject: str, state: str, question: str, options: list[Option],
                     threshold: float | None = None, timeout_s: float = 2.0, log: bool = True) -> Decision: ...
    def record_rules(self, *, feature: str, subject: str, choice: str, reason: str | None,
                     options: list[Option] | None = None) -> Decision: ...


class NullDecider:
    """Decider used when no decision model is configured: never available, every question abstains."""

    async def available(self) -> bool:
        return False

    def threshold(self, feature: str) -> float:
        return DEFAULT_THRESHOLDS.get(feature, 0.8)

    def feature_enabled(self, feature: str) -> bool:
        return False

    async def decide(self, *, feature: str, subject: str, state: str, question: str, options: list[Option],
                     threshold: float | None = None, timeout_s: float = 2.0, log: bool = True) -> Decision:
        return Decision(feature=feature, subject=subject, source="none", choice=None, confidence=None)

    def record_rules(self, *, feature: str, subject: str, choice: str, reason: str | None,
                     options: list[Option] | None = None) -> Decision:
        return Decision(feature=feature, subject=subject, source="rules", choice=choice, confidence=1.0, reason=reason)


def decider_of(obj) -> DeciderProtocol:
    """`obj.decider` if set, else a NullDecider (fail open when the decider is not wired)."""
    d = getattr(obj, "decider", None)
    return d if d is not None else NullDecider()
