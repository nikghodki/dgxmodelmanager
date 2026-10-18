"""The Decider interface the smart features depend on (contract.md), plus a NullDecider that never answers.

The real implementation (`dgxmm/decider/service.py`) is wired as `ctx.decider`; features only use this Protocol.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from dgxmm.decider import registry
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
# Deploy-time features (lane A1; corpora in tests/fixtures/decider_corpus/<feature>/, d3-mini, 2026-10-18): all at
# the default 0.8. Model precision on the questions DGXMM really asks: every accepted answer was right, but no Wilson
# lower bound reaches the 0.95 target (small samples; config/draft/engine labels are self-authored rubrics):
# - compat 0.8: 60 unlisted architectures, no answer >= 0.75 (the model says "unsupported" at 0.40-0.72, wrong for
#   the 3 the Transformers backend can serve), so the rules' "unknown" is shown.
# - config 0.8: 76 wizard cases, 10/10 at >= 0.8 (Wilson lo 0.72, coverage 0.13); 0.76 at 0.5.
# - draft 0.8: 10 multi-candidate targets, 6/6 at >= 0.8 (Wilson lo 0.61); the rules order matches as well.
# - engine 0.8: asked only when both formats fit; 25/25 at >= 0.8 (Wilson lo 0.87; one label, vllm).
# - remote_code 0.8: 29/29 "routine" on clean scans at >= 0.8 (Wilson lo 0.88); it can only add a warning.
# Operations features (docs/decider-evaluation.md, "Operations features"; d3-mini, measured 2026-10-17):
# - health 0.85: 77 five-minute windows (63 recorded on a throwaway deployment under induced load, 14 with injected
#   errors). Rules answer 64 (precision 1.00); on the 13 the rules leave open the model was right 13/13 at >= 0.75
#   and 12/12 at >= 0.85. On every window the model alone reaches the 95 % target only from 0.85 (0.96): below it,
#   it calls a single stray 5xx "error_creep", which the rules now treat as healthy.
# - reading 0.95: 49 readings (22 from the box's benchmark / calibration history, 27 induced). The model is not
#   reliable on the readings the rules leave open (first runs without history, GPU contention DGXMM cannot see):
#   precision 0.49 at 0.5 and 1.00 only from 0.95, where none of those readings qualifies. They stay unlabelled
#   (the recorded quality applies), which is the fail-open direction.
# - routing 0.8: 47 judged cases (34 with several candidates). At >= 0.8 the model was right 20/20 (Wilson lower
#   bound 0.84); with the deterministic ranking below it the pipeline matched the judged answer in 34/34.
# - tuning: rules only (no threshold of its own); tested on synthetic feedback histories.
# - command 0.7: 190 hand-written commands (corpus command/): top-1 precision 1.00 at every threshold from 0.3 (Wilson lower
#   bound 0.97 at 0.7, coverage 0.94); kept at 0.7 so genuinely ambiguous targets are offered as a choice.
# - schedule 0.8: 60 weekly-pattern cases; with the rules skipping one-week and short-window candidates, every model
#   answer on the rest was >= 0.82 and right (37/37); the model alone said "propose" almost always (precision 0.66).
# The values live in the feature registry (dgxmm/decider/registry.py); this is the same dict object (group -> default):
# diagnosis 0.8, parser 0.8, plan 0.6, kind 0.8, workload 0.75; health 0.85, reading 0.95, command 0.7, the rest 0.8.
DEFAULT_THRESHOLDS = registry.DEFAULT_THRESHOLDS


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
