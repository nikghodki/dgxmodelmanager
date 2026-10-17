"""Ways to make a deployment fit (spec §5.4). Pure functions."""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from dgxmm.planner.fit import FitResult

GIB = 1024**3
MAX_STOP_PLANS = 4    # stop plans offered (single deployments and minimal combinations), best-ranked first
MAX_SLEEP_PLANS = 3
MAX_COMBO_SIZE = 3    # larger sets are only offered as the single greedy fallback plan
COMBO_CANDIDATES = 8  # combinations are drawn from the best-ranked candidates only (bounded work)
_QUANT_SUFFIX = re.compile(
    r"([-_.](nvfp4|mxfp4|fp8([-_]dynamic)?|fp4|awq|gptq|int4|int8|w4a16|w8a8|bnb[-_]4bit|4bit|8bit|gguf|"
    r"(ud[-_])?q\d(_[0-9]|_k)?(_[msl]|_xl)?|iq\d\w*|bf16|fp16))+$", re.I)  # Q8_0, Q4_K_M, UD-Q4_K_XL, IQ4_XS


class PlanAction(BaseModel):
    type: Literal["sleep", "stop"]
    deployment_id: int


@dataclass(frozen=True)
class ActiveDep:
    id: int
    name: str
    state: str
    priority: int
    reserved: int
    actual: int
    can_sleep: bool
    sleep_frees: int
    # ranking hints (filled from traffic facts by the deployer; the safety filter still decides what is safe)
    idle_min: float | None = None  # minutes since the last request activity; None = unknown / never recorded
    disruptive: bool = False       # serving, recently active, traffic unknown, starting, or others depend on it


@dataclass
class PlanOption:
    id: str
    title: str
    detail: str
    new_required: int
    frees: int
    fits: bool
    actions: list[PlanAction] = field(default_factory=list)
    patch: dict | None = None
    model_ref: str | None = None
    # decision-model annotations (dgxmm/decider/plans.py): set by the safety filter / recommender
    recommended: bool = False
    recommendation: dict | None = None  # {source, confidence, reason}
    unsafe_reason: str | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "detail": self.detail, "new_required": self.new_required,
                "frees": self.frees, "fits": self.fits, "actions": [a.model_dump() for a in self.actions],
                "patch": self.patch, "model_ref": self.model_ref, "recommended": self.recommended,
                "recommendation": self.recommendation, "unsafe_reason": self.unsafe_reason}


def base_model_name(ref: str) -> str:
    name = ref.rstrip("/").split("/")[-1]
    name = re.sub(r"\.gguf$", "", name, flags=re.I)
    return _QUANT_SUFFIX.sub("", name).lower()


def _gib(n: int) -> str:
    return f"{n / GIB:.1f} GiB"


def build_plans(*, fit: FitResult, cfg: dict, context_key: str, seqs_key: str, kv_patches: list[dict],
                estimate_with: Callable[[dict], int | None], active: list[ActiveDep], variants: list[str],
                estimate_ref: Callable[[str], int | None]) -> list[PlanOption]:
    free = fit.free_for_new
    shortfall = fit.required - free
    out: list[PlanOption] = []

    def reduction(pid: str, title: str, patch: dict, detail: str) -> None:
        req = estimate_with(patch)
        if req is not None:
            out.append(PlanOption(pid, title, detail, req, 0, req <= free, patch=patch))

    for p in kv_patches:
        reduction("kv", "Use a smaller KV-cache dtype", p, f"Set {', '.join(f'{k}={v}' for k, v in p.items())}; quality impact is usually negligible.")

    ctx = int(cfg.get(context_key) or 0)
    c, chosen = ctx // 2, None
    while c >= 4096:
        patch = {context_key: c, **({"kv_tokens": None} if cfg.get("kv_tokens") else {})}
        req = estimate_with(patch)
        if req is not None and req <= free:
            chosen = (patch, req)
            break
        c //= 2
    if chosen:
        out.append(PlanOption("context", f"Reduce context to {chosen[0][context_key]:,} tokens",
                              f"From {ctx:,}. Long prompts beyond this will be rejected.", chosen[1], 0, True, patch=chosen[0]))

    seqs = int(cfg.get(seqs_key) or 0)
    s = seqs // 2
    while s >= 1:
        req = estimate_with({seqs_key: s})
        if req is not None and req <= free:
            out.append(PlanOption("seqs", f"Serve {s} concurrent sequences instead of {seqs}",
                                  "Lower throughput under load; single-user latency unchanged.", req, 0, True, patch={seqs_key: s}))
            break
        s //= 2

    for ref in variants:
        req = estimate_ref(ref)
        if req is not None:
            out.append(PlanOption(f"variant:{ref}", f"Use {ref} instead", "Smaller quantization already on disk.",
                                  req, 0, req <= free, model_ref=ref))

    out += eviction_plans(active, shortfall, fit.required)
    out.sort(key=lambda p: not p.fits)  # stable: fitting options first, original order otherwise
    return out


def _stop_frees(d: ActiveDep) -> int:
    return d.actual if d.state == "sleeping" else max(d.reserved, d.actual)


def _rank(d: ActiveDep) -> tuple:
    """Least disruptive first: not busy, lower priority, then the longest idle (asleep counts as idle forever)."""
    idle = math.inf if d.state == "sleeping" else (d.idle_min if d.idle_min is not None else 0.0)
    return (d.disruptive, d.priority, -idle)


def _evictions(kind: str, cands: list[ActiveDep], shortfall: int, limit: int) -> list[list[ActiveDep]]:
    """Sets of deployments to stop/sleep: every single one that covers the shortfall, then minimal combinations
    (no member can be dropped) of up to MAX_COMBO_SIZE, ranked; if none covers it, one greedy set in rank order."""
    frees = _stop_frees if kind == "stop" else (lambda d: d.sleep_frees)
    ranked = sorted(cands, key=lambda d: (_rank(d), frees(d), d.id))
    singles = sorted((d for d in ranked if frees(d) >= shortfall), key=lambda d: (_rank(d), frees(d), d.id))
    sets: list[list[ActiveDep]] = [[d] for d in singles[:limit]]
    if len(sets) < limit:
        pool = [d for d in ranked if frees(d) < shortfall][:COMBO_CANDIDATES]
        combos = []
        for size in range(2, MAX_COMBO_SIZE + 1):
            for combo in itertools.combinations(pool, size):
                total = sum(frees(d) for d in combo)
                if total >= shortfall and all(total - frees(d) < shortfall for d in combo):
                    worst = max(_rank(d) for d in combo)
                    combos.append((worst, size, total, sorted(d.id for d in combo), list(combo)))
        combos.sort(key=lambda c: c[:4])
        sets += [c[4] for c in combos[:limit - len(sets)]]
    if not sets and ranked:  # nothing covers it within the bound: one greedy plan, as before (may not fit)
        acc, freed = [], 0
        for d in ranked:
            acc.append(d)
            freed += frees(d)
            if freed >= shortfall:
                break
        sets.append(acc)
    return sets


def eviction_plans(active: list[ActiveDep], shortfall: int, required: int) -> list[PlanOption]:
    """Sleep plans then stop plans, each group ranked (see _evictions); ids are stable: "stop:3", "sleep:2+5"."""
    out: list[PlanOption] = []
    sleepers = [d for d in active if d.can_sleep and d.state == "ready"]
    for group in _evictions("sleep", sleepers, shortfall, MAX_SLEEP_PLANS):
        freed = sum(d.sleep_frees for d in group)
        out.append(PlanOption("sleep:" + "+".join(str(i) for i in sorted(d.id for d in group)),
                              "Put " + ", ".join(d.name for d in group) + " to sleep",
                              f"Frees about {_gib(freed)} (weights stay in memory for a fast wake-up).",
                              required, freed, freed >= shortfall,
                              actions=[PlanAction(type="sleep", deployment_id=d.id) for d in group]))
    for group in _evictions("stop", active, shortfall, MAX_STOP_PLANS):
        freed = sum(_stop_frees(d) for d in group)
        out.append(PlanOption("stop:" + "+".join(str(i) for i in sorted(d.id for d in group)),
                              "Stop " + ", ".join(d.name for d in group),
                              f"Frees about {_gib(freed)}. Can be restarted later.",
                              required, freed, freed >= shortfall,
                              actions=[PlanAction(type="stop", deployment_id=d.id) for d in group]))
    return out
