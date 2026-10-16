"""Ways to make a deployment fit (spec §5.4). Pure functions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from dgxmm.planner.fit import FitResult

GIB = 1024**3
_QUANT_SUFFIX = re.compile(
    r"([-_.](nvfp4|mxfp4|fp8([-_]dynamic)?|fp4|awq|gptq|int4|int8|w4a16|w8a8|bnb[-_]4bit|4bit|8bit|gguf|"
    r"q\d(_k)?(_[msl])?|iq\d\w*|bf16|fp16))+$", re.I)


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

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "detail": self.detail, "new_required": self.new_required,
                "frees": self.frees, "fits": self.fits, "actions": [a.model_dump() for a in self.actions],
                "patch": self.patch, "model_ref": self.model_ref}


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

    ordered = sorted(active, key=lambda d: (d.priority, -d.reserved))
    sleepers = [d for d in ordered if d.can_sleep and d.state == "ready"]
    if sleepers:
        acts, freed = [], 0
        for d in sleepers:
            acts.append(PlanAction(type="sleep", deployment_id=d.id))
            freed += d.sleep_frees
            if freed >= shortfall:
                break
        out.append(PlanOption("sleep", "Put " + ", ".join(d.name for d in sleepers[:len(acts)]) + " to sleep",
                              f"Frees about {_gib(freed)} (weights stay in memory for a fast wake-up).",
                              fit.required, freed, freed >= shortfall, actions=acts))
    if ordered:
        acts, freed = [], 0
        for d in ordered:
            acts.append(PlanAction(type="stop", deployment_id=d.id))
            freed += d.actual if d.state == "sleeping" else max(d.reserved, d.actual)
            if freed >= shortfall:
                break
        out.append(PlanOption("stop", "Stop " + ", ".join(d.name for d in ordered[:len(acts)]),
                              f"Frees about {_gib(freed)}. Lowest priority first; can be restarted later.",
                              fit.required, freed, freed >= shortfall, actions=acts))
    out.sort(key=lambda p: not p.fits)  # stable: fitting options first, original order otherwise
    return out
