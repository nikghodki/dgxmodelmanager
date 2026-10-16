"""Which downloaded models can run together (spec §4.6)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    ref: str
    engine: str
    config: dict = field(hash=False, compare=False)
    required: int = 0
    favourite: bool = False


@dataclass
class Combo:
    models: list[Candidate]
    total: int
    headroom: int

    def as_dict(self) -> dict:
        return {"models": [{"ref": m.ref, "engine": m.engine, "config": m.config, "required": m.required,
                            "favourite": m.favourite} for m in self.models],
                "total": self.total, "headroom": self.headroom}


def default_config(engine_name: str, artifact) -> dict:
    max_ctx = getattr(getattr(artifact, "arch", None), "max_context", None) or 32768
    if engine_name == "llamacpp":
        return {"ctx_size": min(32768, max_ctx), "parallel": 4}
    return {"max_model_len": min(32768, max_ctx), "max_num_seqs": 8, "max_num_batched_tokens": 8192}


def best_combos(cands: list[Candidate], capacity: int, max_models: int = 4, top: int = 10,
                must_include: tuple[str, ...] = ()) -> list[Combo]:
    pool = sorted((c for c in cands if 0 < c.required <= capacity), key=lambda c: c.required)
    combos: list[Combo] = []
    for k in range(min(max_models, len(pool)), 0, -1):
        for combo in itertools.combinations(pool, k):
            total = sum(c.required for c in combo)
            if total > capacity:
                continue
            refs = {c.ref for c in combo}
            if must_include and not set(must_include) <= refs:
                continue
            combos.append(Combo(list(combo), total, capacity - total))
    combos.sort(key=lambda x: (-len(x.models), -sum(m.favourite for m in x.models), -x.headroom))
    return combos[:top]


def planner_view(deployer, inventory, budget, favourites: set[str], ctx: int | None = None,
                 must_include: tuple[str, ...] = ()) -> dict:
    from dgxmm.deployer.service import DeployError
    cands: list[Candidate] = []
    for a in inventory.list():
        if not a.complete or a.kind != "llm":
            continue
        engine = next((n for n, e in deployer.engines.items() if e.supports(a)), None)
        if engine is None:
            continue
        cfg = default_config(engine, a)
        if ctx:
            key = deployer.engines[engine].context_key
            cfg[key] = min(ctx, cfg[key])
        try:
            prep = deployer.prepare(engine, a.ref, cfg)
        except DeployError:
            continue
        cands.append(Candidate(a.ref, engine, prep.cfg.model_dump(mode="json"), prep.estimate.total, a.ref in favourites))
    capacity = budget.usable - budget.external
    return {"capacity": capacity, "external": budget.external, "committed": budget.committed,
            "candidates": [{"ref": c.ref, "engine": c.engine, "required": c.required, "favourite": c.favourite,
                            "config": c.config} for c in sorted(cands, key=lambda c: c.required)],
            "combos": [x.as_dict() for x in best_combos(cands, capacity, must_include=must_include)]}
