from __future__ import annotations

import asyncio
import logging

from dgxmm.catalog.hub import fit_badge, guess_quant, quick_required
from dgxmm.catalog.llmfit import LlmfitError
from dgxmm.catalog.support import engine_support
from dgxmm.config import GIB
from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.memory.estimate import VllmSizing, default_kv_tokens, estimate_vllm
from dgxmm.planner.fit import check_fit
from dgxmm.planner.plans import base_model_name

log = logging.getLogger(__name__)


def _llmfit_summary(entry: dict | None, required: int | None) -> dict | None:
    if not entry:
        return None
    mem = entry.get("memory_required_gb") or entry.get("recommended_ram_gb") or entry.get("min_vram_gb")
    mismatch = bool(mem and required and abs(mem * 1e9 - required) / required > 0.20)
    return {"fit_level": entry.get("fit_level"), "score": entry.get("score"), "estimated_tps": entry.get("estimated_tps"),
            "measured_tps": entry.get("measured_tps"), "memory_required_gb": mem, "runtime": entry.get("runtime"),
            "best_quant": entry.get("best_quant"), "use_case": entry.get("use_case"), "mismatch": mismatch}


class CatalogService:
    def __init__(self, settings, hub, llmfit, inventory, budget, deployer, archs):
        self.settings, self.hub, self.llmfit, self.inventory = settings, hub, llmfit, inventory
        self.budget, self.deployer, self.archs = budget, deployer, archs

    async def _budget(self):
        return self.budget.current() or await self.budget.refresh()

    def _evictable(self) -> int:
        return sum(max(d.reserved, d.actual) for d in self.deployer.active_deps())

    def _llm_map(self, q: str, free: int, ctx: int) -> tuple[dict, str | None]:
        try:
            return self.llmfit.search_map(q, max(free, 0) / GIB, ctx), None
        except LlmfitError as e:
            return {}, f"llmfit unavailable: {e}"

    async def search(self, q: str, gguf: bool = False, ctx: int = 32768, limit: int = 30) -> dict:
        budget = await self._budget()
        models = await asyncio.to_thread(self.hub.search, q, limit, gguf)
        llm, banner = await asyncio.to_thread(self._llm_map, q, budget.free_for_new, ctx)
        archs, evictable = self.archs.get(), self._evictable()
        results = []
        for m in models:
            required = quick_required(m.est_bytes, self.settings) if m.est_bytes else None
            local = self.inventory.get(m.repo_id)
            arch = (local.architecture if local else None)
            results.append({**m.as_dict(), "downloaded": local is not None and local.complete,
                            "fit": fit_badge(required, budget, evictable) if required else {"badge": "unknown"},
                            "llmfit": _llmfit_summary(llm.get(m.repo_id.lower()), required),
                            "support": {"vllm": None if gguf else (arch in archs if (archs is not None and arch) else None),
                                        "llamacpp": m.has_gguf},
                            "quant": guess_quant(m.repo_id)})
        return {"results": results, "banner": banner, "free_for_new": budget.free_for_new}

    async def details(self, repo_id: str, ctx: int = 32768) -> dict:
        budget = await self._budget()
        d = await asyncio.to_thread(self.hub.details, repo_id)
        spec = None
        if d.config:
            try:
                spec = parse_hf_config(d.config)
            except Exception as e:
                log.warning("cannot parse config of %s: %s", repo_id, e)
        st = {f.name: f.size for f in d.files if f.name.endswith(".safetensors")}
        if any(k.startswith("model") for k in st):
            st = {k: v for k, v in st.items() if not k.startswith("consolidated")}
        weights = sum(st.values())
        out: dict = {"repo_id": repo_id, "gated": d.gated, "sha": d.sha,
                     "files": [{"name": f.name, "size": f.size} for f in d.files],
                     "architecture": spec.architecture if spec else None, "max_context": spec.max_context if spec else None,
                     "quant": spec.quant if spec else guess_quant(repo_id)}
        evictable = self._evictable()
        if spec and weights:
            n_ctx = min(ctx, spec.max_context)
            est = estimate_vllm(spec, weights, VllmSizing(n_ctx, 8, 8192, "auto", default_kv_tokens(n_ctx, 8)),
                                self.settings)
            out["estimate"] = est.as_dict()
            out["fit"] = {**fit_badge(est.total, budget, evictable), **check_fit(est, budget).as_dict()}
        out["gguf_files"] = [{"name": f.name, "size": f.size, "quant": guess_quant(f.name),
                              "fit": fit_badge(int(f.size * 1.1) + GIB, budget, evictable)}
                             for f in d.files if f.name.lower().endswith(".gguf") and "mmproj" not in f.name.lower()]
        archs = self.archs.get()
        out["support"] = {"vllm": engine_support(out["architecture"], False, archs)["vllm"] if weights else False,
                          "llamacpp": bool(out["gguf_files"])}
        llm, banner = await asyncio.to_thread(self._llm_map, repo_id, budget.free_for_new, ctx)
        out["llmfit"] = _llmfit_summary(llm.get(repo_id.lower()), out.get("estimate", {}).get("total"))
        out["banner"] = banner
        base = base_model_name(repo_id)
        try:
            variants = await asyncio.to_thread(self.hub.search, base, 20, False)
            out["variants"] = [{"repo_id": v.repo_id, "quant": guess_quant(v.repo_id), "est_bytes": v.est_bytes}
                               for v in variants if v.repo_id != repo_id and base_model_name(v.repo_id) == base]
        except Exception:
            out["variants"] = []
        out["downloaded"] = bool(self.inventory.get(repo_id))
        return out
