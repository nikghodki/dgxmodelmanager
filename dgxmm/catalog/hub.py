"""Thin, testable wrapper over huggingface_hub (spec §7)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from dgxmm.config import GIB, Settings

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
               "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "F4": 0.5, "BOOL": 1}
_QUANT = re.compile(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|Q\d|NVFP4|MXFP4|FP8|AWQ|GPTQ|BF16|F16|INT4|INT8|W4A16)", re.I)


@dataclass
class HubModel:
    repo_id: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    library: str | None
    tags: list[str]
    gated: Any
    est_bytes: int | None
    params: int | None
    has_gguf: bool
    last_modified: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HubFile:
    name: str
    size: int


@dataclass
class HubDetails:
    repo_id: str
    files: list[HubFile]
    config: dict | None
    gated: Any
    sha: str | None = None
    extra: dict = field(default_factory=dict)


def guess_quant(name: str) -> str | None:
    m = _QUANT.findall(name)
    return m[-1].upper() if m else None


def quick_required(est_bytes: int, settings: Settings) -> int:
    """Rough requirement for list views: weights + 5% + vLLM runtime + 2 GiB default KV allowance."""
    runtime = int((settings.vllm_runtime_fixed_gib + settings.vllm_cuda_graph_gib) * GIB)
    return int(est_bytes * 1.05) + runtime + 2 * GIB


def fit_badge(required: int, budget, evictable: int) -> dict:
    if required <= budget.free_for_new:
        badge = "fits_now"
    elif required <= budget.free_for_new + evictable:
        badge = "fits_after_evict"
    else:
        badge = "too_big"
    return {"badge": badge, "required": required, "free": budget.free_for_new}


class HfHubClient:
    def __init__(self, token: str | None = None, api=None, http: httpx.Client | None = None):
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api, self.token = api, token
        self.http = http or httpx.Client(timeout=20, follow_redirects=True)

    @staticmethod
    def _bytes(st) -> tuple[int | None, int | None]:
        if not st or not getattr(st, "parameters", None):
            return None, None
        params = sum(st.parameters.values())
        return int(sum(n * DTYPE_BYTES.get(d, 2) for d, n in st.parameters.items())), params

    def search(self, query: str | None, limit: int = 30, gguf: bool = False) -> list[HubModel]:
        kw: dict[str, Any] = dict(search=query or None, sort="downloads", limit=limit,
                                  expand=["downloads", "likes", "pipeline_tag", "library_name", "tags", "gated",
                                          "safetensors", "lastModified", "gguf"])
        if gguf:
            kw["library"] = "gguf"
        else:
            kw["pipeline_tag"] = "text-generation"
        out = []
        for m in self.api.list_models(**kw):
            est, params = self._bytes(getattr(m, "safetensors", None))
            tags = list(getattr(m, "tags", None) or [])
            lm = getattr(m, "last_modified", None)
            out.append(HubModel(m.id, int(getattr(m, "downloads", 0) or 0), int(getattr(m, "likes", 0) or 0),
                                getattr(m, "pipeline_tag", None), getattr(m, "library_name", None), tags,
                                getattr(m, "gated", False), est, params,
                                bool(getattr(m, "gguf", None)) or "gguf" in tags, lm.isoformat() if hasattr(lm, "isoformat") else lm))
        return out

    def details(self, repo_id: str) -> HubDetails:
        info = self.api.model_info(repo_id, files_metadata=True)
        files = [HubFile(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
        config = None
        if any(f.name == "config.json" for f in files):
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            r = self.http.get(f"https://huggingface.co/{repo_id}/resolve/{info.sha or 'main'}/config.json", headers=headers)
            if r.status_code == 200:
                try:
                    config = r.json()
                except ValueError:
                    config = None
        return HubDetails(repo_id, files, config, getattr(info, "gated", False), getattr(info, "sha", None))
