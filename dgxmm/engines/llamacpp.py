"""llama.cpp llama-server engine (ghcr.io/ggml-org/llama.cpp:server-cuda)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dgxmm.config import Mount, Settings
from dgxmm.engines.argguard import LLAMACPP_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import (SERVED_NAME_RE, ContainerSpec, EngineMetrics, LaunchContext, container_name,
                                default_served_name, managed_labels, openai_health)
from dgxmm.engines.vllm import metrics_from
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.memory.estimate import LlamaCppSizing, MemoryEstimate, estimate_llamacpp

CacheType = Literal["f16", "bf16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"]
LLAMA_METRICS = {
    "gauges": {"running": ("llamacpp:requests_processing",), "waiting": ("llamacpp:requests_deferred",),
               "kv_usage": ("llamacpp:kv_cache_usage_ratio",)},
    "counters": {"prompt_tokens": ("llamacpp:prompt_tokens",), "generation_tokens": ("llamacpp:tokens_predicted",)},
    "histograms": {},
}


class LlamaCppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str | None = Field(None, pattern=SERVED_NAME_RE)
    ctx_size: int = Field(32768, ge=512, le=1_048_576)
    parallel: int = Field(4, ge=1, le=64)
    n_gpu_layers: int = Field(999, ge=0, le=999)
    cache_type_k: CacheType = "f16"
    cache_type_v: CacheType = "f16"
    flash_attn: Literal["auto", "on", "off"] = "auto"
    n_batch: int = Field(2048, ge=32, le=16384)
    n_ubatch: int = Field(512, ge=32, le=8192)
    extra_args: list[str] = []

    @field_validator("extra_args")
    @classmethod
    def _extra(cls, v: list[str]) -> list[str]:
        return validate_extra_args(v, LLAMACPP_EXTRA_FLAGS)

    @model_validator(mode="after")
    def _check(self):
        if self.n_ubatch > self.n_batch:
            raise ValueError("n_ubatch must be <= n_batch")
        if self.cache_type_v not in ("f16", "bf16", "f32") and self.flash_attn == "off":
            raise ValueError("a quantized V cache needs flash attention (flash_attn auto or on)")
        return self


def container_path(local_path: str, settings: Settings) -> tuple[Mount, str]:
    p = Path(local_path)
    hf_local = Path(settings.hf_cache.local)
    if hf_local in p.parents:
        return (Mount(host=settings.hf_cache.host, target="/root/.cache/huggingface", mode="ro"),
                "/root/.cache/huggingface/" + str(p.relative_to(hf_local)))
    for i, m in enumerate(settings.gguf_roots):
        root = Path(m.local)
        if root in p.parents:
            return Mount(host=m.host, target=f"/models/{i}", mode="ro"), f"/models/{i}/{p.relative_to(root)}"
    raise ValueError(f"{local_path} is not under an allowed GGUF root")


class LlamaCppEngine:
    name = "llamacpp"
    supports_sleep = False
    config_model = LlamaCppConfig
    context_key = "ctx_size"
    seqs_key = "parallel"

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_config(self, raw: dict[str, Any]) -> LlamaCppConfig:
        return LlamaCppConfig.model_validate(raw)

    def supports(self, artifact: ModelArtifact) -> bool:
        return artifact.source == "gguf" and artifact.kind == "llm"

    def draft_ref(self, cfg: LlamaCppConfig) -> str | None:
        return None

    def served_name(self, cfg: LlamaCppConfig, model_ref: str) -> str:
        return cfg.alias or default_served_name(model_ref)

    def sleep_enabled(self, cfg: LlamaCppConfig) -> bool:
        return False

    def estimate(self, artifact: ModelArtifact, cfg: LlamaCppConfig, draft, settings: Settings) -> MemoryEstimate:
        if artifact.arch is None:
            raise ValueError("GGUF header could not be parsed")
        return estimate_llamacpp(artifact.arch, artifact.weights_bytes,
                                 LlamaCppSizing(cfg.ctx_size, cfg.parallel, cfg.cache_type_k, cfg.cache_type_v, cfg.n_ubatch),
                                 settings)

    def container_spec(self, artifact: ModelArtifact, cfg: LlamaCppConfig, draft, launch: LaunchContext,
                       settings: Settings) -> ContainerSpec:
        mount, path = container_path(artifact.local_path, settings)
        cmd = ["-m", path, "--host", "0.0.0.0", "--port", str(launch.port), "-c", str(cfg.ctx_size),
               "-np", str(cfg.parallel), "-ngl", str(cfg.n_gpu_layers), "-ctk", cfg.cache_type_k,
               "-ctv", cfg.cache_type_v, "-fa", cfg.flash_attn, "-b", str(cfg.n_batch), "-ub", str(cfg.n_ubatch),
               "--metrics", "--slots", "-a", self.served_name(cfg, artifact.ref), *cfg.extra_args]
        return ContainerSpec(name=container_name(launch.name), image=settings.images.llamacpp, command=cmd,
                             environment={}, mounts=[mount], labels=managed_labels(launch.deployment_id, self.name),
                             network=settings.docker_network, port=launch.port, publish_bind=settings.publish_bind,
                             ipc_host=False)

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str) -> bool:
        return await openai_health(http, base_url, served_name)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return metrics_from(text, LLAMA_METRICS)

    async def sleep(self, http, base_url) -> None:
        raise NotImplementedError("llama.cpp has no sleep mode")

    async def wake(self, http, base_url) -> None:
        raise NotImplementedError("llama.cpp has no sleep mode")

    def kv_patch_options(self, cfg: LlamaCppConfig) -> list[dict]:
        if cfg.cache_type_k == "f16":
            return [{"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on" if cfg.flash_attn == "off" else cfg.flash_attn}]
        return []
