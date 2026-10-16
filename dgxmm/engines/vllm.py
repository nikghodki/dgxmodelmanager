"""vLLM engine (OpenAI server in vllm/vllm-openai)."""
from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dgxmm.config import Mount, Settings
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import (SERVED_NAME_RE, ContainerSpec, EngineMetrics, LaunchContext, container_name,
                                default_served_name, managed_labels, openai_health)
from dgxmm.engines.prom import buckets, counter, gauge, parse_samples
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate, VllmSizing, default_kv_tokens, estimate_vllm, vllm_gpu_util

PARSER_RE = r"^[a-z0-9_]{1,32}$"


class SpeculativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["dflash", "eagle", "eagle3", "mtp", "ngram", "draft_model"]
    model: str | None = Field(None, pattern=REPO_RE)
    num_speculative_tokens: int = Field(3, ge=1, le=16)

    @model_validator(mode="after")
    def _needs_model(self):
        if self.method in ("dflash", "eagle", "eagle3", "draft_model") and not self.model:
            raise ValueError(f"speculative method {self.method} needs a draft model")
        return self


class VllmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    served_model_name: str | None = Field(None, pattern=SERVED_NAME_RE)
    max_model_len: int = Field(32768, ge=256, le=1_048_576)
    max_num_seqs: int = Field(8, ge=1, le=1024)
    max_num_batched_tokens: int = Field(8192, ge=256, le=262_144)
    kv_tokens: int | None = Field(None, ge=256, le=64_000_000)
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = Field(None, pattern=PARSER_RE)
    reasoning_parser: str | None = Field(None, pattern=PARSER_RE)
    speculative: SpeculativeConfig | None = None
    enable_sleep_mode: bool = True
    trust_remote_code: bool = False
    safetensors_load_strategy: Literal["lazy", "eager", "prefetch"] | None = None
    extra_args: list[str] = []

    @field_validator("extra_args")
    @classmethod
    def _extra(cls, v: list[str]) -> list[str]:
        return validate_extra_args(v, VLLM_EXTRA_FLAGS)

    @model_validator(mode="after")
    def _pool(self):
        if self.kv_tokens is not None and self.kv_tokens < self.max_model_len:
            raise ValueError("kv_tokens must be >= max_model_len")
        return self


VLLM_METRICS = {
    "gauges": {"running": ("vllm:num_requests_running",), "waiting": ("vllm:num_requests_waiting",),
               "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")},
    "counters": {"prompt_tokens": ("vllm:prompt_tokens",), "generation_tokens": ("vllm:generation_tokens",),
                 "prefix_queries": ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"),
                 "prefix_hits": ("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"),
                 "spec_draft": ("vllm:spec_decode_num_draft_tokens",),
                 "spec_accepted": ("vllm:spec_decode_num_accepted_tokens",),
                 "preemptions": ("vllm:num_preemptions",), "requests": ("vllm:request_success",)},
    "histograms": {"ttft": ("vllm:time_to_first_token_seconds",),
                   "itl": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
                   "e2e": ("vllm:e2e_request_latency_seconds",)},
}


def metrics_from(text: str, table: dict) -> EngineMetrics:
    s = parse_samples(text)
    m = EngineMetrics()
    for key, names in table["gauges"].items():
        v = gauge(s, *names)
        if v is not None:
            m.gauges[key] = v
    for key, names in table["counters"].items():
        v = counter(s, *names)
        if v is not None:
            m.counters[key] = v
    for key, names in table.get("histograms", {}).items():
        b = buckets(s, *names)
        if b:
            m.histograms[key] = b
    return m


class VllmEngine:
    name = "vllm"
    supports_sleep = True
    config_model = VllmConfig
    context_key = "max_model_len"
    seqs_key = "max_num_seqs"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.arch_source = None

    def validate_config(self, raw: dict[str, Any]) -> VllmConfig:
        return VllmConfig.model_validate(raw)

    def supports(self, artifact: ModelArtifact) -> bool:
        if not (artifact.source == "hf" and artifact.format == "safetensors" and artifact.kind == "llm"):
            return False
        archs = self.arch_source() if self.arch_source else None
        return archs is None or artifact.architecture in archs

    def draft_ref(self, cfg: VllmConfig) -> str | None:
        return cfg.speculative.model if cfg.speculative and cfg.speculative.model else None

    def served_name(self, cfg: VllmConfig, model_ref: str) -> str:
        return cfg.served_model_name or default_served_name(model_ref)

    def sleep_enabled(self, cfg: VllmConfig) -> bool:
        return cfg.enable_sleep_mode

    @staticmethod
    def pool_tokens(cfg: VllmConfig) -> int:
        """KV pool the reservation is sized for: kv_tokens, or the default shared pool."""
        return cfg.kv_tokens or default_kv_tokens(cfg.max_model_len, cfg.max_num_seqs)

    @classmethod
    def sizing(cls, cfg: VllmConfig) -> VllmSizing:
        return VllmSizing(cfg.max_model_len, cfg.max_num_seqs, cfg.max_num_batched_tokens, cfg.kv_cache_dtype,
                          cls.pool_tokens(cfg))

    def estimate(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None, settings: Settings) -> MemoryEstimate:
        if artifact.arch is None:
            raise ValueError("model config.json could not be parsed")
        return estimate_vllm(artifact.arch, artifact.weights_bytes, self.sizing(cfg), settings,
                             draft_spec=draft.arch if draft else None, draft_bytes=draft.weights_bytes if draft else 0)

    def container_spec(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None,
                       launch: LaunchContext, settings: Settings) -> ContainerSpec:
        util = vllm_gpu_util(launch.reserved, launch.mem_total)
        cmd = ["--model", artifact.ref]
        if artifact.revision:
            cmd += ["--revision", artifact.revision]
        cmd += ["--served-model-name", self.served_name(cfg, artifact.ref), "--host", "0.0.0.0", "--port", str(launch.port),
                "--max-model-len", str(cfg.max_model_len), "--max-num-seqs", str(cfg.max_num_seqs),
                "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
                "--gpu-memory-utilization", f"{util:.2f}", "--kv-cache-dtype", cfg.kv_cache_dtype,
                "--enable-prefix-caching" if cfg.enable_prefix_caching else "--no-enable-prefix-caching",
                "--enable-chunked-prefill" if cfg.enable_chunked_prefill else "--no-enable-chunked-prefill"]
        if cfg.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        if cfg.tool_call_parser:
            cmd += ["--tool-call-parser", cfg.tool_call_parser]
        if cfg.reasoning_parser:
            cmd += ["--reasoning-parser", cfg.reasoning_parser]
        if cfg.speculative:
            sc: dict[str, Any] = {"method": cfg.speculative.method,
                                  "num_speculative_tokens": cfg.speculative.num_speculative_tokens}
            if cfg.speculative.model:
                sc["model"] = cfg.speculative.model
            cmd += ["--speculative-config", json.dumps(sc)]
        if cfg.enable_sleep_mode:
            cmd.append("--enable-sleep-mode")
        if cfg.trust_remote_code:
            cmd.append("--trust-remote-code")
        if cfg.safetensors_load_strategy:
            cmd += ["--safetensors-load-strategy", cfg.safetensors_load_strategy]
        cmd += cfg.extra_args
        env = {"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        if cfg.enable_sleep_mode:
            env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /sleep and /wake_up
        mounts = [Mount(host=settings.hf_cache.host, target="/root/.cache/huggingface"), *settings.vllm_cache_mounts]
        return ContainerSpec(name=container_name(launch.name), image=settings.images.vllm, command=cmd, environment=env,
                             mounts=mounts, labels=managed_labels(launch.deployment_id, self.name),
                             network=settings.docker_network, port=launch.port, publish_bind=settings.publish_bind)

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str) -> bool:
        return await openai_health(http, base_url, served_name)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return metrics_from(text, VLLM_METRICS)

    async def sleep(self, http: httpx.AsyncClient, base_url: str) -> None:
        (await http.post(f"{base_url}/sleep", params={"level": 1}, timeout=120)).raise_for_status()

    async def wake(self, http: httpx.AsyncClient, base_url: str) -> None:
        (await http.post(f"{base_url}/wake_up", timeout=300)).raise_for_status()

    def kv_patch_options(self, cfg: VllmConfig) -> list[dict]:
        return [{"kv_cache_dtype": "fp8"}] if cfg.kv_cache_dtype == "auto" else []
