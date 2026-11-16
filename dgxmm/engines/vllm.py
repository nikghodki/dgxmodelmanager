"""vLLM engine (OpenAI server in vllm/vllm-openai)."""
from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dgxmm.config import LOCAL_NODE_ID, Mount, Settings
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import (RANK_LABEL, REMOTE_NETWORK, SERVED_NAME_RE, ContainerSpec, EngineMetrics,
                                LaunchContext, RankSpec, SpecError, auth_headers, container_name, remote_hf_mount,
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

    def _serve_args(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None, util: float,
                    host: str | None, port: int | None) -> list[str]:
        """The `vllm serve` arguments; ``host`` None: no --host/--port (a headless cluster worker serves no API)."""
        cmd = ["--model", artifact.ref]
        if artifact.revision:
            cmd += ["--revision", artifact.revision]
        cmd += ["--served-model-name", self.served_name(cfg, artifact.ref)]
        if host is not None:
            cmd += ["--host", host, "--port", str(port)]
        cmd += ["--max-model-len", str(cfg.max_model_len), "--max-num-seqs", str(cfg.max_num_seqs),
                "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
                "--gpu-memory-utilization", f"{util:.4f}", "--kv-cache-dtype", cfg.kv_cache_dtype,
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
                # pinned like --revision: HF_HUB_OFFLINE resolves a bare repo id only through refs/main, which a
                # pinned download never writes
                if draft is not None and draft.revision:
                    sc["revision"] = draft.revision
            cmd += ["--speculative-config", json.dumps(sc)]
        if cfg.enable_sleep_mode:
            cmd.append("--enable-sleep-mode")
        if cfg.trust_remote_code:
            cmd.append("--trust-remote-code")
        if cfg.safetensors_load_strategy:
            cmd += ["--safetensors-load-strategy", cfg.safetensors_load_strategy]
        cmd += cfg.extra_args
        return cmd

    @staticmethod
    def _env(cfg: VllmConfig, api_key: str | None) -> dict[str, str]:
        env = {"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        if cfg.enable_sleep_mode:
            env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /sleep and /wake_up
        if api_key:  # the environment, never argv (--api-key would show in ps and docker inspect's Cmd)
            env["VLLM_API_KEY"] = api_key
        return env

    def container_spec(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None,
                       launch: LaunchContext, settings: Settings) -> ContainerSpec:
        util = vllm_gpu_util(launch.reserved, launch.mem_total)
        if launch.node != LOCAL_NODE_ID and not settings.remote_dev_endpoints:
            cfg = cfg.model_copy(update={"enable_sleep_mode": False})  # no dev routes on a remote node (I7)
        if settings.vllm_sleep != "level1":  # "sleep" stops the deployment: no sleep mode, no dev routes
            cfg = cfg.model_copy(update={"enable_sleep_mode": False})
        cmd = self._serve_args(artifact, cfg, draft, util, "0.0.0.0", launch.port)
        env = self._env(cfg, launch.api_key)
        if launch.node == LOCAL_NODE_ID:
            mounts = [Mount(host=settings.hf_cache.host, target="/root/.cache/huggingface"), *settings.vllm_cache_mounts]
            network, publish_bind = settings.docker_network, settings.publish_bind
        else:  # remote node: its own HF cache only, read-only; its default bridge, published on its cluster IP
            hf, publish_bind = remote_hf_mount(settings, launch.node)
            mounts, network = [hf], REMOTE_NETWORK
        return ContainerSpec(name=container_name(launch.name), image=settings.images.vllm, command=cmd, environment=env,
                             mounts=mounts, labels=managed_labels(launch.deployment_id, self.name),
                             network=network, port=launch.port, publish_bind=publish_bind)

    def cluster_specs(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None,
                      launch: LaunchContext, placement, nodes, master_port: int, settings: Settings) -> list[RankSpec]:
        """One container per node of ``placement`` (rank = position, the head first), for multi-node tensor
        parallelism over NCCL (mp backend): tp = nnodes = number of nodes, one GPU per node. Every rank runs on the
        host network with the RDMA devices; the head (rank 0) serves the API on its cluster IP, the others are
        headless. ``launch``: the head's port and the engine key (head only); ``launch.rank_memory`` each rank's
        reservation and its node's memory total (its --gpu-memory-utilization)."""
        head = nodes.config(placement.head)
        n = len(placement.nodes)
        # no sleep mode on clusters (yet): no --enable-sleep-mode and no VLLM_SERVER_DEV_MODE dev endpoints
        cfg = cfg.model_copy(update={"enable_sleep_mode": False})
        out: list[RankSpec] = []
        for rank, node_id in enumerate(placement.nodes):
            node = nodes.config(node_id)
            if not node.cluster_ip or node.nccl is None or not head.cluster_ip:
                raise SpecError("cluster_node_not_ready", f"node {node_id} has no cluster_ip or nccl configured")
            reserved, total = (launch.rank_memory or {}).get(node_id, (launch.reserved, launch.mem_total))
            is_head = rank == 0
            cmd = self._serve_args(artifact, cfg, draft, vllm_gpu_util(reserved, total),
                                   head.cluster_ip if is_head else None, launch.port if is_head else None)
            # last on the command line: argparse keeps the last value, so nothing in extra_args can override them
            cmd += ["--tensor-parallel-size", str(n), "--nnodes", str(n), "--node-rank", str(rank),
                    "--master-addr", head.cluster_ip, "--master-port", str(master_port)]
            if not is_head:
                cmd.append("--headless")
            env = self._env(cfg, launch.api_key if is_head else None)
            env.update({"NCCL_IB_HCA": node.nccl.ib_hca, "NCCL_SOCKET_IFNAME": node.nccl.socket_ifname,
                        "GLOO_SOCKET_IFNAME": node.nccl.socket_ifname, "VLLM_HOST_IP": node.cluster_ip})
            if node_id == LOCAL_NODE_ID:
                mounts = [Mount(host=settings.hf_cache.host, target="/root/.cache/huggingface"),
                          *settings.vllm_cache_mounts]
            else:
                mounts = [remote_hf_mount(settings, node_id)[0]]
            spec = ContainerSpec(name=f"{container_name(launch.name)}-r{rank}", image=settings.images.vllm,
                                 command=cmd, environment=env, mounts=mounts,
                                 labels={**managed_labels(launch.deployment_id, self.name), RANK_LABEL: str(rank)},
                                 network="host", port=launch.port if is_head else None, publish_bind=None,
                                 network_mode="host",
                                 devices=["/dev/infiniband"], cap_add=["IPC_LOCK"], ulimits={"memlock": (-1, -1)},
                                 shm_size="16g")
            out.append(RankSpec(node_id, rank, spec))
        return out

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str,
                     api_key: str | None = None, completion: bool = True) -> bool:
        return await openai_health(http, base_url, served_name, api_key=api_key, completion=completion)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return metrics_from(text, VLLM_METRICS)

    async def sleep(self, http: httpx.AsyncClient, base_url: str, api_key: str | None = None) -> None:
        (await http.post(f"{base_url}/sleep", params={"level": 1}, headers=auth_headers(api_key),
                         timeout=120)).raise_for_status()

    async def wake(self, http: httpx.AsyncClient, base_url: str, api_key: str | None = None) -> None:
        (await http.post(f"{base_url}/wake_up", headers=auth_headers(api_key), timeout=300)).raise_for_status()

    def kv_patch_options(self, cfg: VllmConfig) -> list[dict]:
        return [{"kv_cache_dtype": "fp8"}] if cfg.kv_cache_dtype == "auto" else []
