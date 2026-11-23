"""llama.cpp llama-server engine (ghcr.io/ggml-org/llama.cpp:server-cuda; cluster deployments: images.llamacpp_rpc,
built from deploy/llamacpp-rpc with the RPC backend)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dgxmm.config import IMAGE_REF, LOCAL_NODE_ID, Mount, Settings
from dgxmm.engines.argguard import LLAMACPP_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import (RANK_LABEL, REMOTE_NETWORK, SERVED_NAME_RE, ContainerSpec, EngineMetrics,
                                LaunchContext, RankSpec, SpecError, container_name, default_served_name,
                                engine_image, managed_labels, openai_health, remote_hf_mount)
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
# Where a cluster head sees the HF cache: not under /root, since the ranks run as DGXMM's uid (rank_user()).
RPC_HF_TARGET = "/models/hf"
MIB = 1 << 20


def rank_user() -> str | None:
    """The uid:gid every llama.cpp cluster rank runs as: DGXMM's own (compose `user:`, DGXMM_UID:DGXMM_GID), which
    owns the HF caches (mode 700 is common) on every node. None when DGXMM itself runs as root: the image's USER (an
    unprivileged uid) then, never root."""
    uid, gid = os.getuid(), os.getgid()
    return None if uid == 0 else f"{uid}:{gid}"


class LlamaCppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # the engine image; None = images.llamacpp. Must be on images.llamacpp_allowed (engines.base.engine_image).
    # Single-node only: cluster deployments run images.llamacpp_rpc
    image: str | None = Field(None, max_length=400, pattern=IMAGE_REF)
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


def tensor_split(shares: list[int]) -> str:
    """--tensor-split proportions from byte shares, in MiB (llama.cpp normalises them): whole percentages would put
    up to 1 % of the model (about 1 GB for a large one) on a rank beyond its reservation. A non-zero share below
    1 MiB counts as 1; all zero: equal parts."""
    if sum(shares) <= 0:
        return ",".join("1" for _ in shares)
    return ",".join(str(max(s // MIB, 1) if s > 0 else 0) for s in shares)


def remote_container_path(local_path: str, settings: Settings, node_id: str) -> tuple[Mount, str, str]:
    """On a remote node only GGUF files inside the HF cache can be served (from the node's own HF cache, read-only):
    (mount, path in the container, publish address)."""
    p = Path(local_path)
    hf_local = Path(settings.hf_cache.local)
    if hf_local not in p.parents:
        raise SpecError("not_on_node", f"{local_path} is outside the HF cache; only models in the HF cache can run "
                                       f"on node {node_id}")
    mount, publish_bind = remote_hf_mount(settings, node_id)
    return mount, f"{mount.target}/{p.relative_to(hf_local)}", publish_bind


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

    def supports(self, artifact: ModelArtifact, image: str | None = None) -> bool:
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
        if launch.node == LOCAL_NODE_ID:
            mount, path = container_path(artifact.local_path, settings)
            network, publish_bind = settings.docker_network, settings.publish_bind
        else:  # remote node: its own HF cache only, read-only; its default bridge, published on its cluster IP
            mount, path, publish_bind = remote_container_path(artifact.local_path, settings, launch.node)
            network = REMOTE_NETWORK
        cmd = ["-m", path, "--host", "0.0.0.0", "--port", str(launch.port), "-c", str(cfg.ctx_size),
               "-np", str(cfg.parallel), "-ngl", str(cfg.n_gpu_layers), "-ctk", cfg.cache_type_k,
               "-ctv", cfg.cache_type_v, "-fa", cfg.flash_attn, "-b", str(cfg.n_batch), "-ub", str(cfg.n_ubatch),
               "--metrics", "--slots", "-a", self.served_name(cfg, artifact.ref), *cfg.extra_args]
        # the key in the environment, never argv (--api-key would show in ps and docker inspect's Cmd)
        env = {"LLAMA_API_KEY": launch.api_key} if launch.api_key else {}
        return ContainerSpec(name=container_name(launch.name), image=engine_image(settings, self.name, cfg.image),
                             command=cmd,
                             environment=env, mounts=[mount], labels=managed_labels(launch.deployment_id, self.name),
                             network=network, port=launch.port, publish_bind=publish_bind, ipc_host=False)

    def cluster_specs(self, artifact: ModelArtifact, cfg: LlamaCppConfig, draft, launch: LaunchContext, placement,
                      nodes, rpc_port: int, settings: Settings) -> list[RankSpec]:
        """One container per node of ``placement`` (rank = position, the head first) for a llama.cpp deployment over
        the ggml RPC backend. Workers run ``ggml-rpc-server`` bound to their node's cluster IP (RPC has no
        authentication: never 0.0.0.0) on ``rpc_port``, without a local tensor cache (no -c: it grew without bound,
        outside the disk accounting; each load sends the worker's share over the link again) and without any
        mount. The head runs ``llama-server`` on the model (the HF cache read-only at RPC_HF_TARGET) with the
        single-node flags, serves the API on its
        cluster IP (engine key in LLAMA_API_KEY) and splits the layers over ``--rpc`` (the workers, in rank order)
        and its own GPU with ``--tensor-split``: llama.cpp puts the RPC devices first, so the proportions list the
        workers, then the head. The proportions are each rank's share of the shardable part of its reservation
        (``launch.rank_memory``, memory.estimate.per_rank_bytes: the reservation less the per-rank overheads). Every
        rank runs on the host network with the RDMA devices (the RPC transport auto-negotiates RDMA), as DGXMM's
        uid (rank_user(): never root)."""
        if draft is not None:
            raise SpecError("invalid_container_spec", "llama.cpp has no draft models")
        if cfg.image:  # a chosen image is single-node only (docs/security.md, "Engine images")
            raise SpecError("image_single_node_only", "a chosen engine image is supported on single-node deployments "
                                                      "only; cluster deployments run images.llamacpp_rpc")
        cfg_nodes = {n: nodes.config(n) for n in placement.nodes}
        for node_id, node in cfg_nodes.items():
            if not node.cluster_ip:
                raise SpecError("cluster_node_not_ready", f"node {node_id} has no cluster_ip configured")
        head_id, workers = placement.nodes[0], list(placement.nodes[1:])
        head = cfg_nodes[head_id]
        labels = managed_labels(launch.deployment_id, self.name)
        rdma = dict(network="host", network_mode="host", publish_bind=None, devices=["/dev/infiniband"],
                    cap_add=["IPC_LOCK"], ulimits={"memlock": (-1, -1)}, ipc_host=False, user=rank_user())
        out: list[RankSpec] = []
        for rank, node_id in enumerate(placement.nodes):
            if rank == 0:
                continue
            node = cfg_nodes[node_id]
            spec = ContainerSpec(name=f"{container_name(launch.name)}-r{rank}", image=settings.images.llamacpp_rpc,
                                 command=["ggml-rpc-server", "--host", node.cluster_ip, "-p", str(rpc_port)],
                                 environment={}, mounts=[],
                                 labels={**labels, RANK_LABEL: str(rank)}, port=None, **rdma)
            out.append(RankSpec(node_id, rank, spec))
        if head_id == LOCAL_NODE_ID:
            mount, path = container_path(artifact.local_path, settings)
        else:
            mount, path, _ = remote_container_path(artifact.local_path, settings, head_id)
        if path.startswith(mount.target + "/"):  # the HF cache: outside /root (not traversable by a non-root uid)
            new_target = RPC_HF_TARGET if mount.target.startswith("/root/") else mount.target
            path = new_target + path[len(mount.target):]
            mount = Mount(host=mount.host, target=new_target, mode=mount.mode)
        split = tensor_split(self._shares(artifact, cfg, launch, [*workers, head_id], settings))
        # -lm none: no mmap; on the unified-memory GB10 the mapped file would stay in the page cache next to the
        # copies on the GPUs. The cluster flags come last (llama.cpp keeps the last value), after extra_args.
        cmd = ["llama-server", "-m", path, "-c", str(cfg.ctx_size), "-np", str(cfg.parallel),
               "-ngl", str(cfg.n_gpu_layers), "-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v,
               "-fa", cfg.flash_attn, "-b", str(cfg.n_batch), "-ub", str(cfg.n_ubatch), "--metrics", "--slots",
               "-a", self.served_name(cfg, artifact.ref), "-lm", "none", *cfg.extra_args,
               "--host", head.cluster_ip, "--port", str(launch.port),
               "--rpc", ",".join(f"{cfg_nodes[w].cluster_ip}:{rpc_port}" for w in workers), "--tensor-split", split]
        env = {"LLAMA_API_KEY": launch.api_key} if launch.api_key else {}
        spec = ContainerSpec(name=f"{container_name(launch.name)}-r0", image=settings.images.llamacpp_rpc,
                             command=cmd, environment=env, mounts=[mount], labels={**labels, RANK_LABEL: "0"},
                             port=launch.port, **rdma)
        return [RankSpec(head_id, 0, spec), *out]

    def _shares(self, artifact: ModelArtifact, cfg: LlamaCppConfig, launch: LaunchContext, order: list[str],
                settings: Settings) -> list[int]:
        """Each node's (in ``order``) share of the layers: its reservation less the overheads every rank needs in
        full (activations / compute buffers, runtime; memory.estimate.per_rank_bytes)."""
        reserved = [int((launch.rank_memory or {}).get(n, (launch.reserved, 0))[0]) for n in order]
        try:
            est = self.estimate(artifact, cfg, None, settings)
            fixed = est.activations + est.runtime
        except ValueError:
            fixed = 0
        shares = [max(r - fixed, 0) for r in reserved]
        return shares if sum(shares) > 0 else reserved

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str,
                     api_key: str | None = None, completion: bool = True) -> bool:
        return await openai_health(http, base_url, served_name, api_key=api_key, completion=completion)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return metrics_from(text, LLAMA_METRICS)

    async def sleep(self, http, base_url, api_key: str | None = None) -> None:
        raise NotImplementedError("llama.cpp has no sleep mode")

    async def wake(self, http, base_url, api_key: str | None = None) -> None:
        raise NotImplementedError("llama.cpp has no sleep mode")

    def kv_patch_options(self, cfg: LlamaCppConfig) -> list[dict]:
        if cfg.cache_type_k == "f16":
            return [{"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on" if cfg.flash_attn == "off" else cfg.flash_attn}]
        return []
