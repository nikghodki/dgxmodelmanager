"""The decision model engine: a d3 package (vllm-sr/d3-mini by default, or vllm-sr/d3-nano) served by its own
d3_server.py (System One API).

A *system* engine: one deployment at most, never registered in LiteLLM, never a plan candidate (the watchdog stops it
first under memory pressure, and whenever its own memory runs away: dgxmm/watchdog.py), no Prometheus metrics.
The image (``dgxmm-decider:<version>``) is built locally from deploy/decider.
"""
from __future__ import annotations

import logging
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from dgxmm.config import GIB, Mount, Settings, _expand_host
from dgxmm.decider.client import capped_request, json_object, parse_answer, probe
from dgxmm.decider.types import Option
from dgxmm.decider.secret import ensure_key, read_key
from dgxmm.engines.base import ContainerSpec, EngineMetrics, LaunchContext, SpecError, container_name, managed_labels
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate



@dataclass(frozen=True)
class DeciderModel:
    repo: str
    revision: str          # pinned snapshot; deploy/decider/<name>-<revision>.sha256 pins its code/config/manifest
    served_name: str
    reservation_bytes: int  # fixed reservation from a measured peak on GB10 (docs/decider-evaluation.md)
    params: str
    note: str


log = logging.getLogger(__name__)

DECIDER_MODELS: dict[str, DeciderModel] = {
    "vllm-sr/d3-mini": DeciderModel("vllm-sr/d3-mini", "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e", "d3-mini",
                                    13 * GIB, "4.5B", "measured peak 12.6 GiB (weights + runtime + batch 8)"),
    "vllm-sr/d3-nano": DeciderModel("vllm-sr/d3-nano", "aeb70d4dd53ed25b855c3b0a66394b8fe265db7f", "d3-nano",
                                    9 * GIB, "2.2B", "measured: 8.2 GiB released at stop after load, NVML 6.0 GiB"),
}
DEFAULT_MODEL = "vllm-sr/d3-mini"
DECIDER_REPO = DEFAULT_MODEL  # the default model (kept for callers that need "the" decider repo)
DECIDER_REVISION = DECIDER_MODELS[DEFAULT_MODEL].revision
ALLOWED_MODELS = {r: m.revision for r, m in DECIDER_MODELS.items()}  # repo -> pinned snapshot revision
SERVED_NAME = DECIDER_MODELS[DEFAULT_MODEL].served_name
SYSTEM_LABEL = "dgxmm.system"
MODEL_LABEL = "dgxmm.decider.model"  # which decision model a decider container serves (reconcile reads it)
HF_TARGET = "/models/hf"  # HF_HOME in the image (readable by its unprivileged user, unlike /root)
KEY_DIR = "/run/dgxmm-decider"  # tmpfs, mode 0700, owned by the decider user
KEY_FILE = f"{KEY_DIR}/api_key"  # written after start via docker exec stdin (DockerController.write_secret_file)
SHARED_BLOBS = "hub/blobs"  # huggingface_hub >= 2 shared content store (see container_spec)


def repo_dir(repo: str) -> str:
    """The repo's directory in the HF cache: snapshot + blobs (relative symlinks stay inside)."""
    return "hub/models--" + repo.replace("/", "--")


def hash_list_name(repo: str, revision: str) -> str:
    """deploy/decider/<model name>-<revision>.sha256 (the entrypoint derives the same name from D3_REPO)."""
    return f"{repo.rsplit('/', 1)[-1]}-{revision}.sha256"


# No Triton/kernel cache mount: vLLM's ~/.triton must not be shared (its containers load compiled launcher .so
# files from it), and the dgxmm container cannot see the host's ~/.cache to create a DGXMM-owned one. Kernels are
# compiled into the container's own filesystem at warm-up (the measured 2-4 min start-ups already ran this way).


def host_path(path: str) -> str:
    """Absolute host path for a bind mount (docker does not expand "~")."""
    p = posixpath.normpath(_expand_host(path))
    if not posixpath.isabs(p):
        raise ValueError(f"decider mount source {path!r} is not an absolute host path")
    return p



def shared_blob_mounts(snapshot: str, repo: str, settings: Settings) -> list[Mount]:
    """huggingface_hub >= 2 (also DGXMM's own downloads) keeps file contents in a shared content-addressed store,
    hub/blobs/<xx>/<hash>, and a repo's blobs are relative symlinks into it. Bind exactly the store files this
    snapshot links to, read-only, at the same path in the container; never the store directory, its .refs/.lock
    entries or other repos' files. Links that resolve outside the store (or to anything but a regular file) are not
    mounted: the entrypoint's hash check then refuses to start."""
    local = Path(settings.hf_cache.local).resolve()
    store = (local / SHARED_BLOBS).resolve()
    repo_root = (local / repo_dir(repo)).resolve()
    if not store.is_dir():
        return []
    if not store.is_relative_to(local):
        raise SpecError("invalid_mounts", f"the HF blob store {SHARED_BLOBS} is a symlink to {store}, outside the HF "
                                          f"cache ({local}); the decision model only mounts files inside the cache. "
                                          f"Move the store back into {local / SHARED_BLOBS}.")
    cache = settings.hf_cache.host.rstrip("/")
    out: dict[str, Mount] = {}
    for p in sorted(Path(snapshot).rglob("*")):
        if not p.is_symlink():
            continue
        try:
            real = p.resolve(strict=True)
        except (OSError, RuntimeError):
            continue  # broken link
        if real.is_relative_to(repo_root):
            continue  # inside the mounted repo directory (older layout, e.g. d3-mini)
        if not real.is_relative_to(store) or not real.is_file():
            log.warning("decider: %s resolves to %s (outside the HF blob store or not a file); not mounted", p, real)
            continue
        rel = real.relative_to(local)
        if any(part.startswith(".") for part in rel.relative_to(SHARED_BLOBS).parts):
            continue  # .refs / .lock bookkeeping
        out[str(rel)] = Mount(host=host_path(f"{cache}/{rel.as_posix()}"), target=f"{HF_TARGET}/{rel.as_posix()}",
                              mode="ro")
    return list(out.values())


def warmup_requests() -> list[dict]:
    """Questions shaped like the features' (long log-like state with many described options; a template excerpt;
    a short one). d3_server compiles its kernels at start, but the first requests of new lengths are still slow,
    so a deployment only counts as ready after these were answered."""
    log = "\n".join(f"(EngineCore pid=212) INFO 10-16 12:{i % 60:02d}:00 [gpu_model_runner.py:{1000 + i}] "
                     f"Loading weights shard {i}/120 took {0.1 + i / 1000:.3f}s" for i in range(60))
    causes = {f"cause_{i}": f"Failure cause number {i}: the engine stopped because of condition {i}." for i in range(12)}
    template = "\n".join(f"{{%- if tools %}}<tool_call>{{{{ tool_{i} | tojson }}}}</tool_call>{{%- endif %}}"
                         for i in range(40))
    parsers = {f"parser_{i}": f"Format {i}: tool calls are wrapped in marker {i}." for i in range(16)}
    return [
        {"state": f"Engine: vllm\nLOG (warm-up):\n<<<\n{log}\n>>>",
         "questions": {"q": {"type": "choice", "instructions": "Why did the deployment fail?", "criteria": causes}}},
        {"state": f"CHAT TEMPLATE EXCERPT (warm-up):\n<<<\n{template}\n>>>",
         "questions": {"q": {"type": "choice", "instructions": "Which tool-call parser?", "criteria": parsers}}},
        {"state": "Memory shortfall: 4.0 GiB",
         "questions": {"q": {"type": "choice", "instructions": "Which plan?",
                             "criteria": {"a": "Use fp8 KV cache.", "b": "Stop an idle deployment."}}}},
    ]


async def warmup(http: httpx.AsyncClient, base_url: str, key: str | None, timeout_s: float = 60.0) -> bool:
    """Every warm-up question gets a valid answer (capped reply, validated like a real answer: parse_answer)."""
    headers = {"authorization": f"Bearer {key}"} if key else {}
    for body in warmup_requests():
        try:
            status, content = await capped_request(http, "POST", f"{base_url}/v1/systemone", json=body,
                                                   headers=headers, timeout_s=timeout_s)
            if status != 200:
                return False
            parse_answer(content, [Option(k, k) for k in body["questions"]["q"]["criteria"]])
        except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError, RecursionError):
            return False
    return True


class DecisionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = Field(8, ge=1, le=32)


class DecisionEngine:
    name = "decision"
    supports_sleep = False
    publishes_port = False  # reached only by container name on its network (never via engine_base_host)
    system = True           # deployments of this engine are system deployments (deployments.system)
    fixed_estimate = True   # no calibration: the reservation is a measured constant
    scrape_metrics = False  # d3_server has no /metrics
    config_model = DecisionConfig
    context_key = "max_input_tokens"  # not a config field: no context-reduction plans
    seqs_key = "batch_size"

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_config(self, raw: dict[str, Any]) -> DecisionConfig:
        return DecisionConfig.model_validate(raw)

    def supports(self, artifact: ModelArtifact) -> bool:
        return (artifact.source == "hf" and artifact.ref in ALLOWED_MODELS
                and artifact.revision == ALLOWED_MODELS[artifact.ref])

    def draft_ref(self, cfg: DecisionConfig) -> str | None:
        return None

    def served_name(self, cfg: DecisionConfig, model_ref: str) -> str:
        m = DECIDER_MODELS.get(model_ref)
        return m.served_name if m is not None else SERVED_NAME

    def sleep_enabled(self, cfg: DecisionConfig) -> bool:
        return False

    def estimate(self, artifact: ModelArtifact, cfg: DecisionConfig, draft, settings: Settings) -> MemoryEstimate:
        m = DECIDER_MODELS.get(artifact.ref)
        if m is None:
            raise ValueError(f"{artifact.ref} is not a supported decision model")
        weights = min(artifact.weights_bytes, m.reservation_bytes)
        return MemoryEstimate(weights=weights, draft_weights=0, kv_cache=0, state=0, activations=0,
                              runtime=m.reservation_bytes - weights, margin=0, confidence="measured",
                              notes=(f"fixed reservation for {m.served_name} ({m.note})",))

    def container_spec(self, artifact: ModelArtifact, cfg: DecisionConfig, draft, launch: LaunchContext,
                       settings: Settings) -> ContainerSpec:
        """A locked-down container (docs/security.md): the bearer key is a 0400 file in a tmpfs written after start
        (never env / command / inspect); no published port, on an internal network only DGXMM joins; unprivileged
        user, read-only root filesystem, no capabilities, no-new-privileges, pids and memory limits, private IPC;
        read-only bind mounts of the selected model's repo dir and of exactly the shared-store files its snapshot
        links to; no docker socket."""
        key = ensure_key(settings)
        m = DECIDER_MODELS[artifact.ref]
        env = {"HF_HUB_OFFLINE": "1", "DECISION_API_KEY_FILE": KEY_FILE, "D3_REPO": m.repo, "D3_REVISION": m.revision}
        # Only the selected model's repo directory, read-only, plus the shared-store files its snapshot links to: the
        # rest of the HF cache (e.g. the host's HF token) stays out of a container that runs code shipped with the
        # model snapshot.
        rdir = repo_dir(m.repo)
        cache = settings.hf_cache.host.rstrip("/")
        mounts = [Mount(host=host_path(f"{cache}/{rdir}"), target=f"{HF_TARGET}/{rdir}", mode="ro")]
        mounts += shared_blob_mounts(artifact.local_path, m.repo, settings)
        uid, _, gid = settings.decider_user.partition(":")
        tmpfs = {
            # "exec" explicitly (docker's tmpfs default is noexec): Triton / TorchInductor dlopen the kernels they
            # compile here
            "/tmp": f"rw,exec,nosuid,nodev,size={int(settings.decider_tmp_gib * GIB)},mode=1777",
            KEY_DIR: f"rw,nosuid,nodev,noexec,size=65536,mode=0700,uid={uid},gid={gid or uid}",
        }
        internal = bool(settings.decider_network)
        return ContainerSpec(name=container_name(launch.name), image=settings.images.decision,
                             command=["--port", str(launch.port), "--batch-size", str(cfg.batch_size)],
                             environment=env, mounts=mounts,
                             labels={**managed_labels(launch.deployment_id, self.name), SYSTEM_LABEL: "true",
                                     MODEL_LABEL: m.repo},
                             network=settings.decider_network or settings.docker_network, port=launch.port,
                             publish_bind=None, ipc_host=False, user=settings.decider_user, read_only=True,
                             tmpfs=tmpfs, cap_drop=["ALL"], security_opt=["no-new-privileges"],
                             pids_limit=settings.decider_pids_limit,
                             mem_limit=int(settings.decider_memory_limit_gib * GIB), internal_network=internal,
                             secret_files={KEY_FILE: key.encode()})

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str) -> bool:
        try:
            status, content = await capped_request(http, "GET", f"{base_url}/health", timeout_s=5)
        except ValueError:  # reply larger than the cap
            return False
        if status != 200 or json_object(content).get("status") != "ready":
            return False
        key = read_key(self.settings)
        return await probe(http, base_url, key, timeout_s=60) and await warmup(http, base_url, key)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return EngineMetrics()

    async def sleep(self, http, base_url) -> None:
        raise NotImplementedError("the decision model has no sleep mode")

    async def wake(self, http, base_url) -> None:
        raise NotImplementedError("the decision model has no sleep mode")

    def kv_patch_options(self, cfg: DecisionConfig) -> list[dict]:
        return []


def is_system_engine(engine) -> bool:
    return bool(getattr(engine, "system", False))
