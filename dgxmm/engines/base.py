"""Engine plug-in interface (spec §3.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from dgxmm.config import LOCAL_NODE_ID, Mount, Settings, unsafe_publish_reason
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate

SERVED_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$"
MANAGED_LABEL = "dgxmm.managed"


class SpecError(ValueError):
    """The container cannot be built from what is on disk (e.g. a mount source in the wrong place): the deployment
    fails with this code and message instead of an internal error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class ContainerSpec:
    name: str
    image: str
    command: list[str]
    environment: dict[str, str]
    mounts: list[Mount]
    labels: dict[str, str]
    network: str
    port: int
    publish_bind: str | None
    gpus: bool = True
    ipc_host: bool = True
    entrypoint: list[str] | None = None
    # Hardening (used by the decision engine; the defaults keep the vLLM / llama.cpp containers unchanged)
    user: str | None = None                   # "uid:gid"; None = the image's user
    read_only: bool = False                   # read-only root filesystem
    tmpfs: dict[str, str] = field(default_factory=dict)  # path -> mount options
    cap_drop: list[str] = field(default_factory=list)
    security_opt: list[str] = field(default_factory=list)
    pids_limit: int | None = None
    mem_limit: int | None = None              # bytes; swap is disabled for the container when set
    internal_network: bool = False            # `network` must be an internal network that DGXMM itself joins
    # Files written into the running container right after it starts (path -> content), through `docker exec`
    # stdin into a tmpfs: never in the environment, the command line or `docker inspect`. Not part of repr().
    secret_files: dict[str, bytes] = field(default_factory=dict, repr=False)
    # Multi-node cluster deployment (host networking, RDMA devices, resource limits)
    network_mode: str | None = None                     # "host" for multi-node tensor parallel; incompatible with internal_network
    devices: list[str] = field(default_factory=list)    # host device paths (e.g. "/dev/infiniband")
    cap_add: list[str] = field(default_factory=list)    # capabilities to add (e.g. "IPC_LOCK")
    ulimits: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (soft, hard); -1 = unlimited
    shm_size: str | None = None                         # shared memory size (e.g. "16g")


@dataclass
class EngineMetrics:
    gauges: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchContext:
    deployment_id: int
    name: str
    port: int
    reserved: int
    mem_total: int
    node: str = LOCAL_NODE_ID  # where the container runs (NodeConfig.id)
    # the engine's bearer key (Deployment.engine_api_key), handed over in the environment; None = no key (a
    # deployment from before engine keys, or a system deployment). Not part of repr().
    api_key: str | None = field(default=None, repr=False)


REMOTE_NETWORK = "bridge"  # a remote node's default Docker network: DGXMM's own networks exist only on the local node
HF_TARGET = "/root/.cache/huggingface"


def remote_hf_mount(settings: Settings, node_id: str) -> tuple[Mount, str]:
    """(read-only mount of the remote node's HF cache, the node's publish address). Only the node's own HF cache is
    mounted there: the local node's other host paths (e.g. vllm_cache_mounts) do not exist on it."""
    node = settings.node(node_id)
    if not node.hf_cache_host:
        raise SpecError("node_hf_cache_missing", f"node {node_id} has no hf_cache_host configured")
    bind = node.engine_address()
    if (why := unsafe_publish_reason(bind)) is not None:  # defensive: Settings validates it already
        raise SpecError("unsafe_publish_bind", f"node {node_id}: publish address {bind!r} {why}")
    return Mount(host=node.hf_cache_host, target=HF_TARGET, mode="ro"), bind


class Engine(Protocol):
    name: str
    supports_sleep: bool
    config_model: type[BaseModel]
    context_key: str
    seqs_key: str

    def validate_config(self, raw: dict[str, Any]) -> BaseModel: ...
    def supports(self, artifact: ModelArtifact) -> bool: ...
    def draft_ref(self, cfg: Any) -> str | None: ...
    def served_name(self, cfg: Any, model_ref: str) -> str: ...
    def sleep_enabled(self, cfg: Any) -> bool: ...
    def estimate(self, artifact: ModelArtifact, cfg: Any, draft: ModelArtifact | None, settings: Settings) -> MemoryEstimate: ...
    def container_spec(self, artifact: ModelArtifact, cfg: Any, draft: ModelArtifact | None,
                       launch: LaunchContext, settings: Settings) -> ContainerSpec: ...
    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str,
                     api_key: str | None = None) -> bool: ...
    def parse_metrics(self, text: str) -> EngineMetrics: ...
    async def sleep(self, http: httpx.AsyncClient, base_url: str, api_key: str | None = None) -> None: ...
    async def wake(self, http: httpx.AsyncClient, base_url: str, api_key: str | None = None) -> None: ...
    def kv_patch_options(self, cfg: Any) -> list[dict]: ...


def container_name(dep_name: str) -> str:
    return f"dgxmm-{dep_name}"


def managed_labels(deployment_id: int, engine: str) -> dict[str, str]:
    return {MANAGED_LABEL: "true", "dgxmm.deployment_id": str(deployment_id), "dgxmm.engine": engine}


def default_served_name(model_ref: str) -> str:
    import re
    stem = model_ref.rstrip("/").split("/")[-1]
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    stem = re.sub(r"-0000\d-of-0000\d$", "", stem)
    return re.sub(r"[^A-Za-z0-9._-]", "-", stem).lower()[:63] or "model"


def auth_headers(api_key: str | None) -> dict[str, str]:
    """The Authorization header for an engine's bearer key; none without a key (as before engine keys)."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


async def openai_health(http: httpx.AsyncClient, base_url: str, served_name: str, health_path: str = "/health",
                        api_key: str | None = None) -> bool:
    headers = auth_headers(api_key)
    r = await http.get(f"{base_url}{health_path}", headers=headers, timeout=5)
    if r.status_code != 200:
        return False
    r = await http.post(f"{base_url}/v1/completions", headers=headers,
                        json={"model": served_name, "prompt": "Hi", "max_tokens": 1}, timeout=120)
    return r.status_code == 200
