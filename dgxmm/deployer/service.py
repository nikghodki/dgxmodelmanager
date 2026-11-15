"""Deployment lifecycle. The only caller of DockerController.run/stop_and_remove."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from dgxmm.config import LOCAL_NODE_ID, Settings
from dgxmm.db import Database, Deployment, DeploymentEvent
from dgxmm.deployer.logtail import SCAN_LINES, oom_evidence, stored_tail
from dgxmm.deployer.gate import GateTimeout, GateWithdrawn, StartupGate
from dgxmm.deployer.states import ACTIVE_STATES, assert_transition
from dgxmm.docker_ctl import SecretDeliveryFailed
from dgxmm.engines.base import LaunchContext, SpecError, container_name
from dgxmm.engines.decision import is_system_engine
from dgxmm.events import EventBroker
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact
from dgxmm.inventory.kinds import classify_kind
from dgxmm.memory.estimate import MemoryEstimate, vllm_gpu_util
from dgxmm.memory.redact import redact_log
from dgxmm.nodes.registry import NodeUnreachable, RemoteDocker, UnknownNode, is_transport_error
from dgxmm.planner.fit import FitResult, check_fit
from dgxmm.planner.plans import ActiveDep, PlanAction, base_model_name, build_plans

log = logging.getLogger(__name__)
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,40}$"
ALIAS_PATTERN = r"^[A-Za-z0-9._:/-]{1,64}$"  # LiteLLM model name
# on_demand: started by the first request, stopped after its idle timeout (dgxmm/ondemand); always_on: as before
DeploymentMode = Literal["on_demand", "always_on"]
IDLE_TIMEOUT_MIN_S, IDLE_TIMEOUT_MAX_S = 60, 86400  # per-deployment idle_timeout_s (None = ondemand_idle_timeout_s)


# docker's error when the host port is bound by a process outside docker ("failed to bind host port ...: address
# already in use") or by a container it knows ("port is already allocated").
_PORT_CONFLICT = re.compile(r"address already in use|port is already allocated", re.I)
PORT_ATTEMPTS = 4
NODE_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"  # NodeConfig.id ("local" is the local node)
UNREACHABLE_BACKOFF_MAX_S = 60.0  # longest wait between monitor polls while a remote node is unreachable
REMOTE_LIVENESS_TIMEOUT_S = 5.0  # longest a liveness check waits on a remote node's Docker (the budget loop runs it)



CLUSTER_ENGINES = ("vllm", "llamacpp")  # engines that can run one deployment across several nodes


class Placement(BaseModel):
    """Deployment placement: single-node or multi-node cluster."""
    kind: Literal["single", "cluster"]
    nodes: list[str]  # head first

    @property
    def head(self) -> str:
        """Head node (rank 0)."""
        return self.nodes[0]

    @property
    def workers(self) -> list[str]:
        """Worker nodes (ranks 1+)."""
        return self.nodes[1:]

    def label(self, settings: Settings) -> str:
        """Human-readable placement label."""
        if self.kind == "single":
            node_id = self.nodes[0]
            node_cfg = next((n for n in settings.nodes if n.id == node_id), None)
            return node_cfg.label if node_cfg else node_id
        else:
            # cluster: join node labels with " + "
            labels = []
            for node_id in self.nodes:
                node_cfg = next((n for n in settings.nodes if n.id == node_id), None)
                labels.append(node_cfg.label if node_cfg else node_id)
            return f"{' + '.join(labels)} (cluster)"


class DeployRequest(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    engine: str = Field(pattern=r"^[a-z]{1,16}$")
    model_ref: str = Field(max_length=512)
    config: dict[str, Any] = {}
    priority: int = Field(0, ge=0, le=100)
    alias: str | None = Field(None, pattern=ALIAS_PATTERN)
    # always_on unless asked for: on demand needs LiteLLM (it starts models on requests that come through it), so
    # an API script gets it only when it says so (the deploy wizard picks on demand when LiteLLM is set up and synced)
    mode: DeploymentMode = "always_on"  # system deployments are always_on whatever is sent
    idle_timeout_s: int | None = Field(None, ge=IDLE_TIMEOUT_MIN_S, le=IDLE_TIMEOUT_MAX_S)
    node: str = Field(LOCAL_NODE_ID, pattern=NODE_ID_PATTERN)  # where it runs (a configured node id)
    placement: Placement | None = None

    def model_post_init(self, __context):
        """Map node field to placement if not explicitly set, or validate consistency."""
        if self.placement is None:
            # No placement provided, create from node
            self.placement = Placement(kind="single", nodes=[self.node])
        else:
            # Placement provided, check for conflict with explicit node
            if "node" in self.model_fields_set and self.node != self.placement.nodes[0]:
                raise ValueError(
                    f"conflicting node and placement: node={self.node!r} but "
                    f"placement.nodes[0]={self.placement.nodes[0]!r}"
                )
            # Sync node to placement head (handles default node case)
            self.node = self.placement.nodes[0]



class DeployError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, **extra):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


class InsufficientMemory(DeployError):
    def __init__(self, fit: FitResult):
        gib = 1024**3
        super().__init__("insufficient_memory",
                         f"needs {fit.required / gib:.1f} GiB but only {fit.free_for_new / gib:.1f} GiB is free", 409)
        self.fit = fit


@dataclass
class MonitorState:
    """Start-up monitor bookkeeping across polls: ``errors`` counts consecutive errors (5 fail the deployment);
    ``unreachable`` counts consecutive polls on which the deployment's remote node could not be reached (never
    counted as errors; they only lengthen the wait between polls)."""
    start: float
    errors: int = 0
    unreachable: int = 0


@dataclass
class Prepared:
    engine: Any
    artifact: ModelArtifact
    cfg: BaseModel
    draft: ModelArtifact | None
    estimate: MemoryEstimate


def new_engine_key() -> str:
    """A deployment's engine API key (Deployment.engine_api_key)."""
    return secrets.token_urlsafe(32)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


StateHook = Callable[[int, str, str], Awaitable[None]]
STARTUP_STATES = ("starting", "loading")  # states that hold a start-up slot (deployer/gate.py)
PENDING_MODE_APPLIES = frozenset({"ready", "failed", "stopped"})  # a pending mode switch takes effect here
GIB_F = float(1024**3)
_SKIPPED = object()  # a remote liveness lookup that was not done (node stale, busy or too slow)


class Deployer:
    def __init__(self, *, settings: Settings, db: Database, docker, inventory, engines: dict, http: httpx.AsyncClient,
                 events: EventBroker, budget=None, nodes=None, clock: Callable[[], float] = time.monotonic,
                 node_probe=None):
        self.settings, self.db, self.docker, self.inventory = settings, db, docker, inventory
        self.engines, self.http, self.events, self.budget, self.nodes, self.clock = engines, http, events, budget, nodes, clock
        self.node_probe = node_probe  # NodeProbe: staleness of remote nodes (None: the budget probe's, if it has one)
        self.remote_liveness_timeout_s = REMOTE_LIVENESS_TIMEOUT_S
        self._liveness_inflight: dict[str, asyncio.Future] = {}  # node -> a remote lookup that outlived its wait
        # node -> {dep id: (state, container id)} of the rows of a node reconcile skipped (deployer/reconcile.py)
        self.reconcile_pending: dict[str, dict[int, tuple]] = {}
        self.mem_lock = asyncio.Lock()
        self._locks: dict[int, asyncio.Lock] = {}
        self._monitors: dict[int, asyncio.Task] = {}
        self.state_hooks: list[StateHook] = []
        self.removal_hooks: list[Callable[[int], None]] = []  # fn(dep_id) after a deployment row is deleted
        # fn(dep_id) just before (rows that cascade with it still exist, e.g. to audit their deletion)
        self.before_removal_hooks: list[Callable[[int], None]] = []
        self.calibration = None  # set in Task 32
        self.delete_guard: Callable[[str], bool] | None = None  # WeightsDeleter.is_being_deleted
        # LiteLLMSync.error_for: why a deployment is not (correctly) registered in LiteLLM, as of the last sync
        self.litellm_error: Callable[[int, str], dict | None] | None = None
        # set by LiteLLMSync: an entry LiteLLM serves under a model name that DGXMM did not create, or None
        self.foreign_model_name: Callable[[str], Awaitable[dict | None]] | None = None
        # dep_id -> model refs while a container is being (re)launched; until it reaches "starting" the DB row may
        # still be planned/stopped/failed, so the weights deleter treats these as blocking.
        self.launching: dict[int, tuple[str, ...]] = {}
        # decision model (dgxmm/decider): returns the Decider (ctx.decider) at call time; None = rules only
        self.decider_getter: Callable[[], Any] | None = None
        self.metrics = None  # MetricsStore: traffic facts for the plan safety filter
        self.rotating: set[int] = set()  # decider deployments whose key is being rotated (dgxmm.decider.rotation)
        # at most settings.max_concurrent_startups engines starting/loading at once (deployer/gate.py)
        self.gate = StartupGate(lambda: self.settings.max_concurrent_startups)
        self._queued: dict[int, asyncio.Task] = {}  # dep_id -> task waiting for its start-up slot
        self._waking: dict[int, asyncio.Task] = {}  # dep_id -> wake-up waiting for its start-up slot
        # dep_id -> (error code, message) of the last background wake-up that failed (read by the schedule runner)
        self.wake_failures: dict[int, tuple[str, str]] = {}
        # OnDemandController.status (set by AppContext): in-flight / queued / phase of an on-demand deployment
        self.ondemand_status: Callable[[int, str], dict] | None = None
        # cluster deployments (deployer/cluster.py): dep_id -> retry of rank-container removal; first retry wait
        self._cleanups: dict[int, asyncio.Task] = {}
        self.cluster_retry_s = cluster.CLUSTER_RETRY_BASE_S

    def _check_not_deleting(self, *refs: str | None) -> None:
        for ref in refs:
            if ref and self.delete_guard is not None and self.delete_guard(ref):
                raise DeployError("model_being_deleted", f"{ref} is being deleted from disk", 409)

    def _require_single_system(self, engine_name: str) -> None:
        """Only one deployment of a system engine (the decision model) may exist at a time."""
        with self.db.session() as s:
            other = s.scalar(select(Deployment).where(Deployment.engine == engine_name))
            if other is not None:
                raise DeployError("decider_exists", f"deployment {other.name} already runs the {engine_name} engine",
                                  409, holder={"id": other.id, "name": other.name, "state": other.state})

    def _refuse_system(self, dep_id: int, what: str) -> None:
        dep = self._get(dep_id)
        if dep is not None and dep.system:
            raise DeployError("system_deployment", f"{dep.name} is a system deployment and is never {what} by a plan",
                              409)

    @staticmethod
    def _refuse_system_change(dep: Deployment, what: str) -> None:
        """System deployments (the decision model) change only through their own page: never switched to another
        model or reconfigured by the generic actions."""
        if dep.system:
            raise DeployError("system_deployment", f"{dep.name} is a system deployment and cannot be {what}; change "
                                                   "the decision model on the Decision model page", 409)

    # ---------- helpers ----------
    def _lock(self, dep_id: int) -> asyncio.Lock:
        return self._locks.setdefault(dep_id, asyncio.Lock())

    def _get(self, dep_id: int) -> Deployment | None:
        with self.db.session() as s:
            return s.get(Deployment, dep_id)

    def _require(self, dep_id: int) -> Deployment:
        dep = self._get(dep_id)
        if dep is None:
            raise DeployError("not_found", f"deployment {dep_id} not found", 404)
        return dep

    def _publishes(self, engine: str | None) -> bool:
        return getattr(self.engines.get(engine), "publishes_port", True) if engine else True

    def base_url(self, name: str, port: int, engine: str | None = None, node: str = LOCAL_NODE_ID) -> str:
        """engine_base_host (if set) for engines with a published port; the container name otherwise (the decider
        has no published port and is reachable only on its docker network). On a remote node: its cluster IP."""
        if self._is_remote(node):  # a node removed from config.yaml: DeployError unknown_node
            return f"http://{self.nodes.config(self._require_node(node)).engine_address()}:{port}"
        host = (self.settings.engine_base_host if self._publishes(engine) else None) or container_name(name)
        return f"http://{host}:{port}"

    def base_url_for(self, dep: "Deployment | dict") -> str:
        """base_url of a deployment (row or view): on the node it runs on. The one helper for callers outside the
        deployer (LiteLLM, metrics, on-demand gateway, decider). A cluster: its head's cluster IP."""
        if cluster.is_cluster(dep):
            return cluster.head_url(self, dep)
        if isinstance(dep, dict):
            name, port, engine, dep_id = dep["name"], dep["port"], dep.get("engine"), dep.get("id")
            node = self._node_of(self._get(dep_id)) if self._has_remote() and dep_id is not None else LOCAL_NODE_ID
        else:
            name, port, engine, node = dep.name, dep.port, dep.engine, self._node_of(dep)
        return self.base_url(name, port, engine, node)

    def api_key_for(self, dep: "Deployment | dict") -> str | None:
        """The engine's bearer key of a deployment (row or view; views never carry it), None without one (a
        deployment from before engine keys, a system deployment). For every DGXMM client of the engine."""
        if isinstance(dep, dict):
            dep = self._get(dep["id"]) if dep.get("id") is not None else None
        return dep.engine_api_key if dep is not None else None

    def _ensure_api_key(self, dep_id: int, prep: Prepared) -> str | None:
        """The key the engine is launched with: the deployment's own, generated now for a row that has none (from
        before engine keys); kept across restarts. None for system engines (their own key mechanism)."""
        if is_system_engine(prep.engine):
            return None
        with self.db.session() as s:
            row = s.get(Deployment, dep_id)
            if row is None:
                return None
            if not row.engine_api_key:
                row.engine_api_key = new_engine_key()
            return row.engine_api_key

    # ---------- nodes ----------
    def _has_remote(self) -> bool:
        return self.nodes is not None and any(self.nodes.is_remote(n) for n in self.nodes.ids())

    def _node_stale(self, node_id: str) -> bool:
        probe = self.node_probe if self.node_probe is not None else getattr(self.budget, "probe", None)
        stale = getattr(probe, "stale", None)
        return stale is not None and bool(stale(node_id))

    @staticmethod
    def _node_of(dep: Deployment | None) -> str:
        """The node a deployment runs on (NULL and rows from before migration 41: the local node)."""
        return (getattr(dep, "node", None) or LOCAL_NODE_ID) if dep is not None else LOCAL_NODE_ID

    def _is_remote(self, node_id: str) -> bool:
        return node_id != LOCAL_NODE_ID and self.nodes is not None

    def node_configured(self, node_id: str) -> bool:
        """False for a deployment's node that was removed from config.yaml: nothing can be reached there, so stop and
        remove only release the DB state (ruling R11)."""
        return node_id == LOCAL_NODE_ID or (self.nodes is not None and node_id in self.nodes.ids())

    def _require_node(self, node_id: str) -> str:
        if node_id != LOCAL_NODE_ID and (self.nodes is None or node_id not in self.nodes.ids()):
            raise DeployError("unknown_node", f"node {node_id!r} is not configured", 400)
        return node_id

    def _docker_for(self, node_id: str = LOCAL_NODE_ID):
        """The DockerController for a node: ``self.docker`` for the local node (exactly as before); for a remote node
        its controller, in which transport failures raise NodeUnreachable (nodes.registry.RemoteDocker). The
        controller is built on first use inside the (worker-thread) call, never on the event loop."""
        if not self._is_remote(node_id):
            return self.docker
        self._require_node(node_id)

        def controller():
            try:
                return self.nodes.docker(node_id)
            except UnknownNode:
                raise
            except Exception as e:
                if is_transport_error(e):
                    raise NodeUnreachable(f"node {node_id} is unreachable: {e}") from e
                raise
        return RemoteDocker(controller, node_id)

    def _node_budget(self, node_id: str, local):
        """The budget to fit against on ``node_id`` (``local``: the budget refresh() just returned). A remote node
        without a fresh memory reading refuses new starts (503 node_stale)."""
        if not self._is_remote(node_id):
            return local
        b = self.budget.current(node_id)
        if b is None or self._node_stale(node_id):
            raise DeployError("node_stale", f"no recent memory reading from node {node_id}; it may be offline", 503)
        return b

    def _ref_allowed(self, ref: str) -> bool:
        if re.fullmatch(REPO_RE, ref) and ".." not in ref:
            return True
        try:
            p = Path(ref)
            p_resolved = p.resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        roots = [Path(m.local).resolve() for m in self.settings.gguf_roots] + [Path(self.settings.hf_cache.local).resolve()]
        # Accept if: absolute path, has .gguf suffix, and the resolved path is under an allowed root
        return p.is_absolute() and p.suffix.lower() == ".gguf" and any(r in p_resolved.parents for r in roots)

    def _transition(self, dep_id: int, to: str, message: str = "", log_tail: str | None = None,
                    actor: str | None = None, **fields) -> dict:
        with self.db.session() as s:
            dep = s.get(Deployment, dep_id)
            frm = dep.state
            assert_transition(frm, to)
            dep.state = to
            for k, v in fields.items():
                setattr(dep, k, v)
            if to in PENDING_MODE_APPLIES and dep.pending_mode:  # a switch that waited for the start (set_mode)
                dep.mode, dep.pending_mode = dep.pending_mode, None
            s.add(DeploymentEvent(deployment_id=dep_id, from_state=frm, to_state=to, message=message, log_tail=log_tail,
                                  actor=actor[:64] if actor else None))
        if to not in STARTUP_STATES and self.gate.holders.get(dep_id) not in (None, "wake"):
            self.gate.release(dep_id)  # the start-up is over (ready / failed / stopping ...): next in queue may go
        view = self.view(dep_id)
        self.events.publish("deployment", view)
        self._publish_waiters(exclude=dep_id)
        for hook in self.state_hooks:
            asyncio.get_running_loop().create_task(self._run_hook(hook, dep_id, frm, to))
        return view

    @staticmethod
    async def _run_hook(hook: StateHook, dep_id: int, frm: str, to: str) -> None:
        try:
            await hook(dep_id, frm, to)
        except Exception:
            log.exception("state hook failed for deployment %s", dep_id)

    def _next_port(self, dep_id: int, tried: set[int]) -> int | None:
        node = self._node_of(self._get(dep_id))
        try:
            if node != LOCAL_NODE_ID:
                return self._allocate_port(exclude_dep=dep_id, skip=tried, node_id=node)
            return self._allocate_port(exclude_dep=dep_id, skip=tried)
        except (DeployError, NodeUnreachable):
            return None

    def _used_ports(self, exclude_dep: int | None = None, node_id: str = "local") -> set[int]:
        """Ports held by other active/planned deployments on the given node and by unmanaged containers (blocking: docker call)."""
        return self._db_ports(exclude_dep, node_id) | self._unmanaged_ports(node_id)

    def _db_ports(self, exclude_dep: int | None, node_id: str) -> set[int]:
        """Ports of the other active/planned deployments on ``node_id`` (DB only, no docker call), and of those being
        (re)launched right now (``launching``: e.g. a remote restart whose row is still stopped/failed while its
        container starts after mem_lock; counted like usage_rows does)."""
        with self.db.session() as s:
            rows = list(s.scalars(select(Deployment)))
        out = {d.port for d in rows
               if d.port and (d.state in ACTIVE_STATES | {"planned"} or d.id in self.launching)
               and d.id != exclude_dep and (d.node or "local") == node_id and not cluster.is_cluster(d)}
        # a cluster's ranks run on the host network and publish nothing: its API and master ports are held on
        # every node of its placement
        for d in rows:
            if cluster.is_cluster(d) and (d.state in ACTIVE_STATES | {"planned"} or d.id in self.launching) \
                    and d.id != exclude_dep and node_id in cluster.nodes_of(d):
                out |= cluster.held_ports(d)
        return out

    def _unmanaged_ports(self, node_id: str) -> set[int]:
        """Host ports of containers DGXMM does not manage on ``node_id`` (blocking: docker call). A remote node that
        cannot be reached raises NodeUnreachable (its ports are unknown, so nothing may be placed there)."""
        try:
            return {p for c in self._docker_for(node_id).list_all() for p in c.host_ports if not c.managed}
        except NodeUnreachable:
            raise
        except Exception:
            log.warning("could not list containers for the port check", exc_info=True)
            return set()

    async def _require_weights(self, node_id: str, prep: Prepared) -> None:
        """A remote node must have the snapshot in its HF cache (409 weights_missing, with the copy_weights action
        that puts it there). Checked through the node's probe container, before mem_lock and in a worker thread
        (never SSH I/O on the event loop or under the lock). A stale node is refused first (as under the lock); a
        check that cannot run is 503, never reported as missing weights."""
        from dgxmm.nodes.weights_sync import gguf_copy_ref, gguf_in_hf_cache, weights_present
        art, hf_local = prep.artifact, self.settings.hf_cache.local
        if not self._is_remote(node_id):
            return
        if art.source == "gguf" and not gguf_in_hf_cache(art, hf_local):
            return  # refused by the engine (remote_container_path: only GGUFs in the HF cache run on a node)
        if art.source not in ("hf", "gguf"):
            return
        if self._node_stale(node_id):
            raise DeployError("node_stale", f"no recent memory reading from node {node_id}; it may be offline", 503)
        probe = self.node_probe if self.node_probe is not None else getattr(self.budget, "probe", None)
        if getattr(probe, "exec", None) is None:
            raise DeployError("weights_check_failed", f"cannot check the weights on node {node_id}: no node probe",
                              503)
        try:
            present = await asyncio.to_thread(weights_present, probe, node_id, art, hf_local)
        except Exception as e:
            if is_transport_error(e):
                raise DeployError("node_unreachable", f"node {node_id} is unreachable: {e}", 503)
            raise DeployError("weights_check_failed", f"could not check the weights of {art.ref} on node {node_id}: "
                                                      f"{e}", 503)
        if not present:
            ref, revision = (art.ref, art.revision) if art.source == "hf" else (gguf_copy_ref(art, hf_local) or
                                                                                 (None, None))
            extra = {"action": {"type": "copy_weights", "node": node_id, "ref": ref, "revision": revision}} \
                if ref else {}  # a GGUF outside any snapshot: nothing copy_weights could copy
            what = f"{ref} (revision {revision})" if ref else art.ref
            raise DeployError("weights_missing", f"{what} is not in the HF cache of node {node_id}; copy it there "
                                                 "first", 409, **extra)

    async def _remote_unmanaged_ports(self, node_id: str) -> set[int]:
        """_unmanaged_ports of a remote node, taken before mem_lock (never SSH I/O while holding it)."""
        try:
            return await asyncio.to_thread(self._unmanaged_ports, node_id)
        except NodeUnreachable as e:
            raise DeployError("node_unreachable", str(e), 503)

    def _remote_port(self, dep_id: int | None, current: int | None, node_id: str, unmanaged: set[int]) -> int:
        """Under mem_lock: ``current`` if still free on the node (DB rows rechecked now + the unmanaged ports listed
        before the lock), else the lowest free port."""
        used = self._db_ports(dep_id, node_id) | unmanaged
        return current if current and current not in used else self._allocate_port(exclude_dep=dep_id, used=used,
                                                                                    node_id=node_id)

    def _allocate_port(self, exclude_dep: int | None = None, skip: set[int] | frozenset = frozenset(),
                       used: set[int] | None = None, node_id: str = "local") -> int:
        used = (self._used_ports(exclude_dep, node_id=node_id) if used is None else set(used)) | set(skip)
        lo, hi = self.settings.port_range
        for port in range(lo, hi + 1):
            if port not in used:
                return port
        raise DeployError("no_ports", "port pool exhausted", 503)

    def allocate_port(self, node_id: str) -> int:
        """Allocate a port on the specified node. Public API for tests and external callers."""
        return self._allocate_port(node_id=node_id)


    # ---------- planning ----------
    def prepare(self, engine_name: str, model_ref: str, raw_cfg: dict, existing_ref: str | None = None) -> Prepared:
        """`existing_ref`: the model an existing deployment already runs. Such a deployment stays servable when the
        inventory's kind rules later reclassify its model (e.g. a *ForCausalLM decision checkpoint is now kind
        `decision`): for it the engine is asked with the architecture-only kind DGXMM used before."""
        engine = self.engines.get(engine_name)
        if engine is None:
            raise DeployError("unknown_engine", f"engine {engine_name!r} is not available", 422)
        if not self._ref_allowed(model_ref):
            raise DeployError("invalid_model_ref", "model reference is not a repo id or an allowed GGUF path", 422)
        self._check_not_deleting(model_ref)
        try:
            cfg = engine.validate_config(raw_cfg)
        except (ValidationError, ValueError) as e:
            raise DeployError("invalid_config", str(e), 422)
        artifact = self.inventory.get(model_ref)
        if artifact is None:
            raise DeployError("model_not_found", f"{model_ref} is not downloaded", 404)
        if not artifact.complete:
            raise DeployError("model_incomplete", "; ".join(artifact.problems) or "incomplete download", 409)
        if not engine.supports(artifact) and not (
                existing_ref == model_ref and artifact.kind != "llm"
                and engine.supports(dataclasses.replace(artifact, kind=classify_kind(artifact.architecture)))):
            raise DeployError("engine_unsupported", f"{engine_name} cannot serve {artifact.format} model {model_ref}", 422)
        draft = None
        if dref := engine.draft_ref(cfg):
            self._check_not_deleting(dref)
            draft = self.inventory.get(dref)
            if draft is None or not draft.complete:
                raise DeployError("draft_not_found", f"draft model {dref} is not downloaded", 404)
        try:
            est = engine.estimate(artifact, cfg, draft, self.settings)
        except ValueError as e:
            raise DeployError("cannot_estimate", str(e), 422)
        if self.calibration is not None and not getattr(engine, "fixed_estimate", False):
            est = self.calibration.adjust(engine.name, artifact.ref, config_hash(cfg.model_dump(mode="json")), est)
        return Prepared(engine, artifact, cfg, draft, est)

    def _replacing_credit(self, replacing: int | None) -> int:
        # Credit stop-equivalent memory (max(reserved, actual)) since switch() auto-escalates
        # from sleep to stop when sleep won't free enough. This makes preview accurate.
        if replacing is None:
            return 0
        dep = self._get(replacing)
        if dep is None or dep.state not in ACTIVE_STATES:
            return 0
        current = self.budget.current()
        actual = next((u.actual for u in current.deployments if u.deployment_id == replacing), 0) if current else 0
        return max(dep.reserved_bytes, actual) if dep.state != "sleeping" else actual

    def _action_frees(self, action: PlanAction, budget) -> int:
        """Memory a plan action releases: sleep frees KV/activations of a ready deployment;
        stop frees max(reserved, actual), or only the measured memory of a sleeping one."""
        dep = self._get(action.deployment_id)
        if dep is None or dep.state not in ACTIVE_STATES:
            return 0
        actual = next((u.actual for u in budget.deployments if u.deployment_id == dep.id), 0) if budget else 0
        if action.type == "sleep":
            if not self._can_sleep(dep):
                return 0
            est = dep.estimate or {}
            return max(dep.reserved_bytes - est.get("weights", 0) - est.get("draft_weights", 0)
                       - est.get("runtime", 0), 0)
        return actual if dep.state == "sleeping" else max(dep.reserved_bytes, actual)

    async def preview(self, engine: str, model_ref: str, raw_cfg: dict, replacing: int | None = None,
                      priority: int | None = None, consult=None) -> dict:
        """`priority`: the request's priority (plan safety: never evict a higher-priority deployment). None = the
        replaced deployment's own priority (switch / reconfigure), else 0. `consult()`: may the decision model be
        asked to rank the plans (the caller's rate limit; None = yes)."""
        prep = self.prepare(engine, model_ref, raw_cfg)
        budget = await self.budget.refresh()
        fit = check_fit(prep.estimate, budget, extra_free=self._replacing_credit(replacing))
        out = {"estimate": prep.estimate.as_dict(), "fit": fit.as_dict(), "budget": budget.as_dict(),
               "served_name": prep.engine.served_name(prep.cfg, model_ref),
               "config": prep.cfg.model_dump(mode="json")}
        if engine == "vllm":
            out["gpu_memory_utilization"] = vllm_gpu_util(prep.estimate.total, budget.total)
        if hasattr(prep.engine, "pool_tokens"):
            out["kv_pool_tokens"] = prep.engine.pool_tokens(prep.cfg)
        if priority is None:
            old = self._get(replacing) if replacing is not None else None
            priority = old.priority if old is not None else 0
        if fit.fits:
            out["plans"] = []
            return out
        out["notes"] = self.system_memory_notes(engine, replacing)
        facts = await self.plan_facts(replacing)
        plans = self.plans_for(prep, fit, replacing, facts=facts)
        out["plans"] = await self.recommend_plans(plans, prep, fit, raw_cfg, replacing, priority,
                                                  facts=facts, consult=consult) if plans else plans
        return out

    def system_memory_notes(self, engine: str, replacing: int | None = None) -> list[str]:
        """A request that does not fit, while a system deployment (the decision model) holds memory: say how to free
        it. Never a plan action: plans leave system deployments alone (an admin turns the decider off explicitly)."""
        if is_system_engine(self.engines.get(engine)):
            return []
        out = []
        for v in self.list_views():
            if v["system"] and v["id"] != replacing and v["reserved_bytes"] and v["state"] in ACTIVE_STATES \
                    and v["state"] != "stopping":
                gib = f"{v['reserved_bytes'] / GIB_F:.1f}".removesuffix(".0")
                out.append(f"The decision model holds {gib} GiB — disable it on the Decision model page to free it")
        return out

    async def plan_facts(self, exclude: int | None = None) -> dict | None:
        """Traffic / dependency facts of the active deployments (dgxmm/decider/plans.py); None if unavailable.
        Used both to rank the stop/sleep plans and by the plan safety filter."""
        from dgxmm.decider.plans import facts_for
        try:
            return await asyncio.to_thread(lambda: facts_for(self.list_views(), self.metrics, exclude=exclude,
                                                             db=self.db))
        except Exception:
            log.exception("could not read plan facts; plans are ranked by priority and size only")
            return None

    async def recommend_plans(self, plans: list[dict], prep: Prepared, fit: FitResult, raw_cfg: dict,
                              replacing: int | None = None, priority: int = 0, facts: dict | None = None,
                              consult=None) -> list[dict]:
        """Safety-filter the plans and mark the recommended one (dgxmm/decider/plans.py). Fails open."""
        from dgxmm.decider.plans import facts_for, recommend
        from dgxmm.decider.protocol import NullDecider
        try:
            decider = (self.decider_getter() if self.decider_getter else None) or NullDecider()
            model = getattr(prep.engine, "config_model", None)
            defaults = {k: f.default for k, f in model.model_fields.items()} if model is not None else {}
            deps = facts if facts is not None else await asyncio.to_thread(
                lambda: facts_for(self.list_views(), self.metrics, exclude=replacing, db=self.db))
            return await recommend(plans, deps=deps, request_priority=priority,
                                   decider=decider, shortfall=max(fit.required - fit.free_for_new, 0),
                                   model_ref=prep.artifact.ref, engine=prep.engine.name,
                                   cfg=prep.cfg.model_dump(mode="json"), raw_cfg=raw_cfg or {}, defaults=defaults,
                                   context_key=getattr(prep.engine, "context_key", ""),
                                   seqs_key=getattr(prep.engine, "seqs_key", ""), consult=consult)
        except Exception:
            log.exception("plan recommendation failed; returning plans unannotated")
            return plans

    # ---------- lifecycle ----------
    def validate_placement(self, placement: Placement, engine: str, system: bool = False) -> None:
        """Validate placement configuration. Raises DeployError if invalid."""
        if placement.kind == "cluster":
            # System deployments cannot be clusters
            if system:
                raise DeployError("cluster_unsupported",
                                  "system deployments cannot be clusters", 400)
            # Only vllm and llamacpp support clustering
            if engine not in CLUSTER_ENGINES:
                raise DeployError("cluster_unsupported_engine",
                                  f"engine {engine} does not support cluster deployments", 400)
            
            # Cluster needs at least 2 distinct nodes
            if len(set(placement.nodes)) < 2:
                raise DeployError("cluster_needs_two_nodes", 
                                  "cluster deployment requires at least 2 distinct nodes", 400)
            
            # All nodes must be configured with cluster_ip and nccl
            for node_id in placement.nodes:
                node_cfg = next((n for n in self.settings.nodes if n.id == node_id), None)
                if node_cfg is None:
                    raise DeployError("cluster_node_not_ready",
                                      f"node {node_id} is not configured", 400)
                if node_cfg.cluster_ip is None or node_cfg.nccl is None:
                    raise DeployError("cluster_node_not_ready",
                                      f"node {node_id} is not ready for cluster deployments "
                                      "(missing cluster_ip or nccl config)", 400)

    def placement_options(self, req: DeployRequest, prep: Prepared | None = None) -> list:
        """Where ``req`` could run (planner.placement.PlacementOption list: each node, plus the cluster) and which
        placement to suggest; ``req.placement`` / ``req.node`` are ignored. From the budgets of the last refresh
        (the caller refreshes first). ``prep``: ``prepare()`` of ``req`` if the caller already ran it. Blocking
        (inventory, calibration DB): call it in a worker thread."""
        from dgxmm.planner.placement import build_options
        if prep is None:
            prep = self.prepare(req.engine, req.model_ref, req.config)
        budgets = self.budget.all() if self.budget is not None else {}
        return build_options(prep.engine.name, prep.estimate, self.settings, budgets, stale=self._node_stale,
                             allow_cluster=not is_system_engine(prep.engine))

    async def deploy(self, req: DeployRequest, user_id: int | None) -> dict:
        if req.placement.kind == "cluster":
            return await cluster.deploy(self, req, user_id)

        node = self._require_node(req.node)
        prep = self.prepare(req.engine, req.model_ref, req.config)
        system = is_system_engine(prep.engine)
        
        # Validate placement
        self.validate_placement(req.placement, req.engine, system)
        if system and node != LOCAL_NODE_ID:
            raise DeployError("node_unsupported", f"the {req.engine} engine runs only on the local node", 422)
        with self.db.session() as s:
            if s.scalar(select(Deployment).where(Deployment.name == req.name)):
                raise DeployError("name_in_use", f"a deployment named {req.name} already exists", 409)
        # System deployments are never registered in LiteLLM, so they neither hold nor need a model name.
        model_name = None if system else req.alias or prep.engine.served_name(prep.cfg, req.model_ref)
        self.require_model_name(model_name, None)
        await self.require_not_foreign(model_name)
        if system:
            self._require_single_system(req.engine)
        remote = self._is_remote(node)
        await self._require_weights(node, prep)  # before mem_lock, in a worker thread (SSH)
        # a remote node: its container list is read before mem_lock, its launch happens after it (never SSH I/O
        # while holding the lock that every local deploy, restart and wake needs)
        unmanaged = await self._remote_unmanaged_ports(node) if remote else set()
        async with self.mem_lock:
            self.require_model_name(model_name, None)  # again under the lock: concurrent deploys
            if system:
                self._require_single_system(req.engine)
            budget = self._node_budget(node, await self.budget.refresh())
            fit = check_fit(prep.estimate, budget)
            if not fit.fits:
                raise InsufficientMemory(fit)
            port = self._remote_port(None, None, node, unmanaged) if remote else self._allocate_port()
            with self.db.session() as s:
                dep = Deployment(name=req.name, engine=req.engine, model_ref=req.model_ref, node=node,
                                 config=prep.cfg.model_dump(mode="json"), state="planned", port=port,
                                 priority=req.priority, alias=req.alias, reserved_bytes=prep.estimate.total,
                                 estimate=prep.estimate.as_dict(), created_by=user_id, system=system or None,
                                 # the decision model is never managed by the on-demand controller
                                 mode="always_on" if system else req.mode,
                                 idle_timeout_s=None if system else req.idle_timeout_s,
                                 # the decision model has its own key (dgxmm/decider/secret.py)
                                 engine_api_key=None if system else new_engine_key(),
                                 placement=req.placement.model_dump(),
                                 ranks=[])
                s.add(dep)
                s.flush()
                dep_id = dep.id
            if remote:
                launched = self._claim_or_queue(dep_id, req.name, prep)
            else:
                launched = await self._launch_or_queue(dep_id, req.name, port, prep, budget.total)
        if remote and launched:
            await self._launch(dep_id, req.name, port, prep, budget.total)
        if launched:
            self._spawn_monitor(dep_id)
        return self.view(dep_id)

    # ---------- start-up gate ----------
    async def _launch_or_queue(self, dep_id: int, name: str, port: int, prep: Prepared, mem_total: int) -> bool:
        """Launch now if a start-up slot is free (True), else put the deployment in state "queued" and launch it
        from a background task when its turn comes (False). Called under mem_lock after the fit check."""
        if self.gate.try_acquire(dep_id, "launch", name):
            await self._launch(dep_id, name, port, prep, mem_total)
            return True
        self._queue(dep_id, name)
        return False

    def _claim_or_queue(self, dep_id: int, name: str, prep: Prepared) -> bool:
        """A remote deployment, under mem_lock after the fit check: take a start-up slot (True; the caller launches
        after releasing the lock) or queue it (False). With a slot, its reservation is counted from now
        (``launching`` -> usage_rows) so the memory stays claimed between the lock and the container start."""
        if self.gate.try_acquire(dep_id, "launch", name):
            self.launching[dep_id] = self._refs(prep)
            return True
        self._queue(dep_id, name)
        return False

    @staticmethod
    def _refs(prep: Prepared) -> tuple[str, ...]:
        return tuple(r for r in (prep.artifact.ref, prep.draft.ref if prep.draft else None) if r)

    def _queue(self, dep_id: int, name: str) -> None:
        self.gate.enqueue(dep_id, "launch", name)
        info = self.gate.info(dep_id) or {}
        self._transition(dep_id, "queued", message=f"queued: {info.get('reason', 'waiting for a start-up slot')}",
                         last_error=None, exit_code=None, oom_killed=None)
        self._spawn_queued(dep_id)

    def _spawn_queued(self, dep_id: int) -> None:
        old = self._queued.get(dep_id)
        if old and not old.done():
            return
        task = asyncio.get_running_loop().create_task(self._run_queued(dep_id))
        task.add_done_callback(lambda t: self._queued_done(dep_id, t))
        self._queued[dep_id] = task

    def _queued_done(self, dep_id: int, task: asyncio.Task) -> None:
        if self._queued.get(dep_id) is task:
            self._queued.pop(dep_id, None)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("queued start of deployment %s died with uncaught exception", dep_id)

    def resume_queued(self, dep_id: int) -> None:
        """After a dgxmm restart: put a deployment found in state "queued" back in the start-up queue."""
        dep = self._get(dep_id)
        if dep is not None:
            self.gate.enqueue(dep_id, "launch", dep.name)
        self._spawn_queued(dep_id)

    def _fail_queued(self, dep_id: int, reason: str) -> None:
        dep = self._get(dep_id)
        if dep is not None and dep.state == "queued":
            self._transition(dep_id, "failed", message=reason, last_error=reason)

    async def _run_queued(self, dep_id: int) -> None:
        dep = self._get(dep_id)
        if dep is None or dep.state != "queued":
            self.gate.withdraw(dep_id)
            return
        timeout = self.settings.startup_queue_timeout_s
        try:
            await self.gate.acquire(dep_id, timeout, "launch", dep.name)
        except GateTimeout:
            self._fail_queued(dep_id, f"timed out after {timeout:g}s waiting for a start-up slot "
                                      f"(still starting: {', '.join(self.gate.blocking()) or 'none'})")
            return
        except GateWithdrawn:
            return  # stopped or removed while it waited
        launched = False
        try:
            launched = await self._launch_queued(dep_id)
        except Exception as e:  # anything after the slot was granted: never leave a stuck queued row
            log.exception("queued start of deployment %s failed", dep_id)
            msg = e.message if isinstance(e, DeployError) else f"{type(e).__name__}: {e}"
            try:
                self._fail_queued(dep_id, f"cannot start: {msg}")
            except Exception:
                log.exception("could not mark queued deployment %s failed", dep_id)
        finally:
            if not launched:
                self.gate.release(dep_id)
        if launched:
            self._spawn_monitor(dep_id)

    async def _launch_queued(self, dep_id: int) -> bool:
        """Launch a queued deployment that holds a start-up slot; False if it was stopped/failed instead."""
        async with self._lock(dep_id):
            dep = self._get(dep_id)
            if dep is None or dep.state != "queued":
                return False  # stopped / removed while it waited
            try:
                prep = self.prepare(dep.engine, dep.model_ref, dep.config)
            except DeployError as e:
                self._fail_queued(dep_id, f"cannot start: {e.message}")
                return False
            if cluster.is_cluster(dep):
                return await cluster.launch_queued(self, dep, prep)
            node = self._node_of(dep)
            remote = self._is_remote(node)
            unmanaged = await self._remote_unmanaged_ports(node) if remote else set()  # before mem_lock
            async with self.mem_lock:
                budget = self._node_budget(node, await self.budget.refresh())
                # the queued row's reservation is already counted as committed; credit it back
                fit = check_fit(prep.estimate, budget, extra_free=max(dep.reserved_bytes, 0))
                if not fit.fits:
                    self._fail_queued(dep_id, f"not enough memory at launch time: needs "
                                              f"{fit.required / GIB_F:.1f} GiB but only "
                                              f"{fit.free_for_new / GIB_F:.1f} GiB is free")
                    return False
                port = self._remote_port(dep_id, dep.port, node, unmanaged) if remote else \
                    await self._free_port(dep_id, dep.port)
                with self.db.session() as s:
                    row = s.get(Deployment, dep_id)
                    row.port, row.reserved_bytes, row.estimate = port, prep.estimate.total, prep.estimate.as_dict()
                if not remote:
                    await self._launch(dep_id, dep.name, port, prep, budget.total)  # errors: see _run_queued
                    return True
            # remote: launched after mem_lock; the row stays "queued" (committed) until the container starts
            await self._launch(dep_id, dep.name, port, prep, budget.total)
            return True

    async def _free_port(self, dep_id: int, current: int | None) -> int:
        """``current`` unless another deployment or an unmanaged container uses it now, else a new port."""
        node = self._node_of(self._get(dep_id))
        used = await asyncio.to_thread(self._used_ports, dep_id, node)
        return current if current and current not in used else self._allocate_port(exclude_dep=dep_id, used=used)

    def _publish_waiters(self, exclude: int | None = None) -> None:
        """Queue positions / reasons changed: push the waiting deployments' views (SSE) so the UI stays current."""
        for w in list(self.gate.waiters):
            if w.dep_id == exclude:
                continue
            try:
                self.events.publish("deployment", self.view(w.dep_id))
            except DeployError:
                pass

    def startup_queue(self, dep_id: int) -> dict | None:
        return self.gate.info(dep_id)

    async def _launch(self, dep_id: int, name: str, port: int, prep: Prepared, mem_total: int) -> None:
        refs = tuple(r for r in (prep.artifact.ref, prep.draft.ref if prep.draft else None) if r)
        # Publish "launching" BEFORE consulting the guard; the deleter publishes its markers before re-planning.
        # Either it sees us launching (refuses) or we see its markers (we refuse).
        self.launching[dep_id] = refs
        started = False
        try:
            try:
                self._check_not_deleting(*refs)
            except DeployError:
                self._drop_if_planned(dep_id)
                raise  # restart/reconfigure: row keeps its stopped/failed state
            node = self._node_of(self._get(dep_id))
            docker = self._docker_for(node)
            api_key = self._ensure_api_key(dep_id, prep)
            tried: set[int] = set()
            while True:
                launch = LaunchContext(dep_id, name, port, prep.estimate.total, mem_total, node, api_key=api_key)
                try:
                    spec = prep.engine.container_spec(prep.artifact, prep.cfg, prep.draft, launch, self.settings)
                except ValueError as e:  # SpecError (a typed reason) or a config the engine cannot express
                    code = e.code if isinstance(e, SpecError) else "invalid_container_spec"
                    self._transition(dep_id, "failed", message=str(e), last_error=str(e))
                    raise DeployError(code, str(e), 422)
                if self.settings.install_id:
                    spec.labels = {**spec.labels, "dgxmm.install": self.settings.install_id}
                try:
                    await asyncio.to_thread(docker.remove_by_name, spec.name)
                    cid = await asyncio.to_thread(docker.run, spec)
                    break
                except SecretDeliveryFailed as e:
                    if e.exit_code is None:
                        self._transition(dep_id, "failed", message=f"docker error: {e}", last_error=f"docker error: {e}")
                        raise DeployError("docker_error", f"could not start container: {e}", 502)
                    # the container ran and exited (e.g. the decider's snapshot integrity check): a real failure with
                    # its own exit code and log, so diagnosis and the user see why
                    tail = await asyncio.to_thread(stored_tail, e.log_tail) if e.log_tail else e.log_tail
                    self._transition(dep_id, "failed", message=str(e), log_tail=tail, last_error=str(e),
                                     exit_code=e.exit_code)
                    raise DeployError("container_exited", str(e), 502)
                except NodeUnreachable as e:
                    self._drop_if_planned(dep_id)  # a fresh deploy creates nothing; a restart keeps its state
                    raise DeployError("node_unreachable", str(e), 503)
                except Exception as e:
                    tried.add(port)
                    # A HOST process (invisible from this container) holds the port: docker fails at start.
                    # Take the next free port a few times before giving up.
                    retry = _PORT_CONFLICT.search(str(e)) and len(tried) < PORT_ATTEMPTS
                    if not retry:
                        nxt = None
                    elif self._is_remote(node):  # lists the remote node's containers: off the event loop
                        nxt = await asyncio.to_thread(self._next_port, dep_id, tried)
                    else:
                        nxt = self._next_port(dep_id, tried)
                    if nxt is None:
                        try:  # docker creates the container before starting it: don't leave it behind
                            await asyncio.to_thread(docker.remove_by_name, spec.name)
                        except Exception:
                            log.warning("could not remove container %s after a failed start", spec.name,
                                        exc_info=True)
                        self._transition(dep_id, "failed", message=f"docker error: {e}", last_error=f"docker error: {e}")
                        raise DeployError("docker_error", f"could not start container: {e}", 502)
                    log.warning("port %s of deployment %s is held by another process; trying %s", port, dep_id, nxt)
                    port = nxt
                    with self.db.session() as s:
                        s.get(Deployment, dep_id).port = port
            self._transition(dep_id, "starting", message=f"container {cid[:12]} created", container_id=cid,
                             last_error=None, exit_code=None, oom_killed=None)
            started = True
        finally:
            self.launching.pop(dep_id, None)
            if not started and self.gate.holders.get(dep_id) == "launch":
                self.gate.release(dep_id)

    def _drop_if_planned(self, dep_id: int) -> None:
        """A fresh deploy that cannot launch: don't leave a row holding the name."""
        dep = self._get(dep_id)
        if dep is not None and dep.state == "planned":
            with self.db.session() as s:
                row = s.get(Deployment, dep_id)
                if row is not None:
                    s.delete(row)

    def _spawn_monitor(self, dep_id: int) -> None:
        old = self._monitors.get(dep_id)
        if old and not old.done():
            old.cancel()
        task = asyncio.get_running_loop().create_task(self._monitor(dep_id))
        task.add_done_callback(lambda t: self._monitor_done(dep_id, t))
        self._monitors[dep_id] = task

    def _monitor_done(self, dep_id: int, task: asyncio.Task) -> None:
        if self.gate.holders.get(dep_id) not in (None, "wake") and not task.cancelled():
            dep = self._get(dep_id)
            if dep is None or dep.state not in STARTUP_STATES:
                self.gate.release(dep_id)  # never leak a slot if the monitor ended without a transition
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("monitor task for deployment %s died with uncaught exception", dep_id)

    def nodes_by_id(self) -> dict[int, str]:
        """deployment id -> the node it runs on."""
        with self.db.session() as s:
            return {i: n or LOCAL_NODE_ID for i, n in s.execute(select(Deployment.id, Deployment.node))}

    def has_monitor(self, dep_id: int) -> bool:
        mon = self._monitors.get(dep_id)
        return mon is not None and not mon.done()

    def resume_monitor(self, dep_id: int) -> None:
        self._spawn_monitor(dep_id)

    async def _monitor(self, dep_id: int) -> None:
        st = MonitorState(start=self.clock())
        while True:
            await asyncio.sleep(self._monitor_wait(st))
            if await self._monitor_tick(dep_id, st):
                return

    def _monitor_wait(self, st: MonitorState) -> float:
        """The poll interval; while the deployment's remote node is unreachable it doubles each poll, up to
        UNREACHABLE_BACKOFF_MAX_S."""
        poll = self.settings.monitor_poll_s
        if not st.unreachable:
            return poll
        return min(UNREACHABLE_BACKOFF_MAX_S, poll * 2 ** min(st.unreachable, 30))

    async def _monitor_tick(self, dep_id: int, st: MonitorState) -> bool:
        """One monitor poll with its error accounting; True when monitoring is over."""
        try:
            if await self._monitor_step(dep_id, st.start):
                return True
            st.errors = st.unreachable = 0  # Reset on success
            return False
        except NodeUnreachable as e:  # only raised for a remote node: never counts toward the 5-errors rule
            if self.clock() - st.start > self.settings.startup_timeout_s:
                # never hold the (global) start-up slot forever: fail it; container_id is kept, so a stop once the
                # node is back removes the container
                msg = f"startup timed out after {self.settings.startup_timeout_s}s while its node was unreachable"
                try:
                    if cluster.is_cluster(self._get(dep_id)):  # every rank removed; reservations held until gone
                        await cluster.fail(self, dep_id, f"{msg}: {e}")
                    else:
                        self._transition(dep_id, "failed", message=msg, last_error=f"{msg}: {e}")
                except Exception:
                    log.exception("failed to transition deployment %s to failed after its startup timeout", dep_id)
                    self.gate.release(dep_id)
                return True
            st.unreachable += 1
            log.warning("deployment %s: its node is unreachable (poll %d, retrying in %.1fs): %s", dep_id,
                        st.unreachable, self._monitor_wait(st), e)
            return False
        except Exception as e:
            st.errors += 1
            log.exception("monitor error for deployment %s (consecutive: %d)", dep_id, st.errors)
            if st.errors >= 5:
                try:
                    if cluster.is_cluster(self._get(dep_id)):  # every rank removed; reservations held until gone
                        await cluster.fail(self, dep_id, f"monitor error: {e}")
                    else:
                        self._transition(dep_id, "failed", message="monitor error", last_error=f"monitor error: {e}")
                except Exception:
                    log.exception("failed to transition deployment %s to failed after monitor errors", dep_id)
                return True
            return False

    async def _monitor_step(self, dep_id: int, start: float) -> bool:
        """Check a starting/loading deployment once; True when it reached a final state (or is no longer starting)."""
        dep = self._get(dep_id)
        if dep is None or dep.state not in ("starting", "loading"):
            return True
        if cluster.is_cluster(dep):
            return await cluster.monitor_step(self, dep, start)
        node = self._node_of(dep)
        info = await asyncio.to_thread(self._docker_for(node).get, dep.container_id)
        if info is None:
            self._transition(dep_id, "failed", message="container disappeared", last_error="container disappeared")
            return True
        if info.status in ("exited", "dead"):
            await self._fail_exited(dep, info.exit_code, info.oom_killed)
            return True
        if dep.state == "starting" and info.status == "running":
            self._transition(dep_id, "loading", message="container running; loading model")
        if self._is_remote(node):  # the node's last memory reading (budget refresh)
            b = self.budget.current(node)
            available = b.available if b is not None else None
        else:
            available = (await asyncio.to_thread(self.budget.probe.meminfo)).available
        if available is not None and available < self.settings.margin_bytes:
            await self._abort(dep, "aborted: available memory fell below the safety margin during startup")
            return True
        if self.clock() - start > self.settings.startup_timeout_s:
            await self._abort(dep, f"startup timed out after {self.settings.startup_timeout_s}s")
            return True
        if info.status == "running" and await self._healthy(dep):
            self._transition(dep_id, "ready", message="health check and test completion passed")
            return True
        return False

    async def _healthy(self, dep: Deployment) -> bool:
        engine = self.engines[dep.engine]
        cfg = engine.validate_config(dep.config)
        base = cluster.head_url(self, dep) if cluster.is_cluster(dep) else \
            self.base_url(dep.name, dep.port, dep.engine, self._node_of(dep))
        try:
            return await engine.health(self.http, base, engine.served_name(cfg, dep.model_ref),
                                       api_key=dep.engine_api_key)
        except (httpx.HTTPError, OSError):
            return False

    def _stored_logs(self, container_id: str, node: str = LOCAL_NODE_ID) -> str:
        """The container's condensed, redacted log tail (worker thread: regexes over untrusted text)."""
        return stored_tail(self._docker_for(node).logs(container_id, SCAN_LINES))

    async def _fail_exited(self, dep: Deployment, exit_code: int | None, oom_killed: bool = False) -> None:
        """A container that exited on its own. It counts as out of memory only with evidence: Docker's OOMKilled, or
        the kernel OOM killer's line in its log. Exit 137 alone is a SIGKILL (docker kill, a stop timeout…): it is
        labelled as such and never recorded for calibration."""
        try:
            logs = await asyncio.to_thread(self._stored_logs, dep.container_id, self._node_of(dep))
        except Exception:
            log.exception("failed to capture logs for deployment %s", dep.id)
            logs = ""
        kernel = not oom_killed and bool(logs) and await asyncio.to_thread(oom_evidence, logs)
        oom = bool(oom_killed or kernel)
        if oom:
            hint = (" (out of memory: Docker reports the container was OOM-killed)" if oom_killed
                    else " (out of memory: the kernel OOM killer killed a process)")
        elif exit_code == 137:
            hint = " (killed by SIGKILL, e.g. docker kill or a stop timeout; Docker reports no out-of-memory kill)"
        else:
            hint = ""
        msg = f"container exited with code {exit_code}{hint}"
        try:
            await asyncio.to_thread(self._docker_for(self._node_of(dep)).stop_and_remove, dep.container_id)
        except Exception:
            log.exception("failed to cleanup container for deployment %s", dep.id)
        self._transition(dep.id, "failed", message=msg, log_tail=logs, last_error=msg, exit_code=exit_code,
                         oom_killed=oom, container_id=None)
        if oom and self.calibration is not None:
            self.calibration.record_oom(dep.engine, dep.model_ref, config_hash(dep.config), dep.reserved_bytes)

    LIVE_STATES = ("ready", "sleeping", "unknown")

    async def check_liveness(self) -> list[int]:
        """Re-check running deployments: a crashed / OOM-killed / vanished engine container goes to
        failed via the same path as during startup (state hooks then deregister it from LiteLLM).
        Runs in the budget loop, so a down remote node must never stall it: deployments on a stale node are skipped,
        and a remote lookup waits at most ``remote_liveness_timeout_s`` (state unchanged; while that lookup is still
        running in its worker thread, the node is skipped)."""
        with self.db.session() as s:
            ids = [d.id for d in s.scalars(select(Deployment)) if d.state in self.LIVE_STATES and d.container_id]
        failed: list[int] = []
        skip_nodes: set[str] = set()
        for dep_id in ids:
            lock = self._lock(dep_id)
            if lock.locked():  # sleep/wake/stop in progress; never stall the budget loop behind it
                continue
            try:
                async with lock:
                    dep = self._get(dep_id)
                    if dep is None or dep.state not in self.LIVE_STATES or not dep.container_id:
                        continue
                    if cluster.is_cluster(dep):
                        if await cluster.check_live(self, dep, skip_nodes):
                            failed.append(dep_id)
                        continue
                    node = self._node_of(dep)
                    if self._is_remote(node):
                        if node in skip_nodes or not self.node_configured(node):
                            continue
                        info = await self._remote_lookup(node, dep.container_id)
                        if info is _SKIPPED:
                            skip_nodes.add(node)
                            continue
                    else:
                        info = await asyncio.to_thread(self.docker.get, dep.container_id)
                    if info is None:
                        self._transition(dep_id, "failed", message="container disappeared",
                                         last_error="container disappeared", container_id=None)
                        failed.append(dep_id)
                    elif info.status in ("exited", "dead"):
                        await self._fail_exited(dep, info.exit_code, info.oom_killed)
                        failed.append(dep_id)
            except NodeUnreachable as e:  # state unchanged while its node is offline
                skip_nodes.add(self._node_of(self._get(dep_id)))
                log.warning("liveness check of deployment %s skipped: %s", dep_id, e)
            except Exception:
                log.warning("liveness check failed for deployment %s", dep_id, exc_info=True)
        return failed

    async def _remote_lookup(self, node: str, container_id: str):
        """A remote container's info, or _SKIPPED when the node is stale, still busy with an earlier lookup, or does
        not answer within remote_liveness_timeout_s (the worker thread then finishes on its own)."""
        if self._node_stale(node):
            return _SKIPPED
        pending = self._liveness_inflight.get(node)
        if pending is not None and not pending.done():
            return _SKIPPED
        fut = asyncio.ensure_future(asyncio.to_thread(self._docker_for(node).get, container_id))
        self._liveness_inflight[node] = fut
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # never "exception was never retrieved"
        try:
            return await asyncio.wait_for(asyncio.shield(fut), self.remote_liveness_timeout_s)
        except asyncio.TimeoutError:
            log.warning("liveness check on node %s timed out after %.1fs; skipped", node,
                        self.remote_liveness_timeout_s)
            return _SKIPPED

    async def _abort(self, dep: Deployment, reason: str) -> None:
        try:
            logs = await asyncio.to_thread(self._stored_logs, dep.container_id, self._node_of(dep))
        except Exception:
            log.exception("failed to capture logs for deployment %s", dep.id)
            logs = ""
        try:
            await asyncio.to_thread(self._docker_for(self._node_of(dep)).stop_and_remove, dep.container_id)
        except Exception:
            log.exception("failed to cleanup container for deployment %s", dep.id)
        self._transition(dep.id, "failed", message=reason, log_tail=logs, last_error=reason, container_id=None)

    async def wait_ready(self, dep_id: int, timeout: float) -> str:
        deadline = self.clock() + timeout
        while True:
            dep = self._require(dep_id)
            if dep.state in ("ready", "failed", "stopped", "sleeping"):
                return dep.state
            if dep.state == "queued":  # time in the start-up queue does not count (the queue has its own timeout)
                deadline = self.clock() + timeout
            if self.clock() > deadline:
                return dep.state
            await asyncio.sleep(0.02)

    async def stop(self, dep_id: int, user_id: int | None = None, reason: str = "user", precheck=None,
                   message: str | None = None, require_states: tuple[str, ...] | None = None) -> dict:
        """``precheck``: optional async fn -> reason or None, called under the deployment's lock before a ready
        deployment is stopped (the schedule runner re-checks traffic there); a reason refuses with 409 ``precheck``.
        ``message``: the "stopping" event's text instead of "stop requested (<reason>)" (e.g. the on-demand
        controller's "Put to sleep after 5 min idle"). ``require_states``: refuse (409 ``precheck``) unless the
        deployment is in one of these states under the lock; the precheck then runs in any of them."""
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if require_states is not None and dep.state not in require_states:
                raise DeployError("precheck", f"deployment is {dep.state}", 409)
            if dep.state == "stopped":
                return self.view(dep_id)
            if precheck is not None and (dep.state == "ready" or require_states is not None) and \
                    (why := await precheck()):
                raise DeployError("precheck", why, 409)
            mon = self._monitors.pop(dep_id, None)
            if mon and not mon.done():
                mon.cancel()
            if dep.state == "queued":
                q = self._queued.pop(dep_id, None)
                if q and not q.done():
                    q.cancel()
                self.gate.withdraw(dep_id)
                return self._transition(dep_id, "stopped", message=f"cancelled while queued ({reason})",
                                        actor=reason, container_id=None)
            if cluster.is_cluster(dep):
                return await cluster.stop(self, dep, reason, message)
            node = self._node_of(dep)
            if not self.node_configured(node):  # nothing to reach there: release the DB state only (R11)
                if dep.state not in ("planned", "failed", "stopping"):
                    self._transition(dep_id, "stopping", message=message or f"stop requested ({reason})", actor=reason)
                return self._transition(dep_id, "stopped", actor=reason, container_id=None,
                                        message=f"stopped without contacting node {node}, which is not configured "
                                                f"({reason}); a container left there is not removed")
            docker = self._docker_for(node)
            if dep.state in ("planned", "failed"):
                try:
                    await asyncio.to_thread(docker.stop_and_remove, dep.container_id)
                except NodeUnreachable as e:
                    raise DeployError("node_unreachable", f"could not stop container: {e}", 503)
                await self._remove_leftover(dep.name, dep_id, self._node_of(dep))
                return self._transition(dep_id, "stopped", message=f"cleaned up ({reason})", actor=reason,
                                        container_id=None)
            if dep.state != "stopping":
                self._transition(dep_id, "stopping", message=message or f"stop requested ({reason})", actor=reason)
            try:
                await asyncio.to_thread(docker.stop_and_remove, dep.container_id)
            except NodeUnreachable as e:  # failed, container id kept: a stop once the node is back removes it
                self._transition(dep_id, "failed", message=f"stop failed: {e}", last_error=f"stop failed: {e}")
                raise DeployError("node_unreachable", f"could not stop container: {e}", 503)
            except Exception as e:
                self._transition(dep_id, "failed", message=f"stop failed: {e}", last_error=f"stop failed: {e}")
                raise DeployError("docker_error", f"could not stop container: {e}", 502)
            return self._transition(dep_id, "stopped", message="container removed", container_id=None)

    async def restart(self, dep_id: int, user_id: int | None = None, config_override: dict | None = None,
                      model_ref: str | None = None, allow_rotating: bool = False) -> dict:
        if dep_id in self.rotating and not allow_rotating:
            raise DeployError("rotation_in_progress", "the decider's API key is being rotated; it restarts by itself", 409)
        if cluster.is_cluster(self._get(dep_id)):  # a row's placement never changes
            return await self._restart_cluster(dep_id, config_override, model_ref)
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", f"deployment is {dep.state}; stop it first", 409)
            ref = model_ref or dep.model_ref
            node = self._require_node(self._node_of(dep))
            remote = self._is_remote(node)
            prep = self.prepare(dep.engine, ref, config_override if config_override is not None else dep.config,
                                existing_ref=dep.model_ref)
            if not dep.system:
                await self.require_not_foreign(dep.alias or prep.engine.served_name(prep.cfg, ref))
            await self._require_weights(node, prep)  # before mem_lock, in a worker thread (SSH)
            unmanaged = await self._remote_unmanaged_ports(node) if remote else set()  # before mem_lock
            async with self.mem_lock:
                if not dep.system:
                    self.require_model_name(dep.alias or prep.engine.served_name(prep.cfg, ref), dep_id)
                budget = self._node_budget(node, await self.budget.refresh())
                fit = check_fit(prep.estimate, budget)
                if not fit.fits:
                    raise InsufficientMemory(fit)
                port = self._remote_port(dep_id, dep.port, node, unmanaged) if remote else \
                    await self._free_port(dep_id, dep.port)  # active/planned rows + unmanaged containers
                with self.db.session() as s:
                    row = s.get(Deployment, dep_id)
                    row.model_ref, row.config, row.port = ref, prep.cfg.model_dump(mode="json"), port
                    row.reserved_bytes, row.estimate = prep.estimate.total, prep.estimate.as_dict()
                if remote:
                    launched = self._claim_or_queue(dep_id, dep.name, prep)
                else:
                    launched = await self._launch_or_queue(dep_id, dep.name, port, prep, budget.total)
            if remote and launched:  # after mem_lock, still under the deployment's own lock
                await self._launch(dep_id, dep.name, port, prep, budget.total)
        if launched:
            self._spawn_monitor(dep_id)
        return self.view(dep_id)

    async def _restart_cluster(self, dep_id: int, config_override: dict | None, model_ref: str | None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", f"deployment is {dep.state}; stop it first", 409)
            launched = await cluster.restart(self, dep, model_ref or dep.model_ref,
                                             config_override if config_override is not None else dep.config)
        if launched:
            self._spawn_monitor(dep_id)
        return self.view(dep_id)

    async def remove(self, dep_id: int) -> None:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", "stop the deployment before removing it", 409)
            node = self._node_of(dep)
            if cluster.is_cluster(dep):
                await cluster.remove(self, dep)
            elif self.node_configured(node):  # else: nothing to reach there, only the row goes (R11)
                try:
                    await asyncio.to_thread(self._docker_for(node).stop_and_remove, dep.container_id)
                except NodeUnreachable as e:
                    raise DeployError("node_unreachable", f"could not remove container: {e}", 503)
                await self._remove_leftover(dep.name, dep_id, node)
            for hook in self.before_removal_hooks:
                try:
                    hook(dep_id)
                except Exception:
                    log.exception("before-removal hook failed for deployment %s", dep_id)
            with self.db.session() as s:
                s.delete(s.get(Deployment, dep_id))
            self.gate.forget(dep_id)
            if self.metrics is not None:
                try:
                    await asyncio.to_thread(self.metrics.forget, dep_id)
                except Exception:
                    log.exception("failed to drop metrics of removed deployment %s", dep_id)
            for hook in self.removal_hooks:
                try:
                    hook(dep_id)
                except Exception:
                    log.exception("removal hook failed for deployment %s", dep_id)
        self.events.publish("deployment_removed", {"id": dep_id})

    async def _remove_leftover(self, name: str, dep_id: int, node: str = LOCAL_NODE_ID) -> None:
        """A container that never started (state "created") has no container_id in the DB: remove it by name, but
        only if its labels say it belongs to this deployment."""
        fn = getattr(self._docker_for(node), "remove_leftover", None)
        if fn is not None:
            await asyncio.to_thread(fn, container_name(name), dep_id)

    async def aclose(self) -> None:
        for t in [*self._monitors.values(), *self._queued.values(), *self._waking.values(), *self._cleanups.values()]:
            t.cancel()
        self._cleanups.clear()
        self._monitors.clear()
        self._queued.clear()
        self._waking.clear()

    # ---------- views ----------
    def usage_rows(self) -> list[tuple[int, str, str, int, str | None, str]]:
        """A remote deployment that is being launched outside mem_lock (planned/stopped/failed row, in
        ``launching``) counts as "queued": its reservation is committed until the container starts."""
        with self.db.session() as s:
            rows = []
            for d in s.scalars(select(Deployment)):
                node = d.node or "local"
                if cluster.is_cluster(d):  # one row per rank, on its node
                    if d.state in ACTIVE_STATES or d.id in self.launching:
                        rows.extend(cluster.usage_rows(d, launching=d.id in self.launching))
                    continue
                if d.state in ACTIVE_STATES:
                    rows.append((d.id, d.name, d.state, d.reserved_bytes, d.container_id, node))
                elif node != LOCAL_NODE_ID and d.id in self.launching:
                    rows.append((d.id, d.name, "queued", d.reserved_bytes, d.container_id, node))
            return rows

    def _actual(self, dep_id: int) -> int | None:
        b = self.budget.current() if self.budget else None
        if not b:
            return None
        return next((u.actual for u in b.deployments if u.deployment_id == dep_id), None)

    def _read_placement(self, d: Deployment) -> dict:
        """Read placement from a Deployment row, handling NULL."""
        if d.placement is not None:
            return d.placement
        # NULL placement: read as single on the row's node
        node = d.node if d.node else "local"
        return {"kind": "single", "nodes": [node]}

    def _placement_label(self, placement: dict) -> str:
        """Generate human-readable placement label."""
        if placement["kind"] == "single":
            node_id = placement["nodes"][0]
            node_cfg = next((n for n in self.settings.nodes if n.id == node_id), None)
            return node_cfg.label if node_cfg else node_id
        else:
            # cluster: join node labels with " + " and append "(cluster)"
            labels = []
            for node_id in placement["nodes"]:
                node_cfg = next((n for n in self.settings.nodes if n.id == node_id), None)
                labels.append(node_cfg.label if node_cfg else node_id)
            return f"{' + '.join(labels)} (cluster)"

    def _view(self, d: Deployment) -> dict:
        engine = self.engines.get(d.engine)
        served = None
        if engine is not None:
            try:
                served = engine.served_name(engine.validate_config(d.config), d.model_ref)
            except (ValidationError, ValueError):
                served = None
        return {
            "id": d.id, "name": d.name, "engine": d.engine, "model_ref": d.model_ref, "state": d.state,
            "port": d.port, "priority": d.priority, "alias": d.alias, "reserved_bytes": d.reserved_bytes,
            "actual_bytes": self._actual(d.id), "estimate": d.estimate, "config": d.config,
            "litellm_status": d.litellm_status, "last_error": d.last_error, "exit_code": d.exit_code,
            "container_id": d.container_id, "served_name": served, "adopted_from": d.adopted_from,
            "system": bool(d.system),
            "mode": self._mode(d), "idle_timeout_s": d.idle_timeout_s,
            # "always_on" while a switch to always on waits for the deployment to be ready (it is on demand until then)
            "pending_mode": d.pending_mode,
            # filled by the on-demand controller; None for always-on deployments (not managed by it)
            "ondemand": self.ondemand_status(d.id, d.state) if self._mode(d) == "on_demand" and self.ondemand_status
                        is not None else None,
            "startup_queue": self.gate.info(d.id),  # why it waits for a start-up slot (state queued / waking)
            "model_name": d.alias or served,  # what clients pass as "model" to the single LiteLLM endpoint
            "litellm_error": self.litellm_error(d.id, d.state) if self.litellm_error is not None else None,
            **self._endpoints(d),
            "sleep_capable": bool(engine and engine.supports_sleep and d.config.get("enable_sleep_mode"))
                             and not cluster.is_cluster(d),  # clusters cannot sleep yet (cluster_unsupported)
            "placement": self._read_placement(d),
            "placement_label": self._placement_label(self._read_placement(d)),
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        }

    def _endpoints(self, d: Deployment) -> dict:
        node = self._node_of(d)
        if not self.node_configured(node):  # its node was removed from config.yaml: no address (R11)
            return {"endpoint": None, "host_endpoint": None}
        if cluster.is_cluster(d):  # the head serves the API on its cluster IP (host network)
            url = f"{cluster.head_url(self, d)}/v1" if d.port else None
            return {"endpoint": url, "host_endpoint": url}
        if self._is_remote(node):  # no docker DNS across hosts: both are the node's published address
            url = f"http://{self.nodes.config(node).engine_address()}:{d.port}/v1" if d.port else None
            return {"endpoint": url, "host_endpoint": url}
        return {"endpoint": f"http://{container_name(d.name)}:{d.port}/v1" if d.port else None,
                "host_endpoint": f"http://{self.settings.publish_bind}:{d.port}/v1"
                                 if d.port and self.settings.publish_bind and self._publishes(d.engine) else None}

    def view(self, dep_id: int, with_events: bool = False) -> dict:
        with self.db.session() as s:
            d = s.get(Deployment, dep_id)
            if d is None:
                raise DeployError("not_found", f"deployment {dep_id} not found", 404)
            out = self._view(d)
            if with_events:
                rows = s.scalars(select(DeploymentEvent).where(DeploymentEvent.deployment_id == dep_id)
                                 .order_by(DeploymentEvent.id.desc()).limit(50))
                out["events"] = [{"ts": e.ts.isoformat(), "from": e.from_state, "to": e.to_state,
                                  "message": e.message,
                                  # rows from older versions were stored unredacted
                                  "log_tail": redact_log(e.log_tail) if e.log_tail else e.log_tail} for e in rows]
        return out

    def list_views(self) -> list[dict]:
        with self.db.session() as s:
            return [self._view(d) for d in s.scalars(select(Deployment).order_by(Deployment.name))]

    def active_deps(self, exclude: int | None = None, facts: dict | None = None) -> list[ActiveDep]:
        """Stop/sleep candidates. ``facts`` (plan_facts) adds the ranking hints: idle minutes and whether stopping
        it would disrupt something (the safety filter would reject it)."""
        out = []
        for v in self.list_views():
            if v["id"] == exclude or v["system"] or v["state"] not in ("ready", "sleeping", "unknown", "loading", "starting"):
                continue  # system deployments are never stop/sleep candidates
            est = v["estimate"] or {}
            sleep_frees = max(v["reserved_bytes"] - est.get("weights", 0) - est.get("draft_weights", 0)
                              - est.get("runtime", 0), 0)
            f = (facts or {}).get(v["id"])
            idle, disruptive = None, v["state"] in ("starting", "loading", "unknown")
            if f is not None:
                idle = f.last_traffic_min if f.last_traffic_min is not None else (
                    f.ready_for_min if f.traffic_known else None)
                disruptive = bool(f.active_reason() or f.unknown_reason() or f.dependents)
            out.append(ActiveDep(v["id"], v["name"], v["state"], v["priority"], v["reserved_bytes"],
                                 v["actual_bytes"] or 0, v["sleep_capable"] and v["state"] == "ready", sleep_frees,
                                 idle_min=idle, disruptive=disruptive))
        return out

    def plans_for(self, prep: Prepared, fit: FitResult, replacing: int | None = None,
                  facts: dict | None = None) -> list[dict]:
        cfg = prep.cfg.model_dump(mode="json")
        name = prep.engine.name

        def est_with(patch: dict) -> int | None:
            try:
                return self.prepare(name, prep.artifact.ref, {**cfg, **patch}).estimate.total
            except DeployError:
                return None

        def est_ref(ref: str) -> int | None:
            try:
                return self.prepare(name, ref, cfg).estimate.total
            except DeployError:
                return None

        base = base_model_name(prep.artifact.ref)
        variants = [a.ref for a in self.inventory.list()
                    if a.ref != prep.artifact.ref and a.complete and a.kind == "llm" and prep.engine.supports(a)
                    and base_model_name(a.ref) == base and a.weights_bytes < prep.artifact.weights_bytes]
        plans = build_plans(fit=fit, cfg=cfg, context_key=prep.engine.context_key, seqs_key=prep.engine.seqs_key,
                            kv_patches=prep.engine.kv_patch_options(prep.cfg), estimate_with=est_with,
                            active=self.active_deps(exclude=replacing, facts=facts), variants=variants,
                            estimate_ref=est_ref)
        return [p.as_dict() for p in plans]

    def _stored_estimate(self, dep: Deployment) -> MemoryEstimate:
        if dep.estimate:
            return MemoryEstimate.from_dict(dep.estimate)
        return MemoryEstimate(dep.reserved_bytes, 0, 0, 0, 0, 0, 0, "low")

    async def sleep(self, dep_id: int, user_id: int | None = None, precheck=None) -> dict:
        """``precheck``: as for ``stop``."""
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            cluster.refuse(dep, "sleep")
            if dep.state != "ready":
                raise DeployError("not_ready", f"deployment is {dep.state}", 409)
            if precheck is not None and (why := await precheck()):
                raise DeployError("precheck", why, 409)
            engine = self.engines[dep.engine]
            cfg = engine.validate_config(dep.config)
            if not (engine.supports_sleep and engine.sleep_enabled(cfg)):
                raise DeployError("sleep_unsupported", "this deployment was not started with sleep mode", 409)
            try:
                await engine.sleep(self.http, self.base_url(dep.name, dep.port, node=self._node_of(dep)),
                                   api_key=dep.engine_api_key)
            except httpx.HTTPError as e:
                raise DeployError("engine_error", f"sleep failed: {e}", 502)
            return self._transition(dep_id, "sleeping", message="sleeping (level 1: KV freed, weights offloaded)")

    async def wake(self, dep_id: int, user_id: int | None = None) -> dict:
        """Wake a sleeping deployment. Waking re-allocates the KV cache, so it waits for the start-up slot (never
        while another engine profiles memory at start-up). The caller waits at most ``wake_sync_wait_s``; if the slot
        is still taken then, the wake-up continues in the background (up to ``wake_queue_timeout_s``) and the view
        is returned with ``startup_queue`` set (the API answers 202), like a queued deploy."""
        dep = self._require(dep_id)
        if dep.state != "sleeping":
            raise DeployError("not_sleeping", f"deployment is {dep.state}", 409)
        old = self._waking.get(dep_id)
        task = old if old is not None and not old.done() else asyncio.get_running_loop().create_task(
            self._wake_when_free(dep_id, dep.name))
        if task is not old:
            self._waking[dep_id] = task
            task.add_done_callback(lambda t: self._wake_done(dep_id, t))
        try:
            return await asyncio.wait_for(asyncio.shield(task), self.settings.wake_sync_wait_s)
        except asyncio.TimeoutError:
            return self.view(dep_id)  # still waiting for its turn: startup_queue says why

    def _wake_done(self, dep_id: int, task: asyncio.Task) -> None:
        if self._waking.get(dep_id) is task:
            self._waking.pop(dep_id, None)
        if task.cancelled():
            return
        e = task.exception()
        if e is None:
            return
        # nobody may be waiting any more (background wake-up): keep the reason on the deployment
        msg = e.message if isinstance(e, DeployError) else f"{type(e).__name__}: {e}"
        self.wake_failures[dep_id] = (e.code if isinstance(e, DeployError) else "error", msg)
        try:
            with self.db.session() as s:
                row = s.get(Deployment, dep_id)
                if row is not None and row.state == "sleeping":
                    row.last_error = f"wake failed: {msg}"
            if self._get(dep_id) is not None:
                self.events.publish("deployment", self.view(dep_id))
        except Exception:
            log.exception("could not record the failed wake of deployment %s", dep_id)

    async def _wake_when_free(self, dep_id: int, name: str) -> dict:
        dep = self._get(dep_id)
        if dep is not None and not dep.system:
            await self.require_not_foreign(self._model_name(dep))
        timeout = self.settings.wake_queue_timeout_s
        try:
            await self.gate.acquire(dep_id, timeout, "wake", name)
        except GateTimeout:
            busy = ", ".join(self.gate.blocking()) or "another deployment"
            raise DeployError("startup_busy", f"{busy} is still starting; waited {timeout:g}s to wake {name}. "
                                              "Try again when it is ready", 409)
        except GateWithdrawn:
            dep = self._get(dep_id)
            if dep is None:
                raise DeployError("not_found", f"deployment {dep_id} was removed", 404)
            raise DeployError("not_sleeping", f"deployment is {dep.state}", 409)
        try:
            async with self._lock(dep_id):
                dep = self._require(dep_id)
                if dep.state != "sleeping":
                    raise DeployError("not_sleeping", f"deployment is {dep.state}", 409)
                engine = self.engines[dep.engine]
                async with self.mem_lock:
                    self.require_model_name(self._model_name(dep), dep_id)
                    budget = self._node_budget(self._node_of(dep), await self.budget.refresh())
                    actual = next((u.actual for u in budget.deployments if u.deployment_id == dep_id), 0)
                    need = max(dep.reserved_bytes - actual, 0)
                    if need > budget.free_for_new:
                        raise InsufficientMemory(FitResult(False, need, budget.free_for_new,
                                                           budget.free_for_new - need, self._stored_estimate(dep),
                                                           budget))
                    try:
                        await engine.wake(self.http, self.base_url(dep.name, dep.port, node=self._node_of(dep)),
                                          api_key=dep.engine_api_key)
                    except httpx.HTTPError as e:
                        raise DeployError("engine_error", f"wake failed: {e}", 502)
                    return self._transition(dep_id, "ready", message="woken up", last_error=None)
        finally:
            self.gate.release(dep_id)

    async def apply_actions(self, actions, user_id: int | None = None) -> list[tuple[str, int]]:
        applied: list[tuple[str, int]] = []
        try:
            for raw in actions:
                a = raw if isinstance(raw, PlanAction) else PlanAction.model_validate(raw)
                self._refuse_system(a.deployment_id, "slept" if a.type == "sleep" else "stopped")
                if a.type == "sleep":
                    await self.sleep(a.deployment_id, user_id)
                else:
                    await self.stop(a.deployment_id, user_id, reason="plan")
                applied.append((a.type, a.deployment_id))
        except Exception:
            await self.undo_actions(applied)
            raise
        return applied

    async def undo_actions(self, applied: list[tuple[str, int]]) -> list[str]:
        failures = []
        for typ, dep_id in reversed(applied):
            try:
                if typ == "sleep":
                    await self.wake(dep_id)
                else:
                    await self.restart(dep_id)
            except Exception as e:
                log.exception("could not undo %s of deployment %s", typ, dep_id)
                failures.append(f"{typ} deployment {dep_id}: {e}")
        return failures

    # States in which a deployment holds its LiteLLM model name (it is, or is about to be, registered).
    # Sleeping/stopped/failed deployments do not; they are re-checked when they wake or start again.
    NAME_HOLDING_STATES = frozenset({"planned", "starting", "loading", "ready", "unknown"})

    def _served_name(self, d: Deployment) -> str | None:
        engine = self.engines.get(d.engine)
        if engine is None:
            return None
        try:
            return engine.served_name(engine.validate_config(d.config), d.model_ref)
        except (ValidationError, ValueError):
            return None

    def _model_name(self, d: Deployment) -> str | None:
        return d.alias or self._served_name(d)

    def model_name_holder(self, model_name: str, dep_id: int | None) -> Deployment | None:
        """The other deployment that owns ``model_name``: any deployment whose alias it is (aliases are reserved
        while the deployment exists, whatever its state), or an active deployment registered under it.
        Removing a deployment deletes its row, which frees its alias."""
        with self.db.session() as s:
            for d in s.scalars(select(Deployment).where(Deployment.id != (dep_id or -1)).order_by(Deployment.id)):
                if d.system:
                    continue  # never registered in LiteLLM
                if d.alias == model_name or (d.state in self.NAME_HOLDING_STATES and self._model_name(d) == model_name):
                    return d
        return None

    def require_model_name(self, model_name: str | None, dep_id: int | None) -> None:
        if not model_name:
            return
        d = self.model_name_holder(model_name, dep_id)
        if d is not None:
            raise DeployError("model_name_taken",
                              f"the model name {model_name} is already used by deployment {d.name} ({d.state}); "
                              "choose a different model name (alias)", 409,
                              holder={"id": d.id, "name": d.name, "state": d.state})

    async def require_not_foreign(self, model_name: str | None) -> None:
        """409 ``model_name_taken`` when LiteLLM already serves ``model_name`` from an entry DGXMM did not create
        (a static config.yaml entry or another tool's): registering next to it would make LiteLLM load-balance
        between the two. Fails open (no LiteLLM, or its model list unreadable: not taken). No holder deployment."""
        if not model_name or self.foreign_model_name is None:
            return
        found = await self.foreign_model_name(model_name)
        if found is not None:
            raise DeployError("model_name_taken",
                              f"LiteLLM already has a model named {model_name} that DGXMM didn't create (static "
                              "config or another tool). Pick another name, or remove that entry from LiteLLM first.",
                              409, holder=None, external=found)

    def set_alias(self, dep_id: int, alias: str | None) -> dict:
        """Change the LiteLLM model name. Empty/None clears it (the served name is used)."""
        alias = alias or None
        if alias is not None and not re.fullmatch(ALIAS_PATTERN, alias):
            raise DeployError("invalid_alias", "model name must be 1-64 characters: letters, digits and . _ : / -", 422)
        dep = self._require(dep_id)
        if alias is not None:
            self.require_model_name(alias, dep_id)
        elif dep.state in self.NAME_HOLDING_STATES:
            self.require_model_name(self._served_name(dep), dep_id)
        self._set_alias(dep_id, alias)
        return self.view(dep_id)

    async def rename(self, dep_id: int, alias: str | None) -> dict:
        """set_alias under mem_lock: deploy/start/wake check and claim model names while holding it, so a rename
        cannot take a name inside their await window."""
        dep = self._require(dep_id)
        new = alias or None
        target = new if new is not None else (self._served_name(dep) if dep.state in self.NAME_HOLDING_STATES else None)
        if target is not None and target != self._model_name(dep) and not dep.system:
            await self.require_not_foreign(target)  # before the lock: a LiteLLM call
        async with self.mem_lock:
            return self.set_alias(dep_id, alias)

    @staticmethod
    def _mode(d: Deployment) -> str:
        """NULL (a row an older DGXMM wrote after migration 40) and system deployments are always on."""
        return "always_on" if d.system or not d.mode else d.mode

    def set_mode(self, dep_id: int, mode: str, idle_timeout_s: int | None, pending_mode: str | None = None) -> dict:
        """Persist a deployment's mode, idle timeout (None = the global default) and pending mode (None clears it;
        applied by ``_transition`` when the deployment reaches ready, failed or stopped). Only the stored values:
        starting or stopping, and re-registering in LiteLLM, are the on-demand controller's job
        (OnDemandController.set_mode)."""
        if mode not in ("on_demand", "always_on") or pending_mode not in (None, "on_demand", "always_on"):
            raise DeployError("invalid_mode", "mode must be on_demand or always_on", 422)
        if idle_timeout_s is not None and not IDLE_TIMEOUT_MIN_S <= idle_timeout_s <= IDLE_TIMEOUT_MAX_S:
            raise DeployError("invalid_idle_timeout",
                              f"idle_timeout_s must be {IDLE_TIMEOUT_MIN_S}-{IDLE_TIMEOUT_MAX_S} seconds", 422)
        dep = self._require(dep_id)
        if dep.system:
            raise DeployError("system_deployment", f"{dep.name} is a system deployment and is always on", 409)
        if "on_demand" in (mode, pending_mode):
            cluster.refuse(dep, "on-demand mode")
        with self.db.session() as s:
            row = s.get(Deployment, dep_id)
            row.mode, row.idle_timeout_s, row.pending_mode = mode, idle_timeout_s, pending_mode
        view = self.view(dep_id)
        self.events.publish("deployment", view)
        return view

    def note(self, dep_id: int, message: str, actor: str | None = None) -> None:
        """A deployment event without a state change (e.g. the on-demand controller's "Started for a request");
        ignored for a deployment that no longer exists."""
        with self.db.session() as s:
            dep = s.get(Deployment, dep_id)
            if dep is None:
                return
            s.add(DeploymentEvent(deployment_id=dep_id, from_state=dep.state, to_state=dep.state, message=message,
                                  actor=actor[:64] if actor else None))
        self.events.publish("deployment", self.view(dep_id))

    def _set_alias(self, dep_id: int, alias: str | None) -> None:
        with self.db.session() as s:
            s.get(Deployment, dep_id).alias = alias
        self.events.publish("deployment", self.view(dep_id))

    def _can_sleep(self, dep: Deployment) -> bool:
        engine = self.engines.get(dep.engine)
        return bool(dep.state == "ready" and engine and engine.supports_sleep and dep.config.get("enable_sleep_mode"))

    async def switch(self, old_id: int, req: DeployRequest, actions, user_id: int | None) -> dict:
        old = self._require(old_id)
        self._refuse_system_change(old, "switched")
        cluster.refuse(old, "switching")
        if req.placement.kind == "cluster":
            raise cluster.unsupported("switching")
        acts = [a if isinstance(a, PlanAction) else PlanAction.model_validate(a) for a in actions]

        # Check memory fit before touching anything; auto-escalate sleep→stop if needed
        prep = self.prepare(req.engine, req.model_ref, req.config)
        budget = await self.budget.refresh()

        # Memory freed by the user's chosen actions (each deployment counted once)
        seen: set[int] = set()
        unique: list[PlanAction] = []
        for a in acts:
            if a.deployment_id not in seen:
                seen.add(a.deployment_id)
                unique.append(a)
        acts = unique
        freed = sum(self._action_frees(a, budget) for a in acts)

        # The old deployment's own credit, unless the user already chose what to do with it.
        # Prefer sleep when it is enough; otherwise stop (agrees with preview's _replacing_credit).
        if old_id not in seen and old.state in ("ready", "sleeping", "unknown"):
            sleep_act = PlanAction(type="sleep", deployment_id=old_id)
            stop_act = PlanAction(type="stop", deployment_id=old_id)
            if self._can_sleep(old) and check_fit(prep.estimate, budget,
                                                  extra_free=freed + self._action_frees(sleep_act, budget)).fits:
                chosen = sleep_act
            elif old.state == "sleeping" and check_fit(prep.estimate, budget, extra_free=freed).fits:
                chosen = None  # already asleep and nothing more is needed
            else:
                chosen = stop_act
            if chosen is not None:
                acts.insert(0, chosen)
                freed += self._action_frees(chosen, budget)

        # Final fit check
        fit = check_fit(prep.estimate, budget, extra_free=freed)
        if not fit.fits:
            raise InsufficientMemory(fit)

        alias = old.alias
        if "mode" not in req.model_fields_set:  # the replacement keeps the old deployment's mode unless one is given
            req = req.model_copy(update={"mode": self._mode(old), "idle_timeout_s": old.idle_timeout_s
                                         if "idle_timeout_s" not in req.model_fields_set else req.idle_timeout_s})
        with self.db.session() as s:
            name_taken = s.scalar(select(Deployment).where(Deployment.name == req.name)) is not None
        applied = await self.apply_actions(acts, user_id)
        new_id = None
        try:
            if alias:
                self._set_alias(old_id, None)
            new = await self.deploy(req.model_copy(update={"alias": req.alias or alias}), user_id)
            new_id = new["id"]
            state = await self.wait_ready(new_id, self.settings.startup_timeout_s + 30)
            if state != "ready":
                raise DeployError("switch_failed", f"new deployment ended in state {state}; rolled back", 502)
            return self.view(new_id)
        except Exception as original_error:
            rollback_failures = []
            if new_id is None and not name_taken:  # deploy() created the row but raised (e.g. docker error)
                with self.db.session() as s:
                    row = s.scalar(select(Deployment).where(Deployment.name == req.name))
                    new_id = row.id if row is not None else None
            if new_id is not None:
                try:
                    await self.stop(new_id, user_id, reason="switch rollback")
                    await self.remove(new_id)  # free the name so the switch can be retried
                except Exception as e:
                    log.exception("rollback: could not stop/remove %s", new_id)
                    rollback_failures.append(f"stop new deployment {new_id}: {e}")
            if alias:
                try:
                    self._set_alias(old_id, alias)
                except Exception as e:
                    log.exception("rollback: could not restore alias %s", alias)
                    rollback_failures.append(f"restore alias {alias}: {e}")
            undo_failures = await self.undo_actions(applied)
            rollback_failures.extend(undo_failures)
            if rollback_failures:
                raise DeployError("rollback_incomplete",
                                  f"switch failed ({original_error}); rollback incomplete: {', '.join(rollback_failures)}",
                                  502)
            raise

    async def reconfigure(self, dep_id: int, new_config: dict, model_ref: str | None = None, actions=(),
                          user_id: int | None = None) -> dict:
        dep = self._require(dep_id)
        self._refuse_system_change(dep, "reconfigured")
        cluster.refuse(dep, "reconfiguring")
        old_cfg, old_ref = dict(dep.config), dep.model_ref
        old_reserved, old_estimate, old_port = dep.reserved_bytes, dict(dep.estimate), dep.port
        was_active = dep.state in ("ready", "sleeping", "unknown", "starting", "loading")
        prep = self.prepare(dep.engine, model_ref or old_ref, new_config, existing_ref=old_ref)
        if not actions:
            budget = await self.budget.refresh()
            fit = check_fit(prep.estimate, budget, extra_free=self._replacing_credit(dep_id))
            if not fit.fits:
                raise InsufficientMemory(fit)
        applied = await self.apply_actions(actions, user_id)
        try:
            if was_active:
                await self.stop(dep_id, user_id, reason="reconfigure")
            await self.restart(dep_id, user_id, config_override=new_config, model_ref=model_ref)
            state = await self.wait_ready(dep_id, self.settings.startup_timeout_s + 30)
            if state != "ready":
                raise DeployError("reconfigure_failed", f"new configuration ended in state {state}; rolled back", 502)
            return self.view(dep_id)
        except Exception as original_error:
            rollback_failures = []
            now = self._get(dep_id)
            if now is not None and now.state not in ("stopped", "failed"):
                try:
                    await self.stop(dep_id, user_id, reason="reconfigure rollback")
                except Exception as e:
                    log.exception("rollback stop of %s failed", dep_id)
                    rollback_failures.append(f"stop deployment {dep_id}: {e}")
            try:
                with self.db.session() as s:  # always restore the old config, even if we cannot restart it
                    row = s.get(Deployment, dep_id)
                    row.config, row.model_ref = old_cfg, old_ref
                    if not was_active:
                        row.reserved_bytes, row.estimate, row.port = old_reserved, old_estimate, old_port
            except Exception as e:
                log.exception("rollback config restore of %s failed", dep_id)
                rollback_failures.append(f"restore config: {e}")
            if was_active:
                try:
                    await self.restart(dep_id, user_id, config_override=old_cfg, model_ref=old_ref)
                except Exception as e:
                    log.exception("rollback restart of %s failed", dep_id)
                    rollback_failures.append(f"restart deployment {dep_id}: {e}")
            undo_failures = await self.undo_actions(applied)
            rollback_failures.extend(undo_failures)
            if rollback_failures:
                raise DeployError("rollback_incomplete",
                                  f"reconfigure failed ({original_error}); rollback incomplete: {', '.join(rollback_failures)}",
                                  502)
            raise


from dgxmm.deployer import cluster  # noqa: E402  (cluster.py imports this module)
