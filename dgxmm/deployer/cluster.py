"""Cluster deployments: one model across several nodes, one rank container per node: vLLM with tensor parallelism
(NCCL over RDMA), llama.cpp with the ggml RPC backend (ggml-rpc-server workers, the llama-server head splitting the
layers over them with --rpc / --tensor-split; RDMA auto-negotiated). Rank lifecycle helpers used by ``Deployer``
(dgxmm/deployer/service.py), which dispatches here on ``placement.kind == "cluster"``; the single-node paths never
come here.

- The row: ``node`` / ``port`` / ``container_id`` are the head's (rank 0: it serves the API on its cluster IP);
  ``ranks`` holds one entry per node {"node", "rank", "container_id", "port", "reserved"} (+ the cluster port:
  vLLM's "master_port" on rank 0, llama.cpp's "rpc_port" on every worker); ``reserved_bytes`` is the sum of the
  ranks' reservations.
- Memory: each rank's reservation (memory.estimate.per_rank_bytes) is fitted and counted on its own node:
  ``usage_rows`` yields one row per (deployment, node).
- Ports: host-network ranks publish nothing, so the head's API port and the cluster port (vLLM master port, llama.cpp
  RPC port; one value from settings.cluster_port_range) are counted on every node of the placement (``held_ports``),
  and a later deploy on any of them cannot take them.
- Start: workers first, then the head; health on the head only. Stop: the head first, then the workers.
- Failure: any rank exiting or vanishing fails the whole deployment: it goes "stopping", every rank is removed (head
  first), and only once every rank container is confirmed gone does it become "failed" (or "stopped"); until then the
  reservation stays committed and removal is retried with backoff (``cleanup``).
- Sleep, switch, on-demand mode and reconfigure are refused (``cluster_unsupported``)."""
from __future__ import annotations

import asyncio
import logging

from dgxmm.config import LOCAL_NODE_ID
from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer.logtail import SCAN_LINES, oom_evidence, stored_tail
from dgxmm.deployer.service import _SKIPPED, DeployError, InsufficientMemory, Placement, Prepared, new_engine_key
from dgxmm.deployer.states import ACTIVE_STATES
from dgxmm.engines.base import LaunchContext, SpecError, container_name
from dgxmm.engines.decision import is_system_engine
from dgxmm.memory.budget import RELEASING
from dgxmm.nodes.registry import NodeUnreachable
from dgxmm.planner.fit import FitResult

log = logging.getLogger(__name__)

CLUSTER_RETRY_BASE_S = 1.0  # first wait before retrying the removal of rank containers (doubles each try)
CLUSTER_RETRY_MAX_S = 60.0


# ---------- the row ----------
def placement_of(dep) -> dict:
    p = getattr(dep, "placement", None) if not isinstance(dep, dict) else dep.get("placement")
    return p or {"kind": "single", "nodes": [(dep.get("node") if isinstance(dep, dict) else dep.node) or LOCAL_NODE_ID]}


def is_cluster(dep) -> bool:
    """A deployment (row or view) placed on several nodes as one cluster."""
    if dep is None:
        return False
    p = dep.get("placement") if isinstance(dep, dict) else getattr(dep, "placement", None)
    return bool(p) and p.get("kind") == "cluster"


def nodes_of(dep) -> list[str]:
    return list(placement_of(dep)["nodes"])


def ranks_of(dep) -> list[dict]:
    """A fresh copy of the rank entries (rank order), safe to modify and store back."""
    return sorted((dict(r) for r in (dep.ranks or [])), key=lambda r: r["rank"])


def rank_name(name: str, rank: int) -> str:
    return f"{container_name(name)}-r{rank}"


CLUSTER_PORT_KEYS = ("master_port", "rpc_port")  # vLLM: rank 0's master port; llama.cpp: each worker's RPC port


def cluster_port(dep) -> int | None:
    """The deployment's cluster port: vLLM's master port (rank 0) or llama.cpp's RPC port (the workers')."""
    return next((r[k] for r in ranks_of(dep) for k in CLUSTER_PORT_KEYS if r.get(k)), None)


def held_ports(dep) -> set[int]:
    """Ports a cluster holds on every node of its placement: the head's API port and the cluster port(s)."""
    return {p for p in (dep.port, *(r.get(k) for r in (dep.ranks or []) for k in CLUSTER_PORT_KEYS)) if p}


def head_url(d, dep) -> str:
    """http://<head cluster_ip>:<port>: where the head serves the API (LiteLLM, health, scraper, gateway)."""
    port = dep["port"] if isinstance(dep, dict) else dep.port
    head = d._require_node(nodes_of(dep)[0])  # a head removed from config.yaml: DeployError unknown_node
    return f"http://{d.settings.node(head).cluster_ip}:{port}"


def usage_rows(dep, launching: bool) -> list[tuple]:
    """One budget row per rank, on its node. ``launching``: a row being launched outside mem_lock (counted as
    queued). While stopping, a rank whose container is not confirmed gone stays committed (RELEASING); a rank whose
    container is gone frees its node's memory."""
    state = dep.state if dep.state in ACTIVE_STATES else "queued"
    out = []
    for r in ranks_of(dep):
        if dep.state == "stopping":
            if not r.get("container_id"):
                continue
            row_state = RELEASING
        else:
            row_state = state
        out.append((dep.id, dep.name, row_state, int(r.get("reserved") or 0), r.get("container_id"), r["node"]))
    return out


def _save_ranks(d, dep_id: int, ranks: list[dict], **fields) -> None:
    with d.db.session() as s:
        row = s.get(Deployment, dep_id)
        if row is None:
            return
        row.ranks = [dict(r) for r in ranks]  # a new list: JSON columns are not mutation-tracked
        for k, v in fields.items():
            setattr(row, k, v)


def unsupported(what: str) -> DeployError:
    return DeployError("cluster_unsupported", f"{what} is not supported for cluster deployments yet", 409)


def refuse(dep, what: str) -> None:
    if is_cluster(dep):
        raise unsupported(what)


# ---------- fit and ports ----------
async def _unmanaged(d, nodes: list[str]) -> dict[str, set[int]]:
    """Host ports of unmanaged containers on each node, read before mem_lock (in worker threads)."""
    out = {}
    for n in nodes:
        try:
            out[n] = await asyncio.to_thread(d._unmanaged_ports, n)
        except NodeUnreachable as e:
            raise DeployError("node_unreachable", str(e), 503)
    return out


async def _fit(d, prep: Prepared, placement: Placement, credit: dict[str, int] | None = None):
    """Under mem_lock: (per-node budgets, each rank's reservation, each node's memory total). Raises
    InsufficientMemory (extra ``node``) for the first node its rank does not fit on."""
    from dgxmm.planner.placement import rank_bytes
    local = await d.budget.refresh()
    budgets = {n: d._node_budget(n, local) for n in placement.nodes}
    need = rank_bytes(prep.engine.name, prep.estimate, placement, budgets)
    for n in placement.nodes:
        b = budgets[n]
        free = b.free_for_new + (credit or {}).get(n, 0)
        if need[n] > free:
            e = InsufficientMemory(FitResult(False, need[n], free, free - need[n], prep.estimate, b))
            e.message = f"node {n}: {e.message}"
            e.args = (e.message,)
            e.extra["node"] = n
            raise e
    return budgets, need, {n: budgets[n].total for n in placement.nodes}


def _ports(d, dep_id: int | None, nodes: list[str], unmanaged: dict[str, set[int]], cur_port: int | None,
           cur_master: int | None) -> tuple[int, int]:
    """Under mem_lock: (API port, cluster port), each the current one if still free on every node of the
    placement."""
    used: set[int] = set()
    for n in nodes:
        used |= d._db_ports(dep_id, n) | unmanaged.get(n, set())
    port = cur_port if cur_port and cur_port not in used else d._allocate_port(exclude_dep=dep_id, used=used,
                                                                              node_id=nodes[0])
    lo, hi = d.settings.cluster_port_range
    taken = used | {port}
    if cur_master and lo <= cur_master <= hi and cur_master not in taken:
        return port, cur_master
    for p in range(lo, hi + 1):
        if p not in taken:
            return port, p
    raise DeployError("no_ports", "cluster port pool exhausted", 503)


def _fresh_ranks(engine: str, nodes: list[str], need: dict[str, int], port: int, cport: int) -> list[dict]:
    ranks = []
    for i, n in enumerate(nodes):
        r = {"node": n, "rank": i, "container_id": None, "port": port if i == 0 else None, "reserved": need[n]}
        if engine == "llamacpp":
            if i > 0:  # each worker's ggml-rpc-server listens on it (on its cluster IP)
                r["rpc_port"] = cport
        elif i == 0:
            r["master_port"] = cport
        ranks.append(r)
    return ranks


async def _checks(d, prep: Prepared, placement: Placement) -> dict[str, set[int]]:
    """Before mem_lock: every node configured, the weights on every node (worker threads), unmanaged ports."""
    for n in placement.nodes:
        d._require_node(n)
    for n in placement.nodes:
        await d._require_weights(n, prep)
    return await _unmanaged(d, placement.nodes)


# ---------- deploy / restart / queued ----------
async def deploy(d, req, user_id: int | None) -> dict:
    from sqlalchemy import select
    placement = req.placement
    prep = d.prepare(req.engine, req.model_ref, req.config)
    d.validate_placement(placement, req.engine, is_system_engine(prep.engine))  # refuses system engines
    if req.mode == "on_demand":
        raise unsupported("on-demand mode")
    with d.db.session() as s:
        if s.scalar(select(Deployment).where(Deployment.name == req.name)):
            raise DeployError("name_in_use", f"a deployment named {req.name} already exists", 409)
    model_name = req.alias or prep.engine.served_name(prep.cfg, req.model_ref)
    d.require_model_name(model_name, None)
    await d.require_not_foreign(model_name)
    unmanaged = await _checks(d, prep, placement)
    async with d.mem_lock:
        d.require_model_name(model_name, None)  # again under the lock: concurrent deploys
        _, need, mem_total = await _fit(d, prep, placement)
        port, mport = _ports(d, None, placement.nodes, unmanaged, None, None)
        with d.db.session() as s:
            dep = Deployment(name=req.name, engine=req.engine, model_ref=req.model_ref, node=placement.head,
                             config=prep.cfg.model_dump(mode="json"), state="planned", port=port,
                             priority=req.priority, alias=req.alias, reserved_bytes=sum(need.values()),
                             estimate=prep.estimate.as_dict(), created_by=user_id, system=None, mode="always_on",
                             idle_timeout_s=req.idle_timeout_s, engine_api_key=new_engine_key(),
                             placement=placement.model_dump(),
                             ranks=_fresh_ranks(req.engine, placement.nodes, need, port, mport))
            s.add(dep)
            s.flush()
            dep_id = dep.id
        launched = d._claim_or_queue(dep_id, req.name, prep)
    if launched:
        await launch(d, dep_id, prep, mem_total)
        d._spawn_monitor(dep_id)
    return d.view(dep_id)


async def restart(d, dep: Deployment, ref: str, raw_cfg: dict) -> bool:
    """Under the deployment's lock, ``dep`` stopped or failed. True if launched (the caller spawns the monitor)."""
    prep = d.prepare(dep.engine, ref, raw_cfg, existing_ref=dep.model_ref)
    placement = Placement(**placement_of(dep))
    name = dep.alias or prep.engine.served_name(prep.cfg, ref)
    await d.require_not_foreign(name)
    unmanaged = await _checks(d, prep, placement)
    # the recorded ranks of the previous run must be gone before any new rank starts (same names, same cluster port)
    if not await remove_ranks(d, dep.id):
        raise DeployError("docker_error", "rank containers of the previous run could not all be removed; try again",
                          502)
    async with d.mem_lock:
        d.require_model_name(name, dep.id)
        _, need, mem_total = await _fit(d, prep, placement)
        port, mport = _ports(d, dep.id, placement.nodes, unmanaged, dep.port, cluster_port(dep))
        _save_ranks(d, dep.id, _fresh_ranks(dep.engine, placement.nodes, need, port, mport), model_ref=ref,
                    config=prep.cfg.model_dump(mode="json"), port=port, reserved_bytes=sum(need.values()),
                    estimate=prep.estimate.as_dict())
        launched = d._claim_or_queue(dep.id, dep.name, prep)
    if launched:
        await launch(d, dep.id, prep, mem_total)
    return launched


async def launch_queued(d, dep: Deployment, prep: Prepared) -> bool:
    """A queued cluster that holds a start-up slot (under its lock). False if it failed instead."""
    placement = Placement(**placement_of(dep))
    try:
        unmanaged = await _checks(d, prep, placement)
    except DeployError as e:
        d._fail_queued(dep.id, f"cannot start: {e.message}")
        return False
    async with d.mem_lock:
        # the queued row's reservations are already counted on their nodes; credit them back
        credit = {r["node"]: int(r.get("reserved") or 0) for r in ranks_of(dep)}
        try:
            _, need, mem_total = await _fit(d, prep, placement, credit)
        except InsufficientMemory as e:
            d._fail_queued(dep.id, f"not enough memory at launch time: {e.message}")
            return False
        port, mport = _ports(d, dep.id, placement.nodes, unmanaged, dep.port, cluster_port(dep))
        _save_ranks(d, dep.id, _fresh_ranks(dep.engine, placement.nodes, need, port, mport), port=port,
                    reserved_bytes=sum(need.values()), estimate=prep.estimate.as_dict())
    await launch(d, dep.id, prep, mem_total)  # after mem_lock; the row stays queued (committed) until it starts
    return True


async def launch(d, dep_id: int, prep: Prepared, mem_total: dict[str, int]) -> None:
    """Run every rank (workers first, the head last), recording each container id as soon as it exists. The row goes
    "starting" with the first container; a later failure removes what was started (``fail``)."""
    refs = d._refs(prep)
    d.launching[dep_id] = refs
    started = False
    try:
        try:
            d._check_not_deleting(*refs)
        except DeployError:
            d._drop_if_planned(dep_id)
            raise
        dep = d._get(dep_id)
        placement = Placement(**placement_of(dep))
        ranks = ranks_of(dep)
        api_key = d._ensure_api_key(dep_id, prep)
        head = placement.head
        ctx = LaunchContext(dep_id, dep.name, dep.port, dep.reserved_bytes, mem_total[head], head, api_key=api_key,
                            rank_memory={r["node"]: (int(r["reserved"]), mem_total[r["node"]]) for r in ranks})
        try:
            specs = prep.engine.cluster_specs(prep.artifact, prep.cfg, prep.draft, ctx, placement, d.nodes,
                                              cluster_port(dep), d.settings)
        except ValueError as e:
            code = e.code if isinstance(e, SpecError) else "invalid_container_spec"
            d._transition(dep_id, "failed", message=str(e), last_error=str(e))
            raise DeployError(code, str(e), 422)
        if d.settings.install_id:
            for rs in specs:
                rs.spec.labels = {**rs.spec.labels, "dgxmm.install": d.settings.install_id}
        by_rank = {r["rank"]: r for r in ranks}
        created = False
        for rs in sorted(specs, key=lambda rs: (rs.rank == 0, rs.rank)):  # workers first, then the head
            try:
                docker = d._docker_for(rs.node)
                await asyncio.to_thread(docker.remove_by_name, rs.spec.name)
                cid = await asyncio.to_thread(docker.run, rs.spec)
            except Exception as e:
                code, status = ("node_unreachable", 503) if isinstance(e, NodeUnreachable) else ("docker_error", 502)
                msg = f"could not start rank {rs.rank} on node {rs.node}: {e}"
                if created:  # the other ranks run: the whole deployment fails, every rank is removed
                    await fail(d, dep_id, msg)
                else:
                    try:  # docker creates before it starts: never leave the container behind
                        await asyncio.to_thread(d._docker_for(rs.node).remove_leftover, rs.spec.name, dep_id)
                    except Exception:
                        log.warning("could not remove %s after a failed start", rs.spec.name, exc_info=True)
                    if isinstance(e, NodeUnreachable):
                        d._drop_if_planned(dep_id)  # a fresh deploy creates nothing; a restart keeps its state
                    else:
                        d._transition(dep_id, "failed", message=f"docker error: {msg}",
                                      last_error=f"docker error: {msg}")
                raise DeployError(code, msg, status)
            by_rank[rs.rank]["container_id"] = cid
            ranks = [by_rank[k] for k in sorted(by_rank)]
            # the head's id is the row's container_id: stored in the same commit as the ranks
            _save_ranks(d, dep_id, ranks, **({"container_id": cid} if rs.rank == 0 else {}))
            if not created:
                created = True
                d._transition(dep_id, "starting", message=f"rank {rs.rank} container {cid[:12]} created on node "
                                                          f"{rs.node}", last_error=None, exit_code=None,
                              oom_killed=None)
        d.events.publish("deployment", d.view(dep_id))
        started = True
    finally:
        d.launching.pop(dep_id, None)
        if not started and d.gate.holders.get(dep_id) == "launch":
            d.gate.release(dep_id)


# ---------- monitor / liveness / failure ----------
def _exit_message(rank: dict, exit_code: int | None, oom: bool, oom_killed: bool) -> str:
    if oom:
        hint = (" (out of memory: Docker reports the container was OOM-killed)" if oom_killed
                else " (out of memory: the kernel OOM killer killed a process)")
    elif exit_code == 137:
        hint = " (killed by SIGKILL, e.g. docker kill or a stop timeout; Docker reports no out-of-memory kill)"
    else:
        hint = ""
    return f"rank {rank['rank']} on node {rank['node']}: container exited with code {exit_code}{hint}"


async def _rank_logs(d, rank: dict) -> str:
    try:
        return await asyncio.to_thread(lambda: stored_tail(d._docker_for(rank["node"]).logs(rank["container_id"],
                                                                                            SCAN_LINES)))
    except Exception:
        log.exception("failed to capture logs of rank %s on node %s", rank["rank"], rank["node"])
        return ""


async def _fail_rank(d, dep: Deployment, rank: dict, info) -> None:
    """``rank``'s container vanished (``info`` None) or exited: the whole deployment fails."""
    if info is None:
        await fail(d, dep.id, f"rank {rank['rank']} container on node {rank['node']} disappeared")
        return
    logs = await _rank_logs(d, rank)
    kernel = not info.oom_killed and bool(logs) and await asyncio.to_thread(oom_evidence, logs)
    oom = bool(info.oom_killed or kernel)
    await fail(d, dep.id, _exit_message(rank, info.exit_code, oom, info.oom_killed), log_tail=logs,
               exit_code=info.exit_code, oom_killed=oom)


async def _infos(d, dep: Deployment) -> tuple[list[tuple[dict, object]], NodeUnreachable | None]:
    """(rank, its container info) of every reachable rank (blocking lookups in worker threads), and the
    NodeUnreachable of the first rank whose node could not be reached (None: all reached)."""
    out, unreachable = [], None
    for r in ranks_of(dep):
        try:
            info = await asyncio.to_thread(d._docker_for(r["node"]).get, r["container_id"]) \
                if r.get("container_id") else None
        except NodeUnreachable as e:
            unreachable = unreachable or e
            continue
        out.append((r, info))
    return out, unreachable


async def monitor_step(d, dep: Deployment, start: float) -> bool:
    """Deployer._monitor_step of a cluster: every rank checked; health on the head only. A rank that exited on a
    reachable node fails the deployment even while another node is unreachable; otherwise an unreachable node raises
    NodeUnreachable (Deployer._monitor_tick: backoff, then the start-up timeout)."""
    infos, unreachable = await _infos(d, dep)
    for r, info in infos:
        if info is None or info.status in ("exited", "dead"):
            await _fail_rank(d, dep, r, info)
            return True
    if unreachable is not None:
        raise unreachable
    running = all(info.status == "running" for _, info in infos)
    if dep.state == "starting" and running:
        d._transition(dep.id, "loading", message="rank containers running; loading model")
    for node in nodes_of(dep):
        if d._is_remote(node):
            b = d.budget.current(node)
            available = b.available if b is not None else None
        else:
            available = (await asyncio.to_thread(d.budget.probe.meminfo)).available
        if available is not None and available < d.settings.margin_bytes:
            await fail(d, dep.id, f"aborted: available memory on node {node} fell below the safety margin during "
                                  "startup")
            return True
    if d.clock() - start > d.settings.startup_timeout_s:
        await fail(d, dep.id, f"startup timed out after {d.settings.startup_timeout_s}s")
        return True
    if running and await d._healthy(dep):
        d._transition(dep.id, "ready", message="health check and test completion passed")
        return True
    return False


async def check_live(d, dep: Deployment, skip_nodes: set[str]) -> bool:
    """Deployer.check_liveness of a cluster (under its lock): True if it failed. Skipped (nothing changes) when any
    of its nodes is stale, busy, too slow or unreachable."""
    infos = []
    for r in ranks_of(dep):  # an unreachable rank is skipped; the reachable ones are still evaluated
        node = r["node"]
        if d._is_remote(node):
            if node in skip_nodes or not d.node_configured(node):
                continue
            try:
                info = await d._remote_lookup(node, r.get("container_id"))
            except NodeUnreachable as e:
                skip_nodes.add(node)
                log.warning("liveness check of deployment %s on node %s skipped: %s", dep.id, node, e)
                continue
            if info is _SKIPPED:
                skip_nodes.add(node)
                continue
        else:
            info = await asyncio.to_thread(d.docker.get, r.get("container_id"))
        infos.append((r, info))
    for r, info in infos:
        if info is None or info.status in ("exited", "dead"):
            await _fail_rank(d, dep, r, info)
            return True
    return False


async def remove_ranks(d, dep_id: int) -> bool:
    """Stop and remove every rank container (the head first, then the workers); True once all are confirmed gone. A
    rank on a node that is no longer configured counts as gone (nothing can be reached there, ruling R11)."""
    dep = d._get(dep_id)
    if dep is None:
        return True
    ranks = ranks_of(dep)
    gone = True
    for r in ranks:  # rank order: the head (0) first
        node = r["node"]
        if not d.node_configured(node):
            r["container_id"] = None
            continue
        try:
            docker = d._docker_for(node)
            if r.get("container_id"):
                await asyncio.to_thread(docker.stop_and_remove, r["container_id"])
            name = rank_name(dep.name, r["rank"])
            if not await asyncio.to_thread(docker.remove_leftover, name, dep_id):
                left = await asyncio.to_thread(docker.get, name)  # remove_leftover logs its errors, never raises
                if left is not None and left.labels.get("dgxmm.deployment_id") == str(dep_id):
                    raise RuntimeError(f"container {name} is still there")
            r["container_id"] = None
        except Exception as e:
            gone = False
            log.warning("deployment %s: could not remove rank %s on node %s: %s", dep_id, r["rank"], node, e)
        _save_ranks(d, dep_id, ranks)
    _save_ranks(d, dep_id, ranks, **({"container_id": None} if gone else {}))
    return gone


async def fail(d, dep_id: int, reason: str, **fields) -> None:
    """The whole deployment fails: "stopping" (reservation held), every rank removed, then "failed" once all are
    confirmed gone; otherwise it stays "stopping" and the removal is retried with backoff."""
    dep = d._get(dep_id)
    if dep is None:
        return
    if dep.state != "stopping":
        d._transition(dep_id, "stopping", message=f"{reason}; stopping every rank", last_error=reason)
    final = {"message": reason, "last_error": reason, **fields}
    if await remove_ranks(d, dep_id):
        d._transition(dep_id, "failed", container_id=None, **final)
    else:
        spawn_cleanup(d, dep_id, "failed", final)


def spawn_cleanup(d, dep_id: int, to: str, fields: dict) -> None:
    old = d._cleanups.get(dep_id)
    if old is not None and not old.done():
        old.cancel()
    task = asyncio.get_running_loop().create_task(_cleanup(d, dep_id, to, fields))
    task.add_done_callback(lambda t: _cleanup_done(d, dep_id, t))
    d._cleanups[dep_id] = task


def _cleanup_done(d, dep_id: int, task: asyncio.Task) -> None:
    if d._cleanups.get(dep_id) is task:
        d._cleanups.pop(dep_id, None)
    if not task.cancelled() and task.exception() is not None:
        log.error("cleanup of deployment %s died", dep_id, exc_info=task.exception())


async def _cleanup(d, dep_id: int, to: str, fields: dict) -> None:
    """Retry removing the remaining rank containers (backoff from d.cluster_retry_s, doubling, at most
    CLUSTER_RETRY_MAX_S) while the deployment stays stopping; then ``to`` with ``fields``."""
    delay = d.cluster_retry_s
    while True:
        await asyncio.sleep(delay)
        delay = min(delay * 2, CLUSTER_RETRY_MAX_S)
        async with d._lock(dep_id):
            dep = d._get(dep_id)
            if dep is None or dep.state != "stopping":
                return
            try:
                gone = await remove_ranks(d, dep_id)
            except Exception:
                log.exception("cleanup of deployment %s failed; retrying", dep_id)
                continue
            if gone:
                d._transition(dep_id, to, container_id=None, **fields)
                return


# ---------- stop / remove ----------
async def stop(d, dep: Deployment, reason: str, message: str | None) -> dict:
    """Deployer.stop of a cluster, under its lock (queued rows are handled by the caller)."""
    cleanup = d._cleanups.pop(dep.id, None)
    if cleanup is not None and not cleanup.done():
        cleanup.cancel()
    if dep.state in ("planned", "failed"):
        if not await remove_ranks(d, dep.id):
            raise DeployError("docker_error", "could not remove every rank container; try again", 502)
        return d._transition(dep.id, "stopped", message=f"cleaned up ({reason})", actor=reason, container_id=None)
    if dep.state != "stopping":
        d._transition(dep.id, "stopping", message=message or f"stop requested ({reason})", actor=reason)
    if await remove_ranks(d, dep.id):
        return d._transition(dep.id, "stopped", message="rank containers removed", container_id=None)
    msg = "some rank containers could not be removed yet; retrying (memory stays reserved until they are gone)"
    with d.db.session() as s:
        s.get(Deployment, dep.id).last_error = msg
        s.add(DeploymentEvent(deployment_id=dep.id, from_state="stopping", to_state="stopping", message=msg))
    spawn_cleanup(d, dep.id, "stopped", {"message": "rank containers removed", "last_error": None})
    view = d.view(dep.id)
    d.events.publish("deployment", view)
    return view


async def remove(d, dep: Deployment) -> None:
    """The containers part of Deployer.remove of a cluster (stopped or failed): every rank, by id and by name."""
    for r in ranks_of(dep):
        node = r["node"]
        if not d.node_configured(node):
            continue  # R11: nothing to reach there
        docker = d._docker_for(node)
        try:
            await asyncio.to_thread(docker.stop_and_remove, r.get("container_id"))
            await asyncio.to_thread(docker.remove_leftover, rank_name(dep.name, r["rank"]), dep.id)
        except NodeUnreachable as e:
            raise DeployError("node_unreachable", f"could not remove container: {e}", 503)
