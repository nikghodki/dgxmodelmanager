"""Deployment lifecycle. The only caller of DockerController.run/stop_and_remove."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from dgxmm.config import Settings
from dgxmm.db import Database, Deployment, DeploymentEvent
from dgxmm.deployer.states import ACTIVE_STATES, assert_transition
from dgxmm.engines.base import LaunchContext, container_name
from dgxmm.events import EventBroker
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate, vllm_gpu_util
from dgxmm.planner.fit import FitResult, check_fit
from dgxmm.planner.plans import ActiveDep, PlanAction, base_model_name, build_plans

log = logging.getLogger(__name__)
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,40}$"
ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$"


class DeployRequest(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    engine: str = Field(pattern=r"^[a-z]{1,16}$")
    model_ref: str = Field(max_length=512)
    config: dict[str, Any] = {}
    priority: int = Field(0, ge=0, le=100)
    alias: str | None = Field(None, pattern=ALIAS_PATTERN)


class DeployError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class InsufficientMemory(DeployError):
    def __init__(self, fit: FitResult):
        gib = 1024**3
        super().__init__("insufficient_memory",
                         f"needs {fit.required / gib:.1f} GiB but only {fit.free_for_new / gib:.1f} GiB is free", 409)
        self.fit = fit


@dataclass
class Prepared:
    engine: Any
    artifact: ModelArtifact
    cfg: BaseModel
    draft: ModelArtifact | None
    estimate: MemoryEstimate


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


StateHook = Callable[[int, str, str], Awaitable[None]]


class Deployer:
    def __init__(self, *, settings: Settings, db: Database, docker, inventory, engines: dict, http: httpx.AsyncClient,
                 events: EventBroker, budget=None, clock: Callable[[], float] = time.monotonic):
        self.settings, self.db, self.docker, self.inventory = settings, db, docker, inventory
        self.engines, self.http, self.events, self.budget, self.clock = engines, http, events, budget, clock
        self.mem_lock = asyncio.Lock()
        self._locks: dict[int, asyncio.Lock] = {}
        self._monitors: dict[int, asyncio.Task] = {}
        self.state_hooks: list[StateHook] = []
        self.calibration = None  # set in Task 32

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

    def base_url(self, name: str, port: int) -> str:
        host = self.settings.engine_base_host or container_name(name)
        return f"http://{host}:{port}"

    def _ref_allowed(self, ref: str) -> bool:
        if re.fullmatch(REPO_RE, ref) and ".." not in ref:
            return True
        try:
            p = Path(ref).resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        roots = [Path(m.local).resolve() for m in self.settings.gguf_roots] + [Path(self.settings.hf_cache.local).resolve()]
        return str(p) == ref and p.suffix.lower() == ".gguf" and any(r in p.parents for r in roots)

    def _transition(self, dep_id: int, to: str, message: str = "", log_tail: str | None = None, **fields) -> dict:
        with self.db.session() as s:
            dep = s.get(Deployment, dep_id)
            frm = dep.state
            assert_transition(frm, to)
            dep.state = to
            for k, v in fields.items():
                setattr(dep, k, v)
            s.add(DeploymentEvent(deployment_id=dep_id, from_state=frm, to_state=to, message=message, log_tail=log_tail))
        view = self.view(dep_id)
        self.events.publish("deployment", view)
        for hook in self.state_hooks:
            asyncio.get_running_loop().create_task(self._run_hook(hook, dep_id, frm, to))
        return view

    @staticmethod
    async def _run_hook(hook: StateHook, dep_id: int, frm: str, to: str) -> None:
        try:
            await hook(dep_id, frm, to)
        except Exception:
            log.exception("state hook failed for deployment %s", dep_id)

    def _allocate_port(self, exclude_dep: int | None = None) -> int:
        with self.db.session() as s:
            used = {d.port for d in s.scalars(select(Deployment)) if d.port and d.state in ACTIVE_STATES | {"planned"}
                    and d.id != exclude_dep}
        try:
            used |= {p for c in self.docker.list_all() for p in c.host_ports if not c.managed}
        except Exception:
            log.warning("could not list containers for port allocation", exc_info=True)
        lo, hi = self.settings.port_range
        for port in range(lo, hi + 1):
            if port not in used:
                return port
        raise DeployError("no_ports", "port pool exhausted", 503)

    # ---------- planning ----------
    def prepare(self, engine_name: str, model_ref: str, raw_cfg: dict) -> Prepared:
        engine = self.engines.get(engine_name)
        if engine is None:
            raise DeployError("unknown_engine", f"engine {engine_name!r} is not available", 422)
        if not self._ref_allowed(model_ref):
            raise DeployError("invalid_model_ref", "model reference is not a repo id or an allowed GGUF path", 422)
        try:
            cfg = engine.validate_config(raw_cfg)
        except (ValidationError, ValueError) as e:
            raise DeployError("invalid_config", str(e), 422)
        artifact = self.inventory.get(model_ref)
        if artifact is None:
            raise DeployError("model_not_found", f"{model_ref} is not downloaded", 404)
        if not artifact.complete:
            raise DeployError("model_incomplete", "; ".join(artifact.problems) or "incomplete download", 409)
        if not engine.supports(artifact):
            raise DeployError("engine_unsupported", f"{engine_name} cannot serve {artifact.format} model {model_ref}", 422)
        draft = None
        if dref := engine.draft_ref(cfg):
            draft = self.inventory.get(dref)
            if draft is None or not draft.complete:
                raise DeployError("draft_not_found", f"draft model {dref} is not downloaded", 404)
        try:
            est = engine.estimate(artifact, cfg, draft, self.settings)
        except ValueError as e:
            raise DeployError("cannot_estimate", str(e), 422)
        if self.calibration is not None:
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

    async def preview(self, engine: str, model_ref: str, raw_cfg: dict, replacing: int | None = None) -> dict:
        prep = self.prepare(engine, model_ref, raw_cfg)
        budget = await self.budget.refresh()
        fit = check_fit(prep.estimate, budget, extra_free=self._replacing_credit(replacing))
        out = {"estimate": prep.estimate.as_dict(), "fit": fit.as_dict(), "budget": budget.as_dict(),
               "served_name": prep.engine.served_name(prep.cfg, model_ref),
               "config": prep.cfg.model_dump(mode="json")}
        if engine == "vllm":
            out["gpu_memory_utilization"] = vllm_gpu_util(prep.estimate.total, budget.total)
        out["plans"] = [] if fit.fits else self.plans_for(prep, fit, replacing)
        return out

    # ---------- lifecycle ----------
    async def deploy(self, req: DeployRequest, user_id: int | None) -> dict:
        prep = self.prepare(req.engine, req.model_ref, req.config)
        with self.db.session() as s:
            if s.scalar(select(Deployment).where(Deployment.name == req.name)):
                raise DeployError("name_in_use", f"a deployment named {req.name} already exists", 409)
        async with self.mem_lock:
            budget = await self.budget.refresh()
            fit = check_fit(prep.estimate, budget)
            if not fit.fits:
                raise InsufficientMemory(fit)
            port = self._allocate_port()
            with self.db.session() as s:
                dep = Deployment(name=req.name, engine=req.engine, model_ref=req.model_ref,
                                 config=prep.cfg.model_dump(mode="json"), state="planned", port=port,
                                 priority=req.priority, alias=req.alias, reserved_bytes=prep.estimate.total,
                                 estimate=prep.estimate.as_dict(), created_by=user_id)
                s.add(dep)
                s.flush()
                dep_id = dep.id
            await self._launch(dep_id, req.name, port, prep, budget.total)
        self._spawn_monitor(dep_id)
        return self.view(dep_id)

    async def _launch(self, dep_id: int, name: str, port: int, prep: Prepared, mem_total: int) -> None:
        launch = LaunchContext(dep_id, name, port, prep.estimate.total, mem_total)
        spec = prep.engine.container_spec(prep.artifact, prep.cfg, prep.draft, launch, self.settings)
        try:
            await asyncio.to_thread(self.docker.remove_by_name, spec.name)
            cid = await asyncio.to_thread(self.docker.run, spec)
        except Exception as e:
            self._transition(dep_id, "failed", message=f"docker error: {e}", last_error=f"docker error: {e}")
            raise DeployError("docker_error", f"could not start container: {e}", 502)
        self._transition(dep_id, "starting", message=f"container {cid[:12]} created", container_id=cid,
                         last_error=None, exit_code=None)

    def _spawn_monitor(self, dep_id: int) -> None:
        old = self._monitors.get(dep_id)
        if old and not old.done():
            old.cancel()
        task = asyncio.get_running_loop().create_task(self._monitor(dep_id))
        task.add_done_callback(lambda t: self._monitor_done(dep_id, t))
        self._monitors[dep_id] = task

    def _monitor_done(self, dep_id: int, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("monitor task for deployment %s died with uncaught exception", dep_id)

    def resume_monitor(self, dep_id: int) -> None:
        self._spawn_monitor(dep_id)

    async def _monitor(self, dep_id: int) -> None:
        start = self.clock()
        consecutive_errors = 0
        while True:
            await asyncio.sleep(self.settings.monitor_poll_s)
            try:
                dep = self._get(dep_id)
                if dep is None or dep.state not in ("starting", "loading"):
                    return
                info = await asyncio.to_thread(self.docker.get, dep.container_id)
                if info is None:
                    self._transition(dep_id, "failed", message="container disappeared", last_error="container disappeared")
                    return
                if info.status in ("exited", "dead"):
                    await self._fail_exited(dep, info.exit_code)
                    return
                if dep.state == "starting" and info.status == "running":
                    self._transition(dep_id, "loading", message="container running; loading model")
                mem = await asyncio.to_thread(self.budget.probe.meminfo)
                if mem.available < self.settings.margin_bytes:
                    await self._abort(dep, "aborted: available memory fell below the safety margin during startup")
                    return
                if self.clock() - start > self.settings.startup_timeout_s:
                    await self._abort(dep, f"startup timed out after {self.settings.startup_timeout_s}s")
                    return
                if info.status == "running" and await self._healthy(dep):
                    self._transition(dep_id, "ready", message="health check and test completion passed")
                    return
                consecutive_errors = 0  # Reset on success
            except Exception as e:
                consecutive_errors += 1
                log.exception("monitor error for deployment %s (consecutive: %d)", dep_id, consecutive_errors)
                if consecutive_errors >= 5:
                    try:
                        self._transition(dep_id, "failed", message="monitor error", last_error=f"monitor error: {e}")
                    except Exception:
                        log.exception("failed to transition deployment %s to failed after monitor errors", dep_id)
                    return

    async def _healthy(self, dep: Deployment) -> bool:
        engine = self.engines[dep.engine]
        cfg = engine.validate_config(dep.config)
        try:
            return await engine.health(self.http, self.base_url(dep.name, dep.port), engine.served_name(cfg, dep.model_ref))
        except (httpx.HTTPError, OSError):
            return False

    async def _fail_exited(self, dep: Deployment, exit_code: int | None) -> None:
        try:
            logs = await asyncio.to_thread(self.docker.logs, dep.container_id, 200)
        except Exception:
            log.exception("failed to capture logs for deployment %s", dep.id)
            logs = ""
        hint = " (likely out of memory)" if exit_code == 137 else ""
        msg = f"container exited with code {exit_code}{hint}"
        try:
            await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
        except Exception:
            log.exception("failed to cleanup container for deployment %s", dep.id)
        self._transition(dep.id, "failed", message=msg, log_tail=logs, last_error=msg, exit_code=exit_code,
                         container_id=None)
        if exit_code == 137 and self.calibration is not None:
            self.calibration.record_oom(dep.engine, dep.model_ref, config_hash(dep.config), dep.reserved_bytes)

    async def _abort(self, dep: Deployment, reason: str) -> None:
        try:
            logs = await asyncio.to_thread(self.docker.logs, dep.container_id, 200)
        except Exception:
            log.exception("failed to capture logs for deployment %s", dep.id)
            logs = ""
        try:
            await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
        except Exception:
            log.exception("failed to cleanup container for deployment %s", dep.id)
        self._transition(dep.id, "failed", message=reason, log_tail=logs, last_error=reason, container_id=None)

    async def wait_ready(self, dep_id: int, timeout: float) -> str:
        deadline = self.clock() + timeout
        while True:
            dep = self._require(dep_id)
            if dep.state in ("ready", "failed", "stopped", "sleeping"):
                return dep.state
            if self.clock() > deadline:
                return dep.state
            await asyncio.sleep(0.02)

    async def stop(self, dep_id: int, user_id: int | None = None, reason: str = "user") -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state == "stopped":
                return self.view(dep_id)
            mon = self._monitors.pop(dep_id, None)
            if mon and not mon.done():
                mon.cancel()
            if dep.state in ("planned", "failed"):
                await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
                return self._transition(dep_id, "stopped", message=f"cleaned up ({reason})", container_id=None)
            if dep.state != "stopping":
                self._transition(dep_id, "stopping", message=f"stop requested ({reason})")
            try:
                await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
            except Exception as e:
                self._transition(dep_id, "failed", message=f"stop failed: {e}", last_error=f"stop failed: {e}")
                raise DeployError("docker_error", f"could not stop container: {e}", 502)
            return self._transition(dep_id, "stopped", message="container removed", container_id=None)

    async def restart(self, dep_id: int, user_id: int | None = None, config_override: dict | None = None,
                      model_ref: str | None = None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", f"deployment is {dep.state}; stop it first", 409)
            ref = model_ref or dep.model_ref
            prep = self.prepare(dep.engine, ref, config_override if config_override is not None else dep.config)
            async with self.mem_lock:
                budget = await self.budget.refresh()
                fit = check_fit(prep.estimate, budget)
                if not fit.fits:
                    raise InsufficientMemory(fit)
                # Check all used ports (active/planned DB rows + unmanaged containers)
                with self.db.session() as s:
                    db_used = {d.port for d in s.scalars(select(Deployment))
                               if d.port and d.state in ACTIVE_STATES | {"planned"} and d.id != dep_id}
                try:
                    containers = await asyncio.to_thread(self.docker.list_all)
                    unmanaged_used = {p for c in containers for p in c.host_ports if not c.managed}
                except Exception:
                    log.warning("could not list containers for restart port check", exc_info=True)
                    unmanaged_used = set()
                used_ports = db_used | unmanaged_used
                port = dep.port if dep.port and dep.port not in used_ports else self._allocate_port(exclude_dep=dep_id)
                with self.db.session() as s:
                    row = s.get(Deployment, dep_id)
                    row.model_ref, row.config, row.port = ref, prep.cfg.model_dump(mode="json"), port
                    row.reserved_bytes, row.estimate = prep.estimate.total, prep.estimate.as_dict()
                await self._launch(dep_id, dep.name, port, prep, budget.total)
        self._spawn_monitor(dep_id)
        return self.view(dep_id)

    async def remove(self, dep_id: int) -> None:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", "stop the deployment before removing it", 409)
            await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
            with self.db.session() as s:
                s.delete(s.get(Deployment, dep_id))
        self.events.publish("deployment_removed", {"id": dep_id})

    async def aclose(self) -> None:
        for t in self._monitors.values():
            t.cancel()
        self._monitors.clear()

    # ---------- views ----------
    def usage_rows(self) -> list[tuple[int, str, str, int, str | None]]:
        with self.db.session() as s:
            return [(d.id, d.name, d.state, d.reserved_bytes, d.container_id)
                    for d in s.scalars(select(Deployment)) if d.state in ACTIVE_STATES]

    def _actual(self, dep_id: int) -> int | None:
        b = self.budget.current() if self.budget else None
        if not b:
            return None
        return next((u.actual for u in b.deployments if u.deployment_id == dep_id), None)

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
            "endpoint": f"http://{container_name(d.name)}:{d.port}/v1" if d.port else None,
            "host_endpoint": f"http://{self.settings.publish_bind}:{d.port}/v1" if d.port and self.settings.publish_bind else None,
            "sleep_capable": bool(engine and engine.supports_sleep and d.config.get("enable_sleep_mode")),
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        }

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
                                  "message": e.message, "log_tail": e.log_tail} for e in rows]
        return out

    def list_views(self) -> list[dict]:
        with self.db.session() as s:
            return [self._view(d) for d in s.scalars(select(Deployment).order_by(Deployment.name))]

    def active_deps(self, exclude: int | None = None) -> list[ActiveDep]:
        out = []
        for v in self.list_views():
            if v["id"] == exclude or v["state"] not in ("ready", "sleeping", "unknown", "loading", "starting"):
                continue
            est = v["estimate"] or {}
            sleep_frees = max(v["reserved_bytes"] - est.get("weights", 0) - est.get("draft_weights", 0)
                              - est.get("runtime", 0), 0)
            out.append(ActiveDep(v["id"], v["name"], v["state"], v["priority"], v["reserved_bytes"],
                                 v["actual_bytes"] or 0, v["sleep_capable"] and v["state"] == "ready", sleep_frees))
        return out

    def plans_for(self, prep: Prepared, fit: FitResult, replacing: int | None = None) -> list[dict]:
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
                            active=self.active_deps(exclude=replacing), variants=variants, estimate_ref=est_ref)
        return [p.as_dict() for p in plans]

    def _stored_estimate(self, dep: Deployment) -> MemoryEstimate:
        if dep.estimate:
            return MemoryEstimate.from_dict(dep.estimate)
        return MemoryEstimate(dep.reserved_bytes, 0, 0, 0, 0, 0, 0, "low")

    async def sleep(self, dep_id: int, user_id: int | None = None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state != "ready":
                raise DeployError("not_ready", f"deployment is {dep.state}", 409)
            engine = self.engines[dep.engine]
            cfg = engine.validate_config(dep.config)
            if not (engine.supports_sleep and engine.sleep_enabled(cfg)):
                raise DeployError("sleep_unsupported", "this deployment was not started with sleep mode", 409)
            try:
                await engine.sleep(self.http, self.base_url(dep.name, dep.port))
            except httpx.HTTPError as e:
                raise DeployError("engine_error", f"sleep failed: {e}", 502)
            return self._transition(dep_id, "sleeping", message="sleeping (level 1: KV freed, weights offloaded)")

    async def wake(self, dep_id: int, user_id: int | None = None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state != "sleeping":
                raise DeployError("not_sleeping", f"deployment is {dep.state}", 409)
            engine = self.engines[dep.engine]
            async with self.mem_lock:
                budget = await self.budget.refresh()
                actual = next((u.actual for u in budget.deployments if u.deployment_id == dep_id), 0)
                need = max(dep.reserved_bytes - actual, 0)
                if need > budget.free_for_new:
                    raise InsufficientMemory(FitResult(False, need, budget.free_for_new, budget.free_for_new - need,
                                                       self._stored_estimate(dep), budget))
                try:
                    await engine.wake(self.http, self.base_url(dep.name, dep.port))
                except httpx.HTTPError as e:
                    raise DeployError("engine_error", f"wake failed: {e}", 502)
                return self._transition(dep_id, "ready", message="woken up")

    async def apply_actions(self, actions, user_id: int | None = None) -> list[tuple[str, int]]:
        applied: list[tuple[str, int]] = []
        try:
            for raw in actions:
                a = raw if isinstance(raw, PlanAction) else PlanAction.model_validate(raw)
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

    def _set_alias(self, dep_id: int, alias: str | None) -> None:
        with self.db.session() as s:
            s.get(Deployment, dep_id).alias = alias
        self.events.publish("deployment", self.view(dep_id))

    def _can_sleep(self, dep: Deployment) -> bool:
        engine = self.engines.get(dep.engine)
        return bool(dep.state == "ready" and engine and engine.supports_sleep and dep.config.get("enable_sleep_mode"))

    async def switch(self, old_id: int, req: DeployRequest, actions, user_id: int | None) -> dict:
        old = self._require(old_id)
        acts = [a if isinstance(a, PlanAction) else PlanAction.model_validate(a) for a in actions]

        # Check memory fit before touching anything; auto-escalate sleep→stop if needed
        prep = self.prepare(req.engine, req.model_ref, req.config)
        budget = await self.budget.refresh()

        # Determine what we'll actually free
        if not any(a.deployment_id == old_id for a in acts) and old.state in ("ready", "sleeping", "unknown"):
            if old.state == "sleeping":
                freed = 0  # already sleeping
            elif self._can_sleep(old):
                # Compute sleep_frees
                est = old.estimate or {}
                sleep_frees = max(old.reserved_bytes - est.get("weights", 0) - est.get("draft_weights", 0)
                                  - est.get("runtime", 0), 0)
                # Check if sleep is enough
                fit_with_sleep = check_fit(prep.estimate, budget, extra_free=sleep_frees)
                if fit_with_sleep.fits:
                    acts.insert(0, PlanAction(type="sleep", deployment_id=old_id))
                    freed = sleep_frees
                else:
                    # Sleep not enough; escalate to stop
                    stop_frees = max(old.reserved_bytes, next((u.actual for u in budget.deployments
                                                               if u.deployment_id == old_id), 0))
                    acts.insert(0, PlanAction(type="stop", deployment_id=old_id))
                    freed = stop_frees
            else:
                # Can't sleep; must stop
                stop_frees = max(old.reserved_bytes, next((u.actual for u in budget.deployments
                                                           if u.deployment_id == old_id), 0))
                acts.insert(0, PlanAction(type="stop", deployment_id=old_id))
                freed = stop_frees
        else:
            # Explicit actions provided; compute total freed
            freed = sum(
                (max(d.reserved_bytes, next((u.actual for u in budget.deployments if u.deployment_id == a.deployment_id), 0))
                 if a.type == "stop" else 0)
                for a in acts for d in [self._get(a.deployment_id)] if d
            )

        # Final fit check
        fit = check_fit(prep.estimate, budget, extra_free=freed)
        if not fit.fits:
            raise InsufficientMemory(fit)

        alias = old.alias
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
            if new_id is not None:
                try:
                    await self.stop(new_id, user_id, reason="switch rollback")
                except Exception as e:
                    log.exception("rollback: could not stop %s", new_id)
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
        old_cfg, old_ref = dict(dep.config), dep.model_ref
        old_reserved, old_estimate, old_port = dep.reserved_bytes, dict(dep.estimate), dep.port
        was_active = dep.state in ("ready", "sleeping", "unknown", "starting", "loading")
        prep = self.prepare(dep.engine, model_ref or old_ref, new_config)
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
