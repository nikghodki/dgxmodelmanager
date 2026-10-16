import asyncio
import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from dgxmm.advisor.benchmark import BenchmarkBusy
from dgxmm.api.deployments import deploy_error
from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.db import Benchmark
from dgxmm.deployer.service import DeployError

router = APIRouter(prefix="/api", tags=["metrics"])
log = logging.getLogger(__name__)
WINDOWS = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}


class BenchIn(BaseModel):
    levels: list[int] = Field([1, 4, 8, 16], min_length=1, max_length=6)
    input_len: int = Field(1024, ge=8, le=32768)
    output_len: int = Field(256, ge=1, le=4096)
    duration_s: float = Field(60, ge=1, le=600)
    max_requests: int = Field(32, ge=1, le=512)

    @field_validator("levels")
    @classmethod
    def _levels(cls, v: list[int]) -> list[int]:
        if any(not 1 <= c <= 128 for c in v):
            raise ValueError("concurrency levels must be between 1 and 128")
        return v


def _series(ctx, key, window):
    if window == "live":
        return ctx.metrics.series(key)
    if window not in WINDOWS:
        raise api_error(422, "invalid_window", "window must be live, 1h, 24h, 7d or 30d")
    return ctx.metrics.history(key, WINDOWS[window])


@router.get("/deployments/{dep_id}/metrics")
def dep_metrics(dep_id: int, window: str = Query("live"), p: Principal = Depends(current_principal),
                ctx: AppContext = Depends(get_ctx)):
    return _series(ctx, dep_id, window)


@router.get("/host/metrics")
def host_metrics(window: str = Query("live"), p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return _series(ctx, "host", window)


@router.get("/deployments/{dep_id}/advice")
def advice(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        return ctx.advisor.advise(dep_id)
    except DeployError as e:
        raise deploy_error(e)


@router.post("/deployments/{dep_id}/benchmark", status_code=202)
async def benchmark(dep_id: int, body: BenchIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        v = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    if v["state"] != "ready":
        raise api_error(409, "not_ready", "the deployment must be ready to benchmark")
    if ctx.benchmarks.busy() is not None:
        raise api_error(409, "benchmark_running", "another benchmark is running")

    async def _go():
        try:
            await ctx.benchmarks.run(deployment_id=dep_id, engine=v["engine"], model_ref=v["model_ref"], config=v["config"],
                                     base_url=ctx.deployer.base_url(v["name"], v["port"]), served_name=v["served_name"],
                                     levels=tuple(body.levels), input_len=body.input_len, output_len=body.output_len,
                                     duration_s=body.duration_s, max_requests=body.max_requests)
        except BenchmarkBusy:
            pass
        except Exception:
            log.exception("benchmark failed")

    asyncio.get_running_loop().create_task(_go())
    audit(ctx.db, p.user.id, "benchmark", v["name"], body.model_dump())
    return {"started": True}


@router.get("/deployments/{dep_id}/benchmarks")
def benchmarks(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        v = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    with ctx.db.session() as s:
        rows = s.scalars(select(Benchmark).where(Benchmark.model_ref == v["model_ref"]).order_by(Benchmark.id.desc()).limit(200))
        return [{"id": b.id, "ts": b.ts.isoformat(), "config_hash": b.config_hash, "config": b.config, "engine": b.engine,
                 "concurrency": b.concurrency, "requests": b.requests, "ttft_p50": b.ttft_p50, "ttft_p95": b.ttft_p95,
                 "itl_p50": b.itl_p50, "tps_total": b.tps_total, "tps_per_req": b.tps_per_req,
                 "current": b.config == v["config"]} for b in rows]
