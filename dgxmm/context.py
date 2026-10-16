"""Shared service container stored on app.state.ctx, plus production wiring."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.config import Settings
from dgxmm.db import Database
from dgxmm.events import EventBroker

log = logging.getLogger("dgxmm")


@dataclass
class AppContext:
    settings: Settings
    db: Database
    events: EventBroker
    login_limiter: SlidingWindowLimiter = field(default_factory=lambda: SlidingWindowLimiter(5, 60))
    inventory: Any = None
    budget: Any = None
    deployer: Any = None
    docker: Any = None
    engines: dict = field(default_factory=dict)
    probe: Any = None
    http: Any = None
    watchdog: Any = None
    litellm: Any = None
    catalog: Any = None
    downloads: Any = None
    metrics: Any = None
    archs: Any = None
    hub: Any = None
    advisor: Any = None
    benchmarks: Any = None
    weights: Any = None  # WeightsDeleter (delete / clean up downloaded weights)
    startup: list = field(default_factory=list)          # awaited once, in order, before serving
    background: list = field(default_factory=list)       # long-running coroutines
    budget_listeners: list = field(default_factory=list)  # async fn(budget) after each refresh


async def budget_loop(ctx: AppContext) -> None:
    from dgxmm.api.host import host_payload
    while True:
        try:
            b = await ctx.budget.refresh()
            ctx.events.publish("budget", host_payload(ctx, b))
            for listener in ctx.budget_listeners:
                await listener(b)
        except Exception:
            log.exception("budget refresh failed")
        await asyncio.sleep(ctx.settings.budget_interval_s)


def liveness_listener(deployer):
    """Budget listener: re-check ready/sleeping/unknown containers on every budget refresh."""
    async def _check(_budget) -> None:
        await deployer.check_liveness()
    return _check


def build_context(settings: Settings) -> AppContext:
    import httpx

    from dgxmm.auth.bootstrap import bootstrap_admin
    from dgxmm.deployer.reconcile import reconcile
    from dgxmm.deployer.service import Deployer
    from dgxmm.docker_ctl import DockerController
    from dgxmm.engines.registry import build_engines
    from dgxmm.inventory.service import InventoryService
    from dgxmm.memory.budget import BudgetService
    from dgxmm.memory.hostmem import HostProbe, NvmlGpu

    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    db = Database(f"sqlite:///{settings.db_path}")
    events = EventBroker()
    http = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5))
    inventory = InventoryService(settings)
    docker = DockerController.from_env()
    engines = build_engines(settings)
    probe = HostProbe(settings, NvmlGpu())
    deployer = Deployer(settings=settings, db=db, docker=docker, inventory=inventory, engines=engines, http=http,
                        events=events)
    budget = BudgetService(settings, probe, deployer.usage_rows)
    deployer.budget = budget
    ctx = AppContext(settings=settings, db=db, events=events, inventory=inventory, budget=budget, deployer=deployer,
                     docker=docker, engines=engines, probe=probe, http=http)

    async def _bootstrap():
        bootstrap_admin(db)

    async def _inventory():
        await asyncio.to_thread(inventory.refresh)

    async def _reconcile():
        await budget.refresh()
        await reconcile(deployer)

    async def _budget_loop():
        await budget_loop(ctx)

    async def _inventory_loop():
        await asyncio.sleep(settings.inventory_interval_s)
        await inventory.run()

    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    key = settings.read_secret(settings.litellm_key_file)
    if not key:
        log.warning("no LiteLLM master key at %s; LiteLLM registration disabled", settings.litellm_key_file)
    litellm = LiteLLMSync(LiteLLMClient(settings.litellm_url, key, http) if key else None, deployer, db, events)
    deployer.state_hooks.append(litellm.on_state)
    ctx.litellm = litellm
    ctx.background.append(litellm.run)

    from dgxmm.watchdog import Watchdog
    ctx.watchdog = Watchdog(settings, deployer, events, db)
    ctx.budget_listeners.append(ctx.watchdog.check)
    ctx.budget_listeners.append(liveness_listener(deployer))

    from dgxmm.inventory.weights import WeightsDeleter
    ctx.weights = WeightsDeleter(settings, db, inventory, docker=docker)

    from dgxmm.catalog.support import VllmArchRegistry
    archs = VllmArchRegistry(settings, docker)
    engines["vllm"].arch_source = archs.get
    ctx.archs = archs

    async def _archs():
        await archs.ensure()
    ctx.background.append(_archs)

    from dgxmm.catalog.hub import HfHubClient
    from dgxmm.catalog.llmfit import Llmfit
    from dgxmm.catalog.service import CatalogService
    hf_token = settings.read_secret(settings.hf_token_file)
    hub = HfHubClient(token=hf_token)
    ctx.catalog = CatalogService(settings, hub, Llmfit(settings.llmfit_bin), inventory, budget, deployer, archs)
    ctx.hub = hub

    from dgxmm.downloads import DownloadService
    ctx.downloads = DownloadService(settings, db, inventory, hub, events, token=hf_token)

    async def _recover_downloads():
        ctx.downloads.recover()

    from dgxmm.metrics.scraper import MetricsScraper
    from dgxmm.metrics.store import MetricsStore
    store = MetricsStore(db)
    scraper = MetricsScraper(settings, deployer, http, store, events, budget)
    ctx.metrics = store
    ctx.background.append(scraper.run)

    from dgxmm.memory.calibration import Calibration, CalibrationRecorder
    calibration = Calibration(db)
    deployer.calibration = calibration
    deployer.state_hooks.append(CalibrationRecorder(deployer, calibration).on_state)

    from dgxmm.advisor.benchmark import BenchmarkRunner
    from dgxmm.advisor.rules import AdvisorService
    ctx.advisor = AdvisorService(deployer, inventory, store, db)
    ctx.benchmarks = BenchmarkRunner(db, events)

    ctx.startup += [_bootstrap, _recover_downloads, _inventory, _reconcile]
    ctx.background += [_budget_loop, _inventory_loop]
    return ctx
