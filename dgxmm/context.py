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
UNTRUSTED_PROXIES_REFRESH_S = 30.0  # how often the decider network's subnets are re-read (dgxmm/proxy_headers.py)


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
    rate_limiters: dict = field(default_factory=dict)    # dgxmm.api.ratelimit buckets
    budget_listeners: list = field(default_factory=list)  # async fn(budget) after each refresh
    settings_store: Any = None  # SettingsStore; created from db + settings when not given
    decider: Any = None  # dgxmm.decider.service.Decider; always constructed (disabled by default)
    untrusted_proxy_networks: list = field(default_factory=list)  # never proxies (dgxmm/proxy_headers.py)
    decider_image: Any = None  # dgxmm.decider.image.DeciderImage (status + on-demand build)
    diagnosis: Any = None  # DiagnosisService (failure diagnosis)
    workload: Any = None   # WorkloadService (workload label for advice ordering)
    kinds: Any = None      # KindService (model-derived artifact kinds)
    insights: Any = None   # InsightService (dgxmm/decider/insights.py); always constructed
    insight_engine: Any = None  # InsightEngine (dgxmm/decider/insight_engine.py); runs from ctx.background
    health: Any = None     # HealthService (dgxmm/decider/health.py; job health_check)
    readings: Any = None   # ReadingService (dgxmm/decider/reading.py; job reading_labels)
    routing: Any = None    # RoutingService (dgxmm/decider/routing.py; job routing_advice)
    schedules: Any = None  # ScheduleService (dgxmm/decider/schedule.py): sleep/wake schedules, profiles, proposals

    def __post_init__(self) -> None:
        if self.settings_store is None:
            from dgxmm.settings_store import SettingsStore
            self.settings_store = SettingsStore(self.db, self.settings)
        if self.decider is None:
            from dgxmm.decider.service import Decider
            self.decider = Decider(db=self.db, settings=self.settings, store=self.settings_store,
                                   deployer=lambda: self.deployer, http=lambda: self.http)
        # removing the decider's deployment (from any page) turns the decider off and clears its deployment id
        hooks, on_removed = getattr(self.deployer, "removal_hooks", None), getattr(self.decider, "on_deployment_removed",
                                                                                 None)
        if hooks is not None and on_removed is not None and on_removed not in hooks:
            hooks.append(on_removed)
        if self.insights is None:
            from dgxmm.decider.insights import InsightService
            self.insights = InsightService(self.db, self.events)
        if hooks is not None and self.insights.on_deployment_removed not in hooks:
            hooks.append(self.insights.on_deployment_removed)  # a deployment's insights are deleted with it
        if self.insight_engine is None:
            from dgxmm.decider.insight_engine import InsightEngine
            self.insight_engine = InsightEngine(self)
        if self.decider_image is None:
            from dgxmm.decider.image import DeciderImage
            self.decider_image = DeciderImage(self.settings, lambda: self.docker, self.events)


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
    from dgxmm.docker_ctl import CONTAINER_CACHE_TTL_S, DockerController
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
    probe = HostProbe(settings, NvmlGpu(), containers=docker.lookup)
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
    if not ctx.settings_store.litellm().master_key:
        log.warning("no LiteLLM master key (settings or %s); LiteLLM registration disabled until one is set",
                    settings.litellm_key_file)
    litellm = LiteLLMSync(LiteLLMClient(http, ctx.settings_store.litellm), deployer, db, events, insights=ctx.insights)
    deployer.state_hooks.append(litellm.on_state)
    ctx.litellm = litellm
    ctx.background.append(litellm.run)

    from dgxmm.watchdog import Watchdog
    ctx.watchdog = Watchdog(settings, deployer, events, db)
    ctx.budget_listeners.append(ctx.watchdog.check)
    ctx.budget_listeners.append(liveness_listener(deployer))


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

    from dgxmm.inventory.weights import WeightsDeleter
    ctx.weights = WeightsDeleter(settings, db, inventory, docker=docker, deployer=deployer, downloads=ctx.downloads)

    async def _recover_downloads():
        ctx.downloads.recover()

    from dgxmm.metrics.scraper import MetricsScraper
    from dgxmm.metrics.store import MetricsStore
    store = MetricsStore(db)
    scraper = MetricsScraper(settings, deployer, http, store, events, budget)
    from dgxmm.decider.retention import prune as prune_decider_tables
    scraper.retention.append(lambda: prune_decider_tables(db, settings))
    scraper.retention.append(ctx.insights.prune)  # closed insights 30 days, expiries, snooze ends
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

    from dgxmm.decider.features import wire_features
    wire_features(ctx)

    async def _containers_loop():
        # keeps the read-only container-name cache warm for the budget loop, which only reads the cache
        while True:
            await asyncio.to_thread(docker.directory.refresh)
            await asyncio.sleep(CONTAINER_CACHE_TTL_S)

    async def _decider_network():
        # After DGXMM's own container is recreated (upgrade), re-join the decider's internal network so a running
        # decider stays reachable. Never creates the network (the first decider start does).
        try:
            if await asyncio.to_thread(docker.attach_self, settings.decider_network):
                log.info("attached to the decider network %s", settings.decider_network)
        except Exception:
            log.warning("could not join the decider network %s", settings.decider_network, exc_info=True)

    async def refresh_untrusted_proxies():
        from dgxmm.proxy_headers import parse_networks
        try:
            ctx.untrusted_proxy_networks = parse_networks(
                await asyncio.to_thread(docker.network_subnets, settings.decider_network))
        except Exception:
            log.debug("could not read the decider network's subnets", exc_info=True)

    async def _untrusted_proxies_loop():
        # the decider's network is never a proxy (dgxmm/proxy_headers.py); it is created at the first decider start
        while True:
            await asyncio.sleep(UNTRUSTED_PROXIES_REFRESH_S)
            await refresh_untrusted_proxies()

    async def _decider_upgrade_check():
        from dgxmm.decider import upgrade
        try:
            up = await asyncio.to_thread(upgrade.check, ctx)
        except Exception:
            return
        if up["needed"]:
            log.warning("the running decider predates this DGXMM build (%s): rebuild its image and restart it "
                        "(Decision model page, or POST /api/decider/rotate-key)", "; ".join(up["reasons"]))

    ctx.startup += [_bootstrap, _recover_downloads, _decider_network, refresh_untrusted_proxies, _inventory, _reconcile,
                    _decider_upgrade_check]
    from dgxmm.decider import rotation

    async def _decider_rotation():
        await rotation.watch(ctx)  # admin CLI rotate-decider-key requests

    ctx.background += [_budget_loop, _inventory_loop, _containers_loop, _decider_rotation, _untrusted_proxies_loop]
    return ctx
