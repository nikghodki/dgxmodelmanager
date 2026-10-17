from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import httpx

log = logging.getLogger(__name__)


class MetricsScraper:
    def __init__(self, settings, deployer, http: httpx.AsyncClient, store, events, budget,
                 clock: Callable[[], float] = time.time):
        self.settings, self.deployer, self.http, self.store = settings, deployer, http, store
        self.events, self.budget, self.clock = events, budget, clock
        self.retention: list[Callable[[], object]] = []  # run after each rollup, in the same worker thread

    async def scrape_once(self) -> dict[int, dict]:
        now = self.clock()
        out: dict[int, dict] = {}
        for v in self.deployer.list_views():
            if v["state"] != "ready" or not v["port"]:
                continue
            engine = self.deployer.engines.get(v["engine"])
            if engine is None or not getattr(engine, "scrape_metrics", True):
                continue  # e.g. the decision model: no /metrics
            try:
                r = await self.http.get(self.deployer.base_url(v["name"], v["port"]) + "/metrics", timeout=5)
                r.raise_for_status()
                out[v["id"]] = self.store.add(v["id"], now, engine.parse_metrics(r.text))
            except Exception as e:
                log.debug("metrics scrape failed for %s: %s", v["name"], e)
        b = self.budget.current()
        if b is not None:
            sample = self.budget.last_sample
            self.store.add_host(now, b, sample.gpu if sample else {})
        if out:
            self.events.publish("metrics", out)
        return out

    def _rollup(self) -> None:
        self.store.rollup()
        for fn in self.retention:
            try:
                fn()
            except Exception:
                log.exception("retention pass failed")

    async def run(self) -> None:
        last_rollup = self.clock()
        while True:
            try:
                await self.scrape_once()
                if self.clock() - last_rollup >= 60:
                    await asyncio.to_thread(self._rollup)
                    last_rollup = self.clock()
            except Exception:
                log.exception("metrics loop failed")
            await asyncio.sleep(self.settings.scrape_interval_s)
