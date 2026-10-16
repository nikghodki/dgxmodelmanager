"""Keep LiteLLM's DB-backed model list in sync with ready deployments (spec §6)."""
from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy import select

from dgxmm.db import Database, Deployment

log = logging.getLogger(__name__)


class LiteLLMClient:
    def __init__(self, base_url: str, key: str, http: httpx.AsyncClient):
        self.base, self.key, self.http = base_url.rstrip("/"), key, http

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    async def list_dgxmm(self) -> list[dict]:
        r = await self.http.get(f"{self.base}/model/info", headers=self._h(), timeout=10)
        r.raise_for_status()
        return [m for m in r.json().get("data", []) if (m.get("model_info") or {}).get("dgxmm_id")]

    async def add(self, model_name: str, api_base: str, served: str, dgxmm_id: str) -> None:
        body = {"model_name": model_name,
                "litellm_params": {"model": f"openai/{served}", "api_base": api_base, "api_key": "none"},
                "model_info": {"dgxmm_id": dgxmm_id}}
        r = await self.http.post(f"{self.base}/model/new", headers=self._h(), json=body, timeout=15)
        r.raise_for_status()

    async def delete(self, model_id: str) -> None:
        r = await self.http.post(f"{self.base}/model/delete", headers=self._h(), json={"id": model_id}, timeout=15)
        r.raise_for_status()


class LiteLLMSync:
    def __init__(self, client: LiteLLMClient | None, deployer, db: Database, events):
        self.client, self.deployer, self.db, self.events = client, deployer, db, events
        self._wake = asyncio.Event()

    def desired(self) -> dict[tuple[str, str], tuple[str, str]]:
        want: dict[tuple[str, str], tuple[str, str]] = {}
        for v in self.deployer.list_views():
            if v["state"] != "ready" or not v["served_name"] or not v["port"]:
                continue
            api_base = self.deployer.base_url(v["name"], v["port"]) + "/v1"
            for name in filter(None, [v["served_name"], v["alias"]]):
                want[(str(v["id"]), name)] = (api_base, v["served_name"])
        return want

    def _set_status(self, ready_status: str) -> None:
        changed = []
        with self.db.session() as s:
            for d in s.scalars(select(Deployment)):
                status = ready_status if d.state == "ready" else ("disabled" if ready_status == "disabled" else "none")
                if d.litellm_status != status:
                    d.litellm_status = status
                    changed.append(d.id)
        for dep_id in changed:
            self.events.publish("deployment", self.deployer.view(dep_id))

    async def sync(self) -> bool:
        if self.client is None:
            self._set_status("disabled")
            return True
        try:
            have: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
            for m in await self.client.list_dgxmm():
                key = (str(m["model_info"]["dgxmm_id"]), m["model_name"])
                have.setdefault(key, []).append((m["model_info"]["id"],
                                                (m.get("litellm_params") or {}).get("api_base", ""),
                                                (m.get("litellm_params") or {}).get("model", "")))
            want = self.desired()
            for key, entries in have.items():
                target = want.get(key)
                keep = None
                if target:
                    keep = next((e for e in entries if e[1] == target[0] and e[2] == f"openai/{target[1]}"), None)
                for model_id, _, _ in entries:
                    if keep is None or model_id != keep[0]:
                        await self.client.delete(model_id)
                if target and keep is None:
                    have[key] = []
            for key, (api_base, served) in want.items():
                if not have.get(key):
                    await self.client.add(key[1], api_base, served, key[0])
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
            log.warning("LiteLLM sync failed: %s", e)
            self._set_status("pending")
            return False
        self._set_status("synced")
        return True

    def trigger(self) -> None:
        self._wake.set()

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        if "ready" in (frm, to):
            self.trigger()

    async def run(self) -> None:
        failure_delay = 5.0
        while True:
            try:
                ok = await self.sync()
            except Exception as e:
                log.exception("LiteLLM sync unexpected error: %s", e)
                ok = False
            if ok:
                delay, failure_delay = 60.0, 5.0
            else:
                delay, failure_delay = failure_delay, min(failure_delay * 2, 600.0)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
