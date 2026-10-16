"""LiteLLM integration: keep LiteLLM's DB-backed model list in sync with ready deployments (one entry per
deployment, under its model name = alias or served name) and proxy the admin API (models, virtual keys).

The URL and master key are resolved at call time from the runtime settings store, so changes made in the UI
apply without a restart. The master key is never logged, returned or put into error messages.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from sqlalchemy import select

from dgxmm.db import Database, Deployment
from dgxmm.settings_store import LiteLLMConfig

log = logging.getLogger(__name__)

NO_DB_MESSAGE = ("LiteLLM is running without a database, so virtual keys are not available. Configure DATABASE_URL "
                 "for LiteLLM (with store_model_in_db: true) and restart it.")
_NO_DB_RE = re.compile(r"no (connected )?(db|database)|(db|database) (is )?not connected|not connected to (a |the )?(db|database)",
                       re.I)
_SK_RE = re.compile(r"sk-[A-Za-z0-9_.\-]{2,}")
_HEX_RE = re.compile(r"\b[0-9a-fA-F]{40,}\b")
VERSION_TTL_S = 600.0
UNRESTRICTED = "all-proxy-models"


class LiteLLMError(Exception):
    """A failed LiteLLM call, already mapped to an API status/code with a scrubbed, user-facing message."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def scrub(text: str, *secrets: str | None) -> str:
    """Remove anything key-like from text that came from LiteLLM or httpx before it reaches users or logs."""
    for s in secrets:
        if s:
            text = text.replace(s, "***")
    text = _SK_RE.sub("sk-***", text)
    return _HEX_RE.sub("<hash>", text)[:300]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_message(r: httpx.Response) -> str:
    try:
        j = r.json()
    except ValueError:
        return r.text[:300]
    if isinstance(j, dict):
        err = j.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if j.get("detail"):
            d = j["detail"]
            return str(d.get("error", d) if isinstance(d, dict) else d)
    return str(j)[:300]


class LiteLLMClient:
    def __init__(self, http: httpx.AsyncClient, config: Callable[[], LiteLLMConfig]):
        self.http, self._config = http, config
        self._versions: dict[str, tuple[float, str | None]] = {}

    @classmethod
    def fixed(cls, base_url: str, key: str | None, http: httpx.AsyncClient) -> "LiteLLMClient":
        cfg = LiteLLMConfig(enabled=True, url=base_url.rstrip("/"), public_url="", master_key=key,
                            master_key_source="settings" if key else "none")
        return cls(http, lambda: cfg)

    def config(self) -> LiteLLMConfig:
        return self._config()

    # ---------- transport ----------
    async def _request(self, method: str, path: str, *, cfg: LiteLLMConfig | None = None, auth: bool = True,
                       timeout: float = 15, **kw) -> httpx.Response:
        cfg = cfg or self.config()
        if auth and not cfg.master_key:
            raise LiteLLMError(409, "litellm_not_configured",
                               "no LiteLLM master key is set; add one in the LiteLLM settings or the secret file")
        headers = {"Authorization": f"Bearer {cfg.master_key}"} if auth else {}
        try:
            return await self.http.request(method, cfg.url + path, headers=headers, timeout=timeout, **kw)
        except httpx.HTTPError as e:
            detail = scrub(str(e), cfg.master_key)
            raise LiteLLMError(502, "litellm_unreachable",
                               f"could not reach LiteLLM at {cfg.url}: {type(e).__name__}" + (f" ({detail})" if detail else "")
                               ) from None

    def _error(self, r: httpx.Response, cfg: LiteLLMConfig) -> LiteLLMError:
        msg = scrub(_error_message(r), cfg.master_key)
        if _NO_DB_RE.search(msg):
            return LiteLLMError(409, "litellm_no_db", NO_DB_MESSAGE)
        if r.status_code in (401, 403):
            return LiteLLMError(502, "litellm_auth_failed",
                                "LiteLLM rejected the master key; check it in the LiteLLM settings")
        if r.status_code == 404:
            return LiteLLMError(404, "litellm_not_found", msg or "not found in LiteLLM")
        if 400 <= r.status_code < 500:
            return LiteLLMError(400, "litellm_rejected", f"LiteLLM rejected the request: {msg}")
        return LiteLLMError(502, "litellm_error", f"LiteLLM returned HTTP {r.status_code}: {msg}")

    async def _json(self, method: str, path: str, **kw) -> Any:
        cfg = self.config()
        r = await self._request(method, path, cfg=cfg, **kw)
        if r.status_code >= 400:
            raise self._error(r, cfg)
        try:
            return r.json()
        except ValueError:
            raise LiteLLMError(502, "litellm_error", f"LiteLLM returned a non-JSON response for {path}") from None

    # ---------- models ----------
    async def list_models(self) -> list[dict]:
        data = (await self._json("GET", "/model/info", timeout=10)).get("data", [])
        return [m for m in data if isinstance(m, dict)]

    async def list_dgxmm(self) -> list[dict]:
        return [m for m in await self.list_models() if (m.get("model_info") or {}).get("dgxmm_id")]

    async def add(self, model_name: str, api_base: str, served: str, dgxmm_id: str) -> None:
        body = {"model_name": model_name,
                "litellm_params": {"model": f"openai/{served}", "api_base": api_base, "api_key": "none"},
                "model_info": {"dgxmm_id": dgxmm_id}}
        await self._json("POST", "/model/new", json=body)

    async def delete(self, model_id: str) -> None:
        await self._json("POST", "/model/delete", json={"id": model_id})

    # ---------- health ----------
    async def _readiness(self, cfg: LiteLLMConfig, timeout: float) -> dict:
        r = await self._request("GET", "/health/readiness", cfg=cfg, auth=False, timeout=timeout)
        try:
            j = r.json()
        except ValueError:
            return {}
        return j if isinstance(j, dict) else {}

    async def _version(self, cfg: LiteLLMConfig, readiness: dict, timeout: float) -> str | None:
        if readiness.get("litellm_version"):
            return str(readiness["litellm_version"])
        hit = self._versions.get(cfg.url)
        if hit and time.monotonic() - hit[0] < VERSION_TTL_S:
            return hit[1]
        version = None
        try:  # LiteLLM >= 1.7x no longer reports its version in /health/readiness; the OpenAPI document does
            r = await self._request("GET", "/openapi.json", cfg=cfg, auth=False, timeout=timeout)
            if r.status_code == 200:
                version = ((r.json() or {}).get("info") or {}).get("version")
        except (LiteLLMError, ValueError, AttributeError):
            version = None
        self._versions[cfg.url] = (time.monotonic(), version)
        return version

    async def probe(self, *, auth: bool = False, timeout: float = 5) -> dict:
        """Connection status. Never raises. With ``auth`` also checks the master key via /model/info."""
        cfg = self.config()
        out: dict[str, Any] = {"reachable": False, "version": None, "db": None, "error": None, "checked_at": _now()}
        if auth:
            out.update(auth_ok=None, models=None)
        try:
            live = await self._request("GET", "/health/liveliness", cfg=cfg, auth=False, timeout=timeout)
            if live.status_code >= 400:
                live = await self._request("GET", "/health/readiness", cfg=cfg, auth=False, timeout=timeout)
            out["reachable"] = live.status_code < 500
            if not out["reachable"]:
                out["error"] = f"LiteLLM at {cfg.url} answered HTTP {live.status_code}"
                return out
            ready = await self._readiness(cfg, timeout)
            out["db"] = ready.get("db")
            out["version"] = await self._version(cfg, ready, timeout)
        except LiteLLMError as e:
            out["error"] = e.message
            return out
        if auth:
            if not cfg.master_key:
                out.update(auth_ok=False, error="no master key is set")
                return out
            try:
                out["models"] = len(await self.list_models())
                out["auth_ok"] = True
            except LiteLLMError as e:
                out.update(auth_ok=False if e.code == "litellm_auth_failed" else None, error=e.message)
        return out

    # ---------- virtual keys (need LiteLLM's DB) ----------
    async def require_db(self) -> None:
        cfg = self.config()
        if not cfg.master_key:
            raise LiteLLMError(409, "litellm_not_configured",
                               "no LiteLLM master key is set; add one in the LiteLLM settings or the secret file")
        db = (await self._readiness(cfg, 10)).get("db")
        if db is not None and str(db).lower() != "connected":
            raise LiteLLMError(409, "litellm_no_db", NO_DB_MESSAGE)

    async def key_list(self) -> list[dict]:
        await self.require_db()
        out: list[dict] = []
        page = 1
        while page <= 100:
            j = await self._json("GET", "/key/list", params={"return_full_object": "true", "size": 100, "page": page})
            out += [k for k in j.get("keys", []) if isinstance(k, dict)]
            if page >= int(j.get("total_pages") or 1):
                break
            page += 1
        return out

    async def key_generate(self, body: dict) -> dict:
        await self.require_db()
        return await self._json("POST", "/key/generate", json=body)

    async def key_update(self, token: str, changes: dict) -> dict:
        await self.require_db()
        return await self._json("POST", "/key/update", json={**changes, "key": token})

    async def key_info(self, token: str) -> dict:
        await self.require_db()
        j = await self._json("GET", "/key/info", params={"key": token})
        return {**(j.get("info") or {}), "token": token}

    async def key_delete(self, token: str) -> None:
        await self.require_db()
        await self._json("POST", "/key/delete", json={"keys": [token]})


class LiteLLMSync:
    def __init__(self, client: LiteLLMClient | None, deployer, db: Database, events):
        self.client, self.deployer, self.db, self.events = client, deployer, db, events
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self.last: dict[str, Any] = {"ok": None, "status": None, "registered": 0, "removed": 0, "error": None,
                                     "code": None, "at": None}

    def active(self) -> bool:
        return self.client is not None and self.client.config().active

    def desired(self) -> dict[tuple[str, str], tuple[str, str]]:
        """(deployment id, model name) -> (api_base, served name). Exactly one entry per ready deployment."""
        want: dict[tuple[str, str], tuple[str, str]] = {}
        for v in self.deployer.list_views():
            if v["state"] != "ready" or not v["served_name"] or not v["port"]:
                continue
            api_base = self.deployer.base_url(v["name"], v["port"]) + "/v1"
            want[(str(v["id"]), v["alias"] or v["served_name"])] = (api_base, v["served_name"])
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
        async with self._lock:
            return await self._sync()

    async def _sync(self) -> bool:
        if not self.active():
            self._set_status("disabled")
            self.last = {"ok": True, "status": "disabled", "registered": 0, "removed": 0, "error": None,
                         "code": None, "at": _now()}
            return True
        added = removed = 0
        try:
            have: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
            for m in await self.client.list_dgxmm():
                key = (str(m["model_info"]["dgxmm_id"]), m["model_name"])
                have.setdefault(key, []).append((m["model_info"]["id"],
                                                (m.get("litellm_params") or {}).get("api_base", ""),
                                                (m.get("litellm_params") or {}).get("model", "")))
            want = self.desired()
            # Anything marked as ours that is not wanted (stopped deployment, old alias, or the served-name entry
            # older DGXMM versions registered next to the alias) is deleted; duplicates are collapsed.
            for key, entries in have.items():
                target = want.get(key)
                keep = None
                if target:
                    keep = next((e for e in entries if e[1] == target[0] and e[2] == f"openai/{target[1]}"), None)
                for model_id, _, _ in entries:
                    if keep is None or model_id != keep[0]:
                        await self.client.delete(model_id)
                        removed += 1
                if target and keep is None:
                    have[key] = []
            for key, (api_base, served) in want.items():
                if not have.get(key):
                    await self.client.add(key[1], api_base, served, key[0])
                    added += 1
        except (httpx.HTTPError, LiteLLMError, KeyError, ValueError, TypeError) as e:
            if isinstance(e, LiteLLMError):
                code, msg = e.code, e.message
            else:
                code, msg = "litellm_error", f"unexpected response from LiteLLM ({type(e).__name__})"
            log.warning("LiteLLM sync failed: %s", msg)
            self._set_status("pending")
            self.last = {"ok": False, "status": "pending", "registered": added, "removed": removed, "error": msg,
                         "code": code, "at": _now()}
            return False
        self._set_status("synced")
        self.last = {"ok": True, "status": "synced", "registered": added, "removed": removed, "error": None,
                     "code": None, "at": _now()}
        return True

    async def is_registered(self, dep_id: int, model_name: str) -> bool:
        return any(str(m["model_info"]["dgxmm_id"]) == str(dep_id) and m.get("model_name") == model_name
                   for m in await self.client.list_dgxmm())

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
                log.exception("LiteLLM sync unexpected error: %s", type(e).__name__)
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
