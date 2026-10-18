"""LiteLLM integration: keep LiteLLM's DB-backed model list in sync with ready deployments (one entry per
deployment, under its model name = alias or served name) and proxy the admin API (models, virtual keys).

The URL and master key are resolved at call time from the runtime settings store, so changes made in the UI
apply without a restart. The master key is never logged, returned or put into error messages.
"""
from __future__ import annotations

import asyncio
import json
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
NOT_CONFIGURED_MESSAGE = "no LiteLLM master key is set; add one in the LiteLLM settings or the secret file"
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
    STATUS_TTL_S = 15.0       # cached connection status (GET /integrations/litellm, viewers included)
    DB_TTL_S = 15.0           # cached "does LiteLLM have a DB" answer used before key operations
    MODELS_TTL_S = 15.0       # cached model list for GET /integrations/litellm/models (viewers included)
    VERSION_TTL_S = 600.0
    VERSION_NEG_TTL_S = 30.0  # failed version lookups are retried after this long
    VERSION_FETCH_MAX_S = 30.0  # a version fetch may outlive the probe deadline (it finishes in the background)
    OPENAPI_MAX_BYTES = 4 * 1024 * 1024
    probe_deadline_s = 4.0    # one overall deadline for a status probe

    def __init__(self, http: httpx.AsyncClient, config: Callable[[], LiteLLMConfig],
                 clock: Callable[[], float] = time.monotonic):
        self.http, self._config, self.clock = http, config, clock
        self._versions: dict[str, tuple[float, str | None]] = {}
        self._status: dict[str, tuple[float, dict]] = {}
        self._db: dict[str, tuple[float, Any]] = {}
        self._keys: dict[str, tuple[float, list[dict]]] = {}
        self._models: dict[str, tuple[float, list[dict]]] = {}
        self._status_lock = asyncio.Lock()
        self._version_lock = asyncio.Lock()
        self._version_tasks: dict[str, asyncio.Future] = {}

    @classmethod
    def fixed(cls, base_url: str, key: str | None, http: httpx.AsyncClient) -> "LiteLLMClient":
        cfg = LiteLLMConfig(enabled=True, url=base_url.rstrip("/"), public_url="", master_key=key,
                            master_key_source="settings" if key else "none")
        return cls(http, lambda: cfg)

    def config(self) -> LiteLLMConfig:
        return self._config()

    def _fresh(self, cache: dict, url: str, ttl: float):
        hit = cache.get(url)
        return hit[1] if hit is not None and self.clock() - hit[0] < ttl else None

    # ---------- transport ----------
    async def _request(self, method: str, path: str, *, cfg: LiteLLMConfig | None = None, auth: bool = True,
                       timeout: float = 15, **kw) -> httpx.Response:
        cfg = cfg or self.config()
        if auth and not cfg.master_key:
            raise LiteLLMError(409, "litellm_not_configured", NOT_CONFIGURED_MESSAGE)
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

    async def _json(self, method: str, path: str, *, cfg: LiteLLMConfig | None = None, **kw) -> dict:
        """Authenticated call that must return a JSON object (anything else is a mapped LiteLLMError). ``cfg``
        defaults to the current settings; pass an explicit one only as resolved by the settings store (its key
        is bound to its URL)."""
        cfg = cfg or self.config()
        r = await self._request(method, path, cfg=cfg, **kw)
        if r.status_code >= 400:
            raise self._error(r, cfg)
        try:
            j = r.json()
        except ValueError:
            j = None
        if not isinstance(j, dict):
            raise LiteLLMError(502, "litellm_error", f"LiteLLM returned an unexpected response for {path}")
        return j

    # ---------- models ----------
    async def list_models(self, max_age: float = 0, *, cfg: LiteLLMConfig | None = None) -> list[dict]:
        """Every model LiteLLM serves. ``max_age`` > 0 allows a cached copy that young (DGXMM's own add/delete
        invalidate it); without it the list is always fetched fresh (sync, connection test)."""
        url = (cfg or self.config()).url
        if max_age and (hit := self._fresh(self._models, url, max_age)) is not None:
            return list(hit)
        data = (await self._json("GET", "/model/info", cfg=cfg, timeout=10)).get("data", [])
        if not isinstance(data, list):
            raise LiteLLMError(502, "litellm_error", "LiteLLM returned an unexpected model list")
        out = [m for m in data if isinstance(m, dict)]
        self._models[url] = (self.clock(), out)
        return list(out)

    async def list_dgxmm(self, *, cfg: LiteLLMConfig | None = None) -> list[dict]:
        return [m for m in await self.list_models(cfg=cfg) if (m.get("model_info") or {}).get("dgxmm_id")]

    async def add(self, model_name: str, api_base: str, served: str, dgxmm_id: str) -> None:
        body = {"model_name": model_name,
                "litellm_params": {"model": f"openai/{served}", "api_base": api_base, "api_key": "none"},
                "model_info": {"dgxmm_id": dgxmm_id}}
        try:
            await self._json("POST", "/model/new", json=body)
        finally:
            self.invalidate_models()

    async def delete(self, model_id: str, *, cfg: LiteLLMConfig | None = None) -> None:
        try:
            await self._json("POST", "/model/delete", cfg=cfg, json={"id": model_id})
        finally:
            self.invalidate_models()

    # ---------- health ----------
    async def _readiness(self, cfg: LiteLLMConfig, timeout: float) -> dict:
        r = await self._request("GET", "/health/readiness", cfg=cfg, auth=False, timeout=timeout)
        try:
            j = r.json()
        except ValueError:
            return {}
        if isinstance(j, dict) and "db" in j:
            self._db[cfg.url] = (self.clock(), j.get("db"))
        return j if isinstance(j, dict) else {}

    async def _openapi_version(self, cfg: LiteLLMConfig, timeout: float) -> str | None:
        """Read info.version from /openapi.json (streamed, capped at OPENAPI_MAX_BYTES)."""
        buf = bytearray()
        try:
            async with self.http.stream("GET", cfg.url + "/openapi.json", timeout=timeout) as r:
                if r.status_code != 200:
                    return None
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > self.OPENAPI_MAX_BYTES:
                        return None
            v = ((json.loads(bytes(buf)) or {}).get("info") or {}).get("version")
            return str(v) if v else None
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return None

    async def _version(self, cfg: LiteLLMConfig, readiness: dict, timeout: float) -> str | None:
        if readiness.get("litellm_version"):
            return str(readiness["litellm_version"])

        def cached():
            hit = self._versions.get(cfg.url)
            if hit is None:
                return False, None
            ttl = self.VERSION_TTL_S if hit[1] else self.VERSION_NEG_TTL_S
            return self.clock() - hit[0] < ttl, hit[1]

        ok, v = cached()
        if ok:
            return v
        async with self._version_lock:  # single flight: concurrent lookups share one fetch
            ok, v = cached()
            if ok:
                return v
            task = self._version_tasks.get(cfg.url)
            if task is None or task.done():
                task = asyncio.ensure_future(self._fetch_version(cfg, timeout))
                self._version_tasks[cfg.url] = task
        # Shielded: when the probe deadline cancels this wait the fetch carries on (bounded by
        # VERSION_FETCH_MAX_S) and fills the cache, so the next status read has the version.
        return await asyncio.shield(task)

    async def _fetch_version(self, cfg: LiteLLMConfig, timeout: float) -> str | None:
        # LiteLLM >= 1.7x no longer reports its version in /health/readiness; the OpenAPI document does
        try:
            v = await asyncio.wait_for(self._openapi_version(cfg, timeout), self.VERSION_FETCH_MAX_S)
        except asyncio.TimeoutError:
            v = None
        self._versions[cfg.url] = (self.clock(), v)
        return v

    def _with_version(self, url: str, out: dict) -> dict:
        """A status probed before the version was known gets it once a background fetch has found it."""
        if out.get("reachable") and not out.get("version"):
            hit = self._versions.get(url)
            if hit is not None and hit[1]:
                out["version"] = hit[1]
        return out

    async def _probe_into(self, out: dict, cfg: LiteLLMConfig, timeout: float) -> None:
        live = await self._request("GET", "/health/liveliness", cfg=cfg, auth=False, timeout=timeout)
        if live.status_code != 200 or "alive" not in live.text.lower():
            out["error"] = f"{cfg.url} is not a LiteLLM endpoint (/health/liveliness answered HTTP {live.status_code})"
            return
        out["reachable"] = True
        ready = await self._readiness(cfg, timeout)
        out["db"] = ready.get("db")
        out["version"] = await self._version(cfg, ready, timeout)

    async def probe(self, *, auth: bool = False, timeout: float = 5) -> dict:
        """Connection status. Never raises. With ``auth`` also checks the master key via /model/info."""
        cfg = self.config()
        out: dict[str, Any] = {"reachable": False, "version": None, "db": None, "error": None, "checked_at": _now()}
        if auth:
            out.update(auth_ok=None, models=None)
        try:
            await asyncio.wait_for(self._probe_into(out, cfg, timeout), self.probe_deadline_s)
        except LiteLLMError as e:
            out["error"] = e.message
        except asyncio.TimeoutError:
            if not out["reachable"]:
                out["error"] = f"timed out after {self.probe_deadline_s:g}s waiting for LiteLLM at {cfg.url}"
        if not out["reachable"]:
            return out
        if auth:
            if not cfg.master_key:
                out.update(auth_ok=False, error=cfg.master_key_hint or "no master key is set")
                return out
            try:
                out["models"] = len(await self.list_models())
                out["auth_ok"] = True
            except LiteLLMError as e:
                out.update(auth_ok=False if e.code == "litellm_auth_failed" else None, error=e.message)
        return out

    def cached_status(self) -> dict | None:
        """The status as of the last probe if still fresh (no network call), else None."""
        url = self.config().url
        hit = self._fresh(self._status, url, self.STATUS_TTL_S)
        return self._with_version(url, dict(hit)) if hit is not None else None

    async def status(self, *, force: bool = False) -> dict:
        """Unauthenticated connection status, cached STATUS_TTL_S per URL and fetched single-flight."""
        url = self.config().url
        if not force and (hit := self._fresh(self._status, url, self.STATUS_TTL_S)) is not None:
            return self._with_version(url, dict(hit))
        async with self._status_lock:
            if not force and (hit := self._fresh(self._status, url, self.STATUS_TTL_S)) is not None:
                return self._with_version(url, dict(hit))
            out = await self.probe()
            self._status[url] = (self.clock(), out)
            return dict(out)

    # ---------- virtual keys (need LiteLLM's DB) ----------
    async def require_db(self) -> None:
        cfg = self.config()
        if not cfg.master_key:
            raise LiteLLMError(409, "litellm_not_configured", NOT_CONFIGURED_MESSAGE)
        hit = self._db.get(cfg.url)
        if hit is not None and self.clock() - hit[0] < self.DB_TTL_S:
            db = hit[1]
        else:
            db = (await self._readiness(cfg, 10)).get("db")
        if db is not None and str(db).lower() != "connected":
            raise LiteLLMError(409, "litellm_no_db", NO_DB_MESSAGE)

    def invalidate_keys(self) -> None:
        self._keys.clear()

    def invalidate_models(self) -> None:
        self._models.clear()

    async def key_list(self, max_age: float = 0) -> list[dict]:
        """All keys (every page). ``max_age`` > 0 allows a cached copy that young."""
        url = self.config().url
        if max_age and (hit := self._fresh(self._keys, url, max_age)) is not None:
            return list(hit)
        await self.require_db()
        out: list[dict] = []
        page = 1
        while page <= 100:
            j = await self._json("GET", "/key/list", params={"return_full_object": "true", "size": 100, "page": page})
            keys = j.get("keys", [])
            out += [k for k in keys if isinstance(k, dict)] if isinstance(keys, list) else []
            try:
                pages = int(j.get("total_pages") or 1)
            except (TypeError, ValueError):
                pages = 1
            if page >= pages:
                break
            page += 1
        self._keys[url] = (self.clock(), out)
        return list(out)

    async def key_generate(self, body: dict) -> dict:
        await self.require_db()
        try:
            return await self._json("POST", "/key/generate", json=body)
        finally:
            self.invalidate_keys()

    async def key_update(self, token: str, changes: dict) -> dict:
        await self.require_db()
        try:
            return await self._json("POST", "/key/update", json={**changes, "key": token})
        finally:
            self.invalidate_keys()

    async def key_info(self, token: str) -> dict:
        await self.require_db()
        info = (await self._json("GET", "/key/info", params={"key": token})).get("info")
        return {**(info if isinstance(info, dict) else {}), "token": token}

    async def key_delete(self, token: str) -> None:
        await self.require_db()
        try:
            await self._json("POST", "/key/delete", json={"keys": [token]})
        finally:
            self.invalidate_keys()


_UNKNOWN, _OFF = object(), "off"  # LiteLLMSync._reported: before / after a failed sync; integration off


class LiteLLMSync:
    def __init__(self, client: LiteLLMClient | None, deployer, db: Database, events):
        self.client, self.deployer, self.db, self.events = client, deployer, db, events
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()
        self.last: dict[str, Any] = {"ok": None, "status": None, "registered": 0, "removed": 0, "error": None,
                                     "code": None, "at": None}
        self.conflicts: dict[int, str] = {}       # deployment id -> why it is not registered (name clash)
        self.registrations: dict[int, str] = {}   # deployment id -> model name, as of the last successful sync
        # called (no arguments) after a successful sync that changed what is_registered reports, e.g. routing advice
        self.listeners: list[Callable[[], Any]] = []
        self._reported: object = _UNKNOWN  # what is_registered reported after the previous sync
        if deployer is not None:
            deployer.litellm_error = self.error_for  # deployment views carry the reason (list, detail, SSE)

    def active(self) -> bool:
        return self.client is not None and self.client.config().active

    DEREGISTER_TIMEOUT_S = 15.0

    async def leave(self, cfg: LiteLLMConfig, apply: Callable[[], Any]) -> dict:
        """Remove every DGXMM-owned model (``model_info.dgxmm_id``) from the LiteLLM described by ``cfg`` — the
        settings in effect BEFORE a disable or URL change, so its key is the one bound to that URL and is sent
        nowhere else — then run ``apply`` (which persists the new settings). Both happen under the sync lock, so
        the sync loop cannot re-register in between. Best effort: ``apply`` runs even if LiteLLM is unreachable;
        the result says what was removed and what may remain."""
        async with self._lock:
            removed, listed, error = 0, None, None
            if self.client is not None and cfg.active:
                async def run():
                    nonlocal removed, listed
                    mine = await self.client.list_dgxmm(cfg=cfg)
                    listed = len(mine)
                    for m in mine:
                        await self.client.delete(m["model_info"]["id"], cfg=cfg)
                        removed += 1
                try:
                    await asyncio.wait_for(run(), self.DEREGISTER_TIMEOUT_S)
                except LiteLLMError as e:
                    error = e.message
                except asyncio.TimeoutError:
                    error = f"timed out after {self.DEREGISTER_TIMEOUT_S:g}s talking to LiteLLM at {cfg.url}"
                except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
                    error = f"unexpected response from LiteLLM at {cfg.url} ({type(e).__name__})"
            warning = None
            if error is not None:
                left = listed - removed if listed is not None else (
                    len(self.registrations) if self.last.get("ok") and self.registrations else None)
                what = f"{left} DGXMM model{'' if left == 1 else 's'}" if left else "DGXMM models"
                warning = (f"{what} may remain registered in {cfg.url}; remove them there, or re-enable the "
                           "integration with that URL to clean up")
                log.warning("LiteLLM: could not remove DGXMM models from %s: %s", cfg.url, error)
            else:
                self.registrations = {}
            apply()
            return {"deregistered": removed, "deregister_error": error, "warning": warning}

    def desired(self) -> dict[tuple[str, str], tuple[str, str]]:
        """(deployment id, model name) -> (api_base, served name). Exactly one entry per ready deployment, and
        never two backends under one model name: the Deployer prevents that, but if it happens anyway the
        oldest deployment keeps the name and the others are recorded in ``self.conflicts``."""
        want: dict[tuple[str, str], tuple[str, str]] = {}
        owner: dict[str, dict] = {}
        conflicts: dict[int, str] = {}
        for v in sorted(self.deployer.list_views(), key=lambda v: v["id"]):
            if v["state"] != "ready" or not v["served_name"] or not v["port"] or v.get("system"):
                continue  # system deployments (the decision model) are never LiteLLM models
            name = v["alias"] or v["served_name"]
            if name in owner:
                conflicts[v["id"]] = (f"model name {name} is already registered for deployment {owner[name]['name']}; "
                                      "give this deployment a different model name")
                continue
            owner[name] = v
            api_base = self.deployer.base_url(v["name"], v["port"]) + "/v1"
            want[(str(v["id"]), name)] = (api_base, v["served_name"])
        self.conflicts = conflicts
        if conflicts:
            log.warning("LiteLLM: %d deployment(s) not registered because of a model name clash", len(conflicts))
        return want

    def _set_status(self, ready_status: str) -> None:
        changed = []
        with self.db.session() as s:
            for d in s.scalars(select(Deployment)):
                status = ready_status if d.state == "ready" else ("disabled" if ready_status == "disabled" else "none")
                if d.system:
                    status = "none"
                if d.state == "ready" and ready_status == "synced" and d.id in self.conflicts:
                    status = "error"
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
            self._report(_OFF)
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
            self._reported = _UNKNOWN  # the next good sync notifies
            return False
        self.registrations = {int(k[0]): k[1] for k in want}
        self._set_status("synced")
        self.last = {"ok": True, "status": "synced", "registered": added, "removed": removed, "error": None,
                     "code": None, "at": _now()}
        self._report(dict(self.registrations))
        return True

    def _report(self, reported) -> None:
        """After a successful sync: run the listeners if what is_registered reports (``OFF`` or the registrations)
        changed since the previous sync, including the first good sync after a failed one. A failing listener is
        logged and does not stop the others."""
        if reported == self._reported:
            return
        self._reported = reported
        for fn in list(self.listeners):
            try:
                fn()
            except Exception:
                log.exception("LiteLLM sync listener failed")

    def error_for(self, dep_id: int, state: str) -> dict | None:
        """``{code, message}`` when a ready deployment is not registered as it should be, as of the last sync: a
        model-name clash, or the sync itself failing (messages are already scrubbed of keys). None otherwise;
        non-ready deployments are not registered anyway, so a failing sync is not their problem."""
        if state != "ready":
            return None
        if dep_id in self.conflicts:
            return {"code": "model_name_conflict", "message": self.conflicts[dep_id]}
        if self.last["ok"] is False:
            return {"code": self.last["code"] or "litellm_error", "message": self.last["error"]}
        return None

    def is_registered(self, dep_id: int, model_name: str) -> bool | None:
        """As of the last sync (no LiteLLM call); None (unknown) unless that sync succeeded."""
        if not self.last["ok"] or self.last["status"] != "synced":
            return None
        return self.registrations.get(dep_id) == model_name

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
