"""/api/integrations/litellm — runtime LiteLLM settings, status, sync, models and virtual keys.

Secrets: the master key is write-only (only ``master_key_set``/``master_key_source`` are returned); a full virtual
key is returned exactly once, in the create response. Neither is audited or logged.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, Depends, Path, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.litellm import UNRESTRICTED, LiteLLMClient, LiteLLMError
from dgxmm.settings_store import (LITELLM_ENABLED, LITELLM_MASTER_KEY, LITELLM_MASTER_KEY_URL, LITELLM_PUBLIC_URL,
                                  LITELLM_URL, norm_url)

router = APIRouter(prefix="/api/integrations/litellm", tags=["integrations"])

TOKEN_RE = r"^[0-9a-f]{64}$"  # LiteLLM's hashed key id; full keys (sk-...) are never accepted in URLs
KEY_ALIAS_RE = r"^[A-Za-z0-9._:@/ -]{1,128}$"
MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9._:/@-]{1,128}$")
DURATION_RE = r"^[1-9][0-9]{0,5}(s|m|h|d|w|mo)$"
KEY_FIELDS = ("token", "key_name", "key_alias", "models", "spend", "max_budget", "budget_duration", "rpm_limit",
              "tpm_limit", "expires", "created_at", "blocked", "metadata")


# ---------- helpers shared with dgxmm.api.deployments ----------
def client(ctx: AppContext) -> LiteLLMClient:
    if ctx.litellm is None or ctx.litellm.client is None:
        raise api_error(503, "litellm_unavailable", "the LiteLLM integration is not initialised")
    return ctx.litellm.client


def require_enabled(ctx: AppContext) -> LiteLLMClient:
    c = client(ctx)
    cfg = ctx.settings_store.litellm()
    if not cfg.master_key:
        raise api_error(409, "litellm_not_configured",
                        "no LiteLLM master key is set; add one in the LiteLLM settings or the secret file")
    if not cfg.enabled:
        raise api_error(409, "litellm_disabled", "the LiteLLM integration is disabled in the LiteLLM settings")
    return c


def litellm_error(e: LiteLLMError):
    return api_error(e.status, e.code, e.message)


def key_out(k: dict) -> dict:
    """Public view of a LiteLLM key record: never the full key (``key`` is dropped; ``token`` is the hash)."""
    out = {f: k.get(f) for f in KEY_FIELDS}
    out["models"] = list(out["models"] or [])
    out["metadata"] = out["metadata"] if isinstance(out["metadata"], dict) else {}
    out["created_by_dgxmm"] = out["metadata"].get("created_by") == "dgxmm"
    return out


def key_allows(k: dict, model_name: str) -> bool:
    """Best-effort "can this key call model_name": unrestricted ([] or all-proxy-models) or listed by exact name.
    Not modelled: team-level model lists/key inheritance from a team, access groups, wildcard patterns
    (e.g. "openai/*"), per-model budgets, and expiry/blocked state. Treat the result as informational."""
    models = k.get("models") or []
    return not models or UNRESTRICTED in models or model_name in models


def _http_url(field: str, v: str) -> str:
    v = v.strip()
    parts = urlsplit(v)
    if parts.scheme not in ("http", "https") or not parts.hostname or any(c.isspace() for c in v) \
            or parts.query or parts.fragment or parts.username or parts.password:
        raise api_error(422, "invalid_settings", f"{field} must be an http:// or https:// URL", field=field)
    try:
        parts.port  # noqa: B018  (raises on an invalid port)
    except ValueError:
        raise api_error(422, "invalid_settings", f"{field} has an invalid port", field=field) from None
    return v.rstrip("/")


def settings_out(ctx: AppContext, status: dict) -> dict:
    cfg = ctx.settings_store.litellm()
    return {"enabled": cfg.enabled, "url": cfg.url, "public_url": cfg.public_url, "endpoint": cfg.endpoint,
            "master_key_set": bool(cfg.master_key), "master_key_source": cfg.master_key_source,
            "master_key_hint": cfg.master_key_hint, "default_url": ctx.settings.litellm_url.rstrip("/"), "status": status}


async def _status(ctx: AppContext, force: bool = False) -> dict:
    """Unauthenticated status, cached ~15 s (so viewers polling GET never cause more than one probe)."""
    if ctx.litellm is None or ctx.litellm.client is None:
        return {"reachable": False, "version": None, "db": None, "error": "LiteLLM integration not initialised",
                "checked_at": None}
    return await ctx.litellm.client.status(force=force)


# ---------- settings ----------
class SettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: StrictBool | None = None
    url: str | None = None
    public_url: str | None = None
    master_key: str | None = None  # "" clears (falls back to the secret file); never echoed


@router.get("")
async def get_settings(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return settings_out(ctx, await _status(ctx))


@router.put("")
async def put_settings(raw: Any = Body(...), p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    # Validated by hand so a validation error can never echo the submitted master key back.
    if not isinstance(raw, dict):
        raise api_error(422, "invalid_settings", "body must be a JSON object", field="body")
    try:
        body = SettingsIn.model_validate(raw)
    except ValidationError as e:
        err = e.errors()[0]
        field = str(err["loc"][0]) if err.get("loc") else "body"
        raise api_error(422, "invalid_settings", f"invalid value for {field}", field=field) from None
    fields = body.model_fields_set
    store, changes = ctx.settings_store, {}
    if "url" in fields and body.url is not None:
        url = _http_url("url", body.url) if body.url.strip() else None
        if url is not None and norm_url(url) == norm_url(ctx.settings.litellm_url):
            url = None  # the config default is stored as "unset"
        if url != store.get(LITELLM_URL):
            store.set(LITELLM_URL, url)
            changes["url"] = url or "(default)"
    if "public_url" in fields and body.public_url is not None:
        pub = norm_url(_http_url("public_url", body.public_url)) if body.public_url.strip() else ""
        # The config default (litellm_public_url) is stored as "unset"; clearing it while a config default
        # exists is stored as "" so it stays cleared instead of reverting to the default.
        new = None if pub == norm_url(ctx.settings.litellm_public_url or "") else pub
        if new != store.get(LITELLM_PUBLIC_URL):
            store.set(LITELLM_PUBLIC_URL, new)
            changes["public_url"] = pub
    if "master_key" in fields and body.master_key is not None:
        mk = body.master_key
        if mk and (len(mk) > 512 or any(c.isspace() or not c.isprintable() for c in mk)):
            raise api_error(422, "invalid_settings", "invalid value for master_key", field="master_key")
        # Bound to the URL in effect after this request: it is never used with any other URL (see
        # SettingsStore.litellm), so repointing the URL later cannot disclose it.
        url_now = store.litellm().url
        if mk and (mk != store.get(LITELLM_MASTER_KEY) or url_now != store.get(LITELLM_MASTER_KEY_URL)):
            store.set_master_key(mk, url_now)
            changes["master_key"] = "set"
        elif not mk and store.get(LITELLM_MASTER_KEY) is not None:
            store.set_master_key(None)
            changes["master_key"] = "cleared"
    if "enabled" in fields and body.enabled is not None and store.get(LITELLM_ENABLED) != str(body.enabled).lower():
        store.set_bool(LITELLM_ENABLED, body.enabled)
        changes["enabled"] = body.enabled
    if changes:
        audit(ctx.db, p.user.id, "litellm_settings_update", "litellm", changes)
        if ctx.litellm is not None:
            ctx.litellm.trigger()
    return settings_out(ctx, await _status(ctx, force=bool(changes)))


@router.post("/test")
async def test_connection(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await client(ctx).probe(auth=True, timeout=5)


@router.post("/sync")
async def sync_now(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    client(ctx)
    await ctx.litellm.sync()
    last = ctx.litellm.last
    if not last["ok"]:
        raise api_error(502 if last["code"] != "litellm_no_db" else 409, last["code"] or "litellm_error", last["error"],
                        registered=last["registered"], removed=last["removed"])
    audit(ctx.db, p.user.id, "litellm_sync", "litellm", {"registered": last["registered"], "removed": last["removed"]})
    return {k: last[k] for k in ("ok", "status", "registered", "removed", "error")}


@router.get("/models")
async def list_models(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    c = require_enabled(ctx)
    try:
        models = await c.list_models()
    except LiteLLMError as e:
        raise litellm_error(e)
    deps = {str(v["id"]): v for v in ctx.deployer.list_views()} if ctx.deployer is not None else {}
    is_admin = p.user.role == "admin"
    out = []
    for m in models:
        info, params = m.get("model_info") or {}, m.get("litellm_params") or {}
        dgx = info.get("dgxmm_id")
        dep = deps.get(str(dgx)) if dgx else None
        out.append({"id": info.get("id"), "model_name": m.get("model_name"), "model": params.get("model"),
                    "api_base": params.get("api_base") if is_admin else None,
                    "managed_by_dgxmm": bool(dgx),
                    "deployment": {"id": dep["id"], "name": dep["name"], "state": dep["state"]} if dep else None,
                    "db_model": info.get("db_model")})
    return out


# ---------- virtual keys ----------
class _Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_budget: float | None = Field(None, ge=0)
    budget_duration: str | None = Field(None, pattern=DURATION_RE)
    rpm_limit: int | None = Field(None, ge=0)
    tpm_limit: int | None = Field(None, ge=0)

    def check_models(self, models: list[str] | None) -> None:
        for m in models or []:
            if not MODEL_NAME_RE.fullmatch(m):
                raise api_error(422, "invalid_model", f"invalid model name: {m[:64]!r}")


class KeyCreate(_Limits):
    key_alias: str = Field(pattern=KEY_ALIAS_RE)
    models: list[str] = Field(default_factory=list, max_length=200)  # [] = all models
    duration: str | None = Field(None, pattern=DURATION_RE)  # key expiry, e.g. "30d"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyPatch(_Limits):
    key_alias: str | None = Field(None, pattern=KEY_ALIAS_RE)
    models: list[str] | None = Field(None, max_length=200)
    blocked: StrictBool | None = None


def _token(token: str) -> str:
    if not re.fullmatch(TOKEN_RE, token):
        raise api_error(422, "invalid_token", "expected the key's token hash (64 hex characters)")
    return token


@router.get("/keys")
async def list_keys(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    c = require_enabled(ctx)
    try:
        return [key_out(k) for k in await c.key_list()]
    except LiteLLMError as e:
        raise litellm_error(e)


@router.post("/keys", status_code=201)
async def create_key(body: KeyCreate, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    body.check_models(body.models)
    c = require_enabled(ctx)
    payload = body.model_dump(exclude_none=True)
    payload["metadata"] = {**body.metadata, "created_by": "dgxmm", "dgxmm_user": p.user.username}
    try:
        rec = await c.key_generate(payload)
    except LiteLLMError as e:
        raise litellm_error(e)
    audit(ctx.db, p.user.id, "litellm_key_create", body.key_alias, {"models": body.models})
    return {**key_out(rec), "key": rec.get("key")}


@router.patch("/keys/{token}")
async def update_key(body: KeyPatch, token: str = Path(...), p: Principal = Depends(require_admin),
                     ctx: AppContext = Depends(get_ctx)):
    token = _token(token)
    body.check_models(body.models)
    c = require_enabled(ctx)
    changes = body.model_dump(exclude_unset=True)
    try:
        rec = await c.key_update(token, changes)
    except LiteLLMError as e:
        if e.code == "litellm_not_found":
            raise api_error(404, "litellm_key_not_found", "no such virtual key in LiteLLM")
        raise litellm_error(e)
    out = key_out(rec)
    out["token"] = out["token"] or token
    audit(ctx.db, p.user.id, "litellm_key_update", out.get("key_alias") or token[:12],
          {k: v for k, v in changes.items() if k != "metadata"})
    return out


@router.delete("/keys/{token}", status_code=204)
async def delete_key(token: str = Path(...), p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    token = _token(token)
    c = require_enabled(ctx)
    alias = None
    try:
        alias = (await c.key_info(token)).get("key_alias")
        await c.key_delete(token)
    except LiteLLMError as e:
        if e.code == "litellm_not_found":
            raise api_error(404, "litellm_key_not_found", "no such virtual key in LiteLLM")
        raise litellm_error(e)
    audit(ctx.db, p.user.id, "litellm_key_delete", alias or token[:12], {"token_prefix": token[:12]})
    return Response(status_code=204)
