---
description: "The LiteLLM settings store, the URL-bound master key, one registration per deployment and virtual keys."
---

# LiteLLM integration

**Settings store** (`settings_store.py`). The LiteLLM connection is resolved on every call from the
`app_settings` table, falling back to `config.yaml`: internal URL (`litellm.url` > `litellm_url`), public
URL (`litellm.public_url` > `litellm_public_url`), master key (`litellm.master_key` > the
`litellm_key_file` secret) and `litellm.enabled` (default: on when a key is available). Values are cached
in memory and the cache is updated on write, so a change made on the LiteLLM page applies to the next
call without a restart.

**Master key bound to its URL.** A stored key is saved together with the internal URL in effect at that
moment (`litellm.master_key_url`) and is used only while the internal URL is still that URL. The
secret-file key is used only with the configured default `litellm_url`. Pointing the internal URL at
another host therefore never sends an existing write-only key there; the API reports
`master_key_hint` instead and the UI asks for the key. The key is never returned by the API, never
logged and never audited, and error texts from LiteLLM are scrubbed of the key, `sk-…` tokens and key
hashes before they are logged or returned.

**One registration per deployment** (`LiteLLMSync`). The desired state is one entry per `ready`
deployment, keyed by deployment id and model name (`alias` or served name), with
`model_info.dgxmm_id` marking it as DGXMM's. Each sync lists the models carrying a `dgxmm_id`, deletes
every one that is not desired (stopped deployments, an old model name after a rename, duplicates, and
the second, served-name entry that older versions registered next to an alias), and adds what is
missing. That makes the migration from the old two-name scheme automatic on the first sync, and a
rename re-registers exactly once. Syncs are serialised by a lock. If two ready deployments end up with
one name anyway (the deployer prevents it when names are claimed), the oldest keeps it and the others
get `litellm_status: error` with a `model_name_conflict` reason. The last sync's result and conflicts
feed `litellm_error` in deployment views and the per-deployment `GET /deployments/{id}/litellm`, so
those read paths never call LiteLLM.

**Model names are unique.** The deployer checks names under its memory lock when a deployment is
deployed, started, woken or renamed: a name is taken if it is another deployment's alias (in any state)
or the model name of another deployment that is planned, starting, loading, ready or unknown.

**Virtual keys** are proxied to LiteLLM's `/key/*` API with the master key. Before a key operation DGXMM
checks LiteLLM's `/health/readiness`; without a connected database it answers `409 litellm_no_db`.
Keys are identified by their hash; the full key is returned only in the create response. Keys created
by DGXMM carry `metadata.created_by = "dgxmm"` and the user name.

**Status** (`GET /api/integrations/litellm`, or on its own `GET /api/integrations/litellm/status`) uses
only unauthenticated calls (`/health/liveliness`, `/health/readiness`, and `/openapi.json` for the
version), cached for 15 seconds. The probe has a 4 s deadline; a slower version fetch finishes in the
background and fills in the cached status. `GET /api/integrations/litellm?probe=false` returns the
settings at once with the cached status (or `null`); the LiteLLM page uses it so the probe never holds
the page up. **Test connection** is always fresh and also checks the key with an authenticated
`/model/info`. `GET /api/integrations/litellm/models` caches LiteLLM's model list for 15 seconds (DGXMM's
own registrations invalidate it).

**Leaving a LiteLLM.** A `PUT /api/integrations/litellm` that disables the integration or changes the
internal URL first deletes every model with `model_info.dgxmm_id` from the LiteLLM in effect before the
change, using the key bound to that URL, under the sync lock (so the sync loop cannot re-register in
between), then saves. It is best effort (15 s bound): the response carries `deregistered`,
`deregister_error` and a `warning` naming the old URL when some may remain; the audit entry records the
count and error. The settings are saved either way.

## Next steps

- [LiteLLM configuration](../configuration/litellm.md)
- [LiteLLM (user guide)](../user-guide/litellm.md)
