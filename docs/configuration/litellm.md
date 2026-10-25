---
description: "Connecting DGXMM to an existing LiteLLM or the bundled one, and runtime settings precedence."
---

# LiteLLM

## LiteLLM integration

DGXMM registers every `ready` always-on deployment with a [LiteLLM](https://docs.litellm.ai/) proxy and
removes it again when the deployment stops or sleeps. [On-demand](../user-guide/on-demand.md) deployments
are registered in every state except failed, with DGXMM's gateway as their `api_base`
(`http://dgxmm:8090/dgxmm/ondemand/<id>/v1`), the on-demand key, `timeout`/`stream_timeout` of `ondemand_queue_timeout_s` + 600 s (at least
1200 s), `num_retries: 0` and `model_info.health_check_params`, which marks LiteLLM's health checks so
they never start a model (see [on-demand gateway](../architecture/ondemand.md#litellm-registration)). Each deployment is registered **once**, under its
**model name** (the **Model name (LiteLLM)** set in the deploy wizard or on the deployment's LiteLLM tab,
else the served name), with `model: openai/<served name>`, `api_base: http://dgxmm-<name>:<port>/v1`
and `model_info.dgxmm_id` set to the deployment id. Clients therefore use one endpoint,
`<public URL>/v1`, and pick the deployment with `"model"`. Models in LiteLLM without a `dgxmm_id` are
never changed.

Model names are unique. DGXMM refuses a deploy, start, wake or rename (`409 model_name_taken`, naming
the deployment that holds it) when the name is another deployment's model name, or another running
deployment's served name. Two deployments of the same model therefore need different model names.

Requirements:

1. LiteLLM stores models in its database: `general_settings: { store_model_in_db: true }` with a
   configured database. Virtual keys need that database too.
2. DGXMM has the master key: on the LiteLLM page, or in `secrets/litellm_master_key`. If you install
   with no key file and a container named `litellm-proxy` is running, the installer reads the key from
   that container. Otherwise pass `LITELLM_MASTER_KEY=…`, write the file yourself, or enter the key on
   the LiteLLM page.
3. DGXMM can reach the internal URL (default `http://litellm-proxy:4000`, so a container named
   `litellm-proxy` on the DGXMM network).
4. LiteLLM can reach the engines and DGXMM: put it on the same Docker network
   (`docker network connect dgxmm litellm-proxy`), or install DGXMM with `DGXMM_NETWORK=<LiteLLM's
   network>`. On-demand models need this direct connection to the `dgxmm` container; LiteLLM going
   through a reverse proxy, or running outside Docker, is refused by the gateway (see
   [`dgxmm_internal_url`](index.md#on-demand-models)).

Example for a fresh install next to an existing LiteLLM on network `ai`:

```bash
DGXMM_NETWORK=ai LITELLM_URL=http://litellm-proxy:4000 ./install.sh
```

Then open **Integrations > LiteLLM**, set the **Public endpoint URL** your clients use, and press **Test
connection**.

Each deployment's LiteLLM status is `synced`, `pending` (the last sync failed and DGXMM retries; see
[troubleshooting](../operations/troubleshooting.md#litellm-status-stays-pending)), `error` (not registered because another
deployment holds its model name, or `ondemand_not_configured` for an on-demand deployment without the gateway
key), `disabled` (integration off or no master key) or `none` (not registered: an always-on deployment
that is not ready, or a failed one).
For `pending` and `error`, the deployment's LiteLLM tab and the Deployments table tooltip give the
exact reason. Switching models moves the model name to the new deployment, so clients keep working.

**Upgrading from an older DGXMM:** older versions registered a deployment twice when it had an alias
(once under the served name, once under the alias). The first sync after the upgrade removes the
served-name entry and keeps the alias. Clients that called the served name of an aliased deployment
must switch to the alias.

### Runtime settings and precedence

The LiteLLM page stores these in DGXMM's database (table `app_settings`). DGXMM reads them each time it
talks to LiteLLM, so changes apply immediately:

| Setting | Default when not set on the page | Notes |
|---|---|---|
| Internal URL | `litellm_url` | Where DGXMM sends its API calls. Setting it back to the `litellm_url` value removes the override. |
| Public endpoint URL | `litellm_public_url` (empty) | Only shown to users and in snippets; DGXMM never calls it. Clearing it on the page keeps it empty, even when `litellm_public_url` is set. |
| Master key | the file in `litellm_key_file` | Write-only. Never returned by the API, never logged, never written to the audit log. |
| Integration enabled | on when a master key is available | Turning it off stops registering; existing registrations stay in LiteLLM. |

Precedence is always: value saved on the page > `config.yaml` (or `DGXMM_*` variable) > built-in
default.

The master key is bound to a URL, so it can never be sent to a server it was not meant for:

- a key entered on the page is used only with the internal URL that was in effect when it was saved;
- the key from `secrets/litellm_master_key` is used only with the default `litellm_url`.

If you change the internal URL, no key applies until you enter the master key for the new URL; the
page says so. Changing the URL back makes the earlier key usable again. Registrations on the previous
LiteLLM are not removed when you switch to another one.

## Bundled LiteLLM

For a fresh box without a LiteLLM, `./install.sh --with-litellm` (or `DGXMM_WITH_LITELLM=1`) enables the
compose profile `litellm`, which has two more services:

- `dgxmm-litellm`: LiteLLM pinned to a released, multi-arch image. It uses
  `deploy/litellm/config.yaml` (`store_model_in_db: true`, so DGXMM's registrations persist) and
  publishes `${DGXMM_BIND}:${DGXMM_LITELLM_PORT:-4000}`.
- `dgxmm-litellm-db`: Postgres 16 (`postgres:16.15-trixie`, pinned by digest) with its data in `data/litellm-db`. It runs as `DGXMM_UID`, so the
  files belong to you. It has no published port. Containers on the shared Docker network can reach
  it, so it is password-protected (`secrets/litellm_db_password`).

Neither has a fixed container name: they are called `<compose project>-dgxmm-litellm-1` and so on.
DGXMM and engines reach LiteLLM as `http://dgxmm-litellm:4000` on `DGXMM_NETWORK`.

Keys. The bundled LiteLLM always gets its **own** master key, `secrets/litellm_bundled_master_key`,
generated once and never overwritten. It is never the key of another LiteLLM.

- Fresh install: `config.yaml` is generated with `litellm_url: http://dgxmm-litellm:4000`. If
  `secrets/litellm_master_key` is empty, the installer writes the same key there too, so registration
  works with nothing to configure. It also writes `litellm_public_url: http://<host IPv4>:<port>` (the
  first LAN address, or the `DGXMM_BIND` address), so the UI shows the right endpoint. With
  `DGXMM_BIND=127.0.0.1` it is left out: set the public URL of your reverse proxy on the LiteLLM page.
- Existing install that uses another LiteLLM: `config.yaml` and `secrets/litellm_master_key` are kept
  and the installer prints a note. To switch DGXMM to the bundled LiteLLM, set `litellm_url` to
  `http://dgxmm-litellm:4000` and the master key from `secrets/litellm_bundled_master_key` on DGXMM's
  LiteLLM page (Integrations > LiteLLM). If a container named `litellm-proxy` runs, the installer
  warns that a second LiteLLM will run and needs a free port.

At the end the installer prints the single endpoint, `http://<host>:<port>/v1`. Clients pass the DGXMM
model name as `model`. Create virtual keys on DGXMM's LiteLLM page, or call `POST /key/generate` on
the endpoint with the bundled master key. `./install.sh status` lists both containers.

If the database or LiteLLM fails to start, the installer still finishes DGXMM's setup (and shows the
first-run admin password). It then prints the redacted LiteLLM and database logs and exits non-zero.

**Turning it off:** set `DGXMM_WITH_LITELLM=0` in `.env` and run `./install.sh`. It stops and removes
the two containers and leaves `litellm` out of `COMPOSE_PROFILES` for that run. `data/litellm-db` and
the secrets are kept; remove `litellm` from `COMPOSE_PROFILES` in `.env` too, so a manual
`docker compose up -d` does not start it again. `uninstall --purge` deletes `data/litellm-db` but leaves
the images. Remove them with `docker image rm postgres:<tag> ghcr.io/berriai/litellm:<tag>` (see
`docker-compose.yaml` for the exact tags).

For backups of the LiteLLM database (virtual keys, budgets, spend), see
[backup and restore](../operations/backup.md).

## Next steps

- [LiteLLM (user guide)](../user-guide/litellm.md)
- [LiteLLM integration (architecture)](../architecture/litellm.md)
