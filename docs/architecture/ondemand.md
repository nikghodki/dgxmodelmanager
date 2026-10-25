---
description: "How on-demand deployments work: the gateway LiteLLM calls, the controller that starts, queues and stops, and their LiteLLM entries."
---

# On-demand gateway

On-demand deployments (see the [user guide](../user-guide/on-demand.md)) are started by the first request
and stopped when idle. Two parts of the `dgxmm` process do this:

- **The gateway** (`dgxmm/ondemand/gateway.py`): HTTP routes under `/dgxmm/ondemand/{id}/v1/…` that
  LiteLLM calls instead of the engine. It checks who is calling, asks the controller for a slot, then
  streams the request to the engine container with httpx.
- **The controller** (`dgxmm/ondemand/controller.py`, `ctx.ondemand`): per-deployment in-flight counters,
  the wait queue, make-room decisions and the idle reaper. It drives the existing deployer (`restart`,
  `wake`, `stop`), so every start goes through the fit check and the
  [start-up gate](deployer.md).

Always-on deployments do not use either: LiteLLM calls their engine directly, as before.

## A request for a stopped model

```mermaid
sequenceDiagram
  autonumber
  actor C as Client
  participant P as Reverse proxy
  participant L as LiteLLM
  participant G as DGXMM gateway
  participant O as Controller
  participant D as Deployer
  participant E as Engine container

  C->>P: POST /v1/chat/completions (model = model name)
  P->>L: same request
  L->>G: POST http://dgxmm:8090/dgxmm/ondemand/{id}/v1/chat/completions<br/>Authorization: Bearer <on-demand key>
  Note over G: peer on DGXMM's Docker network, no forwarding headers,<br/>clean path, then the key (constant-time compare)
  G->>O: acquire(id)
  alt deployment ready
    O-->>G: slot
  else stopped, fits
    O->>D: restart (fit check, start-up gate)
    D->>E: create and start the container
    E-->>D: ready
    O-->>G: slot
  else stopped, does not fit
    O->>D: stop idle on-demand deployments (least recently used first)
    O->>D: restart
    Note over O: no idle model to stop: queued (FIFO);<br/>a streaming request gets the queued chunk now
    O-->>G: slot once memory is free, or a timeout after 15 min (503)
  end
  G->>E: the request (key and cookies removed)
  E-->>G: answer (SSE chunks)
  G-->>L: the same bytes, unbuffered
  L-->>P: answer
  P-->>C: answer
  Note over G,O: slot released after the last byte;<br/>the idle timer restarts
```

## Gateway

**Routes** (under the base path, not under `/api`, and not in the OpenAPI schema):

| Route | Behaviour |
|---|---|
| `POST /ondemand/{id}/v1/` + `chat/completions`, `completions`, `embeddings`, `responses`, `messages`, `chat/completions/batch`, `audio/transcriptions`, `audio/translations`, `rerank`, `score` (`TRAFFIC`) | Traffic: takes a slot (may start, make room or queue), then forwards. These are the inference endpoints the pinned vLLM serves under `/v1` (llama.cpp's are a subset). LiteLLM sends an Anthropic-format request (`/v1/messages`) for an `openai/` entry to `{api_base}/responses`. With `x-dgxmm-health-check: 1` (LiteLLM health checks, below) no slot is taken and nothing is started. |
| `GET /ondemand/{id}/v1/models` | Answered from the deployment record (`{"object": "list", "data": [{"id": <served name>, …, "owned_by": "dgxmm"}]}`); never wakes. |
| `GET` / `HEAD /ondemand/{id}/v1/models/{model}` | Forwarded only while the deployment is `ready`, without a slot; otherwise 503 `ondemand_not_ready`. |
| anything else under `/ondemand` | 404 `not_found`; the engine is not called. Its admin and development endpoints (`/sleep`, `/wake_up`, `/collective_rpc`, `/reset_prefix_cache`, `/metrics`, `/health`, …), the `…/render` and `tokenize` helpers, `messages/count_tokens`, LoRA loading and stored responses (`GET /v1/responses/{id}`, `…/cancel`) are therefore unreachable through the gateway. |

**Checks, in order** (all errors are OpenAI-style `{"error": {"message", "type", "code"}}`):

1. **Peer guard** (`PeerGuard`, ASGI middleware, before routing). The raw TCP peer, as Uvicorn received it
   and before `X-Forwarded-For` is applied, must be allowed by the peer policy (below), must not be on the
   decider's network, and the request must carry no `X-Forwarded-For`, `Forwarded` or `X-Real-IP`
   header. Otherwise the answer is 404 `not_found`, as if the route did not exist. Requests that come
   through nginx or another proxy are therefore refused even when the proxy sits on the same network.
2. **Path**: a `.`, `..` or empty segment, `\`, `?`, `#`, a control character or an encoded `/`, `\`,
   `#`, `?` or NUL is 400 `invalid_path`.
3. **Key configured**: without a readable `ondemand_key_file`, every request is 503
   `ondemand_not_configured` (logged once). The file is checked at most once a second and re-read when its
   inode, mtime or size changed: a rotated key replaces the old one without a restart (and starts a
   LiteLLM sync), a file that went missing or empty fails closed again.
4. **Rate limit**: a peer with 20 failed key checks in a minute gets 429 `rate_limited` with
   `Retry-After: 60` for the rest of that minute.
5. **Key**: `Authorization: Bearer <key>`, compared in constant time; else 401 `invalid_api_key`.
6. **Body**: at most 16 MiB (multipart audio uploads included), else 413 `request_too_large`.
7. **Deployment**: unknown id 404 `not_found`; the decision model 503 `not_on_demand`. An always-on
   deployment (switched while LiteLLM still pointed here) is forwarded without a slot while ready, else
   503 `not_on_demand`; the gateway never starts it.

**Peer policy** (`peer_policy`, refreshed at startup and every 30 s with the decider network's
subnets). With `ondemand_allowed_cidrs: auto`, the allowed peers are the subnets of the Docker networks
DGXMM's own container is attached to, except the decider's network, and minus each network's gateway
addresses and the first host address of each subnet: connections to a published host port arrive from
there (docker-proxy, hairpin NAT), so the host and the LAN never pass. When DGXMM runs outside Docker,
auto allows nobody. An explicit list is allowed minus the same exclusions: the attached networks are read
whatever the setting says, so their gateway and first host addresses and the decider's network are refused
even when the list covers them (one warning per distinct policy). Until the first refresh nobody is
allowed.

**Forwarding.** The upstream URL is the engine's base URL on the Docker network plus the checked path,
rebuilt from its segments. `Authorization`, `Cookie`, `Host`, forwarding headers and hop-by-hop headers
(including any named in `Connection`) are removed, and `Accept-Encoding: identity` is sent so the bytes
passed back are the bytes the engine wrote. Status, headers (except hop-by-hop, `Server` and `Date`) and
body are passed back chunk by chunk with no buffering; engine 4xx/5xx answers pass through unchanged.
One httpx client per deployment (256 connections, connect timeout 10 s, pool timeout 60 s, no read or
write timeout, because generation can be long). A client replaced after a restart on a new address is
closed only when its last stream ends.

**Queued notice.** A streaming chat/completions or completions request that has to queue for memory gets 200 with SSE headers and one
`chat.completion.chunk` (`text_completion` for `/completions`) with `id: "ondemand-queued"`, the served
model name and the notice as the assistant's content, as soon as the controller reports the queueing.
The engine's SSE bytes follow unchanged. A later error (timeout, start failure, engine unreachable)
becomes `data: {"error": …}` and `data: [DONE]`. A streaming request that only waits for a start sends
nothing until the engine answers, so its response is exactly the engine's. Responses API and Anthropic
messages streams never get a notice: their typed events must begin with the engine's `response.created` /
`message_start` (LiteLLM's Responses-to-Anthropic adapter and the OpenAI SDK rely on that), so they wait
like that, and a timeout is a plain 503. Non-streaming answers of a
request that queued get the notice prepended to `choices[0].message.content` (chat, when it is a string;
tool-call answers with `null` content are left alone), `choices[0].text`, or the first `output_text` part
of the first `message` item of a Responses answer (none: unchanged), and `Content-Length` is
recomputed. Other endpoints are never modified. A request that asks for structured output gets no notice
at all, streaming or not, so its client can parse the answer: any of `response_format`, `tools`,
`tool_choice`, `functions`, `function_call`, `structured_outputs` or a `guided_*` field set (not `null`),
or a Responses `text.format` other than `text`. It waits like a request that only waits for a start.

**LiteLLM health checks.** LiteLLM's `/health` and background checks send a real completion through the
entry. The entry's `model_info.health_check_params` (merged by LiteLLM into the probe only) adds
`x-dgxmm-health-check: 1`. A traffic request with that header is answered by the gateway itself and is
**never forwarded**, in any state or mode: the header is not a secret (`model_info` is readable through
LiteLLM's `/model/info`, and any LiteLLM client can send it with `extra_headers`), so forwarding it would
let a client run a generation without a slot, invisible to the reaper and make-room. It takes no slot,
starts nothing and changes no counter. `chat/completions`, `completions`, `responses`, `messages` and
`embeddings` get a minimal empty answer of their shape (200, header `x-dgxmm-ondemand-state: <state>`);
a `failed` deployment is 503 `ondemand_not_ready`, other endpoints 503 `ondemand_health_unsupported`.
The answer therefore says the gateway and the deployment record are there, not that the engine is
serving: LiteLLM reports on-demand models as healthy while they sleep. The header needs the key like any
other request, and is removed from forwarded requests.

**Hang-ups.** Each request watches for the client's disconnect. It cancels a pending `acquire` (the
controller removes the waiter or its queue entry) or the upstream request, which closes the engine
connection. The slot is released in all cases, also when it was granted at the moment of the hang-up.

**Other errors**: the engine unreachable before it answered is 502 `ondemand_upstream_error`; a
deployment removed while the request waited is 404 `not_found` (503 `ondemand_stopped` when it was
starting for the request); more than `ondemand_max_queued` waiting requests is 503 `ondemand_queue_full`
with `Retry-After: 60`.

**Key rotation.** When the key changes, the failed-attempt counts are cleared, and for 300 s a request with
the previous key gets 401 without counting as a failed attempt: LiteLLM keeps sending it until it reloads
the entries the sync re-registered (up to about a minute), and its single address must not be locked out.

## Controller

`acquire(id)` returns a slot at once when the deployment is ready. Otherwise, when
`ondemand_max_queued` requests already wait (queued or for a start, all deployments together), it fails
at once with `ondemand_queue_full`; else, until the deadline (`ondemand_queue_timeout_s` from the
request's arrival, so the cold start counts against it):

- **A start is already running** for this deployment: wait for it (not in the queue).
- **Stopped or failed**: `deployer.restart`; **asleep** (vLLM sleep mode): `deployer.wake`. Any other state
  (starting by hand, stopping, switched to always on): wait without starting it.
- **`InsufficientMemory`** from the deployer's fit check:
  - if the model needs more than the *capacity* (usable memory minus what always-on and system deployments
    hold), fail at once with `does_not_fit`, except within about 20 s of an on-demand stop (freed memory may
    still read as used) or after this attempt stopped a victim;
  - else pick **victims**: on-demand deployments that are ready or asleep, have no slot in flight, have been
    idle for their full timeout and whose engine metrics (a sample less than 60 s old) show no running or
    waiting requests, least recently used first, until the shortfall is covered. If all of them are not
    enough, none is stopped and the request is queued;
  - stop the victims (*Put to sleep to make room for X*), wait up to about 20 s for the budget to show the
    freed memory, and run the fit check again.

All start and make-room decisions run under one controller lock, and each attempt is shared by every
waiter of that deployment, so a burst of first requests starts the model once. Attempts are shielded:
a waiter that hangs up never cancels a deployer call half-way, and an attempt starts nothing when no one
waits any more.

**Never stopping a busy model.** Every stop (reaper or make-room) goes through `deployer.stop` with a
precheck that runs under the deployer's lock, without an `await`: it re-reads the deployment, re-checks
that it is still idle by the rules above and marks it as stopping, which `acquire` treats as not ready.
Handing out a slot and approving a stop therefore cannot interleave. Only stops the precheck approved are
recorded as on-demand events and audited.

**Queue.** One entry per deployment, in order of first arrival. The head attempts with make-room; while
the head is blocked, later entries may start only if they fit in the memory free right now minus what was
freed for the head (it is reserved for the head until it starts or gives up). Waiters wake on every state
change, removal, stop and queue change, and otherwise after at most 15 s (sooner when an idle timeout is
about to run out). `on_queued` (the queued notice) fires once per request, the first time it has to wait
for memory.

**Reaper.** `run()` calls `reap_once()` every 15 s: every on-demand deployment that is idle by the rules
above is stopped (*Put to sleep after N min idle*). The idle timer restarts on every slot granted and
released, when the deployment becomes ready, when DGXMM first sees it after a restart, and when it is
switched to on demand.

**Stopped or removed while starting.** When a deployment that is starting for requests goes to `stopping`
or `stopped` (an administrator's Stop, a schedule, the watchdog) or is removed, its waiters get 503
`ondemand_stopped`. That is not a start failure: no backoff, and the next request starts it again.

**Start failures.** When a start ends in `failed`, or a background wake fails, all waiters get
503 `start_failed`. New starts of that deployment are refused with the same error for 60 s, doubling to at
most 900 s; a successful start, `set_mode` or a changed model or configuration resets it. A background
wake that found no memory queues the requests instead.

**Switch to always on.** `set_mode(id, "always_on")` of a ready deployment just stores the mode. A
stopped, failed or asleep one is started (`restart`) or woken (`wake`) first and stays `on_demand` with
`pending_mode = "always_on"` until then; while `set_mode` starts it, `acquire` waits instead of starting it
itself, and the reaper and make-room never pick it. The deployer applies the pending mode in the same
transaction as the state change to `ready` (or `failed` / `stopped`). Until then LiteLLM keeps the gateway
entry, so the name never disappears during the 1 to 3 minute start; the next sync adds the direct entry
and then deletes the gateway one. A refused start (`InsufficientMemory`, a docker error) clears the pending
mode and leaves the deployment as it was.

**Live counters.** A change of a deployment's in-flight or queued counts publishes a `deployment` event
(the view with its `ondemand` field), at most once a second per deployment: the first change at once,
later ones folded into one event a second after the previous one, with the counts as they are then.

**Nothing is persisted**: counters, timers and the queue live in memory. After a DGXMM restart every ready
on-demand deployment waits a full idle timeout before it is stopped; queued connections are lost anyway.

`status(id)` feeds the `ondemand` field of deployment views: `{"inflight", "queued", "last_request_at",
"phase"}` with `phase` one of `queued`, `starting_for_request`, `serving`, `failed` (failed or removed: LiteLLM no
longer lists it), `sleeping` (stopped, asleep or stopping) and `idle`, in that order of precedence. The
UI's badges come from it; a failed deployment keeps the normal Failed badge.

## LiteLLM registration

`LiteLLMSync` (`dgxmm/litellm.py`, see [LiteLLM integration](litellm.md)) registers on-demand deployments
in **every state except `failed`**, so the model name stays callable while the model is stopped. The entry
differs from an always-on one only in its `litellm_params`:

| Parameter | Value |
|---|---|
| `api_base` | `<dgxmm_internal_url>/ondemand/{id}/v1`, by default `http://dgxmm:8090/dgxmm/ondemand/{id}/v1` |
| `api_key` | the on-demand key (`ondemand_key_file`) |
| `timeout`, `stream_timeout` | `max(1200, ondemand_queue_timeout_s + 600)` s (1500 by default): a request may wait the queue limit for memory plus the start |
| `num_retries` | `0`: LiteLLM must not retry or fail over a request that is being held |

and in `model_info`, `health_check_params: {"extra_headers": {"x-dgxmm-health-check": "1"}}`, so LiteLLM's
health checks never wake the model (see the gateway above).

`model_info` carries `dgxmm_id` as for every DGXMM entry and `dgxmm_key_fp`, the first 12 hex digits of the
key's SHA-256 (LiteLLM's `/model/info` does not return keys). An entry whose fingerprint is missing or
differs from the current key is replaced on the next sync, so a new key is registered without any manual
step. Likewise `dgxmm_params_fp` fingerprints the extra `litellm_params` and `model_info`: when
`ondemand_queue_timeout_s` changes, or after an upgrade that changed them (entries from before the
health-check marker), the entries are replaced once; an unchanged setting writes nothing. Always-on
entries are unchanged and carry neither fingerprint.

A mode switch changes the entry's `api_base`. Each sync runs its adds before its deletes, so the new entry
exists before the old one is removed and the name is never missing. Without a readable key, on-demand
deployments are not registered, existing entries are left as they are (not deleted), and their LiteLLM
status is `error` with `ondemand_not_configured`. A sync runs when a deployment enters or leaves `ready`,
`failed` or `planned`, when one is removed, on a mode change, and every 60 s.

## Data model

- `deployments.mode` (`on_demand` / `always_on`, nullable: `NULL` reads as `always_on`),
  `deployments.idle_timeout_s` (60–86400 s, `NULL` = `ondemand_idle_timeout_s`) and
  `deployments.pending_mode` (`always_on` while a switch waits for the start, else `NULL`; added by
  `add_missing_columns`).
- Migration 40 (*deployments: on-demand mode*) adds both columns when they are missing and sets
  `mode = 'always_on'` on every existing row, so existing deployments keep their behaviour. New deployments
  default to `always_on` in the API (the deploy wizard sends `on_demand` when LiteLLM is enabled and its last
  sync succeeded, `ondemand.litellm_ready` in `GET /api/host`); the decision model is always `always_on`.
- Events written by the controller have the actor `ondemand`; its audit rows (`ondemand_start`,
  `ondemand_sleep`, `ondemand_queue_timeout`) have no user and record the actor, deployment id and event
  text. `PATCH /api/deployments/{id}/mode` is audited as `ondemand_mode` with the old and new values.

## Next steps

- [LiteLLM integration](litellm.md)
- [Deployer and start-up gate](deployer.md)
- [Security: on-demand gateway](../security.md#on-demand-gateway)
