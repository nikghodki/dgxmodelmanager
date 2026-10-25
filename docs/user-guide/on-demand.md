---
description: "On-demand models: started by the first request, put to sleep when idle, making room or queueing when memory is short."
---

# On-demand models

A deployment runs in one of two modes:

- **On demand**: the first request through LiteLLM starts the deployment, and it is stopped again after
  it has had no request for its idle timeout (5 minutes by default). While it is stopped it holds no
  memory, but it stays registered in LiteLLM under the same model name, so clients change nothing.
- **Always on**: the deployment runs until you stop it, as in earlier versions. Deployments that existed
  before the upgrade are always on, and so is the [decision model](decision-model.md).

On demand needs LiteLLM: only requests that come through it start a stopped model. So the deploy wizard
picks **On demand** by default only when the [LiteLLM integration](litellm.md) is enabled and its last
sync succeeded, and **Always on** otherwise, with the hint *On demand needs LiteLLM (it starts models on
requests that come through it)*. The API (`POST /api/deployments`) defaults to always on; send
`"mode": "on_demand"` to get an on-demand deployment.

On-demand models are all equal: there are no priorities between them, and the on-demand controller never
cuts off a request that is being answered. Only an administrator's **Stop** or **Remove** and the memory
watchdog can do that.

## What callers see

Clients call the model through LiteLLM as usual (see [calling your models](calling-models.md)). Only
requests that go through LiteLLM reach the on-demand gateway; calls straight to an engine port do not
start anything.

- **The model is ready**: the request is answered as usual. The gateway passes the engine's answer
  through byte for byte, streaming included.
- **The model is stopped and fits**: the request waits on its open connection while the deployment
  starts (loading, compiling and warming up can take 1 to 3 minutes for a large model, longer the
  first time), then gets the normal answer. Nothing is sent before the engine answers. Other requests for
  the same model that arrive meanwhile wait for the same start.
- **The model is stopped and does not fit**: DGXMM first puts idle on-demand models to sleep to make
  room (see [making room](#making-room)). If that is not possible because the memory is held by
  models that are busy or were used recently, the request is **queued**:
  - a **streaming** chat or completions request (`"stream": true`) gets one chunk at once whose text is

    > ⏳ Your message is queued — waiting for memory to free up for `<model>`. It will start automatically.

    followed by a blank line, and then the real answer in the same response once the model is serving.
    A streaming Responses API or Anthropic messages request gets no such chunk (their event streams must
    begin with the engine's own first event): it is held open like a non-streaming one;
  - a **non-streaming** request is held open, and the same line is put in front of the answer's text
    (`choices[0].message.content` for chat, `choices[0].text` for completions, the first `output_text`
    part for the Responses API). The line is added only when the request was really queued, not when it
    only waited for a start. Other answers (embeddings, Anthropic messages, audio, rerank, score) are never
    changed;
  - a request that asks for structured output never gets the line, streaming or not, because its client
    parses the answer: one that sets `response_format`, `tools`, `tool_choice`, `functions`,
    `function_call`, `structured_outputs` or any `guided_*` field (or, for the Responses API, `tools` or a
    `text.format` other than plain text). It waits silently, and its answer is the engine's, unchanged.
- **Still not serving after 15 minutes** (`ondemand_queue_timeout_s`): the request gets HTTP 503 with
  `Retry-After: 60` and an OpenAI-style error. The limit counts from when the request arrived and
  includes the cold start: time in the queue plus the model's start-up. A start that takes longer than
  the limit fails with this same "model … is busy" error even though nothing else held the memory, so set
  the limit well above your slowest cold start:

  ```json
  {"error": {"message": "model <model> is busy; try again later", "type": "server_error", "code": "ondemand_queue_timeout"}}
  ```

  A streaming request that already received the queued chunk gets a final `data: {"error": …}` event
  and `data: [DONE]` instead, because its status line was already sent.
- **The model can never fit**: when the model needs more memory than stopping every other on-demand model
  could free (the usable memory minus what always-on deployments hold), the request fails at once with
  503 `does_not_fit` ("`<model>` needs N GiB but at most M GiB can be freed for on-demand models")
  instead of waiting.
- **The start fails** (the engine crashes or does not become ready): every waiting request gets 503
  `start_failed` with the reason ("`<name>` could not start: …"). For 60 seconds no new start is tried
  for that deployment, then 2, 4 and 8 minutes after further failures, up to 15 minutes. A successful
  start, a mode change or a reconfiguration resets this. A deployment that ends up **failed** is removed
  from LiteLLM until you start it again from its page.
- **An administrator stops or removes the model while it starts for requests**: every waiting request gets
  503 `ondemand_stopped` ("`<name>` was stopped by an administrator … while it was starting for this
  request"). This is not a start failure: the next request starts it again at once.
- **Too many requests waiting**: at most 64 requests (`ondemand_max_queued`, all on-demand models
  together) wait at a time, queued or waiting for a start, since each one holds its request body in
  DGXMM's memory. One more is refused at once with 503 `ondemand_queue_full` and `Retry-After: 60`.
  Requests for a model that is ready are never refused.

LiteLLM returns these errors to the client as its own error response: it drops `Retry-After`, rewrites
the body (the gateway's message stays inside LiteLLM's), and ends a stream that failed after the queued
chunk with its own error events and no `[DONE]` (see
[troubleshooting](../operations/troubleshooting.md#what-the-errors-look-like-through-litellm)). A client that hangs up while it is
queued leaves the queue, and DGXMM starts nothing just for it. A start that has already begun finishes;
the model is then put to sleep after its idle timeout like any other.

## Idle sleep

An on-demand deployment is put to sleep (stopped, its memory returned to the budget) when all of these
hold:

- it has no request in flight through the gateway;
- its last request ended at least its idle timeout ago (the timer also restarts when it becomes ready,
  when DGXMM restarts, and when it is switched to on demand);
- the engine's own metrics (scraped every 5 seconds) show no running or waiting requests, so a direct
  call to the engine port that bypasses LiteLLM keeps it awake while it is being answered. Direct calls do
  not restart the idle timer.

DGXMM checks every 15 seconds. The deployment's state becomes **Stopped** and the event *Put to sleep
after 5 min idle* is recorded. The next request starts it again. A model deployed on demand starts right
away, like any deployment, and is put to sleep only once it has been idle for its timeout.

An on-demand vLLM deployment you put into vLLM sleep mode by hand counts as idle as well and is stopped
after its timeout.

## Making room

When a requested model does not fit in the free memory, DGXMM stops other on-demand models, but only ones
that:

- are on demand (never an always-on deployment or the decision model);
- are idle by the rules [above](#idle-sleep): nothing in flight, idle for their full timeout, and no
  activity in the engine's metrics.

The least recently used go first, and only as many as needed. The order is by last use, not by size: to
make room for a small model, DGXMM may stop a large idle model if that is the one used longest ago, and
that large model then needs a full cold start on its next request. If a model must stay warm, make it
**always on**, or give it a longer idle timeout so that it is not idle (and not a candidate) between its
requests. If even all idle candidates together would not free enough, none is stopped and the request is
queued. Each stop is checked again under the
deployer's lock just before it happens, so a model that got a request in the meantime is not stopped.
Every start, for a request or not, still goes through the normal memory check and the
[start-up queue](deploying.md#the-start-up-queue), so an on-demand start never runs the box out of memory.

## The queue

Queued requests are served first come, first served across all on-demand models. Requests for a model
that is already starting join that start instead of queueing.

- Only the request at the head of the queue may put idle models to sleep for its memory.
- A later request may go ahead of the head when its model fits in the memory that is free right now, without
  stopping anything and without using memory freed for the head. A large model at the head can therefore
  wait longer while smaller ones start.
- The queue is checked again whenever a deployment stops or changes state, when an idle timeout runs
  out, and at least every 15 seconds.

At most `ondemand_max_queued` requests (64 by default) wait at a time; one more gets 503
`ondemand_queue_full` at once. The queue lives in memory: a DGXMM restart drops it, together with the
waiting connections.

## Choosing and changing the mode

- **Deploy wizard**: the **Mode** choice in step 2, *On demand — start on the first request, sleep after
  N min idle* or *Always on*. On demand is preselected when LiteLLM is enabled and synced, always on
  otherwise (see above). For on demand, **Idle minutes before sleeping** (1 to 1440) is filled with the
  DGXMM default; a value you leave at the default follows later changes of that default.
- **Deployment page**: the **Mode** card on the Overview tab (admins). Change the mode or the idle minutes
  and press **Save mode**. Empty idle minutes mean the DGXMM default. Viewers see the mode read-only. The
  decision model has no Mode card: it is always on.
- **API**: `PATCH /dgxmm/api/deployments/{id}/mode` (admin, with the session's CSRF token), body
  `{"mode": "on_demand", "idle_timeout_s": 600}` or `{"mode": "always_on"}`. `idle_timeout_s` is 60 to
  86400 seconds; `null` means the default and leaving it out keeps the current value. The change is
  audited as `ondemand_mode`.

Switching never drops requests. DGXMM re-registers the model in LiteLLM, adding the new entry before it
removes the old one, and requests already running finish on the path they started on.

- **On demand → always on** of a ready deployment changes the entry at once. A stopped, failed or asleep
  deployment is started (or woken) first, through the normal memory check; if it does not fit, the
  switch is refused with `409 insufficient_memory` and nothing changes. While it starts, it stays on
  demand (the API shows `"pending_mode": "always_on"`, the Mode card *Switching to always on*): LiteLLM
  keeps its gateway entry, and requests that arrive meanwhile wait for the start like any on-demand
  request instead of finding no model. Once it is ready it becomes always on and the entry points at the
  engine directly. If the start fails, or it is stopped meanwhile, the switch is applied then.
- **Always on → on demand** starts the idle timer at the moment of the switch.

**Switch model…** keeps the deployment's mode for the replacement. **Stop** on an on-demand deployment
stops it, but the next request starts it again; switch it to always on first if it must stay down. A Stop
while it is starting for requests fails those requests with `ondemand_stopped` (see above).

## What counts as traffic

Only inference requests start a model or keep it awake:

- `POST` to an inference endpoint the engine serves under `/v1` counts: `chat/completions`,
  `completions`, `embeddings`, `responses` (the OpenAI Responses API; LiteLLM uses it for clients that
  speak the Anthropic Messages API, such as Claude Code), `messages`, `chat/completions/batch`,
  `audio/transcriptions`, `audio/translations`, `rerank` and `score`. An endpoint the model's engine does
  not offer (llama.cpp has no audio endpoints, for example) still starts the model, and the engine then
  answers with its own error;
- `GET /v1/models` is answered from the deployment's record and never starts the model (LiteLLM usually
  answers its own `/v1/models` without asking the gateway at all);
- `GET /v1/models/<id>` is passed to the engine only while the model is ready, otherwise the answer is
  503 `ondemand_not_ready`;
- nothing else reaches the engine through the gateway: health, metrics and admin endpoints, the
  `render`/`tokenize` helpers, `messages/count_tokens`, LoRA loading and stored responses
  (`GET /v1/responses/<id>`) answer 404;
- **LiteLLM health checks** (`GET /health` on LiteLLM, and its background checks if you turned them on)
  send a real chat completion. DGXMM marks the on-demand entries so these probes carry an
  `x-dgxmm-health-check: 1` header (`model_info.health_check_params`). **DGXMM answers these probes
  itself**, with an empty answer, in every state: they never reach the engine, never start a model and
  never count as activity (header `x-dgxmm-ondemand-state` names the deployment's state). So LiteLLM's
  `/health` reports an on-demand model as healthy while it is asleep (it is available on demand), and a
  healthy result does **not** prove that the engine is serving; only a `failed` deployment is reported
  unhealthy. Check the deployment page for the engine itself. Because any LiteLLM client could send the
  same header, a request carrying it never gets a real answer;
- request bodies are capped at 16 MiB, audio uploads (`audio/transcriptions`, `audio/translations`)
  included: a larger file gets 413 `request_too_large`.

DGXMM's own health checks and metrics scrapes go straight to the engine container and never count as
traffic.

## Status badges

The Deployments table, the dashboard and the deployment page show on-demand deployments with these badges
in place of the plain state:

| Badge | Meaning |
|---|---|
| **Sleeping (on demand)** | Stopped, stopping or asleep; the next request starts it. |
| **Starting for a request** | Started because a request asked for it; the request is waiting. |
| **Ready** | Running; with requests in flight or idle. |
| **N queued** | N requests wait in the queue for memory for this model. |

The badges and counters follow live: a change of the queued or in-flight requests is pushed to the UI at
most once a second per deployment.

Other states show the normal state badge: for example **Loading** while you started it by hand, and
**Failed** for a deployment whose start failed. A failed deployment is not asleep: it is removed from
LiteLLM and requests no longer reach it until you start it again from its page.

## Events

The deployment's **Events** tab and the audit log record what the on-demand controller did, with the actor
`ondemand`:

| Event | When | Audit action |
|---|---|---|
| *Started for a request* | a request started (or woke) the deployment | `ondemand_start` |
| *Put to sleep after N min idle* | the idle timeout ran out | `ondemand_sleep` (reason `idle`) |
| *Put to sleep to make room for X* | stopped so that deployment X could start | `ondemand_sleep` (reason `make_room`) |
| *Queued request timed out* | a queued request gave up after the queue limit (at most once a minute per deployment) | `ondemand_queue_timeout` |

## Before you rely on it

A request can wait up to 15 minutes (`ondemand_queue_timeout_s`), counted from its arrival: time in the
queue and its model's cold start together, so set that limit above your slowest cold start, and every
proxy between the client and DGXMM must allow long responses. DGXMM sets the LiteLLM side itself,
following the queue limit (1500 s by default); reverse proxies in front of LiteLLM need a read timeout of
at least that, 1800 s (30 minutes) is recommended. See
[troubleshooting](../operations/troubleshooting.md#requests-to-an-on-demand-model-fail-after-about-60-seconds).
The global defaults are in [configuration](../configuration/index.md#on-demand-models), and how it works
inside is in [on-demand gateway](../architecture/ondemand.md).

## Next steps

- [Deployments](deployments.md)
- [Calling your models](calling-models.md)
- [On-demand gateway (architecture)](../architecture/ondemand.md)
