---
description: "Decision-model runtime settings, insights, background jobs and schedules, and the API's rate and size limits."
---

# Decision model, jobs and API limits

## Decision model runtime settings

The Decision model page stores its settings in the `app_settings` table (a backup of `data/` keeps them).
They are not in `config.yaml` and have no environment override:

| `app_settings` key | Default | Set by | Meaning |
|---|---|---|---|
| `decider.enabled` | `false` | **Enable** / **Disable**, `decider enable` / `disable` | Whether the decider should run. Disabled: every feature uses its rules only. |
| `decider.model` | not set (= `vllm-sr/d3-mini`) | **Model** card, `PUT /api/decider/settings {"model": …}` | The model the next (re)start runs: `vllm-sr/d3-mini` or `vllm-sr/d3-nano`. Unknown values fall back to d3-mini. |
| `decider.feature.<feature>.enabled` | `true` | **Features** card | The feature is on: `diagnosis`, `parser`, `plan`, `kind`, `workload`, `health`, `config`, `compat`, `draft`, `engine`, `command`, `schedule`, `reading`, `tuning`, `routing`, `remote_code`. Off turns the feature off: the model is not asked for it, its own rules do not run (compatibility check, config advice, draft and engine suggestions show nothing) and its background job does not run, so it raises no new insights (schedules you created keep running). Exception: the remote-code code scan always runs and shows its findings whenever trust_remote_code would be enabled; off only skips the model's question. The checks that predate the decision model (diagnosis, parser, plan, kind and workload rules, the ⌘K rules) stay; their switch only stops the model from being asked. Disabling the decision model (`decider.enabled`) is different: every feature and job keeps its rules. |
| `decider.feature.<feature>.threshold` | diagnosis `0.8`, parser `0.8`, plan `0.6`, kind `0.8`, workload `0.75`, health `0.85`, config `0.8`, compat `0.8`, draft `0.8`, engine `0.8`, command `0.7`, schedule `0.8`, reading `0.95`, routing `0.8`, remote_code `0.8` (`tuning` is rules only) | **Features** card (in percent), or **Set threshold** on a tuning insight | Minimum confidence for a model answer to be used (0–1). |
| `schedules.paused_all` | `false` | **Pause all schedules** / **Resume all schedules** (Decision model page), `PUT /api/schedules/settings {"paused_all": true}` | No schedule runs while it is set. |
| `decider.deployment_id`, `decider.switch_from` | — | DGXMM | Internal: the decider's deployment, and the previous model while a model switch is not yet confirmed ready. |

Thresholds you saved override the defaults, also when a later release changes a default. The same settings
can be changed with `PUT /api/decider/settings`, for example
`{"features": {"plan": {"threshold": 0.7}, "workload": {"enabled": false}}}` (admin, see
[API limits](#api-limits)).

## Insights, background jobs and schedules

These features have no `config.yaml` keys apart from the time zone ([`tz`](index.md#web-ui-and-sessions),
`DGXMM_TZ` in `.env`) and the decision-log retention above. They are switched per feature on the
Decision model page. Their fixed values, from the code:

| What | Value | Where |
|---|---|---|
| Health check | every 5 minutes, at most 8 model questions per run | `dgxmm/decider/health.py` |
| Health finding clears | after 15 minutes without it (1 hour if the checks were only inconclusive), then visible 1 more hour | `health.py`, `insights.py` |
| Reading labels | on each new reading, plus a sweep every 15 minutes; 8 model questions per run | `reading.py` |
| Traffic profiles / schedule proposals | daily 03:10 / 03:40 (box time zone); proposals need 7 days of metrics, 24 model questions per night | `features.py`, `schedule.py` |
| Schedule runner | every minute; a sleep or stop needs 10 quiet minutes and is retried for 60 minutes; a slot missed by up to 15 minutes still runs; at most 16 schedules per deployment | `schedule.py` |
| Routing advice | daily 05:15, and 15 s (at most 60 s) after deployment and LiteLLM registration changes; 6 model questions per run | `routing.py` |
| Threshold tuning | Sundays 04:30; needs 20 rated answers in 30 days; candidates 0.50–0.95 in steps of 0.05; target 0.95 (commands 0.90) | `tuning.py` |
| Failing job | retried after 1 minute, doubling up to 1 hour; a *degraded* insight after 3 failures in a row (expires 6 hours after the last failure) | `insight_engine.py` |
| Job threads | 2 (the engine's own pool) | `insight_engine.py` |
| Insights | closed ones kept 30 days; proposals expire after 7 days; snooze 1–168 hours | `insights.py` |
| Compatibility check | 5 files (64 KiB each, model card 8 KiB), 5 s per file, 25 s per check; Hub API calls 20 s per connect/read, at most 3 redirects; results cached 7 days | `catalog/hub.py`, `decider/compat.py` |
| Remote-code scan | 30 s per scan (CPU 20 s, 1 GiB address space), then 10 s per file for at most 60 files; files ≤ 1 MiB, ≤ 16 MiB and 300 files in total, ≤ 400 findings; one scan at a time, a failed scan is not repeated for 2 minutes | `decider/remote_code.py`, `decider/code_scan.py` |
| ⌘K commands | at most 300 characters; at most 20 targets offered to the model; nothing shown below 0.3 | `decider/command.py` |

## API limits

| What | Limit |
|---|---|
| Sign-in | 5 attempts per minute per client IP and per username |
| Decision model changes (`PUT /api/decider/settings`, `POST /api/decider/enable`, `disable`, `models/download`, `image/build`, `rotate-key`) | 10 per minute per admin, shared |
| Parser suggestions (`POST /api/decider/suggest-config`) | 30 per minute per user, 120 per minute for all users |
| Plan ranking by the model in `POST /api/deployments/preview` | 30 per minute per user, 120 for all users; over it the preview still answers, with safety marks but no model ranking |
| Decision log (`GET /api/decider/decisions`) | 120 per minute per user |
| Feedback (`POST /api/decider/decisions/{id}/feedback`) | 60 per minute per user |
| Insights: `GET /api/insights` | 120 per minute per user |
| Insights: dismiss, snooze, applied | 60 per minute per user |
| Schedules (`/api/deployments/{id}/schedules`, `/api/schedules/…`, `/api/schedules/settings`) | 60 per minute per user, 240 for all users |
| Each other decision-model feature's routes (config advice, compatibility check, draft and engine suggestions, remote-code check, routing advice, ⌘K commands, free-memory plans, …) | 30 per minute per user, 120 for all users (bucket `decider_<feature>`) |
| Request bodies under `/api/decider` | 64 KiB (`413 request_too_large`, also for chunked bodies) |

Over a limit the API answers `429 rate_limited` with a `Retry-After` header. The limits are in memory and per
process: a restart resets them. They are not configurable.

## Next steps

- [Decision model (user guide)](../user-guide/decision-model.md)
- [Running the decision model](../operations/decision-model.md)
