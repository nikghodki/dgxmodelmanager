---
description: "DGXMM's SQLite tables, migrations and retention."
---

# Data model

SQLite tables (`dgxmm/db.py`):

| Table | Contents |
|---|---|
| `users` | username, argon2id hash, role (`admin` / `viewer`), `must_change_pw` |
| `cli_tokens` | SHA-256 of a short-lived admin CLI token, user, scope, expiry (deleted when the command ends) |
| `sessions` | SHA-256 of the session token, user, CSRF token, client IP, expiry |
| `deployments` | name, engine, model reference, validated config, state, container id, port, priority, alias (the LiteLLM model name), reservation and estimate, LiteLLM status, last error, exit code, whether the last exit was an OOM kill (`oom_killed`), `system` (the decision model). Ids are never reused (`AUTOINCREMENT`, migration 30). |
| `app_settings` | runtime settings changed in the UI (key, value, updated at): the LiteLLM URLs, enabled flag, master key and the URL it is bound to; the decision model's `decider.*` settings |
| `decisions` | one row per decision-model question: feature, subject, source (`rules` / `model` / `none`), choice, confidence, probabilities, latency, model and revision, SHA-256 of the input, error code, feedback (30 days / 50 000 rows) |
| `diagnoses` | per failure event: cause, source, confidence, summary, fix hint, patch, redacted log excerpt (latest 20 per deployment; unique per deployment and event) |
| `artifact_kinds` | model-derived kinds per artifact and snapshot time, with source and confidence |
| `schema_version` | applied migrations: 1 baseline, 2 one diagnosis per failure event, 3 one live insight per dedupe key, 20 one traffic-profile row per deployment and hour, 30 the `deployments` rebuild with `AUTOINCREMENT` (foreign keys off for that step only, checked before and after) |
| `deployment_events` | state transitions with message, optional log tail and, for stops, the `actor` that asked (`user`, `schedule:<id>`, …) |
| `memory_calibration` | estimated vs measured bytes per engine, model and config hash; OOM records; the reading label |
| `benchmarks` | one row per concurrency level: TTFT p50/p95, ITL p50, total and per-request tok/s, and the reading label (`reading`, `quality`) |
| `benchmark_runs` | one row per benchmark run: levels, status, the level that failed and its error |
| `metric_rollups` | 1-minute aggregates per deployment and for the host (30-day retention) |
| `downloads` | repo, revision, files, progress, state, error |
| `favourites` | model references starred in the Planner |
| `audit_log` | user, action, target, details, timestamp |
| `insights` | kind, feature, subject, severity, templated title and body, confidence, source, decision id, action descriptor, status, snooze end, dedupe key, created / updated / expires at (one live row per dedupe key; closed rows 30 days) |
| `deployment_schedules` | deployment, action (`sleep` / `wake` / `stop` / `start`), days bitmask, `HH:MM`, time zone, enabled, created by, created / updated at, last run (time, local date, result) |
| `traffic_profiles` | per deployment and hour of the week: requests per minute p50 / p95, active fraction, observed minutes, weeks observed and idle (unique per deployment and hour) |
| `compat_cache` | compatibility verdicts, reason keys and flags per repo and commit sha (7 days; never file contents) |

Live metrics (about the last hour at the scrape interval) are kept in memory only.

Schema changes run at startup: new tables and missing nullable columns and non-unique indexes are added, then
each versioned migration in `dgxmm/migrations.py` runs once, in a transaction, recorded in `schema_version`
(see [upgrading an existing install](../getting-started/upgrade.md#upgrading-an-existing-install)).
Retention runs once a minute with the metrics rollups; the limits are listed per table above and in
[startup and background loops](background-loops.md).

## Next steps

- [Backup and restore](../operations/backup.md)
- [HTTP API](request-flow.md#http-api)
