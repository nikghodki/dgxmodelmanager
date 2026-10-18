---
description: "Authentication, sessions, CSRF, roles, audit, secrets and the container guard."
---

# Application security model

The threat model (assets, trust boundaries, the decision model's container, network, key and supply chain,
residual risks) is in [security.md](../security.md).

- **Authentication** (`auth/`): username and password with argon2id hashes. New passwords need at
  least 12 characters. For unknown users, login still verifies a dummy hash, so response timing does not reveal which usernames exist. Failed and rate-limited
  logins are audited.
- **Sessions**: a random token in an HTTP-only, `SameSite=Strict` cookie `dgxmm_session` scoped to
  `/dgxmm`. Only its SHA-256 is stored. Sessions expire after `session_ttl_s` (12 h) of inactivity
  (the expiry slides on each request). The cookie is `Secure` when the request arrived over HTTPS
  (`secure_cookies: auto`). Changing a password signs out the user's other sessions. An admin reset
  signs out all of them.
- **CSRF**: every non-GET request must send the session's token in `X-CSRF-Token`.
- **Admin CLI tokens**: `python -m dgxmm.admin decider status|enable|disable` creates a short-lived admin
  token in `cli_tokens` (hashed), uses it over the admin unix socket only and deletes it afterwards. It
  is refused on the TCP port and on requests with forwarding headers, accepted only on the decider
  routes and the inventory refresh, and audited with `"via": "cli"`.
- **Rate limiting**: 5 login attempts per 60 seconds, per client IP and per username; per-user limits on
  the decision-model routes (see [API limits](../configuration/decision-model.md#api-limits)).
- **First login**: users created by bootstrap, by an admin or by a reset must change their password
  before any other API call.
- **Roles**: `viewer` has read-only access. Viewers can see deployments (with their LiteLLM endpoint
  and model name), models, metrics, the catalog, the planner and the masked process list. They cannot
  run deployment previews (`POST /deployments/preview`, which may ask the decision model), see users, the
  live container logs, the audit log, the adoptable-container list,
  delete previews (they contain host paths), LiteLLM API bases or virtual keys, and the UI hides the
  admin pages. `admin` is needed for every change. The last admin cannot be deleted or demoted.
- **Audit**: logins (also failed and rate-limited ones), password changes, deploy, start, stop, remove,
  sleep, wake, switch and reconfigure (plus their failures), model name changes, adopt, downloads and
  cancellations, inventory rescans, weight deletions and clean-ups (also refused and failed ones),
  LiteLLM settings changes, syncs and virtual-key changes, benchmarks, user changes, watchdog stops and
  command-line password resets are written to `audit_log`. Audit rows never contain the master key or a
  virtual key.
- **Secrets**: the LiteLLM master key is write-only (stored in `app_settings` or read from the secret
  file), bound to its URL, and never returned, logged or audited. A full virtual key appears only in the
  response that creates it. Process command lines are masked before they reach any client, and process
  environments are never read.
- **Container guard**: `docker_ctl.py` refuses to create a container without `dgxmm.managed=true` and
  to stop or remove one that lacks it. The one exception is the delete helper it creates itself
  (`dgxmm.helper=true`, see [deleting weights](../architecture/storage.md#deleting-weights)). Containers it does not manage are
  only ever read.
- **Engine arguments**: configs are strict schemas, and extra CLI flags must match a per-engine
  allowlist (`engines/argguard.py`). Flags DGXMM controls (`--model`, `--port`, `--host`,
  `--gpu-memory-utilization`, …) cannot be overridden. Model references must be a repo id or a `.gguf`
  file under a configured root.
- **Trust boundary**: the Docker socket gives DGXMM root-equivalent control of the host. Only give the
  admin role to people you would trust with root, and keep the UI off the public internet (see
  [operations](hardening.md)).

## Next steps

- [Threat model](../security.md)
- [Security hardening](hardening.md)
