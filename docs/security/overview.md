---
description: "The essentials of DGXMM's security: the Docker socket, network exposure, the container guard, secrets, process command lines and the decision model."
---

# Security overview

!!! danger "Access to the Docker socket is equivalent to root on the host"
    DGXMM mounts the Docker socket so it can start engine containers. Only give the `admin` role to
    people you would trust with root.

- The default bind is `0.0.0.0` (your LAN). Do not expose that port to the internet. For remote access,
  install with `DGXMM_BIND=127.0.0.1` behind an HTTPS reverse proxy (see
  [reverse proxy](../configuration/networking.md#reverse-proxy)), or use a VPN.
- DGXMM only stops or removes containers labelled `dgxmm.managed=true`, plus the short-lived delete
  helpers it starts itself (`dgxmm.helper=true`). Engine ports are published on `127.0.0.1` only and
  have no authentication: give other machines access through LiteLLM and virtual keys.
- The LiteLLM master key is write-only: DGXMM never shows or logs it, and only sends it to the LiteLLM
  URL it was entered for. A virtual key is shown once, when it is created.
- Process command lines are visible to every signed-in user. Secrets in them are masked on a
  best-effort basis, so do not pass secrets as command-line arguments (see
  [known limitations](../operations/known-limitations.md)).
- The optional decision model runs as an unprivileged, read-only container on an internal network with no
  published port; its bearer key is delivered as a file and can be rotated. It never runs anything: its
  answers prefill dialogs or raise insights, and only schedules an admin created act on their own.

!!! warning "A clean code scan is not a review"
    The remote-code check is a static scan with blind spots. Read a repository's code before you enable
    trust_remote_code.

| Read next | For |
|---|---|
| [Threat model](../security.md) | assets, trust boundaries, every control of the decision model, residual risks |
| [Application security model](application.md) | authentication, sessions, CSRF, roles, audit, secrets, the container guard |
| [Security hardening](hardening.md) | how to deploy DGXMM safely |
| [Security policy](reporting.md) | how to report a vulnerability privately |

## Next steps

- [Security hardening](hardening.md)
- [Threat model](../security.md)
