---
description: "The essentials of DGXMM's security: the Docker socket, network exposure, the container guard, secrets, process command lines and the decision model."
---

# Security overview

!!! danger "Access to the Docker socket is equivalent to root on the host"
    DGXMM mounts the Docker socket so it can start engine containers. Only give the `admin` role to
    people you would trust with root.

!!! danger "Node SSH key gives root-equivalent Docker access on remote nodes"
    When using multiple nodes, the SSH key configured in `node_ssh_key_file` grants Docker socket access
    on remote nodes, which is equivalent to root. Protect this key with mode 0600, store it only on the
    node running DGXMM, and only give it to the dedicated user account on each remote node. Rotate it
    immediately if compromised.

- The default bind is `0.0.0.0` (your LAN). Do not expose that port to the internet. For remote access,
  install with `DGXMM_BIND=127.0.0.1` behind an HTTPS reverse proxy (see
  [reverse proxy](../configuration/networking.md#reverse-proxy)), or use a VPN.
- DGXMM only stops or removes containers labelled `dgxmm.managed=true`, plus the short-lived delete
  helpers it starts itself (`dgxmm.helper=true`). Engine ports are published on `127.0.0.1` only and
  have no authentication: give other machines access through LiteLLM and virtual keys.
- The LiteLLM master key is write-only: DGXMM never shows or logs it, and only sends it to the LiteLLM
  URL it was entered for. A virtual key is shown once, when it is created.
- Engine containers have API keys (generated per deployment, delivered as `--api-key` to vLLM and llama.cpp).
  LiteLLM and DGXMM's health checks send the key. Note that vLLM enforces API keys only on `/v1`, `/v2`,
  `/inference`, and `/cohere` routes; other routes including development endpoints stay open when
  `--enable-sleep-mode` is set (sleep mode is enabled when dev mode is, and dev mode is on for single
  deployments that support sleep; cluster deployments never get sleep mode).
- RPC and NCCL ports (for cluster deployments) are bound to the cluster link only (point-to-point RDMA
  network), never exposed on the LAN. Remote engine API ports are published on the node's `publish_bind`
  address (typically the cluster IP), not `0.0.0.0`.
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
