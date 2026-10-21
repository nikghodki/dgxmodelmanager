---
description: "How to deploy DGXMM safely: the Docker socket, network exposure, proxies, engine endpoints and secrets."
---

# Security hardening

See also the threat model in [security.md](../security.md).

- **Docker socket = root.** Anyone with the DGXMM admin role can start containers on the host. Only
  make trusted people admins, and give everyone else the `viewer` role.
- **Do not expose DGXMM directly to the internet.** For access from outside your LAN, use a VPN, or
  install with `DGXMM_BIND=127.0.0.1` behind an HTTPS reverse proxy (see
  [configuration](../configuration/networking.md#reverse-proxy)) and set `secure_cookies: always`.
- **Firewalls:** ports published by Docker bypass host firewall rules such as `ufw`. Restrict access
  with `DGXMM_BIND`, not only with the firewall.
- **Client IPs behind a proxy:** DGXMM only trusts `X-Forwarded-For` from IPs/subnets configured in
  `trusted_proxies` (default: `127.0.0.1,::1`, loopback-only). Behind a reverse proxy, set this to the
  proxy's IP or Docker network subnet (for example `DGXMM_TRUSTED_PROXIES=172.23.0.0/16` in `.env`
  for a proxy container on the `gateway` network) so that client IPs in logs and rate limiting are
  correct. `*` trusts every client and is not recommended. Have
  the proxy overwrite the header (`proxy_set_header X-Forwarded-For $remote_addr;` in nginx) rather than
  append to it. The per-username login limit applies either way.
- **Engine endpoints have no authentication.** Keep `publish_bind: 127.0.0.1` and give other machines
  access through LiteLLM with virtual keys (one per app or person, limited to the models they need).
- **Host PID namespace.** The DGXMM container runs with `pid: host` so that NVML reports the GPU memory of
  the engine containers. DGXMM already reads the host's `/proc` read-only for the process view; with
  `pid: host` its processes could also signal host processes of the same uid (DGXMM never does). It has no
  extra capabilities. It already holds the Docker socket, which is equivalent to root.
- **Process command lines are visible to every user**, masked. Do not put secrets on command lines of
  processes on the box (see [Processes](../operations/troubleshooting.md#processes-masked-or-numeric-values)).
- **The LiteLLM master key** entered on the LiteLLM page is stored in `data/dgxmm.sqlite3`. Protect
  `data/` like `secrets/`.
- **Delete `secrets/admin_password`** after you have changed the first password. It is not needed
  afterwards.
- **Protect backups**: they contain password hashes, session data and your HF and LiteLLM keys.
- **Pin images**: the vLLM image is pinned by digest. Pin the llama.cpp image too if you want
  reproducible deployments (see [engine images](../configuration/models-and-engines.md#engine-images)).

## Next steps

- [Threat model](../security.md)
- [Reporting a vulnerability](reporting.md)
