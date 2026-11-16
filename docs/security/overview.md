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
  helpers it starts itself (`dgxmm.helper=true`). Local engine ports are published on `127.0.0.1` only. Each
  engine requires its own API key (below), which DGXMM never shows: give people and other machines access
  through LiteLLM and virtual keys.
- The LiteLLM master key is write-only: DGXMM never shows or logs it, and only sends it to the LiteLLM
  URL it was entered for. A virtual key is shown once, when it is created.
- Engine containers have API keys (generated per deployment, delivered in the environment, `VLLM_API_KEY` /
  `LLAMA_API_KEY`, never on the command line). LiteLLM and DGXMM's health checks send the key. vLLM enforces it
  only on `/v1`, `/v2`, `/inference` and `/cohere`: other routes (`/invocations`, `/tokenize`, `/metrics`, ...)
  stay open, and so do the development routes (`/sleep`, `/collective_rpc`, `/update_weights`, ...) that sleep
  mode turns on (`VLLM_SERVER_DEV_MODE`). With the default `vllm_sleep: stop` no engine gets sleep mode (a sleep
  stops the model). With `vllm_sleep: level1` it is used by local single deployments only: remote single
  deployments get it only with `remote_dev_endpoints: true` (default off), clusters never.
- Remote engine API ports are published on the node's `publish_bind` address (default its cluster IP), never
  `0.0.0.0`. Cluster ranks run on the host network. What DGXMM controls: the vLLM head serves its API on its
  cluster IP (`--host`), NCCL and Gloo bootstrap use the cluster interface (`NCCL_SOCKET_IFNAME`,
  `GLOO_SOCKET_IFNAME`, `VLLM_HOST_IP`), `--master-addr` is the head's cluster IP, a llama.cpp worker's
  `ggml-rpc-server` (no authentication) listens on its cluster IP and the llama.cpp head on its cluster IP. Whether
  vLLM's torch TCPStore (the master port) and its ZMQ sockets also listen on every interface is not set by DGXMM;
  it is to be verified with `ss -ltnp` on each node during acceptance.
- The cluster link is reachable from every container on the DGXMM host (Docker forwards bridge traffic to it),
  unlike local engines on `127.0.0.1`. Restricting it is an owner action (DGXMM never changes firewall rules):
  allow only the DGXMM and LiteLLM containers in the `DOCKER-USER` chain, for example with the cluster subnet
  `192.0.2.0/30` and the two containers' addresses on the `dgxmm` network (pin them, e.g. compose
  `ipv4_address`, so the rules stay valid):

    ```bash
    sudo iptables -I DOCKER-USER -d 192.0.2.0/30 -j DROP
    sudo iptables -I DOCKER-USER -s <dgxmm address> -d 192.0.2.0/30 -j RETURN
    sudo iptables -I DOCKER-USER -s <litellm address> -d 192.0.2.0/30 -j RETURN
    sudo iptables -I DOCKER-USER -d 192.0.2.0/30 -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
    ```

    (`-I` inserts at the top, so the DROP ends up last.) Host-network containers (the cluster ranks) and host
  processes are not forwarded traffic and are not affected; persist the rules with your distribution's tooling.
- Docker over SSH to remote nodes pins each node's host key (`StrictHostKeyChecking yes`, no trust on first use)
  through the ssh configuration DGXMM writes to `<data_dir>/ssh/config` (never a `~/.ssh`), used by every ssh it
  runs. llama.cpp cluster ranks run as DGXMM's uid, never root, and write nothing on the nodes.
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
