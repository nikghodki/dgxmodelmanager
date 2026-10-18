---
description: "Docker networks, published ports, and running DGXMM behind an HTTPS reverse proxy."
---

# Networking and reverse proxy

## Networking

- **UI port**: `DGXMM_BIND:DGXMM_PORT` on the host (default `0.0.0.0:8090`) maps to port 8090 in the
  container. Change it in `.env` and re-run `./install.sh`.
- **URL path**: the UI and API are served under `/dgxmm/` (`/` redirects there). The prefix is built
  into the frontend, so a reverse proxy must forward `/dgxmm/` unchanged. Do not strip the prefix.
  Serving DGXMM under another path is not supported.
- **Docker network**: DGXMM and every engine join `DGXMM_NETWORK` (`docker_network`). Containers on the
  same network (LiteLLM, your proxy, other apps) can reach an engine at `http://dgxmm-<name>:<port>/v1`.
- **Engine ports** are also published at `publish_bind:<port>` (default `127.0.0.1`, ports 8100–8199),
  so tools on the host can call `http://127.0.0.1:<port>/v1`.

## Reverse proxy

For HTTPS or internet-facing access, bind DGXMM to loopback and put a proxy in front:

```bash
DGXMM_BIND=127.0.0.1 ./install.sh
```

Set `secure_cookies: always` in `config.yaml` when the proxy terminates HTTPS. The proxy must disable
response buffering for the event stream, allow long-lived connections, and pass `X-Forwarded-Proto`.

Set [`trusted_proxies`](index.md#web-ui-and-sessions) (or `DGXMM_TRUSTED_PROXIES` in `.env`) to the address
DGXMM sees the proxy connect from, otherwise its `X-Forwarded-For` header is ignored:

- **Proxy container on the DGXMM network**: the proxy container's IP or the network's subnet, for
  example `DGXMM_TRUSTED_PROXIES=172.18.0.0/16`.
- **Proxy on the host** (through the published port): the DGXMM network's gateway IP, from
  `docker network inspect <network> -f '{{(index .IPAM.Config 0).Gateway}}'`. A loopback install
  prints this hint at the end.

**nginx in a Docker container on the DGXMM network**: the shipped `deploy/nginx-dgxmm.conf`. It
resolves `dgxmm` through Docker's DNS, so nginx still starts while DGXMM is down. Install with
`DGXMM_NETWORK=<your proxy's network>` (or `docker network connect <network> <nginx-container>`) and include
it in your `server { }` block:

```nginx
    # DGX Model Manager. Variable upstream + Docker DNS so nginx still starts when dgxmm is down.
    location /dgxmm/ {
        resolver 127.0.0.11 valid=10s ipv6=off;
        set $dgxmm_upstream http://dgxmm:8090;
        proxy_pass $dgxmm_upstream;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;          # SSE
        proxy_read_timeout 1h;
        client_max_body_size 1m;
    }
```

**nginx on the host**: use the same block with `proxy_pass http://127.0.0.1:8090;` and without the
`resolver` and `set` lines. Set `trusted_proxies` to the gateway IP.

**Caddy on the host** (it gets certificates automatically):

```caddy
dgx.example.com {
    redir / /dgxmm/
    handle /dgxmm/* {
        reverse_proxy 127.0.0.1:8090 {
            flush_interval -1
        }
    }
}
```

Use `handle`, not `handle_path`: the `/dgxmm` prefix must reach DGXMM.

## Next steps

- [Security hardening](../security/hardening.md)
- [LiteLLM](litellm.md)
