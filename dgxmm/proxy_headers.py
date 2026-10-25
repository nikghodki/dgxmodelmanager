"""Proxy headers on the TCP listener: uvicorn's ProxyHeadersMiddleware (X-Forwarded-For / -Proto from the
``trusted_proxies`` peers), except that a peer on the decision model's internal docker network is never a proxy,
whatever ``trusted_proxies`` says (docs/security.md, "Decider → DGXMM").

DGXMM joins that network to reach the decider, so the decider (which runs the model snapshot's code) can open
connections to DGXMM's listener. With a broad ``trusted_proxies`` (e.g. ``172.16.0.0/12``) it would otherwise be
trusted to say who the client is: spoof X-Forwarded-For (dodging the per-IP login limit) or X-Forwarded-Proto. Its
forwarding headers are removed instead, so it is seen as itself.

The decider network's subnets are read from Docker by a background loop (context.build_context) into
``ctx.untrusted_proxy_networks``; the middleware itself never calls Docker."""
from __future__ import annotations

import ipaddress
import logging
from typing import Callable, Iterable

from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

log = logging.getLogger(__name__)
# scope key: the TCP peer as uvicorn received it, before ProxyHeadersMiddleware rewrites scope["client"] (read by the
# on-demand gateway, dgxmm/ondemand/gateway.py, which only serves peers on DGXMM's docker networks)
RAW_PEER_KEY = "dgxmm.raw_peer"
FORWARDING = frozenset({b"x-forwarded-for", b"x-forwarded-proto", b"x-forwarded-host", b"x-forwarded-port",
                        b"forwarded", b"x-real-ip"})
Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_networks(cidrs: Iterable[str]) -> list[Network]:
    out = []
    for c in cidrs:
        try:
            out.append(ipaddress.ip_network(str(c).strip(), strict=False))
        except ValueError:
            log.warning("ignoring invalid subnet %r of the decider network", c)
    return out


def in_networks(host: str | None, networks: Iterable[Network]) -> bool:
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped
    return any(ip.version == n.version and ip in n for n in networks)


class GuardedProxyHeaders:
    def __init__(self, app, trusted_hosts: str, untrusted: Callable[[], Iterable[Network]]):
        self.app, self.untrusted = app, untrusted
        self.proxied = ProxyHeadersMiddleware(app, trusted_hosts=trusted_hosts)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            client = scope.get("client")
            scope = {**scope, RAW_PEER_KEY: client[0] if client else None}
            if in_networks(client[0] if client else None, self.untrusted()):
                headers = [(k, v) for k, v in scope.get("headers", []) if k.lower() not in FORWARDING]
                return await self.app({**scope, "headers": headers}, receive, send)
        return await self.proxied(scope, receive, send)


def tcp_app(app, settings, ctx) -> GuardedProxyHeaders:
    """The ASGI app of the TCP listener (the admin unix socket serves `app` itself, without proxy headers)."""
    return GuardedProxyHeaders(app, settings.trusted_proxies, lambda: ctx.untrusted_proxy_networks)
