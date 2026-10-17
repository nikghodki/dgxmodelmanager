"""Tests for server configuration (__main__.py)."""
from fastapi.testclient import TestClient

from dgxmm.__main__ import server_kwargs
from dgxmm.config import Settings
from dgxmm.proxy_headers import tcp_app


class _Ctx:
    untrusted_proxy_networks: list = []


def _client_seen(settings, peer, headers):
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["client"][0])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})
    TestClient(tcp_app(app, settings, _Ctx()), client=(peer, 40000)).get("/", headers=headers)
    return seen[0]


def test_default_trusted_proxies():
    s = Settings()
    assert s.trusted_proxies == "127.0.0.1,::1"
    assert server_kwargs(s)["proxy_headers"] is False  # applied by dgxmm.proxy_headers, not uvicorn
    assert _client_seen(s, "127.0.0.1", {"X-Forwarded-For": "203.0.113.7"}) == "203.0.113.7"


def test_custom_trusted_proxies():
    s = Settings(trusted_proxies="198.51.100.0/24,172.16.0.0/12")
    assert _client_seen(s, "172.18.0.2", {"X-Forwarded-For": "203.0.113.7"}) == "203.0.113.7"


def test_server_kwargs_passthrough():
    s = Settings(trusted_proxies="198.51.100.1")
    kw = server_kwargs(s)
    assert kw["host"] == "0.0.0.0"
    assert kw["port"] == 8090
    assert kw["log_level"] == "info"


def test_untrusted_xff_config():
    """X-Forwarded-For is ignored from peers that are not trusted proxies."""
    s = Settings(trusted_proxies="198.51.100.100")
    # from an untrusted peer the spoofed header is ignored; from the trusted proxy it is applied
    assert _client_seen(s, "198.51.100.5", {"X-Forwarded-For": "203.0.113.8"}) == "198.51.100.5"
    assert _client_seen(s, "198.51.100.100", {"X-Forwarded-For": "203.0.113.8"}) == "203.0.113.8"
