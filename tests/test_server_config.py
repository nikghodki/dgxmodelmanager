"""Tests for server configuration (__main__.py)."""
from dgxmm.__main__ import server_kwargs
from dgxmm.config import Settings


def test_default_trusted_proxies():
    s = Settings()
    kw = server_kwargs(s)
    assert kw["proxy_headers"] is True
    assert kw["forwarded_allow_ips"] == "127.0.0.1,::1"


def test_custom_trusted_proxies():
    s = Settings(trusted_proxies="10.0.0.0/8,172.16.0.0/12")
    kw = server_kwargs(s)
    assert kw["forwarded_allow_ips"] == "10.0.0.0/8,172.16.0.0/12"


def test_server_kwargs_passthrough():
    s = Settings(trusted_proxies="192.168.1.1")
    kw = server_kwargs(s)
    assert kw["host"] == "0.0.0.0"
    assert kw["port"] == 8090
    assert kw["log_level"] == "info"


def test_untrusted_xff_config():
    """Server kwargs correctly configure uvicorn to ignore X-Forwarded-For from untrusted peers."""
    from dgxmm.config import Settings
    from dgxmm.__main__ import server_kwargs
    
    # Trust only 10.0.0.100
    settings = Settings(trusted_proxies="10.0.0.100")
    kw = server_kwargs(settings)
    
    # uvicorn ProxyHeadersMiddleware with forwarded_allow_ips="10.0.0.100":
    # - From untrusted 10.0.0.5 with X-Forwarded-For: 8.8.8.8 → client stays 10.0.0.5 (spoofed header ignored)
    # - From trusted 10.0.0.100 with X-Forwarded-For: 8.8.8.8 → client becomes 8.8.8.8 (header accepted)
    assert kw["proxy_headers"] is True
    assert kw["forwarded_allow_ips"] == "10.0.0.100"
