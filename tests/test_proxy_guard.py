"""L3: DGXMM joins the decider's internal network, so the decider can reach DGXMM's TCP listener. A peer on that
network is never a trusted proxy, whatever trusted_proxies says: its X-Forwarded-* headers are dropped."""
import ipaddress

from fastapi.testclient import TestClient

from dgxmm.__main__ import server_kwargs
from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.config import Settings
from dgxmm.proxy_headers import GuardedProxyHeaders, in_networks, parse_networks, tcp_app

DECIDER_NET = [ipaddress.ip_network("172.30.0.0/16")]
BROAD = "172.16.0.0/12"  # a broad trusted_proxies that also covers the decider network


def echo_app():
    seen = []

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        seen.append((scope["client"][0], scope["scheme"], {k.decode(): v.decode() for k, v in scope["headers"]}))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    return app, seen


def call(app, peer, headers):
    c = TestClient(app, base_url="http://testserver", client=(peer, 40000))
    assert c.get("/", headers=headers).status_code == 200


def test_decider_peer_is_never_a_proxy_even_when_trusted_proxies_covers_it():
    inner, seen = echo_app()
    g = GuardedProxyHeaders(inner, BROAD, lambda: DECIDER_NET)
    call(g, "172.30.0.7", {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https",
                           "Forwarded": "for=203.0.113.9", "X-Real-IP": "203.0.113.9"})
    host, scheme, headers = seen[-1]
    assert host == "172.30.0.7" and scheme == "http"
    assert not {"x-forwarded-for", "x-forwarded-proto", "forwarded", "x-real-ip"} & set(headers)


def test_other_trusted_proxies_still_work_and_untrusted_peers_are_ignored():
    inner, seen = echo_app()
    g = GuardedProxyHeaders(inner, BROAD, lambda: DECIDER_NET)
    call(g, "172.18.0.4", {"X-Forwarded-For": "203.0.113.9", "X-Forwarded-Proto": "https"})  # e.g. nginx
    assert seen[-1][:2] == ("203.0.113.9", "https")
    call(g, "198.51.100.20", {"X-Forwarded-For": "203.0.113.9"})  # not a trusted proxy
    assert seen[-1][0] == "198.51.100.20"


def test_the_decider_network_cannot_dodge_the_per_ip_login_limit(api_stack):
    api_stack.ctx.login_limiter = SlidingWindowLimiter(5, 60)
    c = TestClient(GuardedProxyHeaders(api_stack.app, BROAD, lambda: DECIDER_NET), base_url="http://testserver",
                   client=("172.30.0.7", 40000))
    codes = [c.post("/dgxmm/api/auth/login", headers={"X-Forwarded-For": f"203.0.113.{i}"},
                    json={"username": f"nobody{i}", "password": "wrong-password-123"}).status_code for i in range(7)]
    assert codes[:5] == [401] * 5 and codes[5:] == [429, 429]


def test_networks_parsing_and_matching():
    nets = parse_networks(["172.30.0.0/16", "fd00:30::/64", "bogus"])
    assert len(nets) == 2
    assert in_networks("172.30.1.2", nets) and in_networks("::ffff:172.30.1.2", nets) and in_networks("fd00:30::5", nets)
    assert not in_networks("172.31.0.1", nets) and not in_networks(None, nets) and not in_networks("testclient", nets)


def test_tcp_listener_uses_the_guard_and_uvicorn_proxy_headers_are_off():
    s = Settings(trusted_proxies=BROAD)
    kw = server_kwargs(s)
    assert kw["proxy_headers"] is False and "forwarded_allow_ips" not in kw

    class Ctx:
        untrusted_proxy_networks = DECIDER_NET
    g = tcp_app(object(), s, Ctx())
    assert isinstance(g, GuardedProxyHeaders) and g.untrusted() == DECIDER_NET
    assert g.proxied.trusted_hosts.always_trust is False


def test_admin_socket_serves_the_bare_app_and_tcp_the_guarded_one(tmp_path):
    from dgxmm.admin_listener import AdminListener, DualServer
    app, guarded = object(), object()
    d = DualServer(app, {"host": "127.0.0.1", "port": 0, "proxy_headers": False}, tmp_path / "admin.sock",
                   main_app=guarded)
    assert d.main.config.app is guarded
    assert isinstance(d.admin.config.app, AdminListener) and d.admin.config.app.app is app


def test_decider_network_subnets_are_read_from_docker(stack):
    stack.sdk.add_network("dgxmm-decider", internal=True)
    stack.sdk.networks.items["dgxmm-decider"].attrs["IPAM"] = {"Config": [{"Subnet": "172.30.0.0/16",
                                                                           "Gateway": "172.30.0.1"}]}
    assert stack.deployer.docker.network_subnets("dgxmm-decider") == ["172.30.0.0/16"]
    assert stack.deployer.docker.network_subnets("missing") == [] and stack.deployer.docker.network_subnets("") == []
