"""I4 (revised by spec §5): viewers see host process command lines only in REDACTED form (GET /api/host, the
`budget` SSE event and GET /api/host/processes); /proc environ is never read or returned."""
from pathlib import Path

from dgxmm.api.host import host_payload
from dgxmm.memory.hostmem import HostProbe
from tests.test_api_deployments import admin
from tests.test_hostmem import MEMINFO, make_proc

SECRET = "s3cr3t-api-key-value"


def test_probe_returns_only_redacted_cmdline(settings):
    proc = Path(settings.host_proc)
    (proc / "meminfo").write_text(MEMINFO)
    make_proc(proc, 200, "python", 1_000_000, "/user.slice/session-3.scope")
    (proc / "200" / "cmdline").write_bytes(b"python\0serve.py\0--api-key\0" + SECRET.encode() + b"\0")
    [p] = HostProbe(settings, None).sample(set()).external_top
    assert set(p.__dict__) == {"pid", "name", "rss", "gpu", "command", "user", "container"}
    assert p.command == "python serve.py --api-key ***"


def test_host_payload_and_api_return_redacted_commands_to_viewers(api_stack):
    from dgxmm.memory.hostmem import HostSample, ProcMem
    admin(api_stack)
    api_stack.get("/dgxmm/api/host")  # populates budget
    stack = api_stack.stack
    p = ProcMem(pid=1, name="python", rss=10, gpu=0, command="python x.py --token ***", user="root",
                container={"id": "abc", "name": "litellm-proxy", "image": "litellm"})
    stack.budget.last_sample = HostSample(mem=stack.probe.meminfo(), managed={}, external_top=[p])
    payload = host_payload(api_stack.ctx, stack.budget.current())
    assert payload["external_top"] == [{"pid": 1, "name": "python", "rss": 10, "gpu": 0, "command": "python x.py --token ***",
                                        "user": "root", "container": {"id": "abc", "name": "litellm-proxy", "image": "litellm"}}]
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert r.status_code == 200
    body = api_stack.get("/dgxmm/api/host").json()
    assert body["external_top"][0]["command"] == "python x.py --token ***"
    assert "environ" not in str(body)
