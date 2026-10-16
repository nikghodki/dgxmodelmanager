"""I4: viewers must not see host process command lines (GET /api/host and the `budget` SSE event)."""
from pathlib import Path

from dgxmm.api.host import host_payload
from dgxmm.memory.hostmem import HostProbe
from tests.test_api_deployments import admin
from tests.test_hostmem import MEMINFO, make_proc


def test_probe_does_not_collect_cmdline(settings):
    proc = Path(settings.host_proc)
    (proc / "meminfo").write_text(MEMINFO)
    make_proc(proc, 200, "python", 1_000_000, "/user.slice/session-3.scope")
    [p] = HostProbe(settings, None).sample(set()).external_top
    assert set(p.__dict__) == {"pid", "name", "rss", "gpu"}


def test_host_payload_and_api_have_no_cmdline(api_stack):
    from dgxmm.memory.hostmem import HostSample, ProcMem
    admin(api_stack)
    b = api_stack.get("/dgxmm/api/host")  # populates budget
    stack = api_stack.stack
    p = ProcMem(pid=1, name="python", rss=10, gpu=0)
    stack.budget.last_sample = HostSample(mem=stack.probe.meminfo(), managed={}, external_top=[p])
    payload = host_payload(api_stack.ctx, stack.budget.current())
    assert payload["external_top"] == [{"pid": 1, "name": "python", "rss": 10, "gpu": 0}]
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert r.status_code == 200
    body = api_stack.get("/dgxmm/api/host").json()
    assert "cmdline" not in str(body)
