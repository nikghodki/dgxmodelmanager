from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import AuditLog
from dgxmm.deployer.service import DeployRequest
from dgxmm.memory.budget import compute_budget
from dgxmm.memory.hostmem import HostMemory
from dgxmm.watchdog import Watchdog

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def budget(settings, avail_gib):
    return compute_budget(HostMemory(121 * GIB, int(avail_gib * GIB), 0, 0, 0, 0), [], settings)


async def test_levels_and_events(stack):
    q = stack.events.subscribe()
    w = Watchdog(stack.settings, stack.deployer, stack.events, stack.db)
    assert await w.check(budget(stack.settings, 50)) == "ok"
    assert await w.check(budget(stack.settings, 11)) == "warning"
    assert await w.check(budget(stack.settings, 5)) == "critical"
    topics = []
    while not q.empty():
        topics.append(q.get_nowait())
    alerts = [d["level"] for t, d in topics if t == "alert"]
    assert alerts == ["warning", "critical"]
    assert w.state()["level"] == "critical"


async def test_auto_protect_stops_lowest_priority_once_per_minute(stack):
    stack.http.all_healthy = True
    lo = await stack.deployer.deploy(DeployRequest(name="lo", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, priority=0), 1)
    hi = await stack.deployer.deploy(DeployRequest(name="hi", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, priority=50,
                                                 alias="hi"), 1)  # distinct LiteLLM model names (same model twice)
    for d in (lo, hi):
        await stack.deployer.wait_ready(d["id"], 5)
    off = Watchdog(stack.settings, stack.deployer, stack.events, stack.db)
    await off.check(budget(stack.settings, 3))
    assert stack.deployer.view(lo["id"])["state"] == "ready"
    stack.settings.auto_protect = True
    now = [1000.0]
    on = Watchdog(stack.settings, stack.deployer, stack.events, stack.db, clock=lambda: now[0])
    await on.check(budget(stack.settings, 3))
    assert stack.deployer.view(lo["id"])["state"] == "stopped"
    await on.check(budget(stack.settings, 3))
    assert stack.deployer.view(hi["id"])["state"] == "ready"  # rate-limited
    now[0] += 61
    await on.check(budget(stack.settings, 3))
    assert stack.deployer.view(hi["id"])["state"] == "stopped"
    with stack.db.session() as s:
        assert [a.target for a in s.scalars(select(AuditLog).where(AuditLog.action == "watchdog_stop"))] == ["lo", "hi"]
