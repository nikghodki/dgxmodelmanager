"""Security re-review, fix round 1: overflowing probabilities, capped health / probe / warm-up replies, the
watchdog's fallback when the decider cannot be stopped, and the auto_protect default in the docs."""
import json
from pathlib import Path

import httpx
import pytest

from dgxmm.config import GIB
from dgxmm.decider.client import MAX_REPLY_BYTES, DeciderError, probe
from dgxmm.deployer.service import DeployRequest
from dgxmm.memory.budget import compute_budget
from dgxmm.memory.hostmem import HostMemory
from tests.test_decider_client import OPTS, make
from tests.test_decider_engine import d3  # noqa: F401  (fixture)
from tests.test_decider_supervisor import running_decider, sup  # noqa: F401  (fixture)
from tests.test_watchdog import SMALL
from tests.test_watchdog_decider import watchdog_for

ROOT = Path(__file__).resolve().parents[1]


async def test_a_huge_integer_probability_is_a_bad_reply_counted_by_the_breaker():
    body = ('{"answers": {"q": {"choice": "gpu_oom", "probabilities": {"gpu_oom": 1' + "0" * 400 + '}}}}').encode()
    c = make(lambda req: httpx.Response(200, content=body))
    with pytest.raises(DeciderError) as e:
        await c.choose(feature="diagnosis", state="s", question="q", options=OPTS)
    assert e.value.code == "bad_reply" and c.breaker.failures == 1


def _streamed(sent):
    async def gen():
        for _ in range(1000):
            sent.append(1)
            yield b" " * 65536
    return gen()


@pytest.mark.parametrize("path", ["/health", "/v1/systemone"])
async def test_client_health_reads_capped_replies(path):
    sent = []

    def handler(req):
        if req.url.path == path:
            return httpx.Response(200, content=_streamed(sent))
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json={"answers": {"probe": {"noul": 0.9}}})
    c = make(handler)
    assert await c.health() is False and len(sent) <= 3


@pytest.mark.parametrize("content", [b'{"answers": {"probe": {"noul": NaN}}}', b'{"answers": "probe"}', b"[1]",
                                     b'{"answers": {"probe": 1}}'])
async def test_probe_parses_strictly(content):
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=content)))
    assert await probe(http, "http://d:1", "k") is False


def _engine_handler(warm):
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        q = json.loads(req.content)["questions"]
        if "probe" in q:
            return httpx.Response(200, json={"answers": {"probe": {"noul": 0.9}}})
        return warm(q["q"])
    return handler


GOOD = lambda q: httpx.Response(200, json={"answers": {"q": {"choice": next(iter(q["criteria"])),  # noqa: E731
                                                              "probabilities": {next(iter(q["criteria"])): 1.0}}}})


@pytest.mark.parametrize("warm", [
    lambda q: httpx.Response(200, content=b'{"answers": {"q": {"choice": "a", "probabilities": {"a": NaN}}}}'),
    lambda q: httpx.Response(200, json={"answers": {"q": {"choice": "nope", "probabilities": {"nope": 1.0}}}}),
    lambda q: httpx.Response(200, json={"answers": {"q": {}}}),
    lambda q: httpx.Response(200, content=b"x" * (MAX_REPLY_BYTES + 1)),
], ids=["nan", "unoffered", "empty", "oversized"])
async def test_engine_warmup_validates_every_answer(d3, warm):  # noqa: F811
    eng = d3.deployer.engines["decision"]
    from dgxmm.decider.secret import ensure_key
    ensure_key(d3.settings)
    good = httpx.AsyncClient(transport=httpx.MockTransport(_engine_handler(GOOD)))
    assert await eng.health(good, "http://x:1", "d3-mini") is True
    bad = httpx.AsyncClient(transport=httpx.MockTransport(_engine_handler(warm)))
    assert await eng.health(bad, "http://x:1", "d3-mini") is False


async def test_engine_health_reads_a_capped_health_reply(d3):  # noqa: F811
    sent = []
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=_streamed(sent))))
    assert await d3.deployer.engines["decision"].health(http, "http://x:1", "d3-mini") is False and len(sent) <= 3


async def test_watchdog_falls_back_to_user_deployments_when_the_decider_cannot_be_stopped(sup, monkeypatch):  # noqa: F811
    st = sup.stack
    view = await running_decider(sup)
    st.http.all_healthy = True
    lo = await st.deployer.deploy(DeployRequest(name="lo", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL,
                                                priority=0), 1)
    await st.deployer.wait_ready(lo["id"], 5)
    st.settings.auto_protect = True
    real_stop = st.deployer.stop

    async def stop(dep_id, *a, **kw):
        if dep_id == view["id"]:
            raise RuntimeError("docker refused")
        return await real_stop(dep_id, *a, **kw)
    monkeypatch.setattr(st.deployer, "stop", stop)
    now = [1000.0]
    w = watchdog_for(sup, clock=lambda: now[0])
    critical = compute_budget(HostMemory(121 * GIB, 3 * GIB, 0, 0, 0, 0), [], st.settings)
    await w.check(critical)
    assert "failed to stop" in w.action and st.deployer.view(lo["id"])["state"] == "ready"
    now[0] += 61
    await w.check(critical)  # the decider is skipped while its state is unchanged: the user order applies
    assert st.deployer.view(lo["id"])["state"] == "stopped"
    assert st.deployer.view(view["id"])["state"] == "ready"
    # once its state changes it is a candidate again
    w._unstoppable[view["id"]] = "loading"
    assert w._victim()["id"] == view["id"]


def test_security_doc_states_the_auto_protect_default_and_how_to_enable_it():
    doc = " ".join((ROOT / "docs" / "security.md").read_text().replace("**", "").split())
    assert "`auto_protect` defaults to `false`" in doc and "auto_protect: true" in doc
