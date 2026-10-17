"""L1: the decider's answers are untrusted. Non-finite, negative or > 1 probabilities, sums far from 1, a choice that
is not the argmax, options that were not offered and oversized replies are errors (fail open, counted by the
breaker); stored legacy NaN / Infinity values never make an API answer fail."""
import json
import math

import httpx
import pytest

from dgxmm.db import DecisionRecord, DiagnosisRow
from dgxmm.decider.client import MAX_OPTIONS, MAX_REPLY_BYTES, DeciderError
from dgxmm.decider.common import safe_decide
from dgxmm.decider.types import Option
from tests.test_decider_client import OPTS, make, reply

KEYS = [o.key for o in OPTS]


def raw(choice, probs_text):
    return ('{"answers": {"q": {"type": "choice", "choice": "%s", "probabilities": %s}}}' % (choice, probs_text)).encode()


BAD = {
    "nan": raw("gpu_oom", '{"gpu_oom": NaN, "port_in_use": 0.1}'),
    "infinity": raw("gpu_oom", '{"gpu_oom": Infinity, "port_in_use": 0.1}'),
    "negative": raw("gpu_oom", '{"gpu_oom": 1.2, "port_in_use": -0.2}'),
    "above_one": raw("gpu_oom", '{"gpu_oom": 1.5}'),
    "sum_low": raw("gpu_oom", '{"gpu_oom": 0.4, "port_in_use": 0.1}'),
    "sum_high": raw("gpu_oom", '{"gpu_oom": 0.9, "port_in_use": 0.9}'),
    "not_argmax": raw("other", '{"gpu_oom": 0.99, "other": 0.01}'),
    "unoffered": raw("gpu_oom", '{"gpu_oom": 0.9, "evil": 0.1}'),
    "bool": raw("gpu_oom", '{"gpu_oom": true}'),
    "string": raw("gpu_oom", '{"gpu_oom": "1.0"}'),
    "huge_number": raw("gpu_oom", '{"gpu_oom": 1e999}'),
    "oversized": json.dumps({**reply(), "pad": "x" * (MAX_REPLY_BYTES + 10)}).encode(),
}


@pytest.mark.parametrize("name", sorted(BAD))
async def test_invalid_answers_are_bad_replies_counted_by_the_breaker(name):
    c = make(lambda req: httpx.Response(200, content=BAD[name], headers={"content-type": "application/json"}))
    with pytest.raises(DeciderError) as e:
        await c.choose(feature="diagnosis", state="s", question="q", options=OPTS)
    assert e.value.code == "bad_reply" and c.breaker.failures == 1


async def test_a_streamed_reply_is_cut_at_the_cap_without_reading_it_all():
    sent = []

    async def stream():
        for _ in range(1000):  # 64 MB if read to the end
            sent.append(1)
            yield b" " * 65536

    c = make(lambda req: httpx.Response(200, content=stream()))
    with pytest.raises(DeciderError) as e:
        await c.choose(feature="diagnosis", state="s", question="q", options=OPTS)
    assert e.value.code == "bad_reply" and len(sent) <= 3


async def test_confidence_is_the_chosen_options_probability_and_small_drift_is_renormalised():
    probs = {"gpu_oom": 0.6, "port_in_use": 0.405}  # sums to 1.005 (rounding)
    c = make(lambda req: httpx.Response(200, json=reply("gpu_oom", probs)))
    ans = await c.choose(feature="diagnosis", state="s", question="q", options=OPTS)
    assert ans.choice == "gpu_oom" and ans.confidence == pytest.approx(0.6 / 1.005)
    assert sum(ans.probabilities.values()) == pytest.approx(1.0) and c.breaker.failures == 0


async def test_too_many_options_are_refused_before_asking():
    c = make(lambda req: pytest.fail("must not be sent"))
    opts = [Option(f"o{i}", f"O{i}") for i in range(MAX_OPTIONS + 1)]
    with pytest.raises(DeciderError) as e:
        await c.choose(feature="diagnosis", state="s", question="q", options=opts)
    assert e.value.code == "bad_request"


async def test_an_invalid_answer_fails_open():
    from dgxmm.decider.service import Decider  # noqa: F401  (the real decider maps DeciderError to source none)
    c = make(lambda req: httpx.Response(200, content=BAD["infinity"]))

    class D:
        def feature_enabled(self, f):
            return True

        async def available(self):
            return True

        async def decide(self, **kw):
            await c.choose(feature=kw["feature"], state=kw["state"], question=kw["question"], options=kw["options"])

    d = await safe_decide(D(), feature="diagnosis", subject="x", state="s", question="q", options=OPTS)
    assert d.source == "none" and d.choice is None


def test_legacy_non_finite_rows_never_break_the_api(api_stack):
    from tests.test_api_deployments import SMALL, admin
    h = admin(api_stack)
    r = api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "tiny", "engine": "vllm",
                                                                    "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    dep_id = r.json()["id"]
    with api_stack.ctx.db.session() as s:
        s.add(DecisionRecord(feature="plan", subject="preview:x", source="model", choice="kv", confidence=math.inf,
                             probabilities={"kv": math.inf, "stop": math.nan}, latency_ms=math.nan))
        s.add(DiagnosisRow(deployment_id=dep_id, event_id=1, cause="gpu_oom", source="model", confidence=math.inf,
                           summary="", fix_hint="", log_excerpt=""))
    r = api_stack.get("/dgxmm/api/decider/decisions", headers=h)
    assert r.status_code == 200, r.text
    row = r.json()[0]
    assert row["confidence"] is None and row["probabilities"] == {} and row["latency_ms"] is None
    r = api_stack.get(f"/dgxmm/api/deployments/{dep_id}/diagnosis", headers=h)
    assert r.status_code == 200 and r.json()["diagnosis"]["confidence"] is None
    assert api_stack.get("/dgxmm/api/decider/status", headers=h).status_code == 200
