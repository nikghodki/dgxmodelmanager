"""POST /api/deployments/preview: admin-only, and its plan ranking is under the decider rate limit (M2)."""
from __future__ import annotations

from dgxmm.api.ratelimit import RATE_LIMITS
from dgxmm.decider import plans as dp
from tests.fakes import FakeDecider
from tests.test_decider_plans import idle, plan, rec

PLANS = [plan("kv", patch={"kv_cache_dtype": "fp8"}), plan("stop", actions=[("stop", 2)])]


def test_viewer_gets_403_on_preview(api_stack):
    from tests.test_api_deployments import SMALL, admin
    admin(api_stack)
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    h = {"X-CSRF-Token": r.json()["csrf_token"]}
    r = api_stack.post("/dgxmm/api/deployments/preview", headers=h,
                       json={"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required"


async def test_over_the_rate_limit_the_ranking_falls_back_to_rules_without_asking_the_model():
    fake = FakeDecider(answers={"plan": ("kv", 0.9)})
    out = await dp.recommend(PLANS, deps={2: idle(2, "old")}, decider=fake, shortfall=1, model_ref="org/new",
                             engine="vllm", cfg={}, raw_cfg={}, defaults={}, context_key="max_model_len",
                             seqs_key="max_num_seqs", cache=dp._Cache(), consult=lambda: False)
    assert fake.calls == [] and not any(p["recommended"] for p in out)
    asked = []
    await dp.recommend(PLANS, deps={2: idle(2, "old")}, decider=fake, shortfall=1, model_ref="org/new",
                       engine="vllm", cfg={}, raw_cfg={}, defaults={}, context_key="max_model_len",
                       seqs_key="max_num_seqs", cache=dp._Cache(), consult=lambda: asked.append(1) or True)
    assert asked == [1] and len(fake.calls) == 1


async def test_rules_answers_are_recorded_once_per_cache_period():
    fake = FakeDecider()
    cache = dp._Cache()
    for _ in range(5):
        out = await rec([plan("stop", actions=[("stop", 2)])], {2: idle(2, "old")}, fake, cache=cache)
        assert out[0]["recommended"] and out[0]["recommendation"]["source"] == "rules"
    assert len(fake.rules) == 1


def test_preview_ranking_uses_the_decider_plan_bucket(api_stack, monkeypatch):
    """Every preview that would ask the model takes one request from the admin's `decider_plan` bucket."""
    from tests.test_api_deployments import SMALL, admin
    h = admin(api_stack)
    per_user = RATE_LIMITS["decider_plan"][0]
    seen = []

    async def fake_preview(engine, model_ref, cfg, replacing=None, priority=None, consult=None):
        seen.append(consult())
        return {"plans": []}

    monkeypatch.setattr(api_stack.ctx.deployer, "preview", fake_preview)
    for _ in range(per_user + 2):
        r = api_stack.post("/dgxmm/api/deployments/preview", headers=h,
                           json={"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
        assert r.status_code == 200  # never 429: the ranking falls back to rules
    assert seen == [True] * per_user + [False, False]
