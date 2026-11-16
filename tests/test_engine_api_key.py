"""Per-deployment engine API keys (multi-node Task 8): every vLLM / llama.cpp deployment gets its own key
(``deployments.engine_api_key``, migration 42), handed to the engine in its environment, and every DGXMM client of
the engine (health check, sleep/wake, metrics scraper, benchmark, on-demand gateway, LiteLLM) sends it as a bearer
token. The key is never part of a view, event, audit record or the decider's inputs."""
import json
import sqlite3

import httpx
import pytest
from sqlalchemy import create_engine, select

from dgxmm.db import AuditLog, Database, Deployment
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import LaunchContext, openai_health
from dgxmm.engines.llamacpp import LlamaCppConfig, LlamaCppEngine
from dgxmm.engines.vllm import VllmConfig, VllmEngine
from dgxmm.migrations import migrate
from tests.test_engines_llamacpp import gguf_art  # noqa: F401  (fixture)
from tests.test_engines_vllm import tiny  # noqa: F401  (fixture)

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
KEY = "k-0123456789abcdef"


def launch(api_key=None):
    return LaunchContext(1, "tiny", 8100, 10 * 1024 ** 3, 121 * 1024 ** 3, api_key=api_key)


def req(name="tiny", **kw):
    return DeployRequest(mode="always_on", name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, **kw)


def key_of(stack, dep_id):
    with stack.db.session() as s:
        return s.get(Deployment, dep_id).engine_api_key


def set_key(stack, dep_id, value):
    with stack.db.session() as s:
        s.get(Deployment, dep_id).engine_api_key = value


async def ready(stack, name="tiny", **kw):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req(name, **kw), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return d


def auth_on(stack, port, path):
    return [a for p, pa, a in stack.http.auth if p == port and pa == path]


# ---------- engines ----------

def test_vllm_spec_sets_api_key_env(settings, tiny):
    spec = VllmEngine(settings).container_spec(tiny, VllmConfig(), None, launch(KEY), settings)
    assert spec.environment["VLLM_API_KEY"] == KEY
    assert KEY not in json.dumps(spec.command)  # environment, never argv (ps, logs of the command line)
    assert "--api-key" not in spec.command


def test_vllm_spec_without_key_unchanged(settings, tiny):
    spec = VllmEngine(settings).container_spec(tiny, VllmConfig(), None, launch(None), settings)
    assert "VLLM_API_KEY" not in spec.environment


def test_llamacpp_spec_sets_api_key_env(settings, gguf_art):
    eng = LlamaCppEngine(settings)
    spec = eng.container_spec(gguf_art, LlamaCppConfig(), None, launch(KEY), settings)
    assert spec.environment["LLAMA_API_KEY"] == KEY
    assert KEY not in json.dumps(spec.command) and "--api-key" not in spec.command
    assert "LLAMA_API_KEY" not in eng.container_spec(gguf_art, LlamaCppConfig(), None, launch(None),
                                                      settings).environment


def _recording(seen):
    def handler(r: httpx.Request):
        seen.append((r.url.path, r.headers.get("authorization")))
        return httpx.Response(200, json={"choices": [{"text": "x"}]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_health_sends_bearer():
    seen = []
    assert await openai_health(_recording(seen), "http://e:8100", "m", api_key="k")
    assert seen == [("/health", "Bearer k"), ("/v1/completions", "Bearer k")]


async def test_health_without_key_sends_no_authorization():
    seen = []
    assert await openai_health(_recording(seen), "http://e:8100", "m")
    assert seen == [("/health", None), ("/v1/completions", None)]


async def test_vllm_sleep_wake_send_bearer(settings):
    seen = []
    eng, http = VllmEngine(settings), _recording(seen)
    await eng.sleep(http, "http://e:8100", api_key="k")
    await eng.wake(http, "http://e:8100", api_key="k")
    await eng.sleep(http, "http://e:8100")
    assert seen == [("/sleep", "Bearer k"), ("/wake_up", "Bearer k"), ("/sleep", None)]


# ---------- deployer: generation, persistence, delivery ----------

async def test_deploy_generates_key_and_starts_engine_with_it(stack):
    d = await ready(stack)
    key = key_of(stack, d["id"])
    assert key and len(key) >= 43  # secrets.token_urlsafe(32)
    run = stack.sdk.run_calls[-1]
    assert run["environment"]["VLLM_API_KEY"] == key
    assert key not in json.dumps(run["command"])
    assert set(auth_on(stack, d["port"], "/health")) == {f"Bearer {key}"}
    assert set(auth_on(stack, d["port"], "/v1/completions")) == {f"Bearer {key}"}
    other = await ready(stack, "other", alias="other")
    assert key_of(stack, other["id"]) not in (None, key)  # one key per deployment


@pytest.mark.usefixtures("vllm_level1_sleep")
async def test_key_kept_across_sleep_wake_stop_restart(stack):
    d = await ready(stack)
    key = key_of(stack, d["id"])
    await stack.deployer.sleep(d["id"])
    await stack.deployer.wake(d["id"])
    assert auth_on(stack, d["port"], "/sleep") == [f"Bearer {key}"]
    assert auth_on(stack, d["port"], "/wake_up") == [f"Bearer {key}"]
    await stack.deployer.stop(d["id"])
    await stack.deployer.restart(d["id"])
    assert key_of(stack, d["id"]) == key
    assert stack.sdk.run_calls[-1]["environment"]["VLLM_API_KEY"] == key


async def test_reconfigure_keeps_key(stack):
    d = await ready(stack)
    key = key_of(stack, d["id"])
    await stack.deployer.reconfigure(d["id"], {**SMALL, "max_model_len": 2048})
    assert key_of(stack, d["id"]) == key
    assert stack.sdk.run_calls[-1]["environment"]["VLLM_API_KEY"] == key


@pytest.mark.usefixtures("vllm_level1_sleep")
async def test_legacy_row_without_key_unchanged_until_relaunch(stack):
    """A deployment from before this version (NULL key): no Authorization header, until its next launch gives it
    a key."""
    d = await ready(stack)
    set_key(stack, d["id"], None)
    stack.http.auth.clear()
    await stack.deployer.sleep(d["id"])
    await stack.deployer.wake(d["id"])
    assert auth_on(stack, d["port"], "/sleep") == [None] and auth_on(stack, d["port"], "/wake_up") == [None]
    assert stack.deployer.api_key_for(stack.deployer.view(d["id"])) is None
    await stack.deployer.stop(d["id"])
    await stack.deployer.restart(d["id"])
    key = key_of(stack, d["id"])
    assert key and stack.sdk.run_calls[-1]["environment"]["VLLM_API_KEY"] == key
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    assert auth_on(stack, d["port"], "/v1/completions")[-1] == f"Bearer {key}"


async def test_system_deployment_gets_no_engine_key(stack, fake_hf_repo):
    from tests.test_decider_engine import D3_CONFIG, deploy_decider
    from dgxmm.engines.decision import DECIDER_REPO, DECIDER_REVISION
    from dgxmm.config import GIB
    fake_hf_repo(DECIDER_REPO, rev=DECIDER_REVISION, config=D3_CONFIG,
                 files={"model-00001-of-00002.safetensors": 5 * GIB, "model-00002-of-00002.safetensors": 4 * GIB,
                        "d3_server.py": 100})
    stack.inv.refresh()
    dec = await deploy_decider(stack)
    assert key_of(stack, dec["id"]) is None
    env = stack.sdk.run_calls[-1]["environment"] or {}
    assert "VLLM_API_KEY" not in env and "LLAMA_API_KEY" not in env


# ---------- clients ----------

async def test_metrics_scraper_sends_bearer(stack):
    from dgxmm.metrics.scraper import MetricsScraper
    from dgxmm.metrics.store import MetricsStore
    d = await ready(stack)
    legacy = await ready(stack, "legacy", alias="legacy")
    set_key(stack, legacy["id"], None)
    await MetricsScraper(stack.settings, stack.deployer, stack.http.client(), MetricsStore(stack.db), stack.events,
                         stack.budget).scrape_once()
    assert auth_on(stack, d["port"], "/metrics") == [f"Bearer {key_of(stack, d['id'])}"]
    assert auth_on(stack, legacy["port"], "/metrics") == [None]


async def test_benchmark_sends_bearer(db):
    from dgxmm.advisor.benchmark import BenchmarkRunner
    from dgxmm.events import EventBroker
    from tests.test_benchmark import sse_handler
    seen, auth = [], []
    inner = sse_handler(seen)

    def handler(r):
        auth.append(r.headers.get("authorization"))
        return inner(r)
    runner = BenchmarkRunner(db, EventBroker(),
                             http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await runner.run(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1",
                     served_name="q", levels=(1,), input_len=8, output_len=5, duration_s=5, max_requests=2,
                     api_key="k")
    assert auth and set(auth) == {"Bearer k"}


def test_benchmark_api_passes_the_key(api_stack):
    import time
    from tests.test_api_deployments import admin
    from tests.test_api_metrics import wire
    h = admin(api_stack)
    wire(api_stack)
    got = {}

    async def fake_run(**kw):
        got.update(kw)
        return []
    api_stack.ctx.benchmarks.run = fake_run
    api_stack.stack.http.all_healthy = True
    dep = api_stack.post("/dgxmm/api/deployments", headers=h,
                         json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    for _ in range(200):
        if api_stack.get(f"/dgxmm/api/deployments/{dep['id']}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    assert api_stack.post(f"/dgxmm/api/deployments/{dep['id']}/benchmark", headers=h,
                          json={"levels": [1]}).status_code == 202
    for _ in range(200):
        if got:
            break
        time.sleep(0.02)
    assert got["api_key"] == key_of(api_stack.stack, dep["id"])


async def test_litellm_registers_with_engine_key_and_replaces_keyless_entry(stack):
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync, key_fingerprint
    from tests.fakes import FakeLiteLLM
    d = await ready(stack)
    set_key(stack, d["id"], None)  # registered before the deployment had a key
    fake = FakeLiteLLM()
    sync = LiteLLMSync(LiteLLMClient.fixed("http://litellm:4000", "sk-test", fake.client()), stack.deployer,
                       stack.db, stack.events)
    assert await sync.sync()
    [entry] = [m for m in fake.models if m["model_info"].get("dgxmm_id")]
    assert entry["litellm_params"]["api_key"] == "none"
    set_key(stack, d["id"], KEY)  # its next launch gave it a key
    assert await sync.sync()
    [entry] = [m for m in fake.models if m["model_info"].get("dgxmm_id")]
    assert entry["litellm_params"]["api_key"] == KEY
    assert entry["model_info"]["dgxmm_key_fp"] == key_fingerprint(KEY) and KEY not in json.dumps(entry["model_info"])
    assert sync.last["registered"] == 1 and sync.last["removed"] == 1
    set_key(stack, d["id"], KEY + "-2")  # changed: replaced again
    assert await sync.sync()
    [entry] = [m for m in fake.models if m["model_info"].get("dgxmm_id")]
    assert entry["litellm_params"]["api_key"] == KEY + "-2"
    assert await sync.sync() and sync.last["registered"] == 0  # stable: nothing re-registered


# ---------- never serialised ----------

def test_api_key_never_in_deployment_api_response(api_stack):
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    q = api_stack.stack.events.subscribe()
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 201, r.text
    dep = r.json()
    key = key_of(api_stack.stack, dep["id"])
    assert key
    bodies = [r.text, api_stack.get(f"/dgxmm/api/deployments/{dep['id']}").text,
              api_stack.get("/dgxmm/api/deployments").text, api_stack.get("/dgxmm/api/audit").text]
    bodies += [json.dumps(q.get_nowait(), default=str) for _ in range(q.qsize())]
    for body in bodies:
        assert key not in body and "engine_api_key" not in body and "_api_key" not in body
    with api_stack.stack.db.session() as s:
        for a in s.scalars(select(AuditLog)):
            assert key not in json.dumps(a.detail, default=str)


async def test_api_key_never_in_decider_snapshot(stack):
    from dgxmm.decider.command import build_targets
    from dgxmm.decider.plans import facts_for
    d = await ready(stack)
    key = key_of(stack, d["id"])
    views = stack.deployer.list_views()
    snap = [repr(t) for t in build_targets(views, [], admin=True)]
    facts = facts_for(views, None, db=stack.db)
    for blob in (json.dumps(views, default=str), json.dumps(snap), json.dumps(facts, default=str),
                 json.dumps(stack.deployer.view(d["id"], with_events=True), default=str)):
        assert key not in blob


async def test_key_never_logged_during_lifecycle(stack, caplog):
    import logging
    caplog.set_level(logging.DEBUG)
    d = await ready(stack)
    key = key_of(stack, d["id"])
    await stack.deployer.sleep(d["id"])
    await stack.deployer.wake(d["id"])
    await stack.deployer.stop(d["id"])
    assert key not in caplog.text


# ---------- migration 42 ----------

def test_migration_42_adds_nullable_column(tmp_path):
    path = tmp_path / "old.sqlite3"
    Database(f"sqlite:///{path}").engine.dispose()
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("DELETE FROM schema_version WHERE version = 42")
    conn.execute("ALTER TABLE deployments DROP COLUMN engine_api_key")
    conn.execute("INSERT INTO deployments (id, name, engine, model_ref, config, state, priority, reserved_bytes, "
                 "estimate, litellm_status, port, created_at, updated_at) VALUES (1, 'old', 'vllm', 'Qwen/Qwen3-0.6B', "
                 "'{}', 'stopped', 0, 0, '{}', 'none', 8100, '2026-01-01 00:00:00', '2026-01-01 00:00:00')")
    conn.close()
    eng = create_engine(f"sqlite:///{path}")
    assert 42 in migrate(eng)
    eng.dispose()
    conn = sqlite3.connect(str(path))
    cols = {r[1]: r for r in conn.execute('PRAGMA table_info("deployments")')}
    assert "engine_api_key" in cols and cols["engine_api_key"][3] == 0  # nullable
    assert conn.execute("SELECT engine_api_key FROM deployments").fetchall() == [(None,)]
    conn.close()
