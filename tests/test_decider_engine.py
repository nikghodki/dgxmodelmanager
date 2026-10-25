"""The "decision" engine and system deployments: spec, estimate, health, and exclusion from plans, watchdog,
LiteLLM, metrics scraping, model-name holding and reconcile."""
import json
import os
import stat

import httpx
import pytest

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployError, DeployRequest
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.decision import DECIDER_MODELS, DECIDER_REPO, DECIDER_REVISION, DecisionEngine
from dgxmm.planner.fit import check_fit
from dgxmm.planner.plans import PlanAction
from tests.conftest import QWEN_TINY

D3_CONFIG = {"architectures": ["Qwen3_5Model"], "model_type": "qwen3_5", "dtype": "bfloat16"}
SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


@pytest.fixture
def d3(stack, fake_hf_repo):
    fake_hf_repo(DECIDER_REPO, rev=DECIDER_REVISION, config=D3_CONFIG,
                 files={"model-00001-of-00002.safetensors": 5 * GIB, "model-00002-of-00002.safetensors": 4 * GIB,
                        "d3_server.py": 100})
    fake_hf_repo("vllm-sr/d3-other", rev="b" * 40, config=D3_CONFIG, files={"model.safetensors": GIB})
    stack.inv.refresh()
    return stack


async def deploy_decider(stack, name="decider"):
    return await stack.deployer.deploy(DeployRequest(name=name, engine="decision", model_ref=DECIDER_REPO, config={},
                                                     priority=100), 1)


def test_supports_only_the_pinned_allowlisted_snapshot(d3):
    eng = d3.deployer.engines["decision"]
    assert eng.supports(d3.inv.get(DECIDER_REPO))
    assert not eng.supports(d3.inv.get("vllm-sr/d3-other"))
    assert not eng.supports(d3.inv.get("Qwen/Qwen3-Tiny"))


def test_fixed_13gib_estimate_without_calibration(d3):
    class Cal:
        def adjust(self, *a):
            raise AssertionError("calibration must not touch the fixed reservation")
    d3.deployer.calibration = Cal()
    prep = d3.deployer.prepare("decision", DECIDER_REPO, {})
    assert prep.estimate.total == DECIDER_MODELS[DECIDER_REPO].reservation_bytes == 13 * GIB
    assert prep.estimate.margin == 0 and prep.estimate.confidence == "measured"


async def test_fit_applies_the_safety_margin(d3):
    prep = d3.deployer.prepare("decision", DECIDER_REPO, {})
    reserve = d3.settings.margin_bytes + d3.settings.os_reserve_bytes
    d3.probe.available = 13 * GIB + reserve
    assert check_fit(prep.estimate, await d3.budget.refresh()).fits
    d3.probe.available = 13 * GIB + reserve - 1  # the safety margin and OS reserve still apply
    assert not check_fit(prep.estimate, await d3.budget.refresh()).fits


def test_config_is_minimal(d3):
    eng = d3.deployer.engines["decision"]
    assert eng.validate_config({}).batch_size == 8
    with pytest.raises(Exception):
        eng.validate_config({"extra_args": ["--model", "x"]})
    with pytest.raises(Exception):
        eng.validate_config({"batch_size": 0})


def test_container_spec_gpu_ipc_ro_cache_offline_and_generated_secret(d3):
    eng = d3.deployer.engines["decision"]
    art = d3.inv.get(DECIDER_REPO)
    spec = eng.container_spec(art, eng.validate_config({}), None, LaunchContext(7, "decider", 8123, 13 * GIB, 121 * GIB),
                              d3.settings)
    assert spec.image == "dgxmm-decider:3" and spec.gpus and not spec.ipc_host
    assert spec.command == ["--port", "8123", "--batch-size", "8"]
    # only the d3-mini repo dir (snapshot + blobs), never the whole HF cache (the host HF token lives there)
    repo = "/hub/models--vllm-sr--d3-mini"
    assert [(m.host, m.target, m.mode) for m in spec.mounts if m.mode == "ro"] == [
        (d3.settings.hf_cache.host + repo, "/models/hf" + repo, "ro")]
    assert all(m.target != "/models/hf" for m in spec.mounts)
    # no mount shared with vLLM containers (their ~/.triton holds launcher .so files they load)
    shared = {m.host for m in d3.settings.vllm_cache_mounts}
    assert not shared & {m.host for m in spec.mounts}
    # docker never expands "~": every bind source must be an absolute host path; nothing is mounted read-write
    assert all(os.path.isabs(m.host) and "~" not in m.host for m in spec.mounts)
    assert len(spec.mounts) == 1 and all(m.mode == "ro" for m in spec.mounts)
    assert spec.environment["HF_HUB_OFFLINE"] == "1"
    key_path = d3.settings.decider_key_path
    key = key_path.read_text()
    assert spec.secret_files == {"/run/dgxmm-decider/api_key": key.encode()} and len(key) >= 32
    assert "DECISION_API_KEY" not in spec.environment
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600
    assert spec.labels["dgxmm.system"] == "true" and spec.labels["dgxmm.engine"] == "decision"
    # second spec reuses the same key
    again = eng.container_spec(art, eng.validate_config({}), None, LaunchContext(7, "decider", 8123, 1, 1), d3.settings)
    assert again.secret_files["/run/dgxmm-decider/api_key"] == key.encode()


async def test_health_needs_ready_then_authenticated_probe(d3):
    eng = d3.deployer.engines["decision"]
    from dgxmm.decider.secret import ensure_key
    key = ensure_key(d3.settings)
    seen = []

    def handler(req):
        seen.append((req.url.path, req.headers.get("authorization")))
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        q = json.loads(req.content)["questions"]
        if "q" in q:  # a warm-up choice question: must get a valid answer (parse_answer)
            k = next(iter(q["q"]["criteria"]))
            return httpx.Response(200, json={"answers": {"q": {"choice": k, "probabilities": {k: 1.0}}}})
        return httpx.Response(200, json={"answers": {"probe": {"noul": 0.8}}})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await eng.health(http, "http://x:1", "d3-mini")
    # /health, the probe, then the warm-up questions (engines/decision.py warmup_requests), all with the key
    from dgxmm.engines.decision import warmup_requests
    assert seen == [("/health", None)] + [("/v1/systemone", f"Bearer {key}")] * (1 + len(warmup_requests()))

    def loading(req):
        return httpx.Response(200, json={"status": "loading"})

    assert not await eng.health(httpx.AsyncClient(transport=httpx.MockTransport(loading)), "http://x:1", "d3-mini")

    def not_a_dict(req):
        return httpx.Response(200, json=["ready"])

    assert not await eng.health(httpx.AsyncClient(transport=httpx.MockTransport(not_a_dict)), "http://x:1", "d3-mini")


async def test_deploy_marks_system_and_view_shows_it(d3):
    dep = await deploy_decider(d3)
    assert dep["system"] is True and dep["engine"] == "decision"
    run = d3.sdk.run_calls[-1]
    assert run["device_requests"] and "ipc_mode" not in run
    assert d3.settings.decider_key_path.read_text() not in str(dep)  # secret never in the view
    d3.http.healthy.add(dep["port"])
    assert await d3.deployer.wait_ready(dep["id"], 5) == "ready"
    assert d3.http.systemone_auth[-1] == f"Bearer {d3.settings.decider_key_path.read_text()}"
    other = await d3.deployer.deploy(DeployRequest(mode="always_on", name="tiny", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                   config=SMALL), 1)
    assert other["system"] is False


async def test_only_one_decider_deployment(d3):
    await deploy_decider(d3)
    with pytest.raises(DeployError) as e:
        await deploy_decider(d3, "decider2")
    assert e.value.code == "decider_exists" and e.value.status == 409


async def test_decider_does_not_hold_a_litellm_model_name(d3):
    await deploy_decider(d3)
    assert d3.deployer.model_name_holder("d3-mini", None) is None


async def test_system_deployment_never_a_plan_candidate(d3):
    d3.http.all_healthy = True
    dec = await deploy_decider(d3)
    tiny = await d3.deployer.deploy(DeployRequest(mode="always_on", name="tiny", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={**SMALL, "enable_sleep_mode": True}), 1)
    for d in (dec, tiny):
        assert await d3.deployer.wait_ready(d["id"], 5) == "ready"
    assert [a.id for a in d3.deployer.active_deps()] == [tiny["id"]]
    d3.probe.available = 20 * GIB
    p = await d3.deployer.preview("vllm", "org/big-model", SMALL)
    assert p["plans"]
    for plan in p["plans"]:
        assert all(a["deployment_id"] != dec["id"] for a in plan["actions"])
    with pytest.raises(DeployError) as e:
        await d3.deployer.apply_actions([PlanAction(type="stop", deployment_id=dec["id"])])
    assert e.value.code == "system_deployment"
    assert d3.deployer.view(dec["id"])["state"] == "ready"


async def test_watchdog_stops_the_decider_before_user_deployments(d3):
    # the decider is advisory and fails open: it is the first auto-protect victim (tests/test_watchdog_decider.py)
    from dgxmm.memory.budget import compute_budget
    from dgxmm.memory.hostmem import HostMemory
    from dgxmm.watchdog import Watchdog
    d3.http.all_healthy = True
    d3.settings.auto_protect = True
    dec = await deploy_decider(d3)
    assert await d3.deployer.wait_ready(dec["id"], 5) == "ready"
    w = Watchdog(d3.settings, d3.deployer, d3.events, d3.db)
    await w.check(compute_budget(HostMemory(121 * GIB, 3 * GIB, 0, 0, 0, 0), [], d3.settings))
    assert d3.deployer.view(dec["id"])["state"] == "stopped" and "memory critical" in w.action


async def test_litellm_never_registers_the_decider(d3):
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    from tests.fakes import FakeLiteLLM
    d3.http.all_healthy = True
    dec = await deploy_decider(d3)
    tiny = await d3.deployer.deploy(DeployRequest(mode="always_on", name="tiny", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=SMALL), 1)
    for d in (dec, tiny):
        await d3.deployer.wait_ready(d["id"], 5)
    fake = FakeLiteLLM()
    sync = LiteLLMSync(LiteLLMClient.fixed("http://litellm:4000", "sk-test", fake.client()), d3.deployer, d3.db,
                       d3.events)
    assert [k[0] for k in sync.desired()] == [str(tiny["id"])]
    assert await sync.sync()
    names = [m["model_name"] for m in fake.models]
    assert "d3-mini" not in names and "qwen3-tiny" in names
    assert d3.deployer.view(dec["id"])["litellm_status"] == "none"
    assert d3.deployer.view(tiny["id"])["litellm_status"] == "synced"


async def test_metrics_scraper_skips_the_decider(d3):
    from dgxmm.metrics.scraper import MetricsScraper
    from dgxmm.metrics.store import MetricsStore
    d3.http.all_healthy = True
    dec = await deploy_decider(d3)
    await d3.deployer.wait_ready(dec["id"], 5)
    scraper = MetricsScraper(d3.settings, d3.deployer, d3.deployer.http, MetricsStore(d3.db), d3.events, d3.budget)
    await scraper.scrape_once()
    assert not [c for c in d3.http.calls if c[2] == "/metrics" and c[1] == dec["port"]]


async def test_reconcile_records_an_orphan_decider_as_system(d3):
    from dgxmm.deployer.reconcile import reconcile
    from dgxmm.engines.base import managed_labels
    d3.sdk.add_unmanaged("dgxmm-decider", image="dgxmm-decider:3", command=["--port", "8150", "--batch-size", "8"],
                         labels={**managed_labels(99, "decision"), "dgxmm.system": "true"}, ports=[(8150, "127.0.0.1")])
    await reconcile(d3.deployer)
    v = next(v for v in d3.deployer.list_views() if v["engine"] == "decision")
    assert v["system"] is True and v["model_ref"] == DECIDER_REPO and v["port"] == 8150


def test_decider_not_adoptable(d3):
    from dgxmm.deployer.adopt import list_adoptable
    d3.sdk.add_unmanaged("hand-decider", image="dgxmm-decider:3", command=["--port", "8000"])
    assert all(c["name"] != "hand-decider" for c in list_adoptable(d3.deployer))


def test_engines_api_lists_decision_as_system(api_stack):
    from tests.test_api_deployments import admin
    admin(api_stack)
    eng = {e["name"]: e for e in api_stack.get("/dgxmm/api/engines").json()}
    assert eng["decision"]["system"] is True and eng["vllm"]["system"] is False
    assert eng["decision"]["image"] == "dgxmm-decider:3"


def test_image_pins_snapshot_hashes_and_ignores_snapshot_pycache():
    from dgxmm.decider.image import CONTEXT_FILES, default_build_dir
    d = default_build_dir()
    hashes = d / f"d3-mini-{DECIDER_REVISION}.sha256"
    assert hashes.name in CONTEXT_FILES
    names = [line.split()[1] for line in hashes.read_text().splitlines()]
    for must in ("d3_server.py", "d3_runtime.py", "d3_fast.py", "modeling_d3.py", "MODEL_MANIFEST.json",
                 "config.json", "decision_config.json", "readout.safetensors", "tokenizer.json"):
        assert must in names
    assert all(len(line.split()[0]) == 64 for line in hashes.read_text().splitlines())
    ep = (d / "entrypoint.sh").read_text()
    # launch.py verifies private copies of the pinned files (tests/test_decider_launch.py)
    assert '--hashes "$hashes"' in ep and "exec python3 -I -B" in ep and "pycache_prefix=/tmp/pyc" in ep
    df = (d / "Dockerfile").read_text()
    assert "PYTHONPYCACHEPREFIX=/tmp/pyc" in df and "PYTHONDONTWRITEBYTECODE=1" in df
    assert hashes.name in df


def test_host_path_is_absolute_and_expanded(monkeypatch):
    from dgxmm.engines.decision import host_path
    monkeypatch.setenv("DGXMM_HOST_HOME", "/home/u")
    assert host_path("~/.cache/huggingface/hub/x") == "/home/u/.cache/huggingface/hub/x"
    with pytest.raises(ValueError):
        host_path("relative/dir")
