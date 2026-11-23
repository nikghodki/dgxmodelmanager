"""A deployment's engine image chosen from the allowlist (images.<engine>_allowed; docs/security.md, "Engine
images"): config, enforcement in prepare and at container build, preview / deploy / switch / reconfigure / restart,
the pull on the node that runs it, the cluster refusal, the per-image architecture registry and the API."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dgxmm.catalog.support import VllmArchRegistry
from dgxmm.config import GIB, Images
from dgxmm.deployer import cluster
from dgxmm.deployer.service import DeployError, DeployRequest, Placement
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.base import LaunchContext, SpecError, engine_image
from dgxmm.engines.llamacpp import LlamaCppEngine
from dgxmm.engines.vllm import VllmEngine
from dgxmm.inventory.hf_scan import scan_hf_cache
from tests.conftest import QWEN_TINY
from tests.fakes import FakeDockerClient
from tests.test_api_deployments import SMALL, admin

D1, D2 = "sha256:" + "1" * 64, "sha256:" + "2" * 64
NEW_VLLM = f"vllm/vllm-openai:new-model@{D1}"
NEW_LLAMA = f"ghcr.io/ggml-org/llama.cpp:server-cuda-b99999@{D2}"


@pytest.fixture
def allowed(settings):
    settings.images.vllm_allowed = [NEW_VLLM]
    settings.images.llamacpp_allowed = [NEW_LLAMA]
    return settings


# ---------- config ----------
def test_allowlists_default_to_empty_and_the_default_image_is_unchanged():
    i = Images()
    assert i.vllm_allowed == [] and i.llamacpp_allowed == [] and i.allowed("vllm") == [] and i.allowed("decision") == []
    assert i.vllm == Images.model_fields["vllm"].default


@pytest.mark.parametrize("ref", ["vllm/vllm-openai:nightly", "vllm/vllm-openai", f"vllm/vllm-openai@sha256:{'1' * 63}",
                                 f"Vllm/Openai@{D1}", f"x;rm -rf /@{D1}", f"vllm/vllm-openai:nightly@md5:{'1' * 32}"])
def test_allowlist_entries_must_be_pinned_by_digest(ref):
    with pytest.raises(ValidationError, match="pinned by digest|vllm_allowed"):
        Images(vllm_allowed=[ref])


def test_allowlist_accepts_digest_pins_with_or_without_tag_and_dedups():
    refs = [NEW_VLLM, f"vllm/vllm-openai@{D2}", f"registry.example.com:5000/team/vllm:t1@{D1}", NEW_VLLM]
    assert Images(vllm_allowed=refs).vllm_allowed == refs[:3]


def test_engine_config_image_field_is_an_image_reference(settings):
    e = VllmEngine(settings)
    assert e.validate_config({"image": NEW_VLLM}).image == NEW_VLLM
    assert e.validate_config({}).image is None
    for bad in ("x;rm -rf /", "--privileged", "a" * 401, "vllm/vllm-openai:bad tag"):
        with pytest.raises(ValidationError):
            e.validate_config({"image": bad})
    with pytest.raises(ValidationError):
        LlamaCppEngine(settings).validate_config({"image": "$(id)"})


# ---------- engine_image ----------
def test_engine_image_default_allowed_refused_and_wrong_engine(allowed):
    s = allowed
    assert engine_image(s, "vllm", None) == s.images.vllm
    assert engine_image(s, "vllm", s.images.vllm) == s.images.vllm  # naming the default is the default
    assert engine_image(s, "vllm", NEW_VLLM) == NEW_VLLM
    assert engine_image(s, "llamacpp", NEW_LLAMA) == NEW_LLAMA
    with pytest.raises(SpecError) as e:
        engine_image(s, "vllm", f"evil/miner@{D1}")
    assert e.value.code == "image_not_allowed" and "images.vllm_allowed" in e.value.message
    with pytest.raises(SpecError):  # a llama.cpp image is not a vLLM one, and the other way round
        engine_image(s, "vllm", NEW_LLAMA)
    with pytest.raises(SpecError):
        engine_image(s, "llamacpp", NEW_VLLM)


# ---------- container specs ----------
@pytest.fixture
def tiny(settings, fake_hf_repo):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    [a] = scan_hf_cache(Path(settings.hf_cache.local) / "hub")
    return a


def _launch():
    return LaunchContext(deployment_id=1, name="t", port=8100, reserved=10 * GIB, mem_total=100 * GIB)


def test_vllm_spec_runs_the_chosen_image_and_never_passes_it_to_vllm(allowed, tiny):
    e = VllmEngine(allowed)
    spec = e.container_spec(tiny, e.validate_config({"image": NEW_VLLM}), None, _launch(), allowed)
    assert spec.image == NEW_VLLM
    assert not any("image" in a or NEW_VLLM in a for a in spec.command)
    assert e.container_spec(tiny, e.validate_config({}), None, _launch(), allowed).image == allowed.images.vllm


def test_spec_refuses_an_image_removed_from_the_allowlist_after_deploy(allowed, tiny):
    e = VllmEngine(allowed)
    cfg = e.validate_config({"image": NEW_VLLM})
    allowed.images.vllm_allowed = []
    with pytest.raises(SpecError) as err:
        e.container_spec(tiny, cfg, None, _launch(), allowed)
    assert err.value.code == "image_not_allowed"


def test_cluster_specs_refuse_a_chosen_image(allowed, tiny):
    e = VllmEngine(allowed)
    with pytest.raises(SpecError) as err:
        e.cluster_specs(tiny, e.validate_config({"image": NEW_VLLM}), None, _launch(), None, None, 29500, allowed)
    assert err.value.code == "image_single_node_only"


def test_dspark_speculative_config_reaches_vllm(settings, tiny):
    e = VllmEngine(settings)
    cfg = e.validate_config({"reasoning_parser": "inkling", "speculative": {
        "method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}})
    cmd = e.container_spec(tiny, cfg, None, _launch(), settings).command
    assert json.loads(cmd[cmd.index("--speculative-config") + 1]) == {
        "method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}
    assert cmd[cmd.index("--reasoning-parser") + 1] == "inkling"
    assert e.draft_ref(cfg) is None  # the drafter is bundled in the checkpoint
    with pytest.raises(ValidationError):
        e.validate_config({"speculative": {"method": "dspark", "draft_sample_method": "random"}})
    plain = e.validate_config({"speculative": {"method": "mtp", "num_speculative_tokens": 2}})
    cmd = e.container_spec(tiny, plain, None, _launch(), settings).command
    assert "draft_sample_method" not in cmd[cmd.index("--speculative-config") + 1]


# ---------- architecture registry per image ----------
async def test_registry_reads_every_allowed_image_and_answers_per_image(allowed):
    sdk = FakeDockerClient()
    outputs = {allowed.images.vllm: ["LlamaForCausalLM"], NEW_VLLM: ["LlamaForCausalLM", "NewLoopForCausalLM"]}
    orig = sdk.containers.run

    def run(image, *a, **kw):
        sdk.oneshot_output = "ARCHS=" + json.dumps(outputs[image])
        return orig(image, *a, **kw)
    sdk.containers.run = run
    reg = VllmArchRegistry(allowed, DockerController(sdk))
    assert await reg.ensure() == {"LlamaForCausalLM"}  # get(): the default image's, as before
    assert reg.get_for(NEW_VLLM) == {"LlamaForCausalLM", "NewLoopForCausalLM"}
    assert reg.get_any() == {"LlamaForCausalLM", "NewLoopForCausalLM"}
    assert reg.get_for(f"other/image@{D2}") is None  # not an image DGXMM may run
    again = VllmArchRegistry(allowed, DockerController(FakeDockerClient()))
    assert again.get_for(NEW_VLLM) == {"LlamaForCausalLM", "NewLoopForCausalLM"}  # cached per image


def test_vllm_supports_by_image(allowed, fake_hf_repo):
    fake_hf_repo("org/new-loop", config={**QWEN_TINY, "architectures": ["NewLoopForCausalLM"]})
    [a] = [x for x in scan_hf_cache(Path(allowed.hf_cache.local) / "hub") if x.ref == "org/new-loop"]
    e = VllmEngine(allowed)
    per_image = {allowed.images.vllm: {"LlamaForCausalLM"}, NEW_VLLM: {"NewLoopForCausalLM"}}
    e.image_arch_source = per_image.get
    e.arch_source = lambda: set().union(*per_image.values())
    assert e.supports(a)  # some allowed image serves it: offered in the inventory
    assert e.supports(a, image=NEW_VLLM) and not e.supports(a, image=allowed.images.vllm)
    assert e.images_supporting(a) == [NEW_VLLM]


# ---------- deployer ----------
@pytest.fixture
def new_arch(stack, fake_hf_repo):
    """A model only the allowed image serves."""
    fake_hf_repo("org/new-loop", config={**QWEN_TINY, "architectures": ["NewLoopForCausalLM"]},
                 files={"model.safetensors": 2 * GIB})
    stack.inv.refresh()
    e = stack.deployer.engines["vllm"]
    per_image = {stack.settings.images.vllm: {"Qwen3ForCausalLM"}, NEW_VLLM: {"Qwen3ForCausalLM", "NewLoopForCausalLM"}}
    e.image_arch_source = per_image.get
    e.arch_source = lambda: set().union(*per_image.values())
    return stack


def _req(name="q", ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, alias=alias, config={**SMALL, **cfg})


def _image_of_last_run(sdk):
    return sdk.run_calls[-1]["image"]


def test_prepare_enforces_the_allowlist(allowed, stack):
    d = stack.deployer
    assert d.prepare("vllm", "Qwen/Qwen3-Tiny", {**SMALL, "image": NEW_VLLM}).cfg.image == NEW_VLLM
    for bad in (f"evil/miner@{D1}", NEW_LLAMA, "vllm/vllm-openai:nightly"):
        with pytest.raises(DeployError) as e:
            d.prepare("vllm", "Qwen/Qwen3-Tiny", {**SMALL, "image": bad})
        assert (e.value.code, e.value.status) == ("image_not_allowed", 422)


def test_prepare_names_the_images_that_serve_an_architecture_the_default_does_not(allowed, new_arch):
    d = new_arch.deployer
    with pytest.raises(DeployError) as e:
        d.prepare("vllm", "org/new-loop", SMALL)
    assert e.value.code == "engine_unsupported" and NEW_VLLM in e.value.message and e.value.extra["images"] == [NEW_VLLM]
    assert d.prepare("vllm", "org/new-loop", {**SMALL, "image": NEW_VLLM}).cfg.image == NEW_VLLM


async def test_preview_and_deploy_carry_the_image_and_default_is_unchanged(allowed, stack):
    d = stack.deployer
    pv = await d.preview("vllm", "Qwen/Qwen3-Tiny", {**SMALL, "image": NEW_VLLM})
    assert pv["config"]["image"] == NEW_VLLM
    stack.http.all_healthy = True
    v = await d.deploy(_req(image=NEW_VLLM), user_id=1)
    assert v["config"]["image"] == NEW_VLLM and _image_of_last_run(stack.sdk) == NEW_VLLM
    assert await d.wait_ready(v["id"], 5) == "ready"  # frees the start-up slot
    await d.deploy(_req(name="plain", alias="plain"), user_id=1)
    assert _image_of_last_run(stack.sdk) == stack.settings.images.vllm


async def test_restart_reconfigure_and_switch_carry_the_image(allowed, stack):
    d = stack.deployer
    stack.http.all_healthy = True
    v = await d.deploy(_req(image=NEW_VLLM, alias="chat"), user_id=1)
    assert await d.wait_ready(v["id"], 5) == "ready"
    await d.stop(v["id"])
    await d.restart(v["id"])  # an on-demand wake is a restart of the stored config
    assert _image_of_last_run(stack.sdk) == NEW_VLLM
    assert await d.wait_ready(v["id"], 5) == "ready"
    r = await d.reconfigure(v["id"], {**SMALL, "max_model_len": 2048})  # back to the default image
    assert "image" not in {k for k, val in r["config"].items() if val} and _image_of_last_run(stack.sdk) == stack.settings.images.vllm
    r = await d.reconfigure(v["id"], {**SMALL, "image": NEW_VLLM})
    assert r["config"]["image"] == NEW_VLLM and _image_of_last_run(stack.sdk) == NEW_VLLM
    s = await d.switch(v["id"], _req(name="b", image=NEW_VLLM), [], user_id=1)
    assert s["alias"] == "chat" and s["config"]["image"] == NEW_VLLM and _image_of_last_run(stack.sdk) == NEW_VLLM
    with pytest.raises(DeployError) as e:
        await d.reconfigure(s["id"], {**SMALL, "image": f"evil/miner@{D1}"})
    assert e.value.code == "image_not_allowed"


async def test_restart_refused_once_the_image_left_the_allowlist(allowed, stack):
    d = stack.deployer
    v = await d.deploy(_req(image=NEW_VLLM), user_id=1)
    await d.stop(v["id"])
    allowed.images.vllm_allowed = []
    with pytest.raises(DeployError) as e:
        await d.restart(v["id"])
    assert e.value.code == "image_not_allowed"
    assert d.view(v["id"])["state"] == "stopped"


async def test_cluster_placement_refuses_a_chosen_image(allowed, stack):
    prep = stack.deployer.prepare("vllm", "Qwen/Qwen3-Tiny", {**SMALL, "image": NEW_VLLM})
    with pytest.raises(DeployError) as e:
        cluster.require_single_image(prep, Placement(kind="cluster", nodes=["local", "dgx-2"]))
    assert (e.value.code, e.value.status) == ("image_single_node_only", 422)
    cluster.require_single_image(prep, Placement(kind="single", nodes=["local"]))  # single node: fine


async def test_missing_image_is_pulled_on_the_remote_node_with_an_event(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.settings.images.vllm_allowed = [NEW_VLLM]
    remote, local = d.fake("dgx-2"), d.fake("local")
    remote.missing_images.add(NEW_VLLM)
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "config": {**qwen_req.config, "image": NEW_VLLM}}),
                       user_id=1)
    assert remote.pulls == [NEW_VLLM]
    assert local.pulls == []  # pulled where it runs, never on DGXMM's own node
    assert remote.containers.list()[0].image_name == NEW_VLLM
    events = [e["message"] for e in d.view(v["id"], with_events=True)["events"]]
    assert any(m.startswith(f"pulling image {NEW_VLLM} on node dgx-2") for m in events)


async def test_present_image_is_not_pulled(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert d.fake("dgx-2").pulls == []


async def test_failed_pull_fails_the_deployment_with_its_reason(deployer_two_nodes, qwen_req):
    import docker.errors
    d = deployer_two_nodes
    d.settings.images.vllm_allowed = [NEW_VLLM]
    remote = d.fake("dgx-2")
    remote.missing_images.add(NEW_VLLM)
    remote.pull_error = docker.errors.APIError("manifest unknown")
    with pytest.raises(DeployError) as e:
        await d.deploy(qwen_req.model_copy(update={"node": "dgx-2", "config": {**qwen_req.config, "image": NEW_VLLM}}),
                       user_id=1)
    assert (e.value.code, e.value.status) == ("image_pull_failed", 502)
    assert "dgx-2" in e.value.message and "manifest unknown" in e.value.message
    [v] = d.list_views()
    assert v["state"] == "failed" and remote.containers.list() == []


# ---------- API ----------
def test_api_engines_list_the_allowlists(allowed, api_stack):
    admin(api_stack)
    rows = {e["name"]: e for e in api_stack.get("/dgxmm/api/engines").json()}
    assert rows["vllm"]["allowed_images"] == [NEW_VLLM] and rows["llamacpp"]["allowed_images"] == [NEW_LLAMA]
    assert rows["vllm"]["image"] == allowed.images.vllm and rows["decision"]["allowed_images"] == []


def test_api_refuses_an_image_off_the_allowlist_and_audits_the_chosen_one(allowed, api_stack):
    h = admin(api_stack)
    body = {"name": "q", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {**SMALL, "image": f"evil/x@{D1}"}}
    r = api_stack.post("/dgxmm/api/deployments/preview", headers=h, json={k: body[k] for k in ("engine", "model_ref", "config")})
    assert r.status_code == 422 and "image_not_allowed" in r.text
    r = api_stack.post("/dgxmm/api/deployments", headers=h, json=body)
    assert r.status_code == 422 and "image_not_allowed" in r.text
    assert api_stack.stack.sdk.run_calls == []
    r = api_stack.post("/dgxmm/api/deployments", headers=h, json={**body, "config": {**SMALL, "image": NEW_VLLM}})
    assert r.status_code == 201, r.text
    [entry] = [a for a in api_stack.get("/dgxmm/api/audit").json() if a["action"] == "deploy"]
    assert entry["detail"]["image"] == NEW_VLLM
