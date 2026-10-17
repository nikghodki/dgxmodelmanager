from dgxmm.config import GIB
from dgxmm.memory.budget import compute_budget
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.memory.hostmem import HostMemory
from dgxmm.planner.fit import check_fit
from dgxmm.planner.plans import ActiveDep, base_model_name, build_plans


def est(total):
    return MemoryEstimate(total, 0, 0, 0, 0, 0, 0, "estimated")


def fit_needing(required_gib, free_gib, settings):
    total = 121 * GIB
    mem = HostMemory(total=total, available=int((free_gib + 12) * GIB), free=0, cached=0, swap_total=0, swap_free=0)
    return check_fit(est(int(required_gib * GIB)), compute_budget(mem, [], settings))


CFG = {"max_model_len": 65536, "max_num_seqs": 8, "kv_cache_dtype": "auto"}


def estimator(base_gib):
    """Pretend memory = weights 20 GiB + KV proportional to context × seqs (× 0.5 when fp8)."""
    def _est(patch):
        c = {**CFG, **patch}
        kv = (base_gib - 20) * (c["max_model_len"] / 65536) * (c["max_num_seqs"] / 8) * (0.5 if c["kv_cache_dtype"] == "fp8" else 1)
        return int((20 + kv) * GIB)
    return _est


def test_reductions_and_evictions(settings):
    fit = fit_needing(60, 40, settings)  # 20 GiB short
    active = [ActiveDep(1, "chat", "ready", 10, 30 * GIB, 30 * GIB, True, 12 * GIB),
              ActiveDep(2, "embed", "ready", 0, 25 * GIB, 25 * GIB, False, 0)]
    plans = build_plans(fit=fit, cfg=CFG, context_key="max_model_len", seqs_key="max_num_seqs",
                        kv_patches=[{"kv_cache_dtype": "fp8"}], estimate_with=estimator(60), active=active,
                        variants=["Qwen/Model-FP8"], estimate_ref=lambda ref: 35 * GIB)
    by = {p.id: p for p in plans}
    assert by["kv"].fits and by["kv"].patch == {"kv_cache_dtype": "fp8"} and by["kv"].new_required == 40 * GIB
    assert by["context"].patch == {"max_model_len": 32768} and by["context"].fits
    assert by["seqs"].patch == {"max_num_seqs": 4}
    assert by["variant:Qwen/Model-FP8"].model_ref == "Qwen/Model-FP8"
    stop = by["stop:2"]  # ids name the deployments: one plan per eligible deployment / minimal combination
    assert [a.deployment_id for a in stop.actions] == [2] and stop.fits  # lowest priority first, 25 GiB >= 20
    sleep = by["sleep:1"]
    assert [a.deployment_id for a in sleep.actions] == [1] and not sleep.fits  # sleeping frees only 12 GiB
    assert [p.fits for p in plans] == sorted([p.fits for p in plans], reverse=True)  # fitting options first


def test_no_plans_when_nothing_helps(settings):
    fit = fit_needing(200, 40, settings)
    plans = build_plans(fit=fit, cfg=CFG, context_key="max_model_len", seqs_key="max_num_seqs", kv_patches=[],
                        estimate_with=lambda p: 199 * GIB, active=[], variants=[], estimate_ref=lambda r: None)
    assert all(not p.fits for p in plans)


def test_base_model_name_strips_real_gguf_quant_names():
    """Real file on the DGX: unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf. Q8_0 was not stripped, so the GGUF
    variant never matched its base model (no try_llamacpp advice, no variant plans)."""
    for f in ("/data/hf/hub/models--unsloth--Qwen3-0.6B-GGUF/snapshots/5096/Qwen3-0.6B-Q8_0.gguf",
              "Qwen3-0.6B-Q4_0.gguf", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-UD-Q4_K_XL.gguf", "Qwen3-0.6B-IQ4_XS.gguf",
              "Qwen3-0.6B-Q6_K.gguf", "Qwen3-0.6B-Q5_1.gguf", "Qwen3-0.6B-BF16.gguf"):
        assert base_model_name(f) == "qwen3-0.6b", f
    assert base_model_name("Qwen/Qwen2.5-7B-Instruct") == "qwen2.5-7b-instruct"
    assert base_model_name("acme/model-q") == "model-q"


def test_base_model_name():
    assert base_model_name("nvidia/Qwen3.6-35B-A3B-NVFP4") == "qwen3.6-35b-a3b"
    assert base_model_name("Qwen/Qwen3.6-35B-A3B") == "qwen3.6-35b-a3b"
    assert base_model_name("unsloth/Qwen3-8B-GGUF") == "qwen3-8b"
    assert base_model_name("RedHatAI/Llama-3.1-8B-Instruct-FP8-dynamic") == "llama-3.1-8b-instruct"


async def test_preview_includes_plans_when_short(stack):
    from dgxmm.deployer.service import DeployRequest
    small = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
    a = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="org/big-model", config=small), 1)
    stack.http.healthy.add(a["port"])
    await stack.deployer.wait_ready(a["id"], 5)
    stack.probe.available -= a["reserved_bytes"]  # GB10: the ready model's memory is part of `used`
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    assert not p["fit"]["fits"]
    stop = next(x for x in p["plans"] if x["id"] == f"stop:{a['id']}")
    assert stop["fits"] and stop["actions"] == [{"type": "stop", "deployment_id": a["id"]}]
