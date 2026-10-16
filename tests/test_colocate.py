from dgxmm.config import GIB
from dgxmm.planner.colocate import Candidate, best_combos, default_config, planner_view


def c(ref, gib, fav=False):
    return Candidate(ref, "vllm", {}, int(gib * GIB), fav)


def test_best_combos_ranking():
    cands = [c("a", 40), c("b", 30), c("c", 20), c("d", 10, fav=True), c("e", 90)]
    combos = best_combos(cands, capacity=int(75 * GIB), top=3)
    assert [sorted(m.ref for m in x.models) for x in combos][0] == ["b", "c", "d"]  # 3 models, has favourite, most headroom
    assert all(x.total <= 75 * GIB for x in combos)
    assert all("e" not in [m.ref for m in x.models] for x in combos)
    must = best_combos(cands, capacity=int(75 * GIB), must_include=("a",))
    assert all("a" in [m.ref for m in x.models] for x in must)


def test_default_config_caps_context():
    from types import SimpleNamespace
    art = SimpleNamespace(arch=SimpleNamespace(max_context=8192))
    assert default_config("vllm", art)["max_model_len"] == 8192
    assert default_config("llamacpp", art) == {"ctx_size": 8192, "parallel": 4}


async def test_planner_view(stack):
    await stack.budget.refresh()
    v = planner_view(stack.deployer, stack.inv, stack.budget.current(), favourites={"Qwen/Qwen3-Tiny"})
    refs = {x["ref"] for x in v["candidates"]}
    assert {"Qwen/Qwen3-Tiny", "org/big-model"} <= refs and "org/partial" not in refs
    assert v["combos"][0]["models"][0]["ref"] in refs and v["capacity"] > 0
