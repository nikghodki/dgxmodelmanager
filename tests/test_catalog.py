from types import SimpleNamespace

import pytest

from dgxmm.catalog.hub import HfHubClient, HubDetails, HubFile, HubModel, fit_badge, guess_quant, quick_required
from dgxmm.catalog.service import CatalogService
from dgxmm.config import GIB
from tests.conftest import QWEN_TINY


class FakeApi:
    def __init__(self):
        self.calls = []

    def list_models(self, **kw):
        self.calls.append(kw)
        return [SimpleNamespace(id="Qwen/Qwen3-Tiny", downloads=100, likes=5, pipeline_tag="text-generation",
                                library_name="transformers", tags=["qwen3"], gated=False, last_modified=None,
                                safetensors=SimpleNamespace(parameters={"BF16": 1_000_000_000}, total=1_000_000_000), gguf=None),
                SimpleNamespace(id="huge/Huge-1T", downloads=5, likes=1, pipeline_tag="text-generation",
                                library_name="transformers", tags=[], gated="manual", last_modified=None,
                                safetensors=SimpleNamespace(parameters={"BF16": 500_000_000_000}, total=0), gguf=None)]

    def model_info(self, repo_id, files_metadata=False):
        return SimpleNamespace(id=repo_id, sha="abc", gated=False, siblings=[
            SimpleNamespace(rfilename="config.json", size=500), SimpleNamespace(rfilename="model.safetensors", size=2 * GIB),
            SimpleNamespace(rfilename="Tiny-Q4_K_M.gguf", size=GIB // 2)])


class FakeHub:
    def search(self, q, limit=30, gguf=False):
        return [HubModel("Qwen/Qwen3-Tiny", 100, 5, "text-generation", "transformers", [], False, 2 * GIB, 10**9, False, None),
                HubModel("huge/Huge-1T", 5, 1, "text-generation", "transformers", [], "manual", 1000 * GIB, 5 * 10**11, False, None)]

    def details(self, repo_id):
        return HubDetails(repo_id, [HubFile("config.json", 500), HubFile("model.safetensors", 2 * GIB),
                                    HubFile("Tiny-Q4_K_M.gguf", GIB // 2)], QWEN_TINY, False, "abc")


class FakeLlmfit:
    def __init__(self, fail=False):
        self.fail = fail

    def search_map(self, q, free_gib, ctx):
        from dgxmm.catalog.llmfit import LlmfitError
        if self.fail:
            raise LlmfitError("not installed")
        return {"qwen/qwen3-tiny": {"name": "Qwen/Qwen3-Tiny", "fit_level": "Perfect", "estimated_tps": 40.5,
                                    "memory_required_gb": 50.0, "score": 90}}


def test_hub_client_maps_fields():
    api = FakeApi()
    hub = HfHubClient(api=api)
    [m, big] = hub.search("qwen", 10)
    assert m.repo_id == "Qwen/Qwen3-Tiny" and m.est_bytes == 2 * 10**9 and big.gated == "manual"
    assert api.calls[0]["pipeline_tag"] == "text-generation" and api.calls[0]["sort"] == "downloads"
    hub.search("qwen", 10, gguf=True)
    assert api.calls[1]["library"] == "gguf" and "pipeline_tag" not in api.calls[1]


def test_badges_and_quant_guess(settings):
    from dgxmm.memory.budget import compute_budget
    from dgxmm.memory.hostmem import HostMemory
    b = compute_budget(HostMemory(121 * GIB, 100 * GIB, 0, 0, 0, 0), [], settings)
    assert fit_badge(10 * GIB, b, 0)["badge"] == "fits_now"
    assert fit_badge(b.free_for_new + GIB, b, 2 * GIB)["badge"] == "fits_after_evict"
    assert fit_badge(1000 * GIB, b, 0)["badge"] == "too_big"
    assert quick_required(10 * GIB, settings) == int(10 * GIB * 1.05) + int(4.5 * GIB) + 2 * GIB
    assert guess_quant("Qwen3-8B-Q4_K_M.gguf") == "Q4_K_M" and guess_quant("x-NVFP4") == "NVFP4" and guess_quant("plain") is None


async def test_catalog_search_and_details(stack):
    cat = CatalogService(stack.settings, FakeHub(), FakeLlmfit(), stack.inv, stack.budget, stack.deployer,
                         SimpleNamespace(get=lambda: {"Qwen3ForCausalLM"}))
    res = await cat.search("qwen")
    tiny, huge = res["results"]
    assert tiny["downloaded"] is True and tiny["fit"]["badge"] == "fits_now"
    assert tiny["llmfit"]["fit_level"] == "Perfect" and tiny["llmfit"]["mismatch"] is True
    assert huge["fit"]["badge"] == "too_big" and huge["gated"] == "manual" and res["banner"] is None
    d = await cat.details("Qwen/Qwen3-Tiny", ctx=4096)
    assert d["support"] == {"vllm": True, "llamacpp": True}
    assert d["estimate"]["weights"] == 2 * GIB and d["fit"]["badge"] == "fits_now"
    assert d["gguf_files"][0]["quant"] == "Q4_K_M"


async def test_catalog_degrades_without_llmfit(stack):
    cat = CatalogService(stack.settings, FakeHub(), FakeLlmfit(fail=True), stack.inv, stack.budget, stack.deployer,
                         SimpleNamespace(get=lambda: None))
    res = await cat.search("qwen")
    assert "llmfit unavailable" in res["banner"] and res["results"][0]["llmfit"] is None
    assert res["results"][0]["support"]["vllm"] is None


class _InfoApi:
    def __init__(self, private=False):
        self.kw, self.private = [], private

    def model_info(self, repo_id, **kw):
        self.kw.append(kw)
        return SimpleNamespace(id=repo_id, sha="abc", gated=False, private=self.private,
                               siblings=[SimpleNamespace(rfilename="config.json", size=500)])

    def list_models(self, **kw):
        self.kw.append(kw)
        return []


def test_hub_details_and_search_are_anonymous_for_viewers():
    """Review follow-up (L1): a viewer's catalog lookups go out without the admin's HF token, and a private repo
    is never described to a viewer (same rule as the compatibility check)."""
    import httpx
    seen = []

    def handler(r):
        seen.append(r.headers.get("authorization"))
        return httpx.Response(200, json=QWEN_TINY)
    api = _InfoApi()
    hub = HfHubClient(token="hf_secret", api=api, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert hub.details("org/m").config == QWEN_TINY
    assert api.kw[-1].get("token") is None and seen[-1] == "Bearer hf_secret"
    assert hub.details("org/m", anonymous=True).config == QWEN_TINY
    assert api.kw[-1]["token"] is False and seen[-1] is None
    hub.search("qwen", 5, anonymous=True)
    assert api.kw[-1]["token"] is False
    hub.search("qwen", 5)
    assert "token" not in api.kw[-1]
    private = HfHubClient(token="hf_secret", api=_InfoApi(private=True),
                          http=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(LookupError):
        private.details("org/secret", anonymous=True)
    assert private.details("org/secret").repo_id == "org/secret"  # the admin still sees it
