from pathlib import Path

from dgxmm.inventory.hf_scan import classify_kind, scan_hf_cache
from dgxmm.inventory.service import InventoryService
from tests.conftest import QWEN_TINY


def hub(settings):
    return Path(settings.hf_cache.local) / "hub"


def test_complete_repo(settings, fake_hf_repo):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY,
                files={"model-00001-of-00002.safetensors": 600, "model-00002-of-00002.safetensors": 400})
    [a] = scan_hf_cache(hub(settings))
    assert a.ref == "Qwen/Qwen3-Tiny" and a.source == "hf" and a.complete and a.kind == "llm"
    assert a.weights_bytes == 1000 and a.arch.num_layers == 2 and a.quant == "bf16"


def test_picks_complete_snapshot_when_main_is_broken(settings, fake_hf_repo):
    # mirrors unsloth/Qwen3.8-27B-NVFP4 on the box: refs/main snapshot has dangling links
    fake_hf_repo("unsloth/Model-NVFP4", rev="b" * 40, config=QWEN_TINY, files={"model.safetensors": 5000}, main=False)
    fake_hf_repo("unsloth/Model-NVFP4", rev="c" * 40, files={}, broken=["config.json", "model.safetensors"])
    [a] = scan_hf_cache(hub(settings))
    assert a.complete and a.revision == "b" * 40 and a.weights_bytes == 5000
    assert any("refs/main" in p for p in a.problems)


def test_incomplete_and_broken_repos_are_reported_not_raised(settings, fake_hf_repo):
    fake_hf_repo("org/partial", config=QWEN_TINY, incomplete=True)
    fake_hf_repo("org/broken", files={}, broken=["config.json"])
    (hub(settings) / "models--org--empty").mkdir()
    (hub(settings) / "datasets--org--data").mkdir()
    arts = {a.ref: a for a in scan_hf_cache(hub(settings))}
    assert set(arts) == {"org/partial", "org/broken", "org/empty"}
    assert not arts["org/partial"].complete and "download in progress or interrupted" in arts["org/partial"].problems
    assert not arts["org/broken"].complete and not arts["org/empty"].complete


def test_consolidated_weights_ignored_when_sharded_exist(settings, fake_hf_repo):
    fake_hf_repo("mistral/M", config=QWEN_TINY, files={"model.safetensors": 100, "consolidated.safetensors": 100})
    [a] = scan_hf_cache(hub(settings))
    assert a.weights_bytes == 100


def test_kind_classification():
    assert classify_kind("Qwen3ForCausalLM") == "llm"
    assert classify_kind("Gemma4ForConditionalGeneration") == "llm"
    assert classify_kind("WhisperForConditionalGeneration") == "other"
    assert classify_kind("DFlash2DraftModel") == "draft"
    assert classify_kind("ModernBertForMaskedLM") == "other"


def test_inventory_service_get(settings, fake_hf_repo):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    inv = InventoryService(settings)
    inv.refresh()
    assert inv.get("Qwen/Qwen3-Tiny").complete and inv.get("nope/nope") is None
