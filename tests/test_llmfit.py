import json
import subprocess

import pytest

from dgxmm.catalog.llmfit import Llmfit, LlmfitError


class Runner:
    def __init__(self, outputs):
        self.outputs, self.calls = outputs, []

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        out = self.outputs[cmd[cmd.index("--max-context") + 2]]
        if isinstance(out, Exception):
            raise out
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")


def test_search_merges_recommend_and_caches():
    r = Runner({"search": [{"name": "Qwen/Qwen3-8B", "min_vram_gb": 6.0}],
                "recommend": {"models": [{"name": "Qwen/Qwen3-8B", "fit_level": "Perfect", "estimated_tps": 40.0}]}})
    now = [0.0]
    lf = Llmfit(runner=r, clock=lambda: now[0])
    m = lf.search_map("qwen", 70.4, 32768)
    assert m["qwen/qwen3-8b"] == {"name": "Qwen/Qwen3-8B", "min_vram_gb": 6.0, "fit_level": "Perfect", "estimated_tps": 40.0}
    assert r.calls[0][:8] == ["llmfit", "--json", "--memory", "70G", "--ram", "70G", "--max-context", "32768"]
    lf.search_map("qwen", 70.4, 32768)
    assert len(r.calls) == 2  # cached
    now[0] += 601
    lf.search_map("qwen", 70.4, 32768)
    assert len(r.calls) == 4


def test_errors():
    with pytest.raises(LlmfitError):
        Llmfit(runner=Runner({"recommend": FileNotFoundError("llmfit")})).recommend(10, 4096)
    bad = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
    with pytest.raises(LlmfitError, match="boom"):
        Llmfit(runner=bad).recommend(10, 4096)
    junk = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="not json", stderr="")
    with pytest.raises(LlmfitError):
        Llmfit(runner=junk).recommend(10, 4096)
