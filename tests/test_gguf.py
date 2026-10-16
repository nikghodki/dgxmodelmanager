import os
from pathlib import Path

from dgxmm.inventory.gguf_meta import archspec_from_gguf, read_gguf_meta
from dgxmm.inventory.gguf_scan import scan_gguf_roots
from dgxmm.inventory.hf_scan import scan_hf_cache
from dgxmm.inventory.archspec import LayerGroup
from tests.conftest import write_gguf


def test_read_and_map_dense(tmp_path):
    p = write_gguf(tmp_path / "Qwen3-8B-Q4_K_M.gguf", vocab=1000)
    meta = read_gguf_meta(p)
    assert meta["general.architecture"] == "qwen3" and meta["qwen3.block_count"] == 36
    assert meta["tokenizer.ggml.tokens"] == 1000  # string arrays are reduced to their length
    s = archspec_from_gguf(meta)
    assert s.layer_groups == (LayerGroup("full", 36, kv_heads=8, head_dim=128),)
    assert s.quant == "gguf:Q4_K_M" and s.vocab_size == 1000 and s.max_context == 40960 and s.confident


def test_per_layer_kv_and_recurrent_layers(tmp_path):
    p = write_gguf(tmp_path / "hyb.gguf", arch="granitehybrid", layers=4, kv=[0, 0, 0, 8], vocab=10,
                   extra={"granitehybrid.ssm.inner_size": 8192, "granitehybrid.ssm.state_size": 128,
                          "granitehybrid.ssm.conv_kernel": 4, "granitehybrid.ssm.group_count": 1})
    s = archspec_from_gguf(read_gguf_meta(p))
    g = {x.kind: x for x in s.layer_groups}
    assert g["full"].count == 1 and g["mamba"].count == 3
    assert g["mamba"].state_elems == 8192 * 128 + 3 * (8192 + 2 * 128)


def test_sliding_pattern(tmp_path):
    p = write_gguf(tmp_path / "g.gguf", arch="gemma3", layers=6, kv=4, vocab=10,
                   extra={"gemma3.attention.sliding_window": 1024, "gemma3.attention.sliding_window_pattern": 6})
    g = {x.kind: x for x in archspec_from_gguf(read_gguf_meta(p)).layer_groups}
    assert g["sliding"].count == 5 and g["sliding"].window == 1024 and g["full"].count == 1


def test_scan_roots_shards_mmproj_and_hf_snapshots(settings, fake_hf_repo):
    root = Path(settings.gguf_roots[0].local)
    write_gguf(root / "a" / "Qwen3-8B-Q8_0.gguf", vocab=10, ftype=7, pad_bytes=1000)
    write_gguf(root / "big" / "M-Q4_K_M-00001-of-00002.gguf", vocab=10)
    (root / "big" / "M-Q4_K_M-00002-of-00002.gguf").write_bytes(b"x" * 500)
    write_gguf(root / "half" / "H-00001-of-00002.gguf", vocab=10)  # second shard missing
    write_gguf(root / "a" / "mmproj-F16.gguf", vocab=10)
    (root / "a" / "broken.gguf").symlink_to(root / "nope.gguf")
    hub = Path(settings.hf_cache.local) / "hub"
    snap = fake_hf_repo("unsloth/Qwen3-0.6B-GGUF", files={})
    blob = write_gguf(snap.parent.parent / "blobs" / "gg", vocab=10, ftype=7)
    os.symlink(os.path.relpath(blob, snap), snap / "Qwen3-0.6B-Q8_0.gguf")
    arts = {Path(a.ref).name: a for a in scan_gguf_roots([root, hub])}
    assert arts["Qwen3-8B-Q8_0.gguf"].quant == "gguf:Q8_0" and arts["Qwen3-8B-Q8_0.gguf"].kind == "llm"
    big = arts["M-Q4_K_M-00001-of-00002.gguf"]
    assert big.complete and big.weights_bytes == (root / "big" / "M-Q4_K_M-00001-of-00002.gguf").stat().st_size + 500
    assert "M-Q4_K_M-00002-of-00002.gguf" not in arts
    assert not arts["H-00001-of-00002.gguf"].complete
    assert arts["mmproj-F16.gguf"].kind == "other"
    assert not arts["broken.gguf"].complete
    assert arts["Qwen3-0.6B-Q8_0.gguf"].complete and arts["Qwen3-0.6B-Q8_0.gguf"].source == "gguf"
    assert all(a.ref != "unsloth/Qwen3-0.6B-GGUF" for a in scan_hf_cache(hub))  # not reported twice
