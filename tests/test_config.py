import pytest

from dgxmm.config import GIB, load_settings


def test_defaults_when_file_missing(tmp_path):
    s = load_settings(str(tmp_path / "missing.yaml"))
    assert s.safety_margin_gib == 8 and s.margin_bytes == 8 * GIB
    assert s.os_reserve_bytes == 4 * GIB
    assert s.port_range == (8100, 8199)
    assert s.images.vllm == "vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8"
    assert s.auto_protect is False
    assert s.base_path == "/dgxmm"
    assert s.docker_network == "dgxmm"


def test_yaml_then_env_override(tmp_path, monkeypatch):
    p = tmp_path / "c.yaml"
    p.write_text("safety_margin_gib: 10\nauto_protect: true\n")
    monkeypatch.setenv("DGXMM_SAFETY_MARGIN_GIB", "12")
    s = load_settings(str(p))
    assert s.safety_margin_gib == 12
    assert s.auto_protect is True


def test_rejects_descending_port_range(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("port_range: [9000, 8000]\n")
    with pytest.raises(ValueError):
        load_settings(str(p))


def test_read_secret(tmp_path):
    f = tmp_path / "k"
    f.write_text("  sk-123\n")
    s = load_settings(str(tmp_path / "none.yaml"))
    assert s.read_secret(str(f)) == "sk-123"
    assert s.read_secret(str(tmp_path / "absent")) is None


def test_host_home_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DGXMM_HOST_HOME", "/srv/alice")
    s = load_settings(str(tmp_path / "missing.yaml"))
    assert s.hf_cache.host == "/srv/alice/.cache/huggingface"
    assert s.gguf_roots[0].host == "/srv/alice/models"
    assert [m.host for m in s.vllm_cache_mounts] == [
        "/srv/alice/.cache/vllm", "/srv/alice/.cache/flashinfer", "/srv/alice/.triton"]


def test_host_home_falls_back_to_path_home(tmp_path, monkeypatch):
    monkeypatch.delenv("DGXMM_HOST_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    s = load_settings(str(tmp_path / "missing.yaml"))
    assert s.hf_cache.host == f"{tmp_path}/h/.cache/huggingface"
    assert s.gguf_roots[0].host == f"{tmp_path}/h/models"


def test_no_hardcoded_home():
    import dgxmm.config as c
    from pathlib import Path
    assert "nikghodki" not in Path(c.__file__).read_text()


def test_tilde_host_paths_expand_against_host_home(tmp_path, monkeypatch):
    monkeypatch.setenv("DGXMM_HOST_HOME", "/srv/bob")
    p = tmp_path / "c.yaml"
    p.write_text(
        "hf_cache: {host: ~/.cache/huggingface, local: /data/hf}\n"
        "gguf_roots: [{host: ~/models, local: /data/gguf/0}]\n"
        "vllm_cache_mounts: [{host: ~/.triton, target: /root/.triton}, {host: /abs/x, target: /x}]\n")
    s = load_settings(str(p))
    assert s.hf_cache.host == "/srv/bob/.cache/huggingface"
    assert s.gguf_roots[0].host == "/srv/bob/models"
    assert [m.host for m in s.vllm_cache_mounts] == ["/srv/bob/.triton", "/abs/x"]
    assert s.hf_cache.local == "/data/hf"


def test_example_config_loads_and_is_generic(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setenv("DGXMM_HOST_HOME", "/srv/carol")
    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    text = example.read_text()
    assert "nikghodki" not in text
    s = load_settings(str(example))
    assert s.hf_cache.host == "/srv/carol/.cache/huggingface"
    assert s.gguf_roots[0].host == "/srv/carol/models"
    assert s.docker_network == "dgxmm"


def test_example_config_vllm_image_is_public_digest():
    from pathlib import Path
    import yaml
    example = Path(__file__).resolve().parent.parent / "config.example.yaml"
    images = yaml.safe_load(example.read_text())["images"]
    assert images["vllm"].startswith("vllm/vllm-openai@sha256:") and "pinned" not in images["vllm"]
