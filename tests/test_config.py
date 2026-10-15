import pytest

from dgxmm.config import GIB, load_settings


def test_defaults_when_file_missing(tmp_path):
    s = load_settings(str(tmp_path / "missing.yaml"))
    assert s.safety_margin_gib == 8 and s.margin_bytes == 8 * GIB
    assert s.os_reserve_bytes == 4 * GIB
    assert s.port_range == (8100, 8199)
    assert s.images.vllm == "vllm/vllm-openai:pinned-2026-08-27"
    assert s.auto_protect is False
    assert s.base_path == "/dgxmm"


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
