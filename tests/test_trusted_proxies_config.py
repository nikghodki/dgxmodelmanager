"""Tests for trusted_proxies configuration and validation."""
import pytest
from dgxmm.config import Settings


def test_default_trusted_proxies():
    s = Settings()
    assert s.trusted_proxies == "127.0.0.1,::1"


def test_trusted_proxies_from_yaml_string():
    s = Settings(trusted_proxies="10.0.0.0/8,172.16.0.0/12")
    assert s.trusted_proxies == "10.0.0.0/8,172.16.0.0/12"


def test_trusted_proxies_from_yaml_list():
    s = Settings(trusted_proxies=["10.0.0.0/8", "172.16.0.0/12"])
    assert s.trusted_proxies == "10.0.0.0/8,172.16.0.0/12"


def test_trusted_proxies_wildcard():
    s = Settings(trusted_proxies="*")
    assert s.trusted_proxies == "*"


def test_trusted_proxies_invalid_cidr():
    with pytest.raises(ValueError, match="Invalid IP/CIDR"):
        Settings(trusted_proxies="not-an-ip")


def test_trusted_proxies_invalid_list_entry():
    with pytest.raises(ValueError, match="Invalid IP/CIDR"):
        Settings(trusted_proxies=["10.0.0.1", "bad-entry"])


def test_empty_env_var_uses_default():
    """Empty DGXMM_TRUSTED_PROXIES env var should fall through to config.yaml/default."""
    import os
    old = os.environ.get("DGXMM_TRUSTED_PROXIES")
    try:
        os.environ["DGXMM_TRUSTED_PROXIES"] = ""
        from dgxmm.config import load_settings
        s = load_settings()
        assert s.trusted_proxies == "127.0.0.1,::1"
    finally:
        if old is None:
            os.environ.pop("DGXMM_TRUSTED_PROXIES", None)
        else:
            os.environ["DGXMM_TRUSTED_PROXIES"] = old
