"""Runtime settings: optional YAML file, then DGXMM_<FIELD> environment overrides."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
import ipaddress
import logging

from pydantic import BaseModel, Field, field_validator, model_validator

GIB = 1024**3


def host_home() -> str:
    """Home directory of the host user DGXMM runs as.

    Inside the container this comes from DGXMM_HOST_HOME (set by docker-compose from .env);
    when running directly on the host it falls back to the current user's home.
    """
    return os.environ.get("DGXMM_HOST_HOME") or str(Path.home())


def _expand_host(path: str) -> str:
    """Expand a leading ``~`` against the *host* home (not the container's)."""
    if path == "~":
        return host_home()
    if path.startswith("~/"):
        return host_home().rstrip("/") + path[1:]
    return path


class PathMapping(BaseModel):
    host: str   # path on the DGX host (what engine containers mount)
    local: str  # the same directory as seen inside the dgxmm container

    @field_validator("host")
    @classmethod
    def _expand(cls, v: str) -> str:
        return _expand_host(v)


class Mount(BaseModel):
    host: str
    target: str
    mode: str = "rw"

    @field_validator("host")
    @classmethod
    def _expand(cls, v: str) -> str:
        return _expand_host(v)


def _default_hf_cache() -> PathMapping:
    return PathMapping(host="~/.cache/huggingface", local="/data/hf")


def _default_gguf_roots() -> list[PathMapping]:
    return [PathMapping(host="~/models", local="/data/gguf/0")]


def _default_vllm_cache_mounts() -> list[Mount]:
    return [
        Mount(host="~/.cache/vllm", target="/root/.cache/vllm"),
        Mount(host="~/.cache/flashinfer", target="/root/.cache/flashinfer"),
        Mount(host="~/.triton", target="/root/.triton"),
    ]


class Images(BaseModel):
    vllm: str = "vllm/vllm-openai@sha256:22d3b1217fda569653f32f1e00a2a4e64119c1601c67ff94cc72d7ca5ef352c8"
    llamacpp: str = "ghcr.io/ggml-org/llama.cpp:server-cuda"


class Settings(BaseModel):
    base_path: str = "/dgxmm"
    data_dir: str = "/app/data"
    static_dir: str = "/app/static"
    host_proc: str = "/host/proc"
    host_cgroup: str = "/host/sys/fs/cgroup"
    hf_cache: PathMapping = Field(default_factory=_default_hf_cache)
    gguf_roots: list[PathMapping] = Field(default_factory=_default_gguf_roots)
    vllm_cache_mounts: list[Mount] = Field(default_factory=_default_vllm_cache_mounts)
    images: Images = Images()
    docker_network: str = "dgxmm"  # must match DGXMM_NETWORK in .env
    engine_base_host: str | None = None  # None → reach engines by container name
    port_range: tuple[int, int] = (8100, 8199)
    publish_bind: str = "127.0.0.1"
    safety_margin_gib: float = 8.0
    os_reserve_gib: float = 4.0
    startup_timeout_s: int = 900
    monitor_poll_s: float = 3.0
    auto_protect: bool = False
    budget_interval_s: float = 2.0
    scrape_interval_s: float = 5.0
    inventory_interval_s: float = 300.0
    litellm_url: str = "http://litellm-proxy:4000"
    litellm_key_file: str = "/run/secrets/litellm_master_key"
    hf_token_file: str = "/run/secrets/hf_token"
    session_ttl_s: int = 12 * 3600
    secure_cookies: str = "auto"  # auto | always | never
    trusted_proxies: str = "127.0.0.1,::1"  # IPs/CIDRs that may set X-Forwarded-For

    @field_validator("trusted_proxies", mode="before")
    @classmethod
    def _validate_trusted_proxies(cls, v: str | list[str]) -> str:
        if isinstance(v, list):
            v = ",".join(v)
        v = v.strip()
        if v == "*":
            logging.warning("trusted_proxies='*' trusts X-Forwarded-For from any client (security risk)")
            return v
        if not v:
            return "127.0.0.1,::1"
        parts = [p.strip() for p in v.split(",") if p.strip()]
        for part in parts:
            if "/" in part:
                try:
                    ipaddress.ip_network(part, strict=False)
                except ValueError:
                    raise ValueError(f"Invalid IP/CIDR in trusted_proxies: {part}")
            else:
                try:
                    ipaddress.ip_address(part)
                except ValueError:
                    raise ValueError(f"Invalid IP/CIDR in trusted_proxies: {part}")
        return ",".join(parts)
    min_free_disk_gib: float = 50.0
    vllm_runtime_fixed_gib: float = 3.0
    vllm_cuda_graph_gib: float = 1.5
    llamacpp_runtime_fixed_gib: float = 0.5
    llmfit_bin: str = "llmfit"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        lo, hi = self.port_range
        if not (1024 <= lo <= hi <= 65535):
            raise ValueError("port_range must be ascending and within 1024-65535")
        if self.safety_margin_gib < 0 or self.os_reserve_gib < 0:
            raise ValueError("safety_margin_gib and os_reserve_gib must be >= 0")
        if self.secure_cookies not in ("auto", "always", "never"):
            raise ValueError("secure_cookies must be auto, always or never")
        return self

    @property
    def margin_bytes(self) -> int:
        return int(self.safety_margin_gib * GIB)

    @property
    def os_reserve_bytes(self) -> int:
        return int(self.os_reserve_gib * GIB)

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "dgxmm.sqlite3"

    @staticmethod
    def read_secret(path: str) -> str | None:
        p = Path(path)
        if not p.is_file():
            return None
        value = p.read_text().strip()
        return value or None


def load_settings(path: str | None = None) -> Settings:
    path = path or os.environ.get("DGXMM_CONFIG", "config.yaml")
    data: dict = {}
    p = Path(path)
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}
    for name in Settings.model_fields:
        raw = os.environ.get(f"DGXMM_{name.upper()}")
        if raw is not None and raw.strip():  # treat empty as unset
            data[name] = yaml.safe_load(raw)
    return Settings(**data)
