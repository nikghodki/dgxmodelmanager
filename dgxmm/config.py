"""Runtime settings: optional YAML file, then DGXMM_<FIELD> environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

import yaml
import ipaddress
import logging

from pydantic import BaseModel, Field, field_validator, model_validator

GIB = 1024**3


def validate_http_url(v: str, field: str = "URL") -> str:
    """An http(s) base URL with a host and no credentials, query or fragment; returned stripped, without a trailing
    slash. Raises ValueError (message names ``field``). Shared by config.yaml and the LiteLLM settings API."""
    v = v.strip()
    parts = urlsplit(v)
    if parts.scheme not in ("http", "https") or not parts.hostname or any(c.isspace() for c in v) \
            or parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError(f"{field} must be an http:// or https:// URL without credentials, query or fragment")
    try:
        parts.port  # noqa: B018  (raises ValueError on an invalid port)
    except ValueError:
        raise ValueError(f"{field} has an invalid port") from None
    return v.rstrip("/")


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
    # llama.cpp build b11669 (server-cuda-b11669; multi-arch manifest list, arm64 + amd64). To update: see
    # config.example.yaml (images.llamacpp)
    llamacpp: str = "ghcr.io/ggml-org/llama.cpp@sha256:e4cfff28678327b0ae7eda129e700e56a0fa55aa23db2f016256ea155c15c2b5"
    # built locally from deploy/decider (never pulled). Bump the tag when the image's contract with the app changes
    # (2: non-root, read-only rootfs, key delivered as a file): an older image is then "missing" and gets rebuilt.
    decision: str = "dgxmm-decider:3"


class Settings(BaseModel):
    base_path: str = "/dgxmm"
    # data_dir also holds data/litellm-db, the bundled LiteLLM's live Postgres (owned by Postgres, not
    # DGXMM). DGXMM must never walk, size, clean up or delete anything under it.
    data_dir: str = "/app/data"
    static_dir: str = "/app/static"
    host_proc: str = "/host/proc"
    host_cgroup: str = "/host/sys/fs/cgroup"
    host_passwd: str = "/host/etc/passwd"  # read-only mount; uid → user name in the processes view
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
    # Engines starting/loading at the same time (vLLM's start-up memory profile breaks when another engine allocates
    # or frees memory meanwhile). Others wait in state "queued". 0 = no limit.
    max_concurrent_startups: int = Field(1, ge=0)
    startup_queue_timeout_s: float = Field(1800.0, gt=0)  # a queued deployment fails after waiting this long
    wake_queue_timeout_s: float = Field(300.0, gt=0)  # a wake-up gives up after waiting this long for a start-up
    wake_sync_wait_s: float = Field(50.0, ge=0)  # the wake request waits this long, then the wake-up continues async
    monitor_poll_s: float = 3.0
    auto_protect: bool = False
    budget_interval_s: float = 2.0
    scrape_interval_s: float = 5.0
    inventory_interval_s: float = 300.0
    litellm_url: str = "http://litellm-proxy:4000"
    litellm_key_file: str = "/run/secrets/litellm_master_key"
    # Base URL clients use to reach LiteLLM (UI shows <url>/v1). Default for the litellm.public_url runtime
    # setting; a value saved on the LiteLLM page wins. install.sh --with-litellm seeds it on fresh configs.
    litellm_public_url: str = ""
    hf_token_file: str = "/run/secrets/hf_token"
    # Bearer key of the decision model (generated, mode 600, on first use). None -> <data_dir>/secrets/decider_api_key
    # (data/ is the only directory the app can write; compose secrets are read-only and must pre-exist).
    decider_key_file: str | None = None
    decider_build_dir: str | None = None  # docker build context of the decider image; None -> deploy/decider
    # Per-install id (DGXMM_INSTALL_ID from .env, generated by ./install.sh): label dgxmm.install on the containers
    # DGXMM creates and the images it builds, so uninstall --purge removes only this install's.
    install_id: str | None = None
    # IANA time zone of the box (e.g. Europe/Berlin) for wall-clock schedules (daily/weekly insight jobs, sleep/wake
    # schedules). DGXMM_TZ in .env, detected from the host by ./install.sh. None -> $TZ, /etc/localtime, else UTC.
    tz: str | None = Field(None, pattern=r"^[A-Za-z0-9_+-]{1,64}(/[A-Za-z0-9_+-]{1,64}){0,3}$")
    # Docker network of the decider: an *internal* network (no published port, no outbound traffic) that only DGXMM
    # joins (it attaches its own container). Empty -> docker_network (shared with every container on it; the bearer
    # key is then the only protection). See docs/security.md.
    decider_network: str = Field("dgxmm-decider", pattern=r"^$|^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
    # uid:gid the decider runs as (the image's unprivileged user); never root
    decider_user: str = Field("10001:10001", pattern=r"^[1-9][0-9]{0,9}:[1-9][0-9]{0,9}$")
    # cgroup limits of the decider container. GPU memory (cudaMalloc on unified memory) is NOT charged to the
    # container's memory cgroup on GB10: this limit caps the host-side memory (process + page cache of the weights
    # it reads + tmpfs). Measured cgroup peaks 9.3-12.7 GiB (hardened; the page cache varies), steady state ~3 GiB
    # (2.1 GiB anon). The decider's 13 GiB budget reservation is checked against the MemAvailable drop instead:
    # 12.45 GiB measured with the hardened container (GPU + host, including 0.23 GiB used in /tmp).
    decider_memory_limit_gib: float = Field(16.0, ge=2.0)
    decider_pids_limit: int = Field(512, ge=64)  # threads count as pids (measured peaks 82-98)
    # tmpfs /tmp (CUDA JIT + Triton caches; 234 MiB used after warm-up and questions). Capped so that the worst case
    # (12.45 - 0.23 + 0.75 GiB) stays inside the 13 GiB reservation; counts against the memory limit too.
    decider_tmp_gib: float = Field(0.75, gt=0)
    # Back-pressure towards the decider (d3_server runs one request at a time): requests at the server at once, and
    # how long a caller may wait for a slot before its question is answered by rules instead (docs/decider-evaluation.md)
    decider_max_in_flight: int = Field(2, ge=1, le=16)
    decider_queue_timeout_s: float = Field(1.0, ge=0)  # interactive callers; background ones up to half their deadline
    # Decider supervision (dgxmm/decider/supervisor.py): probes, auto-restart after a crash/hang with backoff
    decider_probe_interval_s: float = Field(30.0, gt=0)
    decider_probe_timeout_s: float = Field(10.0, gt=0)
    decider_hang_probes: int = Field(3, ge=1)  # failed probes in a row (container running) = hung -> restart
    decider_restart_backoff_s: float = Field(30.0, gt=0)  # first retry delay; doubles per restart
    decider_restart_backoff_max_s: float = Field(900.0, gt=0)
    decider_max_restarts_per_hour: int = Field(5, ge=1)  # then give up until the decider is enabled again
    decider_stable_s: float = Field(600.0, gt=0)  # healthy this long after a restart -> backoff resets
    # measured memory above the reservation by more than this (2 samples in a row): the watchdog stops the decider
    decider_overuse_margin_gib: float = Field(2.0, ge=0)
    # retention of the decision log (dgxmm/decider/retention.py): rows older than this many days, and beyond the
    # newest decider_max_rows, are deleted once a minute
    decider_retention_days: int = Field(30, ge=1)
    decider_max_rows: int = Field(50_000, ge=1000)
    # rows with feedback (what threshold tuning learns from) are never removed by the row cap, only after this
    decider_feedback_retention_days: int = Field(180, ge=1)
    helper_image: str | None = None  # image for root delete helpers; None → this container's image, else dgxmm:latest
    helper_timeout_s: float = 120.0
    session_ttl_s: int = 12 * 3600
    secure_cookies: str = "auto"  # auto | always | never
    trusted_proxies: str = "127.0.0.1,::1"  # IPs/CIDRs that may set X-Forwarded-For

    @field_validator("litellm_public_url")
    @classmethod
    def _validate_public_url(cls, v: str) -> str:
        v = (v or "").strip()
        return validate_http_url(v, "litellm_public_url") if v else ""

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
    def decider_key_path(self) -> Path:
        return Path(self.decider_key_file) if self.decider_key_file else Path(self.data_dir) / "secrets" / "decider_api_key"

    @property
    def admin_socket_path(self) -> Path:
        """Unix socket of the admin listener (CLI tokens are accepted only there)."""
        return Path(self.data_dir) / "run" / "admin.sock"

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
