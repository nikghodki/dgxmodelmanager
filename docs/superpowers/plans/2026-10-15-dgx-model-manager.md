# DGX Model Manager (DGXMM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build DGXMM: a FastAPI + React web app that runs on the DGX Spark. It inventories model weights, deploys and switches vLLM / llama.cpp containers behind a memory-safety gate, registers deployments in LiteLLM, shows live metrics, plans which models can run together, browses and downloads Hugging Face models (with llmfit scoring), and gives tuning advice backed by benchmarks.

**Architecture:** One Python 3.12 service (`dgxmm`) running in a Docker container on the DGX.
- It is the only writer of engine containers. It controls them through the Docker socket and touches only containers labelled `dgxmm.managed=true`.
- It reads host memory from a read-only bind of host `/proc` and `/sys/fs/cgroup`, and per-process GPU memory from NVML.
- State lives in SQLite.
- The React app is built into static files that FastAPI serves under `/dgxmm/`. The nginx reverse proxy forwards `/dgxmm/` to it.
- Live data is pushed to the browser over Server-Sent Events.

**Tech Stack:**
- Backend: Python 3.12, FastAPI, uvicorn, pydantic v2, SQLAlchemy 2 (SQLite), argon2-cffi, docker SDK 7, httpx, huggingface_hub, gguf, nvidia-ml-py, prometheus-client, llmfit.
- Frontend: React 18, TypeScript 5, Vite 5, Tailwind 3, @tanstack/react-query 5, react-router-dom 6, recharts 2.
- Tests: pytest + pytest-asyncio; vitest for the frontend. Final UI smoke test uses the Playwright MCP browser.

**Spec:** `docs/superpowers/specs/2026-10-15-dgx-model-manager-design.md` (same repo). Read it before starting. Where this plan refines the spec, Task 37 records the change back into the spec.

## Global Constraints

- All code lives in and runs on the DGX: the checkout `~/dgxmodelmanager` (host `dgx-host`, aarch64, user `dgxuser`, uid 1000, in the `docker` group).
- **One commit for the whole app.** Do **not** commit after individual tasks. Task 37 makes a single commit containing everything. The spec commit `1aafb85` already exists and stays.
- Python ≥ 3.12. The tests venv is at `~/dgxmodelmanager/.venv`.
- Engine containers are named `dgxmm-<name>` and labelled `dgxmm.managed=true`, `dgxmm.deployment_id=<id>`, `dgxmm.engine=<engine>`. Never use the old `dmm-` prefix.
- DGXMM may only stop or remove containers carrying `dgxmm.managed=true` (spec §10).
- Default vLLM image: `vllm/vllm-openai:pinned-2026-08-27`. Default llama.cpp image: `ghcr.io/ggml-org/llama.cpp:server-cuda` (an arm64 build exists; verified 2026-10-15).
- Docker network: `gateway` (the existing nginx reverse proxy and LiteLLM (`litellm-proxy`) are on it).
- Port pool: 8100–8199. Engine host ports bind to `127.0.0.1` only.
- Safety margin 8 GiB; OS reserve 4 GiB; startup timeout 900 s; `auto_protect` defaults to **false**.
- Memory units: bytes internally; GiB (1024³) in config and UI.
- Never ask for `--gpu-memory-utilization`. Always compute it: `ceil(reserved / MemTotal × 100) / 100`.
- Secrets (`litellm_master_key`, `hf_token`) are read from files under `/run/secrets`. They are never logged or returned by the API, and never stored in SQLite.
- Every state-changing API route requires a session cookie, a matching `X-CSRF-Token` header, and (except auth routes) role `admin`.
- API error bodies are always `{"detail": {"code": "<snake_case>", "message": "<human text>", ...}}`.
- Base path is `/dgxmm`: API at `/dgxmm/api/...`, UI at `/dgxmm/`.

## How to run commands (read once)

Implementers usually work from a laptop and reach the DGX over SSH (host `dgx-host` below).

- Shell on the DGX: `ssh -o BatchMode=yes dgx-host '<command>'`. In this plan, **`DGX$`** means "run in the checkout on the DGX".
- Edit locally in a mirror at `/tmp/dgxmm` and push before running anything:
  `rsync -az --delete --exclude .venv --exclude node_modules --exclude data --exclude secrets --exclude .git /tmp/dgxmm/ dgx-host:dgxmodelmanager/`
  The mirror never contains `.git`, so `--delete` cannot remove the repo history.
  First-time setup: `rsync -az dgx-host:dgxmodelmanager/docs /tmp/dgxmm/`.
- Python tests: `DGX$ .venv/bin/pytest -q <path>`.
- Frontend commands run in a throwaway Node container (Node is not installed on the host):
  `DGX$ docker run --rm -u 1000:1000 -v $PWD/frontend:/app -w /app -e HOME=/tmp node:22-bookworm-slim <npm command>`. Below, **`NPM$`** means this.
- `sudo` on the DGX needs the user's password. Any step marked **[USER]** must be handed to the user to run.

## Review Focus

These are the five input classes most likely to bite a real user that the spec implies but no feature test would naturally cover. Each one has a pinned test in the task that owns the code.

1. **HF cache snapshots that are broken or partial.** Dangling symlinks (e.g. `unsloth/Qwen3.8-27B-NVFP4`, whose `refs/main` snapshot points at missing blobs), `*.incomplete` blobs, repos with no snapshot. Expected: the inventory picks the newest *complete* snapshot, marks the rest with problems, and never crashes. Deploying an incomplete artifact is refused with `model_incomplete`. → tests in Task 8 and Task 11.
2. **Two deploys racing for the same free memory.** Expected: the global memory lock serialises pre-flight and launch, so the second deploy sees the first one's reservation and is refused. → test in Task 11.
3. **`config.json` with missing or odd fields.** No `head_dim`, no `num_key_value_heads`, nested `text_config`, vision wrappers, unknown `layer_types`. Expected: the estimator still returns a number, marks it low-confidence and adds a 20% margin. It never raises. → tests in Task 4 and Task 5.
4. **DGXMM restarting while an engine is still loading.** Expected: reconciliation resumes monitoring a running container in `starting`/`loading` instead of failing it, and fails only containers that are truly gone or exited. → test in Task 12.
5. **Hostile or malformed input in names, paths and flags.** Model refs like `../../etc`, deployment names with spaces or `;`, `extra_args` such as `--model x` or `--seed 1;rm -rf /`, GGUF paths outside allowed roots. Expected: 422 with a clear code, and nothing reaches Docker. → tests in Task 9 and Task 13.

---

## File Structure

```
dgx-model-manager/
├── pyproject.toml                 # package + deps + pytest config
├── config.example.yaml            # documented defaults (copied to config.yaml on box)
├── Dockerfile                     # node build stage → python runtime
├── docker-compose.yaml            # dgxmm service on `gateway` network
├── deploy/nginx-dgxmm.conf        # location /dgxmm/ snippet for the nginx reverse proxy
├── scripts/validate_on_box.sh     # manual end-to-end check on the DGX
├── README.md
├── dgxmm/
│   ├── __main__.py                # uvicorn entrypoint
│   ├── config.py                  # Settings, load_settings, GIB
│   ├── db.py                      # SQLAlchemy models + Database
│   ├── audit.py                   # audit(db, ...)
│   ├── events.py                  # EventBroker (SSE fan-out)
│   ├── context.py                 # AppContext + build_context (wiring)
│   ├── app.py                     # create_app (routers, static, lifespan loops)
│   ├── docker_ctl.py              # DockerController (label guard) + ContainerInfo
│   ├── litellm.py                 # LiteLLMClient + LiteLLMSync
│   ├── watchdog.py                # memory watchdog
│   ├── downloads.py               # DownloadService
│   ├── auth/      passwords.py sessions.py ratelimit.py bootstrap.py deps.py
│   ├── inventory/ archspec.py artifact.py hf_scan.py gguf_meta.py gguf_scan.py service.py
│   ├── memory/    kv.py estimate.py hostmem.py budget.py calibration.py
│   ├── planner/   fit.py plans.py colocate.py
│   ├── engines/   base.py argguard.py prom.py vllm.py llamacpp.py registry.py
│   ├── deployer/  states.py service.py reconcile.py adopt.py
│   ├── metrics/   store.py scraper.py
│   ├── catalog/   hub.py llmfit.py support.py service.py
│   ├── advisor/   rules.py benchmark.py
│   └── api/       errors.py auth.py users.py host.py inventory.py engines.py deployments.py
│                  stream.py audit.py catalog.py downloads.py planner.py metrics.py
├── tests/  conftest.py fakes.py fixtures/configs/*.json  test_*.py
└── frontend/
    ├── package.json vite.config.ts tsconfig.json tailwind.config.js postcss.config.js index.html
    └── src/ main.tsx index.css api.ts types.ts format.ts format.test.ts auth.tsx stream.ts engineFields.ts
        components/ ui.tsx Layout.tsx MemoryBar.tsx EstimatePanel.tsx PlanChooser.tsx ConfigForm.tsx Charts.tsx
        pages/ Login.tsx ChangePassword.tsx Dashboard.tsx Models.tsx DeployWizard.tsx DeploymentDetail.tsx
               Users.tsx Audit.tsx Catalog.tsx Downloads.tsx Planner.tsx
```

Responsibilities follow the spec's module list (§3).
- `deployer/service.py` is the only caller of `DockerController.run/stop/remove`.
- `memory/*` is pure computation plus host reading; it has no Docker calls.
- `planner/*` is pure functions over estimates and budgets.
- `api/*` modules are thin: validate input, call a service, shape the output.

Milestones from spec §14 map to tasks:
- **M1 (core and safety):** Tasks 1–15.
- **M2 (switching):** Tasks 16–21.
- **M3 (llama.cpp):** Tasks 22–24.
- **M4 (catalog and downloads):** Tasks 25–30.
- **M5 (metrics and advice):** Tasks 31–36.
- **Task 37:** final verification and the single commit.


## Milestone M1: Core and safety

### Task 1: Project scaffold and settings

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `config.example.yaml`, `dgxmm/__init__.py`, `dgxmm/config.py`
- Create: `tests/__init__.py` (empty), `tests/test_config.py`

**Interfaces:**
- Produces: `GIB: int`, `PathMapping(host, local)`, `Mount(host, target, mode)`, `Images(vllm, llamacpp)`, `Settings` (fields below; properties `margin_bytes`, `os_reserve_bytes`, `db_path`; method `read_secret(path) -> str | None`), `load_settings(path: str | None = None) -> Settings`.

- [ ] **Step 1: Create the mirror and the base files**

```bash
mkdir -p /tmp/dgxmm && rsync -az dgx-host:dgxmodelmanager/docs /tmp/dgxmm/
mkdir -p /tmp/dgxmm/{dgxmm,tests}
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "dgxmm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "pydantic>=2.7",
  "pyyaml>=6",
  "sqlalchemy>=2.0.30",
  "argon2-cffi>=23.1",
  "docker>=7.1",
  "httpx>=0.27",
  "huggingface_hub>=0.25",
  "gguf>=0.10",
  "nvidia-ml-py>=12.560",
  "prometheus-client>=0.20",
  "llmfit>=1.1.17",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "pytest-timeout>=2.3"]

[tool.setuptools.packages.find]
include = ["dgxmm*"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
timeout = 60
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/
secrets/
config.yaml
frontend/node_modules/
frontend/dist/
.pytest_cache/
```

`dgxmm/__init__.py`:
```python
"""DGX Model Manager."""
__version__ = "0.1.0"
```

- [ ] **Step 2: Write the failing test** — `tests/test_config.py`

```python
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
```

- [ ] **Step 3: Create the venv and run the test to watch it fail**

```bash
rsync -az --delete --exclude .venv --exclude node_modules --exclude data --exclude secrets --exclude .git /tmp/dgxmm/ dgx-host:dgxmodelmanager/
DGX$ python3 -m venv .venv && .venv/bin/pip install -q -e '.[dev]'
DGX$ .venv/bin/pytest -q tests/test_config.py
```
Expected: FAIL with `ModuleNotFoundError: No module named 'dgxmm.config'`.

- [ ] **Step 4: Implement `dgxmm/config.py`**

```python
"""Runtime settings: optional YAML file, then DGXMM_<FIELD> environment overrides."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

GIB = 1024**3
_HOME = "/home/dgxuser"


class PathMapping(BaseModel):
    host: str   # path on the DGX host (what engine containers mount)
    local: str  # the same directory as seen inside the dgxmm container


class Mount(BaseModel):
    host: str
    target: str
    mode: str = "rw"


class Images(BaseModel):
    vllm: str = "vllm/vllm-openai:pinned-2026-08-27"
    llamacpp: str = "ghcr.io/ggml-org/llama.cpp:server-cuda"


class Settings(BaseModel):
    base_path: str = "/dgxmm"
    data_dir: str = "/app/data"
    static_dir: str = "/app/static"
    host_proc: str = "/host/proc"
    host_cgroup: str = "/host/sys/fs/cgroup"
    hf_cache: PathMapping = PathMapping(host=f"{_HOME}/.cache/huggingface", local="/data/hf")
    gguf_roots: list[PathMapping] = [PathMapping(host=f"{_HOME}/models", local="/data/gguf/0")]
    vllm_cache_mounts: list[Mount] = [
        Mount(host=f"{_HOME}/.cache/vllm", target="/root/.cache/vllm"),
        Mount(host=f"{_HOME}/.cache/flashinfer", target="/root/.cache/flashinfer"),
        Mount(host=f"{_HOME}/.triton", target="/root/.triton"),
    ]
    images: Images = Images()
    docker_network: str = "gateway"
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
        if raw is not None:
            data[name] = yaml.safe_load(raw)
    return Settings(**data)
```

`config.example.yaml` (every key optional; values shown are the defaults):
```yaml
# DGXMM configuration. Copy to config.yaml next to docker-compose.yaml.
safety_margin_gib: 8        # memory DGXMM never allocates
os_reserve_gib: 4           # kept back for the OS and desktop
port_range: [8100, 8199]    # engine container ports
publish_bind: 127.0.0.1     # host interface engine ports are published on
startup_timeout_s: 900
auto_protect: false         # true = watchdog stops lowest-priority deployment when critical
docker_network: gateway
litellm_url: http://litellm-proxy:4000
images:
  vllm: vllm/vllm-openai:pinned-2026-08-27
  llamacpp: ghcr.io/ggml-org/llama.cpp:server-cuda
hf_cache:
  host: /home/dgxuser/.cache/huggingface
  local: /data/hf
gguf_roots:
  - host: /home/dgxuser/models
    local: /data/gguf/0
min_free_disk_gib: 50
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_config.py` (after rsync). Expected: `4 passed`.

---

### Task 2: Database models

**Files:**
- Create: `dgxmm/db.py`, `tests/conftest.py`, `tests/test_db.py`

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces:
  - `utcnow() -> datetime`, `as_aware(dt) -> datetime`.
  - `Database(url)` with `.session()` (a context manager that commits on success and rolls back on error) and `.engine`.
  - ORM classes: `User`, `UserSession`, `Deployment`, `DeploymentEvent`, `MemoryCalibration`, `Benchmark`, `MetricRollup`, `Download`, `AuditLog`.
  - Fixtures `settings`, `db`.

- [ ] **Step 1: Write `tests/conftest.py` (shared fixtures, extended in later tasks)**

```python
from pathlib import Path

import pytest

from dgxmm.config import PathMapping, Settings
from dgxmm.db import Database


@pytest.fixture
def settings(tmp_path) -> Settings:
    for d in ("data", "static", "proc", "cgroup", "hf/hub", "gguf"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return Settings(
        data_dir=str(tmp_path / "data"),
        static_dir=str(tmp_path / "static"),
        host_proc=str(tmp_path / "proc"),
        host_cgroup=str(tmp_path / "cgroup"),
        hf_cache=PathMapping(host="/host/hf", local=str(tmp_path / "hf")),
        gguf_roots=[PathMapping(host="/host/gguf", local=str(tmp_path / "gguf"))],
        litellm_key_file=str(tmp_path / "litellm_key"),
        hf_token_file=str(tmp_path / "hf_token"),
        engine_base_host="engines.test",
        monitor_poll_s=0.01,
    )


@pytest.fixture
def db(settings) -> Database:
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    return Database(f"sqlite:///{settings.db_path}")
```

- [ ] **Step 2: Write the failing test** — `tests/test_db.py`

```python
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from dgxmm.db import Deployment, DeploymentEvent, User


def test_roundtrip_user_and_deployment(db):
    with db.session() as s:
        s.add(User(username="admin", pw_hash="x", role="admin"))
        s.add(Deployment(name="q8b", engine="vllm", model_ref="Qwen/Qwen3-8B", config={"max_model_len": 4096},
                         state="planned", port=8100, reserved_bytes=10, estimate={}))
    with db.session() as s:
        dep = s.scalar(select(Deployment).where(Deployment.name == "q8b"))
        assert dep.config == {"max_model_len": 4096}
        assert dep.litellm_status == "none"
        s.add(DeploymentEvent(deployment_id=dep.id, from_state="planned", to_state="starting", message="go"))
    with db.session() as s:
        assert len(s.scalars(select(DeploymentEvent)).all()) == 1


def test_unique_username_and_rollback(db):
    with db.session() as s:
        s.add(User(username="a", pw_hash="x", role="viewer"))
    with pytest.raises(IntegrityError):
        with db.session() as s:
            s.add(User(username="a", pw_hash="y", role="viewer"))
    with db.session() as s:
        assert len(s.scalars(select(User)).all()) == 1
```

- [ ] **Step 3: Run it to watch it fail**

`DGX$ .venv/bin/pytest -q tests/test_db.py`. Expected: FAIL with `ModuleNotFoundError: No module named 'dgxmm.db'`.

- [ ] **Step 4: Implement `dgxmm/db.py`**

```python
"""SQLite persistence (SQLAlchemy 2 ORM)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_aware(dt: datetime) -> datetime:
    """SQLite drops tzinfo; every stored datetime is UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    pw_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16))  # admin | viewer
    must_change_pw: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UserSession(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # sha256(token)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    csrf_token: Mapped[str] = mapped_column(String(64))
    ip: Mapped[str] = mapped_column(String(64), default="")
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)
    engine: Mapped[str] = mapped_column(String(16))
    model_ref: Mapped[str] = mapped_column(String(512))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    state: Mapped[str] = mapped_column(String(16))
    container_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    alias: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reserved_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    estimate: Mapped[dict] = mapped_column(JSON, default=dict)
    litellm_status: Mapped[str] = mapped_column(String(16), default="none")
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    adopted_from: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class DeploymentEvent(Base):
    __tablename__ = "deployment_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    from_state: Mapped[str] = mapped_column(String(16))
    to_state: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text, default="")
    log_tail: Mapped[str | None] = mapped_column(Text, nullable=True)


class MemoryCalibration(Base):
    __tablename__ = "memory_calibration"
    id: Mapped[int] = mapped_column(primary_key=True)
    engine: Mapped[str] = mapped_column(String(16))
    repo: Mapped[str] = mapped_column(String(512), index=True)
    config_hash: Mapped[str] = mapped_column(String(32))
    estimated_bytes: Mapped[int] = mapped_column(BigInteger)
    measured_bytes: Mapped[int] = mapped_column(BigInteger)  # 0 for OOM records
    oom: Mapped[bool] = mapped_column(Boolean, default=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Benchmark(Base):
    __tablename__ = "benchmarks"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    engine: Mapped[str] = mapped_column(String(16))
    model_ref: Mapped[str] = mapped_column(String(512), index=True)
    config_hash: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    concurrency: Mapped[int] = mapped_column(Integer)
    input_len: Mapped[int] = mapped_column(Integer)
    output_len: Mapped[int] = mapped_column(Integer)
    requests: Mapped[int] = mapped_column(Integer)
    ttft_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    ttft_p95: Mapped[float | None] = mapped_column(Float, nullable=True)
    itl_p50: Mapped[float | None] = mapped_column(Float, nullable=True)
    tps_total: Mapped[float] = mapped_column(Float)
    tps_per_req: Mapped[float] = mapped_column(Float)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MetricRollup(Base):
    __tablename__ = "metric_rollups"
    id: Mapped[int] = mapped_column(primary_key=True)
    deployment_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)  # NULL = host
    minute: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    data: Mapped[dict] = mapped_column(JSON)


class Download(Base):
    __tablename__ = "downloads"
    id: Mapped[int] = mapped_column(primary_key=True)
    repo_id: Mapped[str] = mapped_column(String(256))
    revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    files: Mapped[list] = mapped_column(JSON, default=list)
    bytes_done: Mapped[int] = mapped_column(BigInteger, default=0)
    bytes_total: Mapped[int] = mapped_column(BigInteger, default=0)
    state: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed|cancelled
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(256), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _pragmas(conn, _):  # noqa: ANN001
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

        Base.metadata.create_all(self.engine)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._factory()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_db.py`. Expected: `2 passed`.

---

### Task 3: Authentication, users, audit, app skeleton

**Files:**
- Create: `dgxmm/auth/__init__.py` (empty), `dgxmm/auth/passwords.py`, `dgxmm/auth/sessions.py`, `dgxmm/auth/ratelimit.py`, `dgxmm/auth/bootstrap.py`, `dgxmm/auth/deps.py`
- Create: `dgxmm/audit.py`, `dgxmm/events.py`, `dgxmm/context.py`, `dgxmm/app.py`, `dgxmm/api/__init__.py` (empty), `dgxmm/api/errors.py`, `dgxmm/api/auth.py`, `dgxmm/api/users.py`
- Modify: `tests/conftest.py` (add `ctx`, `client`, `login` helpers)
- Test: `tests/test_auth.py`

**Interfaces:**
- Consumes: `Settings`, `Database`, `User`, `UserSession`, `AuditLog`.
- Produces:
  - `api_error(status, code, message, **extra) -> HTTPException`.
  - `hash_password`, `verify_password`, `validate_new_password`.
  - `create_session(db, user_id, ip, ttl_s) -> (token, csrf)`, `resolve_session(db, token, ttl_s) -> (UserSession, User) | None`, `delete_session(db, token)`.
  - `SlidingWindowLimiter(limit, window_s).hit(key) -> bool`.
  - `bootstrap_admin(db) -> str | None`.
  - `Principal(user, session)`, `get_ctx`, `current_principal`, `require_admin`.
  - `audit(db, user_id, action, target="", detail=None)`.
  - `EventBroker.publish(topic, data)`, `.subscribe() -> asyncio.Queue`, `.unsubscribe(q)`.
  - `AppContext` dataclass (extended in later tasks).
  - `create_app(ctx) -> FastAPI` (includes routers under `settings.base_path`).

- [ ] **Step 1: Write the failing tests** — `tests/test_auth.py`

```python
from dgxmm.auth.bootstrap import bootstrap_admin
from dgxmm.auth.passwords import hash_password
from dgxmm.db import User


def test_bootstrap_creates_admin_once(db):
    pw = bootstrap_admin(db)
    assert pw and len(pw) >= 16
    assert bootstrap_admin(db) is None


def test_login_me_logout(client, make_user):
    make_user("alice", "correct-horse-battery", "admin")
    r = client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": "wrong-password-x"})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "invalid_credentials"
    r = client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": "correct-horse-battery"})
    assert r.status_code == 200
    csrf = r.json()["csrf_token"]
    me = client.get("/dgxmm/api/auth/me").json()
    assert me["user"]["username"] == "alice" and me["csrf_token"] == csrf
    assert client.post("/dgxmm/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
    assert client.get("/dgxmm/api/auth/me").status_code == 401


def test_csrf_required_for_mutations(client, make_user, login):
    make_user("alice", "correct-horse-battery", "admin")
    login("alice", "correct-horse-battery")
    r = client.post("/dgxmm/api/users", json={"username": "bob", "password": "another-long-pass", "role": "viewer"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "csrf_failed"


def test_must_change_password_gate(client, db, login):
    pw = bootstrap_admin(db)
    csrf = login("admin", pw)
    r = client.get("/dgxmm/api/users")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "password_change_required"
    r = client.post("/dgxmm/api/auth/change-password", headers={"X-CSRF-Token": csrf},
                    json={"current_password": pw, "new_password": "short"})
    assert r.status_code == 422
    r = client.post("/dgxmm/api/auth/change-password", headers={"X-CSRF-Token": csrf},
                    json={"current_password": pw, "new_password": "a-much-longer-password"})
    assert r.status_code == 200
    assert client.get("/dgxmm/api/users").status_code == 200


def test_rate_limit_login(client, make_user):
    make_user("alice", "correct-horse-battery", "admin")
    codes = [client.post("/dgxmm/api/auth/login", json={"username": "alice", "password": "nope-nope-nope"}).status_code
             for _ in range(6)]
    assert codes[:5] == [401] * 5 and codes[5] == 429


def test_viewer_cannot_manage_users(client, make_user, login):
    make_user("vic", "viewer-password-1", "viewer")
    csrf = login("vic", "viewer-password-1")
    r = client.post("/dgxmm/api/users", headers={"X-CSRF-Token": csrf},
                    json={"username": "bob", "password": "another-long-pass", "role": "viewer"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "admin_required"


def test_user_crud_and_guards(client, make_user, login):
    make_user("alice", "correct-horse-battery", "admin")
    csrf = login("alice", "correct-horse-battery")
    h = {"X-CSRF-Token": csrf}
    bad = client.post("/dgxmm/api/users", headers=h, json={"username": "../x", "password": "another-long-pass", "role": "viewer"})
    assert bad.status_code == 422
    r = client.post("/dgxmm/api/users", headers=h, json={"username": "bob", "password": "another-long-pass", "role": "viewer"})
    assert r.status_code == 201
    bob_id = r.json()["id"]
    assert client.patch(f"/dgxmm/api/users/{bob_id}", headers=h, json={"role": "admin"}).json()["role"] == "admin"
    users = client.get("/dgxmm/api/users").json()
    alice_id = next(u["id"] for u in users if u["username"] == "alice")
    r = client.delete(f"/dgxmm/api/users/{alice_id}", headers=h)
    assert r.status_code == 409 and r.json()["detail"]["code"] == "cannot_delete_self"
    assert client.delete(f"/dgxmm/api/users/{bob_id}", headers=h).status_code == 204
```

- [ ] **Step 2: Add the fixtures to `tests/conftest.py`** (append)

```python
from fastapi.testclient import TestClient

from dgxmm.auth.passwords import hash_password
from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.context import AppContext
from dgxmm.db import User
from dgxmm.events import EventBroker


@pytest.fixture
def ctx(settings, db) -> AppContext:
    return AppContext(settings=settings, db=db, events=EventBroker(),
                      login_limiter=SlidingWindowLimiter(5, 60))


@pytest.fixture
def client(ctx):
    from dgxmm.app import create_app
    with TestClient(create_app(ctx), base_url="http://testserver") as c:
        yield c


@pytest.fixture
def make_user(db):
    def _make(username, password, role, must_change=False):
        with db.session() as s:
            u = User(username=username, pw_hash=hash_password(password), role=role, must_change_pw=must_change)
            s.add(u)
            s.flush()
            return u.id
    return _make


@pytest.fixture
def login(client):
    def _login(username, password) -> str:
        r = client.post("/dgxmm/api/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        return r.json()["csrf_token"]
    return _login


@pytest.fixture
def admin_csrf(make_user, login) -> str:
    make_user("root", "root-password-123", "admin")
    return login("root", "root-password-123")
```

- [ ] **Step 3: Run the tests to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_auth.py`. Expected: FAIL (ImportError for `dgxmm.auth...`).

- [ ] **Step 4: Implement the auth primitives**

`dgxmm/auth/passwords.py`:
```python
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

_ph = PasswordHasher()
MIN_LEN = 12
DUMMY_HASH = _ph.hash("dgxmm-timing-equaliser")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(pw_hash: str, password: str) -> bool:
    try:
        return _ph.verify(pw_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def validate_new_password(password: str) -> None:
    if len(password) < MIN_LEN:
        raise ValueError(f"password must be at least {MIN_LEN} characters")
```

`dgxmm/auth/sessions.py`:
```python
import hashlib
import secrets
from datetime import timedelta

from sqlalchemy import delete

from dgxmm.db import Database, User, UserSession, as_aware, utcnow


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db: Database, user_id: int, ip: str, ttl_s: int) -> tuple[str, str]:
    token, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(24)
    with db.session() as s:
        s.add(UserSession(id=_digest(token), user_id=user_id, csrf_token=csrf, ip=ip,
                          expires_at=utcnow() + timedelta(seconds=ttl_s)))
    return token, csrf


def resolve_session(db: Database, token: str | None, ttl_s: int) -> tuple[UserSession, User] | None:
    if not token:
        return None
    with db.session() as s:
        sess = s.get(UserSession, _digest(token))
        if sess is None:
            return None
        user = s.get(User, sess.user_id)
        if user is None or as_aware(sess.expires_at) < utcnow():
            s.delete(sess)
            return None
        sess.expires_at = utcnow() + timedelta(seconds=ttl_s)  # sliding expiry
        return sess, user


def delete_session(db: Database, token: str | None) -> None:
    if token:
        with db.session() as s:
            s.execute(delete(UserSession).where(UserSession.id == _digest(token)))


def delete_user_sessions(db: Database, user_id: int, keep_token: str | None = None) -> None:
    with db.session() as s:
        q = delete(UserSession).where(UserSession.user_id == user_id)
        if keep_token:
            q = q.where(UserSession.id != _digest(keep_token))
        s.execute(q)
```

`dgxmm/auth/ratelimit.py`:
```python
import threading
import time
from collections import deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """Record an attempt; return False when the key is over its limit."""
        now = self.clock()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True
```

`dgxmm/auth/bootstrap.py`:
```python
import logging
import secrets

from sqlalchemy import func, select

from dgxmm.auth.passwords import hash_password
from dgxmm.db import Database, User

log = logging.getLogger("dgxmm.bootstrap")


def bootstrap_admin(db: Database) -> str | None:
    """Create `admin` with a one-time random password when no users exist."""
    with db.session() as s:
        if s.scalar(select(func.count()).select_from(User)):
            return None
        password = secrets.token_urlsafe(18)
        s.add(User(username="admin", pw_hash=hash_password(password), role="admin", must_change_pw=True))
    log.warning("DGXMM first run: created user 'admin' with password %s (change it at first login)", password)
    return password
```

`dgxmm/audit.py`:
```python
from dgxmm.db import AuditLog, Database


def audit(db: Database, user_id: int | None, action: str, target: str = "", detail: dict | None = None) -> None:
    with db.session() as s:
        s.add(AuditLog(user_id=user_id, action=action, target=target[:256], detail=detail or {}))
```

`dgxmm/events.py`:
```python
import asyncio
from typing import Any


class EventBroker:
    """In-process fan-out of (topic, data) events to SSE subscribers."""

    def __init__(self, max_queue: int = 256):
        self._subs: set[asyncio.Queue] = set()
        self._max = max_queue

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(self._max)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, topic: str, data: Any) -> None:
        for q in list(self._subs):
            if q.full():  # slow client: drop its oldest event
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait((topic, data))
```

`dgxmm/context.py` (initial form; later tasks add fields and wiring):
```python
"""Shared service container stored on app.state.ctx."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.config import Settings
from dgxmm.db import Database
from dgxmm.events import EventBroker


@dataclass
class AppContext:
    settings: Settings
    db: Database
    events: EventBroker
    login_limiter: SlidingWindowLimiter = field(default_factory=lambda: SlidingWindowLimiter(5, 60))
    # Filled in by later tasks (None in unit tests that do not need them):
    inventory: Any = None
    budget: Any = None
    deployer: Any = None
    docker: Any = None
    engines: dict = field(default_factory=dict)
    background: list = field(default_factory=list)  # async callables started by lifespan
```

`dgxmm/api/errors.py`:
```python
from fastapi import HTTPException


def api_error(status: int, code: str, message: str, **extra) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, **extra})
```

`dgxmm/auth/deps.py`:
```python
import secrets
from dataclasses import dataclass

from fastapi import Depends, Request

from dgxmm.api.errors import api_error
from dgxmm.auth.sessions import resolve_session
from dgxmm.context import AppContext
from dgxmm.db import User, UserSession

COOKIE = "dgxmm_session"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
PW_CHANGE_ALLOWED = {"/api/auth/me", "/api/auth/logout", "/api/auth/change-password"}


@dataclass
class Principal:
    user: User
    session: UserSession


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def current_principal(request: Request, ctx: AppContext = Depends(get_ctx)) -> Principal:
    found = resolve_session(ctx.db, request.cookies.get(COOKIE), ctx.settings.session_ttl_s)
    if not found:
        raise api_error(401, "not_authenticated", "log in first")
    sess, user = found
    if request.method not in SAFE_METHODS:
        sent = request.headers.get("x-csrf-token", "")
        if not secrets.compare_digest(sent, sess.csrf_token):
            raise api_error(403, "csrf_failed", "missing or invalid X-CSRF-Token header")
    path = request.url.path.removeprefix(ctx.settings.base_path)
    if user.must_change_pw and path not in PW_CHANGE_ALLOWED:
        raise api_error(403, "password_change_required", "change your password before continuing")
    return Principal(user, sess)


def require_admin(p: Principal = Depends(current_principal)) -> Principal:
    if p.user.role != "admin":
        raise api_error(403, "admin_required", "this action needs the admin role")
    return p
```

- [ ] **Step 5: Implement the auth and users routers**

`dgxmm/api/auth.py`:
```python
from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel
from sqlalchemy import select

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import COOKIE, Principal, current_principal, get_ctx
from dgxmm.auth.passwords import DUMMY_HASH, hash_password, validate_new_password, verify_password
from dgxmm.auth.sessions import create_session, delete_session, delete_user_sessions
from dgxmm.context import AppContext
from dgxmm.db import User

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str
    password: str


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str


def user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "role": u.role, "must_change_pw": u.must_change_pw}


def _secure(request: Request, ctx: AppContext) -> bool:
    mode = ctx.settings.secure_cookies
    if mode != "auto":
        return mode == "always"
    return request.headers.get("x-forwarded-proto", request.url.scheme) == "https"


@router.post("/login")
def login(body: LoginIn, request: Request, response: Response, ctx: AppContext = Depends(get_ctx)):
    ip = request.client.host if request.client else "unknown"
    username = body.username.strip().lower()
    ip_ok = ctx.login_limiter.hit(f"ip:{ip}")
    user_ok = ctx.login_limiter.hit(f"user:{username}")
    if not (ip_ok and user_ok):
        audit(ctx.db, None, "login_rate_limited", username, {"ip": ip})
        raise api_error(429, "too_many_attempts", "too many login attempts; wait a minute")
    with ctx.db.session() as s:
        user = s.scalar(select(User).where(User.username == username))
    if user is None:
        verify_password(DUMMY_HASH, body.password)
        raise api_error(401, "invalid_credentials", "wrong username or password")
    if not verify_password(user.pw_hash, body.password):
        audit(ctx.db, user.id, "login_failed", username, {"ip": ip})
        raise api_error(401, "invalid_credentials", "wrong username or password")
    token, csrf = create_session(ctx.db, user.id, ip, ctx.settings.session_ttl_s)
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=_secure(request, ctx),
                        max_age=ctx.settings.session_ttl_s, path=ctx.settings.base_path or "/")
    audit(ctx.db, user.id, "login", username, {"ip": ip})
    return {"user": user_out(user), "csrf_token": csrf}


@router.post("/logout")
def logout(request: Request, response: Response, p: Principal = Depends(current_principal),
           ctx: AppContext = Depends(get_ctx)):
    delete_session(ctx.db, request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE, path=ctx.settings.base_path or "/")
    return {"ok": True}


@router.get("/me")
def me(p: Principal = Depends(current_principal)):
    return {"user": user_out(p.user), "csrf_token": p.session.csrf_token}


@router.post("/change-password")
def change_password(body: ChangePasswordIn, request: Request, p: Principal = Depends(current_principal),
                    ctx: AppContext = Depends(get_ctx)):
    if not verify_password(p.user.pw_hash, body.current_password):
        raise api_error(401, "invalid_credentials", "current password is wrong")
    try:
        validate_new_password(body.new_password)
    except ValueError as e:
        raise api_error(422, "weak_password", str(e))
    with ctx.db.session() as s:
        u = s.get(User, p.user.id)
        u.pw_hash = hash_password(body.new_password)
        u.must_change_pw = False
    delete_user_sessions(ctx.db, p.user.id, keep_token=request.cookies.get(COOKIE))
    audit(ctx.db, p.user.id, "password_changed", p.user.username)
    return {"ok": True}
```

`dgxmm/api/users.py`:
```python
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from dgxmm.api.auth import user_out
from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.auth.passwords import hash_password, validate_new_password
from dgxmm.auth.sessions import delete_user_sessions
from dgxmm.context import AppContext
from dgxmm.db import User

router = APIRouter(prefix="/api/users", tags=["users"])
USERNAME = r"^[a-z0-9_.-]{3,32}$"


class UserIn(BaseModel):
    username: str = Field(pattern=USERNAME)
    password: str
    role: Literal["admin", "viewer"]


class UserPatch(BaseModel):
    role: Literal["admin", "viewer"] | None = None
    password: str | None = None


@router.get("")
def list_users(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        return [user_out(u) for u in s.scalars(select(User).order_by(User.username))]


@router.post("", status_code=201)
def create_user(body: UserIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        validate_new_password(body.password)
    except ValueError as e:
        raise api_error(422, "weak_password", str(e))
    with ctx.db.session() as s:
        if s.scalar(select(User).where(User.username == body.username)):
            raise api_error(409, "username_taken", "that username already exists")
        u = User(username=body.username, pw_hash=hash_password(body.password), role=body.role, must_change_pw=True)
        s.add(u)
        s.flush()
        out = user_out(u)
    audit(ctx.db, p.user.id, "user_created", body.username, {"role": body.role})
    return out


def _admin_count(s) -> int:
    return s.scalar(select(func.count()).select_from(User).where(User.role == "admin"))


@router.patch("/{user_id}")
def update_user(user_id: int, body: UserPatch, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        u = s.get(User, user_id)
        if u is None:
            raise api_error(404, "not_found", "no such user")
        if body.role and body.role != u.role:
            if u.role == "admin" and _admin_count(s) <= 1:
                raise api_error(409, "last_admin", "cannot demote the last admin")
            u.role = body.role
        if body.password:
            try:
                validate_new_password(body.password)
            except ValueError as e:
                raise api_error(422, "weak_password", str(e))
            u.pw_hash = hash_password(body.password)
            u.must_change_pw = True
        out = user_out(u)
    if body.password:
        delete_user_sessions(ctx.db, user_id)
    audit(ctx.db, p.user.id, "user_updated", out["username"], {"role": body.role, "password_reset": bool(body.password)})
    return out


@router.delete("/{user_id}", status_code=204)
def delete_user(user_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    if user_id == p.user.id:
        raise api_error(409, "cannot_delete_self", "you cannot delete your own account")
    with ctx.db.session() as s:
        u = s.get(User, user_id)
        if u is None:
            raise api_error(404, "not_found", "no such user")
        if u.role == "admin" and _admin_count(s) <= 1:
            raise api_error(409, "last_admin", "cannot delete the last admin")
        name = u.username
        s.delete(u)
    audit(ctx.db, p.user.id, "user_deleted", name)
    return Response(status_code=204)
```

`dgxmm/app.py` (initial form; later tasks add routers and loops):
```python
"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from dgxmm.api import auth as auth_api
from dgxmm.api import users as users_api
from dgxmm.context import AppContext

log = logging.getLogger("dgxmm")


def create_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tasks = [asyncio.create_task(fn(), name=getattr(fn, "__name__", "bg")) for fn in ctx.background]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    app = FastAPI(title="DGX Model Manager", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.ctx = ctx
    bp = ctx.settings.base_path
    for module in (auth_api, users_api):
        app.include_router(module.router, prefix=bp)
    _mount_static(app, Path(ctx.settings.static_dir), bp)
    return app


def _mount_static(app: FastAPI, static: Path, bp: str) -> None:
    """Serve the built SPA; unknown non-API paths fall back to index.html."""

    @app.get(bp + "/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            from dgxmm.api.errors import api_error
            raise api_error(404, "not_found", "no such API route")
        candidate = (static / path).resolve()
        if path and candidate.is_file() and static.resolve() in candidate.parents:
            return FileResponse(candidate)
        index = static / "index.html"
        if index.is_file():
            return FileResponse(index)
        from dgxmm.api.errors import api_error
        raise api_error(404, "ui_not_built", "frontend has not been built")
```

Register the SPA route last: every router must be included before `_mount_static`, because the catch-all would otherwise shadow later `GET` API routes. Later tasks add routers to the `for module in (...)` tuple, which already runs before `_mount_static`.

- [ ] **Step 6: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_auth.py`. Expected: `7 passed`.

---

### Task 4: Model architecture parsing (`ArchSpec`)

**Files:**
- Create: `dgxmm/inventory/__init__.py` (empty), `dgxmm/inventory/archspec.py`
- Create fixtures: `tests/fixtures/configs/*.json` (copied from the real HF cache)
- Test: `tests/test_archspec.py`

**Interfaces:**
- Produces: `LayerGroup(kind, count, kv_heads=0, head_dim=0, window=None, mla_dim=0, state_elems=0)`; `ArchSpec` (fields below, property `confident`); `parse_hf_config(cfg: dict) -> ArchSpec`; `detect_quant(qc: dict | None, dtype: str) -> str`.
- `LayerGroup.kind` is one of `"full" | "sliding" | "mla" | "linear" | "mamba"`.

- [ ] **Step 1: Copy the real configs into fixtures**

```bash
DGX$ H=~/.cache/huggingface/hub; F=tests/fixtures/configs; mkdir -p $F
DGX$ cp $H/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/config.json $F/qwen3-0.6b.json
DGX$ cp $H/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/config.json $F/qwen3-8b.json
DGX$ cp $H/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5/config.json $F/qwen3.6-35b-a3b-nvfp4.json
DGX$ cp $H/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475/config.json $F/gemma-4-31b-it.json
DGX$ cp $H/models--ibm-granite--granite-4.2-30b/snapshots/*/config.json $F/granite-4.2-30b.json
DGX$ cp $H/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/57926baca9a82b4d6906b43f2750d55315f5b10f/config.json $F/qwen3.8-27b-nvfp4.json
DGX$ cp $H/models--incoai--Qwen3.8-27B-DFlash2/snapshots/*/config.json $F/qwen3.8-27b-dflash2.json
DGX$ rsync -az dgx-host:dgxmodelmanager/tests/fixtures /tmp/dgxmm/tests/
```
(The last line pulls the fixtures back into the local mirror so later rsyncs don't delete them.)

Facts from the box that the tests below rely on (verified 2026-10-15):
- **Qwen3-8B:** 36 layers, kv 8, head_dim 128, bf16, max_position_embeddings 40960.
- **Qwen3.6-35B-A3B-NVFP4:**
  - `text_config` with 40 layers: 30 `linear_attention` + 10 `full_attention`.
  - kv 2, head_dim 256; linear key/value heads 16/32, key/value head dims 128/128, conv kernel 4.
  - 256 experts, top-8, moe_intermediate 512, shared expert 512.
  - modelopt `MIXED_PRECISION` quant with `W4A16_NVFP4` layers.
- **Gemma-4-31B-it:** `text_config` with 60 layers: 50 sliding (kv 16, head_dim 256, window 1024) + 10 full (`num_global_key_value_heads` 4, `global_head_dim` 512).
- **granite-4.2-30b:** dense, 64 layers, 32 heads, kv 8, `head_dim` null, hidden 4096.
- **Qwen3.8-27B-NVFP4:**
  - 64 layers (48 linear + 16 full), kv 4, head_dim 256.
  - linear key/value heads 16/48, key/value head dims 128/128, conv kernel 4.
  - compressed-tensors with an `nvfp4-pack-quantized` group (4-bit float).
- **Qwen3.8-27B-DFlash2:** `DFlash2DraftModel`, 5 sliding layers, window 2048.

- [ ] **Step 2: Write the failing tests** — `tests/test_archspec.py`

```python
import json
from pathlib import Path

import pytest

from dgxmm.inventory.archspec import LayerGroup, detect_quant, parse_hf_config

FIX = Path(__file__).parent / "fixtures" / "configs"


def load(name):
    return parse_hf_config(json.loads((FIX / f"{name}.json").read_text()))


def groups(spec):
    return {g.kind: g for g in spec.layer_groups}


def test_dense_qwen3_8b():
    s = load("qwen3-8b")
    assert s.layer_groups == (LayerGroup("full", 36, kv_heads=8, head_dim=128),)
    assert s.quant == "bf16" and s.max_context == 40960 and s.confident and not s.is_moe


def test_granite_derives_head_dim_without_flagging():
    s = load("granite-4.2-30b")
    assert s.layer_groups == (LayerGroup("full", 64, kv_heads=8, head_dim=128),)
    assert s.confident


def test_hybrid_moe_qwen36():
    s = load("qwen3.6-35b-a3b-nvfp4")
    g = groups(s)
    assert g["full"] == LayerGroup("full", 10, kv_heads=2, head_dim=256)
    assert g["linear"].count == 30
    # conv (4-1)*(2*16*128 + 32*128) + recurrent 32*128*128
    assert g["linear"].state_elems == 3 * (2 * 2048 + 4096) + 32 * 128 * 128 == 548864
    assert s.is_moe and s.num_experts == 256 and s.experts_per_tok == 8
    assert s.intermediate_size == 512 * 8 + 512
    assert s.quant == "nvfp4"


def test_gemma4_sliding_and_global_layers():
    s = load("gemma-4-31b-it")
    g = groups(s)
    assert g["sliding"] == LayerGroup("sliding", 50, kv_heads=16, head_dim=256, window=1024)
    assert g["full"] == LayerGroup("full", 10, kv_heads=4, head_dim=512)


def test_qwen38_nvfp4_and_draft():
    s = load("qwen3.8-27b-nvfp4")
    g = groups(s)
    assert g["full"].count == 16 and g["full"].kv_heads == 4 and g["linear"].count == 48
    assert g["linear"].state_elems == 3 * (2 * 2048 + 6144) + 48 * 128 * 128
    assert s.quant == "nvfp4"
    d = load("qwen3.8-27b-dflash2")
    assert d.architecture == "DFlash2DraftModel"
    assert groups(d)["sliding"].window == 2048 and groups(d)["sliding"].count == 5


def test_mla_and_mamba_synthetic():
    mla = parse_hf_config({"architectures": ["DeepseekV3ForCausalLM"], "num_hidden_layers": 4, "hidden_size": 1024,
                           "num_attention_heads": 8, "vocab_size": 1000, "max_position_embeddings": 8192,
                           "kv_lora_rank": 512, "qk_rope_head_dim": 64, "intermediate_size": 2048})
    assert mla.layer_groups == (LayerGroup("mla", 4, mla_dim=576),)
    mamba = parse_hf_config({"architectures": ["GraniteMoeHybridForCausalLM"], "num_hidden_layers": 4, "hidden_size": 1024,
                             "num_attention_heads": 8, "num_key_value_heads": 2, "vocab_size": 1000,
                             "max_position_embeddings": 8192, "intermediate_size": 2048,
                             "layer_types": ["mamba", "mamba", "mamba", "attention"],
                             "mamba_n_heads": 16, "mamba_d_head": 64, "mamba_d_state": 128, "mamba_n_groups": 1,
                             "mamba_d_conv": 4})
    g = groups(mamba)
    assert g["full"].count == 1 and g["mamba"].count == 3
    assert g["mamba"].state_elems == 16 * 64 * 128 + 3 * (16 * 64 + 2 * 1 * 128)


@pytest.mark.parametrize("cfg", [
    {},
    {"architectures": ["LlavaForConditionalGeneration"], "text_config": {"hidden_size": 4096}},
    {"num_hidden_layers": 2, "layer_types": ["weird", "full_attention"]},
])
def test_sparse_configs_never_raise_and_flag_low_confidence(cfg):
    s = parse_hf_config(cfg)
    assert s.num_layers > 0 and s.layer_groups
    assert s.confident is False


def test_detect_quant_variants():
    assert detect_quant(None, "float16") == "fp16"
    assert detect_quant({"quant_method": "awq"}, "float16") == "awq"
    assert detect_quant({"quant_method": "modelopt", "quant_algo": "FP8"}, "bfloat16") == "fp8"
    assert detect_quant({"quant_method": "compressed-tensors",
                         "config_groups": {"g": {"weights": {"num_bits": 8, "type": "float"}}}}, "bfloat16") == "fp8"
    assert detect_quant({"quant_method": "gptq", "bits": 4}, "float16") == "gptq"
```

- [ ] **Step 3: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_archspec.py`. Expected: FAIL (ImportError).

- [ ] **Step 4: Implement `dgxmm/inventory/archspec.py`**

```python
"""Turn a Hugging Face config.json into the numbers the memory estimator needs."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

LayerKind = Literal["full", "sliding", "mla", "linear", "mamba"]
_KIND_ORDER = {"full": 0, "sliding": 1, "mla": 2, "linear": 3, "mamba": 4}


@dataclass(frozen=True)
class LayerGroup:
    kind: LayerKind
    count: int
    kv_heads: int = 0
    head_dim: int = 0
    window: int | None = None
    mla_dim: int = 0        # MLA: kv_lora_rank + qk_rope_head_dim (one latent per token)
    state_elems: int = 0    # linear/mamba: conv + recurrent state elements per layer per sequence


@dataclass(frozen=True)
class ArchSpec:
    architecture: str
    model_type: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    max_context: int
    layer_groups: tuple[LayerGroup, ...]
    intermediate_size: int
    is_moe: bool = False
    num_experts: int | None = None
    experts_per_tok: int | None = None
    torch_dtype: str = "bfloat16"
    quant: str = "bf16"
    assumptions: tuple[str, ...] = ()  # fields that were missing and defaulted

    @property
    def confident(self) -> bool:
        return not self.assumptions


def _merged(cfg: dict[str, Any]) -> dict[str, Any]:
    t = cfg.get("text_config")
    if not isinstance(t, dict):
        return dict(cfg)
    m = {**cfg, **t}
    m["architectures"] = cfg.get("architectures") or t.get("architectures")
    m["quantization_config"] = cfg.get("quantization_config") or t.get("quantization_config")
    return m


def _norm_kind(raw: Any) -> LayerKind | None:
    s = str(raw).lower()
    if s in ("full_attention", "attention", "global", "global_attention", "full"):
        return "full"
    if s in ("sliding_attention", "local", "local_attention", "sliding", "chunked_attention"):
        return "sliding"
    if s in ("linear_attention", "linear", "deltanet", "gated_deltanet"):
        return "linear"
    if s.startswith("mamba") or s == "ssm":
        return "mamba"
    return None


def detect_quant(qc: dict | None, dtype: str) -> str:
    if not qc:
        return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(str(dtype), "bf16")
    method = str(qc.get("quant_method", "")).lower()
    algos = {str(qc.get("quant_algo", "")).upper()}
    for layer in (qc.get("quantized_layers") or {}).values():
        if isinstance(layer, dict):
            algos.add(str(layer.get("quant_algo", "")).upper())
    if any("NVFP4" in a for a in algos):
        return "nvfp4"
    if any("MXFP4" in a for a in algos) or "mxfp4" in method:
        return "mxfp4"
    bits: set[tuple[Any, Any]] = set()
    formats = {str(qc.get("format", "")).lower()}
    for g in (qc.get("config_groups") or {}).values():
        w = (g or {}).get("weights") or {}
        bits.add((w.get("num_bits"), w.get("type")))
        formats.add(str((g or {}).get("format", "")).lower())
    if any("nvfp4" in f for f in formats) or (4, "float") in bits:
        return "nvfp4"
    if any("FP8" in a for a in algos) or (8, "float") in bits or method == "fp8":
        return "fp8"
    if method in ("awq", "gptq", "bitsandbytes", "gguf", "hqq", "eetq"):
        return "bnb" if method == "bitsandbytes" else method
    if (4, "int") in bits:
        return "w4a16"
    if (8, "int") in bits:
        return "w8a8"
    return method or "unknown"


def parse_hf_config(cfg: dict[str, Any]) -> ArchSpec:
    c = _merged(cfg)
    assumptions: list[str] = []

    def need(key: str, default: Any) -> Any:
        v = c.get(key)
        if v is None:
            assumptions.append(key)
            return default
        return v

    n_layers = max(int(need("num_hidden_layers", 32)), 1)
    hidden = int(need("hidden_size", 4096))
    n_heads = int(need("num_attention_heads", max(hidden // 128, 1)))
    kv_heads = int(c.get("num_key_value_heads") or n_heads)
    head_dim = int(c.get("head_dim") or max(hidden // max(n_heads, 1), 1))
    vocab = int(need("vocab_size", 32000))
    max_ctx = c.get("max_position_embeddings") or c.get("max_sequence_length") or c.get("seq_length")
    if max_ctx is None:
        assumptions.append("max_position_embeddings")
        max_ctx = 4096
    window = c.get("sliding_window")

    raw_types = c.get("layer_types") or c.get("layers_block_type")
    kinds: list[LayerKind]
    if isinstance(raw_types, list) and raw_types:
        kinds = []
        for raw in raw_types:
            k = _norm_kind(raw)
            if k is None:
                assumptions.append(f"layer_type:{raw}")
                k = "full"
            kinds.append(k)
        if len(kinds) != n_layers:
            assumptions.append("layer_types_length")
            kinds = (kinds * (n_layers // len(kinds) + 1))[:n_layers]
    elif c.get("full_attention_interval") and c.get("linear_num_value_heads"):
        iv = int(c["full_attention_interval"])
        kinds = ["full" if (i + 1) % iv == 0 else "linear" for i in range(n_layers)]
    elif window and c.get("use_sliding_window", True) is not False:
        kinds = ["sliding"] * n_layers
    else:
        kinds = ["full"] * n_layers
    if c.get("kv_lora_rank"):
        kinds = ["mla" if k == "full" else k for k in kinds]
    if "sliding" in kinds and not window:
        assumptions.append("sliding_window")

    linear_state = 0
    if "linear" in kinds:
        nk, nv = int(c.get("linear_num_key_heads") or 16), int(c.get("linear_num_value_heads") or 32)
        dk, dv = int(c.get("linear_key_head_dim") or 128), int(c.get("linear_value_head_dim") or 128)
        kconv = int(c.get("linear_conv_kernel_dim") or 4)
        if not c.get("linear_num_value_heads"):
            assumptions.append("linear_num_value_heads")
        key_dim, value_dim = nk * dk, nv * dv
        linear_state = (kconv - 1) * (2 * key_dim + value_dim) + nv * dk * dv
    mamba_state = 0
    if "mamba" in kinds:
        nh = int(c.get("mamba_n_heads") or 0)
        dh = int(c.get("mamba_d_head") or 64)
        ds = int(c.get("mamba_d_state") or c.get("ssm_state_size") or 128)
        ng = int(c.get("mamba_n_groups") or c.get("n_groups") or 1)
        dconv = int(c.get("mamba_d_conv") or c.get("conv_kernel") or 4)
        if not nh:
            assumptions.append("mamba_n_heads")
            nh = max((2 * hidden) // dh, 1)
        mamba_state = nh * dh * ds + (dconv - 1) * (nh * dh + 2 * ng * ds)

    full_kv = int(c.get("num_global_key_value_heads") or kv_heads)
    full_hd = int(c.get("global_head_dim") or head_dim)
    mla_dim = int(c.get("kv_lora_rank") or 0) + int(c.get("qk_rope_head_dim") or 64) if c.get("kv_lora_rank") else 0
    built = []
    for kind, count in sorted(Counter(kinds).items(), key=lambda kv: _KIND_ORDER[kv[0]]):
        if kind == "full":
            built.append(LayerGroup("full", count, kv_heads=full_kv, head_dim=full_hd))
        elif kind == "sliding":
            built.append(LayerGroup("sliding", count, kv_heads=kv_heads, head_dim=head_dim,
                                    window=int(window) if window else None))
        elif kind == "mla":
            built.append(LayerGroup("mla", count, mla_dim=mla_dim))
        elif kind == "linear":
            built.append(LayerGroup("linear", count, state_elems=linear_state))
        else:
            built.append(LayerGroup("mamba", count, state_elems=mamba_state))

    num_experts = c.get("num_experts") or c.get("num_local_experts") or c.get("n_routed_experts")
    is_moe = bool(num_experts and int(num_experts) > 1)
    per_tok = int(c.get("num_experts_per_tok") or c.get("moe_topk") or 0) or None
    dense_inter = int(c.get("intermediate_size") or 4 * hidden)
    if is_moe:
        moe_inter = int(c.get("moe_intermediate_size") or dense_inter)
        shared = int(c.get("shared_expert_intermediate_size") or 0)
        inter = moe_inter * (per_tok or 1) + shared
    else:
        inter = dense_inter
    dtype = str(c.get("torch_dtype") or c.get("dtype") or "bfloat16")
    archs = c.get("architectures") or []
    return ArchSpec(
        architecture=str(archs[0]) if archs else "unknown",
        model_type=str(c.get("model_type") or "unknown"),
        num_layers=n_layers, hidden_size=hidden, vocab_size=vocab, max_context=int(max_ctx),
        layer_groups=tuple(built), intermediate_size=inter, is_moe=is_moe,
        num_experts=int(num_experts) if num_experts else None, experts_per_tok=per_tok,
        torch_dtype=dtype, quant=detect_quant(c.get("quantization_config"), dtype),
        assumptions=tuple(dict.fromkeys(assumptions)),
    )
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_archspec.py`. Expected: all pass. If one real-fixture assertion fails, print the fixture's relevant keys (`python3 -c "import json;print(json.load(open('tests/fixtures/configs/<f>.json')).get('text_config',{}).keys())"`) and fix the parser. Do not edit the expected numbers, which were verified on the box.

---

### Task 5: KV cache and memory estimator (vLLM)

**Files:**
- Create: `dgxmm/memory/__init__.py` (empty), `dgxmm/memory/kv.py`, `dgxmm/memory/estimate.py`
- Test: `tests/test_estimate.py`

**Interfaces:**
- Consumes: `ArchSpec`, `LayerGroup` (Task 4), `Settings`, `GIB`.
- Produces:
  - `KV_DTYPE_BYTES: dict[str, float]`, `kv_dtype_bytes(kv_dtype: str, model_dtype: str) -> float`.
  - `kv_cache_bytes(spec, *, tokens_total, seqs, context, bytes_per_elem, window_slack=0) -> int`.
  - `state_bytes(spec, seqs, elem_bytes=4) -> int`.
  - `MemoryEstimate` (frozen dataclass; fields `weights, draft_weights, kv_cache, state, activations, runtime, margin, confidence, notes`; property `total`; `as_dict()`; `from_dict(d)`; `with_total(total, confidence, note)`).
  - `estimate_vllm(spec, weights_bytes, cfg: VllmSizing, settings, draft_spec=None, draft_bytes=0) -> MemoryEstimate`.
  - `vllm_gpu_util(total_bytes, mem_total) -> float`.
  - `VllmSizing` (dataclass `max_model_len, max_num_seqs, max_num_batched_tokens, kv_cache_dtype, kv_tokens`).

- [ ] **Step 1: Write the failing tests** — `tests/test_estimate.py`

```python
import json
from pathlib import Path

from dgxmm.config import GIB
from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.memory.estimate import MemoryEstimate, VllmSizing, estimate_vllm, vllm_gpu_util
from dgxmm.memory.kv import kv_cache_bytes, kv_dtype_bytes, state_bytes

FIX = Path(__file__).parent / "fixtures" / "configs"


def spec(name):
    return parse_hf_config(json.loads((FIX / f"{name}.json").read_text()))


def test_dense_kv_bytes():
    s = spec("qwen3-8b")
    assert kv_cache_bytes(s, tokens_total=1, seqs=1, context=1, bytes_per_elem=2) == 36 * 2 * 8 * 128 * 2


def test_vllm_estimate_qwen3_8b_exact(settings):
    s = spec("qwen3-8b")
    est = estimate_vllm(s, 16_384_000_000, VllmSizing(32768, 4, 8192, "auto", None), settings)
    assert est.kv_cache == 147456 * 32768 * 4 == 19_327_352_832
    assert est.activations == 8192 * (4 * 4096 + 2 * 12288) * 2 + 4 * 151936 * 4 * 2
    assert est.runtime == int(4.5 * GIB)
    assert est.margin == 0 and est.confidence == "estimated"
    assert est.total == 16_384_000_000 + 19_327_352_832 + est.activations + int(4.5 * GIB)
    fp8 = estimate_vllm(s, 16_384_000_000, VllmSizing(32768, 4, 8192, "fp8", None), settings)
    assert fp8.kv_cache == 19_327_352_832 // 2


def test_vllm_kv_tokens_pool_overrides_full_context_times_seqs(settings):
    s = spec("qwen3-8b")
    est = estimate_vllm(s, 0, VllmSizing(32768, 8, 8192, "auto", 65536), settings)
    assert est.kv_cache == 147456 * 65536


def test_hybrid_qwen36_kv_and_state(settings):
    s = spec("qwen3.6-35b-a3b-nvfp4")
    est = estimate_vllm(s, 22 * 10**9, VllmSizing(131072, 8, 8192, "fp8", None), settings)
    assert est.kv_cache == 10 * 2 * 2 * 256 * 1 * 131072 * 8 == 10_737_418_240
    assert est.state == 30 * 548864 * 8 * 4


def test_gemma_sliding_layers_cap_at_window(settings):
    s = spec("gemma-4-31b-it")
    kv = kv_cache_bytes(s, tokens_total=32768 * 4, seqs=4, context=32768, bytes_per_elem=2, window_slack=8192)
    full = 10 * 2 * 4 * 512 * 2 * 32768 * 4
    sliding = 50 * 2 * 16 * 256 * 2 * (1024 * 4 + 8192)
    assert kv == full + sliding == 20_803_747_840


def test_draft_model_adds_weights_and_kv(settings):
    s, d = spec("qwen3.8-27b-nvfp4"), spec("qwen3.8-27b-dflash2")
    base = estimate_vllm(s, 22 * 10**9, VllmSizing(65536, 4, 8192, "fp8", None), settings)
    with_draft = estimate_vllm(s, 22 * 10**9, VllmSizing(65536, 4, 8192, "fp8", None), settings,
                               draft_spec=d, draft_bytes=3 * 10**9)
    assert with_draft.draft_weights == 3 * 10**9
    assert with_draft.kv_cache > base.kv_cache


def test_low_confidence_adds_20_percent_margin(settings):
    s = parse_hf_config({"architectures": ["X"], "hidden_size": 2048})
    est = estimate_vllm(s, 10**9, VllmSizing(4096, 1, 2048, "auto", None), settings)
    assert est.confidence == "low"
    assert est.margin == int(0.2 * (est.kv_cache + est.state + est.activations))


def test_state_bytes_and_dtype_table():
    assert kv_dtype_bytes("auto", "bfloat16") == 2 and kv_dtype_bytes("fp8", "bfloat16") == 1
    assert kv_dtype_bytes("q8_0", "bfloat16") == 34 / 32
    s = spec("qwen3-8b")
    assert state_bytes(s, 8) == 0


def test_gpu_util_rounds_up():
    assert vllm_gpu_util(50 * GIB, 100 * GIB) == 0.50
    assert vllm_gpu_util(50 * GIB + 1, 100 * GIB) == 0.51


def test_estimate_dict_roundtrip(settings):
    est = estimate_vllm(spec("qwen3-0.6b"), 1_500_000_000, VllmSizing(8192, 2, 2048, "auto", None), settings)
    assert MemoryEstimate.from_dict(est.as_dict()) == est
    assert est.as_dict()["total"] == est.total
```

- [ ] **Step 2: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_estimate.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/memory/kv.py`**

```python
"""KV-cache and recurrent-state sizing from an ArchSpec."""
from __future__ import annotations

import math

from dgxmm.inventory.archspec import ArchSpec

KV_DTYPE_BYTES: dict[str, float] = {
    "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2, "f16": 2, "f32": 4, "fp32": 4,
    "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
    # llama.cpp block formats: bytes per element including block scales
    "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32, "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32,
}


def kv_dtype_bytes(kv_dtype: str, model_dtype: str) -> float:
    if kv_dtype in ("auto", "", None):
        return 4.0 if model_dtype in ("float32", "fp32") else 2.0
    return KV_DTYPE_BYTES[kv_dtype]


def kv_cache_bytes(spec: ArchSpec, *, tokens_total: int, seqs: int, context: int,
                   bytes_per_elem: float, window_slack: int = 0) -> int:
    """KV bytes for a pool holding `tokens_total` tokens across `seqs` concurrent sequences.

    Sliding-window layers only keep min(context, window) tokens per sequence, plus `window_slack`
    tokens once for the prefill chunk in flight. Linear/mamba layers keep no per-token KV.
    """
    total = 0.0
    for g in spec.layer_groups:
        if g.kind == "full":
            total += g.count * 2 * g.kv_heads * g.head_dim * bytes_per_elem * tokens_total
        elif g.kind == "mla":
            total += g.count * g.mla_dim * bytes_per_elem * tokens_total
        elif g.kind == "sliding":
            w = min(context, g.window or context)
            tokens = min(tokens_total, w * seqs + window_slack)
            total += g.count * 2 * g.kv_heads * g.head_dim * bytes_per_elem * tokens
    return int(math.ceil(total))


def state_bytes(spec: ArchSpec, seqs: int, elem_bytes: int = 4) -> int:
    """Fixed per-sequence state of linear-attention / mamba layers (fp32, conservative)."""
    return sum(g.count * g.state_elems * seqs * elem_bytes for g in spec.layer_groups if g.kind in ("linear", "mamba"))
```

- [ ] **Step 4: Implement `dgxmm/memory/estimate.py`**

```python
"""Per-engine memory estimates. All values in bytes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from dgxmm.config import GIB, Settings
from dgxmm.inventory.archspec import ArchSpec
from dgxmm.memory.kv import kv_cache_bytes, kv_dtype_bytes, state_bytes

LOW_CONFIDENCE_MARGIN = 0.20


@dataclass(frozen=True)
class MemoryEstimate:
    weights: int
    draft_weights: int
    kv_cache: int
    state: int
    activations: int
    runtime: int
    margin: int
    confidence: str  # low | estimated | calibrated | measured
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return (self.weights + self.draft_weights + self.kv_cache + self.state
                + self.activations + self.runtime + self.margin)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = list(self.notes)
        d["total"] = self.total
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEstimate":
        return cls(**{k: (tuple(v) if k == "notes" else v) for k, v in d.items() if k != "total"})

    def with_total(self, total: int, confidence: str, note: str) -> "MemoryEstimate":
        """Adjust the margin so the total equals `total` (used by calibration)."""
        return replace(self, margin=self.margin + (total - self.total), confidence=confidence,
                       notes=self.notes + (note,))


@dataclass(frozen=True)
class VllmSizing:
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    kv_cache_dtype: str
    kv_tokens: int | None


def _low_conf_margin(spec: ArchSpec, *parts: int) -> tuple[int, str, tuple[str, ...]]:
    if spec.confident:
        return 0, "estimated", ()
    return (int(LOW_CONFIDENCE_MARGIN * sum(parts)), "low",
            (f"config.json missing {', '.join(spec.assumptions)}; added 20% margin",))


def estimate_vllm(spec: ArchSpec, weights_bytes: int, cfg: VllmSizing, settings: Settings,
                  draft_spec: ArchSpec | None = None, draft_bytes: int = 0) -> MemoryEstimate:
    b = kv_dtype_bytes(cfg.kv_cache_dtype, spec.torch_dtype)
    tokens_total = cfg.kv_tokens or cfg.max_model_len * cfg.max_num_seqs
    kw = dict(tokens_total=tokens_total, seqs=cfg.max_num_seqs, context=cfg.max_model_len,
              bytes_per_elem=b, window_slack=cfg.max_num_batched_tokens)
    kv = kv_cache_bytes(spec, **kw)
    if draft_spec is not None:
        kv += kv_cache_bytes(draft_spec, **kw)
    st = state_bytes(spec, cfg.max_num_seqs)
    per_tok = (4 * spec.hidden_size + 2 * spec.intermediate_size) * 2
    act = cfg.max_num_batched_tokens * per_tok + cfg.max_num_seqs * spec.vocab_size * 4 * 2
    runtime = int((settings.vllm_runtime_fixed_gib + settings.vllm_cuda_graph_gib) * GIB)
    margin, confidence, notes = _low_conf_margin(spec, kv, st, act)
    return MemoryEstimate(weights=weights_bytes, draft_weights=draft_bytes, kv_cache=kv, state=st,
                          activations=act, runtime=runtime, margin=margin, confidence=confidence, notes=notes)


def vllm_gpu_util(total_bytes: int, mem_total: int) -> float:
    """vLLM's --gpu-memory-utilization is a fraction of TOTAL device memory."""
    return math.ceil(total_bytes / mem_total * 100) / 100
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_estimate.py`. Expected: all pass.

---

### Task 6: Host memory probe

**Files:**
- Create: `dgxmm/memory/hostmem.py`
- Test: `tests/test_hostmem.py`

**Interfaces:**
- Consumes: `Settings.host_proc`, `Settings.host_cgroup`.
- Produces:
  - `HostMemory(total, available, free, cached, swap_total, swap_free)` with property `used`.
  - `parse_meminfo(text) -> HostMemory`.
  - `ProcMem(pid, name, rss, gpu, cmdline)`.
  - `HostSample(mem, managed: dict[str, int], external_top: list[ProcMem], gpu: dict)`.
  - `GpuSource` protocol: `process_gpu_bytes() -> dict[int, int]`, `device_stats() -> dict`.
  - `NvmlGpu` (real implementation), `HostProbe(settings, gpu)` with `.meminfo()` and `.sample(managed_ids) -> HostSample`.

Why per-container memory is not just cgroup `memory.current`: on GB10, CUDA allocations are made by the NVIDIA driver and are not charged to the container's memory cgroup, while page cache *is* charged. So DGXMM counts a container as cgroup `anon + kernel` plus NVML's per-process GPU memory for every host PID in that container's cgroup. If NVML under-reports, the shortfall shows up as "external" memory, which only makes the gate more conservative.

- [ ] **Step 1: Write the failing tests** — `tests/test_hostmem.py`

```python
from pathlib import Path

from dgxmm.memory.hostmem import HostProbe, parse_meminfo

MEMINFO = """MemTotal:       127600524 kB
MemFree:        50703908 kB
MemAvailable:   97402460 kB
Buffers:          159568 kB
Cached:         47960852 kB
SwapTotal:      15999996 kB
SwapFree:       10485756 kB
"""
CID = "a" * 64


class FakeGpu:
    def __init__(self, procs):
        self.procs = procs

    def process_gpu_bytes(self):
        return self.procs

    def device_stats(self):
        return {"util": 50}


def make_proc(root: Path, pid: int, name: str, rss_kb: int, cgroup: str):
    d = root / str(pid)
    d.mkdir(parents=True)
    (d / "status").write_text(f"Name:\t{name}\nVmRSS:\t{rss_kb} kB\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    (d / "cmdline").write_bytes(name.encode() + b"\0--flag\0")


def test_parse_meminfo():
    m = parse_meminfo(MEMINFO)
    assert m.total == 127600524 * 1024 and m.available == 97402460 * 1024
    assert m.used == (127600524 - 97402460) * 1024


def test_sample_attributes_gpu_and_anon_to_managed_container(settings):
    proc, cg = Path(settings.host_proc), Path(settings.host_cgroup)
    (proc / "meminfo").write_text(MEMINFO)
    make_proc(proc, 100, "vllm", 2_000_000, f"/system.slice/docker-{CID}.scope")
    make_proc(proc, 200, "python", 1_000_000, "/user.slice/user-1000.slice/session-3.scope")
    make_proc(proc, 300, "kworker", 0, "/")
    scope = cg / "system.slice" / f"docker-{CID}.scope"
    scope.mkdir(parents=True)
    (scope / "memory.stat").write_text("anon 1000000\nfile 999999999\nkernel 50000\n")
    probe = HostProbe(settings, FakeGpu({100: 30 * 2**30, 200: 20 * 2**30}))
    s = probe.sample({CID})
    assert s.managed[CID] == 1_000_000 + 50_000 + 30 * 2**30
    assert [p.pid for p in s.external_top] == [200]
    assert s.external_top[0].gpu == 20 * 2**30 and s.external_top[0].rss == 1_000_000 * 1024
    assert s.gpu == {"util": 50}


def test_sample_tolerates_vanishing_processes_and_missing_cgroup(settings):
    proc = Path(settings.host_proc)
    (proc / "meminfo").write_text(MEMINFO)
    (proc / "999").mkdir()  # no status/cgroup files: process exited mid-scan
    s = HostProbe(settings, None).sample({CID})
    assert s.managed == {CID: 0}
    assert s.external_top == []
```

- [ ] **Step 2: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_hostmem.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/memory/hostmem.py`**

```python
"""Read host memory, per-container usage and top external consumers."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dgxmm.config import Settings

log = logging.getLogger(__name__)
_DOCKER_SCOPE = re.compile(r"docker-([0-9a-f]{64})\.scope")


@dataclass(frozen=True)
class HostMemory:
    total: int
    available: int
    free: int
    cached: int
    swap_total: int
    swap_free: int

    @property
    def used(self) -> int:
        return self.total - self.available


@dataclass(frozen=True)
class ProcMem:
    pid: int
    name: str
    rss: int
    gpu: int
    cmdline: str


@dataclass
class HostSample:
    mem: HostMemory
    managed: dict[str, int]
    external_top: list[ProcMem]
    gpu: dict = field(default_factory=dict)


class GpuSource(Protocol):
    def process_gpu_bytes(self) -> dict[int, int]: ...
    def device_stats(self) -> dict: ...


def parse_meminfo(text: str) -> HostMemory:
    vals: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            vals[key.strip()] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    return HostMemory(total=vals.get("MemTotal", 0), available=vals.get("MemAvailable", 0),
                      free=vals.get("MemFree", 0), cached=vals.get("Cached", 0),
                      swap_total=vals.get("SwapTotal", 0), swap_free=vals.get("SwapFree", 0))


class NvmlGpu:
    """NVML access; on GB10 device memory is 'Not Supported' but per-process usage works."""

    def __init__(self):
        self._handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # no driver / no device in tests
            log.warning("NVML unavailable: %s", e)

    def process_gpu_bytes(self) -> dict[int, int]:
        if self._handle is None:
            return {}
        out: dict[int, int] = {}
        for fn in (self._nvml.nvmlDeviceGetComputeRunningProcesses, self._nvml.nvmlDeviceGetGraphicsRunningProcesses):
            try:
                for p in fn(self._handle):
                    out[p.pid] = out.get(p.pid, 0) + int(p.usedGpuMemory or 0)
            except Exception:
                continue
        return out

    def device_stats(self) -> dict:
        if self._handle is None:
            return {}
        n, h, stats = self._nvml, self._handle, {}
        for key, fn in (("util", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu),
                        ("temp_c", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                        ("power_w", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000.0)):
            try:
                stats[key] = fn()
            except Exception:
                stats[key] = None
        return stats


class HostProbe:
    def __init__(self, settings: Settings, gpu: GpuSource | None):
        self.proc = Path(settings.host_proc)
        self.cgroup = Path(settings.host_cgroup)
        self.gpu = gpu

    def meminfo(self) -> HostMemory:
        return parse_meminfo((self.proc / "meminfo").read_text())

    def _cgroup_anon(self, cid: str) -> int:
        stat = self.cgroup / "system.slice" / f"docker-{cid}.scope" / "memory.stat"
        try:
            vals = dict(line.split() for line in stat.read_text().splitlines() if " " in line)
        except OSError:
            return 0
        kernel = int(vals.get("kernel", 0)) or sum(int(vals.get(k, 0)) for k in ("kernel_stack", "slab_unreclaimable", "sock"))
        return int(vals.get("anon", 0)) + kernel

    def sample(self, managed_ids: set[str], top_n: int = 8) -> HostSample:
        mem = self.meminfo()
        gpu_by_pid = self.gpu.process_gpu_bytes() if self.gpu else {}
        managed = {cid: self._cgroup_anon(cid) for cid in managed_ids}
        external: list[ProcMem] = []
        for d in self.proc.iterdir():
            if not d.name.isdigit():
                continue
            pid = int(d.name)
            try:
                cg = (d / "cgroup").read_text()
                status = (d / "status").read_text()
            except OSError:
                continue
            m = _DOCKER_SCOPE.search(cg)
            if m and m.group(1) in managed:
                managed[m.group(1)] += gpu_by_pid.get(pid, 0)
                continue
            fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
            rss_kb = int(fields.get("VmRSS", "0 kB").split()[0])
            gpu = gpu_by_pid.get(pid, 0)
            if rss_kb == 0 and gpu == 0:
                continue
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:200]
            except OSError:
                cmd = ""
            external.append(ProcMem(pid=pid, name=fields.get("Name", "?").strip(), rss=rss_kb * 1024, gpu=gpu, cmdline=cmd))
        external.sort(key=lambda p: p.rss + p.gpu, reverse=True)
        return HostSample(mem=mem, managed=managed, external_top=external[:top_n],
                          gpu=self.gpu.device_stats() if self.gpu else {})
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_hostmem.py`. Expected: `3 passed`.

---

### Task 7: Host budget, budget service, and pre-flight fit check

**Files:**
- Create: `dgxmm/memory/budget.py`, `dgxmm/planner/__init__.py` (empty), `dgxmm/planner/fit.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: `HostMemory`, `HostProbe`, `HostSample`, `Settings`, `MemoryEstimate`.
- Produces:
  - `DeploymentUsage(deployment_id, name, state, reserved, actual, container_id)`.
  - `Budget(total, available, margin, os_reserve, usable, committed, managed_actual, external, free_for_new, deployments)` with `.as_dict()`.
  - `COMMITTED_STATES`, `compute_budget(mem, usages, settings) -> Budget`.
  - `BudgetService(settings, probe, usage_provider)` with `async refresh() -> Budget`, `current() -> Budget | None`, `last_sample`.
  - `FitResult(fits, required, free_for_new, headroom, estimate, budget)` with `.as_dict()`.
  - `check_fit(estimate, budget, extra_free=0) -> FitResult`.
- `usage_provider` is a callable returning `list[tuple[int, str, str, int, str | None]]`: `(id, name, state, reserved_bytes, container_id)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_budget.py`

```python
from dgxmm.config import GIB
from dgxmm.memory.budget import BudgetService, DeploymentUsage, compute_budget
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.memory.hostmem import HostMemory, HostSample
from dgxmm.planner.fit import check_fit


def mem(total_gib=120, avail_gib=90):
    return HostMemory(total=total_gib * GIB, available=avail_gib * GIB, free=0, cached=0, swap_total=0, swap_free=0)


def est(total):
    return MemoryEstimate(weights=total, draft_weights=0, kv_cache=0, state=0, activations=0, runtime=0,
                          margin=0, confidence="estimated")


def test_budget_formula(settings):
    usages = [DeploymentUsage(1, "a", "ready", 40 * GIB, 38 * GIB, "c1"),
              DeploymentUsage(2, "b", "starting", 20 * GIB, 2 * GIB, "c2"),
              DeploymentUsage(3, "c", "stopped", 10 * GIB, 0, None)]
    b = compute_budget(mem(120, 50), usages, settings)
    assert b.usable == 120 * GIB - 8 * GIB - 4 * GIB
    assert b.committed == 60 * GIB
    assert b.managed_actual == 40 * GIB
    assert b.external == 70 * GIB - 40 * GIB  # used - managed actual
    assert b.free_for_new == b.usable - b.committed - b.external == 18 * GIB


def test_sleeping_counts_actual_not_reserved(settings):
    b = compute_budget(mem(120, 100), [DeploymentUsage(1, "a", "sleeping", 40 * GIB, 16 * GIB, "c1")], settings)
    assert b.committed == 16 * GIB


def test_external_never_negative(settings):
    b = compute_budget(mem(120, 119), [DeploymentUsage(1, "a", "ready", 10 * GIB, 30 * GIB, "c")], settings)
    assert b.external == 0


def test_check_fit(settings):
    b = compute_budget(mem(120, 100), [], settings)
    ok = check_fit(est(50 * GIB), b)
    assert ok.fits and ok.headroom == b.free_for_new - 50 * GIB
    no = check_fit(est(b.free_for_new + 1), b)
    assert not no.fits and no.headroom == -1
    assert check_fit(est(b.free_for_new + 1), b, extra_free=1).fits


class FakeProbe:
    def __init__(self, sample):
        self._s = sample

    def sample(self, ids):
        return self._s

    def meminfo(self):
        return self._s.mem


async def test_budget_service_maps_container_actuals(settings):
    sample = HostSample(mem=mem(120, 80), managed={"c1": 30 * GIB}, external_top=[])
    svc = BudgetService(settings, FakeProbe(sample), lambda: [(1, "a", "ready", 32 * GIB, "c1")])
    b = await svc.refresh()
    assert b.deployments[0].actual == 30 * GIB and svc.current() is b and svc.last_sample is sample
```

- [ ] **Step 2: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_budget.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/memory/budget.py`**

```python
"""Host memory budget (spec §4.1)."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Callable

from dgxmm.config import Settings
from dgxmm.memory.hostmem import HostMemory, HostSample

COMMITTED_STATES = frozenset({"starting", "loading", "ready", "unknown"})
MEASURED_STATES = COMMITTED_STATES | {"sleeping", "stopping"}

UsageRow = tuple[int, str, str, int, str | None]


@dataclass(frozen=True)
class DeploymentUsage:
    deployment_id: int
    name: str
    state: str
    reserved: int
    actual: int
    container_id: str | None = None


@dataclass(frozen=True)
class Budget:
    total: int
    available: int
    margin: int
    os_reserve: int
    usable: int
    committed: int
    managed_actual: int
    external: int
    free_for_new: int
    deployments: tuple[DeploymentUsage, ...]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["deployments"] = [asdict(u) for u in self.deployments]
        return d


def compute_budget(mem: HostMemory, usages: list[DeploymentUsage], settings: Settings) -> Budget:
    usable = mem.total - settings.margin_bytes - settings.os_reserve_bytes
    committed = 0
    for u in usages:
        if u.state in COMMITTED_STATES:
            committed += max(u.reserved, u.actual)
        elif u.state in ("sleeping", "stopping"):
            committed += u.actual
    managed_actual = sum(u.actual for u in usages if u.state in MEASURED_STATES)
    external = max(mem.used - managed_actual, 0)
    return Budget(total=mem.total, available=mem.available, margin=settings.margin_bytes,
                  os_reserve=settings.os_reserve_bytes, usable=usable, committed=committed,
                  managed_actual=managed_actual, external=external,
                  free_for_new=usable - committed - external, deployments=tuple(usages))


class BudgetService:
    def __init__(self, settings: Settings, probe, usage_provider: Callable[[], list[UsageRow]]):
        self.settings, self.probe, self.usage_provider = settings, probe, usage_provider
        self._current: Budget | None = None
        self.last_sample: HostSample | None = None

    def current(self) -> Budget | None:
        return self._current

    async def refresh(self) -> Budget:
        rows = self.usage_provider()
        ids = {cid for (_, _, _, _, cid) in rows if cid}
        sample = await asyncio.to_thread(self.probe.sample, ids)
        usages = [DeploymentUsage(i, n, st, res, sample.managed.get(cid, 0) if cid else 0, cid)
                  for (i, n, st, res, cid) in rows]
        self.last_sample = sample
        self._current = compute_budget(sample.mem, usages, self.settings)
        return self._current
```

A committed deployment counts as `max(reserved, actual)`. If an engine grows past its estimate, the extra is counted against it rather than slipping into "external".

- [ ] **Step 4: Implement `dgxmm/planner/fit.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from dgxmm.memory.budget import Budget
from dgxmm.memory.estimate import MemoryEstimate


@dataclass(frozen=True)
class FitResult:
    fits: bool
    required: int
    free_for_new: int
    headroom: int
    estimate: MemoryEstimate
    budget: Budget

    def as_dict(self) -> dict:
        return {"fits": self.fits, "required": self.required, "free_for_new": self.free_for_new,
                "headroom": self.headroom, "estimate": self.estimate.as_dict()}


def check_fit(estimate: MemoryEstimate, budget: Budget, extra_free: int = 0) -> FitResult:
    """`extra_free`: memory that will be released first (e.g. the deployment being replaced)."""
    free = budget.free_for_new + extra_free
    return FitResult(fits=estimate.total <= free, required=estimate.total, free_for_new=free,
                     headroom=free - estimate.total, estimate=estimate, budget=budget)
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_budget.py`. Expected: `5 passed`.

---

### Task 8: Hugging Face cache inventory

**Files:**
- Create: `dgxmm/inventory/artifact.py`, `dgxmm/inventory/hf_scan.py`, `dgxmm/inventory/service.py`
- Modify: `tests/conftest.py` (add the `fake_hf_repo` builder)
- Test: `tests/test_inventory.py`

**Interfaces:**
- Consumes: `parse_hf_config`, `ArchSpec`, `Settings.hf_cache`.
- Produces:
  - `ModelArtifact` dataclass (fields below) with `.as_dict()`.
  - `classify_kind(arch: str) -> str`.
  - `scan_hf_cache(hub_dir: Path) -> list[ModelArtifact]`.
  - `InventoryService(settings, scanners=None)` with `refresh() -> list[ModelArtifact]`, `list() -> list[ModelArtifact]`, `get(ref) -> ModelArtifact | None`, `async run()`.
  - `REPO_RE`.

- [ ] **Step 1: Add the fake-repo builder to `tests/conftest.py`** (append)

```python
import json as _json
import os as _os


@pytest.fixture
def fake_hf_repo(settings):
    """Build models--org--name/{blobs,snapshots/<rev>,refs/main} like huggingface_hub does."""
    hub = Path(settings.hf_cache.local) / "hub"

    def _make(repo_id: str, rev: str = "a" * 40, config: dict | None = None,
              files: dict[str, int] | None = None, main: bool = True, broken: list[str] | None = None,
              incomplete: bool = False):
        d = hub / ("models--" + repo_id.replace("/", "--"))
        (d / "blobs").mkdir(parents=True, exist_ok=True)
        snap = d / "snapshots" / rev
        snap.mkdir(parents=True, exist_ok=True)
        entries = dict(files or {"model.safetensors": 1000})
        if config is not None:
            entries["config.json"] = -1
        for name, size in entries.items():
            blob = d / "blobs" / f"{rev[:8]}-{name.replace('/', '_')}"
            if name == "config.json":
                blob.write_text(_json.dumps(config))
            else:
                with open(blob, "wb") as f:
                    f.truncate(size)
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            _os.symlink(_os.path.relpath(blob, (snap / name).parent), snap / name)
        for name in broken or []:
            _os.symlink("../../blobs/does-not-exist", snap / name)
        if incomplete:
            (d / "blobs" / "deadbeef.incomplete").write_bytes(b"x")
        if main:
            (d / "refs").mkdir(exist_ok=True)
            (d / "refs" / "main").write_text(rev)
        return snap
    return _make


QWEN_TINY = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "num_hidden_layers": 2,
             "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "hidden_size": 256,
             "intermediate_size": 512, "vocab_size": 1000, "max_position_embeddings": 4096,
             "torch_dtype": "bfloat16"}
```

- [ ] **Step 2: Write the failing tests** — `tests/test_inventory.py`

```python
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
```

- [ ] **Step 3: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_inventory.py`. Expected: FAIL (ImportError).

- [ ] **Step 4: Implement `dgxmm/inventory/artifact.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dgxmm.inventory.archspec import ArchSpec

REPO_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"


@dataclass
class ModelArtifact:
    ref: str                       # HF repo id, or local GGUF path for source="gguf"
    source: Literal["hf", "gguf"]
    local_path: str                # snapshot dir or .gguf file (dgxmm container namespace)
    revision: str | None
    format: Literal["safetensors", "gguf", "other"]
    weights_bytes: int
    size_bytes: int
    complete: bool
    kind: Literal["llm", "draft", "other"]
    arch: ArchSpec | None
    architecture: str | None
    quant: str | None
    problems: list[str] = field(default_factory=list)
    config: dict[str, Any] | None = None
    files: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        a = self.arch
        return {
            "ref": self.ref, "source": self.source, "revision": self.revision, "format": self.format,
            "weights_bytes": self.weights_bytes, "size_bytes": self.size_bytes, "complete": self.complete,
            "kind": self.kind, "architecture": self.architecture, "quant": self.quant, "problems": self.problems,
            "max_context": a.max_context if a else None, "is_moe": a.is_moe if a else None,
            "num_layers": a.num_layers if a else None, "confident": a.confident if a else None,
            "has_mtp": any("mtp" in f for f in self.files),
        }
```

- [ ] **Step 5: Implement `dgxmm/inventory/hf_scan.py`**

```python
"""Scan a huggingface_hub cache directory (…/hub) without network access."""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.inventory.artifact import ModelArtifact

log = logging.getLogger(__name__)
_LLM_SUFFIX = re.compile(r"(CausalLM|ConditionalGeneration|LMHeadModel)$")
_NON_LLM_PREFIX = ("Whisper", "Bart", "MBart", "T5", "MT5", "Pegasus", "Marian", "Speech", "Wav2Vec",
                   "Seamless", "Blip", "Clip", "Siglip", "Encodec")
_DRAFT_HINTS = ("Draft", "Eagle", "DFlash", "Medusa")


def classify_kind(arch: str | None) -> str:
    if not arch:
        return "other"
    if any(h in arch for h in _DRAFT_HINTS):
        return "draft"
    if _LLM_SUFFIX.search(arch) and not arch.startswith(_NON_LLM_PREFIX):
        return "llm"
    return "other"


def _resolve_size(p: Path) -> int | None:
    try:
        return p.stat().st_size  # follows symlinks; raises when dangling
    except OSError:
        return None


def _inspect_snapshot(snap: Path) -> tuple[dict | None, dict[str, int], list[str]]:
    sizes: dict[str, int] = {}
    broken: list[str] = []
    for root, _dirs, files in os.walk(snap):
        for f in files:
            p = Path(root) / f
            rel = str(p.relative_to(snap))
            size = _resolve_size(p)
            if size is None:
                broken.append(rel)
            else:
                sizes[rel] = size
    config = None
    if "config.json" in sizes:
        try:
            config = json.loads((snap / "config.json").read_text())
        except (OSError, ValueError):
            broken.append("config.json (unreadable)")
    return config, sizes, broken


def _weights(sizes: dict[str, int]) -> int:
    st = {k: v for k, v in sizes.items() if k.endswith(".safetensors")}
    if any(k.startswith("model") for k in st):
        st = {k: v for k, v in st.items() if not k.startswith("consolidated")}
    if st:
        return sum(st.values())
    return sum(v for k, v in sizes.items() if k.endswith((".bin", ".pt", ".pth")) and "training_args" not in k)


def _scan_repo(d: Path) -> ModelArtifact:
    parts = d.name.split("--")
    ref = f"{parts[1]}/{'--'.join(parts[2:])}"
    snaps_dir = d / "snapshots"
    snaps = sorted((p for p in snaps_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True) \
        if snaps_dir.is_dir() else []
    main_rev = (d / "refs" / "main").read_text().strip() if (d / "refs" / "main").is_file() else None
    if main_rev:
        snaps.sort(key=lambda p: p.name != main_rev)  # stable: main first, then newest
    problems: list[str] = []
    if any(b.name.endswith(".incomplete") for b in (d / "blobs").glob("*")) if (d / "blobs").is_dir() else False:
        problems.append("download in progress or interrupted")
    chosen = None
    for snap in snaps:
        config, sizes, broken = _inspect_snapshot(snap)
        if config is not None and not broken and _weights(sizes) > 0:
            chosen = (snap, config, sizes)
            break
        if snap.name == main_rev:
            problems.append(f"refs/main snapshot {snap.name[:8]} is incomplete ({len(broken)} broken links)")
    if chosen is None:
        if not snaps:
            problems.append("no snapshot")
        elif not problems or all("refs/main" not in p for p in problems):
            problems.append("no complete snapshot (missing config.json or weights)")
        return ModelArtifact(ref=ref, source="hf", local_path=str(d), revision=None, format="other",
                             weights_bytes=0, size_bytes=0, complete=False, kind="other", arch=None,
                             architecture=None, quant=None, problems=problems)
    snap, config, sizes = chosen
    try:
        arch = parse_hf_config(config)
    except Exception as e:  # never let one odd config kill the scan
        log.warning("cannot parse %s config: %s", ref, e)
        arch = None
        problems.append(f"config.json not understood: {e}")
    weights = _weights(sizes)
    fmt = "safetensors" if any(k.endswith(".safetensors") for k in sizes) else "other"
    architecture = arch.architecture if arch else None
    return ModelArtifact(
        ref=ref, source="hf", local_path=str(snap), revision=snap.name, format=fmt, weights_bytes=weights,
        size_bytes=sum(sizes.values()), complete=arch is not None and "download in progress or interrupted" not in problems,
        kind=classify_kind(architecture), arch=arch, architecture=architecture,
        quant=arch.quant if arch else None, problems=problems, config=config, files=sorted(sizes),
    )


def scan_hf_cache(hub_dir: Path) -> list[ModelArtifact]:
    if not hub_dir.is_dir():
        return []
    out = []
    for d in sorted(hub_dir.iterdir()):
        if d.is_dir() and d.name.startswith("models--") and d.name.count("--") >= 2:
            try:
                out.append(_scan_repo(d))
            except OSError as e:
                log.warning("skipping %s: %s", d, e)
    return out
```

- [ ] **Step 6: Implement `dgxmm/inventory/service.py`**

```python
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable

from dgxmm.config import Settings
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.inventory.hf_scan import scan_hf_cache

log = logging.getLogger(__name__)


class InventoryService:
    def __init__(self, settings: Settings, scanners: list[Callable[[], list[ModelArtifact]]] | None = None):
        self.settings = settings
        hub = Path(settings.hf_cache.local) / "hub"
        self.scanners = scanners or [lambda: scan_hf_cache(hub)]
        self._items: dict[str, ModelArtifact] = {}
        self._lock = threading.Lock()

    def refresh(self) -> list[ModelArtifact]:
        found: dict[str, ModelArtifact] = {}
        for scan in self.scanners:
            try:
                for a in scan():
                    found[a.ref] = a
            except Exception:
                log.exception("inventory scanner failed")
        with self._lock:
            self._items = found
        return list(found.values())

    def list(self) -> list[ModelArtifact]:
        with self._lock:
            return sorted(self._items.values(), key=lambda a: a.ref.lower())

    def get(self, ref: str) -> ModelArtifact | None:
        with self._lock:
            return self._items.get(ref)

    async def run(self) -> None:
        while True:
            await asyncio.to_thread(self.refresh)
            await asyncio.sleep(self.settings.inventory_interval_s)
```

- [ ] **Step 7: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_inventory.py`. Expected: `6 passed`.

- [ ] **Step 8: Smoke-test against the real cache (read-only)**

```bash
DGX$ .venv/bin/python -c "
from pathlib import Path; from dgxmm.inventory.hf_scan import scan_hf_cache
for a in scan_hf_cache(Path.home()/'.cache/huggingface/hub'):
    print(f'{a.ref:55} {a.kind:6} {a.quant or \"-\":7} {a.weights_bytes/2**30:7.1f}GiB complete={a.complete} {a.problems}')"
```
Expected:
- `unsloth/Qwen3.8-27B-NVFP4` is complete with revision `57926…` and a `refs/main` problem note.
- whisper and BERT show kind `other`; `incoai/Qwen3.8-27B-DFlash2` shows kind `draft`.
- repos whose download never finished are incomplete ("no snapshot").


### Task 9: Engine interface, argument guard, Prometheus parsing, vLLM engine

**Files:**
- Create: `dgxmm/engines/__init__.py` (empty), `dgxmm/engines/base.py`, `dgxmm/engines/argguard.py`, `dgxmm/engines/prom.py`, `dgxmm/engines/vllm.py`, `dgxmm/engines/registry.py`
- Test: `tests/test_engines_vllm.py`

**Interfaces:**
- Consumes: `ModelArtifact`, `estimate_vllm`, `VllmSizing`, `vllm_gpu_util`, `Settings`, `Mount`.
- Produces:
  - `ContainerSpec`, `EngineMetrics(gauges, counters, histograms)`, `LaunchContext(deployment_id, name, port, reserved, mem_total)`.
  - `container_name(dep_name)`, `managed_labels(deployment_id, engine)`, `openai_health(http, base_url, served_name, health_path="/health")`.
  - `validate_extra_args(args, allow) -> list[str]`, `VLLM_EXTRA_FLAGS`, `LLAMACPP_EXTRA_FLAGS`.
  - `parse_samples(text)`, `gauge(samples, *names)`, `counter(samples, *names)`, `buckets(samples, name)`, `histogram_quantile(buckets, q)`.
  - `SpeculativeConfig`, `VllmConfig`, `VllmEngine` implementing the Engine protocol.
  - `build_engines(settings) -> dict[str, Engine]`.
- Engine protocol (every engine implements this):
  - `name: str`, `supports_sleep: bool`, `config_model: type[BaseModel]`
  - `validate_config(raw: dict) -> BaseModel`
  - `supports(artifact) -> bool`
  - `draft_ref(cfg) -> str | None`
  - `served_name(cfg, model_ref: str) -> str`
  - `sleep_enabled(cfg) -> bool`
  - `estimate(artifact, cfg, draft, settings) -> MemoryEstimate`
  - `container_spec(artifact, cfg, draft, launch, settings) -> ContainerSpec`
  - `async health(http, base_url, served_name) -> bool`
  - `parse_metrics(text) -> EngineMetrics`
  - `async sleep(http, base_url)`, `async wake(http, base_url)`
  - `kv_patch_options(cfg) -> list[dict]`, `context_key: str`, `seqs_key: str` (used by the planner in Task 16)

- [ ] **Step 1: Write the failing tests** — `tests/test_engines_vllm.py`

```python
import json

import httpx
import pytest
from pydantic import ValidationError

from dgxmm.config import GIB
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.prom import buckets, histogram_quantile, parse_samples
from dgxmm.engines.vllm import VllmEngine
from dgxmm.inventory.hf_scan import scan_hf_cache
from pathlib import Path
from tests.conftest import QWEN_TINY

VLLM_METRICS = """# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="q"} 2.0
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="q"} 1.0
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="q"} 0.25
# TYPE vllm:prompt_tokens counter
vllm:prompt_tokens_total{engine="0",model_name="q"} 1000.0
# TYPE vllm:generation_tokens counter
vllm:generation_tokens_total{engine="0",model_name="q"} 500.0
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{engine="0",model_name="q"} 200.0
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits_total{engine="0",model_name="q"} 50.0
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="q"} 3.0
vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="q"} 9.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="q"} 10.0
vllm:time_to_first_token_seconds_count{model_name="q"} 10.0
vllm:time_to_first_token_seconds_sum{model_name="q"} 4.2
"""


@pytest.fixture
def tiny(settings, fake_hf_repo):
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    [a] = scan_hf_cache(Path(settings.hf_cache.local) / "hub")
    return a


def test_argguard():
    assert validate_extra_args(["--enforce-eager", "--seed=42", "--max-logprobs", "5"], VLLM_EXTRA_FLAGS) == \
        ["--enforce-eager", "--seed", "42", "--max-logprobs", "5"]
    for bad in (["--model", "x"], ["--seed", "1;rm -rf /"], ["--gpu-memory-utilization", "0.9"],
                ["--enforce-eager=1"], ["--seed"], ["positional"]):
        with pytest.raises(ValueError):
            validate_extra_args(bad, VLLM_EXTRA_FLAGS)


def test_config_validation(settings):
    e = VllmEngine(settings)
    cfg = e.validate_config({"max_model_len": 8192, "kv_cache_dtype": "fp8"})
    assert cfg.max_num_seqs == 8 and cfg.enable_sleep_mode is True
    for bad in ({"gpu_memory_utilization": 0.9}, {"extra_args": ["--port", "1"]}, {"served_model_name": "a b"},
                {"max_model_len": 8192, "kv_tokens": 1024}):
        with pytest.raises((ValidationError, ValueError)):
            e.validate_config(bad)


def test_container_spec(settings, tiny):
    e = VllmEngine(settings)
    cfg = e.validate_config({"max_model_len": 4096, "max_num_seqs": 2, "kv_cache_dtype": "fp8",
                             "reasoning_parser": "qwen3", "speculative": {"method": "mtp", "num_speculative_tokens": 2},
                             "extra_args": ["--enforce-eager"]})
    spec = e.container_spec(tiny, cfg, None, LaunchContext(7, "q", 8105, 30 * GIB, 120 * GIB), settings)
    c = spec.command
    assert spec.name == "dgxmm-q" and spec.image == settings.images.vllm and spec.port == 8105
    assert c[c.index("--model") + 1] == "Qwen/Qwen3-Tiny" and c[c.index("--revision") + 1] == tiny.revision
    assert c[c.index("--gpu-memory-utilization") + 1] == "0.25"
    assert c[c.index("--served-model-name") + 1] == "qwen3-tiny"
    assert json.loads(c[c.index("--speculative-config") + 1]) == {"method": "mtp", "num_speculative_tokens": 2}
    assert "--enable-sleep-mode" in c and c[-1] == "--enforce-eager"
    assert spec.environment["VLLM_SERVER_DEV_MODE"] == "1" and spec.environment["HF_HUB_OFFLINE"] == "1"
    assert spec.labels == {"dgxmm.managed": "true", "dgxmm.deployment_id": "7", "dgxmm.engine": "vllm"}
    assert spec.mounts[0].host == settings.hf_cache.host and spec.mounts[0].target == "/root/.cache/huggingface"
    assert spec.publish_bind == "127.0.0.1" and spec.network == "gateway"


def test_parse_metrics(settings):
    m = VllmEngine(settings).parse_metrics(VLLM_METRICS)
    assert m.gauges == {"running": 2.0, "waiting": 1.0, "kv_usage": 0.25}
    assert m.counters["generation_tokens"] == 500.0 and m.counters["prefix_hits"] == 50.0
    assert m.histograms["ttft"] == [(0.1, 3.0), (1.0, 9.0), (float("inf"), 10.0)]
    q = histogram_quantile(m.histograms["ttft"], 0.5)
    assert 0.1 < q < 1.0


def test_histogram_quantile_edges():
    assert histogram_quantile([], 0.5) is None
    assert histogram_quantile([(1.0, 0.0), (float("inf"), 0.0)], 0.5) is None
    assert histogram_quantile([(1.0, 10.0), (float("inf"), 10.0)], 0.5) == pytest.approx(0.5)


async def test_health_checks_health_then_completion(settings):
    seen = []

    def handler(req: httpx.Request):
        seen.append(req.url.path)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await VllmEngine(settings).health(http, "http://h:1", "q")
    assert seen == ["/health", "/v1/completions"]
```

- [ ] **Step 2: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_engines_vllm.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/engines/base.py`**

```python
"""Engine plug-in interface (spec §3.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import BaseModel

from dgxmm.config import Mount, Settings
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate

SERVED_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$"
MANAGED_LABEL = "dgxmm.managed"


@dataclass
class ContainerSpec:
    name: str
    image: str
    command: list[str]
    environment: dict[str, str]
    mounts: list[Mount]
    labels: dict[str, str]
    network: str
    port: int
    publish_bind: str | None
    gpus: bool = True
    ipc_host: bool = True
    entrypoint: list[str] | None = None


@dataclass
class EngineMetrics:
    gauges: dict[str, float] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[tuple[float, float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchContext:
    deployment_id: int
    name: str
    port: int
    reserved: int
    mem_total: int


class Engine(Protocol):
    name: str
    supports_sleep: bool
    config_model: type[BaseModel]
    context_key: str
    seqs_key: str

    def validate_config(self, raw: dict[str, Any]) -> BaseModel: ...
    def supports(self, artifact: ModelArtifact) -> bool: ...
    def draft_ref(self, cfg: Any) -> str | None: ...
    def served_name(self, cfg: Any, model_ref: str) -> str: ...
    def sleep_enabled(self, cfg: Any) -> bool: ...
    def estimate(self, artifact: ModelArtifact, cfg: Any, draft: ModelArtifact | None, settings: Settings) -> MemoryEstimate: ...
    def container_spec(self, artifact: ModelArtifact, cfg: Any, draft: ModelArtifact | None,
                       launch: LaunchContext, settings: Settings) -> ContainerSpec: ...
    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str) -> bool: ...
    def parse_metrics(self, text: str) -> EngineMetrics: ...
    async def sleep(self, http: httpx.AsyncClient, base_url: str) -> None: ...
    async def wake(self, http: httpx.AsyncClient, base_url: str) -> None: ...
    def kv_patch_options(self, cfg: Any) -> list[dict]: ...


def container_name(dep_name: str) -> str:
    return f"dgxmm-{dep_name}"


def managed_labels(deployment_id: int, engine: str) -> dict[str, str]:
    return {MANAGED_LABEL: "true", "dgxmm.deployment_id": str(deployment_id), "dgxmm.engine": engine}


def default_served_name(model_ref: str) -> str:
    import re
    stem = model_ref.rstrip("/").split("/")[-1]
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    stem = re.sub(r"-0000\d-of-0000\d$", "", stem)
    return re.sub(r"[^A-Za-z0-9._-]", "-", stem).lower()[:63] or "model"


async def openai_health(http: httpx.AsyncClient, base_url: str, served_name: str, health_path: str = "/health") -> bool:
    r = await http.get(f"{base_url}{health_path}", timeout=5)
    if r.status_code != 200:
        return False
    r = await http.post(f"{base_url}/v1/completions",
                        json={"model": served_name, "prompt": "Hi", "max_tokens": 1}, timeout=120)
    return r.status_code == 200
```

- [ ] **Step 4: Implement `dgxmm/engines/argguard.py`**

```python
"""Allowlisted extra CLI flags. DGXMM-managed flags (--model, --port, --host,
--gpu-memory-utilization, …) are deliberately absent so users cannot override them."""
from __future__ import annotations

import re

_JSON_OBJ = r"\{[^;&|`$<>\\]{0,2000}\}"
VLLM_EXTRA_FLAGS: dict[str, str | None] = {
    "--enforce-eager": None,
    "--disable-log-requests": None,
    "--disable-log-stats": None,
    "--async-scheduling": None,
    "--disable-custom-all-reduce": None,
    "--seed": r"\d{1,10}",
    "--max-logprobs": r"\d{1,4}",
    "--max-seq-len-to-capture": r"\d{1,7}",
    "--dtype": r"auto|bfloat16|float16",
    "--quantization": r"[a-z0-9_]{1,32}",
    "--load-format": r"[a-z0-9_]{1,32}",
    "--generation-config": r"auto|vllm",
    "--chat-template-content-format": r"auto|string|openai",
    "--limit-mm-per-prompt": _JSON_OBJ,
    "--compilation-config": _JSON_OBJ,
    "--hf-overrides": _JSON_OBJ,
    "--override-generation-config": _JSON_OBJ,
    "--mm-processor-kwargs": _JSON_OBJ,
    "--mamba-ssm-cache-dtype": r"auto|float32|bfloat16",
}
LLAMACPP_EXTRA_FLAGS: dict[str, str | None] = {
    "--threads": r"\d{1,3}",
    "--threads-batch": r"\d{1,3}",
    "--mlock": None,
    "--no-mmap": None,
    "--jinja": None,
    "--cont-batching": None,
    "--no-warmup": None,
    "--kv-unified": None,
    "--swa-full": None,
    "--seed": r"\d{1,10}",
    "--reasoning-format": r"[a-z_]{1,32}",
    "--reasoning-budget": r"-?\d{1,7}",
    "--chat-template": r"[A-Za-z0-9_.-]{1,64}",
    "--rope-scaling": r"none|linear|yarn",
    "--rope-freq-scale": r"\d+(\.\d+)?",
    "--defrag-thold": r"\d+(\.\d+)?",
    "--temp": r"\d+(\.\d+)?",
}


def validate_extra_args(args: list[str], allow: dict[str, str | None]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(args):
        flag, eq, val = str(args[i]).partition("=")
        if flag not in allow:
            raise ValueError(f"flag not allowed: {flag}")
        pattern = allow[flag]
        if pattern is None:
            if eq:
                raise ValueError(f"{flag} takes no value")
            out.append(flag)
            i += 1
            continue
        if eq:
            i += 1
        else:
            if i + 1 >= len(args):
                raise ValueError(f"{flag} needs a value")
            val = str(args[i + 1])
            i += 2
        if not re.fullmatch(pattern, val):
            raise ValueError(f"invalid value for {flag}")
        out += [flag, val]
    return out
```

- [ ] **Step 5: Implement `dgxmm/engines/prom.py`**

```python
"""Prometheus text helpers shared by engines."""
from __future__ import annotations

import math
from collections import defaultdict

from prometheus_client.parser import text_string_to_metric_families

Samples = dict[str, list[tuple[dict[str, str], float]]]


def parse_samples(text: str) -> Samples:
    out: Samples = defaultdict(list)
    for fam in text_string_to_metric_families(text):
        for s in fam.samples:
            out[s.name].append((dict(s.labels), float(s.value)))
    return out


def gauge(samples: Samples, *names: str) -> float | None:
    for n in names:
        if n in samples:
            return sum(v for _, v in samples[n])
    return None


def counter(samples: Samples, *names: str) -> float | None:
    for n in names:
        for candidate in (n + "_total", n):
            if candidate in samples:
                return sum(v for _, v in samples[candidate])
    return None


def buckets(samples: Samples, *names: str) -> list[tuple[float, float]]:
    for n in names:
        rows = samples.get(n + "_bucket")
        if rows:
            agg: dict[float, float] = defaultdict(float)
            for labels, v in rows:
                le = labels.get("le", "+Inf")
                agg[math.inf if le in ("+Inf", "inf") else float(le)] += v
            return sorted(agg.items())
    return []


def histogram_quantile(bkts: list[tuple[float, float]], q: float) -> float | None:
    """Prometheus-style linear interpolation over cumulative (le, count) buckets."""
    if not bkts or bkts[-1][1] <= 0:
        return None
    rank = q * bkts[-1][1]
    prev_le, prev_count = 0.0, 0.0
    for le, count in bkts:
        if count >= rank:
            if math.isinf(le):
                return prev_le
            if count == prev_count:
                return le
            return prev_le + (le - prev_le) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return prev_le


def delta_buckets(cur: list[tuple[float, float]], prev: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    if not prev:
        return cur
    p = dict(prev)
    return [(le, max(c - p.get(le, 0.0), 0.0)) for le, c in cur]
```

- [ ] **Step 6: Implement `dgxmm/engines/vllm.py`**

```python
"""vLLM engine (OpenAI server in vllm/vllm-openai)."""
from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dgxmm.config import Mount, Settings
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import (SERVED_NAME_RE, ContainerSpec, EngineMetrics, LaunchContext, container_name,
                                default_served_name, managed_labels, openai_health)
from dgxmm.engines.prom import buckets, counter, gauge, parse_samples
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate, VllmSizing, estimate_vllm, vllm_gpu_util

PARSER_RE = r"^[a-z0-9_]{1,32}$"


class SpeculativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["dflash", "eagle", "eagle3", "mtp", "ngram", "draft_model"]
    model: str | None = Field(None, pattern=REPO_RE)
    num_speculative_tokens: int = Field(3, ge=1, le=16)

    @model_validator(mode="after")
    def _needs_model(self):
        if self.method in ("dflash", "eagle", "eagle3", "draft_model") and not self.model:
            raise ValueError(f"speculative method {self.method} needs a draft model")
        return self


class VllmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    served_model_name: str | None = Field(None, pattern=SERVED_NAME_RE)
    max_model_len: int = Field(32768, ge=256, le=1_048_576)
    max_num_seqs: int = Field(8, ge=1, le=1024)
    max_num_batched_tokens: int = Field(8192, ge=256, le=262_144)
    kv_tokens: int | None = Field(None, ge=256, le=64_000_000)
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = "auto"
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    enable_auto_tool_choice: bool = False
    tool_call_parser: str | None = Field(None, pattern=PARSER_RE)
    reasoning_parser: str | None = Field(None, pattern=PARSER_RE)
    speculative: SpeculativeConfig | None = None
    enable_sleep_mode: bool = True
    trust_remote_code: bool = False
    safetensors_load_strategy: Literal["lazy", "eager", "prefetch"] | None = None
    extra_args: list[str] = []

    @field_validator("extra_args")
    @classmethod
    def _extra(cls, v: list[str]) -> list[str]:
        return validate_extra_args(v, VLLM_EXTRA_FLAGS)

    @model_validator(mode="after")
    def _pool(self):
        if self.kv_tokens is not None and self.kv_tokens < self.max_model_len:
            raise ValueError("kv_tokens must be >= max_model_len")
        return self


VLLM_METRICS = {
    "gauges": {"running": ("vllm:num_requests_running",), "waiting": ("vllm:num_requests_waiting",),
               "kv_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")},
    "counters": {"prompt_tokens": ("vllm:prompt_tokens",), "generation_tokens": ("vllm:generation_tokens",),
                 "prefix_queries": ("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"),
                 "prefix_hits": ("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"),
                 "spec_draft": ("vllm:spec_decode_num_draft_tokens",),
                 "spec_accepted": ("vllm:spec_decode_num_accepted_tokens",),
                 "preemptions": ("vllm:num_preemptions",), "requests": ("vllm:request_success",)},
    "histograms": {"ttft": ("vllm:time_to_first_token_seconds",),
                   "itl": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
                   "e2e": ("vllm:e2e_request_latency_seconds",)},
}


def metrics_from(text: str, table: dict) -> EngineMetrics:
    s = parse_samples(text)
    m = EngineMetrics()
    for key, names in table["gauges"].items():
        v = gauge(s, *names)
        if v is not None:
            m.gauges[key] = v
    for key, names in table["counters"].items():
        v = counter(s, *names)
        if v is not None:
            m.counters[key] = v
    for key, names in table.get("histograms", {}).items():
        b = buckets(s, *names)
        if b:
            m.histograms[key] = b
    return m


class VllmEngine:
    name = "vllm"
    supports_sleep = True
    config_model = VllmConfig
    context_key = "max_model_len"
    seqs_key = "max_num_seqs"

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_config(self, raw: dict[str, Any]) -> VllmConfig:
        return VllmConfig.model_validate(raw)

    def supports(self, artifact: ModelArtifact) -> bool:
        return artifact.source == "hf" and artifact.format == "safetensors" and artifact.kind == "llm"

    def draft_ref(self, cfg: VllmConfig) -> str | None:
        return cfg.speculative.model if cfg.speculative and cfg.speculative.model else None

    def served_name(self, cfg: VllmConfig, model_ref: str) -> str:
        return cfg.served_model_name or default_served_name(model_ref)

    def sleep_enabled(self, cfg: VllmConfig) -> bool:
        return cfg.enable_sleep_mode

    @staticmethod
    def sizing(cfg: VllmConfig) -> VllmSizing:
        return VllmSizing(cfg.max_model_len, cfg.max_num_seqs, cfg.max_num_batched_tokens, cfg.kv_cache_dtype, cfg.kv_tokens)

    def estimate(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None, settings: Settings) -> MemoryEstimate:
        if artifact.arch is None:
            raise ValueError("model config.json could not be parsed")
        return estimate_vllm(artifact.arch, artifact.weights_bytes, self.sizing(cfg), settings,
                             draft_spec=draft.arch if draft else None, draft_bytes=draft.weights_bytes if draft else 0)

    def container_spec(self, artifact: ModelArtifact, cfg: VllmConfig, draft: ModelArtifact | None,
                       launch: LaunchContext, settings: Settings) -> ContainerSpec:
        util = vllm_gpu_util(launch.reserved, launch.mem_total)
        cmd = ["--model", artifact.ref]
        if artifact.revision:
            cmd += ["--revision", artifact.revision]
        cmd += ["--served-model-name", self.served_name(cfg, artifact.ref), "--host", "0.0.0.0", "--port", str(launch.port),
                "--max-model-len", str(cfg.max_model_len), "--max-num-seqs", str(cfg.max_num_seqs),
                "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
                "--gpu-memory-utilization", f"{util:.2f}", "--kv-cache-dtype", cfg.kv_cache_dtype,
                "--enable-prefix-caching" if cfg.enable_prefix_caching else "--no-enable-prefix-caching",
                "--enable-chunked-prefill" if cfg.enable_chunked_prefill else "--no-enable-chunked-prefill"]
        if cfg.enable_auto_tool_choice:
            cmd.append("--enable-auto-tool-choice")
        if cfg.tool_call_parser:
            cmd += ["--tool-call-parser", cfg.tool_call_parser]
        if cfg.reasoning_parser:
            cmd += ["--reasoning-parser", cfg.reasoning_parser]
        if cfg.speculative:
            sc: dict[str, Any] = {"method": cfg.speculative.method,
                                  "num_speculative_tokens": cfg.speculative.num_speculative_tokens}
            if cfg.speculative.model:
                sc["model"] = cfg.speculative.model
            cmd += ["--speculative-config", json.dumps(sc)]
        if cfg.enable_sleep_mode:
            cmd.append("--enable-sleep-mode")
        if cfg.trust_remote_code:
            cmd.append("--trust-remote-code")
        if cfg.safetensors_load_strategy:
            cmd += ["--safetensors-load-strategy", cfg.safetensors_load_strategy]
        cmd += cfg.extra_args
        env = {"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        if cfg.enable_sleep_mode:
            env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /sleep and /wake_up
        mounts = [Mount(host=settings.hf_cache.host, target="/root/.cache/huggingface"), *settings.vllm_cache_mounts]
        return ContainerSpec(name=container_name(launch.name), image=settings.images.vllm, command=cmd, environment=env,
                             mounts=mounts, labels=managed_labels(launch.deployment_id, self.name),
                             network=settings.docker_network, port=launch.port, publish_bind=settings.publish_bind)

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str) -> bool:
        return await openai_health(http, base_url, served_name)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return metrics_from(text, VLLM_METRICS)

    async def sleep(self, http: httpx.AsyncClient, base_url: str) -> None:
        (await http.post(f"{base_url}/sleep", params={"level": 1}, timeout=120)).raise_for_status()

    async def wake(self, http: httpx.AsyncClient, base_url: str) -> None:
        (await http.post(f"{base_url}/wake_up", timeout=300)).raise_for_status()

    def kv_patch_options(self, cfg: VllmConfig) -> list[dict]:
        return [{"kv_cache_dtype": "fp8"}] if cfg.kv_cache_dtype == "auto" else []
```

`dgxmm/engines/registry.py`:
```python
from dgxmm.config import Settings
from dgxmm.engines.vllm import VllmEngine


def build_engines(settings: Settings) -> dict:
    engines = {"vllm": VllmEngine(settings)}
    return engines
```

- [ ] **Step 7: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_engines_vllm.py`. Expected: all pass.

---

### Task 10: Docker controller with the managed-label guard

**Files:**
- Create: `dgxmm/docker_ctl.py`, `tests/fakes.py`
- Test: `tests/test_docker_ctl.py`

**Interfaces:**
- Consumes: `ContainerSpec`, `MANAGED_LABEL`.
- Produces:
  - `ContainerInfo(id, name, status, exit_code, labels, image, command, mounts, networks, host_ports)` with property `managed`.
  - `DockerController(client)`:
    - `from_env()` (classmethod)
    - `list_all() -> list[ContainerInfo]`, `list_managed() -> list[ContainerInfo]`, `get(id_or_name) -> ContainerInfo | None`
    - `run(spec) -> str` (container id)
    - `stop_and_remove(cid, timeout=30)`, `remove_by_name(name)`
    - `logs(cid, tail=200) -> str`
    - `run_oneshot(image, entrypoint, command) -> str`
  - Test fakes: `FakeDockerClient` (with `.run_calls`, `.fail_run`, `.oneshot_output`, `.add_unmanaged(...)`, `.container(id)`), `FakeProbe`, `FakeEngines`.

- [ ] **Step 1: Write the fakes** — `tests/fakes.py`

```python
"""Test doubles: Docker SDK client, host probe, engine HTTP endpoints."""
from __future__ import annotations

import itertools

import docker.errors
import httpx

from dgxmm.config import GIB
from dgxmm.memory.hostmem import HostMemory, HostSample

_ids = itertools.count(1)


class FakeContainer:
    def __init__(self, client, cid, name, image, command, labels, env, ports, networks, mounts, status="running"):
        self.client, self.id, self.name, self.image_name = client, cid, name, image
        self.command, self.labels, self.env, self.ports = command or [], labels or {}, env or {}, ports
        self.networks, self.mounts, self.status, self.exit_code = networks, mounts, status, None
        self.log_text, self.stopped = "", False

    @property
    def attrs(self):
        return {
            "State": {"Status": self.status, "ExitCode": self.exit_code if self.exit_code is not None else 0},
            "Config": {"Image": self.image_name, "Cmd": self.command, "Labels": self.labels,
                       "Env": [f"{k}={v}" for k, v in self.env.items()]},
            "Mounts": [{"Source": h, "Destination": t, "Mode": m} for h, t, m in self.mounts],
            "NetworkSettings": {"Networks": {n: {} for n in self.networks},
                                "Ports": {f"{p}/tcp": [{"HostIp": b, "HostPort": str(p)}] for p, b in self.ports}},
        }

    def reload(self):
        pass

    def stop(self, timeout=10):
        self.stopped = True
        self.status = "exited"
        if self.exit_code is None:
            self.exit_code = 0

    def remove(self, force=False):
        self.client._containers.pop(self.id, None)

    def logs(self, tail=200, **_):
        return self.log_text.encode()


class FakeContainers:
    def __init__(self, client):
        self.client = client

    def list(self, all=False, filters=None):
        items = list(self.client._containers.values())
        if filters and "label" in filters:
            k, _, v = filters["label"].partition("=")
            items = [c for c in items if c.labels.get(k) == v]
        return items

    def get(self, key):
        for c in self.client._containers.values():
            if key in (c.id, c.name):
                return c
        raise docker.errors.NotFound("no such container")

    def run(self, image, command=None, name=None, environment=None, labels=None, network=None, volumes=None,
            ports=None, entrypoint=None, remove=False, **kw):
        self.client.run_calls.append(dict(image=image, command=command, name=name, environment=environment,
                                          labels=labels, network=network, volumes=volumes, ports=ports,
                                          entrypoint=entrypoint, remove=remove, **kw))
        if remove:
            return self.client.oneshot_output.encode()
        if self.client.fail_run:
            raise docker.errors.APIError("boom")
        if any(c.name == name for c in self.client._containers.values()):
            raise docker.errors.APIError("Conflict. The container name is already in use")
        cid = f"{next(_ids):064x}"
        mounts = [(h, v["bind"], v.get("mode", "rw")) for h, v in (volumes or {}).items()]
        plist = [(int(k.split("/")[0]), v[0]) for k, v in (ports or {}).items()]
        c = FakeContainer(self.client, cid, name, image, command, labels, environment, plist,
                          [network] if network else [], mounts)
        self.client._containers[cid] = c
        return c


class FakeDockerClient:
    def __init__(self):
        self._containers: dict[str, FakeContainer] = {}
        self.containers = FakeContainers(self)
        self.run_calls: list[dict] = []
        self.fail_run = False
        self.oneshot_output = ""

    def add_unmanaged(self, name, image="vllm/vllm-openai:x", command=None, labels=None, ports=(), status="running",
                      mounts=(), networks=("gateway",), env=None):
        cid = f"{next(_ids):064x}"
        c = FakeContainer(self, cid, name, image, command, labels or {}, env or {}, list(ports), list(networks),
                          list(mounts), status=status)
        self._containers[cid] = c
        return c

    def container(self, cid) -> FakeContainer:
        return self._containers[cid]


class FakeProbe:
    def __init__(self, total_gib: float = 121, available_gib: float = 100):
        self.total, self.available = int(total_gib * GIB), int(available_gib * GIB)
        self.managed_actual: dict[str, int] = {}

    def meminfo(self) -> HostMemory:
        return HostMemory(total=self.total, available=self.available, free=0, cached=0, swap_total=0, swap_free=0)

    def sample(self, ids):
        return HostSample(mem=self.meminfo(), managed={c: self.managed_actual.get(c, 0) for c in ids},
                          external_top=[], gpu={})


class FakeEngines:
    """Answers engine HTTP calls by port: /health, /v1/completions, /metrics, /sleep, /wake_up."""

    def __init__(self):
        self.healthy: set[int] = set()
        self.metrics: dict[int, str] = {}
        self.sleeping: set[int] = set()
        self.fail_paths: set[str] = set()
        self.calls: list[tuple[str, int, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        port, path = request.url.port, request.url.path
        self.calls.append((request.method, port, path))
        if path in self.fail_paths:
            return httpx.Response(500)
        if path == "/health":
            return httpx.Response(200 if port in self.healthy else 503)
        if path == "/v1/completions":
            return httpx.Response(200 if port in self.healthy else 503, json={"choices": [{"text": "x"}]})
        if path == "/metrics":
            return httpx.Response(200, text=self.metrics.get(port, ""))
        if path == "/sleep":
            self.sleeping.add(port)
            return httpx.Response(200)
        if path == "/wake_up":
            self.sleeping.discard(port)
            return httpx.Response(200)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
```

- [ ] **Step 2: Write the failing tests** — `tests/test_docker_ctl.py`

```python
import pytest

from dgxmm.config import Mount
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.base import ContainerSpec, managed_labels
from tests.fakes import FakeDockerClient


def spec(name="dgxmm-a", labels=None):
    return ContainerSpec(name=name, image="img", command=["--x"], environment={"A": "1"},
                         mounts=[Mount(host="/h", target="/t", mode="ro")], labels=labels or managed_labels(1, "vllm"),
                         network="gateway", port=8100, publish_bind="127.0.0.1")


def test_run_passes_expected_sdk_kwargs():
    sdk = FakeDockerClient()
    d = DockerController(sdk)
    cid = d.run(spec())
    call = sdk.run_calls[0]
    assert call["volumes"] == {"/h": {"bind": "/t", "mode": "ro"}}
    assert call["ports"] == {"8100/tcp": ("127.0.0.1", 8100)}
    assert call["ipc_mode"] == "host" and call["restart_policy"] == {"Name": "no"}
    assert call["device_requests"][0]["Capabilities"] == [["gpu"]]
    info = d.get(cid)
    assert info.managed and info.host_ports == {8100} and info.networks == ["gateway"]


def test_guard_refuses_unmanaged_containers():
    sdk = FakeDockerClient()
    other = sdk.add_unmanaged("litellm-proxy")
    d = DockerController(sdk)
    with pytest.raises(PermissionError):
        d.stop_and_remove(other.id)
    with pytest.raises(PermissionError):
        d.run(spec(labels={"foo": "bar"}))
    with pytest.raises(PermissionError):
        d.remove_by_name("litellm-proxy")
    assert sdk.container(other.id).stopped is False


def test_stop_and_remove_managed_and_missing():
    sdk = FakeDockerClient()
    d = DockerController(sdk)
    cid = d.run(spec())
    d.stop_and_remove(cid)
    assert d.get(cid) is None
    d.stop_and_remove(cid)       # already gone: no error
    d.stop_and_remove(None)


def test_list_managed_and_oneshot():
    sdk = FakeDockerClient()
    sdk.add_unmanaged("tts-service")
    d = DockerController(sdk)
    d.run(spec())
    assert [c.name for c in d.list_managed()] == ["dgxmm-a"]
    assert len(d.list_all()) == 2
    sdk.oneshot_output = "ARCHS=[]\n"
    assert d.run_oneshot("img", ["python3", "-c"], ["print(1)"]) == "ARCHS=[]\n"
    assert sdk.run_calls[-1]["remove"] is True
```

- [ ] **Step 3: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_docker_ctl.py`. Expected: FAIL (ImportError).

- [ ] **Step 4: Implement `dgxmm/docker_ctl.py`**

```python
"""The only module that talks to the Docker daemon. Writes are limited to dgxmm-managed containers."""
from __future__ import annotations

from dataclasses import dataclass, field

import docker
import docker.errors
from docker.types import DeviceRequest

from dgxmm.engines.base import MANAGED_LABEL, ContainerSpec


@dataclass
class ContainerInfo:
    id: str
    name: str
    status: str
    exit_code: int | None
    labels: dict[str, str]
    image: str
    command: list[str]
    mounts: list[dict] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    host_ports: set[int] = field(default_factory=set)

    @property
    def managed(self) -> bool:
        return self.labels.get(MANAGED_LABEL) == "true"

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "status": self.status, "exit_code": self.exit_code,
                "image": self.image, "command": self.command, "managed": self.managed,
                "host_ports": sorted(self.host_ports), "networks": self.networks}


def _info(c) -> ContainerInfo:
    a = c.attrs
    state, cfg = a.get("State", {}), a.get("Config", {})
    ports: set[int] = set()
    for bindings in (a.get("NetworkSettings", {}).get("Ports") or {}).values():
        for b in bindings or []:
            if b.get("HostPort"):
                ports.add(int(b["HostPort"]))
    status = state.get("Status", getattr(c, "status", "unknown"))
    return ContainerInfo(
        id=c.id, name=c.name.lstrip("/"), status=status,
        exit_code=state.get("ExitCode") if status in ("exited", "dead") else None,
        labels=dict(cfg.get("Labels") or {}), image=cfg.get("Image", ""), command=list(cfg.get("Cmd") or []),
        mounts=[{"source": m.get("Source"), "target": m.get("Destination"), "mode": m.get("Mode", "")}
                for m in a.get("Mounts") or []],
        networks=list((a.get("NetworkSettings", {}).get("Networks") or {}).keys()), host_ports=ports,
    )


class DockerController:
    def __init__(self, client):
        self.client = client

    @classmethod
    def from_env(cls) -> "DockerController":
        return cls(docker.from_env())

    def list_all(self) -> list[ContainerInfo]:
        return [_info(c) for c in self.client.containers.list(all=True)]

    def list_managed(self) -> list[ContainerInfo]:
        return [_info(c) for c in self.client.containers.list(all=True, filters={"label": f"{MANAGED_LABEL}=true"})]

    def get(self, id_or_name: str | None) -> ContainerInfo | None:
        if not id_or_name:
            return None
        try:
            c = self.client.containers.get(id_or_name)
            c.reload()
            return _info(c)
        except docker.errors.NotFound:
            return None

    def _guard(self, id_or_name: str):
        c = self.client.containers.get(id_or_name)
        if (c.labels or {}).get(MANAGED_LABEL) != "true":
            raise PermissionError(f"container {c.name} is not managed by dgxmm")
        return c

    def run(self, spec: ContainerSpec) -> str:
        if spec.labels.get(MANAGED_LABEL) != "true":
            raise PermissionError("refusing to create a container without the dgxmm.managed label")
        kwargs: dict = dict(image=spec.image, command=spec.command, name=spec.name, detach=True,
                            environment=spec.environment, labels=spec.labels, network=spec.network,
                            volumes={m.host: {"bind": m.target, "mode": m.mode} for m in spec.mounts},
                            restart_policy={"Name": "no"})
        if spec.entrypoint is not None:
            kwargs["entrypoint"] = spec.entrypoint
        if spec.publish_bind:
            kwargs["ports"] = {f"{spec.port}/tcp": (spec.publish_bind, spec.port)}
        if spec.gpus:
            kwargs["device_requests"] = [DeviceRequest(count=-1, capabilities=[["gpu"]])]
        if spec.ipc_host:
            kwargs["ipc_mode"] = "host"
        return self.client.containers.run(**kwargs).id

    def stop_and_remove(self, cid: str | None, timeout: int = 30) -> None:
        if not cid:
            return
        try:
            c = self._guard(cid)
        except docker.errors.NotFound:
            return
        try:
            c.stop(timeout=timeout)
        except docker.errors.APIError:
            pass
        try:
            c.remove(force=True)
        except docker.errors.NotFound:
            pass

    def remove_by_name(self, name: str) -> None:
        try:
            c = self._guard(name)
        except docker.errors.NotFound:
            return
        c.remove(force=True)

    def logs(self, cid: str | None, tail: int = 200) -> str:
        if not cid:
            return ""
        try:
            return self.client.containers.get(cid).logs(tail=tail).decode(errors="replace")
        except docker.errors.NotFound:
            return ""

    def run_oneshot(self, image: str, entrypoint: list[str], command: list[str]) -> str:
        out = self.client.containers.run(image, command=command, entrypoint=entrypoint, remove=True,
                                         stdout=True, stderr=False,
                                         labels={MANAGED_LABEL: "true", "dgxmm.oneshot": "true"})
        return out.decode(errors="replace") if isinstance(out, bytes) else str(out)
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_docker_ctl.py`. Expected: `4 passed`.

---

### Task 11: Deployer (state machine, pre-flight, launch, monitor, stop, restart)

**Files:**
- Create: `dgxmm/deployer/__init__.py` (empty), `dgxmm/deployer/states.py`, `dgxmm/deployer/service.py`
- Modify: `tests/conftest.py` (add the `stack` fixture)
- Test: `tests/test_deployer.py`

**Interfaces:**
- Consumes: `Database`, `Deployment`, `DeploymentEvent`, `DockerController`, `InventoryService`, engines from `build_engines`, `BudgetService`, `check_fit`, `FitResult`, `MemoryEstimate`, `LaunchContext`, `container_name`, `EventBroker`, `REPO_RE`.
- Produces:
  - `STATES`, `TRANSITIONS`, `ACTIVE_STATES`, `InvalidTransition`, `assert_transition(a, b)`.
  - `DeployRequest(name, engine, model_ref, config, priority, alias)`.
  - `DeployError(code, message, status)`, `InsufficientMemory(fit)` (subclass, `code="insufficient_memory"`, `status=409`).
  - `Prepared(engine, artifact, cfg, draft, estimate)`, `config_hash(cfg_dict) -> str`.
  - `Deployer` with:
    - `prepare(engine, model_ref, raw_cfg) -> Prepared`
    - `async preview(engine, model_ref, raw_cfg, replacing=None) -> dict`
    - `async deploy(req, user_id) -> dict`
    - `async stop(dep_id, user_id=None, reason="user") -> dict`
    - `async restart(dep_id, user_id=None, config_override=None, model_ref=None) -> dict`
    - `async remove(dep_id)`, `async wait_ready(dep_id, timeout) -> str`
    - `view(dep_id, with_events=False) -> dict`, `list_views() -> list[dict]`
    - `usage_rows() -> list[UsageRow]`, `resume_monitor(dep_id)`, `base_url(name, port) -> str`
    - attributes `state_hooks: list[Callable[[int, str, str], Awaitable[None]]]` (dep_id, from_state, to_state) and `calibration` (optional, set in Task 32)
    - `async aclose()`

- [ ] **Step 1: Add the `stack` fixture to `tests/conftest.py`** (append)

```python
from types import SimpleNamespace

from dgxmm.config import GIB


@pytest.fixture
def stack(settings, db, fake_hf_repo):
    from dgxmm.deployer.service import Deployer
    from dgxmm.docker_ctl import DockerController
    from dgxmm.engines.registry import build_engines
    from dgxmm.inventory.service import InventoryService
    from dgxmm.memory.budget import BudgetService
    from tests.fakes import FakeDockerClient, FakeEngines, FakeProbe

    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY, files={"model.safetensors": 2 * GIB})
    fake_hf_repo("org/big-model", config=QWEN_TINY, files={"model.safetensors": 45 * GIB})
    fake_hf_repo("org/partial", config=QWEN_TINY, incomplete=True)
    inv = InventoryService(settings)
    inv.refresh()
    sdk, probe, engines_http = FakeDockerClient(), FakeProbe(), FakeEngines()
    events = EventBroker()
    deployer = Deployer(settings=settings, db=db, docker=DockerController(sdk), inventory=inv,
                        engines=build_engines(settings), http=engines_http.client(), events=events)
    deployer.budget = BudgetService(settings, probe, deployer.usage_rows)
    return SimpleNamespace(settings=settings, db=db, inv=inv, sdk=sdk, probe=probe, http=engines_http,
                           deployer=deployer, budget=deployer.budget, events=events)
```

- [ ] **Step 2: Write the failing tests** — `tests/test_deployer.py`

```python
import asyncio

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name="tiny", ref="Qwen/Qwen3-Tiny", **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg})


async def become_ready(stack, dep):
    stack.http.healthy.add(dep["port"])
    return await stack.deployer.wait_ready(dep["id"], 5)


async def test_deploy_reaches_ready_with_computed_util(stack):
    dep = await stack.deployer.deploy(req(), user_id=1)
    assert dep["state"] in ("starting", "loading") and dep["port"] == 8100
    assert await become_ready(stack, dep) == "ready"
    call = stack.sdk.run_calls[0]
    cmd = call["command"]
    util = float(cmd[cmd.index("--gpu-memory-utilization") + 1])
    assert util == pytest.approx(dep["reserved_bytes"] / stack.probe.total, abs=0.011)
    with stack.db.session() as s:
        states = [e.to_state for e in s.scalars(select(DeploymentEvent).order_by(DeploymentEvent.id))]
    assert states == ["starting", "loading", "ready"]


async def test_refuses_when_not_enough_memory(stack):
    stack.probe.available = 20 * GIB  # 101 GiB used by something external
    with pytest.raises(InsufficientMemory) as e:
        await stack.deployer.deploy(req(ref="org/big-model"), user_id=1)
    assert e.value.fit.required > e.value.fit.free_for_new
    assert stack.sdk.run_calls == []


async def test_concurrent_deploys_cannot_both_claim_memory(stack):
    # free_for_new = 121 - 12 - 21 external = 88 GiB; each big deploy needs ~50 GiB
    results = await asyncio.gather(stack.deployer.deploy(req("a", "org/big-model"), 1),
                                   stack.deployer.deploy(req("b", "org/big-model"), 1), return_exceptions=True)
    assert sum(isinstance(r, InsufficientMemory) for r in results) == 1
    assert len(stack.sdk.run_calls) == 1


async def test_exit_137_marks_failed_likely_oom(stack):
    dep = await stack.deployer.deploy(req(), 1)
    c = stack.sdk.container(dep["container_id"])
    c.log_text = "CUDA out of memory"
    c.status, c.exit_code = "exited", 137
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    v = stack.deployer.view(dep["id"], with_events=True)
    assert v["exit_code"] == 137 and "likely out of memory" in v["last_error"]
    assert "CUDA out of memory" in v["events"][0]["log_tail"]


async def test_memory_drop_during_load_aborts(stack):
    dep = await stack.deployer.deploy(req(), 1)
    stack.probe.available = 4 * GIB
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    assert "safety margin" in stack.deployer.view(dep["id"])["last_error"]
    assert stack.sdk._containers == {}


async def test_startup_timeout(stack):
    stack.settings.startup_timeout_s = 0
    dep = await stack.deployer.deploy(req(), 1)
    assert await stack.deployer.wait_ready(dep["id"], 5) == "failed"
    assert "timed out" in stack.deployer.view(dep["id"])["last_error"]


async def test_stop_then_restart_reuses_row(stack):
    dep = await stack.deployer.deploy(req(), 1)
    await become_ready(stack, dep)
    stopped = await stack.deployer.stop(dep["id"], 1)
    assert stopped["state"] == "stopped" and stopped["container_id"] is None and stack.sdk._containers == {}
    again = await stack.deployer.restart(dep["id"], 1, config_override={**SMALL, "max_model_len": 2048})
    assert again["state"] in ("starting", "loading") and again["config"]["max_model_len"] == 2048
    assert await become_ready(stack, again) == "ready"


async def test_input_validation(stack):
    with pytest.raises(ValidationError):
        DeployRequest(name="bad name;", engine="vllm", model_ref="Qwen/Qwen3-Tiny")
    for ref, code in (("../../etc", "invalid_model_ref"), ("nope/nope", "model_not_found"),
                      ("org/partial", "model_incomplete")):
        with pytest.raises(DeployError) as e:
            await stack.deployer.deploy(req(ref=ref), 1)
        assert e.value.code == code
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(extra_args=["--seed", "1;rm"]), 1)
    assert e.value.code == "invalid_config" and e.value.status == 422
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(engine_typo=1), 1)
    assert e.value.code == "invalid_config"


async def test_name_in_use_and_port_allocation(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    assert (a["port"], b["port"]) == (8100, 8101)
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req("a"), 1)
    assert e.value.code == "name_in_use"


async def test_docker_failure_marks_failed(stack):
    stack.sdk.fail_run = True
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(), 1)
    assert e.value.code == "docker_error"
    with stack.db.session() as s:
        assert s.scalar(select(Deployment)).state == "failed"


async def test_preview_reports_util_and_fit(stack):
    p = await stack.deployer.preview("vllm", "Qwen/Qwen3-Tiny", SMALL)
    assert p["fit"]["fits"] and 0 < p["gpu_memory_utilization"] < 1
    assert p["served_name"] == "qwen3-tiny"


async def test_state_hooks_called(stack):
    seen = []

    async def hook(dep_id, a, b):
        seen.append(b)

    stack.deployer.state_hooks.append(hook)
    dep = await stack.deployer.deploy(req(), 1)
    await become_ready(stack, dep)
    await asyncio.sleep(0.05)
    assert seen[-1] == "ready"
```

- [ ] **Step 3: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_deployer.py`. Expected: FAIL (ImportError).

- [ ] **Step 4: Implement `dgxmm/deployer/states.py`**

```python
STATES = ("planned", "starting", "loading", "ready", "sleeping", "stopping", "stopped", "failed", "unknown")
TRANSITIONS: dict[str, set[str]] = {
    "planned": {"starting", "failed", "stopped"},
    "starting": {"loading", "ready", "failed", "stopping"},
    "loading": {"ready", "failed", "stopping"},
    "ready": {"sleeping", "stopping", "failed"},
    "sleeping": {"ready", "stopping", "failed"},
    "stopping": {"stopped", "failed"},
    "stopped": {"starting"},
    "failed": {"starting", "stopped"},
    "unknown": {"stopping", "failed", "ready"},
}
ACTIVE_STATES = frozenset({"starting", "loading", "ready", "sleeping", "stopping", "unknown"})


class InvalidTransition(Exception):
    pass


def assert_transition(current: str, target: str) -> None:
    if target not in TRANSITIONS.get(current, set()):
        raise InvalidTransition(f"cannot go from {current} to {target}")
```

- [ ] **Step 5: Implement `dgxmm/deployer/service.py`**

```python
"""Deployment lifecycle. The only caller of DockerController.run/stop_and_remove."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from dgxmm.config import Settings
from dgxmm.db import Database, Deployment, DeploymentEvent
from dgxmm.deployer.states import ACTIVE_STATES, assert_transition
from dgxmm.engines.base import LaunchContext, container_name
from dgxmm.events import EventBroker
from dgxmm.inventory.artifact import REPO_RE, ModelArtifact
from dgxmm.memory.estimate import MemoryEstimate, vllm_gpu_util
from dgxmm.planner.fit import FitResult, check_fit

log = logging.getLogger(__name__)
NAME_PATTERN = r"^[a-z0-9][a-z0-9-]{0,40}$"
ALIAS_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$"


class DeployRequest(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    engine: str = Field(pattern=r"^[a-z]{1,16}$")
    model_ref: str = Field(max_length=512)
    config: dict[str, Any] = {}
    priority: int = Field(0, ge=0, le=100)
    alias: str | None = Field(None, pattern=ALIAS_PATTERN)


class DeployError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class InsufficientMemory(DeployError):
    def __init__(self, fit: FitResult):
        gib = 1024**3
        super().__init__("insufficient_memory",
                         f"needs {fit.required / gib:.1f} GiB but only {fit.free_for_new / gib:.1f} GiB is free", 409)
        self.fit = fit


@dataclass
class Prepared:
    engine: Any
    artifact: ModelArtifact
    cfg: BaseModel
    draft: ModelArtifact | None
    estimate: MemoryEstimate


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


StateHook = Callable[[int, str, str], Awaitable[None]]


class Deployer:
    def __init__(self, *, settings: Settings, db: Database, docker, inventory, engines: dict, http: httpx.AsyncClient,
                 events: EventBroker, budget=None, clock: Callable[[], float] = time.monotonic):
        self.settings, self.db, self.docker, self.inventory = settings, db, docker, inventory
        self.engines, self.http, self.events, self.budget, self.clock = engines, http, events, budget, clock
        self.mem_lock = asyncio.Lock()
        self._locks: dict[int, asyncio.Lock] = {}
        self._monitors: dict[int, asyncio.Task] = {}
        self.state_hooks: list[StateHook] = []
        self.calibration = None  # set in Task 32

    # ---------- helpers ----------
    def _lock(self, dep_id: int) -> asyncio.Lock:
        return self._locks.setdefault(dep_id, asyncio.Lock())

    def _get(self, dep_id: int) -> Deployment | None:
        with self.db.session() as s:
            return s.get(Deployment, dep_id)

    def _require(self, dep_id: int) -> Deployment:
        dep = self._get(dep_id)
        if dep is None:
            raise DeployError("not_found", f"deployment {dep_id} not found", 404)
        return dep

    def base_url(self, name: str, port: int) -> str:
        host = self.settings.engine_base_host or container_name(name)
        return f"http://{host}:{port}"

    def _ref_allowed(self, ref: str) -> bool:
        if re.fullmatch(REPO_RE, ref) and ".." not in ref:
            return True
        try:
            p = Path(ref).resolve(strict=False)
        except (OSError, RuntimeError):
            return False
        roots = [Path(m.local).resolve() for m in self.settings.gguf_roots] + [Path(self.settings.hf_cache.local).resolve()]
        return str(p) == ref and p.suffix.lower() == ".gguf" and any(r in p.parents for r in roots)

    def _transition(self, dep_id: int, to: str, message: str = "", log_tail: str | None = None, **fields) -> dict:
        with self.db.session() as s:
            dep = s.get(Deployment, dep_id)
            frm = dep.state
            assert_transition(frm, to)
            dep.state = to
            for k, v in fields.items():
                setattr(dep, k, v)
            s.add(DeploymentEvent(deployment_id=dep_id, from_state=frm, to_state=to, message=message, log_tail=log_tail))
        view = self.view(dep_id)
        self.events.publish("deployment", view)
        for hook in self.state_hooks:
            asyncio.get_running_loop().create_task(self._run_hook(hook, dep_id, frm, to))
        return view

    @staticmethod
    async def _run_hook(hook: StateHook, dep_id: int, frm: str, to: str) -> None:
        try:
            await hook(dep_id, frm, to)
        except Exception:
            log.exception("state hook failed for deployment %s", dep_id)

    def _allocate_port(self, exclude_dep: int | None = None) -> int:
        with self.db.session() as s:
            used = {d.port for d in s.scalars(select(Deployment)) if d.port and d.state in ACTIVE_STATES | {"planned"}
                    and d.id != exclude_dep}
        try:
            used |= {p for c in self.docker.list_all() for p in c.host_ports if not c.managed}
        except Exception:
            log.warning("could not list containers for port allocation", exc_info=True)
        lo, hi = self.settings.port_range
        for port in range(lo, hi + 1):
            if port not in used:
                return port
        raise DeployError("no_ports", "port pool exhausted", 503)

    # ---------- planning ----------
    def prepare(self, engine_name: str, model_ref: str, raw_cfg: dict) -> Prepared:
        engine = self.engines.get(engine_name)
        if engine is None:
            raise DeployError("unknown_engine", f"engine {engine_name!r} is not available", 422)
        if not self._ref_allowed(model_ref):
            raise DeployError("invalid_model_ref", "model reference is not a repo id or an allowed GGUF path", 422)
        try:
            cfg = engine.validate_config(raw_cfg)
        except (ValidationError, ValueError) as e:
            raise DeployError("invalid_config", str(e), 422)
        artifact = self.inventory.get(model_ref)
        if artifact is None:
            raise DeployError("model_not_found", f"{model_ref} is not downloaded", 404)
        if not artifact.complete:
            raise DeployError("model_incomplete", "; ".join(artifact.problems) or "incomplete download", 409)
        if not engine.supports(artifact):
            raise DeployError("engine_unsupported", f"{engine_name} cannot serve {artifact.format} model {model_ref}", 422)
        draft = None
        if dref := engine.draft_ref(cfg):
            draft = self.inventory.get(dref)
            if draft is None or not draft.complete:
                raise DeployError("draft_not_found", f"draft model {dref} is not downloaded", 404)
        try:
            est = engine.estimate(artifact, cfg, draft, self.settings)
        except ValueError as e:
            raise DeployError("cannot_estimate", str(e), 422)
        if self.calibration is not None:
            est = self.calibration.adjust(engine.name, artifact.ref, config_hash(cfg.model_dump(mode="json")), est)
        return Prepared(engine, artifact, cfg, draft, est)

    def _replacing_credit(self, replacing: int | None) -> int:
        if replacing is None:
            return 0
        dep = self._get(replacing)
        if dep is None or dep.state not in ACTIVE_STATES:
            return 0
        current = self.budget.current()
        actual = next((u.actual for u in current.deployments if u.deployment_id == replacing), 0) if current else 0
        return max(dep.reserved_bytes, actual) if dep.state != "sleeping" else actual

    async def preview(self, engine: str, model_ref: str, raw_cfg: dict, replacing: int | None = None) -> dict:
        prep = self.prepare(engine, model_ref, raw_cfg)
        budget = await self.budget.refresh()
        fit = check_fit(prep.estimate, budget, extra_free=self._replacing_credit(replacing))
        out = {"estimate": prep.estimate.as_dict(), "fit": fit.as_dict(), "budget": budget.as_dict(),
               "served_name": prep.engine.served_name(prep.cfg, model_ref),
               "config": prep.cfg.model_dump(mode="json")}
        if engine == "vllm":
            out["gpu_memory_utilization"] = vllm_gpu_util(prep.estimate.total, budget.total)
        return out

    # ---------- lifecycle ----------
    async def deploy(self, req: DeployRequest, user_id: int | None) -> dict:
        prep = self.prepare(req.engine, req.model_ref, req.config)
        with self.db.session() as s:
            if s.scalar(select(Deployment).where(Deployment.name == req.name)):
                raise DeployError("name_in_use", f"a deployment named {req.name} already exists", 409)
        async with self.mem_lock:
            budget = await self.budget.refresh()
            fit = check_fit(prep.estimate, budget)
            if not fit.fits:
                raise InsufficientMemory(fit)
            port = self._allocate_port()
            with self.db.session() as s:
                dep = Deployment(name=req.name, engine=req.engine, model_ref=req.model_ref,
                                 config=prep.cfg.model_dump(mode="json"), state="planned", port=port,
                                 priority=req.priority, alias=req.alias, reserved_bytes=prep.estimate.total,
                                 estimate=prep.estimate.as_dict(), created_by=user_id)
                s.add(dep)
                s.flush()
                dep_id = dep.id
            await self._launch(dep_id, req.name, port, prep, budget.total)
        self._spawn_monitor(dep_id)
        return self.view(dep_id)

    async def _launch(self, dep_id: int, name: str, port: int, prep: Prepared, mem_total: int) -> None:
        launch = LaunchContext(dep_id, name, port, prep.estimate.total, mem_total)
        spec = prep.engine.container_spec(prep.artifact, prep.cfg, prep.draft, launch, self.settings)
        try:
            await asyncio.to_thread(self.docker.remove_by_name, spec.name)
            cid = await asyncio.to_thread(self.docker.run, spec)
        except Exception as e:
            self._transition(dep_id, "failed", message=f"docker error: {e}", last_error=f"docker error: {e}")
            raise DeployError("docker_error", f"could not start container: {e}", 502)
        self._transition(dep_id, "starting", message=f"container {cid[:12]} created", container_id=cid,
                         last_error=None, exit_code=None)

    def _spawn_monitor(self, dep_id: int) -> None:
        old = self._monitors.get(dep_id)
        if old and not old.done():
            old.cancel()
        self._monitors[dep_id] = asyncio.get_running_loop().create_task(self._monitor(dep_id))

    def resume_monitor(self, dep_id: int) -> None:
        self._spawn_monitor(dep_id)

    async def _monitor(self, dep_id: int) -> None:
        start = self.clock()
        while True:
            await asyncio.sleep(self.settings.monitor_poll_s)
            dep = self._get(dep_id)
            if dep is None or dep.state not in ("starting", "loading"):
                return
            info = await asyncio.to_thread(self.docker.get, dep.container_id)
            if info is None:
                self._transition(dep_id, "failed", message="container disappeared", last_error="container disappeared")
                return
            if info.status in ("exited", "dead"):
                await self._fail_exited(dep, info.exit_code)
                return
            if dep.state == "starting" and info.status == "running":
                self._transition(dep_id, "loading", message="container running; loading model")
            mem = await asyncio.to_thread(self.budget.probe.meminfo)
            if mem.available < self.settings.margin_bytes:
                await self._abort(dep, "aborted: available memory fell below the safety margin during startup")
                return
            if self.clock() - start > self.settings.startup_timeout_s:
                await self._abort(dep, f"startup timed out after {self.settings.startup_timeout_s}s")
                return
            if info.status == "running" and await self._healthy(dep):
                self._transition(dep_id, "ready", message="health check and test completion passed")
                return

    async def _healthy(self, dep: Deployment) -> bool:
        engine = self.engines[dep.engine]
        cfg = engine.validate_config(dep.config)
        try:
            return await engine.health(self.http, self.base_url(dep.name, dep.port), engine.served_name(cfg, dep.model_ref))
        except (httpx.HTTPError, OSError):
            return False

    async def _fail_exited(self, dep: Deployment, exit_code: int | None) -> None:
        logs = await asyncio.to_thread(self.docker.logs, dep.container_id, 200)
        hint = " (likely out of memory)" if exit_code == 137 else ""
        msg = f"container exited with code {exit_code}{hint}"
        await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
        self._transition(dep.id, "failed", message=msg, log_tail=logs, last_error=msg, exit_code=exit_code,
                         container_id=None)
        if exit_code == 137 and self.calibration is not None:
            self.calibration.record_oom(dep.engine, dep.model_ref, config_hash(dep.config), dep.reserved_bytes)

    async def _abort(self, dep: Deployment, reason: str) -> None:
        logs = await asyncio.to_thread(self.docker.logs, dep.container_id, 200)
        await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
        self._transition(dep.id, "failed", message=reason, log_tail=logs, last_error=reason, container_id=None)

    async def wait_ready(self, dep_id: int, timeout: float) -> str:
        deadline = self.clock() + timeout
        while True:
            dep = self._require(dep_id)
            if dep.state in ("ready", "failed", "stopped", "sleeping"):
                return dep.state
            if self.clock() > deadline:
                return dep.state
            await asyncio.sleep(0.02)

    async def stop(self, dep_id: int, user_id: int | None = None, reason: str = "user") -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state == "stopped":
                return self.view(dep_id)
            mon = self._monitors.pop(dep_id, None)
            if mon and not mon.done():
                mon.cancel()
            if dep.state in ("planned", "failed"):
                await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
                return self._transition(dep_id, "stopped", message=f"cleaned up ({reason})", container_id=None)
            self._transition(dep_id, "stopping", message=f"stop requested ({reason})")
            try:
                await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
            except Exception as e:
                self._transition(dep_id, "failed", message=f"stop failed: {e}", last_error=f"stop failed: {e}")
                raise DeployError("docker_error", f"could not stop container: {e}", 502)
            return self._transition(dep_id, "stopped", message="container removed", container_id=None)

    async def restart(self, dep_id: int, user_id: int | None = None, config_override: dict | None = None,
                      model_ref: str | None = None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", f"deployment is {dep.state}; stop it first", 409)
            ref = model_ref or dep.model_ref
            prep = self.prepare(dep.engine, ref, config_override if config_override is not None else dep.config)
            async with self.mem_lock:
                budget = await self.budget.refresh()
                fit = check_fit(prep.estimate, budget)
                if not fit.fits:
                    raise InsufficientMemory(fit)
                port = dep.port if dep.port else self._allocate_port(exclude_dep=dep_id)
                if port in {p for c in self.docker.list_all() for p in c.host_ports if not c.managed}:
                    port = self._allocate_port(exclude_dep=dep_id)
                with self.db.session() as s:
                    row = s.get(Deployment, dep_id)
                    row.model_ref, row.config, row.port = ref, prep.cfg.model_dump(mode="json"), port
                    row.reserved_bytes, row.estimate = prep.estimate.total, prep.estimate.as_dict()
                await self._launch(dep_id, dep.name, port, prep, budget.total)
        self._spawn_monitor(dep_id)
        return self.view(dep_id)

    async def remove(self, dep_id: int) -> None:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state not in ("stopped", "failed"):
                raise DeployError("not_stopped", "stop the deployment before removing it", 409)
            await asyncio.to_thread(self.docker.stop_and_remove, dep.container_id)
            with self.db.session() as s:
                s.delete(s.get(Deployment, dep_id))
        self.events.publish("deployment_removed", {"id": dep_id})

    async def aclose(self) -> None:
        for t in self._monitors.values():
            t.cancel()
        self._monitors.clear()

    # ---------- views ----------
    def usage_rows(self) -> list[tuple[int, str, str, int, str | None]]:
        with self.db.session() as s:
            return [(d.id, d.name, d.state, d.reserved_bytes, d.container_id)
                    for d in s.scalars(select(Deployment)) if d.state in ACTIVE_STATES]

    def _actual(self, dep_id: int) -> int | None:
        b = self.budget.current() if self.budget else None
        if not b:
            return None
        return next((u.actual for u in b.deployments if u.deployment_id == dep_id), None)

    def _view(self, d: Deployment) -> dict:
        engine = self.engines.get(d.engine)
        served = None
        if engine is not None:
            try:
                served = engine.served_name(engine.validate_config(d.config), d.model_ref)
            except (ValidationError, ValueError):
                served = None
        return {
            "id": d.id, "name": d.name, "engine": d.engine, "model_ref": d.model_ref, "state": d.state,
            "port": d.port, "priority": d.priority, "alias": d.alias, "reserved_bytes": d.reserved_bytes,
            "actual_bytes": self._actual(d.id), "estimate": d.estimate, "config": d.config,
            "litellm_status": d.litellm_status, "last_error": d.last_error, "exit_code": d.exit_code,
            "container_id": d.container_id, "served_name": served, "adopted_from": d.adopted_from,
            "endpoint": f"http://{container_name(d.name)}:{d.port}/v1" if d.port else None,
            "host_endpoint": f"http://{self.settings.publish_bind}:{d.port}/v1" if d.port and self.settings.publish_bind else None,
            "sleep_capable": bool(engine and engine.supports_sleep and d.config.get("enable_sleep_mode")),
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        }

    def view(self, dep_id: int, with_events: bool = False) -> dict:
        with self.db.session() as s:
            d = s.get(Deployment, dep_id)
            if d is None:
                raise DeployError("not_found", f"deployment {dep_id} not found", 404)
            out = self._view(d)
            if with_events:
                rows = s.scalars(select(DeploymentEvent).where(DeploymentEvent.deployment_id == dep_id)
                                 .order_by(DeploymentEvent.id.desc()).limit(50))
                out["events"] = [{"ts": e.ts.isoformat(), "from": e.from_state, "to": e.to_state,
                                  "message": e.message, "log_tail": e.log_tail} for e in rows]
        return out

    def list_views(self) -> list[dict]:
        with self.db.session() as s:
            return [self._view(d) for d in s.scalars(select(Deployment).order_by(Deployment.name))]
```

Notes for the implementer:
- `_fail_exited` removes the exited container so its name and port are free for a retry. The logs are captured first.
- `restart` re-checks that the old port is not taken by an unmanaged container before reusing it.

- [ ] **Step 6: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_deployer.py`. Expected: all pass. If `test_concurrent_deploys_cannot_both_claim_memory` is flaky, the global `mem_lock` is not held across `refresh → check_fit → _launch`; fix the code, not the test.

---

### Task 12: Startup reconciliation

**Files:**
- Create: `dgxmm/deployer/reconcile.py`
- Test: `tests/test_reconcile.py`

**Interfaces:**
- Consumes: `Deployer` (`_transition`, `resume_monitor`, `stop`, `docker`, `db`), `ContainerInfo`.
- Produces: `async reconcile(deployer) -> list[str]` (human-readable notes, also logged).

- [ ] **Step 1: Write the failing tests** — `tests/test_reconcile.py`

```python
from sqlalchemy import select

from dgxmm.db import Deployment
from dgxmm.deployer.reconcile import reconcile
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import managed_labels

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def deploy(stack, name):
    return await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL), 1)


async def test_loading_deployment_is_resumed_not_failed(stack):
    dep = await deploy(stack, "a")
    await stack.deployer.aclose()  # simulate dgxmm restart: monitors gone, container still running
    notes = await reconcile(stack.deployer)
    assert any("resumed" in n for n in notes)
    stack.http.healthy.add(dep["port"])
    assert await stack.deployer.wait_ready(dep["id"], 5) == "ready"


async def test_missing_and_exited_containers_fail(stack):
    a, b = await deploy(stack, "a"), await deploy(stack, "b")
    await stack.deployer.aclose()
    stack.sdk._containers.pop(a["container_id"])
    cb = stack.sdk.container(b["container_id"])
    cb.status, cb.exit_code = "exited", 137
    await reconcile(stack.deployer)
    va, vb = stack.deployer.view(a["id"]), stack.deployer.view(b["id"])
    assert va["state"] == "failed" and "missing" in va["last_error"]
    assert vb["state"] == "failed" and vb["exit_code"] == 137


async def test_orphan_managed_container_becomes_unknown_row(stack):
    stack.sdk.add_unmanaged("dgxmm-ghost", labels=managed_labels(999, "vllm"),
                            command=["--model", "Qwen/Qwen3-Tiny", "--port", "8150"], ports=[(8150, "127.0.0.1")])
    stack.sdk.add_unmanaged("tts-service")
    await reconcile(stack.deployer)
    with stack.db.session() as s:
        rows = s.scalars(select(Deployment)).all()
    assert [(r.name, r.state, r.port, r.model_ref) for r in rows] == [("ghost", "unknown", 8150, "Qwen/Qwen3-Tiny")]
```

- [ ] **Step 2: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_reconcile.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/deployer/reconcile.py`**

```python
"""Bring the DB in line with Docker after a dgxmm restart (spec §11)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.db import Deployment, DeploymentEvent

log = logging.getLogger(__name__)


def _flag(cmd: list[str], flag: str) -> str | None:
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


async def reconcile(deployer) -> list[str]:
    notes: list[str] = []
    containers = {c.id: c for c in await asyncio.to_thread(deployer.docker.list_managed)
                  if not c.labels.get("dgxmm.oneshot")}
    with deployer.db.session() as s:
        rows = list(s.scalars(select(Deployment)))
    known = {r.container_id for r in rows if r.container_id}
    for d in rows:
        if d.state in ("stopped", "failed", "planned"):
            continue
        c = containers.get(d.container_id) if d.container_id else None
        if c is None:
            deployer._transition(d.id, "failed", message="container missing after dgxmm restart",
                                 last_error="container missing after dgxmm restart", container_id=None)
            notes.append(f"{d.name}: container missing → failed")
        elif c.status in ("exited", "dead"):
            await deployer._fail_exited(d, c.exit_code)
            notes.append(f"{d.name}: container exited ({c.exit_code}) → failed")
        elif d.state in ("starting", "loading"):
            deployer.resume_monitor(d.id)
            notes.append(f"{d.name}: still {d.state} → monitoring resumed")
        elif d.state == "stopping":
            await deployer.stop(d.id, reason="reconcile")
            notes.append(f"{d.name}: finishing interrupted stop")
    taken = {r.name for r in rows}
    for cid, c in containers.items():
        if cid in known:
            continue
        name = c.name.removeprefix("dgxmm-") or "orphan"
        base, n = name, 2
        while name in taken:
            name, n = f"{base}-{n}", n + 1
        taken.add(name)
        port = _flag(c.command, "--port")
        port_i = int(port) if port and port.isdigit() else (min(c.host_ports) if c.host_ports else None)
        with deployer.db.session() as s:
            row = Deployment(name=name, engine=c.labels.get("dgxmm.engine", "vllm"),
                             model_ref=_flag(c.command, "--model") or _flag(c.command, "-m") or "unknown",
                             config={}, state="unknown", container_id=cid, port=port_i, reserved_bytes=0,
                             estimate={}, last_error="found running without a database record")
            s.add(row)
            s.flush()
            s.add(DeploymentEvent(deployment_id=row.id, from_state="unknown", to_state="unknown",
                                  message=f"adopted orphan container {c.name}"))
        notes.append(f"{c.name}: orphan managed container → recorded as unknown")
    for n in notes:
        log.info("reconcile: %s", n)
    return notes
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_reconcile.py`. Expected: `3 passed`.

---

### Task 13: HTTP API (host, inventory, engines, deployments, SSE) and app wiring

**Files:**
- Create: `dgxmm/api/host.py`, `dgxmm/api/inventory.py`, `dgxmm/api/engines.py`, `dgxmm/api/deployments.py`, `dgxmm/api/stream.py`, `dgxmm/__main__.py`
- Modify: `dgxmm/context.py` (new fields + `build_context`), `dgxmm/app.py` (routers + startup hooks)
- Modify: `tests/conftest.py` (add `api_stack`)
- Test: `tests/test_api_deployments.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - Routes (all under `/dgxmm`):
    - `GET /api/host`
    - `GET /api/inventory`, `POST /api/inventory/refresh`
    - `GET /api/engines`
    - `POST /api/deployments/preview`, `POST /api/deployments`, `GET /api/deployments`, `GET /api/deployments/{id}`
    - `POST /api/deployments/{id}/stop`, `POST /api/deployments/{id}/start`, `DELETE /api/deployments/{id}`
    - `GET /api/deployments/{id}/logs`
    - `GET /api/stream` (SSE)
  - `host_payload(ctx, budget) -> dict`, `deploy_error(e) -> HTTPException`.
  - `AppContext` gains fields `probe`, `http`, `startup: list`, `budget_listeners: list`, `watchdog`, `litellm`, `catalog`, `downloads`, `metrics`, `advisor`, `benchmarks`.
  - `build_context(settings) -> AppContext`.

- [ ] **Step 1: Update `dgxmm/context.py`** (replace the dataclass body and add `build_context`)

```python
"""Shared service container stored on app.state.ctx, plus production wiring."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.config import Settings
from dgxmm.db import Database
from dgxmm.events import EventBroker

log = logging.getLogger("dgxmm")


@dataclass
class AppContext:
    settings: Settings
    db: Database
    events: EventBroker
    login_limiter: SlidingWindowLimiter = field(default_factory=lambda: SlidingWindowLimiter(5, 60))
    inventory: Any = None
    budget: Any = None
    deployer: Any = None
    docker: Any = None
    engines: dict = field(default_factory=dict)
    probe: Any = None
    http: Any = None
    watchdog: Any = None
    litellm: Any = None
    catalog: Any = None
    downloads: Any = None
    metrics: Any = None
    advisor: Any = None
    benchmarks: Any = None
    startup: list = field(default_factory=list)          # awaited once, in order, before serving
    background: list = field(default_factory=list)       # long-running coroutines
    budget_listeners: list = field(default_factory=list)  # async fn(budget) after each refresh


async def budget_loop(ctx: AppContext) -> None:
    from dgxmm.api.host import host_payload
    while True:
        try:
            b = await ctx.budget.refresh()
            ctx.events.publish("budget", host_payload(ctx, b))
            for listener in ctx.budget_listeners:
                await listener(b)
        except Exception:
            log.exception("budget refresh failed")
        await asyncio.sleep(ctx.settings.budget_interval_s)


def build_context(settings: Settings) -> AppContext:
    import httpx

    from dgxmm.auth.bootstrap import bootstrap_admin
    from dgxmm.deployer.reconcile import reconcile
    from dgxmm.deployer.service import Deployer
    from dgxmm.docker_ctl import DockerController
    from dgxmm.engines.registry import build_engines
    from dgxmm.inventory.service import InventoryService
    from dgxmm.memory.budget import BudgetService
    from dgxmm.memory.hostmem import HostProbe, NvmlGpu

    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    db = Database(f"sqlite:///{settings.db_path}")
    events = EventBroker()
    http = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5))
    inventory = InventoryService(settings)
    docker = DockerController.from_env()
    engines = build_engines(settings)
    probe = HostProbe(settings, NvmlGpu())
    deployer = Deployer(settings=settings, db=db, docker=docker, inventory=inventory, engines=engines, http=http,
                        events=events)
    budget = BudgetService(settings, probe, deployer.usage_rows)
    deployer.budget = budget
    ctx = AppContext(settings=settings, db=db, events=events, inventory=inventory, budget=budget, deployer=deployer,
                     docker=docker, engines=engines, probe=probe, http=http)

    async def _bootstrap():
        bootstrap_admin(db)

    async def _inventory():
        await asyncio.to_thread(inventory.refresh)

    async def _reconcile():
        await budget.refresh()
        await reconcile(deployer)

    async def _budget_loop():
        await budget_loop(ctx)

    async def _inventory_loop():
        await asyncio.sleep(settings.inventory_interval_s)
        await inventory.run()

    ctx.startup += [_bootstrap, _inventory, _reconcile]
    ctx.background += [_budget_loop, _inventory_loop]
    return ctx
```

Later tasks append to `build_context` right before `return ctx`. Each says exactly what to add.

- [ ] **Step 2: Update `dgxmm/app.py`**

Replace the `lifespan` function and the router tuple:
```python
from dgxmm.api import auth as auth_api, users as users_api, host as host_api, inventory as inventory_api
from dgxmm.api import engines as engines_api, deployments as deployments_api, stream as stream_api

ROUTERS = [auth_api, users_api, host_api, inventory_api, engines_api, deployments_api, stream_api]


def create_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for fn in ctx.startup:
            await fn()
        tasks = [asyncio.create_task(fn(), name=getattr(fn, "__name__", "bg")) for fn in ctx.background]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            for t in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if ctx.deployer is not None:
                await ctx.deployer.aclose()

    app = FastAPI(title="DGX Model Manager", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.ctx = ctx
    bp = ctx.settings.base_path
    for module in ROUTERS:
        app.include_router(module.router, prefix=bp)
    _mount_static(app, Path(ctx.settings.static_dir), bp)
    return app
```
(Later tasks append modules to `ROUTERS`.)

- [ ] **Step 3: Implement the routers**

`dgxmm/api/host.py`:
```python
import shutil

from fastapi import APIRouter, Depends

from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/host", tags=["host"])


def host_payload(ctx: AppContext, budget) -> dict:
    sample = ctx.budget.last_sample
    out = {"budget": budget.as_dict(),
           "external_top": [p.__dict__ for p in sample.external_top] if sample else [],
           "gpu": sample.gpu if sample else {},
           "alert": ctx.watchdog.state() if ctx.watchdog else {"level": "ok"}}
    try:
        du = shutil.disk_usage(ctx.settings.hf_cache.local)
        out["disk"] = {"total": du.total, "free": du.free}
    except OSError:
        out["disk"] = None
    return out


@router.get("")
async def host(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    b = ctx.budget.current() or await ctx.budget.refresh()
    return host_payload(ctx, b)
```

`dgxmm/api/inventory.py`:
```python
import asyncio

from fastapi import APIRouter, Depends

from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/inventory", tags=["inventory"])


def _with_support(ctx: AppContext, a) -> dict:
    d = a.as_dict()
    d["engines"] = [name for name, e in ctx.engines.items() if e.supports(a)]
    return d


@router.get("")
def list_inventory(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return [_with_support(ctx, a) for a in ctx.inventory.list()]


@router.post("/refresh")
async def refresh(p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    items = await asyncio.to_thread(ctx.inventory.refresh)
    audit(ctx.db, p.user.id, "inventory_refreshed", "", {"count": len(items)})
    return [_with_support(ctx, a) for a in ctx.inventory.list()]
```

`dgxmm/api/engines.py`:
```python
from fastapi import APIRouter, Depends

from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api/engines", tags=["engines"])


@router.get("")
def engines(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return [{"name": name, "supports_sleep": e.supports_sleep, "defaults": e.config_model().model_dump(mode="json"),
             "schema": e.config_model.model_json_schema(), "image": getattr(ctx.settings.images, name, None)}
            for name, e in ctx.engines.items()]
```

`dgxmm/api/deployments.py`:
```python
import asyncio

from fastapi import APIRouter, Depends, Query, Response
from pydantic import BaseModel

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


class PreviewIn(BaseModel):
    engine: str
    model_ref: str
    config: dict = {}
    replacing: int | None = None


def deploy_error(e: DeployError):
    extra = {"fit": e.fit.as_dict()} if isinstance(e, InsufficientMemory) else {}
    return api_error(e.status, e.code, e.message, **extra)


@router.post("/preview")
async def preview(body: PreviewIn, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        return await ctx.deployer.preview(body.engine, body.model_ref, body.config, replacing=body.replacing)
    except DeployError as e:
        raise deploy_error(e)


@router.post("", status_code=201)
async def create(body: DeployRequest, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await ctx.deployer.deploy(body, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "deploy", body.name, {"model": body.model_ref, "engine": body.engine})
    return view


@router.get("")
def list_deployments(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return ctx.deployer.list_views()


@router.get("/{dep_id}")
def get_deployment(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        return ctx.deployer.view(dep_id, with_events=True)
    except DeployError as e:
        raise deploy_error(e)


@router.post("/{dep_id}/stop")
async def stop(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await ctx.deployer.stop(dep_id, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "stop", view["name"])
    return view


@router.post("/{dep_id}/start")
async def start(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await ctx.deployer.restart(dep_id, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "start", view["name"])
    return view


@router.delete("/{dep_id}", status_code=204)
async def remove(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        name = ctx.deployer.view(dep_id)["name"]
        await ctx.deployer.remove(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "remove", name)
    return Response(status_code=204)


@router.get("/{dep_id}/logs")
async def logs(dep_id: int, tail: int = Query(200, ge=1, le=5000), p: Principal = Depends(require_admin),
               ctx: AppContext = Depends(get_ctx)):
    try:
        view = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    return {"logs": await asyncio.to_thread(ctx.docker.logs, view["container_id"], tail)}
```

`dgxmm/api/stream.py`:
```python
import asyncio
import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext

router = APIRouter(prefix="/api", tags=["stream"])


@router.get("/stream")
async def stream(request: Request, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    q = ctx.events.subscribe()

    async def gen():
        try:
            yield "event: hello\ndata: {}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    topic, data = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"event: {topic}\ndata: {json.dumps(data, default=str)}\n\n"
        finally:
            ctx.events.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```

`dgxmm/__main__.py`:
```python
import logging

import uvicorn

from dgxmm.app import create_app
from dgxmm.config import load_settings
from dgxmm.context import build_context


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    uvicorn.run(create_app(build_context(settings)), host="0.0.0.0", port=8090,
                proxy_headers=True, forwarded_allow_ips="*", log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the `api_stack` fixture to `tests/conftest.py`** (append)

```python
@pytest.fixture
def api_stack(stack, settings, db):
    """TestClient wired to the fake stack (no background loops)."""
    from dgxmm.app import create_app
    from dgxmm.engines.registry import build_engines
    c = AppContext(settings=settings, db=db, events=stack.events, login_limiter=SlidingWindowLimiter(100, 60),
                   inventory=stack.inv, budget=stack.budget, deployer=stack.deployer, docker=stack.deployer.docker,
                   engines=stack.deployer.engines, probe=stack.probe)
    with TestClient(create_app(c), base_url="http://testserver") as tc:
        tc.stack = stack
        tc.ctx = c
        yield tc
```

- [ ] **Step 5: Write the failing tests** — `tests/test_api_deployments.py`

```python
from dgxmm.config import GIB

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def admin(c, make_user=None):
    from dgxmm.auth.passwords import hash_password
    from dgxmm.db import User
    with c.ctx.db.session() as s:
        s.add(User(username="root", pw_hash=hash_password("root-password-123"), role="admin"))
        s.add(User(username="vic", pw_hash=hash_password("viewer-password-1"), role="viewer"))
    r = c.post("/dgxmm/api/auth/login", json={"username": "root", "password": "root-password-123"})
    return {"X-CSRF-Token": r.json()["csrf_token"]}


def test_inventory_host_engines(api_stack):
    h = admin(api_stack)
    inv = api_stack.get("/dgxmm/api/inventory").json()
    tiny = next(a for a in inv if a["ref"] == "Qwen/Qwen3-Tiny")
    assert tiny["engines"] == ["vllm"] and tiny["complete"]
    host = api_stack.get("/dgxmm/api/host").json()
    assert host["budget"]["total"] == 121 * GIB and host["alert"] == {"level": "ok"}
    eng = api_stack.get("/dgxmm/api/engines").json()
    assert eng[0]["name"] == "vllm" and eng[0]["defaults"]["max_model_len"] == 32768


def test_preview_deploy_stop_remove(api_stack):
    h = admin(api_stack)
    p = api_stack.post("/dgxmm/api/deployments/preview", headers=h,
                       json={"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).json()
    assert p["fit"]["fits"]
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 201, r.text
    dep = r.json()
    assert api_stack.get("/dgxmm/api/deployments").json()[0]["name"] == "tiny"
    assert api_stack.get(f"/dgxmm/api/deployments/{dep['id']}").json()["events"]
    assert api_stack.post(f"/dgxmm/api/deployments/{dep['id']}/stop", headers=h).json()["state"] == "stopped"
    assert api_stack.delete(f"/dgxmm/api/deployments/{dep['id']}", headers=h).status_code == 204


def test_insufficient_memory_is_409_with_fit(api_stack):
    h = admin(api_stack)
    api_stack.stack.probe.available = 10 * GIB
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "big", "engine": "vllm", "model_ref": "org/big-model", "config": SMALL})
    assert r.status_code == 409
    d = r.json()["detail"]
    assert d["code"] == "insufficient_memory" and d["fit"]["required"] > d["fit"]["free_for_new"]


def test_hostile_inputs_rejected_before_docker(api_stack):
    h = admin(api_stack)
    cases = [
        {"name": "x;rm", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL},
        {"name": "x", "engine": "vllm", "model_ref": "../../etc/passwd", "config": SMALL},
        {"name": "x", "engine": "vllm", "model_ref": "/etc/passwd.gguf", "config": SMALL},
        {"name": "x", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {**SMALL, "extra_args": ["--model", "evil"]}},
        {"name": "x", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {**SMALL, "gpu_memory_utilization": 0.99}},
    ]
    for body in cases:
        r = api_stack.post("/dgxmm/api/deployments", headers=h, json=body)
        assert r.status_code == 422, (body, r.text)
    assert api_stack.stack.sdk.run_calls == []


def test_viewer_cannot_deploy(api_stack):
    admin(api_stack)
    r = api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    h = {"X-CSRF-Token": r.json()["csrf_token"]}
    r = api_stack.post("/dgxmm/api/deployments", headers=h,
                       json={"name": "tiny", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL})
    assert r.status_code == 403
    assert api_stack.post("/dgxmm/api/deployments/preview", headers=h,
                          json={"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL}).status_code == 200


def test_unknown_api_route_is_json_404(api_stack):
    admin(api_stack)
    r = api_stack.get("/dgxmm/api/nope")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "not_found"
```

- [ ] **Step 6: Run the full test suite**

`DGX$ .venv/bin/pytest -q`. Expected: all tests pass, including the earlier tasks.

---

### Task 14: Frontend foundation and the M1 pages

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/index.html`
- Create: `frontend/src/main.tsx`, `src/index.css`, `src/api.ts`, `src/types.ts`, `src/format.ts`, `src/format.test.ts`, `src/auth.tsx`, `src/stream.ts`, `src/engineFields.ts`
- Create: `src/components/ui.tsx`, `src/components/Layout.tsx`, `src/components/MemoryBar.tsx`, `src/components/EstimatePanel.tsx`, `src/components/ConfigForm.tsx`
- Create: `src/pages/Login.tsx`, `src/pages/ChangePassword.tsx`, `src/pages/Dashboard.tsx`, `src/pages/Models.tsx`, `src/pages/DeployWizard.tsx`, `src/pages/DeploymentDetail.tsx`, `src/pages/Users.tsx`

**Interfaces:**
- Consumes: the REST routes from Tasks 3 and 13, and SSE events `budget`, `deployment`, `deployment_removed`.
- Produces:
  - `api<T>(path, {method, body})`, `setCsrf`, `ApiError(status, detail)`.
  - `fmtBytes`, `fmtPct`, `fmtNum`, `fmtMs`, `stateTone`.
  - `useAuth()`, `useLiveUpdates()`.
  - UI primitives `Card`, `Button`, `Badge`, `Stat`, `Field`, `Input`, `Select`, `Modal`, `ErrorBox`, `Table`.
  - `ENGINE_FIELDS`, `<ConfigForm>`, `<EstimatePanel>`, `<MemoryBar>`.

- [ ] **Step 1: Build tooling files**

`frontend/package.json`:
```json
{
  "name": "dgxmm-ui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run"
  },
  "dependencies": {
    "@tanstack/react-query": "^5.56.2",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.26.2",
    "recharts": "^2.12.7"
  },
  "devDependencies": {
    "@types/react": "^18.3.5",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.1",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.11",
    "typescript": "^5.5.4",
    "vite": "^5.4.6",
    "vitest": "^2.1.1"
  }
}
```

`frontend/vite.config.ts`:
```ts
/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/dgxmm/",
  plugins: [react()],
  server: { proxy: { "/dgxmm/api": { target: "http://127.0.0.1:8090" } } },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
```

`frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "skipLibCheck": true,
    "types": ["vite/client"]
  },
  "include": ["src", "vite.config.ts"]
}
```

`frontend/tailwind.config.js`:
```js
export default { content: ["./index.html", "./src/**/*.{ts,tsx}"], theme: { extend: {} }, plugins: [] };
```

`frontend/postcss.config.js`:
```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`frontend/index.html`:
```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>DGX Model Manager</title>
  </head>
  <body class="bg-slate-50 text-slate-900">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`frontend/src/index.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

.th { @apply px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500; }
.td { @apply px-3 py-2 text-sm align-top; }
```

- [ ] **Step 2: Write the failing unit test** — `frontend/src/format.test.ts`

```ts
import { describe, expect, it } from "vitest";
import { GIB, fmtBytes, fmtPct, stateTone } from "./format";

describe("format", () => {
  it("formats bytes in binary units", () => {
    expect(fmtBytes(0)).toBe("0 B");
    expect(fmtBytes(1536)).toBe("1.5 KiB");
    expect(fmtBytes(8 * GIB)).toBe("8.0 GiB");
    expect(fmtBytes(-2 * GIB)).toBe("-2.0 GiB");
    expect(fmtBytes(null)).toBe("—");
  });
  it("formats ratios and states", () => {
    expect(fmtPct(0.256)).toBe("26%");
    expect(fmtPct(undefined)).toBe("—");
    expect(stateTone("ready")).toBe("green");
    expect(stateTone("failed")).toBe("red");
    expect(stateTone("loading")).toBe("amber");
  });
});
```

Run (after rsync): `NPM$ sh -c "npm install --no-audit --no-fund && npx vitest run"`. Expected: FAIL (cannot resolve `./format`).

- [ ] **Step 3: Core modules**

`frontend/src/format.ts`:
```ts
export const GIB = 1024 ** 3;
export type Tone = "green" | "amber" | "red" | "slate" | "blue" | "violet";

export function fmtBytes(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let v = Math.abs(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${n < 0 ? "-" : ""}${v.toFixed(i === 0 ? 0 : digits)} ${units[i]}`;
}
export const fmtPct = (x?: number | null) => (x == null ? "—" : `${Math.round(x * 100)}%`);
export const fmtNum = (x?: number | null, d = 1) => (x == null ? "—" : x.toFixed(d));
export const fmtMs = (s?: number | null) => (s == null ? "—" : `${Math.round(s * 1000)} ms`);

export function stateTone(state: string): Tone {
  if (state === "ready") return "green";
  if (state === "failed") return "red";
  if (["starting", "loading", "stopping", "planned"].includes(state)) return "amber";
  if (state === "sleeping") return "violet";
  if (state === "unknown") return "blue";
  return "slate";
}
```

`frontend/src/api.ts`:
```ts
const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");
let csrf = "";
export const setCsrf = (t: string) => {
  csrf = t;
};

export class ApiError extends Error {
  constructor(public status: number, public detail: any) {
    super(typeof detail === "string" ? detail : detail?.message ?? `HTTP ${status}`);
  }
  get code(): string | undefined {
    return typeof this.detail === "object" ? this.detail?.code : undefined;
  }
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const method = opts.method ?? "GET";
  const headers: Record<string, string> = {};
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET") headers["X-CSRF-Token"] = csrf;
  const res = await fetch(`${BASE}/api${path}`, {
    method,
    headers,
    credentials: "same-origin",
    body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  if (res.status === 204) return undefined as T;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = Array.isArray(data?.detail)
      ? { code: "validation", message: data.detail.map((d: any) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ") }
      : data?.detail ?? data;
    throw new ApiError(res.status, detail);
  }
  return data as T;
}

export const streamUrl = `${BASE}/api/stream`;
```

`frontend/src/types.ts`:
```ts
export interface User { id: number; username: string; role: "admin" | "viewer"; must_change_pw: boolean }
export interface DeploymentUsage { deployment_id: number; name: string; state: string; reserved: number; actual: number }
export interface Budget {
  total: number; available: number; margin: number; os_reserve: number; usable: number; committed: number;
  managed_actual: number; external: number; free_for_new: number; deployments: DeploymentUsage[];
}
export interface ProcMem { pid: number; name: string; rss: number; gpu: number; cmdline: string }
export interface Alert { level: "ok" | "warning" | "critical"; message?: string; since?: string; action?: string }
export interface HostInfo { budget: Budget; external_top: ProcMem[]; gpu: Record<string, number | null>; alert: Alert; disk: { total: number; free: number } | null }
export interface Artifact {
  ref: string; source: "hf" | "gguf"; revision: string | null; format: string; weights_bytes: number; size_bytes: number;
  complete: boolean; kind: "llm" | "draft" | "other"; architecture: string | null; quant: string | null; problems: string[];
  max_context: number | null; is_moe: boolean | null; num_layers: number | null; confident: boolean | null;
  has_mtp: boolean; engines: string[];
}
export interface Estimate {
  weights: number; draft_weights: number; kv_cache: number; state: number; activations: number; runtime: number;
  margin: number; confidence: string; notes: string[]; total: number;
}
export interface Fit { fits: boolean; required: number; free_for_new: number; headroom: number; estimate: Estimate }
export interface PlanAction { type: "sleep" | "stop" | "patch"; deployment_id?: number | null; patch?: Record<string, unknown> | null }
export interface PlanOption { id: string; title: string; detail: string; new_required: number; frees: number; fits: boolean; actions: PlanAction[]; patch: Record<string, unknown> | null }
export interface Preview {
  estimate: Estimate; fit: Fit; budget: Budget; served_name: string; config: Record<string, unknown>;
  gpu_memory_utilization?: number; plans?: PlanOption[]; llmfit?: Record<string, any> | null;
}
export interface DeploymentEvent { ts: string; from: string; to: string; message: string; log_tail: string | null }
export interface Deployment {
  id: number; name: string; engine: string; model_ref: string; state: string; port: number | null; priority: number;
  alias: string | null; reserved_bytes: number; actual_bytes: number | null; estimate: Estimate | Record<string, never>;
  config: Record<string, any>; litellm_status: string; last_error: string | null; exit_code: number | null;
  container_id: string | null; served_name: string | null; endpoint: string | null; host_endpoint: string | null;
  sleep_capable: boolean; adopted_from: string | null; created_at: string; updated_at: string; events?: DeploymentEvent[];
}
export interface EngineInfo { name: string; supports_sleep: boolean; defaults: Record<string, any>; image: string | null }
```

`frontend/src/auth.tsx`:
```tsx
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { api, setCsrf } from "./api";
import type { User } from "./types";

interface AuthState {
  user: User | null;
  loading: boolean;
  login: (u: string, p: string) => Promise<void>;
  logout: () => Promise<void>;
  reload: () => Promise<void>;
}
const Ctx = createContext<AuthState>(null as unknown as AuthState);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const reload = useCallback(async () => {
    try {
      const me = await api<{ user: User; csrf_token: string }>("/auth/me");
      setCsrf(me.csrf_token);
      setUser(me.user);
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    reload();
  }, [reload]);
  const login = async (username: string, password: string) => {
    const r = await api<{ user: User; csrf_token: string }>("/auth/login", { method: "POST", body: { username, password } });
    setCsrf(r.csrf_token);
    setUser(r.user);
  };
  const logout = async () => {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    setUser(null);
  };
  return <Ctx.Provider value={{ user, loading, login, logout, reload }}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
export const useIsAdmin = () => useAuth().user?.role === "admin";

export function RequireAuth({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  const loc = useLocation();
  if (loading) return <div className="p-8 text-slate-500">Loading…</div>;
  if (!user) return <Navigate to="/login" replace state={{ from: loc.pathname }} />;
  if (user.must_change_pw && loc.pathname !== "/change-password") return <Navigate to="/change-password" replace />;
  return <>{children}</>;
}
```

`frontend/src/stream.ts`:
```ts
import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { streamUrl } from "./api";

const LIVE_POINTS = 120;

export function useLiveUpdates(enabled: boolean) {
  const qc = useQueryClient();
  useEffect(() => {
    if (!enabled) return;
    const es = new EventSource(streamUrl, { withCredentials: true });
    const on = (name: string, fn: (d: any) => void) => es.addEventListener(name, (e) => fn(JSON.parse((e as MessageEvent).data)));
    on("budget", (d) => qc.setQueryData(["host"], d));
    on("deployment", (d) => {
      qc.invalidateQueries({ queryKey: ["deployments"] });
      qc.invalidateQueries({ queryKey: ["deployment", d.id] });
    });
    on("deployment_removed", () => qc.invalidateQueries({ queryKey: ["deployments"] }));
    on("download", () => qc.invalidateQueries({ queryKey: ["downloads"] }));
    on("inventory", () => qc.invalidateQueries({ queryKey: ["inventory"] }));
    on("metrics", (d: Record<string, any>) => {
      for (const [id, sample] of Object.entries(d)) {
        qc.setQueryData<any[]>(["live", Number(id)], (old = []) => [...old, sample].slice(-LIVE_POINTS));
      }
    });
    on("alert", (d) => qc.setQueryData(["alert"], d));
    return () => es.close();
  }, [enabled, qc]);
}
```

`frontend/src/engineFields.ts`:
```ts
export type FieldType = "int" | "bool" | "text" | "select";
export interface FieldDef { key: string; label: string; type: FieldType; options?: string[]; help?: string; advanced?: boolean }

export const ENGINE_FIELDS: Record<string, FieldDef[]> = {
  vllm: [
    { key: "max_model_len", label: "Context length (tokens)", type: "int", help: "Max tokens per request (prompt + output)." },
    { key: "max_num_seqs", label: "Concurrent sequences", type: "int", help: "Requests decoded in parallel." },
    { key: "kv_cache_dtype", label: "KV cache dtype", type: "select", options: ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"], help: "fp8 halves KV memory." },
    { key: "kv_tokens", label: "KV pool tokens (optional)", type: "int", help: "Total KV tokens across sequences. Empty = context × sequences." },
    { key: "max_num_batched_tokens", label: "Max batched tokens", type: "int", advanced: true },
    { key: "enable_prefix_caching", label: "Prefix caching", type: "bool" },
    { key: "enable_chunked_prefill", label: "Chunked prefill", type: "bool", advanced: true },
    { key: "enable_sleep_mode", label: "Sleep mode (fast switching)", type: "bool" },
    { key: "enable_auto_tool_choice", label: "Auto tool choice", type: "bool", advanced: true },
    { key: "tool_call_parser", label: "Tool-call parser", type: "text", advanced: true },
    { key: "reasoning_parser", label: "Reasoning parser", type: "text", advanced: true },
    { key: "served_model_name", label: "Served model name", type: "text", advanced: true },
    { key: "trust_remote_code", label: "Trust remote code", type: "bool", advanced: true },
    { key: "safetensors_load_strategy", label: "Safetensors load strategy", type: "select", options: ["", "lazy", "eager", "prefetch"], advanced: true },
  ],
  llamacpp: [
    { key: "ctx_size", label: "Context size (all slots)", type: "int" },
    { key: "parallel", label: "Parallel slots", type: "int" },
    { key: "cache_type_k", label: "K cache type", type: "select", options: ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"] },
    { key: "cache_type_v", label: "V cache type", type: "select", options: ["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"] },
    { key: "flash_attn", label: "Flash attention", type: "select", options: ["auto", "on", "off"] },
    { key: "n_gpu_layers", label: "GPU layers", type: "int", advanced: true },
    { key: "n_batch", label: "Batch size", type: "int", advanced: true },
    { key: "n_ubatch", label: "Micro-batch size", type: "int", advanced: true },
    { key: "alias", label: "Served model name", type: "text", advanced: true },
  ],
};
```

- [ ] **Step 4: UI primitives and shared components**

`frontend/src/components/ui.tsx`:
```tsx
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import type { Tone } from "../format";

const TONES: Record<Tone, string> = {
  green: "bg-emerald-100 text-emerald-800", amber: "bg-amber-100 text-amber-800", red: "bg-rose-100 text-rose-800",
  slate: "bg-slate-100 text-slate-700", blue: "bg-sky-100 text-sky-800", violet: "bg-violet-100 text-violet-800",
};

export const Card = ({ title, actions, children, className = "" }: { title?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string }) => (
  <section className={`rounded-lg border border-slate-200 bg-white shadow-sm ${className}`}>
    {(title || actions) && (
      <header className="flex items-center justify-between border-b border-slate-100 px-4 py-2">
        <h2 className="text-sm font-semibold text-slate-700">{title}</h2>
        <div className="flex gap-2">{actions}</div>
      </header>
    )}
    <div className="p-4">{children}</div>
  </section>
);

type Variant = "primary" | "secondary" | "danger" | "ghost";
const VARIANTS: Record<Variant, string> = {
  primary: "bg-slate-900 text-white hover:bg-slate-700", secondary: "border border-slate-300 bg-white hover:bg-slate-50",
  danger: "bg-rose-600 text-white hover:bg-rose-500", ghost: "text-slate-600 hover:bg-slate-100",
};
export const Button = ({ variant = "secondary", className = "", ...p }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) => (
  <button {...p} className={`rounded-md px-3 py-1.5 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-50 ${VARIANTS[variant]} ${className}`} />
);

export const Badge = ({ tone = "slate", children }: { tone?: Tone; children: ReactNode }) => (
  <span className={`inline-flex items-center rounded px-2 py-0.5 text-xs font-medium ${TONES[tone]}`}>{children}</span>
);

export const Stat = ({ label, value, sub }: { label: string; value: ReactNode; sub?: ReactNode }) => (
  <div className="rounded-lg border border-slate-200 bg-white p-3">
    <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
    <div className="mt-1 text-xl font-semibold tabular-nums">{value}</div>
    {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
  </div>
);

export const Field = ({ label, help, children }: { label: string; help?: string; children: ReactNode }) => (
  <label className="block text-sm">
    <span className="font-medium text-slate-700">{label}</span>
    <div className="mt-1">{children}</div>
    {help && <span className="mt-0.5 block text-xs text-slate-500">{help}</span>}
  </label>
);

export const Input = (p: InputHTMLAttributes<HTMLInputElement>) => (
  <input {...p} className={`w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm ${p.className ?? ""}`} />
);
export const Select = (p: SelectHTMLAttributes<HTMLSelectElement>) => (
  <select {...p} className={`w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm ${p.className ?? ""}`} />
);

export const ErrorBox = ({ error }: { error: unknown }) =>
  error ? <div className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">{(error as Error).message ?? String(error)}</div> : null;

export function Modal({ open, title, onClose, children, wide }: { open: boolean; title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-8" onClick={onClose}>
      <div className={`w-full ${wide ? "max-w-4xl" : "max-w-lg"} rounded-lg bg-white shadow-xl`} onClick={(e) => e.stopPropagation()}>
        <header className="flex items-center justify-between border-b px-4 py-3">
          <h3 className="font-semibold">{title}</h3>
          <button className="text-slate-400 hover:text-slate-700" onClick={onClose} aria-label="Close">✕</button>
        </header>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

export const Table = ({ head, children }: { head: string[]; children: ReactNode }) => (
  <div className="overflow-x-auto">
    <table className="min-w-full divide-y divide-slate-200">
      <thead><tr>{head.map((h, i) => <th key={i} className="th">{h}</th>)}</tr></thead>
      <tbody className="divide-y divide-slate-100">{children}</tbody>
    </table>
  </div>
);
```

`frontend/src/components/MemoryBar.tsx`:
```tsx
import type { Budget } from "../types";
import { fmtBytes } from "../format";

const COLORS = ["bg-sky-500", "bg-indigo-500", "bg-teal-500", "bg-fuchsia-500", "bg-cyan-600", "bg-blue-700"];

export function MemoryBar({ budget }: { budget: Budget }) {
  const t = budget.total || 1;
  const managed = budget.deployments.filter((d) => ["starting", "loading", "ready", "sleeping", "unknown", "stopping"].includes(d.state));
  const segs = [
    ...managed.map((d, i) => ({
      label: `${d.name} (${d.state})`,
      bytes: d.state === "sleeping" ? d.actual : Math.max(d.reserved, d.actual),
      cls: d.state === "sleeping" ? "bg-violet-400" : COLORS[i % COLORS.length],
    })),
    { label: "External processes", bytes: budget.external, cls: "bg-amber-400" },
    { label: "Safety margin + OS reserve", bytes: budget.margin + budget.os_reserve, cls: "bg-slate-400" },
    { label: "Free for new deployments", bytes: Math.max(budget.free_for_new, 0), cls: "bg-emerald-200" },
  ];
  return (
    <div>
      <div className="flex h-6 w-full overflow-hidden rounded-md border border-slate-200" role="img" aria-label="Memory usage">
        {segs.map((s) => (
          <div key={s.label} className={s.cls} style={{ width: `${(s.bytes / t) * 100}%` }} title={`${s.label}: ${fmtBytes(s.bytes)}`} />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">
        {segs.map((s) => (
          <li key={s.label} className="flex items-center gap-1.5">
            <span className={`inline-block h-2.5 w-2.5 rounded-sm ${s.cls}`} />
            {s.label}: <span className="tabular-nums">{fmtBytes(s.bytes)}</span>
          </li>
        ))}
      </ul>
      {budget.free_for_new < 0 && (
        <p className="mt-2 text-xs text-rose-700">Over-committed by {fmtBytes(-budget.free_for_new)}. New deployments are blocked.</p>
      )}
    </div>
  );
}
```

`frontend/src/components/EstimatePanel.tsx`:
```tsx
import type { Preview } from "../types";
import { fmtBytes } from "../format";
import { Badge } from "./ui";

export function EstimatePanel({ preview }: { preview: Preview }) {
  const e = preview.estimate;
  const rows: [string, number][] = [
    ["Weights", e.weights], ["Draft weights", e.draft_weights], ["KV cache", e.kv_cache], ["Recurrent state", e.state],
    ["Activations", e.activations], ["Runtime (CUDA/graphs)", e.runtime], ["Margin / calibration", e.margin],
  ];
  const f = preview.fit;
  return (
    <div className="space-y-3 text-sm">
      <div className="flex items-center gap-2">
        {f.fits ? <Badge tone="green">Fits</Badge> : <Badge tone="red">Does not fit</Badge>}
        <span>
          needs <b>{fmtBytes(e.total)}</b>, free <b>{fmtBytes(f.free_for_new)}</b>,{" "}
          {f.fits ? <>headroom {fmtBytes(f.headroom)}</> : <>short by {fmtBytes(-f.headroom)}</>}
        </span>
        <Badge tone={e.confidence === "measured" ? "green" : e.confidence === "low" ? "amber" : "slate"}>{e.confidence}</Badge>
      </div>
      <table className="w-full text-xs">
        <tbody>
          {rows.filter(([, v]) => v !== 0).map(([k, v]) => (
            <tr key={k}><td className="py-0.5 text-slate-500">{k}</td><td className="py-0.5 text-right tabular-nums">{fmtBytes(v)}</td></tr>
          ))}
          {preview.gpu_memory_utilization !== undefined && (
            <tr><td className="py-0.5 text-slate-500">vLLM --gpu-memory-utilization</td><td className="py-0.5 text-right tabular-nums">{preview.gpu_memory_utilization.toFixed(2)}</td></tr>
          )}
        </tbody>
      </table>
      {e.notes.map((n) => <p key={n} className="text-xs text-amber-700">{n}</p>)}
      {preview.llmfit && (
        <p className="text-xs text-slate-600">
          llmfit: {preview.llmfit.fit_level ?? "?"} · est. {preview.llmfit.estimated_tps ?? "?"} tok/s · memory{" "}
          {preview.llmfit.memory_required_gb ?? "?"} GB
          {preview.llmfit.mismatch && <span className="ml-1 text-amber-700">(differs from DGXMM by over 20%)</span>}
        </p>
      )}
    </div>
  );
}
```

`frontend/src/components/ConfigForm.tsx`:
```tsx
import { ENGINE_FIELDS } from "../engineFields";
import { Field, Input, Select } from "./ui";

export function ConfigForm({ engine, value, onChange, showAdvanced }: {
  engine: string; value: Record<string, any>; onChange: (v: Record<string, any>) => void; showAdvanced: boolean;
}) {
  const set = (k: string, v: unknown) => {
    const next = { ...value };
    if (v === "" || v === null || (typeof v === "number" && Number.isNaN(v))) delete next[k];
    else next[k] = v;
    onChange(next);
  };
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
      {(ENGINE_FIELDS[engine] ?? []).filter((f) => showAdvanced || !f.advanced).map((f) => (
        <Field key={f.key} label={f.label} help={f.help}>
          {f.type === "bool" ? (
            <input type="checkbox" checked={!!value[f.key]} onChange={(e) => set(f.key, e.target.checked)} />
          ) : f.type === "select" ? (
            <Select value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)}>
              {f.options!.map((o) => <option key={o} value={o}>{o || "(default)"}</option>)}
            </Select>
          ) : f.type === "int" ? (
            <Input type="number" value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value === "" ? "" : parseInt(e.target.value, 10))} />
          ) : (
            <Input value={value[f.key] ?? ""} onChange={(e) => set(f.key, e.target.value)} />
          )}
        </Field>
      ))}
    </div>
  );
}
```

`frontend/src/components/Layout.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { useLiveUpdates } from "../stream";
import type { HostInfo } from "../types";

const NAV = [
  { to: "/", label: "Dashboard", admin: false }, { to: "/models", label: "Models on disk", admin: false },
  { to: "/catalog", label: "Catalog", admin: false }, { to: "/downloads", label: "Downloads", admin: false },
  { to: "/planner", label: "Planner", admin: false }, { to: "/audit", label: "Audit log", admin: true },
  { to: "/users", label: "Users", admin: true },
];

export function Layout() {
  const { user, logout } = useAuth();
  useLiveUpdates(!!user);
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const alert = host.data?.alert;
  return (
    <div className="flex min-h-screen">
      <aside className="w-52 shrink-0 border-r border-slate-200 bg-white">
        <div className="px-4 py-4 text-sm font-bold">DGX Model Manager</div>
        <nav className="flex flex-col gap-0.5 px-2">
          {NAV.filter((n) => !n.admin || user?.role === "admin").map((n) => (
            <NavLink key={n.to} to={n.to} end={n.to === "/"}
              className={({ isActive }) => `rounded px-3 py-1.5 text-sm ${isActive ? "bg-slate-900 text-white" : "text-slate-700 hover:bg-slate-100"}`}>
              {n.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-6 border-t px-4 py-3 text-xs text-slate-500">
          {user?.username} ({user?.role})
          <button className="ml-2 underline" onClick={logout}>log out</button>
        </div>
      </aside>
      <main className="flex-1 space-y-4 p-6">
        {alert && alert.level !== "ok" && (
          <div className={`rounded-md px-4 py-2 text-sm ${alert.level === "critical" ? "bg-rose-600 text-white" : "bg-amber-100 text-amber-900"}`}>
            <b>{alert.level === "critical" ? "Memory critical" : "Memory warning"}:</b> {alert.message}
            {alert.action && <span className="ml-2">({alert.action})</span>}
          </div>
        )}
        <Outlet />
      </main>
    </div>
  );
}
```

- [ ] **Step 5: Pages**

`frontend/src/pages/Login.tsx`:
```tsx
import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { Button, ErrorBox, Field, Input } from "../components/ui";

export function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const loc = useLocation() as { state?: { from?: string } };
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  return (
    <div className="flex min-h-screen items-center justify-center">
      <form className="w-80 space-y-3 rounded-lg border bg-white p-6 shadow-sm"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setErr(null);
          try {
            await login(u, p);
            nav(loc.state?.from ?? "/", { replace: true });
          } catch (x) {
            setErr(x);
          } finally {
            setBusy(false);
          }
        }}>
        <h1 className="text-lg font-semibold">DGX Model Manager</h1>
        <Field label="Username"><Input autoFocus value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" /></Field>
        <Field label="Password"><Input type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete="current-password" /></Field>
        <ErrorBox error={err} />
        <Button variant="primary" className="w-full" disabled={busy || !u || !p}>Log in</Button>
      </form>
    </div>
  );
}
```

`frontend/src/pages/ChangePassword.tsx`:
```tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useAuth } from "../auth";
import { Button, Card, ErrorBox, Field, Input } from "../components/ui";

export function ChangePassword() {
  const { reload, user } = useAuth();
  const nav = useNavigate();
  const [cur, setCur] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [err, setErr] = useState<unknown>(null);
  return (
    <div className="mx-auto mt-16 max-w-md">
      <Card title={user?.must_change_pw ? "Set a new password to continue" : "Change password"}>
        <form className="space-y-3" onSubmit={async (e) => {
          e.preventDefault();
          if (next !== again) return setErr(new Error("passwords do not match"));
          try {
            await api("/auth/change-password", { method: "POST", body: { current_password: cur, new_password: next } });
            await reload();
            nav("/", { replace: true });
          } catch (x) {
            setErr(x);
          }
        }}>
          <Field label="Current password"><Input type="password" value={cur} onChange={(e) => setCur(e.target.value)} /></Field>
          <Field label="New password" help="At least 12 characters."><Input type="password" value={next} onChange={(e) => setNext(e.target.value)} /></Field>
          <Field label="Repeat new password"><Input type="password" value={again} onChange={(e) => setAgain(e.target.value)} /></Field>
          <ErrorBox error={err} />
          <Button variant="primary" disabled={!cur || next.length < 12}>Save</Button>
        </form>
      </Card>
    </div>
  );
}
```

`frontend/src/pages/Dashboard.tsx`:
```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { MemoryBar } from "../components/MemoryBar";
import { Badge, Button, Card, ErrorBox, Stat, Table } from "../components/ui";
import { fmtBytes, fmtNum, fmtPct, stateTone } from "../format";
import type { Deployment, HostInfo } from "../types";

function LiveCell({ id, field, fmt }: { id: number; field: string; fmt: (v: any) => string }) {
  const q = useQuery<any[]>({ queryKey: ["live", id], queryFn: () => Promise.resolve([]), staleTime: Infinity });
  const last = q.data?.[q.data.length - 1];
  return <span className="tabular-nums">{last ? fmt(last[field]) : "—"}</span>;
}

export function Dashboard() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const host = useQuery({ queryKey: ["host"], queryFn: () => api<HostInfo>("/host") });
  const deps = useQuery({ queryKey: ["deployments"], queryFn: () => api<Deployment[]>("/deployments") });
  const act = useMutation({
    mutationFn: ({ id, action }: { id: number; action: string }) => api(`/deployments/${id}/${action}`, { method: "POST" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["deployments"] }),
  });
  const b = host.data?.budget;
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Dashboard</h1>
        {admin && <Link to="/deploy"><Button variant="primary">Deploy a model</Button></Link>}
      </div>
      {b && (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-6">
            <Stat label="Total memory" value={fmtBytes(b.total)} />
            <Stat label="Available (kernel)" value={fmtBytes(b.available)} />
            <Stat label="Free for new" value={fmtBytes(b.free_for_new)} sub={`after ${fmtBytes(b.margin)} safety margin`} />
            <Stat label="Managed models" value={fmtBytes(b.committed)} sub={`${fmtBytes(b.managed_actual)} measured`} />
            <Stat label="External" value={fmtBytes(b.external)} sub="processes DGXMM does not manage" />
            <Stat label="Model disk free" value={fmtBytes(host.data?.disk?.free)} sub={host.data?.gpu?.util != null ? `GPU ${host.data.gpu.util}% · ${host.data.gpu.temp_c ?? "?"}°C` : undefined} />
          </div>
          <Card title="Unified memory"><MemoryBar budget={b} /></Card>
        </>
      )}
      <ErrorBox error={host.error || deps.error || act.error} />
      <Card title="Deployments">
        <Table head={["Name", "Model", "Engine", "State", "Reserved / actual", "Gen tok/s", "KV", "Queue", "LiteLLM", ""]}>
          {(deps.data ?? []).map((d) => (
            <tr key={d.id}>
              <td className="td font-medium"><Link className="underline" to={`/deployments/${d.id}`}>{d.name}</Link></td>
              <td className="td">{d.model_ref}</td>
              <td className="td">{d.engine}</td>
              <td className="td"><Badge tone={stateTone(d.state)}>{d.state}</Badge>{d.last_error && <div className="mt-1 max-w-xs text-xs text-rose-700">{d.last_error}</div>}</td>
              <td className="td tabular-nums">{fmtBytes(d.reserved_bytes)} / {fmtBytes(d.actual_bytes)}</td>
              <td className="td"><LiveCell id={d.id} field="gen_tps" fmt={(v) => fmtNum(v, 1)} /></td>
              <td className="td"><LiveCell id={d.id} field="kv_usage" fmt={fmtPct} /></td>
              <td className="td"><LiveCell id={d.id} field="waiting" fmt={(v) => (v == null ? "—" : String(v))} /></td>
              <td className="td"><Badge tone={d.litellm_status === "synced" ? "green" : d.litellm_status === "error" ? "red" : "slate"}>{d.litellm_status}</Badge></td>
              <td className="td whitespace-nowrap space-x-1">
                {admin && ["ready", "sleeping", "starting", "loading", "unknown"].includes(d.state) && (
                  <Button variant="danger" onClick={() => confirm(`Stop ${d.name}?`) && act.mutate({ id: d.id, action: "stop" })}>Stop</Button>
                )}
                {admin && ["stopped", "failed"].includes(d.state) && (
                  <Button onClick={() => act.mutate({ id: d.id, action: "start" })}>Start</Button>
                )}
              </td>
            </tr>
          ))}
        </Table>
        {deps.data?.length === 0 && <p className="text-sm text-slate-500">Nothing deployed yet.</p>}
      </Card>
      {host.data && host.data.external_top.length > 0 && (
        <Card title="Largest external memory users (not managed; never touched)">
          <Table head={["PID", "Process", "RSS", "GPU", "Command"]}>
            {host.data.external_top.map((p) => (
              <tr key={p.pid}>
                <td className="td tabular-nums">{p.pid}</td><td className="td">{p.name}</td>
                <td className="td tabular-nums">{fmtBytes(p.rss)}</td><td className="td tabular-nums">{fmtBytes(p.gpu)}</td>
                <td className="td max-w-md truncate font-mono text-xs" title={p.cmdline}>{p.cmdline}</td>
              </tr>
            ))}
          </Table>
        </Card>
      )}
    </div>
  );
}
```

`frontend/src/pages/Models.tsx`:
```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Table } from "../components/ui";
import { fmtBytes } from "../format";
import type { Artifact } from "../types";

export function Models() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const refresh = useMutation({
    mutationFn: () => api<Artifact[]>("/inventory/refresh", { method: "POST" }),
    onSuccess: (d) => qc.setQueryData(["inventory"], d),
  });
  const rows = inv.data ?? [];
  const total = rows.reduce((s, a) => s + a.size_bytes, 0);
  return (
    <Card title={`Models on disk (${rows.length}, ${fmtBytes(total)})`}
      actions={admin && <Button onClick={() => refresh.mutate()} disabled={refresh.isPending}>Rescan</Button>}>
      <ErrorBox error={inv.error || refresh.error} />
      <Table head={["Model", "Kind", "Format / quant", "Weights", "Context", "Architecture", "Status", ""]}>
        {rows.map((a) => (
          <tr key={a.ref}>
            <td className="td font-medium break-all">{a.ref}{a.revision && <div className="font-mono text-xs text-slate-400">{a.revision.slice(0, 8)}</div>}</td>
            <td className="td"><Badge tone={a.kind === "llm" ? "blue" : a.kind === "draft" ? "violet" : "slate"}>{a.kind}</Badge></td>
            <td className="td">{a.format} {a.quant && <Badge>{a.quant}</Badge>} {a.has_mtp && <Badge tone="violet">MTP</Badge>}</td>
            <td className="td tabular-nums">{fmtBytes(a.weights_bytes)}</td>
            <td className="td tabular-nums">{a.max_context?.toLocaleString() ?? "—"}</td>
            <td className="td text-xs">{a.architecture}{a.is_moe && <Badge tone="amber">MoE</Badge>}</td>
            <td className="td">
              {a.complete ? <Badge tone="green">complete</Badge> : <Badge tone="red">incomplete</Badge>}
              {a.problems.map((p) => <div key={p} className="mt-1 text-xs text-amber-700">{p}</div>)}
            </td>
            <td className="td">
              {admin && a.complete && a.kind === "llm" && a.engines.length > 0 && (
                <Link to={`/deploy?model=${encodeURIComponent(a.ref)}`}><Button>Deploy</Button></Link>
              )}
            </td>
          </tr>
        ))}
      </Table>
    </Card>
  );
}
```

`frontend/src/pages/DeployWizard.tsx`:
```tsx
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ApiError, api } from "../api";
import { ConfigForm } from "../components/ConfigForm";
import { EstimatePanel } from "../components/EstimatePanel";
import { PlanChooser } from "../components/PlanChooser";
import { Button, Card, ErrorBox, Field, Input, Select } from "../components/ui";
import type { Artifact, EngineInfo, PlanAction, Preview } from "../types";

const slug = (ref: string) => ref.split("/").pop()!.toLowerCase().replace(/\.gguf$/, "").replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "model";

export function DeployWizard() {
  const [params] = useSearchParams();
  const nav = useNavigate();
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const engines = useQuery({ queryKey: ["engines"], queryFn: () => api<EngineInfo[]>("/engines") });
  const deployable = (inv.data ?? []).filter((a) => a.complete && a.kind === "llm" && a.engines.length > 0);
  const [ref, setRef] = useState(params.get("model") ?? "");
  const artifact = deployable.find((a) => a.ref === ref);
  const [engine, setEngine] = useState("");
  const [name, setName] = useState("");
  const [priority, setPriority] = useState(0);
  const [alias, setAlias] = useState("");
  const [config, setConfig] = useState<Record<string, any>>({});
  const [advanced, setAdvanced] = useState(false);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewErr, setPreviewErr] = useState<unknown>(null);
  const [actions, setActions] = useState<PlanAction[]>([]);

  useEffect(() => {
    if (!artifact) return;
    const eng = artifact.engines[0];
    setEngine(eng);
    setName(slug(artifact.ref));
    const defaults = { ...(engines.data?.find((e) => e.name === eng)?.defaults ?? {}) };
    const ctxKey = eng === "vllm" ? "max_model_len" : "ctx_size";
    if (artifact.max_context && defaults[ctxKey] > artifact.max_context) defaults[ctxKey] = artifact.max_context;
    for (const k of Object.keys(defaults)) if (defaults[k] === null || (Array.isArray(defaults[k]) && !defaults[k].length)) delete defaults[k];
    setConfig(defaults);
    setActions([]);
  }, [artifact?.ref, engines.data]); // eslint-disable-line react-hooks/exhaustive-deps

  const body = useMemo(() => ({ engine, model_ref: ref, config }), [engine, ref, config]);
  useEffect(() => {
    if (!engine || !ref) return;
    const t = setTimeout(async () => {
      try {
        setPreview(await api<Preview>("/deployments/preview", { method: "POST", body }));
        setPreviewErr(null);
      } catch (e) {
        setPreviewErr(e);
        setPreview(null);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [body, engine, ref]);

  const deploy = useMutation({
    mutationFn: () => api<{ id: number }>("/deployments", {
      method: "POST",
      body: { name, engine, model_ref: ref, config, priority, alias: alias || null, actions },
    }),
    onSuccess: (d) => nav(`/deployments/${d.id}`),
  });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <Card title="Deploy a model" className="lg:col-span-2">
        <div className="space-y-4">
          <Field label="Model">
            <Select value={ref} onChange={(e) => setRef(e.target.value)}>
              <option value="">Choose a downloaded model…</option>
              {deployable.map((a) => <option key={a.ref} value={a.ref}>{a.ref} ({a.quant})</option>)}
            </Select>
          </Field>
          {artifact && (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Field label="Engine">
                  <Select value={engine} onChange={(e) => { setEngine(e.target.value); setConfig(engines.data?.find((x) => x.name === e.target.value)?.defaults ?? {}); }}>
                    {artifact.engines.map((e) => <option key={e}>{e}</option>)}
                  </Select>
                </Field>
                <Field label="Deployment name"><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>
                <Field label="Priority" help="Lower is evicted first."><Input type="number" value={priority} onChange={(e) => setPriority(parseInt(e.target.value || "0", 10))} /></Field>
                <Field label="LiteLLM alias" help="Optional stable name."><Input value={alias} onChange={(e) => setAlias(e.target.value)} /></Field>
              </div>
              <ConfigForm engine={engine} value={config} onChange={setConfig} showAdvanced={advanced} />
              <button className="text-xs underline" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced settings</button>
            </>
          )}
          <ErrorBox error={deploy.error} />
          <Button variant="primary" disabled={!preview || deploy.isPending || (!preview.fit.fits && actions.length === 0)} onClick={() => deploy.mutate()}>
            {actions.length ? "Apply plan and deploy" : "Deploy"}
          </Button>
          {deploy.error instanceof ApiError && deploy.error.code === "insufficient_memory" && (
            <p className="text-xs text-rose-700">Memory changed since the preview. Choose a plan again.</p>
          )}
        </div>
      </Card>
      <Card title="Memory check">
        <ErrorBox error={previewErr} />
        {preview ? <EstimatePanel preview={preview} /> : <p className="text-sm text-slate-500">Choose a model to see the estimate.</p>}
        {preview && !preview.fit.fits && (
          <PlanChooser plans={preview.plans ?? []} selected={actions}
            onSelect={(opt) => { setActions(opt.actions.filter((a) => a.type !== "patch")); if (opt.patch) setConfig({ ...config, ...opt.patch }); }} />
        )}
      </Card>
    </div>
  );
}
```

`frontend/src/components/PlanChooser.tsx` (Task 21 fills in the plans; the component exists from M1 so the wizard compiles):
```tsx
import type { PlanAction, PlanOption } from "../types";
import { fmtBytes } from "../format";
import { Badge } from "./ui";

export function PlanChooser({ plans, selected, onSelect }: { plans: PlanOption[]; selected: PlanAction[]; onSelect: (p: PlanOption) => void }) {
  if (!plans.length) return <p className="mt-3 text-sm text-slate-500">No automatic plan frees enough memory. Stop something manually or pick a smaller configuration.</p>;
  const key = (a: PlanAction[]) => JSON.stringify(a);
  return (
    <div className="mt-4 space-y-2">
      <h3 className="text-sm font-semibold">Ways to make it fit</h3>
      {plans.map((p) => (
        <button key={p.id} onClick={() => onSelect(p)}
          className={`block w-full rounded-md border px-3 py-2 text-left text-sm hover:bg-slate-50 ${key(p.actions.filter((a) => a.type !== "patch")) === key(selected) && selected.length ? "border-slate-900" : "border-slate-200"}`}>
          <div className="flex items-center justify-between">
            <span className="font-medium">{p.title}</span>
            {p.fits ? <Badge tone="green">fits</Badge> : <Badge tone="amber">still short</Badge>}
          </div>
          <div className="text-xs text-slate-600">{p.detail}</div>
          <div className="text-xs text-slate-500">needs {fmtBytes(p.new_required)}{p.frees ? ` · frees ${fmtBytes(p.frees)}` : ""}</div>
        </button>
      ))}
    </div>
  );
}
```

`frontend/src/pages/DeploymentDetail.tsx`:
```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox } from "../components/ui";
import { fmtBytes, stateTone } from "../format";
import type { Deployment } from "../types";

export function DeploymentDetail() {
  const id = Number(useParams().id);
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const dep = useQuery({ queryKey: ["deployment", id], queryFn: () => api<Deployment>(`/deployments/${id}`) });
  const [logs, setLogs] = useState<string | null>(null);
  const act = useMutation({
    mutationFn: (action: string) => api(`/deployments/${id}/${action}`, { method: "POST" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["deployment", id] }),
  });
  const d = dep.data;
  if (!d) return <ErrorBox error={dep.error} />;
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">{d.name}</h1>
        <Badge tone={stateTone(d.state)}>{d.state}</Badge>
        <span className="text-sm text-slate-500">{d.model_ref} · {d.engine}</span>
        <div className="ml-auto space-x-2">
          {admin && ["ready", "sleeping", "starting", "loading", "unknown"].includes(d.state) && <Button variant="danger" onClick={() => act.mutate("stop")}>Stop</Button>}
          {admin && ["stopped", "failed"].includes(d.state) && <Button onClick={() => act.mutate("start")}>Start</Button>}
          {admin && <Button onClick={async () => setLogs((await api<{ logs: string }>(`/deployments/${id}/logs?tail=300`)).logs)}>Logs</Button>}
        </div>
      </div>
      <ErrorBox error={act.error} />
      {d.last_error && <div className="rounded-md bg-rose-50 px-3 py-2 text-sm text-rose-800">{d.last_error}</div>}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Endpoint">
          <dl className="space-y-1 text-sm">
            <div><dt className="text-slate-500">Served name</dt><dd className="font-mono">{d.served_name}</dd></div>
            <div><dt className="text-slate-500">On gateway network</dt><dd className="font-mono">{d.endpoint}</dd></div>
            <div><dt className="text-slate-500">On the DGX</dt><dd className="font-mono">{d.host_endpoint}</dd></div>
            <div><dt className="text-slate-500">LiteLLM</dt><dd>{d.litellm_status}{d.alias && <> · alias <span className="font-mono">{d.alias}</span></>}</dd></div>
          </dl>
        </Card>
        <Card title="Memory">
          <p className="text-sm">Reserved {fmtBytes(d.reserved_bytes)} · measured {fmtBytes(d.actual_bytes)}</p>
          {"total" in d.estimate && <p className="mt-1 text-xs text-slate-500">estimate confidence: {(d.estimate as any).confidence}</p>}
        </Card>
        <Card title="Configuration">
          <pre className="max-h-64 overflow-auto text-xs">{JSON.stringify(d.config, null, 2)}</pre>
        </Card>
      </div>
      <Card title="Events">
        <ol className="space-y-2 text-sm">
          {(d.events ?? []).map((e, i) => (
            <li key={i}>
              <span className="font-mono text-xs text-slate-500">{new Date(e.ts).toLocaleString()}</span>{" "}
              <Badge tone={stateTone(e.to)}>{e.to}</Badge> {e.message}
              {e.log_tail && <details className="mt-1"><summary className="cursor-pointer text-xs underline">log tail</summary><pre className="max-h-64 overflow-auto bg-slate-900 p-2 text-xs text-slate-100">{e.log_tail}</pre></details>}
            </li>
          ))}
        </ol>
      </Card>
      {logs !== null && (
        <Card title="Container logs (last 300 lines)" actions={<Button variant="ghost" onClick={() => setLogs(null)}>Close</Button>}>
          <pre className="max-h-[32rem] overflow-auto bg-slate-900 p-3 text-xs text-slate-100">{logs || "(empty)"}</pre>
        </Card>
      )}
    </div>
  );
}
```

`frontend/src/pages/Users.tsx`:
```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { useAuth } from "../auth";
import { Badge, Button, Card, ErrorBox, Field, Input, Select, Table } from "../components/ui";
import type { User } from "../types";

export function Users() {
  const qc = useQueryClient();
  const me = useAuth().user;
  const users = useQuery({ queryKey: ["users"], queryFn: () => api<User[]>("/users") });
  const [form, setForm] = useState({ username: "", password: "", role: "viewer" });
  const done = () => qc.invalidateQueries({ queryKey: ["users"] });
  const create = useMutation({ mutationFn: () => api("/users", { method: "POST", body: form }), onSuccess: () => { setForm({ username: "", password: "", role: "viewer" }); done(); } });
  const patch = useMutation({ mutationFn: ({ id, body }: { id: number; body: object }) => api(`/users/${id}`, { method: "PATCH", body }), onSuccess: done });
  const del = useMutation({ mutationFn: (id: number) => api(`/users/${id}`, { method: "DELETE" }), onSuccess: done });
  return (
    <div className="space-y-4">
      <Card title="Users">
        <ErrorBox error={users.error || patch.error || del.error} />
        <Table head={["Username", "Role", "", ""]}>
          {(users.data ?? []).map((u) => (
            <tr key={u.id}>
              <td className="td">{u.username} {u.must_change_pw && <Badge tone="amber">must change password</Badge>}</td>
              <td className="td">
                <Select value={u.role} disabled={u.id === me?.id} onChange={(e) => patch.mutate({ id: u.id, body: { role: e.target.value } })}>
                  <option value="viewer">viewer</option><option value="admin">admin</option>
                </Select>
              </td>
              <td className="td"><Button onClick={() => { const pw = prompt(`New temporary password for ${u.username} (12+ chars)`); if (pw) patch.mutate({ id: u.id, body: { password: pw } }); }}>Reset password</Button></td>
              <td className="td">{u.id !== me?.id && <Button variant="danger" onClick={() => confirm(`Delete ${u.username}?`) && del.mutate(u.id)}>Delete</Button>}</td>
            </tr>
          ))}
        </Table>
      </Card>
      <Card title="Add user">
        <form className="grid grid-cols-1 gap-3 md:grid-cols-4" onSubmit={(e) => { e.preventDefault(); create.mutate(); }}>
          <Field label="Username"><Input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value.toLowerCase() })} /></Field>
          <Field label="Temporary password"><Input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
          <Field label="Role"><Select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}><option>viewer</option><option>admin</option></Select></Field>
          <div className="flex items-end"><Button variant="primary">Add</Button></div>
        </form>
        <ErrorBox error={create.error} />
      </Card>
    </div>
  );
}
```

`frontend/src/main.tsx` (Tasks 21, 30 and 35 add more routes):
```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import { AuthProvider, RequireAuth } from "./auth";
import { Layout } from "./components/Layout";
import { ChangePassword } from "./pages/ChangePassword";
import { Dashboard } from "./pages/Dashboard";
import { DeploymentDetail } from "./pages/DeploymentDetail";
import { DeployWizard } from "./pages/DeployWizard";
import { Login } from "./pages/Login";
import { Models } from "./pages/Models";
import { Users } from "./pages/Users";

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } });

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={qc}>
      <BrowserRouter basename="/dgxmm">
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/change-password" element={<RequireAuth><ChangePassword /></RequireAuth>} />
            <Route element={<RequireAuth><Layout /></RequireAuth>}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/models" element={<Models />} />
              <Route path="/deploy" element={<DeployWizard />} />
              <Route path="/deployments/:id" element={<DeploymentDetail />} />
              <Route path="/users" element={<Users />} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
```

The deploy body now includes `actions`, so update the backend in this task too. Add an `actions` field to the create route by wrapping the request:
In `dgxmm/api/deployments.py`, replace `async def create(body: DeployRequest, …)` with:
```python
class DeployIn(DeployRequest):
    actions: list[dict] = []


@router.post("", status_code=201)
async def create(body: DeployIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    req = DeployRequest(**body.model_dump(exclude={"actions"}))
    try:
        if body.actions:
            await ctx.deployer.apply_actions(body.actions, p.user.id)  # implemented in Task 17
        view = await ctx.deployer.deploy(req, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "deploy", req.name, {"model": req.model_ref, "engine": req.engine, "actions": body.actions})
    return view
```
Until Task 17 lands, a non-empty `actions` list raises `AttributeError`. That is acceptable because the UI only sends actions once plans exist (Task 21).

- [ ] **Step 6: Build and test the frontend**

```bash
rsync … (push mirror)
NPM$ sh -c "npm install --no-audit --no-fund && npx vitest run && npm run build"
```
Expected: vitest `2 passed`; `tsc` reports no errors; `vite build` writes `frontend/dist/index.html` and `frontend/dist/assets/*`. Pull `package-lock.json` back to the mirror: `rsync -az dgx-host:dgxmodelmanager/frontend/package-lock.json /tmp/dgxmm/frontend/`.

---

### Task 15: Packaging, nginx, and the M1 on-box check

**Files:**
- Create: `Dockerfile`, `docker-compose.yaml`, `.dockerignore`, `deploy/nginx-dgxmm.conf`

**Interfaces:**
- Consumes: the built frontend (`frontend/dist`) and the `dgxmm` package.
- Produces: the `dgxmm` container on the `gateway` network, listening on 8090 (published on `127.0.0.1:8090`).

- [ ] **Step 1: Write the packaging files**

`Dockerfile`:
```dockerfile
FROM node:22-bookworm-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/data/hf HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /app
COPY pyproject.toml ./
COPY dgxmm ./dgxmm
RUN pip install --no-cache-dir . && useradd -u 1000 -m dgxmm && mkdir -p /app/data && chown 1000:1000 /app/data
COPY --from=ui /ui/dist /app/static
USER 1000:1000
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/dgxmm/').status==200 else 1)"
CMD ["python", "-m", "dgxmm"]
```

`.dockerignore`:
```
.venv
frontend/node_modules
frontend/dist
data
secrets
tests
docs
.git
```

`docker-compose.yaml`:
```yaml
services:
  dgxmm:
    build: .
    image: dgxmm:latest
    container_name: dgxmm
    restart: unless-stopped
    user: "1000:1000"
    group_add: ["988"]            # docker group on dgx-host (docker.sock access)
    runtime: nvidia               # NVML for per-process GPU memory / util
    environment:
      DGXMM_CONFIG: /app/config.yaml
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: utility
    ports:
      - "127.0.0.1:8090:8090"
    networks: [gateway]
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /proc:/host/proc:ro
      - /sys/fs/cgroup:/host/sys/fs/cgroup:ro
      - /home/dgxuser/.cache/huggingface:/data/hf
      - /home/dgxuser/models:/data/gguf/0
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml:ro
    secrets: [litellm_master_key, hf_token]

networks:
  gateway:
    external: true

secrets:
  litellm_master_key:
    file: ./secrets/litellm_master_key
  hf_token:
    file: ./secrets/hf_token
```

`deploy/nginx-dgxmm.conf` (paste inside the first `server { listen 80; server_name model.example.com; … }` block, which is also the default server for plain-IP access):
```nginx
    # DGX Model Manager. Variable upstream + Docker DNS so nginx still starts when dgxmm is down.
    location /dgxmm/ {
        resolver 127.0.0.11 valid=10s ipv6=off;
        set $dgxmm_upstream http://dgxmm:8090;
        proxy_pass $dgxmm_upstream;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;          # SSE
        proxy_read_timeout 1h;
        client_max_body_size 1m;
    }
```

- [ ] **Step 2: Prepare runtime files on the box**

```bash
DGX$ mkdir -p data secrets ~/models && chmod 700 secrets
DGX$ test -f config.yaml || cp config.example.yaml config.yaml
DGX$ touch secrets/hf_token && chmod 600 secrets/hf_token
DGX$ test -s secrets/litellm_master_key || { docker exec litellm-proxy sh -c 'grep -E "^\s*master_key:" /app/config.yaml | sed -E "s/.*master_key:\s*//"' > secrets/litellm_master_key; chmod 600 secrets/litellm_master_key; }
DGX$ test -s secrets/litellm_master_key && echo "key captured (not printed)"
```
(`secrets/` is git-ignored. The key is never echoed.)

- [ ] **Step 3: Build and start**

```bash
DGX$ docker compose build && docker compose up -d && sleep 5 && docker compose logs --tail=50 dgxmm
```
Expected:
- The log shows `DGXMM first run: created user 'admin' with password …`. **Hand this password to the user privately. Do not paste it anywhere else.**
- `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/dgxmm/` prints `200`.

- [ ] **Step 4: [USER] Wire nginx**

DGXMM only inserts the snippet with the user's explicit OK, because the reverse proxy's `nginx.conf` belongs to another project. With approval:
```bash
DGX$ cp <proxy>/nginx.conf <proxy>/nginx.conf.bak-dgxmm-$(date +%Y%m%d)
```
- Insert the snippet after the line `server_name model.example.com;` with an editor. Show the diff to the user first.
- Then run `docker exec <nginx container> nginx -t && docker exec <nginx container> nginx -s reload`.
- Expected: `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1/dgxmm/` → `200`.

- [ ] **Step 5: M1 on-box check with a tiny model**

This deploys `Qwen/Qwen3-0.6B` (about 6 GiB) on the live box. Tell the user before running it.
```bash
DGX$ cat > /tmp/m1check.sh <<'EOF'
set -e
B=http://127.0.0.1:8090/dgxmm/api; J=/tmp/dgxmm.cookies; rm -f $J
read -rsp "dgxmm admin password: " PW; echo
CSRF=$(curl -s -c $J -H 'content-type: application/json' -d "{\"username\":\"admin\",\"password\":\"$PW\"}" $B/auth/login | python3 -c 'import sys,json;print(json.load(sys.stdin)["csrf_token"])')
H=(-b $J -H "X-CSRF-Token: $CSRF" -H 'content-type: application/json')
curl -s "${H[@]}" $B/inventory | python3 -c 'import sys,json;[print(a["ref"],a["complete"],a["engines"]) for a in json.load(sys.stdin)]'
curl -s "${H[@]}" -d '{"engine":"vllm","model_ref":"Qwen/Qwen3-0.6B","config":{"max_model_len":8192,"max_num_seqs":4}}' $B/deployments/preview | python3 -m json.tool | head -40
curl -s "${H[@]}" -d '{"name":"m1-check","engine":"vllm","model_ref":"Qwen/Qwen3-0.6B","config":{"max_model_len":8192,"max_num_seqs":4}}' $B/deployments | python3 -m json.tool
EOF
DGX$ bash /tmp/m1check.sh
```
If `admin` still has `must_change_pw`, the user changes the password in the UI first.
Expected:
- The inventory lists about 15 repos.
- The preview shows `fits: true` and a `gpu_memory_utilization` of roughly 0.05–0.08.
- Deploy returns `state: starting`.
- Watch `docker logs -f dgxmm-m1-check`. Within about 3 minutes, `GET $B/deployments` shows `ready`.
- `curl -s http://127.0.0.1:<port>/v1/models` lists `qwen3-0.6b`.
- Compare `reserved_bytes` with `actual_bytes` in `GET $B/deployments`. Record both numbers in the task log (Task 32 calibrates from them).
- Then stop and remove: `curl -s "${H[@]}" -X POST $B/deployments/<id>/stop` followed by `curl -s "${H[@]}" -X DELETE $B/deployments/<id>`.


## Milestone M2: Switching

### Task 16: Eviction and reduction planner

**Files:**
- Create: `dgxmm/planner/plans.py`
- Modify: `dgxmm/deployer/service.py` (add `plans_for`; `preview` includes `plans` when it does not fit)
- Modify: `dgxmm/api/deployments.py` (409 `insufficient_memory` responses include `plans`)
- Test: `tests/test_plans.py`

**Interfaces:**
- Consumes: `FitResult`, `Prepared`, engine `context_key` / `seqs_key` / `kv_patch_options`, deployment views.
- Produces:
  - `PlanAction(type: "sleep" | "stop", deployment_id: int)` (pydantic).
  - `ActiveDep(id, name, state, priority, reserved, actual, can_sleep, sleep_frees)`.
  - `PlanOption(id, title, detail, new_required, frees, fits, actions, patch, model_ref)` with `.as_dict()`.
  - `base_model_name(ref) -> str`.
  - `build_plans(*, fit, cfg, context_key, seqs_key, kv_patches, estimate_with, active, variants, estimate_ref) -> list[PlanOption]`.
  - `Deployer.plans_for(prep, fit, replacing=None) -> list[dict]`, `Deployer.active_deps(exclude=None) -> list[ActiveDep]`.

On unified memory, vLLM sleep level 1 moves weights into CPU memory, which is the same pool. So sleeping a model frees only `reserved − weights − draft_weights − runtime` (roughly its KV cache and activations). Stopping it frees the whole reservation.

- [ ] **Step 1: Write the failing tests** — `tests/test_plans.py`

```python
from dgxmm.config import GIB
from dgxmm.memory.budget import compute_budget
from dgxmm.memory.estimate import MemoryEstimate
from dgxmm.memory.hostmem import HostMemory
from dgxmm.planner.fit import check_fit
from dgxmm.planner.plans import ActiveDep, base_model_name, build_plans


def est(total):
    return MemoryEstimate(total, 0, 0, 0, 0, 0, 0, "estimated")


def fit_needing(required_gib, free_gib, settings):
    total = 121 * GIB
    mem = HostMemory(total=total, available=int((free_gib + 12) * GIB), free=0, cached=0, swap_total=0, swap_free=0)
    return check_fit(est(int(required_gib * GIB)), compute_budget(mem, [], settings))


CFG = {"max_model_len": 65536, "max_num_seqs": 8, "kv_cache_dtype": "auto"}


def estimator(base_gib):
    """Pretend memory = weights 20 GiB + KV proportional to context × seqs (× 0.5 when fp8)."""
    def _est(patch):
        c = {**CFG, **patch}
        kv = (base_gib - 20) * (c["max_model_len"] / 65536) * (c["max_num_seqs"] / 8) * (0.5 if c["kv_cache_dtype"] == "fp8" else 1)
        return int((20 + kv) * GIB)
    return _est


def test_reductions_and_evictions(settings):
    fit = fit_needing(60, 40, settings)  # 20 GiB short
    active = [ActiveDep(1, "chat", "ready", 10, 30 * GIB, 30 * GIB, True, 12 * GIB),
              ActiveDep(2, "embed", "ready", 0, 25 * GIB, 25 * GIB, False, 0)]
    plans = build_plans(fit=fit, cfg=CFG, context_key="max_model_len", seqs_key="max_num_seqs",
                        kv_patches=[{"kv_cache_dtype": "fp8"}], estimate_with=estimator(60), active=active,
                        variants=["Qwen/Model-FP8"], estimate_ref=lambda ref: 35 * GIB)
    by = {p.id: p for p in plans}
    assert by["kv"].fits and by["kv"].patch == {"kv_cache_dtype": "fp8"} and by["kv"].new_required == 40 * GIB
    assert by["context"].patch == {"max_model_len": 32768} and by["context"].fits
    assert by["seqs"].patch == {"max_num_seqs": 4}
    assert by["variant:Qwen/Model-FP8"].model_ref == "Qwen/Model-FP8"
    stop = by["stop"]
    assert [a.deployment_id for a in stop.actions] == [2] and stop.fits  # lowest priority first, 25 GiB >= 20
    sleep = by["sleep"]
    assert [a.deployment_id for a in sleep.actions] == [1] and not sleep.fits  # sleeping frees only 12 GiB
    assert [p.fits for p in plans] == sorted([p.fits for p in plans], reverse=True)  # fitting options first


def test_no_plans_when_nothing_helps(settings):
    fit = fit_needing(200, 40, settings)
    plans = build_plans(fit=fit, cfg=CFG, context_key="max_model_len", seqs_key="max_num_seqs", kv_patches=[],
                        estimate_with=lambda p: 199 * GIB, active=[], variants=[], estimate_ref=lambda r: None)
    assert all(not p.fits for p in plans)


def test_base_model_name():
    assert base_model_name("nvidia/Qwen3.6-35B-A3B-NVFP4") == "qwen3.6-35b-a3b"
    assert base_model_name("Qwen/Qwen3.6-35B-A3B") == "qwen3.6-35b-a3b"
    assert base_model_name("unsloth/Qwen3-8B-GGUF") == "qwen3-8b"
    assert base_model_name("RedHatAI/Llama-3.1-8B-Instruct-FP8-dynamic") == "llama-3.1-8b-instruct"


async def test_preview_includes_plans_when_short(stack):
    from dgxmm.deployer.service import DeployRequest
    small = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}
    a = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="org/big-model", config=small), 1)
    stack.http.healthy.add(a["port"])
    await stack.deployer.wait_ready(a["id"], 5)
    p = await stack.deployer.preview("vllm", "org/big-model", small)
    assert not p["fit"]["fits"]
    stop = next(x for x in p["plans"] if x["id"] == "stop")
    assert stop["fits"] and stop["actions"] == [{"type": "stop", "deployment_id": a["id"]}]
```

- [ ] **Step 2: Run them to watch them fail**

`DGX$ .venv/bin/pytest -q tests/test_plans.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/planner/plans.py`**

```python
"""Ways to make a deployment fit (spec §5.4). Pure functions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from dgxmm.planner.fit import FitResult

GIB = 1024**3
_QUANT_SUFFIX = re.compile(
    r"([-_.](nvfp4|mxfp4|fp8([-_]dynamic)?|fp4|awq|gptq|int4|int8|w4a16|w8a8|bnb[-_]4bit|4bit|8bit|gguf|"
    r"q\d(_k)?(_[msl])?|iq\d\w*|bf16|fp16))+$", re.I)


class PlanAction(BaseModel):
    type: Literal["sleep", "stop"]
    deployment_id: int


@dataclass(frozen=True)
class ActiveDep:
    id: int
    name: str
    state: str
    priority: int
    reserved: int
    actual: int
    can_sleep: bool
    sleep_frees: int


@dataclass
class PlanOption:
    id: str
    title: str
    detail: str
    new_required: int
    frees: int
    fits: bool
    actions: list[PlanAction] = field(default_factory=list)
    patch: dict | None = None
    model_ref: str | None = None

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "detail": self.detail, "new_required": self.new_required,
                "frees": self.frees, "fits": self.fits, "actions": [a.model_dump() for a in self.actions],
                "patch": self.patch, "model_ref": self.model_ref}


def base_model_name(ref: str) -> str:
    name = ref.rstrip("/").split("/")[-1]
    name = re.sub(r"\.gguf$", "", name, flags=re.I)
    return _QUANT_SUFFIX.sub("", name).lower()


def _gib(n: int) -> str:
    return f"{n / GIB:.1f} GiB"


def build_plans(*, fit: FitResult, cfg: dict, context_key: str, seqs_key: str, kv_patches: list[dict],
                estimate_with: Callable[[dict], int | None], active: list[ActiveDep], variants: list[str],
                estimate_ref: Callable[[str], int | None]) -> list[PlanOption]:
    free = fit.free_for_new
    shortfall = fit.required - free
    out: list[PlanOption] = []

    def reduction(pid: str, title: str, patch: dict, detail: str) -> None:
        req = estimate_with(patch)
        if req is not None:
            out.append(PlanOption(pid, title, detail, req, 0, req <= free, patch=patch))

    for p in kv_patches:
        reduction("kv", "Use a smaller KV-cache dtype", p, f"Set {', '.join(f'{k}={v}' for k, v in p.items())}; quality impact is usually negligible.")

    ctx = int(cfg.get(context_key) or 0)
    c, chosen = ctx // 2, None
    while c >= 4096:
        patch = {context_key: c, **({"kv_tokens": None} if cfg.get("kv_tokens") else {})}
        req = estimate_with(patch)
        if req is not None and req <= free:
            chosen = (patch, req)
            break
        c //= 2
    if chosen:
        out.append(PlanOption("context", f"Reduce context to {chosen[0][context_key]:,} tokens",
                              f"From {ctx:,}. Long prompts beyond this will be rejected.", chosen[1], 0, True, patch=chosen[0]))

    seqs = int(cfg.get(seqs_key) or 0)
    s = seqs // 2
    while s >= 1:
        req = estimate_with({seqs_key: s})
        if req is not None and req <= free:
            out.append(PlanOption("seqs", f"Serve {s} concurrent sequences instead of {seqs}",
                                  "Lower throughput under load; single-user latency unchanged.", req, 0, True, patch={seqs_key: s}))
            break
        s //= 2

    for ref in variants:
        req = estimate_ref(ref)
        if req is not None:
            out.append(PlanOption(f"variant:{ref}", f"Use {ref} instead", "Smaller quantization already on disk.",
                                  req, 0, req <= free, model_ref=ref))

    ordered = sorted(active, key=lambda d: (d.priority, -d.reserved))
    sleepers = [d for d in ordered if d.can_sleep and d.state == "ready"]
    if sleepers:
        acts, freed = [], 0
        for d in sleepers:
            acts.append(PlanAction(type="sleep", deployment_id=d.id))
            freed += d.sleep_frees
            if freed >= shortfall:
                break
        out.append(PlanOption("sleep", "Put " + ", ".join(d.name for d in sleepers[:len(acts)]) + " to sleep",
                              f"Frees about {_gib(freed)} (weights stay in memory for a fast wake-up).",
                              fit.required, freed, freed >= shortfall, actions=acts))
    if ordered:
        acts, freed = [], 0
        for d in ordered:
            acts.append(PlanAction(type="stop", deployment_id=d.id))
            freed += d.actual if d.state == "sleeping" else max(d.reserved, d.actual)
            if freed >= shortfall:
                break
        out.append(PlanOption("stop", "Stop " + ", ".join(d.name for d in ordered[:len(acts)]),
                              f"Frees about {_gib(freed)}. Lowest priority first; can be restarted later.",
                              fit.required, freed, freed >= shortfall, actions=acts))
    out.sort(key=lambda p: not p.fits)  # stable: fitting options first, original order otherwise
    return out
```

- [ ] **Step 4: Wire the planner into the deployer** — add to `Deployer` in `dgxmm/deployer/service.py`

Add the imports `from dgxmm.planner.plans import ActiveDep, base_model_name, build_plans`, then add these methods:
```python
    def active_deps(self, exclude: int | None = None) -> list[ActiveDep]:
        out = []
        for v in self.list_views():
            if v["id"] == exclude or v["state"] not in ("ready", "sleeping", "unknown", "loading", "starting"):
                continue
            est = v["estimate"] or {}
            sleep_frees = max(v["reserved_bytes"] - est.get("weights", 0) - est.get("draft_weights", 0)
                              - est.get("runtime", 0), 0)
            out.append(ActiveDep(v["id"], v["name"], v["state"], v["priority"], v["reserved_bytes"],
                                 v["actual_bytes"] or 0, v["sleep_capable"] and v["state"] == "ready", sleep_frees))
        return out

    def plans_for(self, prep: Prepared, fit: FitResult, replacing: int | None = None) -> list[dict]:
        cfg = prep.cfg.model_dump(mode="json")
        name = prep.engine.name

        def est_with(patch: dict) -> int | None:
            try:
                return self.prepare(name, prep.artifact.ref, {**cfg, **patch}).estimate.total
            except DeployError:
                return None

        def est_ref(ref: str) -> int | None:
            try:
                return self.prepare(name, ref, cfg).estimate.total
            except DeployError:
                return None

        base = base_model_name(prep.artifact.ref)
        variants = [a.ref for a in self.inventory.list()
                    if a.ref != prep.artifact.ref and a.complete and a.kind == "llm" and prep.engine.supports(a)
                    and base_model_name(a.ref) == base and a.weights_bytes < prep.artifact.weights_bytes]
        plans = build_plans(fit=fit, cfg=cfg, context_key=prep.engine.context_key, seqs_key=prep.engine.seqs_key,
                            kv_patches=prep.engine.kv_patch_options(prep.cfg), estimate_with=est_with,
                            active=self.active_deps(exclude=replacing), variants=variants, estimate_ref=est_ref)
        return [p.as_dict() for p in plans]
```
In `preview`, after computing `fit`, add:
```python
        out["plans"] = [] if fit.fits else self.plans_for(prep, fit, replacing)
```

- [ ] **Step 5: Include plans in 409 responses and undo plan actions on failure** — replace `create` in `dgxmm/api/deployments.py` with:

```python
@router.post("", status_code=201)
async def create(body: DeployIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    req = DeployRequest(**body.model_dump(exclude={"actions"}))
    applied = []
    try:
        if body.actions:
            applied = await ctx.deployer.apply_actions(body.actions, p.user.id)
        view = await ctx.deployer.deploy(req, p.user.id)
    except InsufficientMemory as e:
        await ctx.deployer.undo_actions(applied)
        try:
            plans = (await ctx.deployer.preview(req.engine, req.model_ref, req.config))["plans"]
        except DeployError:
            plans = []
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict(), plans=plans)
    except DeployError as e:
        await ctx.deployer.undo_actions(applied)
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "deploy", req.name, {"model": req.model_ref, "engine": req.engine, "actions": body.actions})
    return view
```
`InsufficientMemory` must be caught before `DeployError` because it is a subclass. `apply_actions` and `undo_actions` arrive in Task 17; Tasks 16 and 17 are executed back to back, so run the API tests after Task 17.

- [ ] **Step 6: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_plans.py tests/test_deployer.py`. Expected: all pass (`tests/test_api_deployments.py` is re-run after Task 17).

---

### Task 17: Sleep, wake, apply plan actions, switch, and reconfigure with rollback

**Files:**
- Modify: `dgxmm/deployer/service.py`
- Modify: `tests/fakes.py` (add `FakeDockerClient.exit_on_start` and `FakeEngines.all_healthy`)
- Test: `tests/test_switch.py`

**Interfaces:**
- Consumes: `PlanAction`, the engine `sleep` / `wake` methods, `InsufficientMemory`.
- Produces `Deployer` methods:
  - `async sleep(dep_id, user_id=None) -> dict`
  - `async wake(dep_id, user_id=None) -> dict`
  - `async apply_actions(actions, user_id=None) -> list[tuple[str, int]]`
  - `async undo_actions(applied)`
  - `async switch(old_id, req, actions, user_id) -> dict`
  - `async reconfigure(dep_id, new_config, model_ref=None, actions=(), user_id=None) -> dict`

- [ ] **Step 1: Extend the fakes** — in `tests/fakes.py`

In `FakeDockerClient.__init__` add `self.exit_on_start: dict[str, int] = {}`.

In `FakeContainers.run`, just before `self.client._containers[cid] = c`, add:
```python
        if name in self.client.exit_on_start:
            c.status, c.exit_code = "exited", self.client.exit_on_start[name]
```

In `FakeEngines.__init__` add `self.all_healthy = False`. In `handler`, replace both `port in self.healthy` checks with `(self.all_healthy or port in self.healthy)`.

- [ ] **Step 2: Write the failing tests** — `tests/test_switch.py`

```python
import pytest

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployError, DeployRequest, InsufficientMemory

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name, ref="Qwen/Qwen3-Tiny", alias=None, **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg}, alias=alias)


async def ready(stack, name, **kw):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(req(name, **kw), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return d


async def test_sleep_and_wake(stack):
    d = await ready(stack, "a")
    assert (await stack.deployer.sleep(d["id"]))["state"] == "sleeping"
    assert ("POST", d["port"], "/sleep") in stack.http.calls
    assert (await stack.deployer.wake(d["id"]))["state"] == "ready"


async def test_wake_refused_when_memory_short(stack):
    d = await ready(stack, "a")
    await stack.deployer.sleep(d["id"])
    stack.probe.available = 9 * GIB
    with pytest.raises(InsufficientMemory):
        await stack.deployer.wake(d["id"])


async def test_sleep_refused_when_disabled(stack):
    d = await ready(stack, "a", enable_sleep_mode=False)
    with pytest.raises(DeployError) as e:
        await stack.deployer.sleep(d["id"])
    assert e.value.code == "sleep_unsupported"


async def test_switch_moves_alias_and_sleeps_old(stack):
    a = await ready(stack, "a", alias="chat")
    b = await stack.deployer.switch(a["id"], req("b"), [], 1)
    assert b["state"] == "ready" and b["alias"] == "chat"
    old = stack.deployer.view(a["id"])
    assert old["state"] == "sleeping" and old["alias"] is None


async def test_switch_failure_rolls_back(stack):
    a = await ready(stack, "a", alias="chat")
    stack.sdk.exit_on_start["dgxmm-b"] = 137
    with pytest.raises(DeployError):
        await stack.deployer.switch(a["id"], req("b"), [], 1)
    old = stack.deployer.view(a["id"])
    assert old["state"] == "ready" and old["alias"] == "chat"
    names = {v["name"]: v["state"] for v in stack.deployer.list_views()}
    assert names["b"] in ("failed", "stopped")


async def test_reconfigure_success_and_rollback(stack):
    a = await ready(stack, "a")
    out = await stack.deployer.reconfigure(a["id"], {**SMALL, "max_model_len": 2048})
    assert out["id"] == a["id"] and out["state"] == "ready" and out["config"]["max_model_len"] == 2048
    stack.sdk.exit_on_start["dgxmm-a"] = 1
    original = {**out["config"]}
    with pytest.raises(DeployError):
        await stack.deployer.reconfigure(a["id"], {**SMALL, "max_model_len": 1024})
    stack.sdk.exit_on_start.clear()
    # rollback restarts the previous config; the fake container now starts fine
    st = await stack.deployer.wait_ready(a["id"], 5)
    v = stack.deployer.view(a["id"])
    assert v["config"]["max_model_len"] == original["max_model_len"]
    assert st in ("ready", "failed")


async def test_apply_actions_undo_on_partial_failure(stack):
    a = await ready(stack, "a")
    with pytest.raises(DeployError):
        await stack.deployer.apply_actions([{"type": "sleep", "deployment_id": a["id"]},
                                            {"type": "stop", "deployment_id": 999}], 1)
    assert stack.deployer.view(a["id"])["state"] == "ready"
```

Run: `DGX$ .venv/bin/pytest -q tests/test_switch.py`. Expected: FAIL (`AttributeError: 'Deployer' object has no attribute 'sleep'`).

Note on `test_reconfigure_success_and_rollback`: the rollback restarts the old config while `exit_on_start` still makes `dgxmm-a` exit, so the rollback attempt may itself fail. The test then clears the flag and only asserts that the *config* was restored and the deployment did not get stuck in a transitional state. That is the real guarantee: we never leave the row with the broken config.

- [ ] **Step 3: Implement the methods** — add to `Deployer` (`dgxmm/deployer/service.py`)

Add imports `from dgxmm.planner.plans import PlanAction` and `from dgxmm.memory.estimate import MemoryEstimate`, then:
```python
    def _stored_estimate(self, dep: Deployment) -> MemoryEstimate:
        if dep.estimate:
            return MemoryEstimate.from_dict(dep.estimate)
        return MemoryEstimate(dep.reserved_bytes, 0, 0, 0, 0, 0, 0, "low")

    async def sleep(self, dep_id: int, user_id: int | None = None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state != "ready":
                raise DeployError("not_ready", f"deployment is {dep.state}", 409)
            engine = self.engines[dep.engine]
            cfg = engine.validate_config(dep.config)
            if not (engine.supports_sleep and engine.sleep_enabled(cfg)):
                raise DeployError("sleep_unsupported", "this deployment was not started with sleep mode", 409)
            try:
                await engine.sleep(self.http, self.base_url(dep.name, dep.port))
            except httpx.HTTPError as e:
                raise DeployError("engine_error", f"sleep failed: {e}", 502)
            return self._transition(dep_id, "sleeping", message="sleeping (level 1: KV freed, weights offloaded)")

    async def wake(self, dep_id: int, user_id: int | None = None) -> dict:
        async with self._lock(dep_id):
            dep = self._require(dep_id)
            if dep.state != "sleeping":
                raise DeployError("not_sleeping", f"deployment is {dep.state}", 409)
            engine = self.engines[dep.engine]
            async with self.mem_lock:
                budget = await self.budget.refresh()
                actual = next((u.actual for u in budget.deployments if u.deployment_id == dep_id), 0)
                need = max(dep.reserved_bytes - actual, 0)
                if need > budget.free_for_new:
                    raise InsufficientMemory(FitResult(False, need, budget.free_for_new, budget.free_for_new - need,
                                                       self._stored_estimate(dep), budget))
                try:
                    await engine.wake(self.http, self.base_url(dep.name, dep.port))
                except httpx.HTTPError as e:
                    raise DeployError("engine_error", f"wake failed: {e}", 502)
                return self._transition(dep_id, "ready", message="woken up")

    async def apply_actions(self, actions, user_id: int | None = None) -> list[tuple[str, int]]:
        applied: list[tuple[str, int]] = []
        try:
            for raw in actions:
                a = raw if isinstance(raw, PlanAction) else PlanAction.model_validate(raw)
                if a.type == "sleep":
                    await self.sleep(a.deployment_id, user_id)
                else:
                    await self.stop(a.deployment_id, user_id, reason="plan")
                applied.append((a.type, a.deployment_id))
        except Exception:
            await self.undo_actions(applied)
            raise
        return applied

    async def undo_actions(self, applied: list[tuple[str, int]]) -> None:
        for typ, dep_id in reversed(applied):
            try:
                if typ == "sleep":
                    await self.wake(dep_id)
                else:
                    await self.restart(dep_id)
            except Exception:
                log.exception("could not undo %s of deployment %s", typ, dep_id)

    def _set_alias(self, dep_id: int, alias: str | None) -> None:
        with self.db.session() as s:
            s.get(Deployment, dep_id).alias = alias
        self.events.publish("deployment", self.view(dep_id))

    def _can_sleep(self, dep: Deployment) -> bool:
        engine = self.engines.get(dep.engine)
        return bool(dep.state == "ready" and engine and engine.supports_sleep and dep.config.get("enable_sleep_mode"))

    async def switch(self, old_id: int, req: DeployRequest, actions, user_id: int | None) -> dict:
        old = self._require(old_id)
        acts = [a if isinstance(a, PlanAction) else PlanAction.model_validate(a) for a in actions]
        if not any(a.deployment_id == old_id for a in acts) and old.state in ("ready", "sleeping", "unknown"):
            if old.state != "sleeping":
                acts.insert(0, PlanAction(type="sleep" if self._can_sleep(old) else "stop", deployment_id=old_id))
        alias = old.alias
        applied = await self.apply_actions(acts, user_id)
        new_id = None
        try:
            if alias:
                self._set_alias(old_id, None)
            new = await self.deploy(req.model_copy(update={"alias": req.alias or alias}), user_id)
            new_id = new["id"]
            state = await self.wait_ready(new_id, self.settings.startup_timeout_s + 30)
            if state != "ready":
                raise DeployError("switch_failed", f"new deployment ended in state {state}; rolled back", 502)
            return self.view(new_id)
        except Exception:
            if new_id is not None:
                try:
                    await self.stop(new_id, user_id, reason="switch rollback")
                except Exception:
                    log.exception("rollback: could not stop %s", new_id)
            if alias:
                self._set_alias(old_id, alias)
            await self.undo_actions(applied)
            raise

    async def reconfigure(self, dep_id: int, new_config: dict, model_ref: str | None = None, actions=(),
                          user_id: int | None = None) -> dict:
        dep = self._require(dep_id)
        old_cfg, old_ref = dict(dep.config), dep.model_ref
        was_active = dep.state in ("ready", "sleeping", "unknown", "starting", "loading")
        prep = self.prepare(dep.engine, model_ref or old_ref, new_config)
        if not actions:
            budget = await self.budget.refresh()
            fit = check_fit(prep.estimate, budget, extra_free=self._replacing_credit(dep_id))
            if not fit.fits:
                raise InsufficientMemory(fit)
        applied = await self.apply_actions(actions, user_id)
        try:
            if was_active:
                await self.stop(dep_id, user_id, reason="reconfigure")
            await self.restart(dep_id, user_id, config_override=new_config, model_ref=model_ref)
            state = await self.wait_ready(dep_id, self.settings.startup_timeout_s + 30)
            if state != "ready":
                raise DeployError("reconfigure_failed", f"new configuration ended in state {state}; rolled back", 502)
            return self.view(dep_id)
        except Exception:
            now = self._get(dep_id)
            if now is not None and now.state not in ("stopped", "failed"):
                await self.stop(dep_id, user_id, reason="reconfigure rollback")
            with self.db.session() as s:  # always restore the old config, even if we cannot restart it
                row = s.get(Deployment, dep_id)
                row.config, row.model_ref = old_cfg, old_ref
            if was_active:
                try:
                    await self.restart(dep_id, user_id, config_override=old_cfg, model_ref=old_ref)
                except Exception:
                    log.exception("rollback restart of %s failed", dep_id)
            await self.undo_actions(applied)
            raise
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_switch.py tests/test_deployer.py`. Expected: all pass.

---

### Task 18: LiteLLM registration

**Files:**
- Create: `dgxmm/litellm.py`
- Modify: `dgxmm/context.py` (wiring in `build_context`)
- Test: `tests/test_litellm.py`

**Interfaces:**
- Consumes: `Deployer.list_views()`, `Deployer.state_hooks`, `Database`, `Deployment.litellm_status`.
- Produces:
  - `LiteLLMClient(base_url, key, http)` with `list_dgxmm()`, `add(model_name, api_base, served, dgxmm_id)`, `delete(model_id)`.
  - `LiteLLMSync(client | None, deployer, db, events)` with `desired()`, `async sync() -> bool`, `trigger()`, `async on_state(dep_id, frm, to)`, `async run()`.

- [ ] **Step 1: Write the failing tests** — `tests/test_litellm.py`

```python
import json
import uuid

import httpx

from dgxmm.deployer.service import DeployRequest
from dgxmm.litellm import LiteLLMClient, LiteLLMSync

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


class FakeLiteLLM:
    def __init__(self):
        self.models = [{"model_name": "hand-written", "litellm_params": {"model": "openai/x"}, "model_info": {"id": "keep-me"}}]
        self.down = False
        self.auth = []

    def handler(self, req: httpx.Request):
        self.auth.append(req.headers.get("authorization"))
        if self.down:
            raise httpx.ConnectError("down")
        if req.url.path == "/model/info":
            return httpx.Response(200, json={"data": self.models})
        body = json.loads(req.content)
        if req.url.path == "/model/new":
            self.models.append({**body, "model_info": {**body["model_info"], "id": str(uuid.uuid4())}})
            return httpx.Response(200, json={"ok": True})
        if req.url.path == "/model/delete":
            self.models = [m for m in self.models if m["model_info"]["id"] != body["id"]]
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def make(stack, fake):
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    return LiteLLMSync(LiteLLMClient("http://litellm:4000", "sk-test", http), stack.deployer, stack.db, stack.events)


async def ready(stack, name, alias=None):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=SMALL, alias=alias), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    return d


async def test_registers_served_name_and_alias_then_removes(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a", alias="chat")
    assert await sync.sync()
    mine = sorted(m["model_name"] for m in fake.models if m["model_info"].get("dgxmm_id"))
    assert mine == ["chat", "qwen3-tiny"]
    entry = next(m for m in fake.models if m["model_name"] == "chat")
    assert entry["litellm_params"] == {"model": "openai/qwen3-tiny", "api_base": f"http://engines.test:{d['port']}/v1", "api_key": "none"}
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"
    assert set(fake.auth) == {"Bearer sk-test"}
    await stack.deployer.stop(d["id"])
    assert await sync.sync()
    assert [m["model_name"] for m in fake.models] == ["hand-written"]  # never touches unmarked entries


async def test_idempotent_and_dedupes(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    await ready(stack, "a")
    await sync.sync()
    fake.models.append({**fake.models[-1], "model_info": {**fake.models[-1]["model_info"], "id": "dup"}})
    await sync.sync()
    await sync.sync()
    assert sum(1 for m in fake.models if m["model_name"] == "qwen3-tiny") == 1


async def test_litellm_down_marks_pending_and_recovers(stack):
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    d = await ready(stack, "a")
    fake.down = True
    assert not await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "pending"
    fake.down = False
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "synced"


async def test_no_key_means_disabled(stack):
    sync = LiteLLMSync(None, stack.deployer, stack.db, stack.events)
    d = await ready(stack, "a")
    assert await sync.sync()
    assert stack.deployer.view(d["id"])["litellm_status"] == "disabled"
```

Run: `DGX$ .venv/bin/pytest -q tests/test_litellm.py`. Expected: FAIL (ImportError).

The test above expects `api_base` to use `engines.test`. `LiteLLMSync` builds `api_base` from `deployer.base_url(name, port) + "/v1"`. In production `engine_base_host` is `None`, so that resolves to `http://dgxmm-<name>:<port>/v1`, which LiteLLM can reach on the `gateway` network.

- [ ] **Step 2: Implement `dgxmm/litellm.py`**

```python
"""Keep LiteLLM's DB-backed model list in sync with ready deployments (spec §6)."""
from __future__ import annotations

import asyncio
import logging

import httpx
from sqlalchemy import select

from dgxmm.db import Database, Deployment

log = logging.getLogger(__name__)


class LiteLLMClient:
    def __init__(self, base_url: str, key: str, http: httpx.AsyncClient):
        self.base, self.key, self.http = base_url.rstrip("/"), key, http

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    async def list_dgxmm(self) -> list[dict]:
        r = await self.http.get(f"{self.base}/model/info", headers=self._h(), timeout=10)
        r.raise_for_status()
        return [m for m in r.json().get("data", []) if (m.get("model_info") or {}).get("dgxmm_id")]

    async def add(self, model_name: str, api_base: str, served: str, dgxmm_id: str) -> None:
        body = {"model_name": model_name,
                "litellm_params": {"model": f"openai/{served}", "api_base": api_base, "api_key": "none"},
                "model_info": {"dgxmm_id": dgxmm_id}}
        r = await self.http.post(f"{self.base}/model/new", headers=self._h(), json=body, timeout=15)
        r.raise_for_status()

    async def delete(self, model_id: str) -> None:
        r = await self.http.post(f"{self.base}/model/delete", headers=self._h(), json={"id": model_id}, timeout=15)
        r.raise_for_status()


class LiteLLMSync:
    def __init__(self, client: LiteLLMClient | None, deployer, db: Database, events):
        self.client, self.deployer, self.db, self.events = client, deployer, db, events
        self._wake = asyncio.Event()

    def desired(self) -> dict[tuple[str, str], tuple[str, str]]:
        want: dict[tuple[str, str], tuple[str, str]] = {}
        for v in self.deployer.list_views():
            if v["state"] != "ready" or not v["served_name"] or not v["port"]:
                continue
            api_base = self.deployer.base_url(v["name"], v["port"]) + "/v1"
            for name in filter(None, [v["served_name"], v["alias"]]):
                want[(str(v["id"]), name)] = (api_base, v["served_name"])
        return want

    def _set_status(self, ready_status: str) -> None:
        changed = []
        with self.db.session() as s:
            for d in s.scalars(select(Deployment)):
                status = ready_status if d.state == "ready" else ("disabled" if ready_status == "disabled" else "none")
                if d.litellm_status != status:
                    d.litellm_status = status
                    changed.append(d.id)
        for dep_id in changed:
            self.events.publish("deployment", self.deployer.view(dep_id))

    async def sync(self) -> bool:
        if self.client is None:
            self._set_status("disabled")
            return True
        try:
            have: dict[tuple[str, str], list[tuple[str, str]]] = {}
            for m in await self.client.list_dgxmm():
                key = (str(m["model_info"]["dgxmm_id"]), m["model_name"])
                have.setdefault(key, []).append((m["model_info"]["id"], (m.get("litellm_params") or {}).get("api_base", "")))
            want = self.desired()
            for key, entries in have.items():
                target = want.get(key)
                keep = None
                if target:
                    keep = next((e for e in entries if e[1] == target[0]), None)
                for model_id, _ in entries:
                    if keep is None or model_id != keep[0]:
                        await self.client.delete(model_id)
                if target and keep is None:
                    have[key] = []
            for key, (api_base, served) in want.items():
                if not have.get(key):
                    await self.client.add(key[1], api_base, served, key[0])
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
            log.warning("LiteLLM sync failed: %s", e)
            self._set_status("pending")
            return False
        self._set_status("synced")
        return True

    def trigger(self) -> None:
        self._wake.set()

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        if "ready" in (frm, to):
            self.trigger()

    async def run(self) -> None:
        failure_delay = 5.0
        while True:
            ok = await self.sync()
            if ok:
                delay, failure_delay = 60.0, 5.0
            else:
                delay, failure_delay = failure_delay, min(failure_delay * 2, 600.0)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
```

- [ ] **Step 3: Wire it into `build_context`** (add before `return ctx`)

```python
    from dgxmm.litellm import LiteLLMClient, LiteLLMSync
    key = settings.read_secret(settings.litellm_key_file)
    if not key:
        log.warning("no LiteLLM master key at %s; LiteLLM registration disabled", settings.litellm_key_file)
    litellm = LiteLLMSync(LiteLLMClient(settings.litellm_url, key, http) if key else None, deployer, db, events)
    deployer.state_hooks.append(litellm.on_state)
    ctx.litellm = litellm
    ctx.background.append(litellm.run)
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_litellm.py`. Expected: `4 passed`.

---

### Task 19: Memory watchdog

**Files:**
- Create: `dgxmm/watchdog.py`
- Modify: `dgxmm/context.py` (wiring)
- Test: `tests/test_watchdog.py`

**Interfaces:**
- Consumes: `Budget`, `Deployer.list_views()`, `Deployer.stop`, `audit`, `EventBroker`.
- Produces: `Watchdog(settings, deployer, events, db, clock=time.monotonic)` with `async check(budget) -> str` (returns `"ok" | "warning" | "critical"`) and `state() -> dict`.

- [ ] **Step 1: Write the failing tests** — `tests/test_watchdog.py`

```python
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import AuditLog
from dgxmm.deployer.service import DeployRequest
from dgxmm.memory.budget import compute_budget
from dgxmm.memory.hostmem import HostMemory
from dgxmm.watchdog import Watchdog

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def budget(settings, avail_gib):
    return compute_budget(HostMemory(121 * GIB, int(avail_gib * GIB), 0, 0, 0, 0), [], settings)


async def test_levels_and_events(stack):
    q = stack.events.subscribe()
    w = Watchdog(stack.settings, stack.deployer, stack.events, stack.db)
    assert await w.check(budget(stack.settings, 50)) == "ok"
    assert await w.check(budget(stack.settings, 11)) == "warning"
    assert await w.check(budget(stack.settings, 5)) == "critical"
    topics = []
    while not q.empty():
        topics.append(q.get_nowait())
    alerts = [d["level"] for t, d in topics if t == "alert"]
    assert alerts == ["warning", "critical"]
    assert w.state()["level"] == "critical"


async def test_auto_protect_stops_lowest_priority_once_per_minute(stack):
    stack.http.all_healthy = True
    lo = await stack.deployer.deploy(DeployRequest(name="lo", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, priority=0), 1)
    hi = await stack.deployer.deploy(DeployRequest(name="hi", engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, priority=50), 1)
    for d in (lo, hi):
        await stack.deployer.wait_ready(d["id"], 5)
    off = Watchdog(stack.settings, stack.deployer, stack.events, stack.db)
    await off.check(budget(stack.settings, 3))
    assert stack.deployer.view(lo["id"])["state"] == "ready"
    stack.settings.auto_protect = True
    now = [1000.0]
    on = Watchdog(stack.settings, stack.deployer, stack.events, stack.db, clock=lambda: now[0])
    await on.check(budget(stack.settings, 3))
    assert stack.deployer.view(lo["id"])["state"] == "stopped"
    await on.check(budget(stack.settings, 3))
    assert stack.deployer.view(hi["id"])["state"] == "ready"  # rate-limited
    now[0] += 61
    await on.check(budget(stack.settings, 3))
    assert stack.deployer.view(hi["id"])["state"] == "stopped"
    with stack.db.session() as s:
        assert [a.target for a in s.scalars(select(AuditLog).where(AuditLog.action == "watchdog_stop"))] == ["lo", "hi"]
```

Run: `DGX$ .venv/bin/pytest -q tests/test_watchdog.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/watchdog.py`**

```python
"""Watch MemAvailable; alert and optionally stop the lowest-priority deployment (spec §4.5)."""
from __future__ import annotations

import logging
import time
from typing import Callable

from dgxmm.audit import audit
from dgxmm.db import utcnow

log = logging.getLogger(__name__)
GIB = 1024**3


class Watchdog:
    def __init__(self, settings, deployer, events, db, clock: Callable[[], float] = time.monotonic):
        self.settings, self.deployer, self.events, self.db, self.clock = settings, deployer, events, db, clock
        self.level, self.message, self.since, self.action = "ok", "", None, None
        self._last_action = float("-inf")

    def state(self) -> dict:
        return {"level": self.level, "message": self.message, "since": self.since, "action": self.action}

    async def check(self, budget) -> str:
        margin = self.settings.margin_bytes
        avail = budget.available
        if avail < margin:
            level = "critical"
            msg = f"only {avail / GIB:.1f} GiB available, below the {margin / GIB:.0f} GiB safety margin"
        elif avail < 1.5 * margin:
            level = "warning"
            msg = f"{avail / GIB:.1f} GiB available, approaching the {margin / GIB:.0f} GiB safety margin"
        else:
            level, msg = "ok", ""
        if level != self.level:
            self.level, self.message, self.since, self.action = level, msg, utcnow().isoformat(), None
            self.events.publish("alert", self.state())
            log.warning("memory watchdog: %s %s", level, msg) if level != "ok" else log.info("memory watchdog: ok")
        else:
            self.message = msg
        if level == "critical" and self.settings.auto_protect and self.clock() - self._last_action >= 60:
            victim = self._victim()
            if victim:
                self._last_action = self.clock()
                try:
                    await self.deployer.stop(victim["id"], None, reason="watchdog: memory critical")
                    self.action = f"stopped {victim['name']} (priority {victim['priority']})"
                    audit(self.db, None, "watchdog_stop", victim["name"], {"available": avail})
                except Exception as e:
                    self.action = f"failed to stop {victim['name']}: {e}"
                    log.exception("watchdog stop failed")
                self.events.publish("alert", self.state())
        return level

    def _victim(self) -> dict | None:
        cands = [v for v in self.deployer.list_views() if v["state"] in ("ready", "sleeping", "unknown")]
        cands.sort(key=lambda v: (v["priority"], -v["reserved_bytes"]))
        return cands[0] if cands else None
```

- [ ] **Step 3: Wire it into `build_context`** (add before `return ctx`)

```python
    from dgxmm.watchdog import Watchdog
    ctx.watchdog = Watchdog(settings, deployer, events, db)
    ctx.budget_listeners.append(ctx.watchdog.check)
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_watchdog.py`. Expected: `2 passed`.

---

### Task 20: Adopt existing vLLM containers

**Files:**
- Create: `dgxmm/deployer/adopt.py`
- Test: `tests/test_adopt.py`

**Interfaces:**
- Consumes: `DockerController.get`, `DockerController.list_all`, `Deployer.prepare`, `Deployment`, `DeploymentEvent`, `validate_extra_args`, `VLLM_EXTRA_FLAGS`.
- Produces: `AdoptError(DeployError)`, `parse_vllm_command(cmd) -> (model_ref, config, ignored)`, `list_adoptable(deployer) -> list[dict]`, `async adopt(deployer, container, name, user_id) -> dict`.

- [ ] **Step 1: Write the failing tests** — `tests/test_adopt.py`

```python
import json

import pytest

from dgxmm.deployer.adopt import adopt, list_adoptable, parse_vllm_command
from dgxmm.deployer.service import DeployError

# A real vLLM command line (a 27B NVFP4 model with a DFlash draft), 2026-10-15
DMM = ["--model", "unsloth/Qwen3.8-27B-NVFP4", "--served-model-name", "qwen3.8-27b", "--host", "0.0.0.0", "--port", "8000",
       "--max-model-len", "256000", "--gpu-memory-utilization", "0.50", "--trust-remote-code", "--enable-prefix-caching",
       "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
       "--enable-chunked-prefill", "--kv-cache-dtype", "fp8", "--max-num-batched-tokens", "16384", "--max-num-seqs", "8",
       "--speculative-config", json.dumps({"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7})]


def test_parse_real_command():
    ref, cfg, ignored = parse_vllm_command(DMM)
    assert ref == "unsloth/Qwen3.8-27B-NVFP4"
    assert cfg == {"served_model_name": "qwen3.8-27b", "max_model_len": 256000, "trust_remote_code": True,
                   "enable_prefix_caching": True, "enable_auto_tool_choice": True, "tool_call_parser": "qwen3_coder",
                   "reasoning_parser": "qwen3", "enable_chunked_prefill": True, "kv_cache_dtype": "fp8",
                   "max_num_batched_tokens": 16384, "max_num_seqs": 8, "enable_sleep_mode": False,
                   "speculative": {"method": "dflash", "model": "incoai/Qwen3.8-27B-DFlash2", "num_speculative_tokens": 7}}
    assert set(ignored) == {"--host", "--port", "--gpu-memory-utilization"}


def test_parse_variants():
    ref, cfg, _ = parse_vllm_command(["vllm", "serve", "nvidia/X", "--safetensors-load-strategy=prefetch", "--enable-sleep-mode"])
    assert ref == "nvidia/X" and cfg["safetensors_load_strategy"] == "prefetch" and cfg["enable_sleep_mode"] is True
    _, cfg, _ = parse_vllm_command(["--model", "a/b", "--enforce-eager", "--no-enable-prefix-caching"])
    assert cfg["extra_args"] == ["--enforce-eager"] and cfg["enable_prefix_caching"] is False
    with pytest.raises(DeployError) as e:
        parse_vllm_command(["--model", "a/b", "--tensor-parallel-size", "2", "--weird-flag"])
    assert e.value.code == "unsupported_flags" and "--weird-flag" in e.value.message


async def test_adopt_creates_stopped_row_and_leaves_container(stack):
    c = stack.sdk.add_unmanaged("vllm-external", image="vllm/vllm-openai:pinned-2026-08-27",
                                command=["--model", "Qwen/Qwen3-Tiny", "--max-model-len", "4096", "--enable-sleep-mode"],
                                status="created")
    stack.sdk.add_unmanaged("litellm-proxy", image="ghcr.io/berriai/litellm:main-latest")
    assert [a["name"] for a in list_adoptable(stack.deployer)] == ["vllm-external"]
    v = await adopt(stack.deployer, "vllm-external", "qwen-tiny", 1)
    assert v["state"] == "stopped" and v["adopted_from"] == "vllm-external" and v["config"]["max_model_len"] == 4096
    assert stack.sdk.container(c.id).status == "created"  # untouched
    assert list_adoptable(stack.deployer)[0]["adopted_as"] == "qwen-tiny"
    with pytest.raises(DeployError):
        await adopt(stack.deployer, "litellm-proxy", "x", 1)
```

Run: `DGX$ .venv/bin/pytest -q tests/test_adopt.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/deployer/adopt.py`**

```python
"""Turn an existing hand-made vLLM container into a DGXMM deployment (spec §5.6)."""
from __future__ import annotations

import asyncio
import json

from sqlalchemy import select

from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer.service import NAME_PATTERN, DeployError
from dgxmm.engines.argguard import VLLM_EXTRA_FLAGS

VALUE_FLAGS = {
    "--served-model-name": ("served_model_name", str), "--max-model-len": ("max_model_len", int),
    "--max-num-seqs": ("max_num_seqs", int), "--max-num-batched-tokens": ("max_num_batched_tokens", int),
    "--kv-cache-dtype": ("kv_cache_dtype", str), "--tool-call-parser": ("tool_call_parser", str),
    "--reasoning-parser": ("reasoning_parser", str), "--safetensors-load-strategy": ("safetensors_load_strategy", str),
}
BOOL_FLAGS = {
    "--enable-prefix-caching": ("enable_prefix_caching", True), "--no-enable-prefix-caching": ("enable_prefix_caching", False),
    "--enable-chunked-prefill": ("enable_chunked_prefill", True), "--no-enable-chunked-prefill": ("enable_chunked_prefill", False),
    "--enable-auto-tool-choice": ("enable_auto_tool_choice", True), "--enable-sleep-mode": ("enable_sleep_mode", True),
    "--trust-remote-code": ("trust_remote_code", True),
}
MANAGED = {"--host", "--port", "--gpu-memory-utilization", "--revision"}


class AdoptError(DeployError):
    pass


def parse_vllm_command(cmd: list[str]) -> tuple[str, dict, list[str]]:
    toks = list(cmd)
    if toks[:2] == ["vllm", "serve"]:
        toks = toks[2:]
    model = None
    if toks and not toks[0].startswith("-"):
        model, toks = toks[0], toks[1:]
    cfg: dict = {"enable_sleep_mode": False}
    ignored, extra, unsupported = [], [], []
    i = 0
    while i < len(toks):
        flag, eq, val = toks[i].partition("=")
        takes_value = flag in VALUE_FLAGS or flag in MANAGED or flag in ("--model", "--speculative-config") \
            or (flag in VLLM_EXTRA_FLAGS and VLLM_EXTRA_FLAGS[flag] is not None)
        if takes_value and not eq:
            val = toks[i + 1] if i + 1 < len(toks) else ""
            i += 2
        else:
            i += 1
        if flag == "--model":
            model = val
        elif flag in MANAGED:
            ignored.append(flag)
        elif flag in VALUE_FLAGS:
            key, typ = VALUE_FLAGS[flag]
            cfg[key] = typ(val)
        elif flag in BOOL_FLAGS:
            key, value = BOOL_FLAGS[flag]
            cfg[key] = value
        elif flag == "--speculative-config":
            cfg["speculative"] = json.loads(val)
        elif flag in VLLM_EXTRA_FLAGS:
            extra += [flag] + ([val] if VLLM_EXTRA_FLAGS[flag] is not None else [])
        else:
            unsupported.append(flag)
    if unsupported:
        raise AdoptError("unsupported_flags", f"cannot adopt: unsupported flags {', '.join(unsupported)}", 422)
    if not model:
        raise AdoptError("no_model", "could not find --model in the container command", 422)
    if extra:
        cfg["extra_args"] = extra
    return model, cfg, ignored


def list_adoptable(deployer) -> list[dict]:
    with deployer.db.session() as s:
        adopted = {d.adopted_from: d.name for d in s.scalars(select(Deployment)) if d.adopted_from}
    out = []
    for c in deployer.docker.list_all():
        if c.managed or "vllm" not in c.image:
            continue
        out.append({**c.as_dict(), "adopted_as": adopted.get(c.name)})
    return sorted(out, key=lambda c: c["name"])


async def adopt(deployer, container: str, name: str, user_id: int | None) -> dict:
    import re
    if not re.fullmatch(NAME_PATTERN, name):
        raise AdoptError("invalid_name", "deployment name must be lowercase letters, digits and dashes", 422)
    info = await asyncio.to_thread(deployer.docker.get, container)
    if info is None:
        raise AdoptError("not_found", f"container {container} not found", 404)
    if info.managed:
        raise AdoptError("already_managed", f"{info.name} is already managed by dgxmm", 409)
    if "vllm" not in info.image:
        raise AdoptError("unsupported_image", f"only vLLM containers can be adopted (image {info.image})", 422)
    ref, cfg, _ = parse_vllm_command(info.command)
    prep = deployer.prepare("vllm", ref, cfg)
    with deployer.db.session() as s:
        if s.scalar(select(Deployment).where(Deployment.name == name)):
            raise AdoptError("name_in_use", f"a deployment named {name} already exists", 409)
        row = Deployment(name=name, engine="vllm", model_ref=ref, config=prep.cfg.model_dump(mode="json"),
                         state="stopped", reserved_bytes=prep.estimate.total, estimate=prep.estimate.as_dict(),
                         adopted_from=info.name, created_by=user_id)
        s.add(row)
        s.flush()
        s.add(DeploymentEvent(deployment_id=row.id, from_state="stopped", to_state="stopped",
                              message=f"adopted settings from container {info.name}; original container left untouched"))
        dep_id = row.id
    view = deployer.view(dep_id)
    deployer.events.publish("deployment", view)
    return view
```

- [ ] **Step 3: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_adopt.py`. Expected: `3 passed`.

---

### Task 21: M2 API routes and UI (sleep/wake/switch/reconfigure, plans, adopt, audit)

**Files:**
- Create: `dgxmm/api/audit.py`, `dgxmm/api/containers.py`
- Modify: `dgxmm/api/deployments.py`, `dgxmm/app.py` (`ROUTERS`)
- Create: `frontend/src/components/SwitchDialog.tsx`, `frontend/src/pages/Audit.tsx`
- Modify: `frontend/src/pages/Dashboard.tsx`, `frontend/src/pages/DeploymentDetail.tsx`, `frontend/src/main.tsx`
- Test: `tests/test_api_switch.py`

**Interfaces:**
- Produces routes:
  - `POST /api/deployments/{id}/sleep`, `/wake`
  - `/switch` (body `{request: DeployRequest, actions: [...]}`)
  - `/reconfigure` (body `{config, model_ref?, actions}`)
  - `GET /api/audit?limit=`
  - `GET /api/containers/adoptable`, `POST /api/containers/{name}/adopt` (body `{name}`)
- Produces `<SwitchDialog mode="switch" | "reconfigure" deployment initialPatch? onClose>`.

- [ ] **Step 1: Write the failing tests** — `tests/test_api_switch.py`

```python
from tests.test_api_deployments import SMALL, admin


def deploy(c, h, name, **extra):
    c.stack.http.all_healthy = True
    r = c.post("/dgxmm/api/deployments", headers=h,
               json={"name": name, "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL, **extra})
    assert r.status_code == 201, r.text
    return r.json()


def wait(c, dep_id, want):
    import time
    for _ in range(200):
        if c.get(f"/dgxmm/api/deployments/{dep_id}").json()["state"] == want:
            return True
        time.sleep(0.02)
    return False


def test_sleep_wake_switch_reconfigure_audit(api_stack):
    h = admin(api_stack)
    a = deploy(api_stack, h, "a", alias="chat")
    assert wait(api_stack, a["id"], "ready")
    assert api_stack.post(f"/dgxmm/api/deployments/{a['id']}/sleep", headers=h).json()["state"] == "sleeping"
    assert api_stack.post(f"/dgxmm/api/deployments/{a['id']}/wake", headers=h).json()["state"] == "ready"
    r = api_stack.post(f"/dgxmm/api/deployments/{a['id']}/switch", headers=h,
                       json={"request": {"name": "b", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL},
                             "actions": []})
    assert r.status_code == 200, r.text
    assert r.json()["alias"] == "chat"
    r = api_stack.post(f"/dgxmm/api/deployments/{r.json()['id']}/reconfigure", headers=h,
                       json={"config": {**SMALL, "max_model_len": 2048}, "actions": []})
    assert r.status_code == 200 and r.json()["config"]["max_model_len"] == 2048
    actions = [e["action"] for e in api_stack.get("/dgxmm/api/audit").json()]
    assert {"deploy", "sleep", "wake", "switch", "reconfigure", "login"} <= set(actions)


def test_adopt_routes(api_stack):
    h = admin(api_stack)
    api_stack.stack.sdk.add_unmanaged("vllm-external", command=["--model", "Qwen/Qwen3-Tiny", "--max-model-len", "4096"])
    rows = api_stack.get("/dgxmm/api/containers/adoptable").json()
    assert [r["name"] for r in rows] == ["vllm-external"]
    r = api_stack.post("/dgxmm/api/containers/vllm-external/adopt", headers=h, json={"name": "qwen-tiny"})
    assert r.status_code == 201 and r.json()["state"] == "stopped"


def test_audit_requires_admin(api_stack):
    admin(api_stack)
    api_stack.post("/dgxmm/api/auth/logout", headers={"X-CSRF-Token": api_stack.get("/dgxmm/api/auth/me").json()["csrf_token"]})
    api_stack.post("/dgxmm/api/auth/login", json={"username": "vic", "password": "viewer-password-1"})
    assert api_stack.get("/dgxmm/api/audit").status_code == 403
```

Run: `DGX$ .venv/bin/pytest -q tests/test_api_switch.py`. Expected: FAIL (404s).

- [ ] **Step 2: Add the deployment routes** — append to `dgxmm/api/deployments.py`

```python
class SwitchIn(BaseModel):
    request: DeployRequest
    actions: list[dict] = []


class ReconfigureIn(BaseModel):
    config: dict
    model_ref: str | None = None
    actions: list[dict] = []


async def _run(ctx, p, action: str, target: str, coro, detail: dict | None = None):
    try:
        view = await coro
    except InsufficientMemory as e:
        raise api_error(409, e.code, e.message, fit=e.fit.as_dict())
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, action, target, detail)
    return view


@router.post("/{dep_id}/sleep")
async def sleep(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "sleep", str(dep_id), ctx.deployer.sleep(dep_id, p.user.id))


@router.post("/{dep_id}/wake")
async def wake(dep_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "wake", str(dep_id), ctx.deployer.wake(dep_id, p.user.id))


@router.post("/{dep_id}/switch")
async def switch(dep_id: int, body: SwitchIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "switch", f"{dep_id}->{body.request.name}",
                      ctx.deployer.switch(dep_id, body.request, body.actions, p.user.id),
                      {"model": body.request.model_ref, "actions": body.actions})


@router.post("/{dep_id}/reconfigure")
async def reconfigure(dep_id: int, body: ReconfigureIn, p: Principal = Depends(require_admin),
                      ctx: AppContext = Depends(get_ctx)):
    return await _run(ctx, p, "reconfigure", str(dep_id),
                      ctx.deployer.reconfigure(dep_id, body.config, body.model_ref, body.actions, p.user.id),
                      {"config": body.config, "model_ref": body.model_ref, "actions": body.actions})
```

`dgxmm/api/audit.py`:
```python
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select

from dgxmm.auth.deps import Principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.db import AuditLog, User

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("")
def list_audit(limit: int = Query(200, ge=1, le=2000), p: Principal = Depends(require_admin),
               ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        names = {u.id: u.username for u in s.scalars(select(User))}
        rows = s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit))
        return [{"ts": a.ts.isoformat(), "user": names.get(a.user_id, "system"), "action": a.action,
                 "target": a.target, "detail": a.detail} for a in rows]
```

`dgxmm/api/containers.py`:
```python
import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from dgxmm.api.deployments import deploy_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.deployer.adopt import adopt, list_adoptable
from dgxmm.deployer.service import DeployError

router = APIRouter(prefix="/api/containers", tags=["containers"])


class AdoptIn(BaseModel):
    name: str


@router.get("/adoptable")
async def adoptable(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return await asyncio.to_thread(list_adoptable, ctx.deployer)


@router.post("/{container}/adopt", status_code=201)
async def adopt_route(container: str, body: AdoptIn, p: Principal = Depends(require_admin),
                      ctx: AppContext = Depends(get_ctx)):
    try:
        view = await adopt(ctx.deployer, container, body.name, p.user.id)
    except DeployError as e:
        raise deploy_error(e)
    audit(ctx.db, p.user.id, "adopt", container, {"as": body.name})
    return view
```

In `dgxmm/app.py`, add `audit as audit_api, containers as containers_api` to the imports and to `ROUTERS`.

- [ ] **Step 3: Frontend: the switch/reconfigure dialog** — `frontend/src/components/SwitchDialog.tsx`

```tsx
import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import type { Artifact, Deployment, PlanAction, Preview } from "../types";
import { ConfigForm } from "./ConfigForm";
import { EstimatePanel } from "./EstimatePanel";
import { PlanChooser } from "./PlanChooser";
import { Button, ErrorBox, Field, Input, Modal, Select } from "./ui";

export function SwitchDialog({ mode, deployment, initialPatch, onClose }: {
  mode: "switch" | "reconfigure"; deployment: Deployment; initialPatch?: Record<string, any>; onClose: () => void;
}) {
  const qc = useQueryClient();
  const inv = useQuery({ queryKey: ["inventory"], queryFn: () => api<Artifact[]>("/inventory") });
  const [ref, setRef] = useState(initialPatch?.model_ref ?? deployment.model_ref);
  const [config, setConfig] = useState<Record<string, any>>(() => {
    const { model_ref: _ignored, ...patch } = initialPatch ?? {};
    return { ...deployment.config, ...patch };
  });
  const [name, setName] = useState(`${deployment.name}-2`.slice(0, 40));
  const [preview, setPreview] = useState<Preview | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [actions, setActions] = useState<PlanAction[]>([]);
  const [advanced, setAdvanced] = useState(false);
  const choices = (inv.data ?? []).filter((a) => a.complete && a.kind === "llm" && a.engines.includes(deployment.engine));

  useEffect(() => {
    const t = setTimeout(async () => {
      try {
        setPreview(await api<Preview>("/deployments/preview", { method: "POST", body: { engine: deployment.engine, model_ref: ref, config, replacing: deployment.id } }));
        setErr(null);
      } catch (e) {
        setErr(e);
        setPreview(null);
      }
    }, 350);
    return () => clearTimeout(t);
  }, [ref, config, deployment.engine, deployment.id]);

  const go = useMutation({
    mutationFn: () => mode === "switch"
      ? api(`/deployments/${deployment.id}/switch`, { method: "POST", body: { request: { name, engine: deployment.engine, model_ref: ref, config, priority: deployment.priority }, actions } })
      : api(`/deployments/${deployment.id}/reconfigure`, { method: "POST", body: { config, model_ref: ref === deployment.model_ref ? null : ref, actions } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["deployments"] }); onClose(); },
  });

  return (
    <Modal open wide title={mode === "switch" ? `Switch ${deployment.name} to another model` : `Reconfigure ${deployment.name}`} onClose={onClose}>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="space-y-3 lg:col-span-2">
          <Field label="Model">
            <Select value={ref} onChange={(e) => setRef(e.target.value)}>
              {choices.map((a) => <option key={a.ref} value={a.ref}>{a.ref} ({a.quant})</option>)}
            </Select>
          </Field>
          {mode === "switch" && <Field label="New deployment name" help={`${deployment.name} is put to sleep (or stopped). Its LiteLLM alias moves to the new deployment.`}><Input value={name} onChange={(e) => setName(e.target.value)} /></Field>}
          <ConfigForm engine={deployment.engine} value={config} onChange={setConfig} showAdvanced={advanced} />
          <button className="text-xs underline" onClick={() => setAdvanced(!advanced)}>{advanced ? "Hide" : "Show"} advanced</button>
          <p className="text-xs text-slate-500">If the new deployment fails to become ready, DGXMM rolls back automatically.</p>
          <ErrorBox error={go.error} />
          <Button variant="primary" disabled={!preview || go.isPending || (!preview.fit.fits && !actions.length)} onClick={() => go.mutate()}>
            {go.isPending ? "Working… (this can take minutes)" : mode === "switch" ? "Switch" : "Apply and restart"}
          </Button>
        </div>
        <div>
          <ErrorBox error={err} />
          {preview && <EstimatePanel preview={preview} />}
          {preview && !preview.fit.fits && (
            <PlanChooser plans={preview.plans ?? []} selected={actions}
              onSelect={(o) => { setActions(o.actions.filter((a) => a.type !== "patch")); if (o.patch) setConfig({ ...config, ...o.patch }); if (o.model_ref) setRef(o.model_ref); }} />
          )}
        </div>
      </div>
    </Modal>
  );
}
```

Add `model_ref?: string | null` to the `PlanOption` interface in `src/types.ts`.

- [ ] **Step 4: Frontend: dashboard actions, adopt list, detail page, audit page**

In `Dashboard.tsx`:
- Import `SwitchDialog`, `useState`, and `Badge` (already imported).
- Add the state `const [dialog, setDialog] = useState<Deployment | null>(null);`.
- In the action cell, before the Stop button, insert:
```tsx
{admin && d.state === "ready" && d.sleep_capable && <Button onClick={() => act.mutate({ id: d.id, action: "sleep" })}>Sleep</Button>}
{admin && d.state === "sleeping" && <Button onClick={() => act.mutate({ id: d.id, action: "wake" })}>Wake</Button>}
{admin && ["ready", "sleeping"].includes(d.state) && <Button onClick={() => setDialog(d)}>Switch…</Button>}
```
- After the deployments `Card`, add:
```tsx
{dialog && <SwitchDialog mode="switch" deployment={dialog} onClose={() => setDialog(null)} />}
<AdoptCard />
```
- Define `AdoptCard` in the same file:
```tsx
function AdoptCard() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const rows = useQuery({ queryKey: ["adoptable"], queryFn: () => api<any[]>("/containers/adoptable") });
  const adopt = useMutation({
    mutationFn: (c: { name: string }) => api(`/containers/${encodeURIComponent(c.name)}/adopt`, { method: "POST", body: { name: c.name.replace(/[^a-z0-9-]/g, "-").replace(/^dmm-/, "").slice(0, 40) } }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["adoptable"] }); qc.invalidateQueries({ queryKey: ["deployments"] }); },
  });
  if (!rows.data?.length) return null;
  return (
    <Card title="vLLM containers not managed by DGXMM">
      <ErrorBox error={adopt.error} />
      <Table head={["Container", "Status", "Image", ""]}>
        {rows.data.map((c) => (
          <tr key={c.id}>
            <td className="td font-mono">{c.name}</td><td className="td">{c.status}</td><td className="td text-xs">{c.image}</td>
            <td className="td">{c.adopted_as ? <Badge tone="green">adopted as {c.adopted_as}</Badge> : admin && <Button onClick={() => adopt.mutate(c)}>Adopt settings</Button>}</td>
          </tr>
        ))}
      </Table>
      <p className="mt-2 text-xs text-slate-500">Adopting copies the container's settings into a stopped DGXMM deployment. The original container is left as it is; stop it before starting the adopted one.</p>
    </Card>
  );
}
```

In `DeploymentDetail.tsx`:
- Import `SwitchDialog` and add `const [dialog, setDialog] = useState<null | { patch?: Record<string, any> }>(null);`.
- Add the header buttons:
```tsx
{admin && d.state === "ready" && d.sleep_capable && <Button onClick={() => act.mutate("sleep")}>Sleep</Button>}
{admin && d.state === "sleeping" && <Button onClick={() => act.mutate("wake")}>Wake</Button>}
{admin && <Button onClick={() => setDialog({})}>Reconfigure…</Button>}
```
- At the end of the returned tree, add `{dialog && <SwitchDialog mode="reconfigure" deployment={d} initialPatch={dialog.patch} onClose={() => setDialog(null)} />}`.

`frontend/src/pages/Audit.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Card, ErrorBox, Table } from "../components/ui";

export function Audit() {
  const q = useQuery({ queryKey: ["audit"], queryFn: () => api<any[]>("/audit?limit=500"), refetchInterval: 15000 });
  return (
    <Card title="Audit log">
      <ErrorBox error={q.error} />
      <Table head={["Time", "User", "Action", "Target", "Detail"]}>
        {(q.data ?? []).map((a, i) => (
          <tr key={i}>
            <td className="td whitespace-nowrap font-mono text-xs">{new Date(a.ts).toLocaleString()}</td>
            <td className="td">{a.user}</td><td className="td">{a.action}</td><td className="td">{a.target}</td>
            <td className="td font-mono text-xs">{Object.keys(a.detail ?? {}).length ? JSON.stringify(a.detail) : ""}</td>
          </tr>
        ))}
      </Table>
    </Card>
  );
}
```

In `main.tsx`, add `import { Audit } from "./pages/Audit";` and `<Route path="/audit" element={<Audit />} />`.

- [ ] **Step 5: Run the backend tests and the frontend build**

`DGX$ .venv/bin/pytest -q` → all pass. `NPM$ sh -c "npx vitest run && npm run build"` → passes with no TypeScript errors.

- [ ] **Step 6: [USER] Enable DB-stored models in LiteLLM**

LiteLLM's config currently has `store_model_in_db: false`, so `/model/new` would be rejected. With the user's OK:
```bash
DGX$ docker inspect litellm-proxy --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' | grep config.yaml
```
- Back up that source file (it belongs to the LiteLLM installation, not to DGXMM).
- Change `store_model_in_db: false` to `store_model_in_db: true` and run `docker restart litellm-proxy`. This briefly interrupts LiteLLM, so tell the user first.
- Verify: `docker logs dgxmm 2>&1 | grep -i litellm`. Within about 60 s there are no `LiteLLM sync failed` lines, and ready deployments show `litellm_status: synced`.


## Milestone M3: llama.cpp

### Task 22: GGUF metadata reader and GGUF inventory

**Files:**
- Create: `dgxmm/inventory/gguf_meta.py`, `dgxmm/inventory/gguf_scan.py`
- Modify: `dgxmm/inventory/hf_scan.py` (leave GGUF-only repos to the GGUF scanner), `dgxmm/inventory/service.py` (default scanners)
- Modify: `tests/conftest.py` (add the `write_gguf` helper)
- Test: `tests/test_gguf.py`

**Interfaces:**
- Consumes: `ArchSpec`, `LayerGroup`, `ModelArtifact`.
- Produces: `FILE_TYPES`, `read_gguf_meta(path) -> dict`, `archspec_from_gguf(meta) -> ArchSpec`, `scan_gguf_roots(roots: list[Path]) -> list[ModelArtifact]`.

- [ ] **Step 1: Add the GGUF writer helper to `tests/conftest.py`** (append)

```python
import gguf as _gguf


def write_gguf(path, arch="qwen3", layers=36, hidden=4096, heads=32, kv=8, head_dim=128, ctx=40960, vocab=151936,
               ftype=15, extra: dict | None = None, pad_bytes: int = 0):
    """Write a metadata-only GGUF (no tensors) that llama.cpp-style readers can parse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    w = _gguf.GGUFWriter(str(path), arch)
    w.add_uint32(f"{arch}.block_count", layers)
    w.add_uint32(f"{arch}.embedding_length", hidden)
    w.add_uint32(f"{arch}.feed_forward_length", hidden * 3)
    w.add_uint32(f"{arch}.attention.head_count", heads)
    if isinstance(kv, list):
        w.add_array(f"{arch}.attention.head_count_kv", kv)
    else:
        w.add_uint32(f"{arch}.attention.head_count_kv", kv)
    w.add_uint32(f"{arch}.attention.key_length", head_dim)
    w.add_uint32(f"{arch}.attention.value_length", head_dim)
    w.add_uint32(f"{arch}.context_length", ctx)
    w.add_uint32("general.file_type", ftype)
    w.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(vocab)])
    for k, v in (extra or {}).items():
        if isinstance(v, list):
            w.add_array(k, v)
        else:
            w.add_uint32(k, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    if pad_bytes:
        with open(path, "ab") as f:
            f.truncate(path.stat().st_size + pad_bytes)
    return path
```

- [ ] **Step 2: Write the failing tests** — `tests/test_gguf.py`

```python
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
```

Run: `DGX$ .venv/bin/pytest -q tests/test_gguf.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/inventory/gguf_meta.py`**

```python
"""GGUF header metadata → ArchSpec (no tensor data is read)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from gguf import GGUFReader, GGUFValueType

from dgxmm.inventory.archspec import ArchSpec, LayerGroup

FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
              19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
              26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
              36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}


def _scalar(part) -> Any:
    v = part.tolist()
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def _value(field) -> Any:
    if not field.types:
        return None
    t0 = field.types[0]
    if t0 == GGUFValueType.ARRAY:
        if len(field.types) > 1 and field.types[1] == GGUFValueType.STRING:
            return len(field.data)  # token lists etc.: only the count is needed
        return [_scalar(field.parts[i]) for i in field.data]
    if t0 == GGUFValueType.STRING:
        return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")
    return _scalar(field.parts[field.data[0]])


def read_gguf_meta(path: Path) -> dict[str, Any]:
    reader = GGUFReader(str(path), "r")
    return {k: _value(f) for k, f in reader.fields.items() if not k.startswith("GGUF.")}


def archspec_from_gguf(meta: dict[str, Any]) -> ArchSpec:
    a = str(meta.get("general.architecture", "unknown"))
    assumptions: list[str] = []

    def g(key: str, default: Any = None, required: bool = False) -> Any:
        v = meta.get(f"{a}.{key}")
        if v is None:
            if required:
                assumptions.append(key)
            return default
        return v

    n_layers = int(g("block_count", 32, True))
    hidden = int(g("embedding_length", 4096, True))
    heads = g("attention.head_count", max(hidden // 128, 1), True)
    heads = max(heads) if isinstance(heads, list) else int(heads)
    kv = g("attention.head_count_kv", heads)
    kv_list = [int(x) for x in kv] if isinstance(kv, list) else [int(kv)] * n_layers
    if len(kv_list) != n_layers:
        assumptions.append("head_count_kv_length")
        kv_list = (kv_list * n_layers)[:n_layers]
    key_len = int(g("attention.key_length") or hidden // heads)
    val_len = int(g("attention.value_length") or key_len)
    head_dim = (key_len + val_len) // 2
    window = g("attention.sliding_window")
    pattern = g("attention.sliding_window_pattern")
    fai = g("full_attention_interval")
    has_ssm = g("ssm.state_size") is not None
    kinds: list[tuple[str, int]] = []
    for i in range(n_layers):
        if kv_list[i] == 0 or (fai and (i + 1) % int(fai) != 0):
            kinds.append(("mamba" if has_ssm else "linear", 0))
        elif window and isinstance(pattern, list) and i < len(pattern):
            kinds.append(("sliding" if pattern[i] else "full", kv_list[i]))
        elif window and isinstance(pattern, int) and pattern > 0:
            kinds.append(("full" if (i + 1) % pattern == 0 else "sliding", kv_list[i]))
        else:
            kinds.append(("full", kv_list[i]))
    state = 0
    if any(k in ("mamba", "linear") for k, _ in kinds):
        d_inner = int(g("ssm.inner_size", 2 * hidden))
        d_state = int(g("ssm.state_size", 128))
        conv = int(g("ssm.conv_kernel", 4))
        groups = int(g("ssm.group_count", 1))
        state = d_inner * d_state + (conv - 1) * (d_inner + 2 * groups * d_state)
    groups_out = []
    order = {"full": 0, "sliding": 1, "linear": 3, "mamba": 4}
    for (kind, kvh), count in sorted(Counter(kinds).items(), key=lambda x: (order[x[0][0]], x[0][1])):
        if kind in ("mamba", "linear"):
            groups_out.append(LayerGroup(kind, count, state_elems=state))  # type: ignore[arg-type]
        else:
            groups_out.append(LayerGroup(kind, count, kv_heads=kvh, head_dim=head_dim,  # type: ignore[arg-type]
                                         window=int(window) if kind == "sliding" and window else None))
    experts = g("expert_count")
    used = g("expert_used_count")
    ff = g("feed_forward_length", 4 * hidden)
    ff = max(ff) if isinstance(ff, list) else int(ff)
    expert_ff = g("expert_feed_forward_length")
    is_moe = bool(experts and int(experts) > 1)
    inter = int(expert_ff) * int(used or 1) + int(g("expert_shared_feed_forward_length", 0) or 0) if is_moe and expert_ff else ff
    vocab = meta.get("tokenizer.ggml.tokens") or g("vocab_size")
    if not vocab:
        assumptions.append("vocab_size")
        vocab = 32000
    ftype = meta.get("general.file_type")
    return ArchSpec(architecture=f"gguf:{a}", model_type=a, num_layers=n_layers, hidden_size=hidden,
                    vocab_size=int(vocab), max_context=int(g("context_length", 4096, True)),
                    layer_groups=tuple(groups_out), intermediate_size=inter, is_moe=is_moe,
                    num_experts=int(experts) if experts else None, experts_per_tok=int(used) if used else None,
                    torch_dtype="bfloat16", quant=f"gguf:{FILE_TYPES.get(ftype, ftype)}",
                    assumptions=tuple(dict.fromkeys(assumptions)))
```

- [ ] **Step 4: Implement `dgxmm/inventory/gguf_scan.py`**

```python
"""Find .gguf files under configured roots and inside HF cache snapshots."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.inventory.gguf_meta import archspec_from_gguf, read_gguf_meta

log = logging.getLogger(__name__)
_SHARD = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)
_META_CACHE: dict[tuple[str, float, int], dict] = {}


def _meta(p: Path) -> dict:
    st = p.stat()
    key = (str(p), st.st_mtime, st.st_size)
    if key not in _META_CACHE:
        _META_CACHE[key] = read_gguf_meta(p)
    return _META_CACHE[key]


def _iter_gguf(root: Path):
    if not root.is_dir():
        return
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".gguf"):
                yield Path(dirpath) / f


def scan_gguf_roots(roots: list[Path]) -> list[ModelArtifact]:
    out: list[ModelArtifact] = []
    for root in roots:
        files = sorted(_iter_gguf(root))
        names = {p.name: p for p in files}
        for p in files:
            m = _SHARD.match(p.name)
            if m and m.group("idx") != "00001":
                continue  # later shards are folded into the first one
            problems: list[str] = []
            shard_paths = [p]
            if m:
                n = int(m.group("n"))
                shard_paths = [p.with_name(f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]
                missing = [s.name for s in shard_paths if s.name not in names or not s.exists()]
                if missing:
                    problems.append(f"missing shards: {', '.join(missing)}")
            try:
                sizes = [s.stat().st_size for s in shard_paths if s.exists()]
            except OSError:
                sizes = []
            if not p.exists():
                out.append(ModelArtifact(ref=str(p), source="gguf", local_path=str(p), revision=None, format="gguf",
                                         weights_bytes=0, size_bytes=0, complete=False, kind="other", arch=None,
                                         architecture=None, quant=None, problems=["broken symlink"]))
                continue
            arch = None
            try:
                arch = archspec_from_gguf(_meta(p))
            except Exception as e:
                problems.append(f"cannot read GGUF header: {e}")
            kind = "other" if "mmproj" in p.name.lower() else ("llm" if arch else "other")
            out.append(ModelArtifact(ref=str(p), source="gguf", local_path=str(p), revision=None, format="gguf",
                                     weights_bytes=sum(sizes), size_bytes=sum(sizes),
                                     complete=arch is not None and not problems, kind=kind, arch=arch,
                                     architecture=arch.architecture if arch else None,
                                     quant=arch.quant if arch else None, problems=problems,
                                     files=[s.name for s in shard_paths]))
    return out
```

- [ ] **Step 5: Leave GGUF-only HF repos to the GGUF scanner**

In `dgxmm/inventory/hf_scan.py`:
- Change `_scan_repo` to return `ModelArtifact | None`.
- At the top of the `if chosen is None:` branch, add:
```python
        if any(any(f.lower().endswith(".gguf") for f in os.listdir(s)) for s in snaps if s.is_dir()):
            return None  # GGUF repo: reported by scan_gguf_roots instead
```
- In `scan_hf_cache`, replace `out.append(_scan_repo(d))` with:
```python
                a = _scan_repo(d)
                if a is not None:
                    out.append(a)
```

In `dgxmm/inventory/service.py`, replace the default `self.scanners = …` line with:
```python
        from dgxmm.inventory.gguf_scan import scan_gguf_roots
        gguf_roots = [Path(m.local) for m in settings.gguf_roots] + [hub]
        self.scanners = scanners or [lambda: scan_hf_cache(hub), lambda: scan_gguf_roots(gguf_roots)]
```

- [ ] **Step 6: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_gguf.py tests/test_inventory.py`. Expected: all pass. If `GGUFWriter` rejects writing with zero tensors in the installed `gguf` version, add a single 1-element F32 tensor in `write_gguf` (`w.add_tensor("dummy", numpy.zeros(1, dtype=numpy.float32))`) and keep the assertions.

---

### Task 23: llama.cpp engine

**Files:**
- Create: `dgxmm/engines/llamacpp.py`
- Modify: `dgxmm/memory/estimate.py` (add `estimate_llamacpp`), `dgxmm/engines/registry.py`
- Test: `tests/test_engines_llamacpp.py`

**Interfaces:**
- Consumes: `kv_cache_bytes`, `state_bytes`, `kv_dtype_bytes`, `LLAMACPP_EXTRA_FLAGS`, `openai_health`, `metrics_from`.
- Produces:
  - `LlamaCppSizing(ctx_size, parallel, cache_type_k, cache_type_v, n_ubatch)`.
  - `estimate_llamacpp(spec, weights_bytes, sizing, settings) -> MemoryEstimate`.
  - `LlamaCppConfig`, `LlamaCppEngine` (Engine protocol; `context_key = "ctx_size"`, `seqs_key = "parallel"`, `supports_sleep = False`).
  - `container_path(local_path, settings) -> tuple[Mount, str]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_engines_llamacpp.py`

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from dgxmm.config import GIB
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import LaunchContext
from dgxmm.engines.llamacpp import LlamaCppEngine
from dgxmm.inventory.gguf_scan import scan_gguf_roots
from dgxmm.memory.estimate import LlamaCppSizing, estimate_llamacpp
from tests.conftest import write_gguf

LLAMA_METRICS = """# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 120
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 80
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
"""


@pytest.fixture
def gguf_art(settings):
    root = Path(settings.gguf_roots[0].local)
    write_gguf(root / "qwen" / "Qwen3-8B-Q4_K_M.gguf", pad_bytes=5 * GIB)
    [a] = scan_gguf_roots([root])
    return a


def test_estimate_exact(settings, gguf_art):
    est = estimate_llamacpp(gguf_art.arch, gguf_art.weights_bytes, LlamaCppSizing(32768, 4, "f16", "f16", 512), settings)
    assert est.kv_cache == 147456 * 32768
    assert est.activations == 512 * 151936 * 4 + 512 * 4096 * 4 * 8 + 256 * 1024**2
    assert est.runtime == int(0.5 * GIB) and est.weights == gguf_art.weights_bytes
    q8 = estimate_llamacpp(gguf_art.arch, 0, LlamaCppSizing(32768, 4, "q8_0", "q8_0", 512), settings)
    assert q8.kv_cache == int(147456 * 32768 * 34 / 64 + 0.999999)


def test_config_validation(settings):
    e = LlamaCppEngine(settings)
    assert e.validate_config({}).ctx_size == 32768
    for bad in ({"cache_type_v": "q8_0", "flash_attn": "off"}, {"n_ubatch": 4096, "n_batch": 1024},
                {"extra_args": ["--model", "x"]}, {"alias": "a b"}):
        with pytest.raises((ValidationError, ValueError)):
            e.validate_config(bad)


def test_container_spec_maps_paths(settings, gguf_art):
    e = LlamaCppEngine(settings)
    cfg = e.validate_config({"ctx_size": 16384, "parallel": 2, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"})
    spec = e.container_spec(gguf_art, cfg, None, LaunchContext(3, "q", 8110, 10 * GIB, 120 * GIB), settings)
    c = spec.command
    assert c[c.index("-m") + 1] == "/models/0/qwen/Qwen3-8B-Q4_K_M.gguf"
    assert spec.mounts[0].host == settings.gguf_roots[0].host and spec.mounts[0].target == "/models/0" and spec.mounts[0].mode == "ro"
    assert c[c.index("-c") + 1] == "16384" and c[c.index("-np") + 1] == "2" and "--metrics" in c
    assert c[c.index("-a") + 1] == "qwen3-8b-q4_k_m" and spec.image == settings.images.llamacpp


def test_metrics_and_engine_flags(settings):
    e = LlamaCppEngine(settings)
    m = e.parse_metrics(LLAMA_METRICS)
    assert m.counters == {"prompt_tokens": 120.0, "generation_tokens": 80.0}
    assert m.gauges == {"running": 1.0, "waiting": 0.0}
    assert e.supports_sleep is False and e.kv_patch_options(e.validate_config({})) == [
        {"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "auto"}]


async def test_deploy_llamacpp_end_to_end(stack, settings):
    write_gguf(Path(settings.gguf_roots[0].local) / "Qwen3-8B-Q4_K_M.gguf", pad_bytes=1 * GIB)
    stack.inv.refresh()
    ref = next(a.ref for a in stack.inv.list() if a.source == "gguf")
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="g", engine="llamacpp", model_ref=ref,
                                                  config={"ctx_size": 8192, "parallel": 2}), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    assert stack.sdk.run_calls[-1]["image"] == settings.images.llamacpp
```

Run: `DGX$ .venv/bin/pytest -q tests/test_engines_llamacpp.py`. Expected: FAIL (ImportError).

(The `q8_0` assertion: K and V are both `q8_0` at 34/32 bytes. `kv_cache_bytes` gets their mean `34/32` as bytes per element; the ceiling of `147456/2 × 32768 × 34/32` equals the integer shown.)

- [ ] **Step 2: Add `estimate_llamacpp` to `dgxmm/memory/estimate.py`**

```python
MIB = 1024**2


@dataclass(frozen=True)
class LlamaCppSizing:
    ctx_size: int
    parallel: int
    cache_type_k: str
    cache_type_v: str
    n_ubatch: int


def estimate_llamacpp(spec: ArchSpec, weights_bytes: int, cfg: LlamaCppSizing, settings: Settings) -> MemoryEstimate:
    b = (kv_dtype_bytes(cfg.cache_type_k, spec.torch_dtype) + kv_dtype_bytes(cfg.cache_type_v, spec.torch_dtype)) / 2
    kv = kv_cache_bytes(spec, tokens_total=cfg.ctx_size, seqs=cfg.parallel, context=max(cfg.ctx_size // cfg.parallel, 1),
                        bytes_per_elem=b, window_slack=cfg.n_ubatch)
    st = state_bytes(spec, cfg.parallel)
    compute = cfg.n_ubatch * spec.vocab_size * 4 + cfg.n_ubatch * spec.hidden_size * 4 * 8 + 256 * MIB
    runtime = int(settings.llamacpp_runtime_fixed_gib * GIB)
    margin, confidence, notes = _low_conf_margin(spec, kv, st, compute)
    return MemoryEstimate(weights=weights_bytes, draft_weights=0, kv_cache=kv, state=st, activations=compute,
                          runtime=runtime, margin=margin, confidence=confidence, notes=notes)
```

In `test_estimate_exact`, `kv_cache_bytes` sees `bytes_per_elem = 2` for f16 (as before). For q8_0 the code computes `ceil(36 * 2 * 8 * 128 * (34/32) * 32768)`. The test expresses this as `int(147456 * 32768 * 34 / 64 + 0.999999)`, since `147456 = 36·2·8·128·2`. Keep both sides as written: they are the same quantity.

- [ ] **Step 3: Implement `dgxmm/engines/llamacpp.py`**

```python
"""llama.cpp llama-server engine (ghcr.io/ggml-org/llama.cpp:server-cuda)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dgxmm.config import Mount, Settings
from dgxmm.engines.argguard import LLAMACPP_EXTRA_FLAGS, validate_extra_args
from dgxmm.engines.base import (SERVED_NAME_RE, ContainerSpec, EngineMetrics, LaunchContext, container_name,
                                default_served_name, managed_labels, openai_health)
from dgxmm.engines.vllm import metrics_from
from dgxmm.inventory.artifact import ModelArtifact
from dgxmm.memory.estimate import LlamaCppSizing, MemoryEstimate, estimate_llamacpp

CacheType = Literal["f16", "bf16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl"]
LLAMA_METRICS = {
    "gauges": {"running": ("llamacpp:requests_processing",), "waiting": ("llamacpp:requests_deferred",),
               "kv_usage": ("llamacpp:kv_cache_usage_ratio",)},
    "counters": {"prompt_tokens": ("llamacpp:prompt_tokens",), "generation_tokens": ("llamacpp:tokens_predicted",)},
    "histograms": {},
}


class LlamaCppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alias: str | None = Field(None, pattern=SERVED_NAME_RE)
    ctx_size: int = Field(32768, ge=512, le=1_048_576)
    parallel: int = Field(4, ge=1, le=64)
    n_gpu_layers: int = Field(999, ge=0, le=999)
    cache_type_k: CacheType = "f16"
    cache_type_v: CacheType = "f16"
    flash_attn: Literal["auto", "on", "off"] = "auto"
    n_batch: int = Field(2048, ge=32, le=16384)
    n_ubatch: int = Field(512, ge=32, le=8192)
    extra_args: list[str] = []

    @field_validator("extra_args")
    @classmethod
    def _extra(cls, v: list[str]) -> list[str]:
        return validate_extra_args(v, LLAMACPP_EXTRA_FLAGS)

    @model_validator(mode="after")
    def _check(self):
        if self.n_ubatch > self.n_batch:
            raise ValueError("n_ubatch must be <= n_batch")
        if self.cache_type_v not in ("f16", "bf16", "f32") and self.flash_attn == "off":
            raise ValueError("a quantized V cache needs flash attention (flash_attn auto or on)")
        return self


def container_path(local_path: str, settings: Settings) -> tuple[Mount, str]:
    p = Path(local_path)
    hf_local = Path(settings.hf_cache.local)
    if hf_local in p.parents:
        return (Mount(host=settings.hf_cache.host, target="/root/.cache/huggingface", mode="ro"),
                "/root/.cache/huggingface/" + str(p.relative_to(hf_local)))
    for i, m in enumerate(settings.gguf_roots):
        root = Path(m.local)
        if root in p.parents:
            return Mount(host=m.host, target=f"/models/{i}", mode="ro"), f"/models/{i}/{p.relative_to(root)}"
    raise ValueError(f"{local_path} is not under an allowed GGUF root")


class LlamaCppEngine:
    name = "llamacpp"
    supports_sleep = False
    config_model = LlamaCppConfig
    context_key = "ctx_size"
    seqs_key = "parallel"

    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_config(self, raw: dict[str, Any]) -> LlamaCppConfig:
        return LlamaCppConfig.model_validate(raw)

    def supports(self, artifact: ModelArtifact) -> bool:
        return artifact.source == "gguf" and artifact.kind == "llm"

    def draft_ref(self, cfg: LlamaCppConfig) -> str | None:
        return None

    def served_name(self, cfg: LlamaCppConfig, model_ref: str) -> str:
        return cfg.alias or default_served_name(model_ref)

    def sleep_enabled(self, cfg: LlamaCppConfig) -> bool:
        return False

    def estimate(self, artifact: ModelArtifact, cfg: LlamaCppConfig, draft, settings: Settings) -> MemoryEstimate:
        if artifact.arch is None:
            raise ValueError("GGUF header could not be parsed")
        return estimate_llamacpp(artifact.arch, artifact.weights_bytes,
                                 LlamaCppSizing(cfg.ctx_size, cfg.parallel, cfg.cache_type_k, cfg.cache_type_v, cfg.n_ubatch),
                                 settings)

    def container_spec(self, artifact: ModelArtifact, cfg: LlamaCppConfig, draft, launch: LaunchContext,
                       settings: Settings) -> ContainerSpec:
        mount, path = container_path(artifact.local_path, settings)
        cmd = ["-m", path, "--host", "0.0.0.0", "--port", str(launch.port), "-c", str(cfg.ctx_size),
               "-np", str(cfg.parallel), "-ngl", str(cfg.n_gpu_layers), "-ctk", cfg.cache_type_k,
               "-ctv", cfg.cache_type_v, "-fa", cfg.flash_attn, "-b", str(cfg.n_batch), "-ub", str(cfg.n_ubatch),
               "--metrics", "--slots", "-a", self.served_name(cfg, artifact.ref), *cfg.extra_args]
        return ContainerSpec(name=container_name(launch.name), image=settings.images.llamacpp, command=cmd,
                             environment={}, mounts=[mount], labels=managed_labels(launch.deployment_id, self.name),
                             network=settings.docker_network, port=launch.port, publish_bind=settings.publish_bind,
                             ipc_host=False)

    async def health(self, http: httpx.AsyncClient, base_url: str, served_name: str) -> bool:
        return await openai_health(http, base_url, served_name)

    def parse_metrics(self, text: str) -> EngineMetrics:
        return metrics_from(text, LLAMA_METRICS)

    async def sleep(self, http, base_url) -> None:
        raise NotImplementedError("llama.cpp has no sleep mode")

    async def wake(self, http, base_url) -> None:
        raise NotImplementedError("llama.cpp has no sleep mode")

    def kv_patch_options(self, cfg: LlamaCppConfig) -> list[dict]:
        if cfg.cache_type_k == "f16":
            return [{"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on" if cfg.flash_attn == "off" else cfg.flash_attn}]
        return []
```

`dgxmm/engines/registry.py`:
```python
from dgxmm.config import Settings
from dgxmm.engines.llamacpp import LlamaCppEngine
from dgxmm.engines.vllm import VllmEngine


def build_engines(settings: Settings) -> dict:
    return {"vllm": VllmEngine(settings), "llamacpp": LlamaCppEngine(settings)}
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q`. Expected: all pass. `test_inventory_host_engines` in `tests/test_api_deployments.py` asserts `eng[0]["name"] == "vllm"`, which still holds because dict order is preserved.

---

### Task 24: llama.cpp on-box check

**Files:** none (verification only).

- [ ] **Step 1: Pull the image and confirm the binary runs on GB10**

```bash
DGX$ docker pull ghcr.io/ggml-org/llama.cpp:server-cuda
DGX$ docker run --rm --gpus all ghcr.io/ggml-org/llama.cpp:server-cuda --version
```
Expected: a version line and a `ggml_cuda_init: found 1 CUDA devices: … NVIDIA GB10` line. If the CUDA init fails (an image built without `sm_121`), stop and tell the user. The fallback is to build `llama-server` from source with `-DCMAKE_CUDA_ARCHITECTURES=121` and set `images.llamacpp` in `config.yaml`. That is a separate task, not covered here.

- [ ] **Step 2: Download a tiny GGUF and deploy it through DGXMM** (tell the user first; about 0.6 GB)

```bash
DGX$ .venv/bin/python -c "from huggingface_hub import hf_hub_download as d; print(d('unsloth/Qwen3-0.6B-GGUF','Qwen3-0.6B-Q8_0.gguf', cache_dir='$HOME/.cache/huggingface/hub'))"
```
- In the UI, open **Models on disk → Rescan**. Expected: an artifact ending in `Qwen3-0.6B-Q8_0.gguf` with engines `llamacpp`.
- Deploy it with `ctx_size 8192`, `parallel 2`. Expected: `ready` within about a minute, and `curl -s http://127.0.0.1:<port>/v1/models` lists `qwen3-0.6b-q8_0`.
- Note `reserved_bytes` and `actual_bytes`, then stop and remove it.

---

## Milestone M4: Catalog and downloads

### Task 25: vLLM architecture registry

**Files:**
- Create: `dgxmm/catalog/__init__.py` (empty), `dgxmm/catalog/support.py`
- Modify: `dgxmm/engines/vllm.py` (optional architecture source), `dgxmm/context.py` (wiring)
- Test: `tests/test_support.py`

**Interfaces:**
- Consumes: `DockerController.run_oneshot`, `Settings.images.vllm`, `Settings.data_dir`.
- Produces:
  - `VllmArchRegistry(settings, docker)` with `get() -> set[str] | None` and `async ensure() -> set[str] | None`.
  - `engine_support(architecture, is_gguf, archs) -> dict`.
  - `VllmEngine.arch_source: Callable[[], set[str] | None] | None`.

- [ ] **Step 1: Write the failing tests** — `tests/test_support.py`

```python
from pathlib import Path

from dgxmm.catalog.support import VllmArchRegistry, engine_support
from dgxmm.docker_ctl import DockerController
from dgxmm.engines.vllm import VllmEngine
from dgxmm.inventory.hf_scan import scan_hf_cache
from tests.conftest import QWEN_TINY
from tests.fakes import FakeDockerClient


async def test_registry_extracts_once_and_caches(settings):
    sdk = FakeDockerClient()
    sdk.oneshot_output = "INFO noise\nARCHS=[\"LlamaForCausalLM\", \"Qwen3ForCausalLM\"]\n"
    reg = VllmArchRegistry(settings, DockerController(sdk))
    assert reg.get() is None
    assert await reg.ensure() == {"LlamaForCausalLM", "Qwen3ForCausalLM"}
    assert sdk.run_calls[-1]["entrypoint"][0] == "python3" and sdk.run_calls[-1]["image"] == settings.images.vllm
    again = VllmArchRegistry(settings, DockerController(FakeDockerClient()))
    assert again.get() == {"LlamaForCausalLM", "Qwen3ForCausalLM"}  # read from data_dir cache


async def test_registry_failure_is_none(settings):
    sdk = FakeDockerClient()
    sdk.oneshot_output = "Traceback: boom"
    assert await VllmArchRegistry(settings, DockerController(sdk)).ensure() is None


def test_engine_support_and_vllm_gate(settings, fake_hf_repo):
    assert engine_support("Qwen3ForCausalLM", False, None) == {"vllm": None, "llamacpp": False}
    assert engine_support("Qwen3ForCausalLM", False, {"Qwen3ForCausalLM"}) == {"vllm": True, "llamacpp": False}
    assert engine_support("gguf:qwen3", True, {"Qwen3ForCausalLM"}) == {"vllm": False, "llamacpp": True}
    fake_hf_repo("Qwen/Qwen3-Tiny", config=QWEN_TINY)
    [a] = scan_hf_cache(Path(settings.hf_cache.local) / "hub")
    e = VllmEngine(settings)
    assert e.supports(a)
    e.arch_source = lambda: {"LlamaForCausalLM"}
    assert not e.supports(a)
    e.arch_source = lambda: None
    assert e.supports(a)
```

Run: `DGX$ .venv/bin/pytest -q tests/test_support.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/catalog/support.py`**

```python
"""Which architectures the pinned vLLM image can serve (spec §7)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)
SCRIPT = ("import json; from vllm import ModelRegistry; "
          "print('ARCHS=' + json.dumps(sorted(ModelRegistry.get_supported_archs())))")


class VllmArchRegistry:
    def __init__(self, settings, docker):
        self.image = settings.images.vllm
        self.docker = docker
        self.path = Path(settings.data_dir) / f"vllm_archs_{re.sub(r'[^A-Za-z0-9]+', '_', self.image)}.json"
        self._archs: set[str] | None = None
        self._lock = asyncio.Lock()

    def get(self) -> set[str] | None:
        if self._archs is None and self.path.is_file():
            try:
                self._archs = set(json.loads(self.path.read_text()))
            except (OSError, ValueError):
                self._archs = None
        return self._archs

    async def ensure(self) -> set[str] | None:
        if self.get() is not None:
            return self._archs
        async with self._lock:
            if self.get() is not None:
                return self._archs
            try:
                out = await asyncio.to_thread(self.docker.run_oneshot, self.image, ["python3", "-c", SCRIPT], [])
                line = next(l for l in out.splitlines() if l.startswith("ARCHS="))
                archs = json.loads(line[len("ARCHS="):])
            except Exception as e:
                log.warning("could not read vLLM model registry from %s: %s", self.image, e)
                return None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(archs))
            self._archs = set(archs)
            return self._archs


def engine_support(architecture: str | None, is_gguf: bool, archs: set[str] | None) -> dict:
    if is_gguf:
        return {"vllm": False, "llamacpp": True}
    vllm = None if archs is None or not architecture else architecture in archs
    return {"vllm": vllm, "llamacpp": False}
```

- [ ] **Step 3: Gate `VllmEngine.supports`**

In `dgxmm/engines/vllm.py`:
- In `VllmEngine.__init__`, add `self.arch_source = None`.
- Replace `supports` with:
```python
    def supports(self, artifact: ModelArtifact) -> bool:
        if not (artifact.source == "hf" and artifact.format == "safetensors" and artifact.kind == "llm"):
            return False
        archs = self.arch_source() if self.arch_source else None
        return archs is None or artifact.architecture in archs
```

In `build_context` (before `return ctx`):
```python
    from dgxmm.catalog.support import VllmArchRegistry
    archs = VllmArchRegistry(settings, docker)
    engines["vllm"].arch_source = archs.get
    ctx.archs = archs

    async def _archs():
        await archs.ensure()
    ctx.background.append(_archs)
```
Add `archs: Any = None` to `AppContext`.

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_support.py`. Expected: `3 passed`.

---

### Task 26: Hugging Face Hub client and catalog service

**Files:**
- Create: `dgxmm/catalog/hub.py`, `dgxmm/catalog/service.py`
- Test: `tests/test_catalog.py`

**Interfaces:**
- Consumes: `HfApi`, `httpx`, `parse_hf_config`, `estimate_vllm`, `VllmSizing`, `check_fit`, `BudgetService`, `InventoryService`, `Deployer.active_deps`, `VllmArchRegistry`, `Llmfit` (Task 27), `base_model_name`, `Settings.read_secret`.
- Produces:
  - `HubModel`, `HubFile`, `HubDetails`, `HfHubClient(token=None, api=None, http=None)` with `.search(query, limit, gguf)` and `.details(repo_id)`.
  - `quick_required(est_bytes, settings) -> int`, `fit_badge(required, budget, evictable) -> dict`, `guess_quant(name) -> str | None`.
  - `CatalogService(settings, hub, llmfit, inventory, budget, deployer, archs)` with `async search(q, gguf=False, ctx=32768, limit=30)` and `async details(repo_id, ctx=32768)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_catalog.py`

```python
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
```

Run: `DGX$ .venv/bin/pytest -q tests/test_catalog.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/catalog/hub.py`**

```python
"""Thin, testable wrapper over huggingface_hub (spec §7)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from dgxmm.config import GIB, Settings

DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I8": 1, "U8": 1,
               "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "F4": 0.5, "BOOL": 1}
_QUANT = re.compile(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|Q\d|NVFP4|MXFP4|FP8|AWQ|GPTQ|BF16|F16|INT4|INT8|W4A16)", re.I)


@dataclass
class HubModel:
    repo_id: str
    downloads: int
    likes: int
    pipeline_tag: str | None
    library: str | None
    tags: list[str]
    gated: Any
    est_bytes: int | None
    params: int | None
    has_gguf: bool
    last_modified: str | None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HubFile:
    name: str
    size: int


@dataclass
class HubDetails:
    repo_id: str
    files: list[HubFile]
    config: dict | None
    gated: Any
    sha: str | None = None
    extra: dict = field(default_factory=dict)


def guess_quant(name: str) -> str | None:
    m = _QUANT.findall(name)
    return m[-1].upper() if m else None


def quick_required(est_bytes: int, settings: Settings) -> int:
    """Rough requirement for list views: weights + 5% + vLLM runtime + 2 GiB default KV allowance."""
    runtime = int((settings.vllm_runtime_fixed_gib + settings.vllm_cuda_graph_gib) * GIB)
    return int(est_bytes * 1.05) + runtime + 2 * GIB


def fit_badge(required: int, budget, evictable: int) -> dict:
    if required <= budget.free_for_new:
        badge = "fits_now"
    elif required <= budget.free_for_new + evictable:
        badge = "fits_after_evict"
    else:
        badge = "too_big"
    return {"badge": badge, "required": required, "free": budget.free_for_new}


class HfHubClient:
    def __init__(self, token: str | None = None, api=None, http: httpx.Client | None = None):
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api, self.token = api, token
        self.http = http or httpx.Client(timeout=20, follow_redirects=True)

    @staticmethod
    def _bytes(st) -> tuple[int | None, int | None]:
        if not st or not getattr(st, "parameters", None):
            return None, None
        params = sum(st.parameters.values())
        return int(sum(n * DTYPE_BYTES.get(d, 2) for d, n in st.parameters.items())), params

    def search(self, query: str | None, limit: int = 30, gguf: bool = False) -> list[HubModel]:
        kw: dict[str, Any] = dict(search=query or None, sort="downloads", limit=limit,
                                  expand=["downloads", "likes", "pipeline_tag", "library_name", "tags", "gated",
                                          "safetensors", "lastModified", "gguf"])
        if gguf:
            kw["library"] = "gguf"
        else:
            kw["pipeline_tag"] = "text-generation"
        out = []
        for m in self.api.list_models(**kw):
            est, params = self._bytes(getattr(m, "safetensors", None))
            tags = list(getattr(m, "tags", None) or [])
            lm = getattr(m, "last_modified", None)
            out.append(HubModel(m.id, int(getattr(m, "downloads", 0) or 0), int(getattr(m, "likes", 0) or 0),
                                getattr(m, "pipeline_tag", None), getattr(m, "library_name", None), tags,
                                getattr(m, "gated", False), est, params,
                                bool(getattr(m, "gguf", None)) or "gguf" in tags, lm.isoformat() if hasattr(lm, "isoformat") else lm))
        return out

    def details(self, repo_id: str) -> HubDetails:
        info = self.api.model_info(repo_id, files_metadata=True)
        files = [HubFile(s.rfilename, int(s.size or 0)) for s in (info.siblings or [])]
        config = None
        if any(f.name == "config.json" for f in files):
            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
            r = self.http.get(f"https://huggingface.co/{repo_id}/resolve/{info.sha or 'main'}/config.json", headers=headers)
            if r.status_code == 200:
                try:
                    config = r.json()
                except ValueError:
                    config = None
        return HubDetails(repo_id, files, config, getattr(info, "gated", False), getattr(info, "sha", None))
```

(`details` fetches `config.json` over HTTP rather than with `hf_hub_download`. That keeps half-downloaded repos out of the HF cache, where they would otherwise show up in the inventory as incomplete.)

- [ ] **Step 3: Implement `dgxmm/catalog/service.py`**

```python
from __future__ import annotations

import asyncio
import logging

from dgxmm.catalog.hub import fit_badge, guess_quant, quick_required
from dgxmm.catalog.llmfit import LlmfitError
from dgxmm.catalog.support import engine_support
from dgxmm.config import GIB
from dgxmm.inventory.archspec import parse_hf_config
from dgxmm.memory.estimate import VllmSizing, estimate_vllm
from dgxmm.planner.fit import check_fit
from dgxmm.planner.plans import base_model_name

log = logging.getLogger(__name__)


def _llmfit_summary(entry: dict | None, required: int | None) -> dict | None:
    if not entry:
        return None
    mem = entry.get("memory_required_gb") or entry.get("recommended_ram_gb") or entry.get("min_vram_gb")
    mismatch = bool(mem and required and abs(mem * 1e9 - required) / required > 0.20)
    return {"fit_level": entry.get("fit_level"), "score": entry.get("score"), "estimated_tps": entry.get("estimated_tps"),
            "measured_tps": entry.get("measured_tps"), "memory_required_gb": mem, "runtime": entry.get("runtime"),
            "best_quant": entry.get("best_quant"), "use_case": entry.get("use_case"), "mismatch": mismatch}


class CatalogService:
    def __init__(self, settings, hub, llmfit, inventory, budget, deployer, archs):
        self.settings, self.hub, self.llmfit, self.inventory = settings, hub, llmfit, inventory
        self.budget, self.deployer, self.archs = budget, deployer, archs

    async def _budget(self):
        return self.budget.current() or await self.budget.refresh()

    def _evictable(self) -> int:
        return sum(max(d.reserved, d.actual) for d in self.deployer.active_deps())

    def _llm_map(self, q: str, free: int, ctx: int) -> tuple[dict, str | None]:
        try:
            return self.llmfit.search_map(q, max(free, 0) / GIB, ctx), None
        except LlmfitError as e:
            return {}, f"llmfit unavailable: {e}"

    async def search(self, q: str, gguf: bool = False, ctx: int = 32768, limit: int = 30) -> dict:
        budget = await self._budget()
        models = await asyncio.to_thread(self.hub.search, q, limit, gguf)
        llm, banner = await asyncio.to_thread(self._llm_map, q, budget.free_for_new, ctx)
        archs, evictable = self.archs.get(), self._evictable()
        results = []
        for m in models:
            required = quick_required(m.est_bytes, self.settings) if m.est_bytes else None
            local = self.inventory.get(m.repo_id)
            arch = (local.architecture if local else None)
            results.append({**m.as_dict(), "downloaded": local is not None and local.complete,
                            "fit": fit_badge(required, budget, evictable) if required else {"badge": "unknown"},
                            "llmfit": _llmfit_summary(llm.get(m.repo_id.lower()), required),
                            "support": {"vllm": None if gguf else (arch in archs if (archs is not None and arch) else None),
                                        "llamacpp": m.has_gguf},
                            "quant": guess_quant(m.repo_id)})
        return {"results": results, "banner": banner, "free_for_new": budget.free_for_new}

    async def details(self, repo_id: str, ctx: int = 32768) -> dict:
        budget = await self._budget()
        d = await asyncio.to_thread(self.hub.details, repo_id)
        spec = None
        if d.config:
            try:
                spec = parse_hf_config(d.config)
            except Exception as e:
                log.warning("cannot parse config of %s: %s", repo_id, e)
        st = {f.name: f.size for f in d.files if f.name.endswith(".safetensors")}
        if any(k.startswith("model") for k in st):
            st = {k: v for k, v in st.items() if not k.startswith("consolidated")}
        weights = sum(st.values())
        out: dict = {"repo_id": repo_id, "gated": d.gated, "sha": d.sha,
                     "files": [{"name": f.name, "size": f.size} for f in d.files],
                     "architecture": spec.architecture if spec else None, "max_context": spec.max_context if spec else None,
                     "quant": spec.quant if spec else guess_quant(repo_id)}
        evictable = self._evictable()
        if spec and weights:
            est = estimate_vllm(spec, weights, VllmSizing(min(ctx, spec.max_context), 8, 8192, "auto", None), self.settings)
            out["estimate"] = est.as_dict()
            out["fit"] = {**fit_badge(est.total, budget, evictable), **check_fit(est, budget).as_dict()}
        out["gguf_files"] = [{"name": f.name, "size": f.size, "quant": guess_quant(f.name),
                              "fit": fit_badge(int(f.size * 1.1) + GIB, budget, evictable)}
                             for f in d.files if f.name.lower().endswith(".gguf") and "mmproj" not in f.name.lower()]
        archs = self.archs.get()
        out["support"] = {"vllm": engine_support(out["architecture"], False, archs)["vllm"] if weights else False,
                          "llamacpp": bool(out["gguf_files"])}
        llm, banner = await asyncio.to_thread(self._llm_map, repo_id, budget.free_for_new, ctx)
        out["llmfit"] = _llmfit_summary(llm.get(repo_id.lower()), out.get("estimate", {}).get("total"))
        out["banner"] = banner
        base = base_model_name(repo_id)
        try:
            variants = await asyncio.to_thread(self.hub.search, base, 20, False)
            out["variants"] = [{"repo_id": v.repo_id, "quant": guess_quant(v.repo_id), "est_bytes": v.est_bytes}
                               for v in variants if v.repo_id != repo_id and base_model_name(v.repo_id) == base]
        except Exception:
            out["variants"] = []
        out["downloaded"] = bool(self.inventory.get(repo_id))
        return out
```

- [ ] **Step 4: Run the tests** (after Task 27 creates `dgxmm/catalog/llmfit.py`)

Tasks 26 and 27 are implemented back to back; run `DGX$ .venv/bin/pytest -q tests/test_catalog.py tests/test_llmfit.py` once both exist. Expected: all pass. In `test_catalog_search_and_details`, the llmfit mismatch is True because llmfit says 50 GB and DGXMM's quick estimate is about 8.6 GiB.

---

### Task 27: llmfit integration

**Files:**
- Create: `dgxmm/catalog/llmfit.py`
- Modify: `dgxmm/context.py` (wiring for Tasks 25–27)
- Test: `tests/test_llmfit.py`

**Interfaces:**
- Produces: `LlmfitError`, `Llmfit(bin="llmfit", runner=subprocess.run, ttl=600, clock=time.monotonic, timeout=60)` with `search_map(query, free_gib, ctx) -> dict[str, dict]` and `recommend(free_gib, ctx, limit=50) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_llmfit.py`

```python
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
```

Run: `DGX$ .venv/bin/pytest -q tests/test_llmfit.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/catalog/llmfit.py`**

```python
"""llmfit (pip package `llmfit`) as a scoring/discovery source. Never the OOM gate."""
from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Callable


class LlmfitError(Exception):
    pass


class Llmfit:
    def __init__(self, bin: str = "llmfit", runner: Callable = subprocess.run, ttl: float = 600,
                 clock: Callable[[], float] = time.monotonic, timeout: float = 60):
        self.bin, self.runner, self.ttl, self.clock, self.timeout = bin, runner, ttl, clock, timeout
        self._cache: dict[tuple, tuple[float, Any]] = {}

    def _run(self, args: list[str]) -> Any:
        key = tuple(args)
        hit = self._cache.get(key)
        if hit and self.clock() - hit[0] < self.ttl:
            return hit[1]
        try:
            proc = self.runner([self.bin, "--json", *args], capture_output=True, text=True, timeout=self.timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise LlmfitError(str(e)) from e
        if proc.returncode != 0:
            raise LlmfitError((proc.stderr or "").strip()[:300] or f"exit code {proc.returncode}")
        try:
            data = json.loads(proc.stdout)
        except ValueError as e:
            raise LlmfitError("llmfit returned invalid JSON") from e
        self._cache[key] = (self.clock(), data)
        return data

    @staticmethod
    def _models(data: Any) -> list[dict]:
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict)]
        if isinstance(data, dict):
            for k in ("models", "results", "data"):
                if isinstance(data.get(k), list):
                    return [m for m in data[k] if isinstance(m, dict)]
        return []

    @staticmethod
    def _mem(free_gib: float, ctx: int) -> list[str]:
        g = f"{max(free_gib, 1):.0f}G"
        return ["--memory", g, "--ram", g, "--max-context", str(ctx)]

    def recommend(self, free_gib: float, ctx: int, limit: int = 50) -> list[dict]:
        return self._models(self._run(self._mem(free_gib, ctx) + ["recommend", "--limit", str(limit)]))

    def search_map(self, query: str, free_gib: float, ctx: int) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if query:
            for m in self._models(self._run(self._mem(free_gib, ctx) + ["search", query])):
                if m.get("name"):
                    out[m["name"].lower()] = dict(m)
        for m in self.recommend(free_gib, ctx):
            if m.get("name"):
                out.setdefault(m["name"].lower(), {}).update(m)
        return out
```

`Runner` in the test keys its canned outputs on the subcommand, which sits two positions after `--max-context`.

- [ ] **Step 3: Wire the catalog into `build_context`** (before `return ctx`)

```python
    from dgxmm.catalog.hub import HfHubClient
    from dgxmm.catalog.llmfit import Llmfit
    from dgxmm.catalog.service import CatalogService
    hf_token = settings.read_secret(settings.hf_token_file)
    hub = HfHubClient(token=hf_token)
    ctx.catalog = CatalogService(settings, hub, Llmfit(settings.llmfit_bin), inventory, budget, deployer, archs)
    ctx.hub = hub
```
Add `hub: Any = None` to `AppContext`.

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_llmfit.py tests/test_catalog.py`. Expected: all pass.

---

### Task 28: Download service

**Files:**
- Create: `dgxmm/downloads.py`
- Modify: `dgxmm/context.py` (wiring)
- Test: `tests/test_downloads.py`

**Interfaces:**
- Consumes: `HubDetails`, `Database`, `Download`, `InventoryService`, `EventBroker`, `REPO_RE`.
- Produces:
  - `DownloadError(code, message, status)`.
  - `DownloadService(settings, db, inventory, hub, events, downloader=None, disk_usage=shutil.disk_usage, token=None)` with:
    - `start(repo_id, revision=None, include=None, user_id=None) -> dict` (sync; call it in a thread)
    - `async launch(dl_id)`
    - `cancel(dl_id) -> dict`
    - `list(limit=50) -> list[dict]`
    - `view(dl_id) -> dict`

- [ ] **Step 1: Write the failing tests** — `tests/test_downloads.py`

```python
import asyncio
from collections import namedtuple
from pathlib import Path

import pytest

from dgxmm.catalog.hub import HubDetails, HubFile
from dgxmm.config import GIB
from dgxmm.downloads import DownloadError, DownloadService
from tests.conftest import QWEN_TINY

Usage = namedtuple("Usage", "total used free")


class Hub:
    def __init__(self, files):
        self.files = files

    def details(self, repo_id):
        return HubDetails(repo_id, self.files, QWEN_TINY, False, "abc")


def fake_downloader(fail_on=None, exc=None):
    calls = []

    def _dl(repo_id, filename, revision=None, cache_dir=None, token=None):
        calls.append(filename)
        if filename == fail_on:
            raise exc
        snap = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc"
        (snap.parent.parent / "refs").mkdir(parents=True, exist_ok=True)
        (snap.parent.parent / "refs" / "main").write_text("abc")
        snap.mkdir(parents=True, exist_ok=True)
        p = snap / filename
        if filename == "config.json":
            import json
            p.write_text(json.dumps(QWEN_TINY))
        else:
            p.write_bytes(b"x" * 100)
        return str(p)
    _dl.calls = calls
    return _dl


FILES = [HubFile("config.json", 100), HubFile("model.safetensors", 3 * GIB), HubFile("original/consolidated.pth", 9 * GIB),
         HubFile("README.md", 10), HubFile("model.Q4.gguf", GIB)]


def svc(stack, downloader, free=1000 * GIB, files=FILES):
    return DownloadService(stack.settings, stack.db, stack.inv, Hub(files), stack.events, downloader=downloader,
                           disk_usage=lambda p: Usage(0, 0, free))


async def wait_state(s, dl_id, states=("done", "failed", "cancelled")):
    for _ in range(300):
        v = s.view(dl_id)
        if v["state"] in states:
            return v
        await asyncio.sleep(0.01)
    return s.view(dl_id)


async def test_downloads_default_files_and_refreshes_inventory(stack):
    dl = fake_downloader()
    s = svc(stack, dl)
    v = s.start("org/new-model", user_id=1)
    assert v["bytes_total"] == 100 + 3 * GIB + 10 and v["files"] == ["config.json", "model.safetensors", "README.md"]
    await s.launch(v["id"])
    done = await wait_state(s, v["id"])
    assert done["state"] == "done" and dl.calls == ["config.json", "model.safetensors", "README.md"]
    assert stack.inv.get("org/new-model") is not None


async def test_refuses_low_disk_bad_input_and_gguf_without_choice(stack):
    s = svc(stack, fake_downloader(), free=40 * GIB)
    with pytest.raises(DownloadError) as e:
        s.start("org/new-model")
    assert e.value.code == "insufficient_disk"
    s = svc(stack, fake_downloader())
    for bad in ("../etc", "org/x;rm"):
        with pytest.raises(DownloadError):
            s.start(bad)
    with pytest.raises(DownloadError):
        s.start("org/x", include=["../../*"])
    gg = svc(stack, fake_downloader(), files=[HubFile("a.Q4_K_M.gguf", GIB), HubFile("a.Q8_0.gguf", 2 * GIB)])
    with pytest.raises(DownloadError) as e:
        gg.start("org/gguf-only")
    assert e.value.code == "choose_file"
    v = gg.start("org/gguf-only", include=["a.Q4_K_M.gguf"])
    assert v["files"] == ["a.Q4_K_M.gguf"]


async def test_permission_error_has_actionable_message(stack):
    err = PermissionError(13, "Permission denied", "/data/hf/hub/models--org--m/blobs/x")
    s = svc(stack, fake_downloader(fail_on="model.safetensors", exc=err))
    v = s.start("org/m")
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "failed" and "owned by root" in out["error"] and "chown" in out["error"]


async def test_cancel_between_files(stack):
    dl = fake_downloader()
    s = svc(stack, dl)
    v = s.start("org/c")
    s.cancel(v["id"])
    await s.launch(v["id"])
    out = await wait_state(s, v["id"])
    assert out["state"] == "cancelled" and dl.calls == []
```

Run: `DGX$ .venv/bin/pytest -q tests/test_downloads.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/downloads.py`**

```python
"""Background Hugging Face downloads with progress (spec §7)."""
from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
import shutil
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from dgxmm.db import Database, Download
from dgxmm.inventory.artifact import REPO_RE

log = logging.getLogger(__name__)
GIB = 1024**3
DEFAULT_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py", "*.tiktoken", "tokenizer*", "*.md"]
EXCLUDE = ["original/*", "*.pth", "*.pt", "*.bin", "*.onnx", "*.msgpack", "*.h5", "*.gguf"]
PATTERN_RE = r"^[A-Za-z0-9._*?\[\]/-]{1,200}$"


class DownloadError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def _default_downloader():
    from huggingface_hub import hf_hub_download
    return hf_hub_download


class DownloadService:
    def __init__(self, settings, db: Database, inventory, hub, events, downloader: Callable | None = None,
                 disk_usage: Callable = shutil.disk_usage, token: str | None = None):
        self.settings, self.db, self.inventory, self.hub, self.events = settings, db, inventory, hub, events
        self.downloader = downloader or _default_downloader()
        self.disk_usage, self.token = disk_usage, token
        self.hub_dir = Path(settings.hf_cache.local) / "hub"
        self._cancel: set[int] = set()
        self._tasks: dict[int, asyncio.Task] = {}
        self._sizes: dict[int, dict[str, int]] = {}

    def view(self, dl_id: int) -> dict:
        with self.db.session() as s:
            d = s.get(Download, dl_id)
            return {"id": d.id, "repo_id": d.repo_id, "revision": d.revision, "files": d.files, "state": d.state,
                    "bytes_done": d.bytes_done, "bytes_total": d.bytes_total, "error": d.error,
                    "created_at": d.created_at.isoformat()}

    def list(self, limit: int = 50) -> list[dict]:
        with self.db.session() as s:
            ids = [d.id for d in s.scalars(select(Download).order_by(Download.id.desc()).limit(limit))]
        return [self.view(i) for i in ids]

    def _update(self, dl_id: int, **fields) -> None:
        with self.db.session() as s:
            d = s.get(Download, dl_id)
            for k, v in fields.items():
                setattr(d, k, v)
        self.events.publish("download", self.view(dl_id))

    def start(self, repo_id: str, revision: str | None = None, include: list[str] | None = None,
              user_id: int | None = None) -> dict:
        if not re.fullmatch(REPO_RE, repo_id) or ".." in repo_id:
            raise DownloadError("invalid_repo", "repo id must look like org/name", 422)
        if revision is not None and not re.fullmatch(r"^[A-Za-z0-9._/-]{1,64}$", revision):
            raise DownloadError("invalid_revision", "invalid revision", 422)
        for p in include or []:
            if not re.fullmatch(PATTERN_RE, p) or ".." in p:
                raise DownloadError("invalid_pattern", f"invalid file pattern {p!r}", 422)
        with self.db.session() as s:
            if s.scalar(select(Download).where(Download.repo_id == repo_id, Download.state.in_(("queued", "running")))):
                raise DownloadError("already_downloading", f"{repo_id} is already downloading", 409)
        try:
            details = self.hub.details(repo_id)
        except Exception as e:
            name = type(e).__name__
            if name == "GatedRepoError":
                raise DownloadError("gated", "gated repo: accept the licence on huggingface.co and add an HF token", 403)
            if name in ("RepositoryNotFoundError", "RevisionNotFoundError"):
                raise DownloadError("not_found", f"{repo_id} not found on the Hub", 404)
            raise DownloadError("hub_error", f"could not reach the Hub: {e}", 502)
        if include:
            files = [f for f in details.files if any(fnmatch.fnmatch(f.name, p) for p in include)]
        else:
            files = [f for f in details.files if any(fnmatch.fnmatch(f.name, p) for p in DEFAULT_PATTERNS)
                     and not any(fnmatch.fnmatch(f.name, x) for x in EXCLUDE)]
            if not any(f.name.endswith(".safetensors") for f in files):
                if any(f.name.lower().endswith(".gguf") for f in details.files):
                    raise DownloadError("choose_file", "this repo contains GGUF files; choose which file to download", 422)
                raise DownloadError("no_weights", "no safetensors weights found in this repo", 422)
        if not files:
            raise DownloadError("no_files", "no files match the selection", 422)
        total = sum(f.size for f in files)
        free = self.disk_usage(str(self.hub_dir if self.hub_dir.exists() else self.hub_dir.parent)).free
        min_free = int(self.settings.min_free_disk_gib * GIB)
        if free - total < min_free:
            raise DownloadError("insufficient_disk",
                                f"needs {total / GIB:.1f} GiB; only {free / GIB:.1f} GiB free and "
                                f"{self.settings.min_free_disk_gib:.0f} GiB must stay free", 507)
        with self.db.session() as s:
            row = Download(repo_id=repo_id, revision=revision or details.sha, files=[f.name for f in files],
                           bytes_total=total, state="queued", created_by=user_id)
            s.add(row)
            s.flush()
            dl_id = row.id
        self._sizes[dl_id] = {f.name: f.size for f in files}
        return self.view(dl_id)

    async def launch(self, dl_id: int) -> None:
        self._tasks[dl_id] = asyncio.get_running_loop().create_task(self._run(dl_id))
        await asyncio.sleep(0)

    def cancel(self, dl_id: int) -> dict:
        self._cancel.add(dl_id)
        v = self.view(dl_id)
        if v["state"] == "queued":
            self._update(dl_id, state="cancelled")
        return self.view(dl_id)

    def _incomplete_bytes(self, repo_id: str) -> int:
        blobs = self.hub_dir / ("models--" + repo_id.replace("/", "--")) / "blobs"
        try:
            parts = list(blobs.glob("*.incomplete"))
            return max((p.stat().st_size for p in parts), default=0)
        except OSError:
            return 0

    async def _run(self, dl_id: int) -> None:
        v = self.view(dl_id)
        if v["state"] == "cancelled" or dl_id in self._cancel:
            self._update(dl_id, state="cancelled")
            return
        sizes = self._sizes.get(dl_id, {})
        self._update(dl_id, state="running")
        done = 0
        for name in v["files"]:
            if dl_id in self._cancel:
                self._update(dl_id, state="cancelled", bytes_done=done)
                return
            fut = asyncio.ensure_future(asyncio.to_thread(self.downloader, repo_id=v["repo_id"], filename=name,
                                                          revision=v["revision"], cache_dir=str(self.hub_dir),
                                                          token=self.token))
            while not fut.done():
                await asyncio.sleep(0.25)
                if not fut.done():
                    self._update(dl_id, bytes_done=done + self._incomplete_bytes(v["repo_id"]))
            try:
                fut.result()
            except PermissionError as e:
                path = getattr(e, "filename", "") or str(self.hub_dir)
                repo_dir = self.hub_dir / ("models--" + v["repo_id"].replace("/", "--"))
                self._update(dl_id, state="failed", bytes_done=done,
                             error=(f"permission denied writing {path}. The repo folder is probably owned by root "
                                    f"(created by a container). Fix on the DGX: sudo chown -R 1000:1000 "
                                    f"{repo_dir.as_posix().replace(self.settings.hf_cache.local, self.settings.hf_cache.host)}"))
                return
            except Exception as e:
                self._update(dl_id, state="failed", bytes_done=done, error=f"{type(e).__name__}: {e}")
                return
            done += sizes.get(name, 0)
            self._update(dl_id, bytes_done=done)
        self._update(dl_id, state="done", bytes_done=done)
        await asyncio.to_thread(self.inventory.refresh)
        self.events.publish("inventory", {"reason": "download", "repo_id": v["repo_id"]})
```

Progress is polled every 0.25 s while a file downloads. `bytes_done` = finished files + the size of the repo's newest `*.incomplete` blob.

- [ ] **Step 3: Wire it into `build_context`** (before `return ctx`)

```python
    from dgxmm.downloads import DownloadService
    ctx.downloads = DownloadService(settings, db, inventory, hub, events, token=hf_token)
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_downloads.py`. Expected: `4 passed`.

---

### Task 29: Co-location planner and favourites

**Files:**
- Create: `dgxmm/planner/colocate.py`
- Modify: `dgxmm/db.py` (add the `Favourite` table)
- Test: `tests/test_colocate.py`

**Interfaces:**
- Consumes: `Deployer.prepare`, `InventoryService.list`, `Budget`, engine defaults.
- Produces:
  - `Candidate(ref, engine, config, required, favourite=False)`.
  - `Combo(models, total, headroom)` with `.as_dict()`.
  - `default_config(engine_name, artifact) -> dict`.
  - `best_combos(cands, capacity, max_models=4, top=10, must_include=()) -> list[Combo]`.
  - `planner_view(deployer, inventory, budget, favourites, ctx=None, must_include=()) -> dict`.
  - `Favourite(ref)` ORM class.

- [ ] **Step 1: Add the table** — in `dgxmm/db.py`

```python
class Favourite(Base):
    __tablename__ = "favourites"
    ref: Mapped[str] = mapped_column(String(512), primary_key=True)
```

- [ ] **Step 2: Write the failing tests** — `tests/test_colocate.py`

```python
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
```

Run: `DGX$ .venv/bin/pytest -q tests/test_colocate.py`. Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `dgxmm/planner/colocate.py`**

```python
"""Which downloaded models can run together (spec §4.6)."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    ref: str
    engine: str
    config: dict = field(hash=False, compare=False)
    required: int = 0
    favourite: bool = False


@dataclass
class Combo:
    models: list[Candidate]
    total: int
    headroom: int

    def as_dict(self) -> dict:
        return {"models": [{"ref": m.ref, "engine": m.engine, "config": m.config, "required": m.required,
                            "favourite": m.favourite} for m in self.models],
                "total": self.total, "headroom": self.headroom}


def default_config(engine_name: str, artifact) -> dict:
    max_ctx = getattr(getattr(artifact, "arch", None), "max_context", None) or 32768
    if engine_name == "llamacpp":
        return {"ctx_size": min(32768, max_ctx), "parallel": 4}
    return {"max_model_len": min(32768, max_ctx), "max_num_seqs": 8, "max_num_batched_tokens": 8192}


def best_combos(cands: list[Candidate], capacity: int, max_models: int = 4, top: int = 10,
                must_include: tuple[str, ...] = ()) -> list[Combo]:
    pool = sorted((c for c in cands if 0 < c.required <= capacity), key=lambda c: c.required)
    combos: list[Combo] = []
    for k in range(min(max_models, len(pool)), 0, -1):
        for combo in itertools.combinations(pool, k):
            total = sum(c.required for c in combo)
            if total > capacity:
                continue
            refs = {c.ref for c in combo}
            if must_include and not set(must_include) <= refs:
                continue
            combos.append(Combo(list(combo), total, capacity - total))
    combos.sort(key=lambda x: (-len(x.models), -sum(m.favourite for m in x.models), -x.headroom))
    return combos[:top]


def planner_view(deployer, inventory, budget, favourites: set[str], ctx: int | None = None,
                 must_include: tuple[str, ...] = ()) -> dict:
    from dgxmm.deployer.service import DeployError
    cands: list[Candidate] = []
    for a in inventory.list():
        if not a.complete or a.kind != "llm":
            continue
        engine = next((n for n, e in deployer.engines.items() if e.supports(a)), None)
        if engine is None:
            continue
        cfg = default_config(engine, a)
        if ctx:
            key = deployer.engines[engine].context_key
            cfg[key] = min(ctx, cfg[key])
        try:
            prep = deployer.prepare(engine, a.ref, cfg)
        except DeployError:
            continue
        cands.append(Candidate(a.ref, engine, prep.cfg.model_dump(mode="json"), prep.estimate.total, a.ref in favourites))
    capacity = budget.usable - budget.external
    return {"capacity": capacity, "external": budget.external, "committed": budget.committed,
            "candidates": [{"ref": c.ref, "engine": c.engine, "required": c.required, "favourite": c.favourite,
                            "config": c.config} for c in sorted(cands, key=lambda c: c.required)],
            "combos": [x.as_dict() for x in best_combos(cands, capacity, must_include=must_include)]}
```

Ranking order: more models first, then more favourites, then more headroom. Ranking purely by headroom (as spec §4.6 literally says) would always put a single tiny model on top. Task 37 records this refinement in the spec.

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_colocate.py`. Expected: `3 passed`.

---

### Task 30: Catalog, downloads and planner API and UI

**Files:**
- Create: `dgxmm/api/catalog.py`, `dgxmm/api/downloads.py`, `dgxmm/api/planner.py`
- Modify: `dgxmm/app.py` (`ROUTERS`)
- Create: `frontend/src/pages/Catalog.tsx`, `frontend/src/pages/Downloads.tsx`, `frontend/src/pages/Planner.tsx`
- Modify: `frontend/src/main.tsx`
- Test: `tests/test_api_catalog.py`

**Interfaces:**
- Produces routes:
  - `GET /api/catalog/search?q=&gguf=&ctx=`, `GET /api/catalog/details?repo=&ctx=`
  - `GET /api/downloads`, `POST /api/downloads` (admin; `{repo_id, revision?, include?}`), `POST /api/downloads/{id}/cancel` (admin)
  - `GET /api/planner?ctx=&must=`, `PUT /api/planner/favourites/{ref:path}`, `DELETE /api/planner/favourites/{ref:path}`

- [ ] **Step 1: Write the failing tests** — `tests/test_api_catalog.py`

```python
from types import SimpleNamespace

from dgxmm.catalog.service import CatalogService
from dgxmm.downloads import DownloadService
from tests.test_api_deployments import admin
from tests.test_catalog import FakeHub, FakeLlmfit
from tests.test_downloads import FILES, Hub, Usage, fake_downloader
from dgxmm.config import GIB


def wire(c):
    st = c.stack
    c.ctx.catalog = CatalogService(st.settings, FakeHub(), FakeLlmfit(), st.inv, st.budget, st.deployer,
                                   SimpleNamespace(get=lambda: None))
    c.ctx.downloads = DownloadService(st.settings, st.db, st.inv, Hub(FILES), st.events, downloader=fake_downloader(),
                                      disk_usage=lambda p: Usage(0, 0, 1000 * GIB))


def test_catalog_routes(api_stack):
    admin(api_stack)
    wire(api_stack)
    r = api_stack.get("/dgxmm/api/catalog/search", params={"q": "qwen"}).json()
    assert r["results"][0]["repo_id"] == "Qwen/Qwen3-Tiny"
    d = api_stack.get("/dgxmm/api/catalog/details", params={"repo": "Qwen/Qwen3-Tiny"}).json()
    assert d["fit"]["badge"] == "fits_now"
    assert api_stack.get("/dgxmm/api/catalog/details", params={"repo": "../x"}).status_code == 422


def test_download_routes(api_stack):
    h = admin(api_stack)
    wire(api_stack)
    r = api_stack.post("/dgxmm/api/downloads", headers=h, json={"repo_id": "org/new-model"})
    assert r.status_code == 202, r.text
    assert api_stack.get("/dgxmm/api/downloads").json()[0]["repo_id"] == "org/new-model"
    bad = api_stack.post("/dgxmm/api/downloads", headers=h, json={"repo_id": "../../etc"})
    assert bad.status_code == 422 and bad.json()["detail"]["code"] == "invalid_repo"


def test_planner_and_favourites(api_stack):
    h = admin(api_stack)
    assert api_stack.put("/dgxmm/api/planner/favourites/Qwen/Qwen3-Tiny", headers=h).status_code == 204
    v = api_stack.get("/dgxmm/api/planner").json()
    assert next(x for x in v["candidates"] if x["ref"] == "Qwen/Qwen3-Tiny")["favourite"] is True
    assert api_stack.delete("/dgxmm/api/planner/favourites/Qwen/Qwen3-Tiny", headers=h).status_code == 204
```

Run: `DGX$ .venv/bin/pytest -q tests/test_api_catalog.py`. Expected: FAIL (404).

- [ ] **Step 2: Implement the routers**

`dgxmm/api/catalog.py`:
```python
import re

from fastapi import APIRouter, Depends, Query

from dgxmm.api.errors import api_error
from dgxmm.auth.deps import Principal, current_principal, get_ctx
from dgxmm.context import AppContext
from dgxmm.inventory.artifact import REPO_RE

router = APIRouter(prefix="/api/catalog", tags=["catalog"])


@router.get("/search")
async def search(q: str = Query("", max_length=100), gguf: bool = False, ctx: int = Query(32768, ge=512, le=1_048_576),
                 p: Principal = Depends(current_principal), c: AppContext = Depends(get_ctx)):
    try:
        return await c.catalog.search(q.strip(), gguf=gguf, ctx=ctx)
    except Exception as e:
        raise api_error(502, "hub_error", f"Hugging Face search failed: {e}")


@router.get("/details")
async def details(repo: str, ctx: int = Query(32768, ge=512, le=1_048_576),
                  p: Principal = Depends(current_principal), c: AppContext = Depends(get_ctx)):
    if not re.fullmatch(REPO_RE, repo) or ".." in repo:
        raise api_error(422, "invalid_repo", "repo id must look like org/name")
    try:
        return await c.catalog.details(repo, ctx=ctx)
    except Exception as e:
        raise api_error(502, "hub_error", f"could not load {repo}: {e}")
```

`dgxmm/api/downloads.py`:
```python
import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.downloads import DownloadError

router = APIRouter(prefix="/api/downloads", tags=["downloads"])


class DownloadIn(BaseModel):
    repo_id: str
    revision: str | None = None
    include: list[str] | None = None


@router.get("")
def list_downloads(p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return ctx.downloads.list()


@router.post("", status_code=202)
async def start(body: DownloadIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        view = await asyncio.to_thread(ctx.downloads.start, body.repo_id, body.revision, body.include, p.user.id)
    except DownloadError as e:
        raise api_error(e.status, e.code, e.message)
    await ctx.downloads.launch(view["id"])
    audit(ctx.db, p.user.id, "download", body.repo_id, {"files": view["files"], "bytes": view["bytes_total"]})
    return view


@router.post("/{dl_id}/cancel")
def cancel(dl_id: int, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    audit(ctx.db, p.user.id, "download_cancelled", str(dl_id))
    return ctx.downloads.cancel(dl_id)
```

`dgxmm/api/planner.py`:
```python
from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select

from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.db import Favourite
from dgxmm.planner.colocate import planner_view

router = APIRouter(prefix="/api/planner", tags=["planner"])


def _favs(ctx) -> set[str]:
    with ctx.db.session() as s:
        return {f.ref for f in s.scalars(select(Favourite))}


@router.get("")
async def planner(ctx_len: int | None = Query(None, alias="ctx", ge=512, le=1_048_576), must: list[str] = Query([]),
                  p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    budget = ctx.budget.current() or await ctx.budget.refresh()
    return planner_view(ctx.deployer, ctx.inventory, budget, _favs(ctx), ctx=ctx_len, must_include=tuple(must))


@router.put("/favourites/{ref:path}", status_code=204)
def add_fav(ref: str, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        if s.get(Favourite, ref) is None:
            s.add(Favourite(ref=ref[:512]))
    return Response(status_code=204)


@router.delete("/favourites/{ref:path}", status_code=204)
def del_fav(ref: str, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    with ctx.db.session() as s:
        f = s.get(Favourite, ref)
        if f:
            s.delete(f)
    return Response(status_code=204)
```

Add `catalog as catalog_api, downloads as downloads_api, planner as planner_api` to `ROUTERS` in `dgxmm/app.py`.

- [ ] **Step 3: Frontend pages**

`frontend/src/pages/Catalog.tsx`:
```tsx
import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Input, Modal, Table } from "../components/ui";
import { fmtBytes } from "../format";

const FIT_TONE: Record<string, "green" | "amber" | "red" | "slate"> = { fits_now: "green", fits_after_evict: "amber", too_big: "red", unknown: "slate" };
const FIT_LABEL: Record<string, string> = { fits_now: "fits now", fits_after_evict: "fits if you evict", too_big: "too big", unknown: "size unknown" };

export function Catalog() {
  const admin = useIsAdmin();
  const [q, setQ] = useState("");
  const [term, setTerm] = useState("qwen");
  const [gguf, setGguf] = useState(false);
  const [ctx, setCtx] = useState(32768);
  const [open, setOpen] = useState<string | null>(null);
  const res = useQuery({ queryKey: ["catalog", term, gguf, ctx], queryFn: () => api<any>(`/catalog/search?q=${encodeURIComponent(term)}&gguf=${gguf}&ctx=${ctx}`) });
  return (
    <div className="space-y-4">
      <Card title="Hugging Face catalog">
        <form className="flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); setTerm(q); }}>
          <div className="w-72"><Input placeholder="Search models (e.g. qwen3 nvfp4)" value={q} onChange={(e) => setQ(e.target.value)} /></div>
          <label className="text-sm"><input type="checkbox" checked={gguf} onChange={(e) => setGguf(e.target.checked)} /> GGUF (llama.cpp)</label>
          <label className="text-sm">Context <Input className="!w-28" type="number" value={ctx} onChange={(e) => setCtx(parseInt(e.target.value || "32768", 10))} /></label>
          <Button variant="primary">Search</Button>
        </form>
        {res.data?.banner && <p className="mt-2 text-xs text-amber-700">{res.data.banner}</p>}
        <ErrorBox error={res.error} />
      </Card>
      <Card title={res.data ? `${res.data.results.length} results · ${fmtBytes(res.data.free_for_new)} free now` : "Results"}>
        <Table head={["Model", "Downloads", "Approx size", "Fit", "Engines", "llmfit", ""]}>
          {(res.data?.results ?? []).map((m: any) => (
            <tr key={m.repo_id}>
              <td className="td break-all font-medium">{m.repo_id} {m.quant && <Badge>{m.quant}</Badge>} {m.gated && <Badge tone="amber">gated</Badge>} {m.downloaded && <Badge tone="green">on disk</Badge>}</td>
              <td className="td tabular-nums">{m.downloads.toLocaleString()}</td>
              <td className="td tabular-nums">{fmtBytes(m.est_bytes)}</td>
              <td className="td"><Badge tone={FIT_TONE[m.fit.badge]}>{FIT_LABEL[m.fit.badge]}</Badge></td>
              <td className="td text-xs">vLLM {m.support.vllm === null ? "?" : m.support.vllm ? "✓" : "✗"} · llama.cpp {m.support.llamacpp ? "✓" : "✗"}</td>
              <td className="td text-xs">{m.llmfit ? <>{m.llmfit.fit_level ?? ""} {m.llmfit.estimated_tps ? `~${m.llmfit.estimated_tps} tok/s` : ""}{m.llmfit.mismatch && <span className="text-amber-700"> ⚠</span>}</> : "—"}</td>
              <td className="td"><Button onClick={() => setOpen(m.repo_id)}>Details</Button></td>
            </tr>
          ))}
        </Table>
      </Card>
      {open && <DetailsModal repo={open} ctx={ctx} admin={admin} onClose={() => setOpen(null)} />}
    </div>
  );
}

function DetailsModal({ repo, ctx, admin, onClose }: { repo: string; ctx: number; admin: boolean; onClose: () => void }) {
  const d = useQuery({ queryKey: ["catalog-details", repo, ctx], queryFn: () => api<any>(`/catalog/details?repo=${encodeURIComponent(repo)}&ctx=${ctx}`) });
  const dl = useMutation({ mutationFn: (include?: string[]) => api("/downloads", { method: "POST", body: { repo_id: repo, include } }) });
  const x = d.data;
  return (
    <Modal open wide title={repo} onClose={onClose}>
      <ErrorBox error={d.error || dl.error} />
      {!x ? <p className="text-sm text-slate-500">Loading…</p> : (
        <div className="space-y-3 text-sm">
          <p>{x.architecture ?? "unknown architecture"} · {x.quant ?? "?"} · max context {x.max_context?.toLocaleString() ?? "?"} {x.gated && <Badge tone="amber">gated: needs an HF token</Badge>}</p>
          {x.estimate && (
            <p>
              vLLM estimate at {ctx.toLocaleString()} context × 8 sequences: <b>{fmtBytes(x.estimate.total)}</b>{" "}
              <Badge tone={FIT_TONE[x.fit.badge]}>{FIT_LABEL[x.fit.badge]}</Badge>
              {x.llmfit && <> · llmfit {x.llmfit.memory_required_gb ?? "?"} GB, {x.llmfit.estimated_tps ?? "?"} tok/s{x.llmfit.mismatch && <span className="text-amber-700"> (estimates differ by over 20%)</span>}</>}
            </p>
          )}
          <p>Engines: vLLM {x.support.vllm === null ? "unknown (registry not loaded yet)" : x.support.vllm ? "supported" : "not supported"} · llama.cpp {x.support.llamacpp ? "GGUF files available" : "no GGUF in this repo"}</p>
          {admin && x.estimate && !x.downloaded && <Button variant="primary" disabled={dl.isPending || dl.isSuccess} onClick={() => dl.mutate(undefined)}>{dl.isSuccess ? "Download queued" : "Download safetensors"}</Button>}
          {x.gguf_files.length > 0 && (
            <Table head={["GGUF file", "Size", "Quant", "Fit", ""]}>
              {x.gguf_files.map((f: any) => (
                <tr key={f.name}>
                  <td className="td break-all font-mono text-xs">{f.name}</td><td className="td">{fmtBytes(f.size)}</td><td className="td">{f.quant}</td>
                  <td className="td"><Badge tone={FIT_TONE[f.fit.badge]}>{FIT_LABEL[f.fit.badge]}</Badge></td>
                  <td className="td">{admin && <Button onClick={() => dl.mutate([f.name])}>Download</Button>}</td>
                </tr>
              ))}
            </Table>
          )}
          {x.variants.length > 0 && (
            <div><h4 className="font-semibold">Other variants</h4><ul className="list-disc pl-5">{x.variants.map((v: any) => <li key={v.repo_id}>{v.repo_id} {v.quant && <Badge>{v.quant}</Badge>} {v.est_bytes && fmtBytes(v.est_bytes)}</li>)}</ul></div>
          )}
        </div>
      )}
    </Modal>
  );
}
```

`frontend/src/pages/Downloads.tsx`:
```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Table } from "../components/ui";
import { fmtBytes } from "../format";

export function Downloads() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["downloads"], queryFn: () => api<any[]>("/downloads") });
  const cancel = useMutation({ mutationFn: (id: number) => api(`/downloads/${id}/cancel`, { method: "POST" }), onSettled: () => qc.invalidateQueries({ queryKey: ["downloads"] }) });
  return (
    <Card title="Downloads">
      <ErrorBox error={q.error || cancel.error} />
      <Table head={["Repo", "Files", "Progress", "State", ""]}>
        {(q.data ?? []).map((d) => {
          const pct = d.bytes_total ? Math.min(100, (d.bytes_done / d.bytes_total) * 100) : 0;
          return (
            <tr key={d.id}>
              <td className="td font-medium">{d.repo_id}</td>
              <td className="td text-xs">{d.files.length} files</td>
              <td className="td w-64">
                <div className="h-2 w-full rounded bg-slate-100"><div className="h-2 rounded bg-sky-500" style={{ width: `${pct}%` }} /></div>
                <div className="text-xs tabular-nums text-slate-500">{fmtBytes(d.bytes_done)} / {fmtBytes(d.bytes_total)}</div>
              </td>
              <td className="td"><Badge tone={d.state === "done" ? "green" : d.state === "failed" ? "red" : d.state === "running" ? "blue" : "slate"}>{d.state}</Badge>{d.error && <div className="mt-1 max-w-md text-xs text-rose-700">{d.error}</div>}</td>
              <td className="td">{admin && ["queued", "running"].includes(d.state) && <Button onClick={() => cancel.mutate(d.id)}>Cancel</Button>}</td>
            </tr>
          );
        })}
      </Table>
      <p className="mt-2 text-xs text-slate-500">Cancelling takes effect after the file currently downloading finishes. Interrupted downloads resume when restarted.</p>
    </Card>
  );
}
```

`frontend/src/pages/Planner.tsx`:
```tsx
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api";
import { useIsAdmin } from "../auth";
import { Badge, Button, Card, ErrorBox, Input, Table } from "../components/ui";
import { fmtBytes } from "../format";

export function Planner() {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const [ctx, setCtx] = useState(32768);
  const [must, setMust] = useState<string[]>([]);
  const q = useQuery({ queryKey: ["planner", ctx, must], queryFn: () => api<any>(`/planner?ctx=${ctx}${must.map((m) => `&must=${encodeURIComponent(m)}`).join("")}`) });
  const fav = useMutation({
    mutationFn: ({ ref, on }: { ref: string; on: boolean }) => api(`/planner/favourites/${ref}`, { method: on ? "PUT" : "DELETE" }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["planner"] }),
  });
  const v = q.data;
  return (
    <div className="space-y-4">
      <Card title="What can run together?">
        <div className="flex items-end gap-3 text-sm">
          <label>Context per model <Input className="!w-28" type="number" value={ctx} onChange={(e) => setCtx(parseInt(e.target.value || "32768", 10))} /></label>
          {v && <span className="text-slate-600">Capacity {fmtBytes(v.capacity)} (total minus margin, OS reserve and {fmtBytes(v.external)} external use; ignores what is deployed now)</span>}
        </div>
        <ErrorBox error={q.error} />
      </Card>
      <Card title="Best combinations">
        {(v?.combos ?? []).map((c: any, i: number) => (
          <div key={i} className="flex items-center justify-between border-b py-2 text-sm last:border-0">
            <div className="flex flex-wrap gap-2">{c.models.map((m: any) => <Badge key={m.ref} tone={m.favourite ? "violet" : "blue"}>{m.ref} · {fmtBytes(m.required)}</Badge>)}</div>
            <div className="tabular-nums text-slate-600">{fmtBytes(c.total)} · {fmtBytes(c.headroom)} spare</div>
          </div>
        ))}
        {v && !v.combos.length && <p className="text-sm text-slate-500">No downloaded model fits in the current capacity.</p>}
      </Card>
      <Card title="Candidates (default configuration)">
        <Table head={["Model", "Engine", "Needs", "Favourite", "Must include"]}>
          {(v?.candidates ?? []).map((c: any) => (
            <tr key={c.ref}>
              <td className="td">{c.ref}</td><td className="td">{c.engine}</td><td className="td tabular-nums">{fmtBytes(c.required)}</td>
              <td className="td">{admin ? <input type="checkbox" checked={c.favourite} onChange={(e) => fav.mutate({ ref: c.ref, on: e.target.checked })} /> : c.favourite ? "★" : ""}</td>
              <td className="td"><input type="checkbox" checked={must.includes(c.ref)} onChange={(e) => setMust(e.target.checked ? [...must, c.ref] : must.filter((m) => m !== c.ref))} /></td>
            </tr>
          ))}
        </Table>
        <Button className="mt-2" variant="ghost" onClick={() => setMust([])}>Clear “must include”</Button>
      </Card>
    </div>
  );
}
```

In `main.tsx`, import the three pages and add the routes:
```tsx
<Route path="/catalog" element={<Catalog />} />
<Route path="/downloads" element={<Downloads />} />
<Route path="/planner" element={<Planner />} />
```

- [ ] **Step 4: Run all tests and the frontend build**

`DGX$ .venv/bin/pytest -q` → all pass. `NPM$ sh -c "npx vitest run && npm run build"` → passes.

- [ ] **Step 5: On-box check of the catalog** (needs network access to huggingface.co from the DGX)

After `docker compose up -d --build`, open **Catalog** in the UI and search `qwen3 nvfp4`. Expected:
- Results with fit badges, `vLLM ✓` for Qwen3 architectures once the registry has loaded (first start runs the one-shot container, about 30 s), and llmfit columns filled in.
- Opening `Qwen/Qwen3-0.6B` shows its estimate and the "on disk" badge.
- If llmfit fails inside the container, the banner reads `llmfit unavailable: …` and everything else still works.


## Milestone M5: Metrics history and tuning advice

### Task 31: Metric store and scraper

**Files:**
- Create: `dgxmm/metrics/__init__.py` (empty), `dgxmm/metrics/store.py`, `dgxmm/metrics/scraper.py`
- Modify: `dgxmm/context.py` (wiring)
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `EngineMetrics`, `delta_buckets`, `histogram_quantile`, `MetricRollup`, `Deployer.list_views`, `Deployer.base_url`, `BudgetService`.
- Produces:
  - `NUMERIC_KEYS`, `HOST_KEYS`.
  - `MetricsStore(db, keep_points=720, clock=time.time)` with:
    - `add(dep_id, ts, metrics) -> dict`
    - `add_host(ts, budget, gpu) -> dict`
    - `series(key) -> list[dict]`
    - `rollup(now=None) -> int`
    - `history(key, hours) -> list[dict]`
    - Here `key` is a deployment id (`int`) or `"host"`.
  - `MetricsScraper(settings, deployer, http, store, events, budget, clock=time.time)` with `async scrape_once() -> dict[int, dict]` and `async run()`.

- [ ] **Step 1: Write the failing tests** — `tests/test_metrics.py`

```python
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from dgxmm.db import MetricRollup
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import EngineMetrics
from dgxmm.metrics.scraper import MetricsScraper
from dgxmm.metrics.store import MetricsStore
from tests.test_engines_vllm import VLLM_METRICS


def em(gen, prompt=0.0, ttft=None, hits=None, queries=None, running=1.0):
    m = EngineMetrics(gauges={"running": running, "waiting": 0.0, "kv_usage": 0.5},
                      counters={"generation_tokens": gen, "prompt_tokens": prompt})
    if ttft:
        m.histograms["ttft"] = ttft
    if hits is not None:
        m.counters.update(prefix_hits=hits, prefix_queries=queries)
    return m


def test_rates_quantiles_and_resets(db):
    s = MetricsStore(db)
    first = s.add(1, 100.0, em(1000, 500, ttft=[(0.1, 0), (1.0, 0), (float("inf"), 0)], hits=0, queries=0))
    assert first["gen_tps"] is None and first["running"] == 1.0
    second = s.add(1, 105.0, em(1500, 1000, ttft=[(0.1, 4), (1.0, 10), (float("inf"), 10)], hits=20, queries=40))
    assert second["gen_tps"] == 100.0 and second["prompt_tps"] == 100.0
    assert 0.1 < second["ttft_p50"] < 1.0 and second["prefix_hit_rate"] == 0.5
    reset = s.add(1, 110.0, em(10))  # engine restarted: counter went backwards
    assert reset["gen_tps"] is None
    assert len(s.series(1)) == 3


def test_rollup_history_and_prune(db):
    import time
    s = MetricsStore(db)
    base = int(time.time()) // 60 * 60 - 600  # ten minutes ago: inside the 30-day retention window
    s.add(1, base + 1, em(0))
    s.add(1, base + 31, em(300))
    s.add(1, base + 61, em(600))  # next minute: not complete yet when rolling at base+70
    assert s.rollup(now=base + 70) == 1
    [row] = s.history(1, hours=24 * 365 * 100)
    assert row["gen_tps"] == 10.0 and row["running"] == 1.0
    assert s.rollup(now=base + 70) == 0  # idempotent
    with db.session() as sess:
        sess.add(MetricRollup(deployment_id=1, minute=datetime.now(timezone.utc) - timedelta(days=31), data={}))
    s.rollup()
    with db.session() as sess:
        assert all(r.minute.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc) - timedelta(days=30)
                   for r in sess.scalars(select(MetricRollup)))


async def test_scraper_reads_ready_deployments(stack):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}), 1)
    await stack.deployer.wait_ready(d["id"], 5)
    stack.http.metrics[d["port"]] = VLLM_METRICS
    await stack.budget.refresh()
    q = stack.events.subscribe()
    t = [1000.0]
    sc = MetricsScraper(stack.settings, stack.deployer, stack.http.client(), MetricsStore(stack.db), stack.events,
                        stack.budget, clock=lambda: t[0])
    out = await sc.scrape_once()
    assert out[d["id"]]["running"] == 2.0 and out[d["id"]]["kv_usage"] == 0.25
    topics = [q.get_nowait()[0] for _ in range(q.qsize())]
    assert "metrics" in topics
    assert sc.store.series("host")[-1]["used"] > 0
```

Run: `DGX$ .venv/bin/pytest -q tests/test_metrics.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/metrics/store.py`**

```python
"""Live samples (ring buffer, ~1 h at 5 s) and 1-minute rollups in SQLite for 30 days (spec §8)."""
from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy import delete, select

from dgxmm.db import Database, MetricRollup, as_aware
from dgxmm.engines.base import EngineMetrics
from dgxmm.engines.prom import delta_buckets, histogram_quantile

NUMERIC_KEYS = ("running", "waiting", "kv_usage", "prompt_tps", "gen_tps", "ttft_p50", "ttft_p95", "itl_p50",
                "itl_p95", "e2e_p50", "prefix_hit_rate", "spec_accept_rate", "preemptions")
HOST_KEYS = ("used", "available", "free_for_new", "external", "committed", "gpu_util", "temp_c", "power_w")
RETENTION = timedelta(days=30)


def _delta(cur: EngineMetrics, prev: EngineMetrics, key: str) -> float | None:
    a, b = cur.counters.get(key), prev.counters.get(key)
    if a is None or b is None or a < b:
        return None
    return a - b


class MetricsStore:
    def __init__(self, db: Database, keep_points: int = 720, clock: Callable[[], float] = time.time):
        self.db, self.keep, self.clock = db, keep_points, clock
        self._rings: dict[int | str, deque] = {}
        self._prev: dict[int, tuple[float, EngineMetrics]] = {}
        self._rolled: dict[int | str, int] = {}

    def _ring(self, key: int | str) -> deque:
        return self._rings.setdefault(key, deque(maxlen=self.keep))

    def add(self, dep_id: int, ts: float, m: EngineMetrics) -> dict:
        s: dict = {k: None for k in NUMERIC_KEYS}
        s.update(ts=ts, running=m.gauges.get("running"), waiting=m.gauges.get("waiting"), kv_usage=m.gauges.get("kv_usage"))
        prev = self._prev.get(dep_id)
        if prev:
            pts, pm = prev
            dt = ts - pts
            if dt > 0:
                for out_key, counter in (("prompt_tps", "prompt_tokens"), ("gen_tps", "generation_tokens")):
                    d = _delta(m, pm, counter)
                    s[out_key] = d / dt if d is not None else None
            for hist in ("ttft", "itl", "e2e"):
                if hist in m.histograms:
                    db = delta_buckets(m.histograms[hist], pm.histograms.get(hist))
                    s[f"{hist}_p50"] = histogram_quantile(db, 0.5)
                    if hist != "e2e":
                        s[f"{hist}_p95"] = histogram_quantile(db, 0.95)
            for out_key, num, den in (("prefix_hit_rate", "prefix_hits", "prefix_queries"),
                                      ("spec_accept_rate", "spec_accepted", "spec_draft")):
                dn, dd = _delta(m, pm, num), _delta(m, pm, den)
                s[out_key] = dn / dd if dn is not None and dd else None
            s["preemptions"] = _delta(m, pm, "preemptions")
        self._prev[dep_id] = (ts, m)
        self._ring(dep_id).append(s)
        return s

    def add_host(self, ts: float, budget, gpu: dict | None) -> dict:
        gpu = gpu or {}
        s = {"ts": ts, "used": budget.total - budget.available, "available": budget.available,
             "free_for_new": budget.free_for_new, "external": budget.external, "committed": budget.committed,
             "gpu_util": gpu.get("util"), "temp_c": gpu.get("temp_c"), "power_w": gpu.get("power_w")}
        self._ring("host").append(s)
        return s

    def series(self, key: int | str) -> list[dict]:
        return list(self._rings.get(key, ()))

    def rollup(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        current_minute = int(now // 60) * 60
        written = 0
        with self.db.session() as sess:
            for key, ring in self._rings.items():
                last = self._rolled.get(key, 0)
                groups: dict[int, list[dict]] = {}
                for s in ring:
                    minute = int(s["ts"] // 60) * 60
                    if last <= minute < current_minute:
                        groups.setdefault(minute, []).append(s)
                keys = HOST_KEYS if key == "host" else NUMERIC_KEYS
                for minute, samples in sorted(groups.items()):
                    data = {}
                    for k in keys:
                        vals = [x[k] for x in samples if x.get(k) is not None]
                        data[k] = sum(vals) / len(vals) if vals else None
                    if key != "host":
                        waits = [x["waiting"] for x in samples if x.get("waiting") is not None]
                        data["waiting_max"] = max(waits) if waits else None
                        data["samples"] = len(samples)
                    sess.add(MetricRollup(deployment_id=None if key == "host" else int(key),
                                          minute=datetime.fromtimestamp(minute, timezone.utc), data=data))
                    written += 1
                self._rolled[key] = current_minute
            sess.execute(delete(MetricRollup).where(MetricRollup.minute < datetime.now(timezone.utc) - RETENTION))
        return written

    def history(self, key: int | str, hours: float) -> list[dict]:
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self.db.session() as sess:
            q = select(MetricRollup).where(MetricRollup.minute >= since).order_by(MetricRollup.minute)
            q = q.where(MetricRollup.deployment_id.is_(None)) if key == "host" else q.where(MetricRollup.deployment_id == int(key))
            return [{"ts": as_aware(r.minute).timestamp(), **r.data} for r in sess.scalars(q)]
```

In `test_rollup_history_and_prune` the rolled minute averages two samples. The first has `gen_tps=None` (no previous sample) and the second has `300/30 = 10.0`, so the mean of non-None values is `10.0`.

- [ ] **Step 3: Implement `dgxmm/metrics/scraper.py`**

```python
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import httpx

log = logging.getLogger(__name__)


class MetricsScraper:
    def __init__(self, settings, deployer, http: httpx.AsyncClient, store, events, budget,
                 clock: Callable[[], float] = time.time):
        self.settings, self.deployer, self.http, self.store = settings, deployer, http, store
        self.events, self.budget, self.clock = events, budget, clock

    async def scrape_once(self) -> dict[int, dict]:
        now = self.clock()
        out: dict[int, dict] = {}
        for v in self.deployer.list_views():
            if v["state"] != "ready" or not v["port"]:
                continue
            engine = self.deployer.engines.get(v["engine"])
            try:
                r = await self.http.get(self.deployer.base_url(v["name"], v["port"]) + "/metrics", timeout=5)
                r.raise_for_status()
                out[v["id"]] = self.store.add(v["id"], now, engine.parse_metrics(r.text))
            except Exception as e:
                log.debug("metrics scrape failed for %s: %s", v["name"], e)
        b = self.budget.current()
        if b is not None:
            sample = self.budget.last_sample
            self.store.add_host(now, b, sample.gpu if sample else {})
        if out:
            self.events.publish("metrics", out)
        return out

    async def run(self) -> None:
        last_rollup = self.clock()
        while True:
            try:
                await self.scrape_once()
                if self.clock() - last_rollup >= 60:
                    await asyncio.to_thread(self.store.rollup)
                    last_rollup = self.clock()
            except Exception:
                log.exception("metrics loop failed")
            await asyncio.sleep(self.settings.scrape_interval_s)
```

- [ ] **Step 4: Wire it into `build_context`** (before `return ctx`)

```python
    from dgxmm.metrics.scraper import MetricsScraper
    from dgxmm.metrics.store import MetricsStore
    store = MetricsStore(db)
    scraper = MetricsScraper(settings, deployer, http, store, events, budget)
    ctx.metrics = store
    ctx.background.append(scraper.run)
```

- [ ] **Step 5: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_metrics.py`. Expected: `3 passed`.

---

### Task 32: Memory calibration

**Files:**
- Create: `dgxmm/memory/calibration.py`
- Modify: `dgxmm/context.py` (wiring)
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `MemoryCalibration`, `MemoryEstimate.with_total`, `Deployer` (`calibration` attribute, `state_hooks`, `budget`, `_get`), `config_hash`.
- Produces:
  - `Calibration(db)` with:
    - `record(engine, repo, config_hash, estimated, measured)`
    - `record_oom(engine, repo, config_hash, estimated)`
    - `adjust(engine, repo, config_hash, est) -> MemoryEstimate`
    - `summary(engine, repo) -> list[dict]`
  - `CalibrationRecorder(deployer, calibration, delay_s=15.0)` with `async on_state(dep_id, frm, to)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_calibration.py`

```python
from dgxmm.config import GIB
from dgxmm.deployer.service import DeployRequest, config_hash
from dgxmm.memory.calibration import Calibration, CalibrationRecorder
from dgxmm.memory.estimate import MemoryEstimate


def est(total, margin=0):
    return MemoryEstimate(total - margin, 0, 0, 0, 0, 0, margin, "estimated")


def test_adjust_exact_ratio_and_oom(db):
    cal = Calibration(db)
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == 10 * GIB  # nothing recorded yet
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB)
    exact = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    assert exact.total == int(12 * GIB * 1.05) and exact.confidence == "measured"
    other = cal.adjust("vllm", "r", "h2", est(20 * GIB))
    assert other.total == int(20 * GIB * 1.2) and other.confidence == "calibrated"
    cal.record("vllm", "r2", "x", 10 * GIB, 30 * GIB)
    assert cal.adjust("vllm", "r2", "y", est(10 * GIB)).total == int(10 * GIB * 1.5)  # ratio clamped
    cal.record_oom("vllm", "r", "h1", 13 * GIB)
    after_oom = cal.adjust("vllm", "r", "h1", est(10 * GIB))
    assert after_oom.total == int(int(12 * GIB * 1.05) * 1.10) and "OOM" in after_oom.notes[-1]
    cal.record("vllm", "r", "h1", 10 * GIB, 12 * GIB)  # a later success clears the OOM penalty
    assert cal.adjust("vllm", "r", "h1", est(10 * GIB)).total == int(12 * GIB * 1.05)


async def test_recorder_measures_after_ready(stack):
    cal = Calibration(stack.db)
    stack.deployer.calibration = cal
    stack.deployer.state_hooks.append(CalibrationRecorder(stack.deployer, cal, delay_s=0).on_state)
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name="a", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config={"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}), 1)
    stack.probe.managed_actual[d["container_id"]] = 3 * GIB
    await stack.deployer.wait_ready(d["id"], 5)
    import asyncio
    await asyncio.sleep(0.05)
    rows = cal.summary("vllm", "Qwen/Qwen3-Tiny")
    assert rows and rows[0]["measured"] == 3 * GIB and rows[0]["config_hash"] == config_hash(d["config"])
```

Run: `DGX$ .venv/bin/pytest -q tests/test_calibration.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/memory/calibration.py`**

```python
"""Learn from measured memory: exact config → measured × 1.05, same model → mean ratio (spec §4.4)."""
from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select

from dgxmm.db import Database, MemoryCalibration
from dgxmm.memory.estimate import MemoryEstimate

log = logging.getLogger(__name__)
GIB = 1024**3


class Calibration:
    def __init__(self, db: Database):
        self.db = db

    def record(self, engine: str, repo: str, config_hash: str, estimated: int, measured: int) -> None:
        with self.db.session() as s:
            s.add(MemoryCalibration(engine=engine, repo=repo, config_hash=config_hash, estimated_bytes=estimated,
                                    measured_bytes=measured, oom=False))

    def record_oom(self, engine: str, repo: str, config_hash: str, estimated: int) -> None:
        with self.db.session() as s:
            s.add(MemoryCalibration(engine=engine, repo=repo, config_hash=config_hash, estimated_bytes=estimated,
                                    measured_bytes=0, oom=True))

    def _rows(self, engine: str, repo: str) -> list[MemoryCalibration]:
        with self.db.session() as s:
            return list(s.scalars(select(MemoryCalibration).where(MemoryCalibration.engine == engine,
                                                                  MemoryCalibration.repo == repo)
                                  .order_by(MemoryCalibration.id.desc())))

    def summary(self, engine: str, repo: str) -> list[dict]:
        return [{"config_hash": r.config_hash, "estimated": r.estimated_bytes, "measured": r.measured_bytes,
                 "oom": r.oom, "ts": r.ts.isoformat()} for r in self._rows(engine, repo)]

    def adjust(self, engine: str, repo: str, config_hash: str, est: MemoryEstimate) -> MemoryEstimate:
        rows = self._rows(engine, repo)
        if not rows:
            return est
        same = [r for r in rows if r.config_hash == config_hash]
        oom_pending = bool(same) and same[0].oom
        exact = next((r for r in same if not r.oom and r.measured_bytes > 0), None)
        if exact:
            est = est.with_total(int(exact.measured_bytes * 1.05), "measured",
                                 f"measured {exact.measured_bytes / GIB:.1f} GiB with this exact configuration")
        else:
            ratios = [r.measured_bytes / r.estimated_bytes for r in rows if not r.oom and r.estimated_bytes > 0][:5]
            if ratios:
                ratio = min(max(sum(ratios) / len(ratios), 0.9), 1.5)
                est = est.with_total(int(est.total * ratio), "calibrated",
                                     f"scaled ×{ratio:.2f} from {len(ratios)} measurement(s) of this model")
        if oom_pending:
            est = est.with_total(int(est.total * 1.10), est.confidence, "last attempt was OOM-killed; added 10%")
        return est


class CalibrationRecorder:
    """After a deployment becomes ready (and KV is allocated), store estimated vs measured memory."""

    def __init__(self, deployer, calibration: Calibration, delay_s: float = 15.0):
        self.deployer, self.calibration, self.delay_s = deployer, calibration, delay_s

    async def on_state(self, dep_id: int, frm: str, to: str) -> None:
        if to != "ready" or frm == "sleeping":
            return
        await asyncio.sleep(self.delay_s)
        dep = self.deployer._get(dep_id)
        if dep is None or dep.state != "ready":
            return
        budget = await self.deployer.budget.refresh()
        actual = next((u.actual for u in budget.deployments if u.deployment_id == dep_id), 0)
        if actual <= 0:
            return
        from dgxmm.deployer.service import config_hash
        raw = dep.estimate.get("total", dep.reserved_bytes) - dep.estimate.get("margin", 0) if dep.estimate else dep.reserved_bytes
        self.calibration.record(dep.engine, dep.model_ref, config_hash(dep.config), int(raw), int(actual))
        log.info("calibration: %s estimated %.1f GiB (formula) measured %.1f GiB", dep.name, raw / GIB, actual / GIB)
```

The recorder stores the *formula* estimate (total minus margin) as `estimated`, so ratios compare the raw formula against reality and do not compound earlier corrections.

In `test_adjust_exact_ratio_and_oom`, the `h2` ratio uses only the one non-OOM row recorded so far (12/10 = 1.2).

- [ ] **Step 3: Wire it into `build_context`** (before `return ctx`)

```python
    from dgxmm.memory.calibration import Calibration, CalibrationRecorder
    calibration = Calibration(db)
    deployer.calibration = calibration
    deployer.state_hooks.append(CalibrationRecorder(deployer, calibration).on_state)
```

- [ ] **Step 4: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_calibration.py tests/test_deployer.py`. Expected: all pass.

---

### Task 33: Advisor rules

**Files:**
- Create: `dgxmm/advisor/__init__.py` (empty), `dgxmm/advisor/rules.py`
- Test: `tests/test_advisor.py`

**Interfaces:**
- Consumes: `ModelArtifact`, `base_model_name`, `config_hash`, metric samples (dicts with `NUMERIC_KEYS`), benchmark rows (dicts).
- Produces:
  - `AdviceInput(engine, cfg, artifact, samples, day, variants, drafts, benchmarks, estimate, reserved)`.
  - `Advice(id, title, rationale, expected_effect, confidence, patch, severity, model_ref)` with `.as_dict()`.
  - `RULES` (15 functions), `advise(inp) -> list[Advice]`.
  - `AdvisorService(deployer, inventory, store, db)` with `advise(dep_id) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_advisor.py`

```python
from types import SimpleNamespace

import pytest

from dgxmm.advisor.rules import AdviceInput, advise
from dgxmm.config import GIB
from dgxmm.deployer.service import config_hash

VCFG = {"max_model_len": 65536, "max_num_seqs": 8, "max_num_batched_tokens": 8192, "kv_cache_dtype": "auto",
        "enable_prefix_caching": True, "enable_chunked_prefill": True, "speculative": None, "kv_tokens": None}


def art(ref="Qwen/Qwen3-8B", quant="bf16", moe=False, files=(), layers=36, source="hf"):
    arch = SimpleNamespace(is_moe=moe, num_layers=layers, max_context=131072)
    return SimpleNamespace(ref=ref, quant=quant, arch=arch, files=list(files), source=source, kind="llm", config={})


def samples(n=200, **kw):
    base = {"running": 1.0, "waiting": 0.0, "kv_usage": 0.5, "ttft_p50": 0.2, "spec_accept_rate": None,
            "preemptions": 0.0, "prefix_hit_rate": 0.3}
    return [{**base, **kw} for _ in range(n)]


def ids(inp):
    return {a.id for a in advise(inp)}


def inp(**kw):
    d = dict(engine="vllm", cfg=dict(VCFG), artifact=art(), samples=samples(), day=samples(300), variants=[], drafts=[],
             benchmarks=[], estimate={"kv_cache": 20 * GIB, "weights": 16 * GIB}, reserved=40 * GIB)
    d.update(kw)
    return AdviceInput(**d)


def test_quant_kv_and_spec_rules():
    nv = art("nvidia/Qwen3-8B-NVFP4", "nvfp4")
    draft = art("org/Qwen3-8B-EAGLE3", "bf16")
    got = advise(inp(variants=[nv], drafts=[draft]))
    by = {a.id: a for a in got}
    assert by["nvfp4_variant"].model_ref == "nvidia/Qwen3-8B-NVFP4"
    assert by["fp8_kv"].patch == {"kv_cache_dtype": "fp8"}
    assert by["spec_decode"].patch["speculative"] == {"method": "eagle3", "model": "org/Qwen3-8B-EAGLE3", "num_speculative_tokens": 3}
    assert "spec_decode" in ids(inp(artifact=art(files=["model_mtp.safetensors"])))


def test_load_driven_rules():
    busy = samples(kv_usage=0.4, waiting=3.0, running=8.0)
    got_busy = {a.id: a for a in advise(inp(samples=busy))}
    assert got_busy["raise_seqs"].patch == {"max_num_seqs": 16} and "kv_pressure" not in got_busy
    hot = samples(kv_usage=0.99, preemptions=2.0, running=8.0)
    a = {x.id: x for x in advise(inp(samples=hot))}["kv_pressure"]
    assert a.severity == "warning" and a.patch == {"kv_cache_dtype": "fp8"}
    idle = samples(300, kv_usage=0.1, prefix_hit_rate=0.0)
    got = ids(inp(day=idle))
    assert {"shrink_reservation", "prefix_useless"} <= got


def test_spec_acceptance_low_and_moe_and_prefill():
    cfg = {**VCFG, "speculative": {"method": "eagle3", "model": "x/y", "num_speculative_tokens": 4}}
    a = {x.id: x for x in advise(inp(cfg=cfg, samples=samples(spec_accept_rate=0.3)))}["spec_accept_low"]
    assert a.patch == {"speculative": {"method": "eagle3", "model": "x/y", "num_speculative_tokens": 2}}
    assert "moe_batch" in ids(inp(artifact=art(moe=True)))
    assert "chunked_prefill_off" in ids(inp(cfg={**VCFG, "enable_chunked_prefill": False}))
    assert "batched_tokens_small" in ids(inp(cfg={**VCFG, "max_num_batched_tokens": 2048}, samples=samples(ttft_p50=2.0)))


def test_llamacpp_rules_and_engine_hint():
    lcfg = {"ctx_size": 65536, "parallel": 4, "n_gpu_layers": 20, "cache_type_k": "f16", "cache_type_v": "f16",
            "flash_attn": "off", "n_batch": 2048, "n_ubatch": 512}
    got = ids(inp(engine="llamacpp", cfg=lcfg, artifact=art(source="gguf", quant="gguf:Q4_K_M")))
    assert {"llamacpp_offload_all", "llamacpp_flash_attn", "llamacpp_quant_kv"} <= got
    gguf = art("/data/gguf/0/Qwen3-8B-Q4_K_M.gguf", "gguf:Q4_K_M", source="gguf")
    assert "try_llamacpp" in ids(inp(samples=samples(running=1.0), variants=[gguf]))


def test_measured_benchmarks_override_rule_estimate():
    patched = {**VCFG, "kv_cache_dtype": "fp8"}
    rows = [{"config_hash": config_hash(VCFG), "concurrency": 1, "tps_per_req": 20.0},
            {"config_hash": config_hash(patched), "concurrency": 1, "tps_per_req": 31.0}]
    a = {x.id: x for x in advise(inp(benchmarks=rows))}["fp8_kv"]
    assert a.confidence == "measured" and "31" in a.expected_effect and "20" in a.expected_effect
```

Run: `DGX$ .venv/bin/pytest -q tests/test_advisor.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/advisor/rules.py`**

```python
"""Tuning advice (spec §9.1). Each rule: AdviceInput -> Advice | None. Pure functions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from dgxmm.planner.plans import base_model_name

GIB = 1024**3


@dataclass
class AdviceInput:
    engine: str
    cfg: dict
    artifact: Any
    samples: list[dict]          # last ~15 min of live samples
    day: list[dict]              # last 24 h of minute rollups
    variants: list[Any]          # other local artifacts of the same base model
    drafts: list[Any]            # local draft models for this base model
    benchmarks: list[dict]       # benchmark rows for this model_ref
    estimate: dict
    reserved: int


@dataclass
class Advice:
    id: str
    title: str
    rationale: str
    expected_effect: str
    confidence: str = "rule"     # rule | estimated | measured
    patch: dict | None = None
    severity: str = "suggestion"  # warning | suggestion | info
    model_ref: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _max(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def _frac(rows: list[dict], pred: Callable[[dict], bool]) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


# ---- vLLM rules ----
def nvfp4_variant(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.artifact.quant not in ("bf16", "fp16"):
        return None
    v = next((v for v in i.variants if v.quant == "nvfp4"), None)
    if not v:
        return None
    return Advice("nvfp4_variant", f"Serve {v.ref} (NVFP4) instead",
                  "GB10 (Blackwell) has native FP4 tensor cores; an NVFP4 variant of this model is already on disk.",
                  "about 3.5× less weight memory and roughly 2× faster decode", model_ref=v.ref)


def fp8_kv(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.cfg.get("kv_cache_dtype") != "auto" or i.cfg.get("max_model_len", 0) <= 32768:
        return None
    frees = i.estimate.get("kv_cache", 0) / 2
    return Advice("fp8_kv", "Use an fp8 KV cache", "Long context makes the KV cache the biggest memory item.",
                  f"halves KV memory (frees about {frees / GIB:.1f} GiB) with negligible quality loss",
                  patch={"kv_cache_dtype": "fp8"})


def _spec_method(ref: str) -> tuple[str, int]:
    r = ref.lower()
    if "dflash" in r:
        return "dflash", 7
    if "eagle3" in r:
        return "eagle3", 3
    if "eagle" in r:
        return "eagle", 3
    return "draft_model", 3


def spec_decode(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.cfg.get("speculative"):
        return None
    mean_running = _mean(i.samples, "running")
    if mean_running is not None and mean_running > 2:
        return None
    if i.drafts:
        d = i.drafts[0]
        method, n = _spec_method(d.ref)
        spec = {"method": method, "model": d.ref, "num_speculative_tokens": n}
    elif any("mtp" in f.lower() for f in i.artifact.files) or any(k in (i.artifact.config or {}) for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers")):
        spec = {"method": "mtp", "num_speculative_tokens": 2}
    else:
        return None
    return Advice("spec_decode", f"Enable speculative decoding ({spec['method']})",
                  "Traffic is mostly one or two requests at a time, where decode is memory-bandwidth bound.",
                  "typically 1.5–2.5× faster single-stream decode", patch={"speculative": spec})


def spec_accept_low(i: AdviceInput) -> Advice | None:
    spec = i.cfg.get("speculative")
    rate = _mean(i.samples, "spec_accept_rate")
    if i.engine != "vllm" or not spec or rate is None or rate >= 0.5:
        return None
    n = int(spec.get("num_speculative_tokens", 1))
    patch = {"speculative": None} if n <= 1 else {"speculative": {**spec, "num_speculative_tokens": max(1, n // 2)}}
    return Advice("spec_accept_low", "Speculative decoding is rarely accepted",
                  f"Only {rate:.0%} of draft tokens are accepted, so drafting costs more than it saves.",
                  "fewer wasted draft tokens; higher throughput", patch=patch)


def raise_seqs(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or len(i.samples) < 12:
        return None
    queued = _frac(i.samples, lambda r: (r.get("waiting") or 0) > 0)
    kv = _mean(i.samples, "kv_usage")
    if queued <= 0.5 or kv is None or kv >= 0.7:
        return None
    new = min(int(i.cfg.get("max_num_seqs", 8)) * 2, 256)
    return Advice("raise_seqs", f"Allow {new} concurrent sequences",
                  f"Requests are queued {queued:.0%} of the time while the KV cache is only {kv:.0%} used.",
                  "shorter queue and higher total tok/s", patch={"max_num_seqs": new})


def kv_pressure(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm":
        return None
    peak = _max(i.samples, "kv_usage")
    pre = sum(r.get("preemptions") or 0 for r in i.samples)
    if peak is None or peak <= 0.95 or pre <= 0:
        return None
    if i.cfg.get("kv_cache_dtype") == "auto":
        patch = {"kv_cache_dtype": "fp8"}
    else:
        patch = {"max_num_seqs": max(1, int(i.cfg.get("max_num_seqs", 8)) // 2)}
    return Advice("kv_pressure", "KV cache is full and requests are being preempted",
                  f"Peak KV usage {peak:.0%} with {pre:.0f} preemptions recently.",
                  "stops preemption stalls", patch=patch, severity="warning")


def shrink_reservation(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or len(i.day) < 60:
        return None
    peak = _max(i.day, "kv_usage")
    kv = i.estimate.get("kv_cache", 0)
    if peak is None or peak >= 0.3 or kv < 4 * GIB:
        return None
    pool = i.cfg.get("kv_tokens") or i.cfg["max_model_len"] * i.cfg.get("max_num_seqs", 1)
    new_pool = max(int(i.cfg["max_model_len"]), int(pool / 2))
    return Advice("shrink_reservation", "Reserve a smaller KV pool",
                  f"The KV cache never went above {peak:.0%} in the last 24 hours.",
                  f"frees about {kv / 2 / GIB:.1f} GiB so another model can run alongside",
                  patch={"kv_tokens": new_pool})


def prefix_useless(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or not i.cfg.get("enable_prefix_caching") or len(i.day) < 60:
        return None
    hit = _mean(i.day, "prefix_hit_rate")
    if hit is None or hit >= 0.02:
        return None
    return Advice("prefix_useless", "Prefix caching is not helping this workload",
                  f"Prefix cache hit rate was {hit:.1%} over the last 24 hours.",
                  "no change needed; informational", severity="info")


def moe_batch(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or not getattr(i.artifact.arch, "is_moe", False) or int(i.cfg.get("max_num_seqs", 8)) >= 16:
        return None
    return Advice("moe_batch", "Batch more requests for this MoE model",
                  "Mixture-of-experts models activate few parameters per token, so larger batches are cheap.",
                  "higher total tok/s under load",
                  patch={"max_num_seqs": 16, "max_num_batched_tokens": max(16384, int(i.cfg.get("max_num_batched_tokens", 8192)))})


def try_llamacpp(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or len(i.samples) < 30:
        return None
    running = _mean(i.samples, "running")
    g = next((v for v in i.variants if v.source == "gguf"), None)
    if running is None or running > 1.2 or g is None:
        return None
    return Advice("try_llamacpp", "Benchmark the llama.cpp GGUF variant",
                  "This deployment almost always serves a single request; llama.cpp is often faster for one stream.",
                  f"possibly higher single-user tok/s with {g.ref.split('/')[-1]}", severity="info", model_ref=g.ref)


def batched_tokens_small(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or int(i.cfg.get("max_num_batched_tokens", 8192)) >= 8192:
        return None
    ttft = _mean(i.samples, "ttft_p50")
    if ttft is None or ttft <= 1.0:
        return None
    return Advice("batched_tokens_small", "Raise max batched tokens to 8192",
                  f"Median time-to-first-token is {ttft:.1f}s and prompts are prefilled in small chunks.",
                  "faster prefill / lower TTFT", patch={"max_num_batched_tokens": 8192})


def chunked_prefill_off(i: AdviceInput) -> Advice | None:
    if i.engine != "vllm" or i.cfg.get("enable_chunked_prefill", True):
        return None
    return Advice("chunked_prefill_off", "Enable chunked prefill",
                  "Without it, one long prompt blocks decoding for every other request.",
                  "smoother latency with mixed prompt lengths", patch={"enable_chunked_prefill": True})


# ---- llama.cpp rules ----
def llamacpp_offload_all(i: AdviceInput) -> Advice | None:
    layers = getattr(i.artifact.arch, "num_layers", 0)
    if i.engine != "llamacpp" or int(i.cfg.get("n_gpu_layers", 999)) > layers:
        return None
    return Advice("llamacpp_offload_all", "Offload all layers to the GPU",
                  "GB10 memory is unified, so keeping layers on the CPU saves no memory and runs slower.",
                  "large decode speed-up", patch={"n_gpu_layers": 999})


def llamacpp_flash_attn(i: AdviceInput) -> Advice | None:
    if i.engine != "llamacpp" or i.cfg.get("flash_attn") != "off":
        return None
    return Advice("llamacpp_flash_attn", "Turn on flash attention", "Flash attention is faster and enables a quantized KV cache.",
                  "faster attention, less memory", patch={"flash_attn": "on"})


def llamacpp_quant_kv(i: AdviceInput) -> Advice | None:
    if i.engine != "llamacpp" or i.cfg.get("cache_type_k") != "f16" or int(i.cfg.get("ctx_size", 0)) <= 32768:
        return None
    return Advice("llamacpp_quant_kv", "Use a q8_0 KV cache", "Long context makes the f16 KV cache large.",
                  "about half the KV memory", patch={"cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on"})


RULES: list[Callable[[AdviceInput], Advice | None]] = [
    nvfp4_variant, fp8_kv, spec_decode, spec_accept_low, raise_seqs, kv_pressure, shrink_reservation, prefix_useless,
    moe_batch, try_llamacpp, batched_tokens_small, chunked_prefill_off, llamacpp_offload_all, llamacpp_flash_attn,
    llamacpp_quant_kv,
]
_SEVERITY = {"warning": 0, "suggestion": 1, "info": 2}


def _with_measurements(a: Advice, i: AdviceInput) -> Advice:
    if not a.patch or not i.benchmarks:
        return a
    from dgxmm.deployer.service import config_hash
    cur_h, new_h = config_hash(i.cfg), config_hash({**i.cfg, **a.patch})

    def tps(h):
        rows = [b for b in i.benchmarks if b["config_hash"] == h and b["concurrency"] == 1]
        return rows[0]["tps_per_req"] if rows else None

    cur, new = tps(cur_h), tps(new_h)
    if cur is not None and new is not None:
        a.expected_effect = f"measured on this box: {new:.0f} vs {cur:.0f} tok/s at concurrency 1"
        a.confidence = "measured"
    return a


def advise(i: AdviceInput) -> list[Advice]:
    out = []
    for rule in RULES:
        try:
            a = rule(i)
        except (KeyError, TypeError, ValueError):
            a = None
        if a:
            out.append(_with_measurements(a, i))
    return sorted(out, key=lambda a: _SEVERITY[a.severity])


class AdvisorService:
    def __init__(self, deployer, inventory, store, db):
        self.deployer, self.inventory, self.store, self.db = deployer, inventory, store, db

    def advise(self, dep_id: int) -> list[dict]:
        from sqlalchemy import select
        from dgxmm.db import Benchmark
        v = self.deployer.view(dep_id)
        artifact = self.inventory.get(v["model_ref"])
        if artifact is None:
            return []
        base = base_model_name(artifact.ref)
        others = [a for a in self.inventory.list() if a.ref != artifact.ref and a.complete]
        variants = [a for a in others if a.kind == "llm" and base_model_name(a.ref) == base]
        drafts = [a for a in others if a.kind == "draft" and base_model_name(a.ref).startswith(base)]
        with self.db.session() as s:
            rows = s.scalars(select(Benchmark).where(Benchmark.model_ref == v["model_ref"]).order_by(Benchmark.id.desc()))
            benches = [{"config_hash": b.config_hash, "concurrency": b.concurrency, "tps_per_req": b.tps_per_req,
                        "tps_total": b.tps_total} for b in rows]
        inp = AdviceInput(engine=v["engine"], cfg=v["config"], artifact=artifact, samples=self.store.series(dep_id)[-180:],
                          day=self.store.history(dep_id, 24), variants=variants, drafts=drafts, benchmarks=benches,
                          estimate=v["estimate"] or {}, reserved=v["reserved_bytes"])
        return [a.as_dict() for a in advise(inp)]
```

`test_load_driven_rules` uses `samples(kv_usage=0.4, waiting=3.0, running=8.0)`. That fires `raise_seqs` (queued 100% of the time, KV 40%) and not `kv_pressure` (peak below 0.95).

- [ ] **Step 3: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_advisor.py`. Expected: all pass.

---

### Task 34: Benchmark runner

**Files:**
- Create: `dgxmm/advisor/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `Benchmark` ORM, `config_hash`, `httpx.AsyncClient`, `EventBroker`.
- Produces: `BenchmarkBusy(Exception)`, `ReqStat`, `percentile(values, p)`, `BenchmarkRunner(db, events, http_factory=None, clock=time.perf_counter)` with `busy() -> int | None` and `async run(...) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** — `tests/test_benchmark.py`

```python
import asyncio
import json

import httpx
import pytest
from sqlalchemy import select

from dgxmm.advisor.benchmark import BenchmarkBusy, BenchmarkRunner, percentile
from dgxmm.db import Benchmark
from dgxmm.events import EventBroker


def sse_handler(seen):
    def handler(req: httpx.Request):
        body = json.loads(req.content)
        seen.append(body)
        chunks = "".join(f"data: {json.dumps({'choices': [{'text': 'x'}]})}\n\n" for _ in range(5))
        chunks += f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 5}})}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, content=chunks.encode(), headers={"content-type": "text/event-stream"})
    return handler


def runner(db, seen, events=None):
    return BenchmarkRunner(db, events or EventBroker(),
                           http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(sse_handler(seen))))


async def test_runs_levels_and_saves_rows(db):
    seen = []
    r = runner(db, seen)
    rows = await r.run(deployment_id=1, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config={"a": 1},
                       base_url="http://h:1", served_name="q", levels=(1, 2), input_len=16, output_len=5,
                       duration_s=5, max_requests=4)
    assert [x["concurrency"] for x in rows] == [1, 2] and all(x["requests"] == 4 for x in rows)
    assert all(x["tps_total"] > 0 and x["ttft_p50"] is not None for x in rows)
    assert seen[0]["stream"] is True and seen[0]["ignore_eos"] is True and seen[0]["max_tokens"] == 5
    assert seen[0]["prompt"] != seen[1]["prompt"]  # nonce defeats the prefix cache
    with db.session() as s:
        assert len(s.scalars(select(Benchmark)).all()) == 2


async def test_only_one_benchmark_at_a_time(db):
    r = runner(db, [])
    kw = dict(deployment_id=1, engine="vllm", model_ref="m/m", config={}, base_url="http://h:1", served_name="q",
              levels=(1,), input_len=8, output_len=5, duration_s=1, max_requests=50)
    first = asyncio.create_task(r.run(**kw))
    await asyncio.sleep(0)
    with pytest.raises(BenchmarkBusy):
        await r.run(**kw)
    await first


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
```

Run: `DGX$ .venv/bin/pytest -q tests/test_benchmark.py`. Expected: FAIL (ImportError).

- [ ] **Step 2: Implement `dgxmm/advisor/benchmark.py`**

```python
"""Small load test against a deployment's OpenAI endpoint (spec §9.2)."""
from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

from dgxmm.db import Benchmark, Database

WORDS = ("memory token cache batch speed latency model server context window prompt decode prefill expert layer "
         "tensor kernel stream request response quality").split()


class BenchmarkBusy(Exception):
    pass


@dataclass
class ReqStat:
    ok: bool
    ttft: float | None = None
    itls: list[float] = field(default_factory=list)
    tokens: int = 0
    e2e: float = 0.0


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class BenchmarkRunner:
    def __init__(self, db: Database, events, http_factory: Callable[[], httpx.AsyncClient] | None = None,
                 clock: Callable[[], float] = time.perf_counter):
        self.db, self.events, self.clock = db, events, clock
        self.http_factory = http_factory or (lambda: httpx.AsyncClient(timeout=httpx.Timeout(600, connect=10)))
        self._lock = asyncio.Lock()
        self._running: int | None = None

    def busy(self) -> int | None:
        return self._running

    async def _one(self, http: httpx.AsyncClient, base_url: str, model: str, input_len: int, output_len: int) -> ReqStat:
        prompt = f"[{uuid.uuid4().hex}] " + " ".join(random.choice(WORDS) for _ in range(input_len))
        body = {"model": model, "prompt": prompt, "max_tokens": output_len, "stream": True, "ignore_eos": True,
                "temperature": 0, "stream_options": {"include_usage": True}}
        t0 = self.clock()
        ttft, last, itls, chunks, usage = None, None, [], 0, None
        try:
            async with http.stream("POST", f"{base_url}/v1/completions", json=body) as r:
                if r.status_code != 200:
                    return ReqStat(ok=False, e2e=self.clock() - t0)
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    if obj.get("usage"):
                        usage = obj["usage"].get("completion_tokens")
                    choices = obj.get("choices") or []
                    if choices and choices[0].get("text"):
                        now = self.clock()
                        chunks += 1
                        if ttft is None:
                            ttft = now - t0
                        else:
                            itls.append(now - last)
                        last = now
        except (httpx.HTTPError, ValueError):
            return ReqStat(ok=False, e2e=self.clock() - t0)
        return ReqStat(ok=True, ttft=ttft, itls=itls, tokens=int(usage or chunks), e2e=self.clock() - t0)

    async def _level(self, http, base_url, model, conc, input_len, output_len, duration_s, max_requests) -> dict:
        stats: list[ReqStat] = []
        start = self.clock()
        issued = 0

        async def worker():
            nonlocal issued
            while issued < max_requests and self.clock() - start < duration_s:
                issued += 1
                stats.append(await self._one(http, base_url, model, input_len, output_len))

        await asyncio.gather(*(worker() for _ in range(conc)))
        wall = max(self.clock() - start, 1e-9)
        ok = [s for s in stats if s.ok]
        per_req = [s.tokens / max(s.e2e - (s.ttft or 0), 1e-9) for s in ok if s.tokens]
        return {"concurrency": conc, "requests": len(ok), "errors": len(stats) - len(ok),
                "ttft_p50": percentile([s.ttft for s in ok if s.ttft is not None], 50),
                "ttft_p95": percentile([s.ttft for s in ok if s.ttft is not None], 95),
                "itl_p50": percentile([x for s in ok for x in s.itls], 50),
                "tps_total": sum(s.tokens for s in ok) / wall,
                "tps_per_req": sum(per_req) / len(per_req) if per_req else 0.0}

    async def run(self, *, deployment_id: int, engine: str, model_ref: str, config: dict, base_url: str,
                  served_name: str, levels=(1, 4, 8, 16), input_len: int = 1024, output_len: int = 256,
                  duration_s: float = 60.0, max_requests: int = 32) -> list[dict]:
        from dgxmm.deployer.service import config_hash
        if self._lock.locked():
            raise BenchmarkBusy(f"a benchmark is already running (deployment {self._running})")
        async with self._lock:
            self._running = deployment_id
            results = []
            try:
                async with self.http_factory() as http:
                    for conc in levels:
                        r = await self._level(http, base_url, served_name, conc, input_len, output_len, duration_s, max_requests)
                        with self.db.session() as s:
                            s.add(Benchmark(deployment_id=deployment_id, engine=engine, model_ref=model_ref,
                                            config_hash=config_hash(config), config=config, concurrency=conc,
                                            input_len=input_len, output_len=output_len, requests=r["requests"],
                                            ttft_p50=r["ttft_p50"], ttft_p95=r["ttft_p95"], itl_p50=r["itl_p50"],
                                            tps_total=r["tps_total"], tps_per_req=r["tps_per_req"]))
                        results.append(r)
                        self.events.publish("benchmark", {"deployment_id": deployment_id, "level": r,
                                                          "done": conc == levels[-1]})
            finally:
                self._running = None
            return results
```

- [ ] **Step 3: Run the tests to confirm they pass**

`DGX$ .venv/bin/pytest -q tests/test_benchmark.py`. Expected: `3 passed`.

---

### Task 35: Metrics, advice and benchmark API and UI

**Files:**
- Create: `dgxmm/api/metrics.py`, `frontend/src/components/Charts.tsx`
- Modify: `dgxmm/app.py` (`ROUTERS`), `dgxmm/context.py` (advisor and benchmark wiring), `frontend/src/stream.ts` (`useLive`), `frontend/src/pages/Dashboard.tsx`, `frontend/src/pages/DeploymentDetail.tsx`
- Test: `tests/test_api_metrics.py`

**Interfaces:**
- Produces routes:
  - `GET /api/deployments/{id}/metrics?window=live|1h|24h|7d`
  - `GET /api/host/metrics?window=`
  - `GET /api/deployments/{id}/advice`
  - `POST /api/deployments/{id}/benchmark` (admin; body `{levels?, input_len?, output_len?, duration_s?}` → 202)
  - `GET /api/deployments/{id}/benchmarks`
- Produces `useLive(id)` and `<MetricChart data lines height />`.

- [ ] **Step 1: Write the failing tests** — `tests/test_api_metrics.py`

```python
import time

import httpx

from dgxmm.advisor.benchmark import BenchmarkRunner
from dgxmm.advisor.rules import AdvisorService
from dgxmm.metrics.store import MetricsStore
from tests.test_api_deployments import SMALL, admin
from tests.test_benchmark import sse_handler
from tests.test_engines_vllm import VLLM_METRICS


def wire(c):
    st = c.stack
    c.ctx.metrics = MetricsStore(st.db)
    c.ctx.advisor = AdvisorService(st.deployer, st.inv, c.ctx.metrics, st.db)
    c.ctx.benchmarks = BenchmarkRunner(st.db, st.events,
                                       http_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(sse_handler([]))))


def test_metrics_advice_benchmark_routes(api_stack):
    h = admin(api_stack)
    wire(api_stack)
    api_stack.stack.http.all_healthy = True
    d = api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "a", "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny",
                                                                    "config": {**SMALL, "max_model_len": 65536 // 16}}).json()
    for _ in range(200):
        if api_stack.get(f"/dgxmm/api/deployments/{d['id']}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    from dgxmm.engines.vllm import VllmEngine
    api_stack.ctx.metrics.add(d["id"], 1.0, VllmEngine(api_stack.ctx.settings).parse_metrics(VLLM_METRICS))
    live = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/metrics?window=live").json()
    assert live[0]["running"] == 2.0
    assert api_stack.get(f"/dgxmm/api/deployments/{d['id']}/metrics?window=24h").json() == []
    assert isinstance(api_stack.get(f"/dgxmm/api/deployments/{d['id']}/advice").json(), list)
    r = api_stack.post(f"/dgxmm/api/deployments/{d['id']}/benchmark", headers=h,
                       json={"levels": [1], "input_len": 8, "output_len": 5, "duration_s": 2})
    assert r.status_code == 202
    for _ in range(200):
        rows = api_stack.get(f"/dgxmm/api/deployments/{d['id']}/benchmarks").json()
        if rows:
            break
        time.sleep(0.02)
    assert rows[0]["concurrency"] == 1
    bad = api_stack.post(f"/dgxmm/api/deployments/{d['id']}/benchmark", headers=h, json={"levels": [1000]})
    assert bad.status_code == 422
    assert api_stack.get("/dgxmm/api/host/metrics?window=live").status_code == 200
```

Run: `DGX$ .venv/bin/pytest -q tests/test_api_metrics.py`. Expected: FAIL (404).

- [ ] **Step 2: Implement `dgxmm/api/metrics.py`**

```python
import asyncio
import logging

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select

from dgxmm.advisor.benchmark import BenchmarkBusy
from dgxmm.api.deployments import deploy_error
from dgxmm.api.errors import api_error
from dgxmm.audit import audit
from dgxmm.auth.deps import Principal, current_principal, get_ctx, require_admin
from dgxmm.context import AppContext
from dgxmm.db import Benchmark
from dgxmm.deployer.service import DeployError

router = APIRouter(prefix="/api", tags=["metrics"])
log = logging.getLogger(__name__)
WINDOWS = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}


class BenchIn(BaseModel):
    levels: list[int] = Field([1, 4, 8, 16], min_length=1, max_length=6)
    input_len: int = Field(1024, ge=8, le=32768)
    output_len: int = Field(256, ge=1, le=4096)
    duration_s: float = Field(60, ge=1, le=600)
    max_requests: int = Field(32, ge=1, le=512)

    @field_validator("levels")
    @classmethod
    def _levels(cls, v: list[int]) -> list[int]:
        if any(not 1 <= c <= 128 for c in v):
            raise ValueError("concurrency levels must be between 1 and 128")
        return v


def _series(ctx, key, window):
    if window == "live":
        return ctx.metrics.series(key)
    if window not in WINDOWS:
        raise api_error(422, "invalid_window", "window must be live, 1h, 24h, 7d or 30d")
    return ctx.metrics.history(key, WINDOWS[window])


@router.get("/deployments/{dep_id}/metrics")
def dep_metrics(dep_id: int, window: str = Query("live"), p: Principal = Depends(current_principal),
                ctx: AppContext = Depends(get_ctx)):
    return _series(ctx, dep_id, window)


@router.get("/host/metrics")
def host_metrics(window: str = Query("live"), p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    return _series(ctx, "host", window)


@router.get("/deployments/{dep_id}/advice")
def advice(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        return ctx.advisor.advise(dep_id)
    except DeployError as e:
        raise deploy_error(e)


@router.post("/deployments/{dep_id}/benchmark", status_code=202)
async def benchmark(dep_id: int, body: BenchIn, p: Principal = Depends(require_admin), ctx: AppContext = Depends(get_ctx)):
    try:
        v = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    if v["state"] != "ready":
        raise api_error(409, "not_ready", "the deployment must be ready to benchmark")
    if ctx.benchmarks.busy() is not None:
        raise api_error(409, "benchmark_running", "another benchmark is running")

    async def _go():
        try:
            await ctx.benchmarks.run(deployment_id=dep_id, engine=v["engine"], model_ref=v["model_ref"], config=v["config"],
                                     base_url=ctx.deployer.base_url(v["name"], v["port"]), served_name=v["served_name"],
                                     levels=tuple(body.levels), input_len=body.input_len, output_len=body.output_len,
                                     duration_s=body.duration_s, max_requests=body.max_requests)
        except BenchmarkBusy:
            pass
        except Exception:
            log.exception("benchmark failed")

    asyncio.get_running_loop().create_task(_go())
    audit(ctx.db, p.user.id, "benchmark", v["name"], body.model_dump())
    return {"started": True}


@router.get("/deployments/{dep_id}/benchmarks")
def benchmarks(dep_id: int, p: Principal = Depends(current_principal), ctx: AppContext = Depends(get_ctx)):
    try:
        v = ctx.deployer.view(dep_id)
    except DeployError as e:
        raise deploy_error(e)
    with ctx.db.session() as s:
        rows = s.scalars(select(Benchmark).where(Benchmark.model_ref == v["model_ref"]).order_by(Benchmark.id.desc()).limit(200))
        return [{"id": b.id, "ts": b.ts.isoformat(), "config_hash": b.config_hash, "config": b.config, "engine": b.engine,
                 "concurrency": b.concurrency, "requests": b.requests, "ttft_p50": b.ttft_p50, "ttft_p95": b.ttft_p95,
                 "itl_p50": b.itl_p50, "tps_total": b.tps_total, "tps_per_req": b.tps_per_req,
                 "current": b.config == v["config"]} for b in rows]
```

The `levels` field validator raises `ValueError`, which pydantic reports as a validation error and FastAPI turns into a 422 (as `test_metrics_advice_benchmark_routes` expects).

Add `metrics as metrics_api` to `ROUTERS` in `dgxmm/app.py`.

In `build_context` (before `return ctx`):
```python
    from dgxmm.advisor.benchmark import BenchmarkRunner
    from dgxmm.advisor.rules import AdvisorService
    ctx.advisor = AdvisorService(deployer, inventory, store, db)
    ctx.benchmarks = BenchmarkRunner(db, events)
```

- [ ] **Step 3: Frontend: live hook and charts**

Append to `frontend/src/stream.ts`:
```ts
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export function useLive(id: number) {
  return useQuery<any[]>({ queryKey: ["live", id], queryFn: () => api<any[]>(`/deployments/${id}/metrics?window=live`), staleTime: Infinity });
}
```
(Merge the `useQuery` import with the existing `useQueryClient` import from `@tanstack/react-query`.)

In `Dashboard.tsx`, inside `LiveCell` replace the line
`const q = useQuery<any[]>({ queryKey: ["live", id], queryFn: () => Promise.resolve([]), staleTime: Infinity });`
with `const q = useLive(id);` and import `useLive` from `../stream`. Both the dashboard and the detail page now share one cache entry, seeded from the API and appended to by SSE.

`frontend/src/components/Charts.tsx`:
```tsx
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface LineDef { key: string; label: string; color: string }
const tick = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export function MetricChart({ data, lines, height = 180, fmt = (v: number) => v.toFixed(1) }: {
  data: any[]; lines: LineDef[]; height?: number; fmt?: (v: number) => string;
}) {
  if (!data.length) return <div className="flex h-24 items-center justify-center text-xs text-slate-400">No data yet</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
        <XAxis dataKey="ts" tickFormatter={tick} tick={{ fontSize: 10 }} minTickGap={40} />
        <YAxis tick={{ fontSize: 10 }} tickFormatter={fmt} width={48} />
        <Tooltip labelFormatter={(v) => new Date((v as number) * 1000).toLocaleString()} formatter={(v: number) => fmt(v)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {lines.map((l) => <Line key={l.key} type="monotone" dataKey={l.key} name={l.label} stroke={l.color} dot={false} isAnimationActive={false} connectNulls />)}
      </LineChart>
    </ResponsiveContainer>
  );
}
```

Series colours (fixed, so the same metric has the same colour everywhere): generation `#2563eb`, prompt `#0d9488`, p50 `#7c3aed`, p95 `#db2777`, KV `#ea580c`, queue `#475569`.

- [ ] **Step 4: Frontend: deployment detail performance, advice and benchmarks**

Add to `DeploymentDetail.tsx` (imports: `MetricChart`, `useLive`, `fmtPct`, `fmtMs`, `fmtNum`; `useQuery`/`useMutation` are already imported):
```tsx
function Performance({ id }: { id: number }) {
  const [win, setWin] = useState("live");
  const live = useLive(id);
  const hist = useQuery({ queryKey: ["hist", id, win], queryFn: () => api<any[]>(`/deployments/${id}/metrics?window=${win}`), enabled: win !== "live" });
  const data = (win === "live" ? live.data : hist.data) ?? [];
  return (
    <Card title="Performance" actions={["live", "24h", "7d"].map((w) => <Button key={w} variant={w === win ? "primary" : "ghost"} onClick={() => setWin(w)}>{w}</Button>)}>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div><h4 className="text-xs font-semibold text-slate-500">Throughput (tok/s)</h4>
          <MetricChart data={data} lines={[{ key: "gen_tps", label: "generation", color: "#2563eb" }, { key: "prompt_tps", label: "prompt", color: "#0d9488" }]} /></div>
        <div><h4 className="text-xs font-semibold text-slate-500">Time to first token</h4>
          <MetricChart data={data} fmt={(v) => fmtMs(v)} lines={[{ key: "ttft_p50", label: "p50", color: "#7c3aed" }, { key: "ttft_p95", label: "p95", color: "#db2777" }]} /></div>
        <div><h4 className="text-xs font-semibold text-slate-500">KV cache and queue</h4>
          <MetricChart data={data} fmt={(v) => fmtNum(v, 2)} lines={[{ key: "kv_usage", label: "KV used (0–1)", color: "#ea580c" }, { key: "waiting", label: "waiting requests", color: "#475569" }]} /></div>
        <div><h4 className="text-xs font-semibold text-slate-500">Inter-token latency / spec-decode acceptance</h4>
          <MetricChart data={data} fmt={(v) => fmtNum(v, 3)} lines={[{ key: "itl_p50", label: "ITL p50 (s)", color: "#7c3aed" }, { key: "spec_accept_rate", label: "accept rate", color: "#0d9488" }]} /></div>
      </div>
    </Card>
  );
}

function AdviceCard({ id, onApply }: { id: number; onApply: (patch: Record<string, any>) => void }) {
  const q = useQuery({ queryKey: ["advice", id], queryFn: () => api<any[]>(`/deployments/${id}/advice`), refetchInterval: 60000 });
  const admin = useIsAdmin();
  return (
    <Card title="Tuning advice">
      <ErrorBox error={q.error} />
      {(q.data ?? []).length === 0 && <p className="text-sm text-slate-500">No suggestions right now.</p>}
      <ul className="space-y-3">
        {(q.data ?? []).map((a) => (
          <li key={a.id} className="rounded-md border border-slate-200 p-3 text-sm">
            <div className="flex items-center gap-2">
              <Badge tone={a.severity === "warning" ? "red" : a.severity === "info" ? "slate" : "blue"}>{a.severity}</Badge>
              <span className="font-medium">{a.title}</span>
              <Badge tone={a.confidence === "measured" ? "green" : "slate"}>{a.confidence}</Badge>
              {admin && (a.patch || a.model_ref) && <Button className="ml-auto" onClick={() => onApply({ ...(a.patch ?? {}), ...(a.model_ref ? { model_ref: a.model_ref } : {}) })}>Apply…</Button>}
            </div>
            <p className="mt-1 text-slate-600">{a.rationale}</p>
            <p className="text-xs text-slate-500">Expected: {a.expected_effect}</p>
          </li>
        ))}
      </ul>
    </Card>
  );
}

function Benchmarks({ id, ready }: { id: number; ready: boolean }) {
  const admin = useIsAdmin();
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["benchmarks", id], queryFn: () => api<any[]>(`/deployments/${id}/benchmarks`), refetchInterval: 10000 });
  const run = useMutation({ mutationFn: () => api(`/deployments/${id}/benchmark`, { method: "POST", body: {} }), onSettled: () => qc.invalidateQueries({ queryKey: ["benchmarks", id] }) });
  const rows = q.data ?? [];
  return (
    <Card title="Benchmarks" actions={admin && <Button disabled={!ready || run.isPending} onClick={() => run.mutate()}>Run benchmark (1, 4, 8, 16)</Button>}>
      <ErrorBox error={run.error} />
      {run.isSuccess && <p className="mb-2 text-xs text-slate-500">Running… takes up to 4 minutes; results appear below as each level finishes.</p>}
      <Table head={["When", "Config", "Concurrency", "Total tok/s", "Per-request tok/s", "TTFT p50", "ITL p50"]}>
        {rows.map((b) => (
          <tr key={b.id}>
            <td className="td whitespace-nowrap text-xs">{new Date(b.ts).toLocaleString()}</td>
            <td className="td font-mono text-xs">{b.config_hash.slice(0, 8)}{b.current && <Badge tone="green">current</Badge>}</td>
            <td className="td tabular-nums">{b.concurrency}</td>
            <td className="td tabular-nums">{fmtNum(b.tps_total, 1)}</td>
            <td className="td tabular-nums">{fmtNum(b.tps_per_req, 1)}</td>
            <td className="td tabular-nums">{fmtMs(b.ttft_p50)}</td>
            <td className="td tabular-nums">{fmtMs(b.itl_p50)}</td>
          </tr>
        ))}
      </Table>
    </Card>
  );
}
```
In the `DeploymentDetail` return tree, add the three cards after the top grid:
```tsx
<Performance id={id} />
<AdviceCard id={id} onApply={(patch) => setDialog({ patch })} />
<Benchmarks id={id} ready={d.state === "ready"} />
```
(`useQueryClient`, `Table` and `fmtMs` / `fmtNum` / `fmtPct` must be imported; remove any unused import so `tsc` passes with `noUnusedLocals`.)

In `Dashboard.tsx`, add a host memory chart above the deployments card:
```tsx
const hostLive = useQuery({ queryKey: ["host-metrics"], queryFn: () => api<any[]>("/host/metrics?window=live"), refetchInterval: 10000 });
…
<Card title="Memory over the last hour">
  <MetricChart data={hostLive.data ?? []} fmt={(v) => fmtBytes(v, 0)} lines={[{ key: "used", label: "used", color: "#2563eb" }, { key: "external", label: "external", color: "#ea580c" }, { key: "free_for_new", label: "free for new", color: "#0d9488" }]} />
</Card>
```

- [ ] **Step 5: Run all tests and the build**

`DGX$ .venv/bin/pytest -q` → all pass. `NPM$ sh -c "npx vitest run && npm run build"` → passes.

---

### Task 36: On-box validation script and README

**Files:**
- Create: `scripts/validate_on_box.sh`, `README.md`

- [ ] **Step 1: Write `scripts/validate_on_box.sh`**

```bash
#!/usr/bin/env bash
# End-to-end check on the DGX against a running dgxmm. Deploys small models; tell users first.
# Usage: DGXMM_USER=admin DGXMM_PASSWORD=... scripts/validate_on_box.sh
set -euo pipefail
B=${DGXMM_URL:-http://127.0.0.1:8090/dgxmm/api}
J=$(mktemp); trap 'rm -f "$J"' EXIT
pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*"; exit 1; }
py() { python3 -c "import sys,json; d=json.load(sys.stdin); $1"; }

CSRF=$(curl -sf -c "$J" -H 'content-type: application/json' \
  -d "{\"username\":\"${DGXMM_USER:-admin}\",\"password\":\"${DGXMM_PASSWORD:?set DGXMM_PASSWORD}\"}" "$B/auth/login" | py 'print(d["csrf_token"])')
H=(-sf -b "$J" -H "X-CSRF-Token: $CSRF" -H 'content-type: application/json')
pass "login"

curl "${H[@]}" "$B/host" | py 'b=d["budget"]; assert b["total"]>100*2**30; print("free_for_new GiB", round(b["free_for_new"]/2**30,1))' && pass "host budget"
curl "${H[@]}" "$B/inventory" | py 'assert any(a["ref"]=="Qwen/Qwen3-0.6B" and a["complete"] for a in d)' && pass "inventory has Qwen3-0.6B"

wait_state() {  # id wanted timeout_s
  for _ in $(seq 1 "$3"); do
    s=$(curl "${H[@]}" "$B/deployments/$1" | py 'print(d["state"])')
    [ "$s" = "$2" ] && return 0
    [ "$s" = "failed" ] && return 1
    sleep 1
  done; return 1
}

run_one() {  # name engine ref config_json
  local id
  id=$(curl "${H[@]}" -d "{\"name\":\"$1\",\"engine\":\"$2\",\"model_ref\":\"$3\",\"config\":$4}" "$B/deployments" | py 'print(d["id"])')
  wait_state "$id" ready 600 || { curl "${H[@]}" "$B/deployments/$id" | py 'print(d["last_error"])'; fail "$1 ready"; }
  pass "$1 ready"
  sleep 20
  curl "${H[@]}" "$B/deployments/$id" | py 'r,a=d["reserved_bytes"],d["actual_bytes"] or 0; print(f"reserved {r/2**30:.2f} GiB measured {a/2**30:.2f} GiB ratio {a/max(r,1):.2f}"); assert a<=r*1.15, "measured exceeds reservation by >15%"' && pass "$1 memory within reservation"
  curl "${H[@]}" -d '{"levels":[1,4],"input_len":256,"output_len":64,"duration_s":20,"max_requests":8}' "$B/deployments/$id/benchmark" >/dev/null
  for _ in $(seq 1 120); do n=$(curl "${H[@]}" "$B/deployments/$id/benchmarks" | py 'print(len([b for b in d if b["current"]]))'); [ "$n" -ge 2 ] && break; sleep 1; done
  curl "${H[@]}" "$B/deployments/$id/benchmarks" | py '[print(f"  c={b[\"concurrency\"]} total={b[\"tps_total\"]:.1f} tok/s ttft={b[\"ttft_p50\"]}") for b in d if b["current"]]' && pass "$1 benchmark"
  curl "${H[@]}" "$B/deployments/$id/advice" | py 'print("  advice:", [a["id"] for a in d])' && pass "$1 advice"
  curl "${H[@]}" "$B/deployments/$id" | py 'print("  litellm:", d["litellm_status"])'
  curl "${H[@]}" -X POST "$B/deployments/$id/stop" >/dev/null && curl "${H[@]}" -X DELETE "$B/deployments/$id" >/dev/null && pass "$1 stopped and removed"
}

run_one val-vllm vllm Qwen/Qwen3-0.6B '{"max_model_len":8192,"max_num_seqs":4}'
G=$(curl "${H[@]}" "$B/inventory" | py 'g=[a["ref"] for a in d if a["source"]=="gguf" and a["complete"] and "0.6B" in a["ref"]]; print(g[0] if g else "")')
if [ -n "$G" ]; then run_one val-llama llamacpp "$G" '{"ctx_size":8192,"parallel":2}'; else echo "SKIP  no small GGUF on disk (Task 24 downloads one)"; fi
echo "ALL CHECKS PASSED"
```

- [ ] **Step 2: Write `README.md`**

```markdown
# DGX Model Manager (DGXMM)

Web UI + API for running LLMs on the DGX Spark (`dgx-host`, GB10, 128 GB unified memory):
see what is on disk, deploy vLLM or llama.cpp containers behind an out-of-memory gate, switch models with
automatic rollback, keep LiteLLM in sync, watch tok/s, TTFT and KV usage, plan which models fit together,
browse/download from Hugging Face (with llmfit scores) and get benchmark-backed tuning advice.

Design: `docs/superpowers/specs/2026-10-15-dgx-model-manager-design.md`. Plan: `docs/superpowers/plans/`.

## Run

    cp config.example.yaml config.yaml          # adjust if needed
    mkdir -p data secrets && chmod 700 secrets
    touch secrets/hf_token                      # optional: paste an HF token for gated repos
    # LiteLLM master key (never printed):
    docker exec litellm-proxy sh -c 'grep -E "^\s*master_key:" /app/config.yaml | sed -E "s/.*master_key:\s*//"' > secrets/litellm_master_key
    chmod 600 secrets/*
    docker compose up -d --build
    docker compose logs dgxmm | grep "first run"   # one-time admin password

Open `http://<dgx>/dgxmm/` (via the nginx reverse proxy) or `http://127.0.0.1:8090/dgxmm/` on the box.
LiteLLM must have `store_model_in_db: true` for automatic registration.

## How memory safety works

- Budget = MemTotal − 8 GiB safety margin − 4 GiB OS reserve − committed reservations − external usage.
- Each deployment's reservation is computed from `config.json` / GGUF headers (weights + KV + state +
  activations + runtime), then corrected by what was measured on this box.
- Per-deployment usage = cgroup anon+kernel memory + NVML per-process GPU memory (CUDA allocations on GB10
  are not charged to the container cgroup). Anything not attributed to a managed container counts as
  "external" and is never touched.
- vLLM's `--gpu-memory-utilization` is always derived as reservation / MemTotal.
- If something doesn't fit, the UI offers plans (fp8 KV, shorter context, fewer sequences, smaller quant,
  sleep or stop lower-priority deployments). Nothing is evicted without confirmation; the watchdog only stops
  a model on its own when `auto_protect: true`.

## Security

- Username/password login (argon2id), HTTP-only SameSite=Strict cookies, CSRF tokens, login rate limiting,
  roles `admin` / `viewer`, audit log of every action.
- DGXMM only stops/removes containers labelled `dgxmm.managed=true`; flags passed to engines are allowlisted.
- The Docker socket is mounted, which is root-equivalent on the host: keep the UI on the LAN/VPN.
- Engine ports are published on 127.0.0.1 only; clients go through LiteLLM (`:4000`).

## Develop

    python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' && .venv/bin/pytest -q
    docker run --rm -u 1000:1000 -v $PWD/frontend:/app -w /app -e HOME=/tmp node:22-bookworm-slim sh -c "npm ci && npx vitest run && npm run build"
    scripts/validate_on_box.sh                   # end-to-end on the DGX (deploys small models)

## Troubleshooting

- *Download fails with "owned by root"*: an engine container created that HF cache folder as root;
  run the `sudo chown -R 1000:1000 …` shown in the error.
- *LiteLLM status stays `pending`*: check `store_model_in_db: true` and the key in `secrets/litellm_master_key`.
- *Catalog says "llmfit unavailable"*: `docker exec dgxmm llmfit --version`; everything else keeps working.
- *Deployment failed with exit 137*: the kernel OOM-killed it; DGXMM adds 10% to the next estimate for that config.
```

- [ ] **Step 3: Make the script executable and lint it**

`DGX$ chmod +x scripts/validate_on_box.sh && bash -n scripts/validate_on_box.sh && echo ok` → `ok`.

---

### Task 37: Final verification, spec amendments, and the single commit

**Files:**
- Modify: `docs/superpowers/specs/2026-10-15-dgx-model-manager-design.md` (append "Amendments from implementation planning")
- Create: `docs/superpowers/plans/2026-10-15-dgx-model-manager.md` (this plan, copied into the repo)

- [ ] **Step 1: Append the amendments to the spec**

```markdown
## 16. Amendments from implementation planning (2026-10-15)

1. **Per-deployment memory** = cgroup `anon + kernel` + NVML per-process GPU memory of the container's PIDs
   (not cgroup `memory.current`): on GB10, CUDA allocations are not charged to the container cgroup while page
   cache is. Under-attribution shows up as "external", which only makes the gate stricter. (§4.1)
2. **Sleep on unified memory** frees only KV/activations; weights are offloaded into the same physical pool.
   The planner counts sleep as freeing `reserved − weights − draft − runtime`. (§4.1, §5.4)
3. **Inventory** is kept in memory and rescanned (no `artifacts` table); deployments store `model_ref`
   (repo id or GGUF path). Broken `refs/main` snapshots are skipped in favour of the newest complete one,
   and vLLM is started with `--revision <snapshot>` and `HF_HUB_OFFLINE=1`. (§3.2, §7)
4. **Apply on advice** uses an in-place *reconfigure* (same deployment row, rollback to previous config on
   failure) rather than creating a second deployment; *switch* is for moving to a different deployment. (§5.5, §9.2)
5. **States** gain `unknown` for managed containers found without a DB row. (§5.3, §11)
6. **Co-location ranking**: more models first, then more favourites, then more headroom (pure headroom
   ranking would always pick one tiny model). Favourites are stored in a `favourites` table. (§4.6)
7. **UI kit**: Tailwind with a small in-repo component set (shadcn-style) instead of the shadcn CLI.
8. **Ports**: engine ports and dgxmm's own `8090` are published on `127.0.0.1` only (for on-box tools and SSH tunnels); network users go through nginx `/dgxmm/` and LiteLLM reaches engines by container name on `gateway`.
9. **Catalog config fetch** uses HTTP (not `hf_hub_download`) so browsing never creates partial repos in the cache.
```

- [ ] **Step 2: Copy the plan into the repo**

```bash
DGX$ mkdir -p docs/superpowers/plans
# (from the laptop) copy the plan into docs/superpowers/plans/ of the checkout on the DGX
```

- [ ] **Step 3: Full verification**

```bash
DGX$ .venv/bin/pytest -q
NPM$ sh -c "npm ci --no-audit --no-fund && npx vitest run && npm run build"
DGX$ docker compose up -d --build && sleep 10 && docker compose ps && docker compose logs --tail=80 dgxmm
DGX$ DGXMM_PASSWORD='<admin password>' scripts/validate_on_box.sh
```
Expected:
- pytest reports 0 failures, vitest passes, and the build succeeds.
- `dgxmm` is `healthy`, and the logs show no tracebacks.
- The validation script ends with `ALL CHECKS PASSED`.

Copy the real outputs (test counts, measured vs reserved ratios, benchmark tok/s) into the final report. If any check fails, fix the cause and re-run. Do not commit a failing tree.

- [ ] **Step 4: UI smoke test with the Playwright MCP browser**

From the laptop, open a port forward over SSH: `ssh -f -N -L 18090:127.0.0.1:8090 dgx-host`. Then, with the Playwright MCP tools, open `http://localhost:18090/dgxmm/` and check each item below, taking a screenshot of each page:
1. The login page renders. Log in as `admin`. Expect the dashboard, or the forced password change on first login.
2. The **Dashboard** memory bar shows segments for external use, margin and free. Totals are about 121 GiB.
3. **Models on disk** lists `Qwen/Qwen3.6-35B-A3B` and `unsloth/Qwen3.8-27B-NVFP4` (complete, with a `refs/main` note), plus whisper/BERT as `other`.
4. **Deploy** → choose `nvidia/Qwen3.6-35B-A3B-NVFP4`. The Memory check panel shows a breakdown with a vLLM utilization value. Do **not** click Deploy.
5. **Catalog** → search `qwen3`. Results show fit badges.
6. **Planner** shows at least one combination.
7. **Audit log** shows the logins.

Close the extra forward afterwards: `pkill -f "18090:127.0.0.1:8090"`.

- [ ] **Step 5: Review what will be committed**

```bash
DGX$ git status --short | head -100
DGX$ git status --short | grep -E '(^\?\?|^ ?A).*(data/|secrets/|config\.yaml$|\.sqlite|node_modules|dist/|\.venv)' && echo "STOP: ignored paths would be committed" || echo "clean"
```
Expected: only source, tests, docs, frontend source, `frontend/package-lock.json`, packaging files and `config.example.yaml`. The check prints `clean`.

- [ ] **Step 6: The single commit**

```bash
DGX$ git add -A
DGX$ git commit -q -F - <<'EOF'
feat: DGX Model Manager (inventory, OOM-safe deploys, switching, LiteLLM, llama.cpp, catalog, metrics, advice)

- Memory budget from /proc + cgroups + NVML; per-engine estimator with calibration; vLLM util derived
- vLLM and llama.cpp engines; deploy/stop/sleep/wake/switch/reconfigure with rollback; adopt existing containers
- Eviction/reduction plans; watchdog; LiteLLM DB-model sync; startup reconciliation
- HF catalog with llmfit scores, vLLM arch registry, downloads with progress; co-location planner
- Metrics scraper with rollups, 15 tuning rules, benchmark runner
- React UI served under /dgxmm/, username/password auth with roles, CSRF, audit log
EOF
DGX$ git log --oneline | head -3
```
Expected: two commits in total, `1aafb85 docs: add DGX Model Manager design spec` and the new feature commit on top.

- [ ] **Step 7: Report**

Tell the user:
- Where the app runs (`http://<dgx>/dgxmm/` once nginx is wired, otherwise `127.0.0.1:8090`).
- The test and validation results, with real numbers.
- The [USER] steps that are still pending: the nginx snippet, LiteLLM `store_model_in_db`, disabling `model-manager-v2`, and rotating the LiteLLM key.

