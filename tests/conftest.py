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
        broken_set = set(broken or [])
        for name, size in entries.items():
            if name in broken_set:
                continue
            blob = d / "blobs" / f"{rev[:8]}-{name.replace('/', '_')}"
            if name == "config.json":
                blob.write_text(_json.dumps(config))
            else:
                with open(blob, "wb") as f:
                    f.truncate(size)
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            _os.symlink(_os.path.relpath(blob, (snap / name).parent), snap / name)
        for name in broken or []:
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
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
    docker = DockerController(sdk)
    docker.own_container_id = lambda: "dgxmm-self"  # as if DGXMM ran in a container (joins the decider network)
    deployer = Deployer(settings=settings, db=db, docker=docker, inventory=inv,
                        engines=build_engines(settings), http=engines_http.client(), events=events)
    deployer.budget = BudgetService(settings, probe, deployer.usage_rows)
    return SimpleNamespace(settings=settings, db=db, inv=inv, sdk=sdk, probe=probe, http=engines_http,
                           deployer=deployer, budget=deployer.budget, events=events)


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
