# Multi-node deployments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From the one DGXMM on dgx-1, deploy any model on dgx-1, on dgx-2, or on both together as a cluster (suggested when it is too big for one DGX), see where each model runs, and serve all of them through the one LiteLLM, with every existing feature unchanged.

**Architecture:** A static node registry (one `DockerController` per node; remote nodes over docker-py `ssh://`), a per-node memory probe and budget, and a `placement` on every deployment (`single` on one node, or `cluster` over several, head first) that routes every Docker call, fit check, health check and LiteLLM `api_base`. Cluster deployments run vLLM tensor-parallel over NCCL/RDMA or llama.cpp over RPC, one container per node ("rank"). Tasks 1-8 deliver single-node placement on either box; Tasks 9-14 add the cluster and feature parity; Tasks 15-16 are UI and on-hardware acceptance.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/SQLite, docker-py >= 7.1 (`use_ssh_client=True`), OpenSSH client,
React/TypeScript + vitest, pytest + `tests/fakes.py` (`FakeDockerClient`).

**Spec:** `docs/superpowers/specs/2026-11-14-multi-node-design.md`

## Global Constraints

- With no `nodes:` key in `config.yaml`, behaviour, API responses and DB contents are identical to today; the
  full existing suite (2644 tests) must stay green.
- The local node's id is always `"local"`; remote ids match `^[a-z0-9][a-z0-9-]{0,31}$`; the UI shows `label`.
- Remote Docker is reached only via `ssh://<user>@<cluster_ip>` with `StrictHostKeyChecking yes` and a
  config-pinned known_hosts line. No trust-on-first-use, no TCP Docker API.
- Remote engine ports publish on the node's `cluster_ip` only, never `0.0.0.0`.
- Every Docker write on any node keeps the `dgxmm.managed` label guard.
- Migrations: new versions start at 41 (40 is the last one). Each runs in its own transaction like `_ondemand_mode`.
- Subprocesses get argv lists, never `shell=True`.
- Commits: author `nikghodki <92228606+nikghodki@users.noreply.github.com>`, no `Co-Authored-By` trailer.
- Run tests with `.venv/bin/python -m pytest -q -p no:cacheprovider`.
- `tests/test_repo_hygiene.py` applies to every tracked file: no real user names, host names or LAN/Tailscale
  addresses. Use user `dgxuser`, host `dgx-host`, and documentation addresses (`192.0.2.x` for the cluster
  link, `198.51.100.x`/`203.0.113.x` elsewhere). Real values live only in the untracked `config.yaml`.

## Review Focus

1. **A model too big for one DGX** (e.g. DeepSeek-V4-Flash-0731, 167 GB, against about 110 GiB free per box) → the
   wizard pre-selects the cluster and marks both single options as not fitting. (Task 11 `test_big_model_recommends_cluster`)
2. **One rank of a cluster model dies** → the whole deployment fails, the other rank is stopped, and its memory stays
   reserved until both containers are confirmed gone. (Task 12 `test_worker_exit_fails_whole_deployment_and_stops_head`,
   `test_cluster_reservation_kept_until_all_ranks_gone`)
3. **dgx-2 goes offline while a model runs there** → the deployment stays as it was, the node shows `stale`, nothing
   on dgx-1 is stopped to compensate. (Task 5 `test_remote_transport_errors_do_not_fail_running_deployment`)
4. **An on-demand cluster model goes idle** → both ranks stop and both nodes get their memory back; the next request
   starts both again. (Task 14 `test_ondemand_reap_stops_all_ranks`)
5. **DGXMM restarts while a remote or cluster deployment is starting** → reconcile adopts each container by its
   deployment-id label, with no duplicate `<name>-2` row. (Task 7 `test_reconcile_adopts_remote_managed_container_by_label`,
   Task 12 `test_reconcile_adopts_cluster_ranks`)

---

### Task 1: Node configuration

**Files:**
- Modify: `dgxmm/config.py` (new models next to `PathMapping`/`Mount`; fields on `Settings`)
- Modify: `config.example.yaml` (commented `nodes:` example)
- Test: `tests/test_nodes_config.py`

**Interfaces:**
- Produces: `class NodeConfig(BaseModel)`: `id: str`, `label: str`, `docker_url: str`, `cluster_ip: str | None = None`,
  `publish_bind: str | None = None`, `hf_cache_host: str | None = None`, `ssh_user: str | None = None`,
  `known_hosts: str | None = None`, `nccl: NcclConfig | None = None`.
- Produces: `class NcclConfig(BaseModel)`: `ib_hca: str`, `socket_ifname: str` (used by Task 12; parsed now).
- Produces: `Settings.nodes: list[NodeConfig] = []`, `Settings.remote_probe_interval_s: float = Field(5.0, gt=0)`,
  `Settings.node_ssh_key_file: str = "/run/secrets/node_ssh_key"`,
  `Settings.all_nodes() -> list[NodeConfig]`, `Settings.node(node_id: str) -> NodeConfig` (raises `KeyError`),
  `LOCAL_NODE_ID = "local"`.

- [ ] **Step 0:** Repair the dev env: `.venv/bin/python -m pip install -e '.[dev]'`, then
  `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_advanced_jobs.py` → passes (it fails today on a missing `tzdata`).
- [ ] **Step 1: Write the failing tests**

```python
def test_no_nodes_means_one_implicit_local_node(settings):
    nodes = settings.all_nodes()
    assert [n.id for n in nodes] == ["local"]
    assert nodes[0].docker_url == "unix:///var/run/docker.sock"
    assert nodes[0].hf_cache_host == settings.hf_cache.host

def test_two_nodes_parse(settings_factory):
    s = settings_factory(nodes=[
        {"id": "local", "label": "dgx-1", "docker_url": "unix:///var/run/docker.sock"},
        {"id": "dgx-2", "label": "dgx-2", "docker_url": "ssh://dgxuser@192.0.2.2",
         "cluster_ip": "192.0.2.2", "ssh_user": "dgxuser",
         "known_hosts": "192.0.2.2 ssh-ed25519 AAAA...", "hf_cache_host": "/home/dgxuser/.cache/huggingface",
         "nccl": {"ib_hca": "rocep1s0f0,roceP2p1s0f0", "socket_ifname": "enp1s0f0np0"}}])
    assert s.node("dgx-2").cluster_ip == "192.0.2.2"
    assert s.node("dgx-2").publish_bind == "192.0.2.2"   # defaults to cluster_ip for remote nodes

def test_remote_node_requires_known_hosts_and_cluster_ip(settings_factory):
    with pytest.raises(ValidationError, match="known_hosts"):
        settings_factory(nodes=[{"id": "local", "label": "a", "docker_url": "unix:///var/run/docker.sock"},
                                {"id": "dgx-2", "label": "b", "docker_url": "ssh://u@198.51.100.2", "cluster_ip": "198.51.100.2"}])

def test_duplicate_or_missing_local_rejected(settings_factory):
    with pytest.raises(ValidationError, match="exactly one local node"):
        settings_factory(nodes=[{"id": "dgx-2", "label": "b", "docker_url": "ssh://u@198.51.100.2",
                                 "cluster_ip": "198.51.100.2", "known_hosts": "198.51.100.2 ssh-ed25519 AAAA"}])
```

- [ ] **Step 2:** Run `pytest tests/test_nodes_config.py -v` → FAIL (`NodeConfig` not defined).
- [ ] **Step 3:** Implement the models and a `Settings` `model_validator(mode="after")`: exactly one node with a
  `unix://` URL and id `local`; unique ids; `ssh://` nodes need `cluster_ip` and `known_hosts`; remote
  `publish_bind` defaults to `cluster_ip`; the local node's `hf_cache_host` defaults to `hf_cache.host`.
  `all_nodes()` returns `[implicit local]` when `nodes` is empty. Add `settings_factory` to `tests/conftest.py` if absent.
- [ ] **Step 4:** Run `pytest tests/test_nodes_config.py -v` → PASS; full suite → green.
- [ ] **Step 5:** Commit `feat(nodes): node configuration (implicit single local node by default)`.

### Task 2: Node registry and remote Docker over SSH

**Files:**
- Create: `dgxmm/nodes/__init__.py`, `dgxmm/nodes/registry.py`
- Modify: `dgxmm/docker_ctl.py` (`DockerController.for_node`), `dgxmm/context.py:121-135` (build the registry, pass it to `Deployer`)
- Modify: `Dockerfile` (add `openssh-client`), `docker-compose.yaml` (optional `node_ssh_key` secret; `HOME=/app/data/home`)
- Test: `tests/test_node_registry.py`

**Interfaces:**
- Consumes: `NodeConfig`, `Settings.all_nodes()`, `Settings.node_ssh_key_file` (Task 1).
- Produces: `class NodeRegistry`: `__init__(settings, client_factory: Callable[[NodeConfig], Any] | None = None)`,
  `ids() -> list[str]`, `local_id: str`, `config(node_id) -> NodeConfig`, `docker(node_id) -> DockerController`
  (lazy, cached; raises `UnknownNode`), `is_remote(node_id) -> bool`.
- Produces: `class UnknownNode(KeyError)`, `class NodeUnreachable(RuntimeError)` in `dgxmm/nodes/registry.py`.
- Produces: `DockerController.for_node(node: NodeConfig) -> DockerController`
  (`docker.DockerClient(base_url=node.docker_url, use_ssh_client=True, timeout=30)` for `ssh://`).

- [ ] **Step 1: Write the failing tests**

```python
def test_registry_builds_local_controller_for_unix_url(settings):
    reg = NodeRegistry(settings, client_factory=lambda n: FakeDockerClient())
    assert reg.ids() == ["local"] and not reg.is_remote("local")
    assert isinstance(reg.docker("local"), DockerController)

def test_registry_writes_ssh_config_with_pinned_known_hosts(two_node_settings, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    NodeRegistry(two_node_settings, client_factory=lambda n: FakeDockerClient())
    cfg = (tmp_path / ".ssh" / "config").read_text()
    assert "Host 192.0.2.2" in cfg and "StrictHostKeyChecking yes" in cfg
    assert f"IdentityFile {two_node_settings.node_ssh_key_file}" in cfg
    kh = (tmp_path / ".ssh" / "known_hosts_dgx-2").read_text()
    assert kh.strip() == two_node_settings.node("dgx-2").known_hosts
    assert oct((tmp_path / ".ssh" / "config").stat().st_mode & 0o777) == "0o600"

def test_unknown_node_raises(settings):
    with pytest.raises(UnknownNode):
        NodeRegistry(settings, client_factory=lambda n: FakeDockerClient()).docker("dgx-9")
```

- [ ] **Step 2:** Run → FAIL (module missing).
- [ ] **Step 3:** Implement. On construction, write `$HOME/.ssh/config` (0600, dir 0700) with one `Host <cluster_ip>`
  block per remote node: `User`, `IdentityFile`, `UserKnownHostsFile ~/.ssh/known_hosts_<id>`,
  `StrictHostKeyChecking yes`, `ServerAliveInterval 15`, `BatchMode yes`. The `context.py` wiring keeps
  `deployer.docker` as `registry.docker("local")`, so existing callers and tests are unchanged.
- [ ] **Step 4:** Run → PASS; full suite green. Manual check on dgx-1:
  `docker exec dgxmm python -c "from dgxmm.nodes.registry import ...; print(reg.docker('dgx-2').list_managed())"` → `[]`.
- [ ] **Step 5:** Commit `feat(nodes): registry with remote Docker over pinned SSH`.

### Task 3: Remote memory probe and per-node budget

**Files:**
- Create: `dgxmm/nodes/probe.py`
- Modify: `dgxmm/memory/budget.py` (`BudgetService` keyed by node), `dgxmm/config.py` (`Images.probe`, pinned by digest)
- Test: `tests/test_node_probe.py`, `tests/test_budget_nodes.py`

**Interfaces:**
- Consumes: `NodeRegistry.docker(node_id)` (Task 2), `parse_meminfo(text) -> HostMemory` (`dgxmm/memory/hostmem.py:175`).
- Produces: `class NodeProbe`: `__init__(registry, settings, clock=time.monotonic)`, `ensure(node_id) -> None`
  (creates `dgxmm-node-probe`: image `settings.images.probe`, no network, read-only rootfs, `/proc:/host/proc:ro`,
  `<hf_cache_host>:/hf:ro`, `cap_drop=["ALL"]`, managed label), `sample(node_id) -> HostMemory | None`,
  `stale(node_id) -> bool`, `exec(node_id, argv: list[str]) -> str`.
- Produces: `BudgetService.current(node_id: str = "local") -> Budget | None`, `BudgetService.all() -> dict[str, Budget]`.
  Usage rows carry `node: str`; `compute_budget` is unchanged and is called once per node with that node's rows.

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_parses_remote_meminfo_via_exec(two_node_registry):
    two_node_registry.fake("dgx-2").exec_output = "MemTotal: 127000000 kB\nMemAvailable: 100000000 kB\n"
    p = NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: 0.0)
    assert p.sample("dgx-2").total == 127000000 * 1024

def test_probe_stale_after_three_intervals(two_node_registry):
    t = [0.0]; p = NodeProbe(two_node_registry, two_node_registry.settings, clock=lambda: t[0])
    p.sample("dgx-2"); t[0] = 3 * two_node_registry.settings.remote_probe_interval_s + 0.1
    assert p.stale("dgx-2") is True and p.stale("local") is False

def test_budget_per_node_filters_usages_by_node(budget_service_two_nodes):
    b = budget_service_two_nodes.all()
    assert {u.name for u in b["dgx-2"].deployments} == {"remote-model"}
    assert {u.name for u in b["local"].deployments} == {"local-model"}
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Local keeps `HostProbe` as today. Remote nodes are sampled on a background loop every
  `remote_probe_interval_s`; an exec failure leaves the last sample in place, so the node goes stale by age.
  Extend `FakeDockerClient` with `exec_output` and `fake(node_id)` helpers in `tests/fakes.py`.
- [ ] **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Commit `feat(nodes): remote memory probe and per-node budgets`.

### Task 4: Deployment node column, migration 41, per-node ports

**Files:**
- Modify: `dgxmm/db.py:58-95` (`Deployment.node: Mapped[str] = mapped_column(String(32), default="local")`)
- Modify: `dgxmm/migrations.py` (`_deployment_node`, `Migration(41, "deployments: node", _deployment_node)`)
- Modify: the port allocator in `dgxmm/deployer/service.py` (the code using `settings.port_range`): uniqueness per `(node, port)`
- Test: `tests/test_migration_node.py`, `tests/test_ports_nodes.py`

**Interfaces:**
- Produces: `Deployment.node` (never NULL after migration 41); `Deployer.allocate_port(node_id: str) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
def test_migration_41_backfills_local(v40_db_with_one_deployment):
    migrate(v40_db_with_one_deployment)
    assert row(v40_db_with_one_deployment, "SELECT node FROM deployments") == ("local",)

def test_ports_unique_per_node(deployer_two_nodes):
    a = deployer_two_nodes.allocate_port("local"); b = deployer_two_nodes.allocate_port("dgx-2")
    assert a == b == 8100
    insert_deployment(deployer_two_nodes, node="local", port=a)
    assert deployer_two_nodes.allocate_port("local") == 8101
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement, following `_ondemand_mode`: add the column if missing, then `UPDATE … SET node='local' WHERE node IS NULL`.
- [ ] **Step 4:** Run → PASS; full suite green. **Step 5:** Commit `feat(nodes): deployments carry a node (migration 41)`.

### Task 5: Deployer places, monitors and stops on the chosen node

**Files:**
- Modify: `dgxmm/deployer/service.py` (`DeployRequest.node`, `deploy`, `prepare`, `_launch`, `_monitor`, `stop`,
  `restart`, `sleep`, `wake`, `check_liveness`, `base_url`), `dgxmm/engines/vllm.py:181` and `dgxmm/engines/llamacpp.py`
  (HF cache mount from `node.hf_cache_host`, read-only)
- Test: `tests/test_deployer_nodes.py`

**Interfaces:**
- Consumes: `NodeRegistry` (Task 2), `NodeProbe.stale`, `BudgetService.current(node_id)` (Task 3), `allocate_port(node_id)` (Task 4).
- Produces: `DeployRequest.node: str = "local"` (pattern as `NodeConfig.id`);
  `Deployer.base_url(name, port, engine=None, node="local") -> str` (remote: `http://<cluster_ip>:<port>`);
  `LaunchContext.node: str`; `DeployError` codes `unknown_node` (400), `node_unreachable` (503), `node_stale` (503).

- [ ] **Step 1: Write the failing tests**

```python
async def test_deploy_on_remote_runs_container_on_that_node_only(deployer_two_nodes, qwen_req):
    await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert len(deployer_two_nodes.fake("dgx-2").containers.list()) == 1
    assert deployer_two_nodes.fake("local").containers.list() == []

async def test_remote_spec_publishes_on_cluster_ip_and_mounts_hf_read_only(deployer_two_nodes, qwen_req):
    await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    c = deployer_two_nodes.fake("dgx-2").containers.list()[0]
    assert c.ports_bind_ip == "192.0.2.2"
    assert ("/home/dgxuser/.cache/huggingface", "ro") in c.mount_sources_modes

async def test_fit_uses_target_node_budget(deployer_two_nodes, qwen_req):
    deployer_two_nodes.set_free("local", 0); deployer_two_nodes.set_free("dgx-2", 100 * 2**30)
    await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2", "name": "q2"}), user_id=1)
    with pytest.raises(InsufficientMemory):
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "local", "name": "q1"}), user_id=1)

async def test_unknown_node_400(deployer_two_nodes, qwen_req):
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-9"}), user_id=1)
    assert (e.value.code, e.value.status) == ("unknown_node", 400)

async def test_stale_node_refuses_start(deployer_two_nodes, qwen_req):
    deployer_two_nodes.mark_stale("dgx-2")
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert (e.value.code, e.value.status) == ("node_stale", 503)

async def test_remote_transport_errors_do_not_fail_running_deployment(deployer_two_nodes, ready_remote_dep):
    deployer_two_nodes.fake("dgx-2").raise_on_all = NodeUnreachable("ssh: connect timed out")
    for _ in range(6):
        await deployer_two_nodes.monitor_tick(ready_remote_dep.id)
    assert deployer_two_nodes.state(ready_remote_dep.id) == "ready"
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Replace every `self.docker` use on a deployment's container with
  `self.nodes.docker(dep.node)`. Map docker-py/SSH transport errors to `NodeUnreachable`. In `_monitor`,
  `NodeUnreachable` from a remote node does not count toward the 5-errors rule (log it and back off up to 60 s);
  local behaviour is unchanged. `wake`'s memory check uses the deployment's node budget.
- [ ] **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Commit `feat(nodes): deploy, monitor and stop on the chosen node`.

### Task 6: Weights presence check and copy to a node

**Files:**
- Create: `dgxmm/nodes/weights_sync.py`
- Modify: `dgxmm/deployer/service.py` (`prepare`: presence check for non-local nodes), `dgxmm/api/` (new `nodes.py` router: `POST /api/nodes/{node_id}/copy-weights`)
- Test: `tests/test_weights_sync.py`

**Interfaces:**
- Consumes: `NodeProbe.exec(node_id, argv)` (Task 3; the probe mounts the node HF cache at `/hf`), `ModelArtifact.ref/.revision`.
- Produces: `weights_present(probe, node_id, artifact) -> bool`;
  `copy_argv(local_hf: str, node: NodeConfig, repo_dir: str, revision: str) -> tuple[list[str], list[str]]`
  (the `tar` argv and the `ssh … tar -x` argv, piped together); `async copy_weights(node_id, ref, revision) -> CopyJob`
  (admin only, audited, appears in Downloads with `kind="copy"`); `DeployError("weights_missing", …, status=409, action={"type": "copy_weights", "node": id, "ref": ref, "revision": rev})`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_deploy_refuses_when_weights_missing_on_node(deployer_two_nodes, qwen_req):
    deployer_two_nodes.fake("dgx-2").exec_output = ""          # `test -f` → absent
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert e.value.code == "weights_missing" and e.value.extra["action"]["type"] == "copy_weights"

def test_copy_argv_is_two_argv_lists_with_pinned_paths(two_node_settings):
    tar, ssh = copy_argv("/hf-local", two_node_settings.node("dgx-2"), "models--unsloth--Qwen3.8-27B-NVFP4", "57926bac")
    assert tar[:3] == ["tar", "-C", "/hf-local/hub"] and "models--unsloth--Qwen3.8-27B-NVFP4" in tar
    assert ssh[0] == "ssh" and "192.0.2.2" in ssh
    assert ssh[-5:] == ["tar", "-C", "/home/dgxuser/.cache/huggingface/hub", "-xf", "-"]

def test_copy_rejects_path_traversal_ref():
    with pytest.raises(ValueError):
        copy_argv("/hf-local", NODE, "models--../../etc", "main")
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. Only the requested snapshot, its blobs and `refs/` are copied
  (no `blobs/` wildcard); repo dirs are validated with the existing repo regex; the process is
  `asyncio.create_subprocess_exec`, no shell. Measured on 2026-11-14: 22 GB in 36 s.
- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `feat(nodes): refuse deploys without weights; copy weights over the cluster link`.

### Task 7: Reconcile, watchdog and LiteLLM per node

**Files:**
- Modify: `dgxmm/deployer/reconcile.py` (loop over nodes), `dgxmm/watchdog.py` (one pressure check per node; victims only on that node), `dgxmm/litellm.py` (uses `base_url(..., node=dep.node)`)
- Test: `tests/test_nodes_reconcile_watchdog.py`

**Interfaces:**
- Consumes: `NodeRegistry`, `BudgetService.all()`, `Deployer.base_url(..., node=)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_reconcile_adopts_remote_managed_container_by_label(deployer_two_nodes, starting_remote_dep):
    deployer_two_nodes.fake("dgx-2").add_managed(deployment_id=starting_remote_dep.id, name="dgxmm-q2")
    reconcile(deployer_two_nodes)
    assert deployer_two_nodes.count_rows() == 1
    assert deployer_two_nodes.row(starting_remote_dep.id).container_id is not None

def test_reconcile_skips_unreachable_node_without_touching_its_rows(deployer_two_nodes, ready_remote_dep):
    deployer_two_nodes.fake("dgx-2").raise_on_all = NodeUnreachable("down")
    reconcile(deployer_two_nodes)
    assert deployer_two_nodes.state(ready_remote_dep.id) == "ready"

async def test_watchdog_only_stops_victims_on_pressured_node(watchdog_two_nodes):
    watchdog_two_nodes.set_available("dgx-2", 1 * 2**30)
    stopped = await watchdog_two_nodes.tick()
    assert {d.node for d in stopped} == {"dgx-2"}

def test_litellm_entry_for_remote_uses_cluster_ip(litellm_sync_two_nodes, ready_remote_dep):
    entry = litellm_sync_two_nodes.desired()[ready_remote_dep.name]
    assert entry["litellm_params"]["api_base"].startswith("http://192.0.2.2:")
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Commit `feat(nodes): reconcile, watchdog and LiteLLM per node`.

### Task 8: Engine API keys

Remote engines become reachable on the cluster link, so they need the per-engine key the security review asked for.

**Files:**
- Modify: `dgxmm/engines/vllm.py` (`VLLM_API_KEY` env), `dgxmm/engines/llamacpp.py` (`LLAMA_API_KEY` env),
  `dgxmm/engines/base.py:109-113` (`openai_health(..., api_key)` sends `Authorization: Bearer`),
  `dgxmm/litellm.py:257` (`add(..., api_key=dep_key)`), `dgxmm/deployer/service.py` (generate a key per deployment, `secrets.token_urlsafe(32)`, stored in the deployment's `config["_api_key"]`, never returned by the API)
- Test: `tests/test_engine_api_key.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_vllm_spec_sets_api_key_env(vllm_engine, artifact, launch_with_key):
    spec = vllm_engine.container_spec(artifact, VllmConfig(), None, launch_with_key, settings)
    assert spec.environment["VLLM_API_KEY"] == launch_with_key.api_key

async def test_health_sends_bearer(respx_mock):
    route = respx_mock.get("http://e:8100/health").respond(200)
    respx_mock.post("http://e:8100/v1/completions").respond(200, json={"choices": [{"text": "x"}]})
    await openai_health(httpx.AsyncClient(), "http://e:8100", "m", api_key="k")
    assert route.calls[0].request.headers["authorization"] == "Bearer k"

def test_api_key_never_in_deployment_api_response(client_admin, ready_dep):
    assert "_api_key" not in json.dumps(client_admin.get(f"/api/deployments/{ready_dep.id}").json())
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement; on-demand gateway and the `sleep`/`wake` calls also send the key.
- [ ] **Step 4:** Run → PASS; full suite green. **Step 5:** Commit `feat(engines): per-deployment API key`.

### Task 9: Container specs for host networking and RDMA

**Files:**
- Modify: `dgxmm/engines/base.py:28-52` (`ContainerSpec`), `dgxmm/docker_ctl.py:197` (`run` maps the new fields)
- Test: `tests/test_docker_ctl_hostnet.py`

**Interfaces:**
- Produces: `ContainerSpec.network_mode: str | None = None` (`"host"` → no network attach, no port publish),
  `devices: list[str] = []`, `cap_add: list[str] = []`, `ulimits: dict[str, tuple[int, int]] = {}`, `shm_size: str | None = None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_maps_host_network_devices_caps_ulimits(fake_docker):
    DockerController(fake_docker).run(spec(network_mode="host", devices=["/dev/infiniband"], cap_add=["IPC_LOCK"],
                                           ulimits={"memlock": (-1, -1)}, shm_size="16g"))
    kw = fake_docker.containers.last_run_kwargs
    assert kw["network_mode"] == "host" and kw["cap_add"] == ["IPC_LOCK"] and kw["shm_size"] == "16g"
    assert kw["devices"] == ["/dev/infiniband:/dev/infiniband:rwm"]
    assert [(u.name, u.soft, u.hard) for u in kw["ulimits"]] == [("memlock", -1, -1)]

def test_host_network_spec_publishes_no_ports(fake_docker):
    DockerController(fake_docker).run(spec(network_mode="host"))
    assert "ports" not in fake_docker.containers.last_run_kwargs
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (defaults keep every existing spec byte-for-byte the same).
- [ ] **Step 4:** Run → PASS; full suite green. **Step 5:** Commit `feat(docker): host networking and RDMA fields in ContainerSpec`.

### Task 10: Placement model (single or cluster), migration 42

**Files:**
- Modify: `dgxmm/deployer/service.py:49` (`Placement`, `DeployRequest.placement`), `dgxmm/db.py` (`Deployment.placement` JSON,
  `Deployment.ranks` JSON), `dgxmm/migrations.py` (`Migration(42, "deployments: placement and ranks", _deployment_placement)`)
- Test: `tests/test_placement_model.py`

**Interfaces:**
- Produces: `class Placement(BaseModel)`: `kind: Literal["single", "cluster"]`, `nodes: list[str]` (head first);
  properties `head -> str`, `workers -> list[str]`, `label(settings) -> str` (`"dgx-2"`, `"dgx-1 + dgx-2 (cluster)"`).
- Produces: `DeployRequest.placement: Placement | None = None`; a validator maps `node` → `Placement("single", [node])`
  and defaults to `single` on `local`. `Deployment.node` stays and holds the head node.
- Produces: `Deployment.ranks: list[dict]`, each `{"node", "rank", "container_id", "port"}` (empty for single deployments).
- Produces: `DeployError` codes `cluster_needs_two_nodes`, `cluster_node_not_ready` (no `cluster_ip`/`nccl`), `cluster_unsupported_engine`.

- [ ] **Step 1: Write the failing tests**

```python
def test_placement_defaults_to_local_single():
    assert DeployRequest(name="a", engine="vllm", model_ref="x/y").placement == Placement(kind="single", nodes=["local"])

def test_node_field_maps_to_single_placement():
    assert DeployRequest(name="a", engine="vllm", model_ref="x/y", node="dgx-2").placement.nodes == ["dgx-2"]

def test_cluster_requires_two_distinct_nodes_with_nccl(deployer_two_nodes, qwen_req):
    with pytest.raises(DeployError, match="cluster_needs_two_nodes"):
        deployer_two_nodes.validate_placement(Placement(kind="cluster", nodes=["local", "local"]), "vllm")

def test_migration_42_backfills_single_placement(v41_db_with_one_deployment):
    migrate(v41_db_with_one_deployment)
    assert json.loads(row(v41_db_with_one_deployment, "SELECT placement FROM deployments")[0]) == \
        {"kind": "single", "nodes": ["local"]}
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Commit `feat(cluster): placement model (migration 42)`.

### Task 11: Per-rank estimates and placement options (the "it is a big model" suggestion)

**Files:**
- Create: `dgxmm/planner/placement.py`
- Modify: `dgxmm/memory/estimate.py` (per-rank estimate), `dgxmm/api/deployments.py` (`POST /api/deployments/placement-options`)
- Test: `tests/test_placement_options.py`

**Interfaces:**
- Consumes: `BudgetService.all()` (Task 3), `Placement` (Task 10), the engine's single-node `estimate(...)`.
- Produces: `rank_bytes(engine, estimate: MemoryEstimate, placement: Placement, budgets: dict[str, Budget]) -> dict[str, int]`.
  vLLM: `weights/tp + kv/tp + runtime` on each rank, where `tp = len(nodes)`. llama.cpp: weights and KV split in
  proportion to each node's `free_for_new`, plus `runtime` on each node.
- Produces: `class PlacementOption`: `placement: Placement`, `label: str`, `per_node: dict[str, int]`, `free: dict[str, int]`,
  `fits: bool`, `recommended: bool`, `reason: str` (`"fits"`, `"too_big_for_one_node"`, `"insufficient_memory"`).
- Produces: `placement_options(req: DeployRequest) -> list[PlacementOption]`: every single node, plus one cluster
  option over all cluster-ready nodes (local head). Recommended: the fitting single node with the most free memory;
  else the cluster if it fits; else none.

- [ ] **Step 1: Write the failing tests**

```python
GiB = 2**30

def test_big_model_recommends_cluster(options_two_nodes):
    opts = options_two_nodes(model_gib=167, free_gib={"local": 110, "dgx-2": 112})
    rec = [o for o in opts if o.recommended]
    assert [o.placement.kind for o in rec] == ["cluster"]
    assert all(not o.fits and o.reason == "too_big_for_one_node" for o in opts if o.placement.kind == "single")

def test_small_model_recommends_single_node_with_most_free(options_two_nodes):
    opts = options_two_nodes(model_gib=22, free_gib={"local": 40, "dgx-2": 110})
    assert [o.placement.nodes for o in opts if o.recommended] == [["dgx-2"]]

def test_nothing_fits_recommends_none(options_two_nodes):
    opts = options_two_nodes(model_gib=510, free_gib={"local": 110, "dgx-2": 110})
    assert not any(o.recommended for o in opts)
    assert all(o.reason == "insufficient_memory" for o in opts if o.placement.kind == "cluster")

def test_vllm_rank_bytes_halve_weights_and_kv():
    est = MemoryEstimate(weights=100 * GiB, kv=20 * GiB, runtime=4 * GiB)
    assert rank_bytes("vllm", est, Placement(kind="cluster", nodes=["local", "dgx-2"]), BUDGETS) == \
        {"local": 64 * GiB, "dgx-2": 64 * GiB}
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement (match the field names of the real `MemoryEstimate`; the
  test's constructor names are placeholders for them). Run in a worker thread from the async handler, so it cannot
  stall the event loop (review finding: planner endpoint).
- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `feat(cluster): per-rank estimates and placement options`.

### Task 12: vLLM cluster deployments (launch, monitor, stop, reconcile)

**Files:**
- Create: `dgxmm/deployer/cluster.py` (rank lifecycle helpers used by `service.py`)
- Modify: `dgxmm/engines/vllm.py` (`cluster_specs`), `dgxmm/deployer/service.py` (`_launch`, `_monitor`, `stop`, `check_liveness`
  dispatch on `placement.kind`), `dgxmm/deployer/reconcile.py`, `dgxmm/config.py` (`Settings.cluster_port_range = (29500, 29599)`)
- Test: `tests/test_cluster_vllm.py`

**Interfaces:**
- Consumes: Tasks 2, 3, 9, 10, 11. `NodeConfig.nccl`, `NodeConfig.cluster_ip`.
- Produces: `VllmEngine.cluster_specs(artifact, cfg, draft, launch: LaunchContext, placement: Placement, nodes: NodeRegistry,
  master_port: int, settings) -> list[RankSpec]`, with `RankSpec(node: str, rank: int, spec: ContainerSpec)`.
  Rank 0 (head) serves the API on `--host <head cluster_ip> --port <port>`; every rank gets
  `--tensor-parallel-size N --nnodes N --node-rank r --master-addr <head cluster_ip> --master-port <master_port>`, and
  ranks > 0 add `--headless`. Specs: `network_mode="host"`, `devices=["/dev/infiniband"]`, `cap_add=["IPC_LOCK"]`,
  `ulimits={"memlock": (-1, -1)}`, `shm_size="16g"`; env `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`,
  `VLLM_HOST_IP=<that node's cluster_ip>`, plus the Task 8 API key. `--gpu-memory-utilization` per rank comes from that
  node's reservation and memory total.
- Produces: the start order is workers, then the head; health checks on the head only; stop order is the head, then workers.

- [ ] **Step 1: Write the failing tests**

```python
def test_cluster_specs_rank_flags(vllm_engine, cluster_launch):
    head, worker = vllm_engine.cluster_specs(**cluster_launch)
    assert (head.node, head.rank, worker.node, worker.rank) == ("local", 0, "dgx-2", 1)
    assert "--headless" not in head.spec.command and "--headless" in worker.spec.command
    for r in (head, worker):
        c = r.spec.command
        assert c[c.index("--master-addr") + 1] == "192.0.2.1" and c[c.index("--nnodes") + 1] == "2"
        assert r.spec.network_mode == "host" and r.spec.environment["NCCL_IB_HCA"] == "rocep1s0f0,roceP2p1s0f0"
    assert worker.spec.environment["VLLM_HOST_IP"] == "192.0.2.2"

async def test_cluster_deploy_creates_one_container_per_node(deployer_two_nodes, deepseek_cluster_req):
    await deployer_two_nodes.deploy(deepseek_cluster_req, user_id=1)
    assert [len(deployer_two_nodes.fake(n).containers.list()) for n in ("local", "dgx-2")] == [1, 1]

async def test_worker_exit_fails_whole_deployment_and_stops_head(deployer_two_nodes, ready_cluster_dep):
    deployer_two_nodes.fake("dgx-2").exit_container(ready_cluster_dep.ranks[1]["container_id"], code=1)
    await deployer_two_nodes.monitor_tick(ready_cluster_dep.id)
    assert deployer_two_nodes.state(ready_cluster_dep.id) == "failed"
    assert deployer_two_nodes.fake("local").containers.list() == []

async def test_cluster_reservation_kept_until_all_ranks_gone(deployer_two_nodes, ready_cluster_dep):
    deployer_two_nodes.fake("local").fail_remove = True
    await deployer_two_nodes.stop(ready_cluster_dep.id)
    assert deployer_two_nodes.budget("local").committed >= ready_cluster_dep.ranks_reserved("local")

def test_reconcile_adopts_cluster_ranks(deployer_two_nodes, starting_cluster_dep):
    for rank in starting_cluster_dep.ranks:
        deployer_two_nodes.fake(rank["node"]).add_managed(deployment_id=starting_cluster_dep.id, rank=rank["rank"])
    reconcile(deployer_two_nodes)
    assert deployer_two_nodes.count_rows() == 1 and all(r["container_id"] for r in deployer_two_nodes.row(starting_cluster_dep.id).ranks)

def test_cluster_litellm_entry_points_to_head(litellm_sync_two_nodes, ready_cluster_dep):
    assert litellm_sync_two_nodes.desired()[ready_cluster_dep.name]["litellm_params"]["api_base"].startswith("http://192.0.2.1:")
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Rank containers are named `dgxmm-<name>-r<rank>` and labelled with
  `dgxmm.deployment_id` and `dgxmm.rank`. A failure path keeps the deployment `stopping` (reservation held) and
  retries removal with backoff until every rank is confirmed gone, and only then sets `failed`/`stopped`.
- [ ] **Step 4:** Run → PASS; full suite green. **Step 5:** Commit `feat(cluster): vLLM tensor-parallel deployments across nodes`.

### Task 13: llama.cpp cluster deployments (RPC)

**Files:**
- Modify: `dgxmm/engines/llamacpp.py` (`cluster_specs`), `dgxmm/config.py` (`Images.llamacpp_rpc`, pinned by digest, built with `GGML_RPC=ON`)
- Test: `tests/test_cluster_llamacpp.py`

**Interfaces:**
- Consumes: Task 12's `RankSpec`, the rank lifecycle in `dgxmm/deployer/cluster.py`, and Task 11's `rank_bytes` for llama.cpp.
- Produces: `LlamacppEngine.cluster_specs(artifact, cfg, launch, placement, nodes, rpc_port: int, settings) -> list[RankSpec]`.
  Workers run `ggml-rpc-server --host <cluster_ip> -p <rpc_port> -c` (cache under that node's HF cache dir). The
  head runs `llama-server ... --rpc <ip:port,...> --tensor-split <a,b>`, with the split from `rank_bytes`.

- [ ] **Step 0: Hardware check.** Confirm the pinned image has `ggml-rpc-server` with RPC enabled
  (`docker run --rm --entrypoint ls <image> /app`). If it doesn't, add `deploy/llamacpp-rpc/Dockerfile` building
  `ggml-rpc-server` and `llama-server` (CUDA, `GGML_RPC=ON`, `CMAKE_CUDA_ARCHITECTURES=121`), and pin it by digest.
- [ ] **Step 1: Write the failing tests**

```python
def test_llamacpp_cluster_specs_bind_rpc_to_cluster_ip(llama_engine, cluster_launch):
    head, worker = llama_engine.cluster_specs(**cluster_launch)
    assert worker.spec.command[:5] == ["ggml-rpc-server", "--host", "192.0.2.2", "-p", "50052"]
    assert head.spec.command[head.spec.command.index("--rpc") + 1] == "192.0.2.2:50052"

def test_tensor_split_from_budgets(llama_engine, cluster_launch_free_60_40):
    head, _ = llama_engine.cluster_specs(**cluster_launch_free_60_40)
    assert head.spec.command[head.spec.command.index("--tensor-split") + 1] == "60,40"
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Commit `feat(cluster): llama.cpp RPC deployments across nodes`.

### Task 14: Feature parity for remote and cluster deployments

**Files:**
- Modify: `dgxmm/ondemand/controller.py`, `dgxmm/watchdog.py`, `dgxmm/decider/schedule.py`, `dgxmm/deployer/service.py`
  (`sleep`, `wake`, `switch`, `reconfigure`), `dgxmm/planner/plans.py`, `dgxmm/decider/snapshot.py` (each deployment row carries `placement`)
- Test: `tests/test_cluster_parity.py`

**Interfaces:**
- Consumes: Tasks 10-13.
- Produces: `Deployer.supports_sleep(dep) -> bool`. True for single deployments as today. For cluster
  deployments it depends on the result of Step 0.

- [ ] **Step 0: Hardware check (decides sleep for clusters).** On dgx-1 and dgx-2, run the Task 12 recipe by hand with
  `--enable-sleep-mode` and `VLLM_SERVER_DEV_MODE=1`, then `POST /sleep?level=1` and `POST /wake_up` on the head.
  Record each node's `MemAvailable` before and after, and a completion after the wake.
  - If both nodes free their memory and the completion after the wake is correct: `supports_sleep` is true for vLLM clusters.
  - Otherwise it is false, and cluster sleep is a stop of every rank. The UI labels it "sleep (stops both nodes)".
  - Write the result into the design doc under Goals 8.
- [ ] **Step 1: Write the failing tests**

```python
async def test_ondemand_reap_stops_all_ranks(ondemand_two_nodes, idle_cluster_dep):
    await ondemand_two_nodes.reap_tick()
    assert all(ondemand_two_nodes.fake(n).containers.list() == [] for n in ("local", "dgx-2"))

async def test_ondemand_start_waits_for_head_health(ondemand_two_nodes, stopped_cluster_dep):
    resp = await ondemand_two_nodes.request(stopped_cluster_dep.name)
    assert resp.status == 200 and ondemand_two_nodes.health_calls == [("192.0.2.1", stopped_cluster_dep.port)]

async def test_watchdog_cluster_victim_frees_both_nodes(watchdog_two_nodes, low_priority_cluster_dep):
    watchdog_two_nodes.set_available("local", 1 * 2**30)
    stopped = await watchdog_two_nodes.tick()
    assert [d.id for d in stopped] == [low_priority_cluster_dep.id]
    assert watchdog_two_nodes.budget("dgx-2").committed == 0

async def test_schedule_stop_then_wake_cluster(scheduler_two_nodes, ready_cluster_dep):
    await scheduler_two_nodes.run_due(at="22:00")   # stop
    await scheduler_two_nodes.run_due(at="07:00")   # wake
    assert scheduler_two_nodes.state(ready_cluster_dep.id) == "ready"

async def test_switch_single_to_cluster(deployer_two_nodes, ready_single_dep, deepseek_cluster_req):
    await deployer_two_nodes.switch(ready_single_dep.id, deepseek_cluster_req, actions=["stop"], user_id=1)
    assert deployer_two_nodes.row_by_name(deepseek_cluster_req.name).placement["kind"] == "cluster"

async def test_cluster_sleep_follows_hardware_check(deployer_two_nodes, ready_cluster_dep):
    await deployer_two_nodes.sleep(ready_cluster_dep.id)
    expected = "sleeping" if deployer_two_nodes.supports_sleep(ready_cluster_dep) else "stopped"
    assert deployer_two_nodes.state(ready_cluster_dep.id) == expected
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. The watchdog's victim choice counts a cluster deployment's
  memory on the pressured node; stopping it frees every node. **Step 4:** Run → PASS; full suite green.
- [ ] **Step 5:** Commit `feat(cluster): on-demand, schedules, watchdog, switch and sleep for remote and cluster deployments`.

### Task 15: API and UI (where each model runs; choose placement when deploying)

**Files:**
- Modify: `dgxmm/api/nodes.py` (`GET /api/nodes` → `[{id, label, remote, reachable, stale, budget}]`),
  `dgxmm/api/deployments.py` (responses carry `placement` and `placement_label`)
- Modify: `frontend/src/pages/DeployWizard.tsx` (a **Where to run** step from `placement-options`: one card per option
  with memory need vs free per node; badges "Recommended", "Does not fit", "Too big for one DGX"; the recommended
  option pre-selected; the step is hidden when there is only one node), `frontend/src/pages/Deployments.tsx`
  (a **Runs on** column: `dgx-1`, `dgx-2`, `dgx-1 + dgx-2 (cluster)`), `frontend/src/pages/DeploymentDetail.tsx`
  (each rank with its node and state), `frontend/src/pages/Dashboard.tsx` (one memory bar per node; a cluster
  deployment's share appears on both)
- Test: `tests/test_api_nodes.py`, `frontend/src/pages/DeployWizard.placement.test.tsx`, `frontend/src/pages/Deployments.placement.test.tsx`

- [ ] **Step 1: Write the failing tests**

```python
def test_get_nodes_lists_both_with_budget(client_viewer_two_nodes):
    r = client_viewer_two_nodes.get("/api/nodes").json()
    assert [n["id"] for n in r] == ["local", "dgx-2"] and "free_for_new" in r[1]["budget"]

def test_deployment_response_has_placement_label(client_admin_two_nodes, ready_cluster_dep):
    d = client_admin_two_nodes.get(f"/api/deployments/{ready_cluster_dep.id}").json()
    assert d["placement_label"] == "dgx-1 + dgx-2 (cluster)"
```

```tsx
it("pre-selects the cluster for a model too big for one DGX", async () => {
  mockPlacementOptions([opt("single", ["local"], { fits: false, reason: "too_big_for_one_node" }),
                        opt("single", ["dgx-2"], { fits: false, reason: "too_big_for_one_node" }),
                        opt("cluster", ["local", "dgx-2"], { fits: true, recommended: true })]);
  render(<DeployWizard />); await pickModel("deepseek-ai/DeepSeek-V4-Flash-0731");
  expect(screen.getByRole("radio", { name: /dgx-1 \+ dgx-2/ })).toBeChecked();
  expect(screen.getAllByText("Too big for one DGX")).toHaveLength(2);
  await user.click(screen.getByText("Deploy"));
  expect(lastPost("/api/deployments").placement).toEqual({ kind: "cluster", nodes: ["local", "dgx-2"] });
});
it("hides the step with a single node", async () => {
  mockNodes([{ id: "local", label: "dgx-1" }]); render(<DeployWizard />);
  expect(screen.queryByText("Where to run")).toBeNull();
});
it("shows where each deployment runs", async () => {
  mockDeployments([dep("q", "dgx-2"), dep("ds", "dgx-1 + dgx-2 (cluster)")]); render(<Deployments />);
  expect(await screen.findByText("dgx-1 + dgx-2 (cluster)")).toBeInTheDocument();
});
```

- [ ] **Step 2:** Run `pytest tests/test_api_nodes.py` and `cd frontend && npx vitest run placement` → FAIL.
- [ ] **Step 3:** Implement. **Step 4:** Both → PASS; `npm run build` OK.
- [ ] **Step 5:** Commit `feat(ui): where each model runs; choose single DGX or cluster when deploying`.

### Task 16: Docs and on-hardware acceptance

**Files:** `docs/configuration/` (nodes section), `docs/security/overview.md` (the SSH key gives root-equivalent Docker
on dgx-2; RPC and NCCL ports only on the cluster link), `docs/user-guide/` (where to run; copying weights; cluster
limits from Task 14 Step 0), `CHANGELOG.md`.

- [ ] **Step 1:** Write the docs; `mkdocs build --strict` passes.
- [ ] **Step 2: Acceptance on the real boxes** (configure `dgx-2` with cluster_ip `192.0.2.2`, user `dgxuser`,
  and a known_hosts line from `ssh-keyscan -t ed25519 192.0.2.2`, checked against the host). Through the UI:
  - `unsloth/Qwen3.8-27B-NVFP4` on **dgx-2** → `ready`, "Runs on: dgx-2", about 11.6 tok/s.
  - `deepseek-ai/DeepSeek-V4-Flash-0731` → the wizard pre-selects **dgx-1 + dgx-2 (cluster)** → `ready`; NCCL logs show `NET/IB`.
  - `unsloth/Qwen3.8-27B-NVFP4` as a **cluster** → at least 18 tok/s (20.3 measured by hand on 2026-11-14).
  - GLM-4.7 UD-Q4_K_XL as a **llama.cpp cluster** → serves at about 9 tok/s or more.
  - LiteLLM on dgx-1 lists all of them; a chat through LiteLLM works for each.
  - Existing features still work: on-demand start/idle stop on a cluster model, a schedule, the watchdog under memory
    pressure, and switch.
- [ ] **Step 3:** Pull dgx-2's cable while a cluster model runs → that deployment fails cleanly (both ranks gone),
  single deployments on dgx-1 are untouched, and dgx-2 shows stale. Plug it back in → fresh; redeploy works.
- [ ] **Step 4:** Commit `docs(nodes): configuration, security and user guide`.
