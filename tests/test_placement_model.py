"""Tests for the placement model (single or cluster) — Task 10."""
import json
import sqlite3
import pytest
from sqlalchemy import select
from dgxmm.db import Deployment
from dgxmm.deployer.service import DeployError, DeployRequest, Placement
from dgxmm.migrations import migrate


def test_placement_defaults_to_local_single():
    """Default placement is single-node on local."""
    req = DeployRequest(name="a", engine="vllm", model_ref="x/y")
    assert req.placement == Placement(kind="single", nodes=["local"])


def test_node_field_maps_to_single_placement():
    """node field maps to a single-node placement."""
    req = DeployRequest(name="a", engine="vllm", model_ref="x/y", node="dgx-2")
    assert req.placement.nodes == ["dgx-2"]
    assert req.placement.kind == "single"


def test_explicit_placement_overrides_node():
    """Explicit placement field takes precedence."""
    req = DeployRequest(name="a", engine="vllm", model_ref="x/y",
                        placement=Placement(kind="single", nodes=["dgx-3"]))
    assert req.placement.nodes == ["dgx-3"]


def test_placement_head_property():
    """placement.head returns the first node."""
    p = Placement(kind="cluster", nodes=["dgx-1", "dgx-2"])
    assert p.head == "dgx-1"


def test_placement_workers_property():
    """placement.workers returns nodes after the head."""
    p = Placement(kind="cluster", nodes=["dgx-1", "dgx-2", "dgx-3"])
    assert p.workers == ["dgx-2", "dgx-3"]


def test_placement_label_single(two_node_settings):
    """Single placement label is the node label."""
    p = Placement(kind="single", nodes=["local"])
    assert p.label(two_node_settings) == "dgx-1"


def test_placement_label_cluster(two_node_settings):
    """Cluster placement label joins node labels."""
    p = Placement(kind="cluster", nodes=["local", "dgx-2"])
    label = p.label(two_node_settings)
    assert "dgx-1" in label
    assert "dgx-2" in label
    assert "cluster" in label.lower()


def test_cluster_requires_two_distinct_nodes(deployer_two_nodes, qwen_req):
    """Cluster with duplicate nodes is rejected."""
    placement = Placement(kind="cluster", nodes=["local", "local"])
    with pytest.raises(DeployError) as e:
        deployer_two_nodes.validate_placement(placement, "vllm")
    assert e.value.code == "cluster_needs_two_nodes"


def test_cluster_requires_configured_nodes(deployer_two_nodes, qwen_req):
    """Cluster with unknown node is rejected."""
    placement = Placement(kind="cluster", nodes=["local", "unknown-node"])
    with pytest.raises(DeployError) as e:
        deployer_two_nodes.validate_placement(placement, "vllm")
    assert e.value.code == "cluster_node_not_ready"


def test_cluster_only_for_vllm_and_llamacpp(deployer_two_nodes, qwen_req):
    """Cluster is only supported for vllm and llamacpp engines."""
    placement = Placement(kind="cluster", nodes=["local", "dgx-2"])
    with pytest.raises(DeployError) as e:
        deployer_two_nodes.validate_placement(placement, "tgi")
    assert e.value.code == "cluster_unsupported_engine"


async def test_cluster_deploy_not_implemented(deployer_two_nodes, qwen_req):
    """Cluster deploy raises not-implemented error until Task 12."""
    req = DeployRequest(name="test", engine="vllm", model_ref="Qwen/Qwen-7B",
                        placement=Placement(kind="cluster", nodes=["local", "dgx-2"]))
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(req, None)
    assert e.value.code == "cluster_not_implemented"


def test_migration_43_backfills_single_placement(tmp_path):
    """Migration 43 backfills placement and ranks for v42 rows."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)")
    conn.execute("""CREATE TABLE deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            engine TEXT NOT NULL,
            model_ref TEXT NOT NULL,
            config TEXT NOT NULL,
            state TEXT NOT NULL,
            container_id TEXT,
            port INTEGER,
            priority INTEGER DEFAULT 0,
            alias TEXT,
            reserved_bytes INTEGER DEFAULT 0,
            estimate TEXT NOT NULL,
            litellm_status TEXT DEFAULT 'none',
            last_error TEXT,
            exit_code INTEGER,
            oom_killed INTEGER,
            adopted_from TEXT,
            system INTEGER,
            mode TEXT DEFAULT 'always_on',
            idle_timeout_s INTEGER,
            node TEXT DEFAULT 'local',
            engine_api_key TEXT
        )""")
    
    conn.execute("INSERT INTO deployments (name, engine, model_ref, config, state, estimate, node) VALUES ('test', 'vllm', 'x/y', '{}', 'running', '{}', 'local')")
    conn.commit()
    
    conn.execute("INSERT INTO schema_migrations VALUES (42, 'deployments: engine API key', datetime('now'))")
    conn.commit()
    conn.close()
    
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    applied = migrate(engine)
    assert 43 in applied
    
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT placement, ranks FROM deployments WHERE name='test'").fetchone()
    placement = json.loads(row[0])
    ranks = json.loads(row[1])
    
    assert placement == {"kind": "single", "nodes": ["local"]}
    assert ranks == []
    conn.close()


def test_migration_43_backfills_placement_from_node_column(tmp_path):
    """Migration 43 uses the node column value for backfill."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)")
    conn.execute("""CREATE TABLE deployments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            engine TEXT NOT NULL,
            model_ref TEXT NOT NULL,
            config TEXT NOT NULL,
            state TEXT NOT NULL,
            container_id TEXT,
            port INTEGER,
            priority INTEGER DEFAULT 0,
            alias TEXT,
            reserved_bytes INTEGER DEFAULT 0,
            estimate TEXT NOT NULL,
            litellm_status TEXT DEFAULT 'none',
            last_error TEXT,
            exit_code INTEGER,
            oom_killed INTEGER,
            adopted_from TEXT,
            system INTEGER,
            mode TEXT DEFAULT 'always_on',
            idle_timeout_s INTEGER,
            node TEXT DEFAULT 'local',
            engine_api_key TEXT
        )""")
    
    conn.execute("INSERT INTO deployments (name, engine, model_ref, config, state, estimate, node) VALUES ('test', 'vllm', 'x/y', '{}', 'running', '{}', 'dgx-2')")
    conn.commit()
    
    conn.execute("INSERT INTO schema_migrations VALUES (42, 'deployments: engine API key', datetime('now'))")
    conn.commit()
    conn.close()
    
    from sqlalchemy import create_engine
    engine = create_engine(f"sqlite:///{db_path}")
    migrate(engine)
    
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT placement FROM deployments WHERE name='test'").fetchone()
    placement = json.loads(row[0])
    
    assert placement == {"kind": "single", "nodes": ["dgx-2"]}
    conn.close()


def test_node_placement_conflict_explicit():
    """Explicit conflicting node and placement is rejected."""
    with pytest.raises(ValueError, match="node.*placement"):
        DeployRequest(name="a", engine="vllm", model_ref="x/y", 
                      node="dgx-2", placement=Placement(kind="single", nodes=["local"]))


def test_node_placement_default_accepts_placement():
    """Default node with explicit placement sets node from placement."""
    req = DeployRequest(name="a", engine="vllm", model_ref="x/y",
                        placement=Placement(kind="single", nodes=["dgx-2"]))
    assert req.node == "dgx-2"


def test_node_placement_from_json():
    """JSON input with conflicting node and placement is rejected."""
    with pytest.raises(ValueError, match="node.*placement"):
        DeployRequest.model_validate({
            "name": "a", "engine": "vllm", "model_ref": "x/y",
            "node": "dgx-2", "placement": {"kind": "single", "nodes": ["local"]}
        })


async def test_deploy_creates_row_with_placement(deployer_two_nodes, qwen_req):
    """Fresh deployment row has placement and ranks."""
    await deployer_two_nodes.deploy(qwen_req, 1)
    with deployer_two_nodes.db.session() as s:
        dep = s.scalar(select(Deployment).where(Deployment.name == qwen_req.name))
        assert dep.placement == {"kind": "single", "nodes": ["local"]}
        assert dep.ranks == []


async def test_view_includes_placement(deployer_two_nodes, qwen_req):
    """Deployment view includes placement and placement_label."""
    result = await deployer_two_nodes.deploy(qwen_req, 1)
    view = deployer_two_nodes.view(result["id"])
    assert view["placement"] == {"kind": "single", "nodes": ["local"]}
    assert view["placement_label"] == "dgx-1"


def test_view_cluster_placement_label(deployer_two_nodes):
    """Cluster deployment view has correct placement_label."""
    # Create a cluster row directly in DB
    with deployer_two_nodes.db.session() as s:
        dep = Deployment(
            name="cluster-test",
            engine="vllm",
            model_ref="test/model",
            config={},
            state="ready",
            estimate={},
            node="local",
            placement={"kind": "cluster", "nodes": ["local", "dgx-2"]},
            ranks=[]
        )
        s.add(dep)
        s.commit()
        dep_id = dep.id
    
    view = deployer_two_nodes.view(dep_id)
    assert view["placement"]["kind"] == "cluster"
    assert "dgx-1" in view["placement_label"]
    assert "dgx-2" in view["placement_label"]
    assert "cluster" in view["placement_label"].lower()


def test_view_null_placement_reads_as_single(deployer_two_nodes):
    """NULL placement reads as single on row's node."""
    # Create a row with NULL placement
    with deployer_two_nodes.db.session() as s:
        dep = Deployment(
            name="null-placement",
            engine="vllm",
            model_ref="test/model",
            config={},
            state="ready",
            estimate={},
            node="dgx-2",
            placement=None,
            ranks=None
        )
        s.add(dep)
        s.commit()
        dep_id = dep.id
    
    view = deployer_two_nodes.view(dep_id)
    assert view["placement"] == {"kind": "single", "nodes": ["dgx-2"]}
    assert view["placement_label"] == "dgx-2"


def test_system_deployment_cluster_rejected_in_validation(deployer_two_nodes):
    """System deployments with cluster placement are rejected during validation."""
    placement = Placement(kind="cluster", nodes=["local", "dgx-2"])
    with pytest.raises(DeployError) as e:
        deployer_two_nodes.validate_placement(placement, "decision", system=True)
    assert e.value.code == "cluster_unsupported"

