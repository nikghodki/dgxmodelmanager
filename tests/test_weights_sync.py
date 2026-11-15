"""Task 6: Weights presence check and copy to a node."""
import pytest

from dgxmm.deployer.service import DeployError
from dgxmm.nodes.weights_sync import copy_argv, weights_present


async def test_deploy_refuses_when_weights_missing_on_node(deployer_two_nodes, qwen_req):
    deployer_two_nodes.fake("dgx-2").exec_output = ""          # `test -f` → absent
    deployer_two_nodes.fake("dgx-2").exec_code = 1
    with pytest.raises(DeployError) as e:
        await deployer_two_nodes.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    assert e.value.code == "weights_missing" and e.value.extra["action"]["type"] == "copy_weights"


def test_copy_argv_is_two_argv_lists_with_pinned_paths(two_node_settings):
    tar, ssh = copy_argv("/hf-local", two_node_settings.node("dgx-2"), "models--unsloth--Qwen3.8-27B-NVFP4", "57926bac")
    assert tar[:3] == ["tar", "-C", "/hf-local/hub"]
    # Check that the repo name appears in some tar argument
    assert any("models--unsloth--Qwen3.8-27B-NVFP4" in arg for arg in tar)
    assert ssh[0] == "ssh" and any("192.0.2.2" in arg for arg in ssh)
    assert ssh[-5:] == ["tar", "-C", "/home/dgxuser/.cache/huggingface/hub", "-xf", "-"]


def test_copy_rejects_path_traversal_ref():
    from dgxmm.config import NodeConfig
    node = NodeConfig(id="test", label="test", docker_url="unix://", hf_cache_host="/cache")
    with pytest.raises(ValueError):
        copy_argv("/hf-local", node, "models--../../etc", "main")


def test_copy_rejects_absolute_path_in_repo():
    from dgxmm.config import NodeConfig
    node = NodeConfig(id="test", label="test", docker_url="unix://", hf_cache_host="/cache")
    with pytest.raises(ValueError):
        copy_argv("/hf-local", node, "/etc/passwd", "main")


def test_copy_rejects_invalid_repo_name():
    from dgxmm.config import NodeConfig
    node = NodeConfig(id="test", label="test", docker_url="unix://", hf_cache_host="/cache")
    with pytest.raises(ValueError):
        copy_argv("/hf-local", node, "no-slash-invalid", "main")


def test_weights_present_checks_snapshot_files(deployer_two_nodes, qwen_req):
    """weights_present uses NodeProbe.exec to check for snapshot files."""
    from dgxmm.nodes.probe import NodeProbe
    from dgxmm.inventory.artifact import ModelArtifact
    
    probe = NodeProbe(deployer_two_nodes.nodes, deployer_two_nodes.settings)
    artifact = ModelArtifact(
        ref="Qwen/Qwen3-Tiny",
        source="hf",
        local_path="/path/to/snapshot",
        revision="abc123",
        format="safetensors",
        weights_bytes=1000,
        size_bytes=1000,
        complete=True,
        kind="llm",
        arch=None,
        architecture=None,
        quant=None,
        files=["config.json", "model.safetensors"]
    )
    
    # Set up probe container in fake Docker
    deployer_two_nodes.fake("dgx-2").add_unmanaged(
        "dgxmm-node-probe",
        image="probe-image:latest",
        status="running"
    )
    
    # Mock exec to return success (all files present)
    deployer_two_nodes.fake("dgx-2").exec_output = ""
    deployer_two_nodes.fake("dgx-2").exec_code = 0
    
    assert weights_present(probe, "dgx-2", artifact) is True
    
    # Now simulate missing weights
    deployer_two_nodes.fake("dgx-2").exec_code = 1
    assert weights_present(probe, "dgx-2", artifact) is False


@pytest.mark.skip("Not implementing full copy job yet - placeholder")
async def test_copy_weights_creates_download_job(deployer_two_nodes):
    """POST /api/nodes/{node_id}/copy-weights creates a Download job with kind=copy."""
    pass
