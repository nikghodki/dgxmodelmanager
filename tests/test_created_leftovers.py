"""A container whose start fails stays in docker state "created" (docker-py's containers.run = create + start). DGXMM
must never leave such containers behind: not on a failed start, not when the deployment is stopped, removed or
rolled back, and not when the decider is disabled."""
import pytest

from dgxmm.deployer.service import DeployError, DeployRequest
from dgxmm.engines.base import container_name, managed_labels

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name="tiny"):
    return DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL, alias=name)


def names(stack):
    return {c.name: c.status for c in stack.sdk._containers.values()}


def test_controller_removes_a_container_whose_start_failed(stack):
    from dgxmm.engines.base import ContainerSpec
    stack.sdk.fail_start.add("dgxmm-x")
    spec = ContainerSpec(name="dgxmm-x", image="img", command=[], environment={}, mounts=[],
                         labels=managed_labels(1, "vllm"), network="dgxmm", port=8100, publish_bind="127.0.0.1")
    with pytest.raises(Exception):
        stack.deployer.docker.run(spec)
    assert "dgxmm-x" not in names(stack)


async def test_failed_start_leaves_no_created_container(stack):
    stack.sdk.fail_start.add(container_name("tiny"))
    with pytest.raises(DeployError) as e:
        await stack.deployer.deploy(req(), 1)
    assert e.value.code == "docker_error"
    assert container_name("tiny") not in names(stack)
    dep = next(v for v in stack.deployer.list_views() if v["name"] == "tiny")
    assert dep["state"] == "failed"
    await stack.deployer.remove(dep["id"])


async def test_remove_and_stop_clean_up_a_leftover_created_container(stack):
    stack.sdk.fail_start.add(container_name("tiny"))
    with pytest.raises(DeployError):
        await stack.deployer.deploy(req(), 1)
    dep = next(v for v in stack.deployer.list_views() if v["name"] == "tiny")
    # a leftover from an older DGXMM (or a crash between create and start): no container_id in the DB
    stack.sdk.add_unmanaged(container_name("tiny"), labels=managed_labels(dep["id"], "vllm"), status="created")
    assert names(stack)[container_name("tiny")] == "created"
    await stack.deployer.stop(dep["id"], 1)
    assert container_name("tiny") not in names(stack)
    stack.sdk.add_unmanaged(container_name("tiny"), labels=managed_labels(dep["id"], "vllm"), status="created")
    await stack.deployer.remove(dep["id"])
    assert container_name("tiny") not in names(stack)


async def test_cleanup_never_touches_unmanaged_containers_with_the_same_name(stack):
    stack.sdk.fail_start.add(container_name("tiny"))
    with pytest.raises(DeployError):
        await stack.deployer.deploy(req(), 1)
    dep = next(v for v in stack.deployer.list_views() if v["name"] == "tiny")
    stack.sdk.add_unmanaged(container_name("tiny"), labels={}, status="created")  # not ours
    await stack.deployer.remove(dep["id"])
    assert container_name("tiny") in names(stack)


async def test_leftover_of_another_deployment_with_the_same_name_is_kept(stack):
    stack.sdk.fail_start.add(container_name("tiny"))
    with pytest.raises(DeployError):
        await stack.deployer.deploy(req(), 1)
    dep = next(v for v in stack.deployer.list_views() if v["name"] == "tiny")
    stack.sdk.add_unmanaged(container_name("tiny"), labels=managed_labels(dep["id"] + 100, "vllm"), status="created")
    await stack.deployer.remove(dep["id"])
    assert container_name("tiny") in names(stack)  # managed, but labelled for another deployment id
