from sqlalchemy import select

from dgxmm.db import Deployment
from dgxmm.deployer.reconcile import reconcile
from dgxmm.deployer.service import DeployRequest
from dgxmm.engines.base import managed_labels

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def deploy(stack, name):
    return await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny", config=SMALL,
                                                     alias=name), 1)  # distinct LiteLLM model names (same model twice)


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
    stack.sdk.add_unmanaged("kokoro-tts")
    await reconcile(stack.deployer)
    with stack.db.session() as s:
        rows = s.scalars(select(Deployment)).all()
    assert [(r.name, r.state, r.port, r.model_ref) for r in rows] == [("ghost", "unknown", 8150, "Qwen/Qwen3-Tiny")]
