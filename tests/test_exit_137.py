"""Exit 137 is out of memory only with evidence (H6 D3): Docker's State.OOMKilled, or the kernel OOM killer's own
line in the log. A plain SIGKILL (docker kill, a stop timeout) is labelled as such and never feeds calibration."""
from __future__ import annotations

import pytest

from dgxmm.decider import diagnosis as dg
from dgxmm.deployer.service import DeployRequest
from dgxmm.docker_ctl import DockerController
from dgxmm.memory.calibration import Calibration

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


async def ready(stack, name="a"):
    stack.http.all_healthy = True
    d = await stack.deployer.deploy(DeployRequest(name=name, engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=dict(SMALL)), 1)
    assert await stack.deployer.wait_ready(d["id"], 5) == "ready"
    return stack.deployer.view(d["id"])


async def crash(stack, *, oom_killed: bool, log: str = "INFO serving requests"):
    cal = Calibration(stack.db)
    stack.deployer.calibration = cal
    d = await ready(stack)
    c = stack.sdk.container(d["container_id"])
    c.log_text = log
    c.status, c.exit_code, c.oom_killed = "exited", 137, oom_killed
    await stack.deployer.check_liveness()
    return stack.deployer.view(d["id"]), [r for r in cal.summary("vllm", "Qwen/Qwen3-Tiny") if r["oom"]]


async def test_docker_oom_killed_is_out_of_memory_and_recorded(stack):
    v, ooms = await crash(stack, oom_killed=True)
    assert v["state"] == "failed" and v["exit_code"] == 137
    assert "out of memory" in v["last_error"] and ooms


async def test_plain_sigkill_is_not_out_of_memory_and_not_recorded(stack):
    v, ooms = await crash(stack, oom_killed=False, log="INFO serving requests\nKilled")
    assert v["state"] == "failed" and v["exit_code"] == 137
    assert "out of memory" not in v["last_error"].lower() and "SIGKILL" in v["last_error"]
    assert ooms == []


async def test_kernel_oom_line_in_the_log_counts_as_out_of_memory(stack):
    v, ooms = await crash(stack, oom_killed=False,
                          log="Out of memory: Killed process 4242 (python3) total-vm:1234kB, anon-rss:99kB")
    assert "out of memory" in v["last_error"] and ooms


async def test_oom_during_startup_uses_the_same_rule(stack):
    d = await stack.deployer.deploy(DeployRequest(name="s", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                  config=dict(SMALL)), 1)
    c = stack.sdk.container(d["container_id"])
    c.status, c.exit_code = "exited", 137
    assert await stack.deployer.wait_ready(d["id"], 5) == "failed"
    assert "out of memory" not in stack.deployer.view(d["id"])["last_error"].lower()


def test_container_info_reads_oom_killed(stack):
    ctl = DockerController(stack.sdk)
    c = stack.sdk.containers.run("img", name="x", detach=True)
    c.status, c.exit_code, c.oom_killed = "exited", 137, True
    assert ctl.get(c.id).oom_killed is True
    c.oom_killed = False
    assert ctl.get(c.id).oom_killed is False


@pytest.mark.parametrize("text,expected", [
    ("Loading weights...", []),
    ("Killed", []),
    ("Out of memory: Killed process 4242 (python3)", ["gpu_oom"]),
    ("Memory cgroup out of memory: Killed process 77 (vllm)", ["gpu_oom"]),
])
def test_diagnosis_rules_match_the_same_evidence(text, expected):
    assert [c for c, _ in dg.rule_hits(text, 137)] == expected


def test_diagnosis_counts_docker_oom_flag_as_evidence():
    assert [c for c, _ in dg.rule_hits("Loading weights...", 137, oom_killed=True)] == ["gpu_oom"]
    assert dg.rule_hits("Loading weights...", 137, oom_killed=False) == []


async def test_the_oom_flag_is_stored_for_diagnosis(stack):
    """The flag is stored with the deployment, so diagnosis sees what the label was based on."""
    v, _ = await crash(stack, oom_killed=True)
    from dgxmm.db import Deployment
    with stack.db.session() as s:
        assert s.get(Deployment, v["id"]).oom_killed is True
