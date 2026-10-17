"""Cross-lane integration (reliability x security): every start of the decider made by the supervisor (crash or
hang restart, model rollback) goes through the deployer's normal run path, so the bearer key file is delivered
again to each new container; and the supervisor never fights a key rotation."""
import pytest

from dgxmm.decider.rotation import rotate_decider_key
from dgxmm.decider.secret import read_key
from dgxmm.engines.decision import KEY_FILE
from tests.test_decider_engine import d3  # noqa: F401  (fixture)
from tests.test_decider_supervisor import crash, running_decider, sup  # noqa: F401  (fixture)


def deliveries(t, cid):
    return [w for w in t.stack.sdk.secret_writes if w[0] == cid and w[1] == KEY_FILE]


async def test_crash_restart_delivers_the_key_file_to_the_new_container(sup):  # noqa: F811
    view = await running_decider(sup)
    first = sup.stack.deployer.view(view["id"])["container_id"]
    assert len(deliveries(sup, first)) == 1
    await crash(sup, view)
    assert await sup.sup.tick() == "restarted"
    new = sup.stack.deployer.view(view["id"])["container_id"]
    assert new and new != first
    (w,) = deliveries(sup, new)
    assert w[2] == read_key(sup.stack.settings).encode()
    run = sup.stack.sdk.run_calls[-1]
    assert read_key(sup.stack.settings) not in repr(run)  # never in env / command / labels


async def test_hang_restart_delivers_the_key_file_again(sup):  # noqa: F811
    view = await running_decider(sup)
    first = sup.stack.deployer.view(view["id"])["container_id"]
    sup.stack.http.healthy.discard(view["port"])
    sup.decider.client = lambda: None  # every probe fails: the decider counts as hung
    for _ in range(sup.stack.settings.decider_hang_probes - 1):
        assert await sup.sup.tick() == "probe_failed"
    assert await sup.sup.tick() == "restarted"
    new = sup.stack.deployer.view(view["id"])["container_id"]
    assert new and new != first and len(deliveries(sup, new)) == 1


async def test_supervisor_leaves_a_rotating_decider_alone(sup):  # noqa: F811
    view = await running_decider(sup)
    await crash(sup, view)
    sup.stack.deployer.rotating.add(view["id"])  # a rotation stopped it and is about to start it again
    try:
        assert await sup.sup.tick() == "rotating"
        assert sup.stack.deployer.view(view["id"])["state"] == "failed"  # untouched
        assert sup.sup.status()["restarts_last_hour"] == 0 and sup.sup.next_attempt_at is None  # no backoff spent
    finally:
        sup.stack.deployer.rotating.discard(view["id"])
    assert await sup.sup.tick() == "restarted"


async def test_rotation_restart_gets_the_new_key_and_the_supervisor_keeps_probing(sup):  # noqa: F811
    from types import SimpleNamespace
    view = await running_decider(sup)
    old_key = read_key(sup.stack.settings)
    ctx = SimpleNamespace(deployer=sup.stack.deployer, decider=sup.decider, settings=sup.stack.settings,
                          db=sup.stack.db)
    out = await rotate_decider_key(ctx, 1)
    assert out["rotated"] and out["restarted"] and out["error"] is None, out
    new_key = read_key(sup.stack.settings)
    assert new_key != old_key
    cid = sup.stack.deployer.view(view["id"])["container_id"]
    (w,) = deliveries(sup, cid)
    assert w[2] == new_key.encode()
    sup.stack.http.healthy.add(sup.stack.deployer.view(view["id"])["port"])
    assert await sup.stack.deployer.wait_ready(view["id"], 5) == "ready"
    assert await sup.sup.tick() == "probe_ok"  # the rotation's restart is not taken for a crash or an admin stop
    assert sup.sup.status()["restarts_last_hour"] == 0


async def test_a_rotation_that_begins_during_the_supervisors_stop_ends_with_the_decider_running_the_new_key(sup):  # noqa: F811,E501
    """The supervisor stopped a hung decider; a rotation started before its restart. The rotation found the decider
    stopped (it restarts only running ones), so the supervisor keeps its pending restart and starts it as soon as the
    rotation is over, with the new key."""
    from dgxmm.decider.secret import rotate_key
    from tests.test_decider_supervisor import hang
    view = await running_decider(sup)
    old_key = read_key(sup.stack.settings)
    await hang(sup, view)
    real_stop = sup.stack.deployer.stop

    async def stop_then_rotation_starts(*a, **kw):
        out = await real_stop(*a, **kw)
        sup.stack.deployer.rotating.add(view["id"])  # the rotation marks it ...
        rotate_key(sup.stack.settings)                # ... and writes the new key
        return out
    sup.stack.deployer.stop = stop_then_rotation_starts
    assert await sup.sup.tick() == "rotating"
    sup.stack.deployer.stop = real_stop
    assert sup.sup.status()["pending_restart"] and sup.sup.next_attempt_at is None  # kept, no backoff spent
    assert sup.stack.deployer.view(view["id"])["state"] == "stopped"
    assert await sup.sup.tick() == "rotating"  # still rotating: still waiting
    sup.stack.deployer.rotating.discard(view["id"])  # the rotation is over (it did not restart a stopped decider)
    assert await sup.sup.tick() == "restarted"
    v = sup.stack.deployer.view(view["id"])
    new_key = read_key(sup.stack.settings)
    assert new_key != old_key
    (w,) = deliveries(sup, v["container_id"])
    assert w[2] == new_key.encode()
    sup.stack.http.healthy.add(v["port"])
    assert await sup.stack.deployer.wait_ready(view["id"], 5) == "ready"
    assert sup.sup.status()["pending_restart"] is None


async def test_rotation_of_a_queued_decider_launches_it_with_the_new_key(sup):  # noqa: F811
    from types import SimpleNamespace
    from dgxmm.deployer.service import DeployRequest
    from tests.test_api_deployments import SMALL
    from tests.test_decider_engine import deploy_decider
    other = await sup.stack.deployer.deploy(DeployRequest(name="busy", engine="vllm", model_ref="Qwen/Qwen3-Tiny",
                                                          config=SMALL, alias="busy"), 1)
    assert other["state"] in ("starting", "loading")  # holds the only start-up slot
    view = await deploy_decider(sup.stack)
    assert view["state"] == "queued"
    sup.decider.set_deployment_id(view["id"])
    sup.decider.set_enabled(True)
    from dgxmm.decider.secret import ensure_key
    old_key = ensure_key(sup.stack.settings)  # the key a launch would read before the rotation
    ctx = SimpleNamespace(deployer=sup.stack.deployer, decider=sup.decider, settings=sup.stack.settings,
                          db=sup.stack.db)
    out = await rotate_decider_key(ctx, 1)
    assert out["rotated"] and out["restarted"] and out["error"] is None, out
    assert sup.stack.deployer.view(view["id"])["state"] == "queued"  # queued again behind "busy"
    sup.stack.http.healthy.add(other["port"])  # the slot frees: the decider launches now
    for _ in range(500):
        v = sup.stack.deployer.view(view["id"])
        if v["container_id"]:
            break
        await __import__("asyncio").sleep(0.01)
    (w,) = deliveries(sup, v["container_id"])
    assert w[2] == read_key(sup.stack.settings).encode() != old_key.encode()


async def test_model_rollback_delivers_the_key_file_to_the_rolled_back_container(sup, fake_hf_repo):  # noqa: F811
    from tests.test_decider_models import NANO, add_models
    add_models(fake_hf_repo, sup.stack.inv, which=(NANO,))
    view = await running_decider(sup)
    mini = view["model_ref"]
    await sup.stack.deployer.stop(view["id"], 1)
    sup.decider.set_model(NANO)
    await sup.stack.deployer.restart(view["id"], 1, model_ref=NANO)
    nano_cid = sup.stack.deployer.view(view["id"])["container_id"]
    assert len(deliveries(sup, nano_cid)) == 1
    sup.decider.set_switch_from(mini)
    await crash(sup, view)
    assert await sup.sup.tick() == "rolled_back"
    v = sup.stack.deployer.view(view["id"])
    assert v["model_ref"] == mini and v["container_id"] != nano_cid
    (w,) = deliveries(sup, v["container_id"])
    assert w[2] == read_key(sup.stack.settings).encode()
    assert sup.stack.sdk.run_calls[-1]["environment"]["D3_REPO"] == mini
