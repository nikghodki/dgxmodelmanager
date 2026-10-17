"""Serialized engine start-ups (deployer startup gate).

vLLM profiles free memory while it starts; another engine allocating or freeing memory at the same time makes the
profile wrong (live incident: a deployment started while the decider was loading failed with "No available memory
for the cache blocks" / "Error in memory profiling"). At most ``max_concurrent_startups`` engines are starting or
loading at once; the others wait in state "queued" with a visible reason, keep their memory reservation, and are
fit-checked again when their turn comes.
"""
import asyncio

import pytest
from sqlalchemy import select

from dgxmm.config import GIB
from dgxmm.db import Deployment, DeploymentEvent
from dgxmm.deployer.reconcile import reconcile
from dgxmm.deployer.service import DeployError, DeployRequest

SMALL = {"max_model_len": 4096, "max_num_seqs": 2, "max_num_batched_tokens": 2048}


def req(name, ref="Qwen/Qwen3-Tiny", **cfg):
    return DeployRequest(name=name, engine="vllm", model_ref=ref, config={**SMALL, **cfg}, alias=name)


async def until(pred, timeout=5.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not pred():
        if loop.time() > end:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


def state(stack, dep_id):
    return stack.deployer.view(dep_id)["state"]


def events(stack, dep_id):
    with stack.db.session() as s:
        return [(e.to_state, e.message) for e in s.scalars(select(DeploymentEvent)
                                                           .where(DeploymentEvent.deployment_id == dep_id)
                                                           .order_by(DeploymentEvent.id))]


async def test_default_allows_one_startup_and_queues_the_next(stack):
    assert stack.settings.max_concurrent_startups == 1
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    assert a["state"] in ("starting", "loading")
    assert b["state"] == "queued"
    assert len(stack.sdk.run_calls) == 1  # b's container is not created yet
    q = stack.deployer.view(b["id"])["startup_queue"]
    assert q["position"] == 1 and q["waiting_for"] == ["a"]
    assert "a" in q["reason"] and "starting" in q["reason"]
    assert any(to == "queued" and "a" in msg for to, msg in events(stack, b["id"]))
    assert stack.deployer.view(a["id"])["startup_queue"] is None
    # a becomes ready -> b launches
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, b["id"]) in ("starting", "loading"))
    assert len(stack.sdk.run_calls) == 2
    stack.http.healthy.add(b["port"])
    assert await stack.deployer.wait_ready(b["id"], 5) == "ready"
    assert stack.deployer.view(b["id"])["startup_queue"] is None


async def test_queued_deployment_keeps_its_memory_reservation(stack):
    await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    budget = await stack.budget.refresh()
    usage = next(u for u in budget.deployments if u.deployment_id == b["id"])
    assert usage.state == "queued" and budget.committed >= usage.reserved > 0


async def test_failed_startup_releases_the_slot(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    c = stack.sdk.container(a["container_id"])
    c.status, c.exit_code = "exited", 1
    await until(lambda: state(stack, b["id"]) in ("starting", "loading"))
    assert state(stack, a["id"]) == "failed"


async def test_fit_is_checked_again_at_launch_time(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b", "org/big-model"), 1)
    assert b["state"] == "queued"
    stack.probe.available = 9 * GIB  # something outside DGXMM took memory meanwhile (still above the margin)
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, b["id"]) == "failed")
    v = stack.deployer.view(b["id"])
    assert "memory" in v["last_error"] and "launch" in v["last_error"]
    assert len(stack.sdk.run_calls) == 1  # never started


async def test_queue_timeout_fails_the_waiting_deployment(stack):
    stack.settings.startup_queue_timeout_s = 0.1
    await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    await until(lambda: state(stack, b["id"]) == "failed")
    assert "timed out" in stack.deployer.view(b["id"])["last_error"]
    assert len(stack.sdk.run_calls) == 1


async def test_stopping_a_queued_deployment_cancels_it(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    c = await stack.deployer.deploy(req("c"), 1)
    assert stack.deployer.view(c["id"])["startup_queue"]["position"] == 2
    stopped = await stack.deployer.stop(b["id"], 1)
    assert stopped["state"] == "stopped"
    assert stack.deployer.view(c["id"])["startup_queue"]["position"] == 1
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, c["id"]) in ("starting", "loading"))
    assert state(stack, b["id"]) == "stopped"
    assert len(stack.sdk.run_calls) == 2  # a and c, never b
    await stack.deployer.remove(b["id"])  # a cancelled queued deployment can be removed


async def test_restart_of_stopped_deployment_queues_behind_a_startup(stack):
    b = await stack.deployer.deploy(req("b"), 1)
    stack.http.healthy.add(b["port"])
    assert await stack.deployer.wait_ready(b["id"], 5) == "ready"
    await stack.deployer.stop(b["id"], 1)
    a = await stack.deployer.deploy(req("a"), 1)
    again = await stack.deployer.restart(b["id"], 1)
    assert again["state"] == "queued"
    stack.http.all_healthy = True  # b got a new port (a took its old one while b was stopped)
    await until(lambda: state(stack, b["id"]) == "ready")


async def test_wake_waits_for_a_running_startup(stack):
    s = await stack.deployer.deploy(req("s", enable_sleep_mode=True), 1)
    stack.http.healthy.add(s["port"])
    assert await stack.deployer.wait_ready(s["id"], 5) == "ready"
    await stack.deployer.sleep(s["id"], 1)
    a = await stack.deployer.deploy(req("a"), 1)
    wake = asyncio.ensure_future(stack.deployer.wake(s["id"], 1))
    await asyncio.sleep(0.1)
    assert not wake.done() and ("POST", s["port"], "/wake_up") not in stack.http.calls
    assert stack.deployer.view(s["id"])["startup_queue"]["waiting_for"] == ["a"]
    stack.http.healthy.add(a["port"])
    assert (await asyncio.wait_for(wake, 5))["state"] == "ready"


async def test_wake_gives_up_after_its_queue_timeout(stack):
    stack.settings.wake_queue_timeout_s = 0.1
    s = await stack.deployer.deploy(req("s", enable_sleep_mode=True), 1)
    stack.http.healthy.add(s["port"])
    await stack.deployer.wait_ready(s["id"], 5)
    await stack.deployer.sleep(s["id"], 1)
    await stack.deployer.deploy(req("a"), 1)
    with pytest.raises(DeployError) as e:
        await stack.deployer.wake(s["id"], 1)
    assert e.value.code == "startup_busy" and e.value.status == 409 and "a" in e.value.message
    assert state(stack, s["id"]) == "sleeping"


async def test_limit_is_configurable(stack):
    stack.settings.max_concurrent_startups = 2
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    c = await stack.deployer.deploy(req("c"), 1)
    assert [x["state"] in ("starting", "loading") for x in (a, b)] == [True, True]
    assert c["state"] == "queued"


async def test_zero_means_no_limit(stack):
    stack.settings.max_concurrent_startups = 0
    deps = [await stack.deployer.deploy(req(n), 1) for n in "abc"]
    assert all(d["state"] in ("starting", "loading") for d in deps)


async def test_concurrent_deploys_never_start_two_engines_at_once(stack):
    names = [f"d{i}" for i in range(5)]
    starting_now: set[int] = set()
    peak = 0
    order: list[int] = []

    async def hook(dep_id, frm, to):
        nonlocal peak
        if to == "starting":
            order.append(dep_id)
        if to in ("starting", "loading"):
            starting_now.add(dep_id)
        else:
            starting_now.discard(dep_id)
        peak = max(peak, len(starting_now))

    stack.deployer.state_hooks.append(hook)
    stack.http.all_healthy = True
    views = await asyncio.gather(*(stack.deployer.deploy(req(n), 1) for n in names))
    for v in views:
        assert await stack.deployer.wait_ready(v["id"], 10) == "ready"
    assert peak == 1
    assert len(order) == 5 and len(stack.sdk.run_calls) == 5
    # FIFO: deployments start in the order they were queued (row id order)
    assert order == sorted(order)


async def test_switch_waits_through_the_queue(stack):
    old = await stack.deployer.deploy(req("old"), 1)
    stack.http.healthy.add(old["port"])
    await stack.deployer.wait_ready(old["id"], 5)
    other = await stack.deployer.deploy(req("other"), 1)  # holds the start-up slot
    stack.settings.startup_timeout_s = 1  # the switch's wait_ready budget (startup_timeout_s + 30)

    async def release_later():
        await asyncio.sleep(0.3)
        stack.http.all_healthy = True

    asyncio.ensure_future(release_later())
    new = await stack.deployer.switch(old["id"], req("new"), [], 1)
    assert new["state"] == "ready" and state(stack, other["id"]) == "ready"


async def test_wait_ready_does_not_count_queue_time(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    waiter = asyncio.ensure_future(stack.deployer.wait_ready(b["id"], 0.2))
    await asyncio.sleep(0.4)
    assert not waiter.done()  # still queued: the timeout has not started
    stack.http.all_healthy = True
    assert await asyncio.wait_for(waiter, 5) == "ready"
    assert state(stack, a["id"]) == "ready"


async def test_reconcile_resumes_queued_and_counts_starting(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    # simulate a dgxmm restart: a new Deployer over the same DB and Docker
    from dgxmm.deployer.service import Deployer
    from dgxmm.memory.budget import BudgetService
    await stack.deployer.aclose()
    d2 = Deployer(settings=stack.settings, db=stack.db, docker=stack.deployer.docker, inventory=stack.inv,
                  engines=stack.deployer.engines, http=stack.deployer.http, events=stack.events)
    d2.budget = BudgetService(stack.settings, stack.probe, d2.usage_rows)
    await reconcile(d2)
    await asyncio.sleep(0.1)
    assert d2.view(b["id"])["state"] == "queued"  # a (still loading) holds the slot after the restart
    assert d2.view(b["id"])["startup_queue"]["waiting_for"] == ["a"]
    stack.http.healthy.add(a["port"])
    await until(lambda: d2.view(b["id"])["state"] in ("starting", "loading"))
    await d2.aclose()


async def test_deleted_model_while_queued_fails_cleanly(stack):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    stack.deployer.delete_guard = lambda ref: ref == "Qwen/Qwen3-Tiny"
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, b["id"]) == "failed")
    assert "being deleted" in stack.deployer.view(b["id"])["last_error"]
    with stack.db.session() as s:
        assert s.get(Deployment, b["id"]) is not None  # the row stays (failed) so the user sees why


def test_queued_deployment_is_listed_with_reason_in_api(api_stack):
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    for n in ("a", "b"):
        r = api_stack.post("/dgxmm/api/deployments", headers=h,
                           json={"name": n, "engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": SMALL,
                                 "alias": n})
        assert r.status_code == 201, r.text
    b = next(x for x in api_stack.get("/dgxmm/api/deployments").json() if x["name"] == "b")
    assert b["state"] == "queued" and b["startup_queue"]["waiting_for"] == ["a"]
    detail = api_stack.get(f"/dgxmm/api/deployments/{b['id']}").json()
    assert detail["startup_queue"]["position"] == 1
    assert api_stack.post(f"/dgxmm/api/deployments/{b['id']}/stop", headers=h).json()["state"] == "stopped"


async def test_queue_changes_are_published_for_waiting_deployments(stack):
    q = stack.events.subscribe()
    await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    c = await stack.deployer.deploy(req("c"), 1)
    while not q.empty():
        q.get_nowait()
    await stack.deployer.stop(b["id"], 1)
    seen = []
    while not q.empty():
        topic, data = q.get_nowait()
        if topic == "deployment":
            seen.append((data["id"], (data.get("startup_queue") or {}).get("position")))
    assert (c["id"], 1) in seen  # c moved up: its new position reaches the UI without a reload


async def test_stop_is_not_blocked_by_a_wake_waiting_for_the_gate(stack):
    s = await stack.deployer.deploy(req("s", enable_sleep_mode=True), 1)
    stack.http.healthy.add(s["port"])
    await stack.deployer.wait_ready(s["id"], 5)
    await stack.deployer.sleep(s["id"], 1)
    await stack.deployer.deploy(req("a"), 1)  # holds the start-up slot
    wake = asyncio.ensure_future(stack.deployer.wake(s["id"], 1))
    await asyncio.sleep(0.05)
    stopped = await asyncio.wait_for(stack.deployer.stop(s["id"], 1), 2)  # does not wait for the wake's turn
    assert stopped["state"] == "stopped"
    stack.http.all_healthy = True
    with pytest.raises(DeployError) as e:
        await asyncio.wait_for(wake, 5)
    assert e.value.code == "not_sleeping"


# ---------- fix round 1 ----------
async def test_any_error_after_the_slot_is_granted_fails_the_queued_row(stack, monkeypatch):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)

    def no_ports(exclude_dep=None):
        raise DeployError("no_ports", "port pool exhausted", 503)
    monkeypatch.setattr(stack.deployer, "_allocate_port", no_ports)
    monkeypatch.setattr(stack.deployer, "_free_port", lambda dep_id, cur: _raise(DeployError("no_ports", "port pool exhausted", 503)))
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, b["id"]) == "failed")
    assert "port pool exhausted" in stack.deployer.view(b["id"])["last_error"]
    assert not stack.deployer.gate.holds(b["id"])
    budget = await stack.budget.refresh()
    assert all(u.deployment_id != b["id"] or u.state == "failed" for u in budget.deployments)


async def _raise(e):
    raise e


async def test_unexpected_error_after_the_slot_is_granted_fails_the_queued_row(stack, monkeypatch):
    a = await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    real = stack.budget.refresh
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if stack.deployer.gate.holds(b["id"]):
            raise OSError("cannot read /proc/meminfo")
        return await real()
    monkeypatch.setattr(stack.budget, "refresh", flaky)
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, b["id"]) == "failed")
    assert "cannot read /proc/meminfo" in stack.deployer.view(b["id"])["last_error"]
    assert not stack.deployer.gate.holds(b["id"])


async def test_removing_a_deployment_whose_wake_waits_is_a_clean_error(stack):
    stack.settings.wake_sync_wait_s = 5
    s = await stack.deployer.deploy(req("s", enable_sleep_mode=True), 1)
    stack.http.healthy.add(s["port"])
    await stack.deployer.wait_ready(s["id"], 5)
    await stack.deployer.sleep(s["id"], 1)
    await stack.deployer.deploy(req("a"), 1)
    wake = asyncio.ensure_future(stack.deployer.wake(s["id"], 1))
    await asyncio.sleep(0.05)
    await stack.deployer.stop(s["id"], 1)
    await stack.deployer.remove(s["id"])
    with pytest.raises(DeployError) as e:  # not a bare CancelledError
        await asyncio.wait_for(wake, 3)
    assert e.value.code == "not_found" and e.value.status == 404


async def test_a_long_wake_wait_returns_queued_and_finishes_in_the_background(stack):
    stack.settings.wake_sync_wait_s = 0.1
    s = await stack.deployer.deploy(req("s", enable_sleep_mode=True), 1)
    stack.http.healthy.add(s["port"])
    await stack.deployer.wait_ready(s["id"], 5)
    await stack.deployer.sleep(s["id"], 1)
    a = await stack.deployer.deploy(req("a"), 1)
    t0 = asyncio.get_running_loop().time()
    v = await stack.deployer.wake(s["id"], 1)
    assert asyncio.get_running_loop().time() - t0 < 1
    assert v["state"] == "sleeping" and v["startup_queue"]["kind"] == "wake"
    again = await stack.deployer.wake(s["id"], 1)  # a second click joins the same wake-up
    assert again["startup_queue"]["position"] == 1
    stack.http.healthy.add(a["port"])
    await until(lambda: state(stack, s["id"]) == "ready")
    assert stack.http.calls.count(("POST", s["port"], "/wake_up")) == 1


def test_wake_api_answers_202_while_queued(api_stack):
    from tests.test_api_deployments import admin
    h = admin(api_stack)
    api_stack.stack.settings.wake_sync_wait_s = 0.1
    api_stack.stack.http.all_healthy = True
    body = {"engine": "vllm", "model_ref": "Qwen/Qwen3-Tiny", "config": {**SMALL, "enable_sleep_mode": True}}
    s = api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "s", "alias": "s", **body}).json()
    import time
    for _ in range(200):
        if api_stack.get(f"/dgxmm/api/deployments/{s['id']}").json()["state"] == "ready":
            break
        time.sleep(0.02)
    assert api_stack.post(f"/dgxmm/api/deployments/{s['id']}/sleep", headers=h).status_code == 200
    api_stack.stack.http.all_healthy = False
    api_stack.post("/dgxmm/api/deployments", headers=h, json={"name": "a", "alias": "a", **body})
    r = api_stack.post(f"/dgxmm/api/deployments/{s['id']}/wake", headers=h)
    assert r.status_code == 202 and r.json()["startup_queue"]["kind"] == "wake"


async def test_raising_the_limit_lets_waiters_start(stack):
    await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    c = await stack.deployer.deploy(req("c"), 1)
    stack.settings.max_concurrent_startups = 3
    stack.deployer.gate.limit_changed()
    await until(lambda: state(stack, b["id"]) in ("starting", "loading")
                and state(stack, c["id"]) in ("starting", "loading"), timeout=3)


async def test_raised_limit_is_picked_up_even_without_a_notification(stack):
    await stack.deployer.deploy(req("a"), 1)
    b = await stack.deployer.deploy(req("b"), 1)
    stack.settings.max_concurrent_startups = 2  # changed behind the gate's back
    await until(lambda: state(stack, b["id"]) in ("starting", "loading"), timeout=3)
