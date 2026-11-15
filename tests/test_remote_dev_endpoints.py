"""Final review I7: remote engines publish on the cluster IP, reachable from every container of this host, so a
remote single-node vLLM deployment gets no sleep mode (VLLM_SERVER_DEV_MODE exposes /sleep, /collective_rpc,
/update_weights... without the engine key) unless remote_dev_endpoints allows it."""
from dgxmm.deployer.service import DeployRequest


def _remote_container(d):
    (c,) = d.fake("dgx-2").containers.list()
    return c


async def test_remote_single_vllm_deploy_has_no_dev_endpoints(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    req = qwen_req.model_copy(update={"node": "dgx-2", "config": {**qwen_req.config, "enable_sleep_mode": True}})
    v = await d.deploy(req, user_id=1)
    c = _remote_container(d)
    assert "--enable-sleep-mode" not in c.command and "VLLM_SERVER_DEV_MODE" not in c.env
    assert d.row(v["id"]).config["enable_sleep_mode"] is False  # what runs is what is stored
    assert d.supports_sleep(d.row(v["id"])) is False


async def test_local_single_vllm_deploy_keeps_sleep_mode(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    req = qwen_req.model_copy(update={"config": {**qwen_req.config, "enable_sleep_mode": True}})
    v = await d.deploy(req, user_id=1)
    (c,) = d.fake("local").containers.list()
    assert "--enable-sleep-mode" in c.command and c.env.get("VLLM_SERVER_DEV_MODE") == "1"
    assert d.supports_sleep(d.row(v["id"])) is True


async def test_remote_dev_endpoints_setting_allows_sleep_mode(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    d.settings.remote_dev_endpoints = True
    req = qwen_req.model_copy(update={"node": "dgx-2", "config": {**qwen_req.config, "enable_sleep_mode": True}})
    v = await d.deploy(req, user_id=1)
    c = _remote_container(d)
    assert "--enable-sleep-mode" in c.command and c.env.get("VLLM_SERVER_DEV_MODE") == "1"
    assert d.supports_sleep(d.row(v["id"])) is True


async def test_a_remote_row_stored_with_sleep_mode_still_starts_without_it(deployer_two_nodes, qwen_req):
    """A row from before this rule (sleep mode in its config): no dev endpoints when it (re)starts, no sleep."""
    d = deployer_two_nodes
    v = await d.deploy(qwen_req.model_copy(update={"node": "dgx-2"}), user_id=1)
    from dgxmm.db import Deployment
    with d.db.session() as s:
        row = s.get(Deployment, v["id"])
        row.config = {**row.config, "enable_sleep_mode": True}
    assert d.supports_sleep(d.row(v["id"])) is False
    engine = d.engines.get("vllm")
    cfg = engine.validate_config(d.row(v["id"]).config)
    from dgxmm.engines.base import LaunchContext
    import inspect
    params = inspect.signature(LaunchContext).parameters
    kw = {k: v for k, v in dict(name="qwen", deployment_id=v["id"], port=v["port"], reserved=2**30,
                                 mem_total=2**37, node="dgx-2").items() if k in params}
    spec = engine.container_spec(d.inventory.get("Qwen/Qwen3-Tiny"), cfg, None, LaunchContext(**kw), d.settings)
    assert "--enable-sleep-mode" not in spec.command and "VLLM_SERVER_DEV_MODE" not in spec.environment
