"""Final review I9: the view says whether the engine needs its key (its direct URLs answer 401 without it), never
the key itself."""


async def test_view_flags_a_keyed_engine_without_revealing_the_key(deployer_two_nodes, qwen_req):
    d = deployer_two_nodes
    v = await d.deploy(qwen_req, user_id=1)
    key = d.row(v["id"]).engine_api_key
    assert key and v["engine_key"] is True
    assert key not in repr(d.view(v["id"]))
