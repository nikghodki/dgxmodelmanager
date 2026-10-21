"""Test LiteLLM sync handles stale entries and exceptions."""
import pytest
from tests.test_litellm import FakeLiteLLM, make, ready


async def test_sync_replaces_entry_when_served_name_changes(stack):
    """When served_name changes on same port, entry is replaced (not kept with old model)."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    
    # Deploy A with served_name "model-a"
    d = await ready(stack, "a")
    await sync.sync()
    
    # Check entry exists
    mine = [m for m in fake.models if m["model_info"].get("dgxmm_id")]
    assert len(mine) == 1
    assert mine[0]["litellm_params"]["model"] == "openai/qwen3-tiny"
    
    # Reconfigure A to change served_name
    await stack.deployer.reconfigure(d["id"], {**d["config"], "served_model_name": "new-name"})
    await stack.deployer.wait_ready(d["id"], 5)
    
    # Sync should replace the entry (delete old, create new)
    await sync.sync()
    mine = [m for m in fake.models if m["model_info"].get("dgxmm_id")]
    assert len(mine) == 1
    assert mine[0]["litellm_params"]["model"] == "openai/new-name"


async def test_sync_run_survives_unexpected_exceptions(stack):
    """run() loop continues after non-httpx exceptions from sync."""
    fake = FakeLiteLLM()
    sync = make(stack, fake)
    
    # Make sync() raise unexpected exception once
    original_list = sync.client.list_models  # the sync reads the whole model list (own + foreign entries)
    call_count = [0]
    
    async def fail_once():
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("unexpected error")
        return await original_list()
    
    sync.client.list_models = fail_once
    
    # First sync should fail but not crash
    result = await sync.sync()
    assert result is False  # Failed
    
    # Second sync should succeed
    result = await sync.sync()
    assert result is True  # Recovered
