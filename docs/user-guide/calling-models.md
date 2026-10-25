---
description: "Call deployments through LiteLLM or directly with curl and the OpenAI SDK."
---

# Calling your models

**Through LiteLLM** (recommended): all deployments share one OpenAI-compatible endpoint
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`), and the `model` field picks the deployment.
The **API access** card on a deployment's Overview tab shows the endpoint and model name and has
ready-made snippets. Create a key on the deployment's LiteLLM tab or the LiteLLM page:

```bash
curl https://llm.example.com/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "default-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
import os
from openai import OpenAI

client = OpenAI(base_url="https://llm.example.com/v1", api_key=os.environ["LITELLM_API_KEY"])
reply = client.chat.completions.create(model="default-chat",
                                       messages=[{"role": "user", "content": "Hello!"}])
print(reply.choices[0].message.content)
```

Each deployment is registered once, under its model name: the **Model name (LiteLLM)** you set, or
the served name. A stable model name makes model changes invisible to clients: deploy with model name
`default-chat`, and use **Switch model…** later. The name moves to the new deployment once it is ready.

An [on-demand](on-demand.md) model that is asleep is started by the request: the first answer can take a
few minutes, so give your client a long timeout (for example `OpenAI(..., timeout=1200)`).

**Directly** (advanced): each engine also serves the same API under its served model name, without
keys, budgets or rate limits. The engines have no authentication, which is why their ports are only
published on the host's loopback interface.

- From the DGX itself, use the "On the DGX" endpoint, for example
  `curl http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "qwen3-0.6b", "messages": [{"role": "user", "content": "Hello!"}]}'`.
- From another container on the DGXMM Docker network, use `http://dgxmm-<deployment name>:<port>/v1`.

Both are listed under **Direct engine endpoints (advanced)** on the Overview tab. Direct calls never start an
on-demand model that is asleep; while one is being answered, it keeps the model from being put to sleep.

## Next steps

- [LiteLLM](litellm.md)
- [Networking and reverse proxy](../configuration/networking.md)
