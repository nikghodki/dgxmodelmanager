---
description: "Sign in to DGXMM for the first time, change the admin password, deploy a model and call it through LiteLLM."
---

# First login

## Sign in

Open one of the URLs the installer printed (`http://<host>:8090/` redirects to `/dgxmm/`), sign in as
`admin` with the printed password (also saved in `secrets/admin_password`) and choose a new one of at
least 12 characters. Every other action is refused until you have changed it.

!!! tip "Delete the saved first password"
    `secrets/admin_password` is not needed after the first password change. Delete it (see
    [security hardening](../security/hardening.md)).

Forgot the password later? See
[troubleshooting](../operations/troubleshooting.md#forgot-the-admin-password).

## Deploy your first model

Deploy a model from **Deploy model**, or find one in **Catalog** first and download it. The wizard shows
a memory check before anything starts (see [deploying a model](../user-guide/deploying.md)). After
**Deploy** the deployment moves through *Starting* → *Loading* → *Ready*; the first vLLM start of a model
can take several minutes.

## Call your first model

Every ready deployment is one model on the LiteLLM endpoint, selected by its model name (shown on the
deployment's **API access** card). Create a virtual key on the **LiteLLM** page (admins) or on the
deployment's LiteLLM tab, then:

```bash
export LITELLM_API_KEY=sk-...                   # the virtual key, shown once when it is created
curl http://<host>:4000/v1/models -H "Authorization: Bearer $LITELLM_API_KEY"
curl http://<host>:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model": "default-chat", "messages": [{"role": "user", "content": "Hello!"}]}'
```

`http://<host>:4000` is the bundled LiteLLM; with your own LiteLLM, use its public URL. Any OpenAI SDK
works the same way (`base_url="http://<host>:4000/v1"`). Python snippets and direct engine endpoints are
in [calling your models](../user-guide/calling-models.md).

## Next steps

- [Dashboard](../user-guide/dashboard.md)
- [Deploying a model](../user-guide/deploying.md)
- [Users and roles](../user-guide/users.md)
