---
description: "Remove DGXMM, with or without its state, engines and decision-model files."
---

# Uninstall

```bash
./install.sh uninstall           # remove the dgxmm container and dgxmm:latest image
./install.sh uninstall --purge   # also remove state and managed engines
```

| | `uninstall` | `uninstall --purge` |
|---|---|---|
| `dgxmm` container and image | removed | removed |
| Decision model container (this install's, label `dgxmm.install`) | stopped and removed; the database still has it enabled, so a reinstall brings it back | removed |
| Decision model image (labels `dgxmm.built=true` and this install's `dgxmm.install`) | kept | removed (other images, e.g. from before install labels, are reported and kept) |
| Decision model snapshots (`<HF cache>/hub/models--vllm-sr--d3-mini`, `models--vllm-sr--d3-nano`, and the shared-store blobs only they use) | kept | kept; removed with `--purge-decider-model` |
| Engine containers (`dgxmm.managed=true`) | left running (a reinstall picks them up again) | removed if labelled for this install (`dgxmm.install`); others (from before install labels, or another install) are reported and kept |
| Docker network | kept | removed if the installer created it |
| Decision model network (`decider_network`, default `dgxmm-decider`) | kept | removed if this install created it (labels `dgxmm.purpose=decider` and this install's `dgxmm.install`); others are reported and kept |
| Bundled LiteLLM containers (`dgxmm-litellm`, `dgxmm-litellm-db`) | removed | removed |
| `data/` (incl. the LiteLLM database `data/litellm-db`), `secrets/`, `config.yaml`, `.env` | kept | deleted |
| LiteLLM / Postgres images | kept | kept (`docker image rm` them by hand) |
| HF cache, models directory | kept | **kept** |

Uninstall refuses to touch a `dgxmm` container that belongs to another checkout. Add `--dry-run` to
see what would happen.

## Next steps

- [Backup and restore](../operations/backup.md)
- [Install](install.md)
