---
description: "Running DGXMM day to day: changing settings, freeing disk space, and where to find backups, logs, the admin CLI and troubleshooting."
---

# Operations

How to install, upgrade, back up, monitor and troubleshoot DGXMM. All commands run in the DGXMM
checkout on the host, as the user that installed it.

| Task | Page |
|---|---|
| Install, upgrade, uninstall | [Install](getting-started/install.md), [upgrade](getting-started/upgrade.md), [uninstall](getting-started/uninstall.md) |
| Back up and restore the database, secrets and configuration | [Backup and restore](operations/backup.md) |
| Read logs, check health | [Logs and monitoring](operations/monitoring.md) |
| Reset a password, manage the decision model from the shell | [Admin command line](operations/admin-cli.md) |
| Run, upgrade and debug the decision model | [Running the decision model](operations/decision-model.md) |
| Fix a problem (including a forgotten admin password) | [Troubleshooting](operations/troubleshooting.md) |
| Know what DGXMM does not do | [Known limitations](operations/known-limitations.md) |
| Deploy it safely | [Security hardening](security/hardening.md) |

## Changing settings

- **`.env` values** (port, bind address, network, model paths): edit `.env`, then `./install.sh`. If you
  change paths or the network, update `config.yaml` to match (see
  [configuration](configuration/index.md#env-and-installer-variables)).
- **`config.yaml`**: edit it, then `docker restart dgxmm`. The file is mounted into the container and
  read at start; `./install.sh` alone does not restart a container whose image and `.env` are
  unchanged.
- **Secrets**: edit the file in `secrets/`, then `docker restart dgxmm` (see
  [secrets](configuration/index.md#secrets) for which files are read without a restart).
- **Trusted proxies**: set `DGXMM_TRUSTED_PROXIES` in `.env`, then `./install.sh` (or set
  `trusted_proxies` in `config.yaml`, then `docker restart dgxmm`).
- **LiteLLM connection** (URLs, master key, on/off): change it on the LiteLLM page. It applies
  immediately and wins over `config.yaml` (see
  [configuration](configuration/litellm.md#runtime-settings-and-precedence)). Turning the integration off or
  changing the internal URL first removes DGXMM's registrations (models with `model_info.dgxmm_id`) from
  the LiteLLM being left, with that LiteLLM's key; the audit log entry records how many were removed. If
  it was unreachable, the change is saved anyway with a warning: remove the leftover DGXMM models in that
  LiteLLM, or re-enable the integration with that URL so the next save/sync cleans them up.

## Freeing disk space

Delete models you no longer need from **Models → On disk** (**Delete…**), and remove interrupted
downloads with **Clean up** or **Clean up incomplete** (see [usage](user-guide/models.md#delete-weights)). Both are
admin-only and audited.

- **Models in use** cannot be deleted: stop the deployments the dialog names first. A model whose
  download is running cannot be deleted either: cancel it on the Downloads page.
- **Stopped or failed deployments** that use a deleted model stay in the list but fail to start with
  "model not found" until you download the model again, reconfigure them to another model, or remove
  them.
- **Root-owned files.** vLLM containers run as root and can leave files in the HF cache that your user
  cannot delete. DGXMM removes them through a short-lived helper container that runs as root with
  only the model's parent directory mounted and no network (see
  [architecture](architecture/storage.md#deleting-weights)). The helper uses DGXMM's own image and is never
  pulled. If it cannot run (for example `helper_image` names a missing image), DGXMM deletes what it can
  itself and fails with `delete_failed` and a hint such as:

  ```bash
  sudo chown -R <uid>:<gid> /home/<user>/.cache/huggingface/hub/models--org--name
  ```

  The uid and gid are `DGXMM_UID` and `DGXMM_GID` from `.env`. Run it, then delete again.
- **Shared files.** When two revisions of a repo share a file, deleting one GGUF only removes its link
  and frees nothing; the dialog says so. Delete the whole repo to free the space.
- **Clean up** skips models with a partial file written in the last 10 minutes, so a download that is
  still being written by another tool is not removed under it.

Deleting files outside DGXMM (for example with `rm` or `huggingface-cli delete-cache`) is fine too.
The next rescan (or **Rescan**) updates the list.

## Next steps

- [Troubleshooting](operations/troubleshooting.md)
- [Backup and restore](operations/backup.md)
