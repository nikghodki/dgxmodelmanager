---
description: "The admin CLI inside the dgxmm container: password resets, calibration and the decision model."
---

# Admin command line

Every admin command runs inside the `dgxmm` container, as `docker exec dgxmm python -m dgxmm.admin
<command>` (`--help` on any command lists its options):

| Command | What it does |
|---|---|
| `reset-password USER` | Sets a new random password (printed once), signs the user out everywhere and requires a change at next sign-in. See [forgot the admin password](troubleshooting.md#forgot-the-admin-password). |
| `list-users` | Lists user names and roles. |
| `reset-calibration REPO [--engine vllm\|llamacpp]` | Forgets the learned memory calibration of a model, so estimates start from the formula again. |
| `decider status [--json] [--check]` | The decision model's state; `--check` exits 1 unless it is enabled and healthy. |
| `decider build [--force]` | Builds the decider image (skipped when it exists). |
| `decider download [--model vllm-sr/d3-nano] [--verify-only]` | Downloads the pinned model snapshot into the HF cache and verifies every file; `--verify-only` only checks it. |
| `decider enable [--wait] [--timeout SECONDS]` | Enables the decider through the running app (normal memory check); `--wait` waits until it is healthy (default 900 s). |
| `decider disable [--remove]` | Disables it and stops its deployment; `--remove` also deletes the deployment. |
| `decider admin-disabled` | Exits 0 when an admin explicitly disabled the decider (used by the installer). |
| `rotate-decider-key [--wait SECONDS]` | New bearer key for the decider; a running decider is restarted with it. Exit 0 rotated, 1 failed, 2 not picked up by DGXMM, 3 still in progress (run it again to keep waiting). Default wait 90 s. |

`decider status|enable|disable` act as the oldest admin (`decider --user NAME …` picks another one) over
the admin unix socket; see [decision model](decision-model.md). Every command that changes something is
written to the audit log.

## Next steps

- [Troubleshooting](troubleshooting.md)
