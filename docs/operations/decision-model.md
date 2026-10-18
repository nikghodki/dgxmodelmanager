---
description: "Installing, enabling, upgrading and operating the optional decision model."
---

# Running the decision model

The decision model ("decider", `vllm-sr/d3-mini` at a pinned revision) is an optional local model that
DGXMM consults for failure diagnosis, parser suggestions, plan recommendations, model kinds and
workload labels. Every feature falls back to its rules when the decider is off or unhealthy. It runs
as a *system* deployment: about 13 GiB of GPU memory, never stopped by plans, never registered in LiteLLM,
and stopped by the memory watchdog before any user deployment (when it uses more than its reservation, or as the
first `auto_protect` victim; the supervisor restarts it later with backoff).

**Install with the installer:**

```bash
./install.sh --with-decider          # or DGXMM_WITH_DECIDER=1 ./install.sh; remembered in .env
```

After DGXMM is healthy the installer runs, inside the `dgxmm` container:

1. `decider build`: builds the decider image (`images.decision`, default `dgxmm-decider:3`) from
   `deploy/decider` with the same code as the UI's **Build image** button. The first build pulls the
   pinned vLLM base (about 21 GB) and takes a few minutes. It is skipped when the image exists.
2. `decider download`: downloads the pinned snapshot into the HF cache (8.5 GiB) with progress,
   resumable after an interruption (re-run the installer). Only files with a known SHA-256 are fetched:
   code, config, tokenizer and `MODEL_MANIFEST.json` from the hash list in `deploy/decider`, then the
   files that manifest lists (weights, licence, card; `assets/` is skipped). Anything else the Hub lists
   is ignored, and unsafe names (absolute, `..`) are refused. Each file is hashed while it downloads and
   only placed in the cache when it matches; a mismatch is discarded, downloaded once more, and if it
   still does not match the step fails with nothing linked. It checks free disk space first, and only
   one download runs at a time. A complete, verified snapshot needs no network. The HF token is never
   sent (the repository is public).
3. `decider enable --wait`: enables the decider through the running app, with the normal memory check.
   If it does not fit, the installer lists the plans that would make room and changes nothing: those
   plans stop or sleep your deployments, so pick one in the UI. Then it waits until the decider is
   healthy.

The installer builds and downloads before enabling (rather than letting the app do it on enable), so
progress shows in the terminal, failures stop the install at the step that failed, and the slow parts
can be retried on their own. Enabling from the UI still offers the in-app image build.

**Admin CLI.** The same commands work by hand:

```bash
docker exec dgxmm python -m dgxmm.admin decider status        # --json, or --check (exit 1 unless healthy)
docker exec dgxmm python -m dgxmm.admin decider build         # --force rebuilds
docker exec dgxmm python -m dgxmm.admin decider download      # --verify-only checks without downloading
docker exec dgxmm python -m dgxmm.admin decider download --model vllm-sr/d3-nano   # the other supported model
docker exec dgxmm python -m dgxmm.admin decider enable --wait # --timeout SECONDS (default 900)
docker exec dgxmm python -m dgxmm.admin decider disable       # --remove also removes the deployment
```

`build` and `download` work on their own. `status`, `enable` and `disable` call the app's API over its
admin unix socket inside the container (`data/run/admin.sock`, mode 0600 in a 0700 directory, served by
a second listener without proxy-header handling) with a short-lived admin token that the command creates
in the database and deletes when it finishes. The token is never printed. It is refused on the TCP port
(whatever `X-Forwarded-For` says) and on any request with forwarding headers, and accepted only on the
decider routes (and the inventory refresh), for at most the wait time plus 10 minutes (so
`--timeout` is at most 10200 s), and only while its user is an admin. It works before the first admin
login (pending password change). The audit log shows the oldest admin as the actor (`--user NAME` picks
another admin), marks these actions `"via": "cli"`, and records each token's issue and revocation.

**Remembered vs. requested.** With `DGXMM_WITH_DECIDER=1` only remembered in `.env`, an install or
upgrade never fails because of the decider: low memory, a failed build, download or enable are warnings
(DGXMM itself is upgraded; the app's memory check decides at enable), and a decider an admin disabled
stays disabled. Passing `--with-decider` on the command line means "set it up now": the memory and disk
checks are fatal, failures exit non-zero, and a disabled decider is enabled again.

**Turning it off:** disable it on the Decision model page (or `decider disable`). The installer leaves it
off; set `DGXMM_WITH_DECIDER=0` in `.env` to stop it trying altogether.

`./install.sh status` shows the decider's state and, with `DGXMM_WITH_DECIDER=1`, exits non-zero while it
is not healthy. The UI's top bar shows a **Decision model** chip (Healthy, Building, Starting, Stopping,
Sleeping, Stopped, Disabled or Error) that opens the Decision model page.

**Install labels.** The installer generates `DGXMM_INSTALL_ID` once (kept in `.env`); DGXMM labels the
containers it creates and the images it builds `dgxmm.install=<id>`, so uninstall touches only this
install's decider container and image. Containers and images from before the label are reported and
kept (remove them by hand).

**Data retention.** The decision log (`decisions`) keeps 30 days and at most the newest 50 000 rows
(`decider_retention_days`, `decider_max_rows` in `config.yaml`). Rows with feedback (what threshold tuning learns
from) are exempt from both and kept 180 days (`decider_feedback_retention_days`, never fewer than
`decider_retention_days`). Stored diagnoses (`diagnoses`) keep the latest
20 per deployment and are deleted with the deployment. The metrics loop prunes both once a minute (with the
metrics rollups, which keep 30 days). Feedback on a pruned decision answers 404.

## Next steps

- [Decision model (user guide)](../user-guide/decision-model.md)
- [Troubleshooting](troubleshooting.md#decision-model-not-healthy)
