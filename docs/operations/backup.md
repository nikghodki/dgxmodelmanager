---
description: "Back up and restore DGXMM's database, secrets and configuration."
---

# Backup and restore

What to keep:

| Path | Contains |
|---|---|
| `data/` | `dgxmm.sqlite3`: users, deployments and their configs, events, calibration, benchmarks, 30-day metrics, audit log, runtime settings (including the decision model's on/off switch, deployment, per-feature switches and thresholds: `app_settings` keys `decider.*`), the decision log (`decisions`, with feedback), failure diagnoses, model-kind cache and `schema_version`. Also cached vLLM architecture lists. |
| `data/run/admin.sock` | The admin CLI's unix socket while DGXMM runs. Not data: `tar` skips it ("socket ignored") and DGXMM recreates it at start. |
| `data/secrets/decider_api_key` | The decision model's API key (mode 600; part of `data/`). If it is lost, a new one is generated and the decider has to be restarted (disable, then enable) to use it. |
| `data/litellm-db/` | Bundled LiteLLM only: its Postgres database (virtual keys, budgets, spend, registered models) |
| `secrets/` | HF token, LiteLLM keys (incl. the bundled LiteLLM's master/salt key and DB password), first admin password |
| `config.yaml`, `.env` | your configuration |
| `docker-compose.override.yaml` | only if you created one |

Model files (HF cache, GGUF directories) are not part of a DGXMM backup. Back them up separately or
download them again. For the decision model: `docker exec dgxmm python -m dgxmm.admin decider download`
(or `./install.sh` with `DGXMM_WITH_DECIDER=1`) fetches and verifies its snapshot again. The decider
image is rebuilt with `decider build`.

**Backup (brief stop):**

```bash
docker stop dgxmm
tar czf dgxmm-backup-$(date +%F).tgz data secrets config.yaml .env
docker start dgxmm
```

With the bundled LiteLLM (`DGXMM_WITH_LITELLM=1`), `data/` also holds a live Postgres
(`data/litellm-db`). Copying a running database's files gives a corrupt backup, so stop the bundled
containers too, and start the database first when you are done:

```bash
docker compose --profile litellm stop dgxmm dgxmm-litellm dgxmm-litellm-db
tar czf dgxmm-backup-$(date +%F).tgz data secrets config.yaml .env
docker compose --profile litellm up -d
```

Or keep LiteLLM running and take a logical dump of its database. Back it up together with `secrets/`:
`litellm_salt_key` must match, or the stored credentials cannot be decrypted.

```bash
docker compose --profile litellm exec -T dgxmm-litellm-db pg_dump -U litellm -d litellm -Fc > litellm-$(date +%F).dump
```

Stopping `dgxmm` does not stop the engines. To back up the database without stopping DGXMM, use
SQLite's online backup:

```bash
docker exec dgxmm python -c "import sqlite3; s = sqlite3.connect('/app/data/dgxmm.sqlite3'); d = sqlite3.connect('/app/data/dgxmm-backup.sqlite3'); s.backup(d); d.close()"
# then copy data/dgxmm-backup.sqlite3 somewhere safe, together with secrets/, config.yaml and .env
```

**Restore** (same or new host):

```bash
git clone https://github.com/nikghodki/dgxmodelmanager.git && cd dgxmodelmanager
tar xzf /path/to/dgxmm-backup-YYYY-MM-DD.tgz        # data/, secrets/, config.yaml, .env
./install.sh
```

With the bundled LiteLLM, the tarball's `data/litellm-db` and `secrets/` come back as they were and
`./install.sh` starts the database before LiteLLM. To restore a `pg_dump` instead: run `./install.sh`
first (this creates an empty database), then stop LiteLLM, load the dump and start it again:

```bash
docker compose --profile litellm stop dgxmm-litellm
docker compose --profile litellm exec -T dgxmm-litellm-db pg_restore -U litellm -d litellm --clean --if-exists < litellm-YYYY-MM-DD.dump
docker compose --profile litellm start dgxmm-litellm
```

Keep the original `secrets/litellm_salt_key` (and `litellm_bundled_master_key`) when you restore a dump.

On a different host, check `.env` first: `DGXMM_UID`/`DGXMM_GID`, `DOCKER_GID` (`stat -c %g
/var/run/docker.sock`) and the paths must fit the new machine. The installer warns when the uid or
Docker group do not match. If you restored from the online backup, rename `dgxmm-backup.sqlite3` to
`dgxmm.sqlite3` and make sure no stale `dgxmm.sqlite3-wal` file sits next to it. Deployments whose
containers do not exist on the new host are marked `failed` at startup. Start them from the UI.

The decision model's settings and decision log come back with `dgxmm.sqlite3`. If the decider was
enabled, DGXMM restarts its deployment like any other (on a new host: download its snapshot and build
its image first, see above, then **Enable** on the Decision model page or `decider enable --wait`). This
procedure (online backup, restore into a new data directory, start) was checked: thresholds, feature
switches, the decision log and the decider key came back unchanged.

## Next steps

- [Upgrade](../getting-started/upgrade.md)
- [Data model](../architecture/data-model.md)
