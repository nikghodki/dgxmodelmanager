---
description: "DGXMM's logs, health endpoints and what to monitor."
---

# Logs and monitoring

## Logs

```bash
./install.sh logs        # last 200 lines of the dgxmm container (DGXMM_LOG_LINES=1000 for more)
./install.sh logs -f     # follow
docker logs dgxmm-<deployment-name>     # an engine container
docker logs dgxmm-decider               # the decision model (while its container exists)
```

In the UI, a deployment's **Logs** button shows its engine log (admins). The **Events** tab keeps the
last log lines of every failure, even after the container has been removed. The DGXMM log covers
startup reconciliation, memory watchdog changes, LiteLLM sync failures, calibration measurements, the
decision model's supervisor (probes, restarts, "the running decider predates this DGXMM build"),
decider retention passes and errors. The log is managed by Docker, and DGXMM writes no log files.
Decision-model questions and answers are not logged there: they are in the decision log on the Decision
model page (`GET /api/decider/decisions`).

## Health

```bash
./install.sh status
```

This shows the container state and health, the checkout it belongs to, whether `.env`, `config.yaml`
and each secret are present, the running engine containers and the URLs. It exits 0 only when DGXMM is
running and healthy, so you can use it in monitoring scripts.

The container health check (every 10 s) requests `http://127.0.0.1:8090/dgxmm/` inside the
container. From the host:

```bash
docker inspect --format '{{.State.Health.Status}}' dgxmm
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8090/dgxmm/     # 200
```

The container restarts automatically (`restart: unless-stopped`), also after a reboot.

Delete helpers only exist while a deletion runs. This should normally list nothing:

```bash
docker ps -a --filter label=dgxmm.helper=true
```

## Next steps

- [Troubleshooting](troubleshooting.md)
- [Admin command line](admin-cli.md)
