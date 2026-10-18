"""Wiring of the operations features (spec features 1, 8, 9, 10): health triage, reading labels, threshold tuning
and routing advice — their insight-engine jobs and the hooks that trigger them. Called from
``features.wire_features``; idempotent (a context is wired once)."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def wire_operations(ctx) -> None:
    engine = getattr(ctx, "insight_engine", None)
    if engine is None or ctx.health is not None:
        return
    from dgxmm.decider import health, tuning
    ctx.health = health.HealthService(ctx)
    engine.register_job(health.JOB, "health", health.CHECK_INTERVAL_S, ctx.health.run,
                        budget_calls=health.BUDGET_CALLS, timeout_s=240.0)
    if ctx.deployer is not None and hasattr(ctx.deployer, "removal_hooks"):
        ctx.deployer.removal_hooks.append(ctx.health.forget)
    engine.register_job(tuning.JOB, "tuning", tuning.SCHEDULE, tuning.job, budget_calls=0, timeout_s=120.0)

    from dgxmm.decider import reading
    ctx.readings = reading.ReadingService(ctx)
    engine.register_job(reading.JOB, "reading", reading.SWEEP_S, ctx.readings.run, budget_calls=reading.BUDGET_CALLS,
                        timeout_s=180.0)
    calibration = getattr(ctx.deployer, "calibration", None) if ctx.deployer is not None else None
    for store in (calibration, getattr(ctx, "benchmarks", None)):
        if store is not None and hasattr(store, "listeners"):
            store.listeners.append(ctx.readings.notify)

    from dgxmm.decider import routing
    ctx.routing = routing.RoutingService(ctx)
    engine.register_job(routing.JOB, "routing", routing.SCHEDULE, ctx.routing.run, budget_calls=routing.BUDGET_CALLS,
                        timeout_s=180.0)
    if ctx.deployer is not None and hasattr(ctx.deployer, "state_hooks"):
        ctx.deployer.state_hooks.append(ctx.routing.on_state)
    # "-> ready" comes before DGXMM's LiteLLM sync has registered the deployment: refresh again once it has (both
    # notices settle into one run)
    sync = getattr(ctx, "litellm", None)
    if sync is not None and hasattr(sync, "listeners"):
        sync.listeners.append(ctx.routing.refresh)
