"""Wiring of the decision-model features into the app (called from context.build_context).

The features themselves are declared in the registry (dgxmm/decider/registry.py). Feature lanes add their hooks and
insight-engine jobs to ``wire_features`` (``ctx.insight_engine.register_job(...)``); the engine itself is started
here once."""
from __future__ import annotations

from dgxmm.decider.protocol import decider_of


def ensure_features(ctx) -> None:
    """Create the feature services on ctx if missing (idempotent; tests build bare AppContexts)."""
    from dgxmm.decider.diagnosis import DiagnosisService
    from dgxmm.decider.kind import KindService
    from dgxmm.decider.workload import WorkloadService
    if ctx.diagnosis is None:
        ctx.diagnosis = DiagnosisService(ctx)
    if ctx.workload is None:
        ctx.workload = WorkloadService(ctx)
    if ctx.kinds is None:
        ctx.kinds = KindService(ctx)


def wire_features(ctx) -> None:
    """Hooks: diagnosis on ->failed, plan recommendation in preview, cached kinds on every inventory scan and a
    background kind classifier."""
    ensure_features(ctx)
    if ctx.deployer is not None:
        ctx.deployer.state_hooks.append(ctx.diagnosis.on_state)
        ctx.deployer.removal_hooks.append(ctx.workload.forget)
        ctx.deployer.removal_hooks.append(ctx.diagnosis.forget)
        ctx.deployer.decider_getter = lambda: decider_of(ctx)
        ctx.deployer.metrics = ctx.metrics
    if ctx.inventory is not None and hasattr(ctx.inventory, "post_scan"):
        ctx.inventory.post_scan.append(ctx.kinds.apply_cached)
        ctx.background.append(ctx.kinds.run)
    if ctx.deployer is not None and ctx.decider is not None and getattr(ctx.decider, "supervisor", None) is None:
        from dgxmm.decider.supervisor import DeciderSupervisor
        ctx.decider.supervisor = DeciderSupervisor(decider=ctx.decider, deployer=lambda: ctx.deployer,
                                                   watchdog=lambda: ctx.watchdog, settings=ctx.settings, db=ctx.db)
        ctx.background.append(ctx.decider.supervisor.run)
        if ctx.watchdog is not None and hasattr(ctx.watchdog, "system_stop_listeners"):
            ctx.watchdog.system_stop_listeners.append(ctx.decider.supervisor.on_watchdog_stop)
    from dgxmm.decider.operations import wire_operations
    wire_operations(ctx)  # health triage, reading labels, threshold tuning, routing advice
    wire_schedules(ctx)
    # ---- advanced features (spec 2026-10-17): register insight-engine jobs above this line ----
    if ctx.insight_engine is not None and ctx.insight_engine.run not in ctx.background:
        ctx.background.append(ctx.insight_engine.run)  # idle (no jobs) until features register some


# Feature 7 (sleep/wake schedules): profiles at night, proposals after them, the runner every minute.
PROFILES_AT, PROPOSALS_AT = "daily 03:10", "daily 03:40"
PROPOSAL_BUDGET = 24  # decider calls per night (MAX_CANDIDATES per deployment)


def wire_schedules(ctx) -> None:
    from dgxmm.decider.schedule import ScheduleService
    if ctx.schedules is None:
        ctx.schedules = ScheduleService(ctx)
    svc = ctx.schedules
    if ctx.deployer is not None and svc.on_deployment_removed not in ctx.deployer.removal_hooks:
        ctx.deployer.removal_hooks.append(svc.on_deployment_removed)
        if hasattr(ctx.deployer, "before_removal_hooks"):
            ctx.deployer.before_removal_hooks.append(svc.before_deployment_removed)
    eng = ctx.insight_engine
    if eng is None or "schedule_runner" in {j["name"] for j in eng.status()}:
        return

    async def profiles(run):
        out = await run.to_thread(svc.build_profiles)
        return profiles_text(out)

    async def proposals(run):
        return await svc.propose(run)

    async def runner(run):
        return await svc.run_due()

    eng.register_job("traffic_profiles", "schedule", PROFILES_AT, profiles)
    eng.register_job("schedule_proposals", "schedule", PROPOSALS_AT, proposals, budget_calls=PROPOSAL_BUDGET)
    # The runner executes schedules the user owns: it is not switched off with the feature's toggle (which only
    # stops new profiles and proposals) nor with the decision model; "pause all schedules" stops it (run_due).
    eng.register_job("schedule_runner", "schedule", 60, runner, timeout_s=600.0, gate=lambda: True)


def profiles_text(out: dict) -> str:
    """The traffic_profiles job's result: "1 profile", "3 profiles"."""
    from dgxmm.decider.common import plural
    return plural(out["profiles"], "profile")
