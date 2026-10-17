"""Wiring of the five decision-model features into the app (called from context.build_context)."""
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
