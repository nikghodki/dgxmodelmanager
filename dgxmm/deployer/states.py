STATES = ("planned", "queued", "starting", "loading", "ready", "sleeping", "stopping", "stopped", "failed", "unknown")
TRANSITIONS: dict[str, set[str]] = {
    "planned": {"queued", "starting", "failed", "stopped"},
    # queued: waiting for a start-up slot (deployer/gate.py); holds its memory reservation and port, no container
    "queued": {"starting", "failed", "stopped"},
    "starting": {"loading", "ready", "failed", "stopping"},
    "loading": {"ready", "failed", "stopping"},
    "ready": {"sleeping", "stopping", "failed"},
    "sleeping": {"ready", "stopping", "failed"},
    "stopping": {"stopped", "failed"},
    "stopped": {"queued", "starting", "failed"},
    "failed": {"queued", "starting", "stopped", "failed"},
    "unknown": {"stopping", "failed", "ready"},
}
ACTIVE_STATES = frozenset({"queued", "starting", "loading", "ready", "sleeping", "stopping", "unknown"})


class InvalidTransition(Exception):
    pass


def assert_transition(current: str, target: str) -> None:
    if target not in TRANSITIONS.get(current, set()):
        raise InvalidTransition(f"cannot go from {current} to {target}")
