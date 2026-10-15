"""Shared service container stored on app.state.ctx."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dgxmm.auth.ratelimit import SlidingWindowLimiter
from dgxmm.config import Settings
from dgxmm.db import Database
from dgxmm.events import EventBroker


@dataclass
class AppContext:
    settings: Settings
    db: Database
    events: EventBroker
    login_limiter: SlidingWindowLimiter = field(default_factory=lambda: SlidingWindowLimiter(5, 60))
    # Filled in by later tasks (None in unit tests that do not need them):
    inventory: Any = None
    budget: Any = None
    deployer: Any = None
    docker: Any = None
    engines: dict = field(default_factory=dict)
    background: list = field(default_factory=list)  # async callables started by lifespan
