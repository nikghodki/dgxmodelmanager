"""Security review L4: the new decision-model routes and jobs do no blocking database or filesystem work on the event
loop: inside ``async def`` bodies, decision-log writes, audit records, database sessions and directory walks go
through a worker thread (``asyncio.to_thread``, ``run.to_thread``, ``arecord_rules``)."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["dgxmm/api/commands.py", "dgxmm/api/schedules.py", "dgxmm/decider/compat.py", "dgxmm/decider/draft.py",
           "dgxmm/decider/engine.py", "dgxmm/decider/remote_code.py", "dgxmm/decider/schedule.py",
           "dgxmm/decider/command.py", "dgxmm/decider/routing.py", "dgxmm/decider/reading.py",
           "dgxmm/decider/health.py"]
BLOCKING = {"safe_record_rules", "audit", "walk", "session", "auto_map_findings", "cache_get", "cache_put"}


def _blocking_calls(tree):
    out = []

    def visit(node, fn):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.Lambda)):
                continue  # runs wherever it is called: a thread, here
            if isinstance(child, ast.AsyncFunctionDef):
                visit(child, child.name)
                continue
            if fn and isinstance(child, ast.Call):
                f = child.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                if name in BLOCKING:
                    out.append(f"{fn}:{child.lineno} {name}()")
            visit(child, fn)
    visit(tree, None)
    return out


@pytest.mark.parametrize("module", MODULES)
def test_no_blocking_database_or_filesystem_calls_on_the_event_loop(module):
    tree = ast.parse((ROOT / module).read_text())
    assert _blocking_calls(tree) == []
