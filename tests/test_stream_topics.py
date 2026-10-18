"""L4: SSE topics are filtered by role: the decider image build log only reaches admins."""
import asyncio

import pytest

from dgxmm.api.stream import ADMIN_ONLY_TOPICS, sse_events, visible

VIEWER_TOPICS = ["alert", "benchmark", "budget", "deployment", "deployment_removed", "diagnosis", "download",
                 "inventory", "metrics", "insights"]


async def _collect(is_admin: bool, events):
    q: asyncio.Queue = asyncio.Queue()
    for e in events:
        q.put_nowait(e)
    seen, n = [], [0]

    async def disconnected():
        n[0] += 1
        return n[0] > len(events)

    async for chunk in sse_events(q, is_admin, disconnected, ping_s=0.01):
        seen.append(chunk)
    return [c.split("\n")[0].removeprefix("event: ") for c in seen if c.startswith("event: ")]


EVENTS = [("decider", {"phase": "step", "message": "SBOM torch==2.9"}), ("deployment", {"id": 1}),
          ("diagnosis", {"deployment_id": 1}), ("budget", {})]


async def test_viewers_never_receive_the_decider_build_log():
    assert await _collect(False, EVENTS) == ["hello", "deployment", "diagnosis", "budget"]


async def test_admins_receive_every_topic():
    assert await _collect(True, EVENTS) == ["hello", "decider", "deployment", "diagnosis", "budget"]


@pytest.mark.parametrize("topic", VIEWER_TOPICS)
def test_documented_viewer_topics(topic):
    assert visible(topic, False) and visible(topic, True)


def test_stream_module_classifies_the_same_viewer_topics():
    from dgxmm.api.stream import VIEWER_TOPICS as CLASSIFIED
    assert CLASSIFIED == set(VIEWER_TOPICS) and not CLASSIFIED & ADMIN_ONLY_TOPICS


def test_admin_only_topics():
    assert ADMIN_ONLY_TOPICS == {"decider"} and not visible("decider", False)


def test_every_published_topic_is_classified():
    """A new topic must be added to the viewer list (and docs/security.md) or to ADMIN_ONLY_TOPICS."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1] / "dgxmm"
    topics = set()
    for f in root.rglob("*.py"):
        topics |= set(re.findall(r"\.publish\(\"([a-z_]+)\"", f.read_text()))
    assert topics <= set(VIEWER_TOPICS) | ADMIN_ONLY_TOPICS, topics - set(VIEWER_TOPICS) - ADMIN_ONLY_TOPICS
    doc = (root.parent / "docs" / "security.md").read_text()
    for t in VIEWER_TOPICS + sorted(ADMIN_ONLY_TOPICS):
        assert f"`{t}`" in doc, t
