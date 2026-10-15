import asyncio
from typing import Any


class EventBroker:
    """In-process fan-out of (topic, data) events to SSE subscribers."""

    def __init__(self, max_queue: int = 256):
        self._subs: set[asyncio.Queue] = set()
        self._max = max_queue

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(self._max)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, topic: str, data: Any) -> None:
        for q in list(self._subs):
            if q.full():  # slow client: drop its oldest event
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait((topic, data))
