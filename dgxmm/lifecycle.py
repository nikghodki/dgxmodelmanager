"""Process-wide "the server is stopping" flag.

Set by ``dgxmm.__main__.run_server`` when the process receives SIGTERM / SIGINT, before uvicorn starts its graceful
shutdown. Long-lived responses (the SSE stream, dgxmm/api/stream.py) read it and end by themselves: otherwise uvicorn
waits the whole grace period for them and then cancels them, logging a CancelledError traceback per open tab."""
from __future__ import annotations

import threading

_stopping = threading.Event()


def request_stop() -> None:
    _stopping.set()


def stopping() -> bool:
    return _stopping.is_set()


def reset() -> None:
    """For tests that run a server in-process more than once."""
    _stopping.clear()
