"""In-process ActionBus (agents -> session) + EventBus re-export (plan §5)."""
from __future__ import annotations

import threading
from queue import Empty, Queue

from interactive_training.core.actions import Action, ActionResult
from interactive_training.core.events import EventBus  # re-exported; subscribers read it

__all__ = ["ActionBus", "EventBus"]


class ActionBus:
    """All action sources (transport, agents) push typed Actions here; the session
    drains it at each control point. Supports blocking submit via per-id acks (§3.7)."""

    def __init__(self):
        self._q: Queue[Action] = Queue()
        self._acks: dict[str, tuple[threading.Event, list]] = {}
        self._lock = threading.Lock()

    def submit(self, action: Action, wait: bool = False, timeout: float | None = None) -> ActionResult | None:
        if wait:
            done = threading.Event()
            holder: list[ActionResult] = []
            with self._lock:
                self._acks[action.id] = (done, holder)
        self._q.put(action)
        if wait:
            done.wait(timeout)
            return holder[0] if holder else None
        return None

    def drain(self) -> list[Action]:
        out: list[Action] = []
        while True:
            try:
                out.append(self._q.get_nowait())
            except Empty:
                break
        return out

    def ack(self, action_id: str, result: ActionResult) -> None:
        with self._lock:
            entry = self._acks.pop(action_id, None)
        if entry is not None:
            entry[1].append(result)
            entry[0].set()
