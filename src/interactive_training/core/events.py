"""Typed, replayable events + per-subscriber EventBus (plan §3.3)."""
from __future__ import annotations

import threading
import time
from collections import deque
from queue import Queue
from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    seq: int
    type: str
    payload: dict = Field(default_factory=dict)
    ts: float = Field(default_factory=time.time)
    branch_id: str = "main"
    round: int = 0


class EventBus:
    """Bounded ring buffer + monotonic seq; every subscriber gets its own queue
    and may replay from any seq (fixes old destructive single-queue, P0.5)."""

    def __init__(self, maxlen: int = 10_000):
        self._buf: deque[Event] = deque(maxlen=maxlen)
        self._subs: set[Queue] = set()
        self._seq = 0
        # Stamped onto every event so the multiround frontend can key charts/markers/
        # journal by round without threading it through ~20 publish() call sites; the
        # session advances it in begin_round() (design multiround_ux §3.1).
        self.current_round = 0
        self._lock = threading.Lock()

    def publish(self, type: str, payload: dict | None = None, branch_id: str = "main") -> Event:
        with self._lock:
            ev = Event(seq=self._seq, type=type, payload=payload or {},
                       branch_id=branch_id, round=self.current_round)
            self._seq += 1
            self._buf.append(ev)
            subs = tuple(self._subs)
        for q in subs:
            q.put(ev)
        return ev

    def subscribe(self, since: int | None = None) -> Queue:
        q: Queue = Queue()
        with self._lock:
            if since is not None:
                for ev in self._buf:
                    if ev.seq >= since:
                        q.put(ev)
            self._subs.add(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def replay(self, since: int = 0) -> list[Event]:
        with self._lock:
            return [ev for ev in self._buf if ev.seq >= since]
