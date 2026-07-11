"""Control-point barrier: apply-before-advance, pause/stop gate, DDP sync (plan §3.7)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepControl:
    """Post-barrier decision returned by session.step() (plan §3.7)."""
    stop: bool = False
    load: str | None = None
    reload_required: bool = False
    evaluate: bool = False
    save: bool = False
    tag: str | None = None
    reset_modules: list[str] = field(default_factory=list)
    knob_updates: dict = field(default_factory=dict)
    applied: list[dict] = field(default_factory=list)


class ControlGate:
    """Single-process pause/resume/stop gate. Pause blocks the loop until resume/stop
    so a stop submitted while paused is still honored (fixes P0.4)."""

    def __init__(self):
        self._resume = threading.Event()
        self._resume.set()
        self._stop = False

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    @property
    def stopped(self) -> bool:
        return self._stop

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop = True
        self._resume.set()

    def wait_if_paused(self, drain: Any | None = None) -> None:
        # While paused, keep re-draining the action bus so resume/stop land.
        while not self._resume.is_set():
            if drain is not None:
                drain()
                if self._stop:
                    return
            self._resume.wait(timeout=0.05)


def broadcast_and_barrier(payload: dict, rank: int, process_group=None) -> dict:
    """Rank 0 owns the decision; broadcast it to all ranks then a collective barrier
    releases them with identical state (plan §3.7 layer 3)."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return payload
    box = [payload if rank == 0 else None]
    dist.broadcast_object_list(box, src=0, group=process_group)
    dist.barrier(group=process_group)
    return box[0]
