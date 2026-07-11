"""Tier 2: zero-edit monkeypatch of optimizer.step into a control point (plan §7.1)."""
from __future__ import annotations

from typing import Any, Callable


def autopatch(session: Any, optimizer: Any, metrics_fn: Callable[[], dict] | None = None):
    """Wrap optimizer.step so every update becomes a control-point barrier with no
    source edits. The post-barrier StepControl is stored on session.last_control."""
    original = optimizer.step

    def patched(*args, **kwargs):
        out = original(*args, **kwargs)
        metrics = metrics_fn() if metrics_fn is not None else {}
        session.last_control = session.step(metrics)
        return out

    optimizer.step = patched
    optimizer._interactive_original_step = original
    return optimizer


def unpatch(optimizer: Any) -> None:
    original = getattr(optimizer, "_interactive_original_step", None)
    if original is not None:
        optimizer.step = original
        del optimizer._interactive_original_step
