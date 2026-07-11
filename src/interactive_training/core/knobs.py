"""Knob registry: generalizes "update lr" to any controllable value (plan §3.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field


@dataclass
class Knob:
    name: str
    get: Callable[[], Any]
    set: Callable[[Any], None]
    dtype: type = float
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""

    def clamp(self, value: Any) -> Any:
        value = self.dtype(value)
        if isinstance(value, (int, float)):
            if self.min is not None:
                value = max(value, self.min)
            if self.max is not None:
                value = min(value, self.max)
        return value


class KnobView(BaseModel):
    name: str
    value: Any = None
    dtype: str = "float"
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""


class KnobRegistry:
    def __init__(self):
        self._knobs: dict[str, Knob] = {}

    def register(self, knob: Knob) -> None:
        self._knobs[knob.name] = knob

    def get(self, name: str) -> Knob:
        return self._knobs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._knobs

    def set_value(self, name: str, value: Any) -> Any:
        knob = self._knobs[name]
        clamped = knob.clamp(value)
        knob.set(clamped)
        return clamped

    def views(self) -> list[KnobView]:
        out = []
        for k in self._knobs.values():
            try:
                value = k.get()
            except Exception:
                value = None
            out.append(KnobView(name=k.name, value=value, dtype=k.dtype.__name__,
                                 min=k.min, max=k.max, step=k.step, description=k.description))
        return out


def optimizer_lr_accessors(optimizer, lr_scheduler=None):
    """(get, set) for lr reproducing the old scheduler base_lrs/initial_lr logic (plan §3.1)."""
    import torch

    def _get():
        if lr_scheduler is not None and getattr(lr_scheduler, "base_lrs", None):
            return float(lr_scheduler.base_lrs[0])
        return float(optimizer.param_groups[0]["lr"])

    def _assign(group_key, group, new_lr):
        if isinstance(group.get(group_key), torch.Tensor):
            group[group_key].fill_(new_lr)
        else:
            group[group_key] = new_lr

    def _set(new_lr):
        new_lr = float(new_lr)
        if lr_scheduler is not None:
            if hasattr(lr_scheduler, "base_lrs"):
                lr_scheduler.base_lrs = [new_lr] * len(lr_scheduler.base_lrs)
            for group in optimizer.param_groups:
                if "lr" in group:
                    _assign("lr", group, new_lr)
                if "initial_lr" in group:
                    _assign("initial_lr", group, new_lr)
        else:
            for group in optimizer.param_groups:
                if "lr" in group:
                    _assign("lr", group, new_lr)

    return _get, _set
