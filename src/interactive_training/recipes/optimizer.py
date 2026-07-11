"""lr / weight_decay / grad_clip knobs (plan §7.3)."""
from __future__ import annotations

from typing import MutableMapping

from interactive_training.recipes._common import bind_dict_knob


def optimizer(session, optim, lr_scheduler=None, cfg: MutableMapping | None = None):
    session.register_optimizer_lr(optim, lr_scheduler)

    def wd_get():
        return float(optim.param_groups[0].get("weight_decay", 0.0))

    def wd_set(v):
        for g in optim.param_groups:
            g["weight_decay"] = float(v)

    session.register_knob("weight_decay", wd_get, wd_set, min=0.0, description="optimizer weight decay")
    if cfg is not None:
        bind_dict_knob(session, cfg, "grad_clip", min=0.0, description="gradient clipping max-norm")
    return session
