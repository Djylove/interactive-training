"""lr / kl_coef / temperature / clip_range knobs + AverageReward goal (plan §7.3)."""
from __future__ import annotations

from typing import MutableMapping

from interactive_training.core.goals import AverageReward
from interactive_training.recipes._common import bind_dict_knob


def rlvr(session, cfg: MutableMapping, optim=None, lr_scheduler=None, metric: str = "reward"):
    if optim is not None:
        session.register_optimizer_lr(optim, lr_scheduler)
    else:
        bind_dict_knob(session, cfg, "lr", min=0.0, description="learning rate")
    bind_dict_knob(session, cfg, "kl_coef", min=0.0, description="GRPO KL penalty coefficient")
    bind_dict_knob(session, cfg, "temperature", min=0.0, description="sampling temperature")
    bind_dict_knob(session, cfg, "clip_range", min=0.0, description="PPO/GRPO clip range")
    if session.goal is None:
        session.goal = AverageReward(metric)
    return session
