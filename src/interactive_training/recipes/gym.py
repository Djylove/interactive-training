"""epsilon / gamma / lr / target_update_freq knobs + AverageReward goal (plan §7.3)."""
from __future__ import annotations

from typing import MutableMapping

from interactive_training.core.goals import AverageReward
from interactive_training.recipes._common import bind_dict_knob


def gym(session, cfg: MutableMapping, metric: str = "episode_reward"):
    bind_dict_knob(session, cfg, "epsilon", min=0.0, max=1.0, description="epsilon-greedy exploration")
    bind_dict_knob(session, cfg, "gamma", min=0.0, max=1.0, description="discount factor")
    bind_dict_knob(session, cfg, "lr", min=0.0, description="learning rate")
    bind_dict_knob(session, cfg, "target_update_freq", dtype=int, min=1, description="target network update frequency")
    if session.goal is None:
        session.goal = AverageReward(metric)
    return session
