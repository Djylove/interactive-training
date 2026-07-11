"""lr_g / lr_d / n_critic knobs + freeze(module) action (plan §7.3)."""
from __future__ import annotations

from typing import MutableMapping

from interactive_training.core.actions import ActionResult
from interactive_training.core.knobs import optimizer_lr_accessors
from interactive_training.recipes._common import bind_dict_knob


def gan(session, g_opt, d_opt, cfg: MutableMapping | None = None):
    g_get, g_set = optimizer_lr_accessors(g_opt)
    d_get, d_set = optimizer_lr_accessors(d_opt)
    session.register_knob("lr_g", g_get, g_set, min=0.0, description="generator learning rate")
    session.register_knob("lr_d", d_get, d_set, min=0.0, description="discriminator learning rate")
    if cfg is not None:
        bind_dict_knob(session, cfg, "n_critic", dtype=int, min=1, description="discriminator steps per generator step")

    @session.action("freeze", "Freeze/unfreeze a module's parameters", ["module_name", "freeze"])
    def _freeze(payload, sess):
        if sess._model is None:
            return ActionResult.fail("no model bound")
        target = sess._model
        for part in payload["module_name"].split("."):
            if not hasattr(target, part):
                return ActionResult.fail(f"module not found: {payload['module_name']}")
            target = getattr(target, part)
        requires = not payload.get("freeze", True)
        for p in target.parameters():
            p.requires_grad_(requires)
        return ActionResult.success(module_name=payload["module_name"], frozen=not requires)

    return session
