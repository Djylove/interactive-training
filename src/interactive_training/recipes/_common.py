"""Shared helper: bind a knob to a key in a user-owned config dict (plan §7.3)."""
from __future__ import annotations

from typing import Any, MutableMapping


def bind_dict_knob(session, cfg: MutableMapping[str, Any], name: str, **meta) -> None:
    session.register_knob(
        name,
        get=lambda: cfg.get(name),
        set=lambda v: cfg.__setitem__(name, v),
        **meta,
    )
