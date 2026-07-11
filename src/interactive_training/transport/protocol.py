"""Single versioned wire schema for Action/Event — no nested JSON strings (plan §5)."""
from __future__ import annotations

from interactive_training.core.actions import Action
from interactive_training.core.events import Event

WIRE_VERSION = 2


def encode_action(action: Action) -> dict:
    return {"v": WIRE_VERSION, **action.model_dump()}


def decode_action(data: dict) -> Action:
    data = {k: v for k, v in data.items() if k != "v"}
    return Action(**data)


def encode_event(event: Event) -> dict:
    return {"v": WIRE_VERSION, **event.model_dump()}


def decode_event(data: dict) -> Event:
    data = {k: v for k, v in data.items() if k != "v"}
    return Event(**data)
