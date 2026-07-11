"""Typed, extensible actions + handler registry (plan §3.2)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from pydantic import BaseModel, Field


class Action(BaseModel):
    type: str
    payload: dict = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: float = Field(default_factory=time.time)
    source: str = "unknown"


class ActionResult(BaseModel):
    ok: bool
    data: dict = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def success(cls, **data: Any) -> "ActionResult":
        return cls(ok=True, data=data)

    @classmethod
    def fail(cls, error: str) -> "ActionResult":
        return cls(ok=False, error=error)


class ActionSchema(BaseModel):
    type: str
    description: str = ""
    payload_keys: list[str] = Field(default_factory=list)


Handler = Callable[[dict, Any], ActionResult]


class ActionRegistry:
    """Maps action type -> handler. One place to add a capability (fixes P1.5)."""

    def __init__(self):
        self._handlers: dict[str, Handler] = {}
        self._docs: dict[str, tuple[str, list[str]]] = {}

    def register(self, type: str, handler: Handler, description: str = "", payload_keys: list[str] | None = None) -> None:
        self._handlers[type] = handler
        self._docs[type] = (description, payload_keys or [])

    def unregister(self, type: str) -> None:
        """Remove an action from both dispatch and advertised schemas."""
        self._handlers.pop(type, None)
        self._docs.pop(type, None)

    def dispatch(self, action: Action, ctx: Any) -> ActionResult:
        handler = self._handlers.get(action.type)
        if handler is None:
            return ActionResult.fail(f"unknown action type: {action.type}")
        try:
            return handler(action.payload, ctx)
        except Exception as exc:  # surface handler failures as distinct results (P0.2)
            return ActionResult.fail(repr(exc))

    def schemas(self) -> list[ActionSchema]:
        return [ActionSchema(type=t, description=d, payload_keys=k) for t, (d, k) in self._docs.items()]
