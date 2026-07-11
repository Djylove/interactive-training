from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    queued = "queued"
    starting = "starting"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    timed_out = "timed_out"


class LiveSession(BaseModel):
    id: str
    status: SessionStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    queue_position: int | None = None
    message: str = ""
    aim_live_url: str = "/live"
    control_reachable: bool = False
    run_hash: str | None = None


class ModeInfo(BaseModel):
    id: str
    title: str
    mode: str
    available: bool
    notice: str


class ModesResponse(BaseModel):
    modes: list[ModeInfo]
    active_live_sessions: int
    queued_live_sessions: int
    max_queue: int


class CreateSessionResponse(BaseModel):
    session: LiveSession
