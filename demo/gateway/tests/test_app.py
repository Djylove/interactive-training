from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from it2_demo.app import create_app
from it2_demo.models import LiveSession, SessionStatus


class FakeManager:
    max_queue = 5
    timeout_seconds = 60
    active_count = 0
    queued_count = 0

    def __init__(self):
        self.session = None
        self.token = "test-token"

    async def start(self):
        return None

    async def stop(self):
        return None

    async def create(self, client_ip):
        self.queued_count = 1
        self.session = LiveSession(
            id="session-1",
            status=SessionStatus.queued,
            created_at=datetime.now(timezone.utc),
            queue_position=1,
            run_hash="run-1",
        )
        return self.session, self.token

    def get(self, session_id, token=None):
        assert session_id == "session-1"
        assert token == self.token
        return self.session

    async def cancel(self, session_id, token):
        assert session_id == "session-1"
        assert token == self.token
        self.session.status = SessionStatus.cancelled
        return self.session

    def authorize_run(self, run_hash, token):
        return run_hash == "run-1" and token == self.token


def test_static_modes_and_assets():
    app = create_app(FakeManager())
    with TestClient(app, base_url="https://testserver") as client:
        modes = client.get("/api/demo/modes")
        assert modes.status_code == 200
        assert {mode["id"] for mode in modes.json()["modes"]} == {
            "muon-video",
            "muon-paper-trace",
            "cpu-live",
        }
        video = client.get("/api/demo/muon/video")
        assert video.status_code == 200
        assert video.json()["fidelity"] == "video_only"
        trace = client.get("/api/demo/muon/paper-trace")
        assert trace.status_code == 200
        assert trace.json()["rounds"] == 11
        rounds = client.get("/api/demo/muon/paper-trace/rounds")
        assert rounds.status_code == 200
        assert len(rounds.json()) == 11


def test_live_queue_cookie_and_cancel():
    app = create_app(FakeManager())
    with TestClient(app, base_url="https://testserver") as client:
        created = client.post("/api/demo/live")
        assert created.status_code == 202
        assert created.json()["session"]["status"] == "queued"
        status = client.get("/api/demo/live/session-1")
        assert status.status_code == 200
        authorized = client.get(
            "/api/demo/authorize-action",
            headers={"X-Original-URI": "/api/live/run-1/actions"},
        )
        assert authorized.status_code == 204
        cancelled = client.delete("/api/demo/live/session-1")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
