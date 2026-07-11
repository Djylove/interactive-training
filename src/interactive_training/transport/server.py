"""Thin FastAPI HTTP+WS transport — no business logic (plan §5)."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from interactive_training.transport.protocol import decode_action, encode_action, encode_event

logger = logging.getLogger(__name__)


class HttpTransport:
    """Daemon-thread FastAPI app that only moves typed Actions/Events; owns no state.

    ``port=0`` binds a free port; after ``start()`` the resolved address is ``url``."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        self.host = host
        self.port = port
        self._server = None
        self._thread: threading.Thread | None = None
        self._ready = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_app(self, session: Any):
        app = FastAPI()

        @app.post("/actions")
        def post_action(body: dict):
            action = decode_action(body)
            session.submit(action)
            return {"id": action.id}

        @app.get("/events")
        def get_events(since: int = 0):
            return {"events": [encode_event(e) for e in session.events.replay(since)]}

        @app.get("/state")
        def get_state():
            return {
                "status": session.state.status,
                "goal": session.goal.model_dump() if session.goal else None,
                "knobs": [v.model_dump() for v in session.knobs.views()],
                "actions": [s.model_dump() for s in session.registry.schemas()],
                "agent": session.agent_snapshot(),
                "context": session.context,
                **session.round_snapshot(),
                **session.state.snapshot(),
            }

        @app.websocket("/events")
        async def ws_events(ws: WebSocket):
            import asyncio
            await ws.accept()
            since = int(ws.query_params.get("since", 0))
            q = session.events.subscribe(since=since)
            try:
                while True:
                    ev = await asyncio.get_event_loop().run_in_executor(None, q.get)
                    await ws.send_json(encode_event(ev))
            except WebSocketDisconnect:
                pass
            finally:
                session.events.unsubscribe(q)

        return app

    def start(self, session: Any) -> None:
        import uvicorn

        app = self.build_app(session)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started and self._thread.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        if self._server.started and self.port == 0:
            self.port = self._server.servers[0].sockets[0].getsockname()[1]
        if self._server.started:
            from interactive_training.transport.composite import wait_http_ready
            self._ready = wait_http_ready(f"{self.url}/state", timeout=10)
            if not self._ready:
                logger.warning("HttpTransport: control endpoint not responding at %s", self.url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
