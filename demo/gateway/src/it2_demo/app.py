from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Cookie, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse

from it2_demo.manager import (
    QueueAtCapacity,
    SessionForbidden,
    SessionManager,
    SessionNotFound,
)
from it2_demo.models import CreateSessionResponse, ModeInfo, ModesResponse


COOKIE_NAME = "it2_demo_token"


def _assets_dir() -> Path:
    configured = os.environ.get("IT2_DEMO_ASSETS")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[3] / "assets"


def create_app(manager: SessionManager | None = None) -> FastAPI:
    session_manager = manager or SessionManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await session_manager.start()
        yield
        await session_manager.stop()

    app = FastAPI(
        title="Interactive Training 2 Demo Gateway",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.session_manager = session_manager

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    @app.get("/api/demo/modes", response_model=ModesResponse)
    async def modes() -> ModesResponse:
        return ModesResponse(
            modes=[
                ModeInfo(
                    id="muon-video",
                    title="Muon video walkthrough",
                    mode="video_only",
                    available=True,
                    notice="Canonical five-round video; the original event log is unavailable.",
                ),
                ModeInfo(
                    id="muon-paper-trace",
                    title="Explore the Muon paper trace",
                    mode="round_memory",
                    available=True,
                    notice="Eleven committed 3,000-step rounds; distinct from the video.",
                ),
                ModeInfo(
                    id="cpu-live",
                    title="Live tiny-BERT CPU control",
                    mode="live_cpu",
                    available=True,
                    notice="Real reduced training, one worker, no LLM calls.",
                ),
            ],
            active_live_sessions=session_manager.active_count,
            queued_live_sessions=session_manager.queued_count,
            max_queue=session_manager.max_queue,
        )

    @app.get("/api/demo/muon/video")
    async def muon_video() -> FileResponse:
        return FileResponse(_assets_dir() / "muon_video" / "manifest.json")

    @app.get("/api/demo/muon/paper-trace")
    async def muon_trace_manifest() -> FileResponse:
        return FileResponse(_assets_dir() / "muon_paper_trace" / "manifest.json")

    @app.get("/api/demo/muon/paper-trace/rounds")
    async def muon_trace_rounds() -> FileResponse:
        return FileResponse(_assets_dir() / "muon_paper_trace" / "rounds.json")

    @app.post(
        "/api/demo/live",
        response_model=CreateSessionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_live_session(request: Request, response: Response):
        client_ip = request.client.host if request.client else "unknown"
        try:
            session, token = await session_manager.create(client_ip)
        except QueueAtCapacity as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        response.set_cookie(
            COOKIE_NAME,
            token,
            httponly=True,
            secure=os.environ.get("IT2_COOKIE_SECURE", "1") == "1",
            samesite="strict",
            max_age=session_manager.timeout_seconds + 600,
            path="/",
        )
        return CreateSessionResponse(session=session)

    @app.get("/api/demo/live/{session_id}")
    async def live_status(
        session_id: str,
        token: str | None = Cookie(default=None, alias=COOKIE_NAME),
    ):
        try:
            return session_manager.get(session_id, token)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except SessionForbidden as exc:
            raise HTTPException(status_code=403, detail="invalid session token") from exc

    @app.delete("/api/demo/live/{session_id}")
    async def cancel_live_session(
        session_id: str,
        token: str | None = Cookie(default=None, alias=COOKIE_NAME),
    ):
        if token is None:
            raise HTTPException(status_code=403, detail="missing session token")
        try:
            return await session_manager.cancel(session_id, token)
        except SessionNotFound as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except SessionForbidden as exc:
            raise HTTPException(status_code=403, detail="invalid session token") from exc

    @app.get("/api/demo/authorize-action", status_code=status.HTTP_204_NO_CONTENT)
    async def authorize_action(
        original_uri: str = Header(alias="X-Original-URI"),
        token: str | None = Cookie(default=None, alias=COOKIE_NAME),
    ) -> Response:
        match = re.match(r"^/api/live/([^/]+)/actions$", original_uri)
        if not match or not session_manager.authorize_run(match.group(1), token):
            raise HTTPException(status_code=403, detail="action not authorized")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


app = create_app()
