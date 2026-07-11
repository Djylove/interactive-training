from __future__ import annotations

import asyncio
import hashlib
import os
import secrets
import signal
import socket
import sys
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from it2_demo.models import LiveSession, SessionStatus


class QueueAtCapacity(RuntimeError):
    pass


class SessionNotFound(KeyError):
    pass


class SessionForbidden(PermissionError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class SessionRecord:
    id: str
    token_hash: str
    client_ip: str
    status: SessionStatus = SessionStatus.queued
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    message: str = "Waiting for the CPU worker"
    port: int | None = None
    process: asyncio.subprocess.Process | None = None
    workdir: Path | None = None
    run_hash: str | None = None


class SessionManager:
    def __init__(self) -> None:
        self.max_queue = int(os.environ.get("IT2_MAX_QUEUE", "5"))
        self.timeout_seconds = int(os.environ.get("IT2_LIVE_TIMEOUT", "360"))
        self.steps = int(os.environ.get("IT2_TRAIN_STEPS", "100"))
        self.session_root = Path(
            os.environ.get("IT2_SESSION_ROOT", "/var/lib/interactive-training/sessions")
        )
        self.aim_repo = os.environ.get(
            "IT2_AIM_REPO", "/var/lib/interactive-training/aim-repo"
        )
        self.python = os.environ.get("IT2_PYTHON", sys.executable)
        self.aim_live_url = os.environ.get("IT2_AIM_LIVE_URL", "/live")
        self.aim_api_url = os.environ.get(
            "IT2_AIM_API_URL", "http://127.0.0.1:39080/api/live/"
        )
        self._sessions: dict[str, SessionRecord] = {}
        self._pending: deque[str] = deque()
        self._active_id: str | None = None
        self._condition = asyncio.Condition()
        self._worker: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        self.session_root.mkdir(parents=True, exist_ok=True)
        if self._worker is None:
            self._worker = asyncio.create_task(self._worker_loop(), name="it2-cpu-worker")

    async def stop(self) -> None:
        self._stopping = True
        async with self._condition:
            self._condition.notify_all()
        if self._active_id:
            record = self._sessions.get(self._active_id)
            if record:
                await self._terminate(record)
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def create(self, client_ip: str) -> tuple[LiveSession, str]:
        async with self._condition:
            active_for_ip = [
                record
                for record in self._sessions.values()
                if record.client_ip == client_ip
                and record.status
                in {SessionStatus.queued, SessionStatus.starting, SessionStatus.running}
            ]
            if active_for_ip:
                raise QueueAtCapacity("This client already has an active or queued session")
            if len(self._pending) >= self.max_queue:
                raise QueueAtCapacity("The public CPU queue is full")

            session_id = secrets.token_urlsafe(12)
            token = secrets.token_urlsafe(32)
            record = SessionRecord(
                id=session_id,
                token_hash=_token_hash(token),
                client_ip=client_ip,
            )
            self._sessions[session_id] = record
            self._pending.append(session_id)
            self._condition.notify()
            return self.public(record), token

    def get(self, session_id: str, token: str | None = None) -> LiveSession:
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFound(session_id)
        if token is not None and not secrets.compare_digest(
            record.token_hash, _token_hash(token)
        ):
            raise SessionForbidden(session_id)
        return self.public(record)

    async def cancel(self, session_id: str, token: str) -> LiveSession:
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionNotFound(session_id)
        if not secrets.compare_digest(record.token_hash, _token_hash(token)):
            raise SessionForbidden(session_id)
        async with self._condition:
            if session_id in self._pending:
                self._pending.remove(session_id)
            if record.process is not None:
                await self._terminate(record)
            record.status = SessionStatus.cancelled
            record.updated_at = _now()
            record.message = "Session cancelled"
            self._condition.notify_all()
        return self.public(record)

    def authorize_run(self, run_hash: str, token: str | None) -> bool:
        if token is None:
            return False
        digest = _token_hash(token)
        return any(
            record.run_hash == run_hash
            and secrets.compare_digest(record.token_hash, digest)
            and record.status in {SessionStatus.starting, SessionStatus.running}
            for record in self._sessions.values()
        )

    def public(self, record: SessionRecord) -> LiveSession:
        position = None
        if record.status == SessionStatus.queued:
            try:
                position = list(self._pending).index(record.id) + 1
            except ValueError:
                position = 1
        return LiveSession(
            id=record.id,
            status=record.status,
            created_at=record.created_at,
            updated_at=record.updated_at,
            queue_position=position,
            message=record.message,
            aim_live_url=self.aim_live_url,
            control_reachable=record.status == SessionStatus.running,
            run_hash=record.run_hash,
        )

    @property
    def active_count(self) -> int:
        return int(self._active_id is not None)

    @property
    def queued_count(self) -> int:
        return len(self._pending)

    async def _worker_loop(self) -> None:
        while not self._stopping:
            async with self._condition:
                await self._condition.wait_for(lambda: self._pending or self._stopping)
                if self._stopping:
                    return
                session_id = self._pending.popleft()
                self._active_id = session_id
                record = self._sessions[session_id]
                record.status = SessionStatus.starting
                record.updated_at = _now()
                record.message = "Starting the isolated CPU trainer"
            try:
                await self._run_session(record)
            finally:
                async with self._condition:
                    self._active_id = None
                    self._condition.notify_all()

    async def _run_session(self, record: SessionRecord) -> None:
        record.workdir = self.session_root / record.id
        record.workdir.mkdir(parents=True, exist_ok=False)
        record.port = self._free_port()
        log_path = record.workdir / "trainer.log"
        log_handle = log_path.open("wb")
        env = self._sanitized_env(record)
        command = [
            self.python,
            "-m",
            "it2_demo.cpu_trainer",
            "--session-id",
            record.id,
            "--session-dir",
            str(record.workdir),
            "--aim-repo",
            self.aim_repo,
            "--port",
            str(record.port),
            "--steps",
            str(self.steps),
        ]
        try:
            record.process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            ready = await self._wait_until_ready(record, timeout=120)
            if not ready:
                raise RuntimeError("trainer did not become ready")
            record.status = SessionStatus.running
            record.updated_at = _now()
            record.message = "Live tiny-BERT CPU training is running; open the Aim workspace"
            record.run_hash = await self._resolve_run_hash(record, timeout=30)
            try:
                code = await asyncio.wait_for(
                    record.process.wait(), timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                record.status = SessionStatus.timed_out
                record.message = "Session reached its public wall-time limit"
                await self._terminate(record)
                return
            if code == 0:
                record.status = SessionStatus.completed
                record.message = "Training completed; the Aim run remains available"
            else:
                record.status = SessionStatus.failed
                record.message = f"Trainer exited with status {code}"
            record.updated_at = _now()
        except Exception as exc:
            record.status = SessionStatus.failed
            record.updated_at = _now()
            record.message = f"Unable to start trainer: {exc}"
            await self._terminate(record)
        finally:
            log_handle.close()

    async def _wait_until_ready(self, record: SessionRecord, timeout: int) -> bool:
        if record.port is None:
            return False
        deadline = asyncio.get_running_loop().time() + timeout
        url = f"http://127.0.0.1:{record.port}/state"
        async with httpx.AsyncClient(timeout=1.5) as client:
            while asyncio.get_running_loop().time() < deadline:
                if record.process and record.process.returncode is not None:
                    return False
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.25)
        return False

    async def _resolve_run_hash(
        self, record: SessionRecord, timeout: int
    ) -> str | None:
        deadline = asyncio.get_running_loop().time() + timeout
        experiment = f"public-cpu-{record.id}"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    response = await client.get(self.aim_api_url)
                    if response.status_code == 200:
                        for session in response.json().get("sessions", []):
                            if session.get("experiment") == experiment:
                                return session.get("run_hash")
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.5)
        return None

    async def _terminate(self, record: SessionRecord) -> None:
        process = record.process
        if process is None or process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except asyncio.TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    def _sanitized_env(self, record: SessionRecord) -> dict[str, str]:
        allowed = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "PATH",
                "HOME",
                "LANG",
                "LC_ALL",
                "SSL_CERT_FILE",
                "REQUESTS_CA_BUNDLE",
                "HF_HOME",
                "TRANSFORMERS_CACHE",
            }
        }
        allowed.update(
            {
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "IT2_PUBLIC_DEMO": "1",
                "IT2_SESSION_ID": record.id,
            }
        )
        return allowed

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
