"""Fan out the transport contract to several components (frontend plan §4)."""
from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def wait_http_ready(url: str, timeout: float = 60.0, poll: float = 0.1) -> bool:
    """Poll until *url* returns HTTP 200, or give up after *timeout* seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(poll)
    return False


class CompositeTransport:
    def __init__(self, transports: list[Any]):
        self.transports = list(transports)

    def start(self, session: Any) -> None:
        for t in self.transports:
            t.start(session)

    def stop(self) -> None:
        for t in reversed(self.transports):
            try:
                t.stop()
            except Exception:
                logger.exception("CompositeTransport: %r failed to stop", t)


def aim_frontend(repo: str | None = None, experiment: str | None = None,
                 host: str = "127.0.0.1", up: bool = False, ui_host: str = "0.0.0.0",
                 ui_port: int = 43800) -> CompositeTransport:
    from interactive_training.transport.aim_transport import AimTransport, AimUp
    from interactive_training.transport.server import HttpTransport

    http = HttpTransport(host=host, port=0)
    transports: list[Any] = [http]
    if up:
        transports.append(AimUp(repo=repo, host=ui_host, port=ui_port))
    transports.append(AimTransport(repo=repo, experiment=experiment, control_url=lambda: http.url))
    return CompositeTransport(transports)
