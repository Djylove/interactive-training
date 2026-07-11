"""Python client for the HTTP transport (tests / remote agents) (plan §5)."""
from __future__ import annotations

import json
import urllib.request
from typing import Any


class Client:
    def __init__(self, base_url: str = "http://127.0.0.1:9876"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())

    def submit(self, type: str, **payload) -> dict:
        return self._request("POST", "/actions", {"type": type, "payload": payload})

    def events(self, since: int = 0) -> list[dict]:
        return self._request("GET", f"/events?since={since}")["events"]

    def state(self) -> dict:
        return self._request("GET", "/state")
