import socket
import time

from interactive_training.core import TrainingSession
from interactive_training.transport.client import Client
from interactive_training.transport.server import HttpTransport


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_http_roundtrip():
    cfg = {"lr": 0.1}
    port = _free_port()
    session = TrainingSession(transport=HttpTransport(port=port))
    session.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v), min=0.0, max=1.0)
    session.start()
    client = Client(f"http://127.0.0.1:{port}")
    try:
        for _ in range(50):
            try:
                client.state()
                break
            except Exception:
                time.sleep(0.1)

        client.submit("set_knob", name="lr", value=0.42)
        session.step({"loss": 1.0})
        assert cfg["lr"] == 0.42

        state = client.state()
        assert state["status"] == "running"
        assert any(k["name"] == "lr" for k in state["knobs"])

        events = client.events(since=0)
        assert any(e["type"] == "metrics" for e in events)
    finally:
        session.end()
