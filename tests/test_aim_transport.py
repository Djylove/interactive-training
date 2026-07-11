import sys
import time
import types

from interactive_training.core import TrainingSession
from interactive_training.transport.aim_transport import AimTransport
from interactive_training.transport.client import Client
from interactive_training.transport.composite import CompositeTransport, aim_frontend


class FakeRun:
    def __init__(self):
        self.tracked = []
        self.params = {}
        self.closed = False
        self.hash = "fake"

    def track(self, value, name=None, step=None, context=None):
        self.tracked.append({"value": value, "name": name, "step": step, "context": context or {}})

    def __setitem__(self, key, value):
        self.params[key] = value

    def close(self):
        self.closed = True


def _stub_aim(monkeypatch):
    aim = types.ModuleType("aim")
    aim.Text = lambda s: s
    monkeypatch.setitem(sys.modules, "aim", aim)


def _make_transport(monkeypatch, control_url=None):
    _stub_aim(monkeypatch)
    transport = AimTransport(control_url=control_url)
    runs = []
    monkeypatch.setattr(AimTransport, "_create_run", lambda self: runs.append(FakeRun()) or runs[-1])
    return transport, runs


def _drain(transport):
    deadline = time.time() + 5
    while not transport._queue.empty() and time.time() < deadline:
        time.sleep(0.02)


def test_aim_transport_maps_events(monkeypatch):
    transport, runs = _make_transport(monkeypatch, control_url=lambda: "http://127.0.0.1:1234")
    cfg = {"lr": 0.1}
    session = TrainingSession(goal="loss", transport=transport)
    session.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v))
    session.start()
    try:
        session.begin_round(0)
        session.step({"loss": 1.5}, step=10)
        session.report_eval({"eval_loss": 2.0})
        session.end_round(score=1.5)
        _drain(transport)
    finally:
        session.end()

    run = runs[0]
    assert run.params["control"] == {"url": "http://127.0.0.1:1234"}
    assert run.params["goal"]["metric"] == "loss"

    loss = [t for t in run.tracked if t["name"] == "loss"]
    assert loss and loss[0]["value"] == 1.5 and loss[0]["step"] == 10
    assert loss[0]["context"]["round"] == 0

    evals = [t for t in run.tracked if t["name"] == "eval_loss"]
    assert evals and evals[0]["context"].get("subset") == "eval"

    scores = [t for t in run.tracked if t["name"] == "round_score"]
    assert scores and scores[0]["value"] == 1.5

    texts = [t for t in run.tracked if t["name"] == "agent_actions"]
    assert any('"status_changed"' in t["value"] for t in texts)
    assert run.closed


def test_aim_transport_run_per_round(monkeypatch):
    transport, runs = _make_transport(monkeypatch)
    session = TrainingSession(transport=transport)
    session.start()
    try:
        session.begin_round(0)
        session.step({"loss": 1.0}, step=1)
        session.end_round(score=1.0)
        session.begin_round(1)
        session.step({"loss": 0.5}, step=1)
        session.end_round(score=0.5)
        _drain(transport)
    finally:
        session.end()

    assert len(runs) == 2
    assert runs[0].closed
    assert [t["value"] for t in runs[1].tracked if t["name"] == "loss"] == [0.5]


def test_session_frontend_flag(monkeypatch):
    _stub_aim(monkeypatch)
    runs = []
    monkeypatch.setattr(AimTransport, "_create_run", lambda self: runs.append(FakeRun()) or runs[-1])

    assert TrainingSession().transport is None

    session = TrainingSession(frontend=True)
    assert isinstance(session.transport, CompositeTransport)
    session.start()
    try:
        http = session.transport.transports[0]
        assert http.port != 0
        session.step({"loss": 1.0}, step=1)
        _drain(session.transport.transports[1])
        assert runs and runs[0].params["control"] == {"url": http.url}
    finally:
        session.end()


def test_composite_with_http_control_url(monkeypatch):
    _stub_aim(monkeypatch)
    runs = []
    monkeypatch.setattr(AimTransport, "_create_run", lambda self: runs.append(FakeRun()) or runs[-1])

    transport = aim_frontend()
    cfg = {"lr": 0.1}
    session = TrainingSession(transport=transport)
    session.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v), min=0.0, max=1.0)
    session.start()
    try:
        http = transport.transports[0]
        assert http.port != 0
        client = Client(http.url)
        for _ in range(50):
            try:
                client.state()
                break
            except Exception:
                time.sleep(0.1)

        client.submit("set_knob", name="lr", value=0.42)
        session.step({"loss": 1.0}, step=1)
        assert cfg["lr"] == 0.42

        _drain(transport.transports[1])
        assert runs[0].params["control"] == {"url": http.url}
    finally:
        session.end()
