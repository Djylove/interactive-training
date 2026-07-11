"""E2E smoke of the frontend control chain (run manually inside the container):

    python tests/e2e_live_smoke.py

Starts a real TrainingSession with the Aim frontend (control HTTP + Aim repo + `aim up`),
then drives it through the /api/live proxy exactly like the new UI panels do:
configure_agent (api_key), set_context, set_agent, save_checkpoint — verifying state,
events, and key redaction end to end.
"""
import json
import tempfile
import threading
import time
import urllib.request

from interactive_training.agents.agent import LLMAgent, OpenAIClient
from interactive_training.core import TrainingSession

SECRET = "sk-e2e-secret-42"


class _FakeClient(OpenAIClient):
    """OpenAIClient with a canned completion — full configure/describe surface,
    no network. Lets the smoke exercise the real agent_call event path."""

    def complete(self, system, user, tools=None):
        return ("run looks healthy, small lr nudge",
                [("set_knob", {"name": "lr", "value": 0.05})],
                {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def http(method, url, body=None):
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def main():
    import torch.nn as nn

    repo = tempfile.mkdtemp(prefix="live_smoke_aim_")
    ui_port = 43977
    session = TrainingSession(
        goal="loss", context="e2e smoke context",
        agent=LLMAgent(client=_FakeClient(), every=10, name="smoke-agent"),
        frontend={"repo": repo, "experiment": "live_smoke", "up": True,
                  "ui_port": ui_port})
    cfg = {"lr": 0.1}
    session.register_knob("lr", lambda: cfg["lr"],
                          lambda v: cfg.__setitem__("lr", v), min=0.0, max=1.0)
    session.bind_model(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)))
    session.start()

    # background loop so control actions get drained at control points
    stop = threading.Event()

    def loop():
        step = 0
        while not stop.is_set():
            step += 1
            session.step({"loss": 1.0 / step, "step": step}, step=step)
            time.sleep(0.05)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{ui_port}/api/live"
    sessions = []
    for _ in range(60):
        try:
            sessions = http("GET", f"{base}/").get("sessions", [])
            if sessions and sessions[0]["reachable"]:
                break
        except Exception:
            pass
        time.sleep(1)
    assert sessions and sessions[0]["reachable"], f"no live session discovered: {sessions}"
    h = sessions[0]["run_hash"]
    print(f"[ok] discovery: run {h} reachable, status={sessions[0]['status']}")

    http("POST", f"{base}/{h}/actions", {
        "type": "configure_agent", "source": "human:web",
        "payload": {"provider": "openrouter", "model": "gpt-5.5",
                    "api_key": SECRET, "every": 25}})
    http("POST", f"{base}/{h}/actions", {
        "type": "set_context", "source": "human:web",
        "payload": {"context": "updated from the frontend"}})
    http("POST", f"{base}/{h}/actions", {
        "type": "set_agent", "source": "human:web", "payload": {"enabled": True}})
    http("POST", f"{base}/{h}/actions", {
        "type": "save_checkpoint", "source": "human:web",
        "payload": {"tag": "e2e-tag"}})
    time.sleep(1)

    state = http("GET", f"{base}/{h}/state")
    agent = state["agent"]
    assert agent["attached"] and agent["active"], agent
    assert agent["model"] == "gpt-5.5" and agent["provider"] == "openrouter", agent
    assert agent["every"] == 25 and agent["api_key_set"] is True, agent
    assert state["context"] == "updated from the frontend", state["context"]
    assert SECRET not in json.dumps(state), "API key leaked into /state!"
    print("[ok] /state: agent configured+active, context updated, key redacted")

    session.checkpoint_saved("/tmp/e2e-ckpt", step=session.state.step, tag="e2e-tag")
    state = http("GET", f"{base}/{h}/state")
    assert any(c["tag"] == "e2e-tag" for c in state["checkpoints"]), state["checkpoints"]
    print(f"[ok] /state: {len(state['checkpoints'])} checkpoint(s) listed, tag preserved")

    tree = state["model_tree"]
    assert tree and tree["children"], f"model_tree missing/empty: {tree}"
    print(f"[ok] /state: model_tree present ({tree['module_type']}, "
          f"{len(tree['children'])} children)")

    events = http("GET", f"{base}/{h}/events?since=0")["events"]
    types = {e["type"] for e in events}
    for expected in ("agent_configured", "agent_enabled", "context_changed",
                     "checkpoint_saved", "action_result", "model_tree", "agent_call"):
        assert expected in types, f"missing event {expected}; got {sorted(types)}"
    assert SECRET not in json.dumps(events), "API key leaked into events!"
    call = next(e for e in events if e["type"] == "agent_call")["payload"]
    assert call["system"] and call["user"] and call["response"], call
    assert call["tool_calls"][0]["name"] == "set_knob", call["tool_calls"]
    print(f"[ok] /events: {sorted(types & {'agent_configured', 'agent_enabled', 'context_changed', 'checkpoint_saved', 'model_tree', 'agent_call'})} present, key redacted")
    print("[ok] agent_call carries full prompt (system+user), response, tool calls")

    stop.set()
    t.join(timeout=5)
    session.end()
    print("[ok] E2E smoke passed")


if __name__ == "__main__":
    main()
