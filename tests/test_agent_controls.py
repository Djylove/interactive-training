"""Runtime agent-control actions the frontend drives: configure_agent / set_agent /
set_context (design doc §3.8) and the checkpoint tag path (§6.4 Checkpoints panel)."""
from interactive_training.core import Action, TrainingSession


def _drain_events(session, since=0):
    return session.events.replay(since)


def test_configure_agent_attaches_and_redacts_key():
    s = TrainingSession()  # launched agent-free; configure_agent attaches a default
    s.submit(Action(type="configure_agent",
                    payload={"provider": "openrouter", "model": "gpt-5.5",
                             "api_key": "sk-secret-123", "every": 25}))
    s.step({"loss": 1.0})

    snap = s.agent_snapshot()
    assert snap["attached"] is True
    assert snap["model"] == "gpt-5.5"
    assert snap["provider"] == "openrouter"
    assert snap["every"] == 25
    assert snap["api_key_set"] is True
    assert "api_key" not in snap  # key is write-only (§3.8)

    # The key must not leak into any event payload or recorded round action.
    for ev in _drain_events(s):
        assert "sk-secret-123" not in str(ev.payload)
    for a in s._round_actions:
        assert a["payload"].get("api_key", "") != "sk-secret-123"

    types = [ev.type for ev in _drain_events(s)]
    assert "agent_attached" in types and "agent_configured" in types


def test_set_agent_toggles_active_flag():
    s = TrainingSession(agent_active=False)
    assert s.agent_snapshot()["active"] is False
    s.submit(Action(type="set_agent", payload={"enabled": True}))
    s.step({"loss": 1.0})
    assert s.agent_snapshot()["active"] is True
    assert any(ev.type == "agent_enabled" and ev.payload["enabled"]
               for ev in _drain_events(s))


def test_set_context_replaces_context():
    s = TrainingSession(context="old context")
    s.submit(Action(type="set_context", payload={"context": "new task description"}))
    s.step({"loss": 1.0})
    assert s.context == "new task description"
    assert any(ev.type == "context_changed" and
               ev.payload["context"] == "new task description"
               for ev in _drain_events(s))


class _FakeClient:
    """LLMClient stub: fixed response + one tool call + usage."""

    def complete(self, system, user, tools=None):
        return ("looks healthy, nudging lr",
                [("set_knob", {"name": "lr", "value": 0.5})],
                {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


def test_agent_call_event_surfaces_prompt_and_response():
    from interactive_training.agents.agent import LLMAgent

    s = TrainingSession(goal="loss", agent=LLMAgent(client=_FakeClient(), every=1))
    cfg = {"lr": 0.1}
    s.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v),
                    min=0.0, max=1.0)
    s.step({"loss": 1.0}, step=1)  # control point -> agent acts -> hook publishes

    calls = [ev for ev in _drain_events(s) if ev.type == "agent_call"]
    assert calls, "no agent_call event published"
    p = calls[0].payload
    assert p["response"] == "looks healthy, nudging lr"
    assert p["system"] and p["user"]
    assert p["tool_calls"] == [{"name": "set_knob",
                                "arguments": {"name": "lr", "value": 0.5}}]
    assert p["usage"]["total_tokens"] == 15


def test_save_checkpoint_carries_tag_through_control():
    s = TrainingSession()
    s.submit(Action(type="save_checkpoint", payload={"tag": "before-lr-drop"}))
    ctrl = s.step({"loss": 1.0})
    assert ctrl.save is True and ctrl.tag == "before-lr-drop"
    ckpt = s.checkpoint_saved("/tmp/ckpt-tagged", step=5, tag=ctrl.tag)
    assert ckpt.tag == "before-lr-drop"
    listed = s.state.checkpoints.list()
    assert listed and listed[-1].tag == "before-lr-drop"
