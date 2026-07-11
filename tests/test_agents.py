from interactive_training.agents.agent import LLMAgent, Observation, Plan
from interactive_training.core import Action, AverageReward, Memory, TrainingSession, ValidationLoss


class ScriptedAgent:
    """A non-LLM agent for testing the act/plan/reflect protocol offline."""
    name = "scripted"
    every = 1

    def __init__(self):
        self.fired = False
        self.planned_rounds = []
        self.reflected_rounds = []

    def act(self, obs: Observation):
        if self.fired:
            return []
        self.fired = True
        return [Action(type="set_knob", payload={"name": "lr", "value": 0.01})]

    def plan(self, ctx):
        self.planned_rounds.append(ctx.round)
        return Plan(config={"lr": 0.1}, strategy="drop lr if loss spikes")

    def reflect(self, ctx, trajectory, score):
        self.reflected_rounds.append(ctx.round)
        return f"score={score}"


def test_attached_agent_computes_actions():
    cfg = {"lr": 0.1}
    s = TrainingSession()
    s.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v))
    s.attach(ScriptedAgent())
    s.step({"loss": 1.0})
    assert cfg["lr"] == 0.01


def test_session_manages_rounds_with_agent(tmp_path):
    goal = ValidationLoss(target=0.0)
    memory = Memory(path=str(tmp_path / "m.jsonl"))
    session = TrainingSession(goal=goal, memory=memory)
    agent = ScriptedAgent()
    session.attach(agent)
    scores = iter([0.5, 0.4, 0.3])
    initial_lrs = []

    def train_round(session, ctx):
        cfg = {"lr": 0.2}
        session.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v))
        session.plan_round(ctx)
        initial_lrs.append(cfg["lr"])
        session.step({"val_loss": next(scores)})

    # round 0 is always a no-agent baseline even if a caller tries baseline_rounds=0.
    session.run_rounds(train_round, max_rounds=2, baseline_rounds=0)
    assert agent.planned_rounds == [1, 2]
    assert agent.reflected_rounds == [1, 2]
    assert initial_lrs == [0.2, 0.1, 0.1]  # baseline keeps defaults; agent rounds get the plan
    assert len(memory.rounds) == 3
    assert memory.summarize().startswith("Round 0 (baseline)")
    assert memory.rounds[0]["baseline"] is True
    assert memory.rounds[1]["baseline"] is False
    assert memory.rounds[0]["reflection"] == ""


def test_multiround_persistent_session_with_human(tmp_path):
    """One persistent session + live transport across rounds; a human steers via HTTP
    (no LLM agent attached). Validates the human/frontend + multi-round path."""
    import socket
    import time

    from interactive_training.transport.client import Client
    from interactive_training.transport.server import HttpTransport

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    goal = ValidationLoss(target=0.0)
    memory = Memory(path=str(tmp_path / "m.jsonl"))
    session = TrainingSession(goal=goal, memory=memory, transport=HttpTransport(port=port))
    client = Client(f"http://127.0.0.1:{port}")
    cfgs = []

    def train_round(session, ctx):
        cfg = {"lr": 0.1}
        cfgs.append(cfg)
        session.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v), min=0.0, max=1.0)
        for _ in range(50):  # wait for the (persistent) server to be reachable
            try:
                client.state()
                break
            except Exception:
                time.sleep(0.1)
        client.submit("set_knob", name="lr", value=0.01 * (ctx.round + 1))
        session.step({"val_loss": 0.5 / (ctx.round + 1)})

    session.run_rounds(train_round, max_rounds=2)  # no agent attached -> human-only
    assert cfgs[0]["lr"] == 0.01 and cfgs[1]["lr"] == 0.02  # human action applied each round
    assert len(memory.rounds) == 2


def test_agent_toggle_via_set_agent():
    cfg = {"lr": 0.1}
    s = TrainingSession(agent_active=False)
    s.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v))
    s.attach(ScriptedAgent())
    s.step({"loss": 1.0}, step=1)
    assert cfg["lr"] == 0.1

    s.submit(Action(type="set_agent", payload={"enabled": True}))
    s.step({"loss": 0.9}, step=2)  # toggle applies at this barrier (after the act window)
    s.step({"loss": 0.8}, step=3)  # agent acts from the next control point
    assert cfg["lr"] == 0.01
    assert any(e.type == "agent_enabled" and e.payload["enabled"] for e in s.events.replay(0))

    s.submit(Action(type="set_agent", payload={"enabled": False}))
    s.step({"loss": 0.7}, step=4)
    assert s._agent_active is False


def test_dormant_agent_skips_plan_and_reflect(tmp_path):
    memory = Memory(path=str(tmp_path / "m.jsonl"))
    session = TrainingSession(goal=ValidationLoss(target=0.0), memory=memory, agent_active=False)
    agent = ScriptedAgent()
    session.attach(agent)

    def train_round(session, ctx):
        cfg = {"lr": 0.2}
        session.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v))
        session.plan_round(ctx)
        session.step({"val_loss": 0.5})

    session.run_rounds(train_round, max_rounds=2)
    assert agent.planned_rounds == [] and agent.reflected_rounds == []
    assert len(memory.rounds) == 3  # rounds still run; they're just human/manual rounds


def test_tool_schema_generation():
    s = TrainingSession()
    s.register_knob("lr", lambda: 0.1, lambda v: None, min=0.0, max=1.0, description="lr")
    obs = s._observation()
    from interactive_training.agents.agent import _tools_from_actions
    tools = _tools_from_actions(obs.available_actions, obs.knobs)
    set_knob = next(t for t in tools if t["function"]["name"] == "set_knob")
    assert "lr" in set_knob["function"]["parameters"]["properties"]["name"]["enum"]
