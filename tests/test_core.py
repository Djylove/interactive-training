import threading
import time

import torch

from interactive_training.core import Action, TrainingSession, ValidationLoss, AverageReward
from interactive_training.core.events import EventBus


def test_eventbus_replay_and_per_subscriber():
    bus = EventBus()
    a = bus.subscribe()
    bus.publish("metrics", {"loss": 1.0})
    bus.publish("metrics", {"loss": 0.9})
    b = bus.subscribe(since=0)
    assert a.qsize() == 2
    assert b.qsize() == 2  # late joiner replays history
    assert [e.seq for e in bus.replay(1)] == [1]


def test_knob_set_applies_before_advance():
    cfg = {"lr": 0.1}
    s = TrainingSession()
    s.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v), min=0.0, max=1.0)
    s.submit(Action(type="set_knob", payload={"name": "lr", "value": 0.5}))
    ctrl = s.step({"loss": 1.0})
    assert cfg["lr"] == 0.5  # applied by the time step returned
    assert ctrl.knob_updates == {"lr": 0.5}


def test_knob_clamped_to_bounds():
    cfg = {"lr": 0.1}
    s = TrainingSession()
    s.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v), min=0.0, max=1.0)
    s.submit(Action(type="set_knob", payload={"name": "lr", "value": 9.0}))
    s.step({"loss": 1.0})
    assert cfg["lr"] == 1.0


def test_blocking_submit_acks():
    s = TrainingSession()
    s.register_knob("x", lambda: 0, lambda v: None)
    results = []

    def worker():
        results.append(s.submit(Action(type="set_knob", payload={"name": "x", "value": 1}), wait=True, timeout=2))

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.05)
    s.step({"loss": 1.0})
    t.join(timeout=2)
    assert results and results[0].ok


def test_pause_blocks_then_stop_while_paused():
    s = TrainingSession()
    s.submit(Action(type="pause"))
    order = []

    def loop():
        order.append("before")
        s.step({"loss": 1.0})  # should block while paused
        order.append("after")

    t = threading.Thread(target=loop)
    t.start()
    time.sleep(0.1)
    assert order == ["before"]  # still blocked
    s.submit(Action(type="stop"))  # stop while paused must release (P0.4)
    t.join(timeout=2)
    assert order == ["before", "after"]
    assert s.control.stopped


def test_checkpoint_branch_and_load():
    s = TrainingSession()
    ckpt = s.checkpoint_saved("/tmp/ckpt-1", step=10)
    s.submit(Action(type="load_checkpoint", payload={"checkpoint_id": ckpt.id, "fork": True}))
    ctrl = s.step({"loss": 1.0})
    assert ctrl.load == "/tmp/ckpt-1" and ctrl.reload_required
    assert s.state.branch_id != "main"  # fork never returns None (P0.7)


def test_reset_module_actually_runs():
    import torch.nn as nn

    s = TrainingSession()
    model = nn.Sequential(nn.Linear(4, 4))
    s.bind_model(model)
    before = model[0].weight.clone()
    s.submit(Action(type="reset_module", payload={"module_name": "0"}))
    ctrl = s.step({"loss": 1.0})
    assert "0" in ctrl.reset_modules
    assert not torch.equal(before, model[0].weight)  # reset actually changed params


def test_custom_action_registration():
    s = TrainingSession()
    seen = {}

    @s.action("switch_dataset", "swap loader", ["name"])
    def _(payload, ctx):
        from interactive_training.core.actions import ActionResult
        seen["name"] = payload["name"]
        return ActionResult.success()

    s.submit(Action(type="switch_dataset", payload={"name": "v2"}))
    s.step({"loss": 1.0})
    assert seen["name"] == "v2"


def test_action_unregister_removes_dispatch_and_schema():
    s = TrainingSession()
    assert any(schema.type == "reset_module" for schema in s.registry.schemas())
    s.registry.unregister("reset_module")
    assert all(schema.type != "reset_module" for schema in s.registry.schemas())
    s.submit(Action(type="reset_module", payload={"module_name": "0"}))
    s.step({"loss": 1.0})
    assert s._pending.applied[-1] == {"type": "reset_module", "ok": False}


def test_goal_scoring():
    g = ValidationLoss()
    hist = [{"val_loss": 1.0}, {"val_loss": 0.5}, {"val_loss": 0.8}]
    assert g.score(hist) == 0.5
    assert AverageReward().score([{"reward": 1}, {"reward": 3}]) == 3


def test_session_coerces_goal_memory_agent(tmp_path):
    class A:
        name = "a"
        every = 1
        def act(self, obs):
            return []

    a = A()
    s = TrainingSession(goal="eval_loss", memory=str(tmp_path / "m.jsonl"), agent=a, max_rounds=7)
    assert s.goal.metric == "eval_loss" and s.goal.direction == "min"
    assert TrainingSession(goal="reward").goal.direction == "max"
    assert s.memory.path is not None and s.max_rounds == 7
    assert a in s._agents  # agent auto-attached


def test_failed_action_distinct_from_success():
    s = TrainingSession()
    s.submit(Action(type="set_knob", payload={"name": "missing", "value": 1}))
    s.step({"loss": 1.0})
    assert s._pending.applied[-1] == {"type": "set_knob", "ok": False}
