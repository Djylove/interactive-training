import torch

from interactive_training.core import Action, TrainingSession
from interactive_training.integrations.autopatch import autopatch, unpatch
from interactive_training import recipes


def _opt(lr=0.1):
    p = torch.nn.Parameter(torch.zeros(3))
    return torch.optim.SGD([p], lr=lr)


def test_optimizer_recipe_knobs():
    cfg = {"grad_clip": 1.0}
    s = TrainingSession()
    recipes.optimizer(s, _opt(0.1), cfg=cfg)
    names = {v.name for v in s.knobs.views()}
    assert {"lr", "weight_decay", "grad_clip"} <= names


def test_gan_recipe_freeze_action():
    g, d = _opt(), _opt()
    s = TrainingSession()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    s.bind_model(model)
    recipes.gan(s, g, d, cfg={"n_critic": 5})
    s.submit(Action(type="freeze", payload={"module_name": "0", "freeze": True}))
    s.step({"loss": 1.0})
    assert all(not p.requires_grad for p in model[0].parameters())


def test_gym_recipe_sets_goal():
    s = TrainingSession()
    recipes.gym(s, {"epsilon": 0.9, "gamma": 0.99, "lr": 0.01, "target_update_freq": 10})
    assert s.goal.metric == "episode_reward" and s.goal.direction == "max"


def test_autopatch_makes_step_a_control_point():
    cfg = {"lr": 0.1}
    s = TrainingSession()
    s.register_knob("lr", lambda: cfg["lr"], lambda v: cfg.__setitem__("lr", v))
    opt = _opt()
    autopatch(s, opt, metrics_fn=lambda: {"loss": 0.0})
    s.submit(Action(type="set_knob", payload={"name": "lr", "value": 0.02}))
    opt.step()
    assert cfg["lr"] == 0.02
    assert len(s.history) == 1
    unpatch(opt)
