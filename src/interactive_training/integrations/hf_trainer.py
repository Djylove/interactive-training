"""Tier 1: make_interactive(...) for HF Trainer/TRL — full old-feature parity (plan §7.2)."""
from __future__ import annotations

from typing import Any

from interactive_training.core.session import TrainingSession


def make_interactive(base_cls: type, extra_optimizer_knobs: bool = True):
    from transformers import TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint

    class InteractiveCallback(TrainerCallback):
        def __init__(self, session: TrainingSession):
            self.session = session
            self.pending_load: str | None = None
            self.pending_tag: str | None = None  # tag of a requested save, until on_save fires

        def on_train_begin(self, args, state, control, **kwargs):
            model = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("lr_scheduler")
            if model is not None:
                self.session.bind_model(model)
            if optimizer is not None and "lr" not in self.session.knobs:
                self.session.register_optimizer_lr(optimizer, scheduler)
                if extra_optimizer_knobs:
                    self._register_tunable_knobs(optimizer, args)

        def _register_tunable_knobs(self, optimizer, args):
            """Expose more tunable hyperparameters beyond lr, via the same optimizer
            param-group mechanism the old version used (weight decay, betas, eps),
            plus HF's gradient-clipping max-norm."""
            s = self.session
            if "weight_decay" not in s.knobs:
                s.register_knob(
                    "weight_decay",
                    get=lambda: float(optimizer.param_groups[0].get("weight_decay", 0.0)),
                    set=lambda v: [g.__setitem__("weight_decay", float(v)) for g in optimizer.param_groups],
                    min=0.0, description="optimizer weight decay")
            if "adam_beta1" not in s.knobs and "betas" in optimizer.param_groups[0]:
                s.register_knob(
                    "adam_beta1",
                    get=lambda: float(optimizer.param_groups[0]["betas"][0]),
                    set=lambda v: [g.__setitem__("betas", (float(v), g["betas"][1])) for g in optimizer.param_groups],
                    min=0.0, max=1.0, description="Adam beta1 (momentum)")
            if "eps" not in s.knobs and "eps" in optimizer.param_groups[0]:
                s.register_knob(
                    "eps",
                    get=lambda: float(optimizer.param_groups[0]["eps"]),
                    set=lambda v: [g.__setitem__("eps", float(v)) for g in optimizer.param_groups],
                    min=0.0, description="optimizer epsilon")
            if args is not None and "max_grad_norm" not in s.knobs:
                s.register_knob(
                    "max_grad_norm",
                    get=lambda: float(getattr(args, "max_grad_norm", 0.0) or 0.0),
                    set=lambda v: setattr(args, "max_grad_norm", float(v)),
                    min=0.0, description="gradient clipping max-norm")

        def on_log(self, args, state, control, **kwargs):
            # Record metrics (incl. eval metrics, which HF logs here) without acting; the
            # agent acts in on_step_end / on_evaluate so it sees the freshest data.
            self._apply(self.session.step(dict(kwargs.get("logs", {})),
                                          step=state.global_step, act=False), control)

        def on_step_end(self, args, state, control, **kwargs):
            # If an eval is scheduled for this step (DefaultFlowCallback set this before us),
            # defer the agent's action to on_evaluate so it can see the new eval result.
            act = not control.should_evaluate
            self._apply(self.session.pump(step=state.global_step, act=act), control)

        def on_evaluate(self, args, state, control, **kwargs):
            self._apply(self.session.pump(step=state.global_step, act=True), control)

        def on_save(self, args, state, control, **kwargs):
            path = get_last_checkpoint(args.output_dir) or args.output_dir
            tag, self.pending_tag = self.pending_tag, None
            self.session.checkpoint_saved(path, state.global_step, tag=tag)

        def _apply(self, ctrl, control):
            if ctrl.stop:
                control.should_training_stop = True
            if ctrl.save:
                control.should_save = True
                self.pending_tag = ctrl.tag
            if ctrl.evaluate:
                control.should_evaluate = True
            if ctrl.reload_required and ctrl.load:
                control.should_training_stop = True
                self.pending_load = ctrl.load

    class Interactive(base_cls):
        def __init__(self, *args, session: TrainingSession | None = None, transport: Any | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            if session is None:
                session = TrainingSession(transport=transport)
            self.session = session
            self._interactive_cb = InteractiveCallback(session)
            self.add_callback(self._interactive_cb)

        def train(self, **kwargs):
            owns_session = not self.session.started  # a multi-round run_rounds owns lifecycle
            self.session.start()
            while True:
                super().train(**kwargs)
                load = self._interactive_cb.pending_load
                if load:
                    kwargs["resume_from_checkpoint"] = load
                    self._interactive_cb.pending_load = None
                    continue
                break
            if owns_session:
                self.session.end()

    Interactive.__name__ = f"Interactive{base_cls.__name__}"
    return Interactive