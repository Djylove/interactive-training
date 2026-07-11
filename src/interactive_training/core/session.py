from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from interactive_training.core.actions import Action, ActionRegistry, ActionResult
from interactive_training.core.control import ControlGate, StepControl, broadcast_and_barrier
from interactive_training.core.determinism import DEFAULT_SEED, seed_everything
from interactive_training.core.events import EventBus
from interactive_training.core.goals import Goal
from interactive_training.core.knobs import Knob, KnobRegistry, optimizer_lr_accessors
from interactive_training.core.memory import Memory
from interactive_training.core.state import TrainingState, build_model_tree, flatten_model_tree
from interactive_training.transport.bus import ActionBus

logger = logging.getLogger(__name__)

_MEMORY_LIMIT = 10
_DDP_HANDLED_ACTIONS = frozenset({
    "set_knob", "pause", "resume", "stop", "load_checkpoint", "save_checkpoint",
    "evaluate", "note",
})
_SENSITIVE_PAYLOAD_KEYS = frozenset({"api_key"})


def _redact_payload(payload: dict) -> dict:
    return {k: ("<redacted>" if k in _SENSITIVE_PAYLOAD_KEYS else v) for k, v in payload.items()}


def _coerce_goal(goal: Goal | str | None) -> Goal | None:
    """Accept a Goal, or a metric-name string (direction inferred: reward/accuracy -> max)."""
    if goal is None or isinstance(goal, Goal):
        return goal
    direction = "max" if any(k in goal.lower() for k in ("reward", "acc")) else "min"
    return Goal(name=goal, metric=goal, direction=direction)


def _coerce_memory(memory: Any) -> Memory | None:
    if memory is None or isinstance(memory, Memory):
        return memory
    return Memory(path=memory)


def _build_frontend(frontend: bool | str | dict | None) -> Any | None:
    """`frontend=True` -> Aim frontend with defaults; a str -> the Aim repo path;
    a dict -> kwargs for `aim_frontend`. Falsy -> no transport."""
    if not frontend:
        return None
    from interactive_training.transport.composite import aim_frontend
    if frontend is True:
        return aim_frontend()
    if isinstance(frontend, str):
        return aim_frontend(repo=frontend)
    return aim_frontend(**frontend)


def _log_frontend(transport: Any, agent: dict | None = None) -> None:
    """Print control / Aim UI endpoints once the transport is listening."""
    from interactive_training.transport.aim_transport import AimTransport, AimUp
    from interactive_training.transport.composite import CompositeTransport
    from interactive_training.transport.server import HttpTransport

    parts = transport.transports if isinstance(transport, CompositeTransport) else [transport]
    http = next((t for t in parts if isinstance(t, HttpTransport)), None)
    aim = next((t for t in parts if isinstance(t, AimTransport)), None)
    ui = next((t for t in parts if isinstance(t, AimUp)), None)
    if http is not None:
        ready = "" if getattr(http, "_ready", False) else "  (not responding)"
        print(f"[frontend] control endpoint: {http.url}{ready}")
    if ui is not None:
        if getattr(ui, "_ready", False):
            print(f"[frontend] aim web UI: {ui.url}  (repo: {ui.repo or '<default>'})")
        else:
            print(f"[frontend] aim web UI: not responding yet at {ui.url}  "
                  f"(repo: {ui.repo or '<default>'})")
    elif aim is not None and aim.repo:
        print(f"[frontend] aim repo: {aim.repo}  (browse: aim up --repo {aim.repo})")
    if agent is not None:
        label = agent.get("model") if agent.get("attached") else None
        print(f"[frontend] agent: {label or ('(attached)' if agent.get('attached') else '(none)')}")


@dataclass
class RoundContext:
    """Per-round context passed to `train_round` and to agent plan/reflect (plan §3.6)."""
    round: int
    memory: Memory
    goal: Goal | None = None
    context: str = ""
    plan: Any = None
    actions: list[dict] = field(default_factory=list)
    watch: list[str] = field(default_factory=list)
    knobs: list = field(default_factory=list)  # KnobViews available when planning (round>0)
    is_baseline: bool = False
    planned: bool = False
    agent_rounds_total: int = 0
    agent_round_index: int = 0
    agent_rounds_after_this: int = 0


class TrainingSession:
    def __init__(self, goal: Goal | str | None = None, transport: Any | None = None,
                 run_id: str | None = None, memory: Any | None = None, round: int = 0,
                 agent: Any | None = None, max_rounds: int | None = None,
                 context: str = "", watch_metrics: list[str] | None = None,
                 seed: int | None = DEFAULT_SEED, frontend: bool | str | dict | None = None,
                 agent_active: bool = True):
        self.goal = _coerce_goal(goal)
        self.seed = seed  # re-applied each round so rounds share one baseline trajectory
        self.context = context  # static run background surfaced to the agent (plan/act/reflect)
        self.watch_metrics = list(watch_metrics or [])
        self.transport = transport if transport is not None else _build_frontend(frontend)
        self.run_id = run_id
        self.memory = _coerce_memory(memory)
        self.round = round
        self.max_rounds = max_rounds
        self.events = EventBus()
        self.actions = ActionBus()
        self.registry = ActionRegistry()
        self.knobs = KnobRegistry()
        self.state = TrainingState()
        self.control = ControlGate()
        self._agents: list[Any] = []
        self._model = None
        self._last_act_step = -1
        self._agent_enabled = True  # gated off during no-agent baseline rounds (run_rounds)
        self._agent_active = agent_active  # user intent, toggled at runtime via `set_agent`
        self._pending = StepControl()
        self.last_control = StepControl()
        self._round_actions: list[dict] = []
        self._round_plan_actions: list[dict] = []
        self._round_initial_config: dict | None = None
        self._round_budget: dict[str, int] = {}
        self._baseline_rounds = 0  # set by run_rounds; 0 in single-round/Tier-1 use
        self._started = False
        self._ended = False
        self._register_builtins()
        if agent is not None:
            for a in (agent if isinstance(agent, (list, tuple)) else [agent]):
                self.attach(a)

    def register_knob(self, name: str, get: Callable[[], Any], set: Callable[[Any], None], **meta) -> None:
        self.knobs.register(Knob(name=name, get=get, set=set, **meta))
        self.events.publish("knobs_registered",
                            {"knobs": [v.model_dump() for v in self.knobs.views()]},
                            self.state.branch_id)

    def register_optimizer_lr(self, optimizer, lr_scheduler=None, **meta) -> None:
        get, set = optimizer_lr_accessors(optimizer, lr_scheduler)
        if lr_scheduler is not None:
            sched = type(lr_scheduler).__name__
            desc = (f"base/peak learning rate. An LR scheduler ({sched}) is active, so the "
                    "effective per-step LR (reported as 'learning_rate') = base x "
                    "schedule_factor(step). Setting this rescales the schedule's amplitude "
                    "(peak), not the instantaneous LR; the schedule shape is preserved.")
        else:
            desc = "optimizer learning rate (applied directly to every step)."
        meta.setdefault("description", desc)
        meta.setdefault("min", 0.0)
        self.register_knob("lr", get, set, **meta)

    def action(self, name: str, description: str = "", payload_keys: list[str] | None = None):
        def deco(fn):
            self.registry.register(name, fn, description, payload_keys)
            return fn
        return deco

    def attach(self, agent: Any) -> "TrainingSession":
        self._agents.append(agent)
        if hasattr(agent, "on_event"):
            # Surface the agent's LLM prompts/responses (`agent_call`) on the event bus
            # so the frontend can render them; branch_id resolved at publish time.
            agent.on_event = lambda type, payload: self.events.publish(
                type, payload, self.state.branch_id)
        return self

    def autopatch(self, optimizer, metrics_fn=None):
        from interactive_training.integrations.autopatch import autopatch
        return autopatch(self, optimizer, metrics_fn)

    def bind_model(self, model) -> None:
        self._model = model
        self.state.model_tree = build_model_tree(model)
        self.events.publish("model_tree", {"tree": self.state.model_tree}, self.state.branch_id)

    def start(self) -> "TrainingSession":
        if self._started:  # idempotent: run_rounds and the Tier-1 trainer both bracket the session
            return self
        self._started = True
        self.state.status = "running"
        self.events.publish("status_changed", {"status": "running"}, self.state.branch_id)
        for a in self._agents:
            self.events.publish("agent_attached",
                                {"name": getattr(a, "name", type(a).__name__),
                                 "every": getattr(a, "every", None),
                                 "active": self._agent_active}, self.state.branch_id)
        if self.transport is not None:
            self.transport.start(self)
            _log_frontend(self.transport, self.agent_snapshot())
        return self

    def end(self) -> None:
        if not self._started or self._ended:
            return
        self._ended = True
        self.state.status = "ended"
        self.events.publish("status_changed", {"status": "ended"}, self.state.branch_id)
        if self.transport is not None:
            self.transport.stop()

    @property
    def started(self) -> bool:
        return self._started and not self._ended

    def begin_round(self, round_idx: int) -> None:
        """Reset per-round knobs/metrics/control while keeping the bus, transport,
        memory, and attached agents alive so a frontend stays connected across rounds."""
        if self.seed is not None:
            # same seed every round -> identical baseline RNG state, so the only thing
            # that changes a round's trajectory is the agent's (or human's) actions.
            seed_everything(self.seed)
        self.round = round_idx
        self.knobs = KnobRegistry()
        self.state = TrainingState()
        self.control = ControlGate()
        self._last_act_step = -1
        self._round_actions = []
        self._round_plan_actions = []
        self._round_initial_config = None
        self._pending = StepControl()
        self.state.status = "running"
        self.events.current_round = round_idx
        self.events.publish("round_started", {"round": round_idx}, self.state.branch_id)

    def end_round(self, score: float | None = None) -> None:
        self.events.publish("round_finished", {"round": self.round, "score": score}, self.state.branch_id)

    def run_rounds(self, train_round: Callable[["TrainingSession", Any], None],
                   max_rounds: int | None = None, goal: Goal | None = None, memory: Memory | None = None,
                   baseline_rounds: int = 1) -> Memory:
        """Drive the multi-round meta-loop from the session itself: run -> score
        -> reflect -> remember, reusing this one persistent session each round (plan §3.6).
        `train_round(session, ctx)` builds + runs one round; attached agents (if any)
        supply startup plans via `session.plan_round(ctx)` and end-of-round reflections.

        Round 0 is always a baseline when agents are attached. The first
        `baseline_rounds` round(s) run with the agent fully disabled (no plan,
        no online actions) so their score anchors the hyperparameter-tuning history as a
        clean baseline the agent can compare later rounds against. Baseline rounds are
        *extra*: they don't count against `max_rounds`, so `max_rounds` agent rounds always
        run after them (total = baseline_rounds + max_rounds)."""
        max_rounds = max_rounds if max_rounds is not None else (self.max_rounds or 5)
        goal = goal or self.goal
        memory = memory if memory is not None else (self.memory or Memory(None))
        self.memory = memory
        reflectors = [a for a in self._agents if hasattr(a, "reflect")]
        # With attached agents, round 0 is always a true no-agent baseline. Without agents,
        # every round is already agent-free, so don't spend an extra baseline round.
        baseline_rounds = max(1, baseline_rounds) if self._agents else 0
        total_rounds = baseline_rounds + max_rounds
        # Stable counts (set before start() so the round-0 Aim run + /state carry them).
        self.max_rounds = max_rounds
        self._baseline_rounds = baseline_rounds
        self.start()
        try:
            for r in range(total_rounds):
                is_baseline = r < baseline_rounds
                agent_round_index = 0 if is_baseline else r - baseline_rounds + 1
                agent_rounds_after_this = max_rounds if is_baseline else max_rounds - agent_round_index
                ctx = RoundContext(round=r, memory=memory, goal=goal, context=self.context,
                                   watch=list(self.watch_metrics), is_baseline=is_baseline,
                                   agent_rounds_total=max_rounds,
                                   agent_round_index=agent_round_index,
                                   agent_rounds_after_this=agent_rounds_after_this)
                # Set before begin_round() so round_snapshot() is correct the instant the
                # `round_started` event fires (the transport/frontend read it then).
                self._round_budget = {
                    "agent_rounds_total": ctx.agent_rounds_total,
                    "agent_round_index": ctx.agent_round_index,
                    "agent_rounds_after_this": ctx.agent_rounds_after_this,
                    "baseline_rounds": baseline_rounds,
                    "is_baseline": is_baseline,
                }
                self.begin_round(r)
                # Baseline rounds ignore any attached agent's online actions, so the round
                # trains purely on its default/plan config -- a reference point for the trend.
                self._agent_enabled = not is_baseline
                train_round(self, ctx)
                ctx.actions = list(self._round_plan_actions) + list(self._round_actions)
                score = goal.score(self.history) if goal else 0.0
                reflection = reflectors[0].reflect(ctx, self.trajectory, score) \
                    if reflectors and not is_baseline and self._agent_active else ""
                if reflection:
                    self.events.publish("agent_reflection", {"round": r, "text": reflection},
                                        self.state.branch_id)
                summ = self.round_summary()
                agent_usage = self._agent_usage()
                if agent_usage:
                    logger.info(
                        "round %d agent usage: cumulative input=%d output=%d cost=$%.6f",
                        r, agent_usage["input_tokens"], agent_usage["output_tokens"],
                        agent_usage["cost_usd"],
                    )
                # keep the round's starting hyperparameter config + decision list + peak
                # step so the agent sees the full tuning history (config -> score trend)
                memory.add_round(r, ctx.plan, score, reflection, extra={
                    "config": summ["config"], "baseline": is_baseline,
                    "best_step": summ["best_step"], "actions": summ["actions"],
                    "agent_usage": agent_usage})
                memory.update_best(goal.direction if goal is not None else "min")
                self.end_round(score)
                # `stop` ends only the current round (begin_round resets the gate next
                # round); the experiment terminates on the goal or max_rounds, not on stop.
                if goal is not None and goal.is_satisfied(score):
                    break
        finally:
            self.end()
        return memory

    def submit(self, action: Action, wait: bool = False, timeout: float | None = None) -> ActionResult | None:
        return self.actions.submit(action, wait=wait, timeout=timeout)

    def run(self):
        """Tier-3 sugar: `with session.run(): ...` brackets start()/end() (plan §7.1)."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            self.start()
            try:
                yield self
            finally:
                self.end()

        return _ctx()

    def step(self, metrics: dict, step: int | None = None, rank: int = 0, process_group=None,
             act: bool = True) -> StepControl:
        record = self.state.log(metrics, step)
        return self._barrier(record, rank, process_group, step=record.get("step"), act=act)

    def pump(self, step: int | None = None, rank: int = 0, process_group=None,
             act: bool = True) -> StepControl:
        """Per-step control-point barrier (no metrics logged); also where the agent
        gets a chance to act, so its cadence is measured in training steps. Pass
        `act=False` to only apply pending control/knob updates without letting the agent
        compute a new action (e.g. to defer acting until after an evaluation lands)."""
        return self._barrier(None, rank, process_group,
                             step=step if step is not None else self.state.step, act=act)

    def _barrier(self, record: dict | None, rank: int, process_group, step: int | None = None,
                 act: bool = True) -> StepControl:
        self._pending = StepControl()
        ddp, rank = self._ddp_info(rank, process_group)

        drained: list[Action] = []
        if rank == 0:
            if record is not None:
                self.events.publish("metrics", record, self.state.branch_id)
            if self._round_initial_config is None and self.knobs.views():
                # the recipe the round actually started from, before any agent action
                self._round_initial_config = {v.name: v.value for v in self.knobs.views()}
            if act:
                self._maybe_act(step)
            drained = self.actions.drain()
            for action in drained:
                self._dispatch(action)
            self.control.wait_if_paused(drain=self._drain_during_pause)

        if ddp:
            # Rank 0 owns the decision; carry the full control state plus a replay list of
            # model/optimizer-mutating actions so every rank ends the barrier in sync.
            replay = [{"type": a.type, "payload": dict(a.payload)} for a in drained
                      if a.type not in _DDP_HANDLED_ACTIONS]
            payload = broadcast_and_barrier(
                {"knobs": self._pending.knob_updates, "stop": self.control.stopped,
                 "load": self._pending.load, "reload": self._pending.reload_required,
                 "save": self._pending.save, "tag": self._pending.tag,
                 "evaluate": self._pending.evaluate, "replay": replay},
                rank, process_group)
            if rank != 0:
                for name, value in payload["knobs"].items():
                    if name in self.knobs:
                        self.knobs.set_value(name, value)
                self._pending.load = payload["load"]
                self._pending.reload_required = payload["reload"]
                self._pending.save = payload["save"]
                self._pending.tag = payload["tag"]
                self._pending.evaluate = payload["evaluate"]
                if payload["stop"]:
                    self.control.stop()
                for a in payload["replay"]:  # e.g. reset_module / freeze on this rank's model
                    self._dispatch(Action(type=a["type"], payload=a["payload"]))

        self._pending.stop = self.control.stopped
        return self._pending

    def report_eval(self, metrics: dict) -> None:
        record = self.state.log(metrics)
        self.events.publish("metrics", {**record, "eval": True}, self.state.branch_id)

    def checkpoint_saved(self, path: str, step: int, tag: str | None = None):
        ckpt = self.state.checkpoints.add(path, step, branch_id=self.state.branch_id, tag=tag)
        self.events.publish("checkpoint_saved", ckpt.model_dump(), self.state.branch_id)
        return ckpt

    def apply_plan(self, plan: Any, record: bool = False,
                   step: int | str | None = "start", source: str = "plan") -> list[dict]:
        config = getattr(plan, "config", None) or {}
        applied = []
        for name, value in config.items():
            key = name
            if key not in self.knobs and key == "learning_rate" and "lr" in self.knobs:
                key = "lr"  # agents habitually name the LR 'learning_rate'
            if key in self.knobs:
                old = self.knobs.get(key).get()
                new = self.knobs.set_value(key, value)
                if old != new:
                    action = {"step": step, "type": "set_knob",
                              "payload": {"name": key, "value": new},
                              "source": source, "ok": True}
                    applied.append(action)
                    if record:
                        self._round_plan_actions.append(action)
            else:
                logger.warning("apply_plan: ignoring unknown knob %r (not a registered knob)", name)
        return applied

    def plan_round(self, ctx: RoundContext, apply: bool = True) -> Any:
        """Ask the planner for this round's initial config after round state is ready.

        Recipes should call this after registering the new round's knobs and before the first
        training step. Baseline rounds skip planning so the initial config stays untouched.
        """
        if ctx.is_baseline or ctx.planned or not self._agent_active:
            return ctx.plan
        planners = [a for a in self._agents if hasattr(a, "plan")]
        if not planners:
            return ctx.plan
        ctx.knobs = self.knobs.views()
        ctx.plan = planners[0].plan(ctx)
        ctx.planned = True
        if ctx.plan is not None:
            self.events.publish("agent_plan",
                                {"round": ctx.round,
                                 "strategy": getattr(ctx.plan, "strategy", "") or "",
                                 "config": getattr(ctx.plan, "config", {}) or {}},
                                self.state.branch_id)
        if apply and ctx.knobs:
            self.apply_plan(ctx.plan, record=True)
        return ctx.plan

    @property
    def config_snapshot(self) -> dict:
        return {v.name: v.value for v in self.knobs.views()}

    def agent_snapshot(self) -> dict:
        snap = {"attached": bool(self._agents), "active": self._agent_active}
        if self._agents and hasattr(self._agents[0], "describe"):
            snap.update(self._agents[0].describe())
        return snap

    def round_snapshot(self) -> dict:
        return {"round": self.round, "rounds": dict(self._round_budget)}

    def wait_until_resumed(self, poll: float = 0.05) -> None:
        self.control.pause()
        self._set_status("paused")
        self.control.wait_if_paused(drain=self._drain_during_pause)
        if not self.control.stopped:
            self.control.resume()
            self._set_status("running")

    def round_summary(self) -> dict:
        final = self.config_snapshot
        initial = dict(self._round_initial_config or final)
        current, actions = dict(initial), []
        for a in sorted(self._round_actions, key=lambda a: (a.get("step") is None, a.get("step"))):
            if not a.get("ok"):
                continue
            step, payload = a.get("step"), a.get("payload") or {}
            if a.get("type") == "set_knob":
                name, value = payload.get("name"), payload.get("value")
                if name in final and current.get(name) != value:
                    actions.append({"step": step, "knob": name, "value": value})
                    current[name] = value
            else:
                actions.append({"step": step, "action": a.get("type")})

        best_step = None
        if self.goal is not None:
            rows = [(h.get("step"), h[self.goal.metric]) for h in self.state.history
                    if self.goal.metric in h]
            if rows:
                pick = min if self.goal.direction == "min" else max
                best_step = pick(rows, key=lambda r: r[1])[0]

        return {"best_step": best_step, "actions": actions, "config": initial}

    def _agent_usage(self) -> dict:
        input_tokens = sum(int(getattr(a, "input_tokens", 0) or 0) for a in self._agents)
        output_tokens = sum(int(getattr(a, "output_tokens", 0) or 0) for a in self._agents)
        cost_usd = sum(float(getattr(a, "cost_usd", 0.0) or 0.0) for a in self._agents)
        if input_tokens == 0 and output_tokens == 0 and cost_usd == 0.0:
            return {}
        return {"input_tokens": input_tokens, "output_tokens": output_tokens,
                "cost_usd": round(cost_usd, 8)}

    @property
    def history(self) -> list[dict]:
        return self.state.history

    @property
    def trajectory(self) -> list[dict]:
        return self.state.history

    def _ddp_info(self, rank: int, process_group) -> tuple[bool, int]:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return True, dist.get_rank(process_group)
        except Exception:
            pass
        return False, rank

    def _maybe_act(self, step: int | None = None) -> None:
        """Give attached agents a chance to act. `every` counts training steps; a step
        may surface via both pump() and step() (the log), so dedupe per step."""
        if not (self._agent_enabled and self._agent_active) or not self._agents \
                or step is None or step == self._last_act_step:
            return
        obs = None
        acted = False
        for agent in self._agents:
            every = getattr(agent, "every", 1) or 1
            if step % every != 0:
                continue
            if obs is None:
                obs = self._observation()
            for action in agent.act(obs) or []:
                action.source = getattr(agent, "name", "agent")
                self.actions.submit(action)
            acted = True
        if acted:
            self._last_act_step = step

    def _drain_during_pause(self) -> None:
        for action in self.actions.drain():
            self._dispatch(action)

    def _dispatch(self, action: Action) -> None:
        result = self.registry.dispatch(action, self)
        self.events.publish("action_result", {"id": action.id, "type": action.type,
                                               "ok": result.ok, "data": result.data,
                                               "error": result.error}, self.state.branch_id)
        self.actions.ack(action.id, result)
        self._pending.applied.append({"type": action.type, "ok": result.ok})
        self._round_actions.append({"step": self.state.step, "type": action.type,
                                    "payload": _redact_payload(dict(action.payload)),
                                    "source": action.source, "ok": result.ok})

    def _observation(self):
        from interactive_training.agents.agent import Observation
        return Observation(
            goal=self.goal,
            context=self.context,
            recent_metrics=self.state.recent(20),
            metrics_history=list(self.state.history),
            actions_taken=list(self._round_plan_actions) + list(self._round_actions),
            knobs={v.name: v for v in self.knobs.views()},
            status=self.state.status,
            round=self.round,
            agent_rounds_total=self._round_budget.get("agent_rounds_total", 0),
            agent_round_index=self._round_budget.get("agent_round_index", 0),
            agent_rounds_after_this=self._round_budget.get("agent_rounds_after_this", 0),
            memory_summary=self.memory.summarize(limit=_MEMORY_LIMIT) if self.memory else "",
            available_actions=self.registry.schemas(),
            watch=list(self.watch_metrics),
            model_modules=flatten_model_tree(self.state.model_tree),
        )

    def _set_status(self, status: str) -> None:
        self.state.status = status
        self.events.publish("status_changed", {"status": status}, self.state.branch_id)

    def _register_builtins(self) -> None:
        r = self.registry
        r.register("set_knob", self._h_set_knob, "Set a registered knob to a value", ["name", "value"])
        r.register("save_checkpoint", self._h_save, "Request a checkpoint save", ["tag"])
        r.register("load_checkpoint", self._h_load, "Load/fork from a checkpoint", ["checkpoint_id", "fork"])
        r.register("pause", self._h_pause, "Pause training")
        r.register("resume", self._h_resume, "Resume training")
        r.register("stop", self._h_stop, "Stop training")
        r.register("evaluate", self._h_evaluate, "Run evaluation", ["split"])
        r.register("reset_module", self._h_reset_module, "Reset a module's parameters", ["module_name"])
        r.register("note", self._h_note, "Record an annotation", ["text"])
        r.register("set_agent", self._h_set_agent,
                   "Enable/disable the attached babysitting agent", ["enabled"])
        r.register("configure_agent", self._h_configure_agent,
                   "Configure the LLM babysitter at runtime (provider/model/api_key/"
                   "reasoning_effort/cadence); attaches one if none exists",
                   ["provider", "model", "base_url", "api_key", "reasoning_effort", "every"])
        r.register("set_context", self._h_set_context,
                   "Replace the free-text training context handed to the agent", ["context"])

    def _h_set_knob(self, p, _):
        name = p["name"]
        if name not in self.knobs:
            return ActionResult.fail(f"unknown knob: {name}")
        value = self.knobs.set_value(name, p["value"])
        self._pending.knob_updates[name] = value
        self.events.publish("knob_changed", {"name": name, "value": value}, self.state.branch_id)
        return ActionResult.success(name=name, value=value)

    def _h_save(self, p, _):
        self._pending.save = True
        self._pending.tag = p.get("tag")
        return ActionResult.success()

    def _h_load(self, p, _):
        ckpt = self.state.checkpoints.get(p["checkpoint_id"])
        if ckpt is None:
            return ActionResult.fail("checkpoint not found")
        if p.get("fork"):
            branch = self.state.branches.fork(ckpt.branch_id, ckpt.id).id
        else:
            branch = ckpt.branch_id
        self._pending.load = ckpt.path
        self._pending.reload_required = True
        self.events.publish("checkpoint_loaded",
                            {"checkpoint_id": ckpt.id, "path": ckpt.path, "branch_id": branch}, branch)
        return ActionResult.success(path=ckpt.path, branch_id=branch)

    def _h_pause(self, p, _):
        self.control.pause()
        self._set_status("paused")
        return ActionResult.success()

    def _h_resume(self, p, _):
        self.control.resume()
        self._set_status("running")
        return ActionResult.success()

    def _h_stop(self, p, _):
        self.control.stop()
        self._pending.stop = True
        self._set_status("stopped")
        return ActionResult.success()

    def _h_evaluate(self, p, _):
        self._pending.evaluate = True
        self.events.publish("evaluate_requested", dict(p), self.state.branch_id)
        return ActionResult.success()

    def _h_reset_module(self, p, _):
        name = p["module_name"]
        self._pending.reset_modules.append(name)
        if self._model is not None and not _reset_named_module(self._model, name):
            return ActionResult.fail(f"module not found: {name}")
        self.events.publish("module_reset", {"module_name": name}, self.state.branch_id)
        return ActionResult.success(module_name=name)

    def _h_note(self, p, _):
        self.events.publish("note", {"text": p.get("text", "")}, self.state.branch_id)
        return ActionResult.success()

    def _h_set_agent(self, p, _):
        self._agent_active = bool(p.get("enabled", True))
        self.events.publish("agent_enabled", {"enabled": self._agent_active}, self.state.branch_id)
        return ActionResult.success(enabled=self._agent_active)

    def _h_configure_agent(self, p, _):
        agent = self._agents[0] if self._agents else self._attach_default_agent()
        if not hasattr(agent, "configure"):
            return ActionResult.fail("attached agent is not runtime-configurable")
        cfg = agent.configure(**{k: p[k] for k in
                                 ("every", "provider", "model", "base_url",
                                  "api_key", "reasoning_effort") if k in p})
        # `cfg` is the redacted describe() — never contains the key (§3.8).
        self.events.publish("agent_configured", cfg, self.state.branch_id)
        return ActionResult.success(**cfg)

    def _h_set_context(self, p, _):
        self.context = str(p.get("context", "") or "")
        self.events.publish("context_changed", {"context": self.context}, self.state.branch_id)
        return ActionResult.success(context=self.context)

    def _attach_default_agent(self):
        """Attach a fresh, unconfigured babysitter so a human can set it up from the
        frontend via `configure_agent` even when the run launched agent-free (§3.8)."""
        from interactive_training.agents.agent import LLMAgent
        agent = LLMAgent()
        self.attach(agent)
        self.events.publish("agent_attached",
                            {"name": getattr(agent, "name", "llm"),
                             "every": getattr(agent, "every", None),
                             "active": self._agent_active}, self.state.branch_id)
        return agent


def _reset_named_module(model, name: str) -> bool:
    target = model
    for part in name.split("."):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    for module in target.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return True
