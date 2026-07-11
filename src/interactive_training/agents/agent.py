"""Agent protocol + Observation + the LLM agent — the only computed action source (plan §4)."""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from interactive_training.core.actions import Action, ActionSchema
from interactive_training.core.goals import Goal
from interactive_training.core.knobs import KnobView

logger = logging.getLogger(__name__)

# Prior-round digest length shown to the agent (keeps prompts from growing unbounded).
_MEMORY_LIMIT = 10


class Observation(BaseModel):
    goal: Goal | None = None
    context: str = ""  # static run background (model, task, algorithm, knob notes)
    recent_metrics: list[dict] = Field(default_factory=list)
    metrics_history: list[dict] = Field(default_factory=list)
    actions_taken: list[dict] = Field(default_factory=list)
    knobs: dict[str, KnobView] = Field(default_factory=dict)
    status: str = "running"
    round: int = 0
    agent_rounds_total: int = 0
    agent_round_index: int = 0
    agent_rounds_after_this: int = 0
    memory_summary: str = ""
    available_actions: list[ActionSchema] = Field(default_factory=list)
    watch: list[str] = Field(default_factory=list)
    model_modules: list[str] = Field(default_factory=list)  # valid targets for reset_module/freeze


class Plan(BaseModel):
    config: dict = Field(default_factory=dict)
    strategy: str = ""


@runtime_checkable
class Agent(Protocol):
    def act(self, obs: Observation) -> list[Action]: ...


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, tools: list[dict] | None = None) -> tuple:
        """Return (text, [(tool_name, arguments), ...], optional usage dict)."""
        ...


class OpenAIClient:
    """Provider adapter (OpenAI / OpenRouter / local); openai imported lazily so core
    keeps no hard dependency (§4). `base_url`/`api_key` target any compatible endpoint;
    `reasoning_effort` and `extra_body` are forwarded to each completion call.

    `api="responses"` issues a native OpenAI Responses request (/v1/responses), which is
    required for reasoning models like gpt-5.x when combining tools with reasoning effort
    (chat.completions rejects that combination). `api="chat"` uses chat.completions."""

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None,
                 api_key: str | None = None, reasoning_effort: str | None = None,
                 extra_body: dict | None = None, api: str = "chat",
                 provider: str = "openai", **client_kwargs):
        self.model = model
        self.provider = provider
        self.base_url = base_url or None
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body
        self.api = api
        self._api_key = api_key or None
        self._client_kwargs = dict(client_kwargs)
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = dict(self._client_kwargs)
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    @property
    def api_key_set(self) -> bool:
        """True if a usable key is configured — either passed explicitly or via the
        environment variable the OpenAI SDK falls back to. The key itself is never
        exposed (§3.8)."""
        import os
        return bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    def configure(self, *, provider: str | None = None, model: str | None = None,
                  base_url: str | None = None, api_key: str | None = None,
                  reasoning_effort: str | None = None,
                  extra_body: dict | None = None) -> None:
        """Apply only the provided fields, then drop the lazy client so the next
        completion rebuilds against the new config (§3.8 timing)."""
        if provider is not None:
            self.provider = provider
        if model is not None:
            self.model = model
        if base_url is not None:
            self.base_url = base_url or None
        if api_key:  # write-only; an empty value means "leave the current key unchanged"
            self._api_key = api_key
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort or None
        if extra_body is not None:
            self.extra_body = extra_body
        self._client = None

    def describe(self) -> dict:
        """Non-secret snapshot for GET /state.agent (§3.2). Never includes the key."""
        return {"provider": self.provider, "model": self.model,
                "base_url": self.base_url, "reasoning_effort": self.reasoning_effort,
                "api_key_set": self.api_key_set}

    def complete(self, system: str, user: str, tools: list[dict] | None = None):
        if self.api == "responses":
            return self._complete_responses(system, user, tools)
        return self._complete_chat(system, user, tools)

    def _complete_chat(self, system: str, user: str, tools: list[dict] | None):
        client = self._ensure()
        call_kwargs: dict = {}
        if self.reasoning_effort is not None:
            call_kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body is not None:
            call_kwargs["extra_body"] = self.extra_body
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=tools or None,
            **call_kwargs,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append((tc.function.name, args))
        return msg.content or "", calls, _usage_from_response(resp)

    def _complete_responses(self, system: str, user: str, tools: list[dict] | None):
        client = self._ensure()
        call_kwargs: dict = {}
        if self.reasoning_effort is not None:
            call_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.extra_body is not None:
            call_kwargs["extra_body"] = self.extra_body
        resp = client.responses.create(
            model=self.model,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=[_chat_tool_to_responses(t) for t in tools] if tools else None,
            **call_kwargs,
        )
        calls = []
        for item in (getattr(resp, "output", None) or []):
            if getattr(item, "type", None) != "function_call":
                continue
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append((item.name, args))
        return getattr(resp, "output_text", "") or "", calls, _usage_from_response(resp)


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _usage_from_response(resp: Any) -> dict:
    usage = _get_field(resp, "usage")
    if usage is None:
        return {}
    input_tokens = _get_field(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _get_field(usage, "prompt_tokens", 0)
    output_tokens = _get_field(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _get_field(usage, "completion_tokens", 0)
    total_tokens = _get_field(usage, "total_tokens")
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _unpack_completion(result: tuple) -> tuple[str, list[tuple[str, dict]], dict]:
    if len(result) == 2:
        text, calls = result
        return text, calls, {}
    text, calls, usage = result[:3]
    return text, calls, usage or {}


def _chat_tool_to_responses(tool: dict) -> dict:
    """Flatten a chat.completions function tool into the Responses API shape
    ({type, name, description, parameters} at top level)."""
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        return {"type": "function", "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {})}
    return tool


def _tools_from_actions(schemas: list[ActionSchema], knobs: dict[str, KnobView]) -> list[dict]:
    tools = []
    for s in schemas:
        props = {}
        if s.type == "set_knob":
            props = {
                "name": {"type": "string", "enum": list(knobs.keys())},
                "value": {"type": "number"},
            }
        else:
            props = {k: {"type": "string"} for k in s.payload_keys}
        desc = s.description
        if s.type == "set_knob" and knobs:
            desc += " | knobs: " + ", ".join(
                f"{v.name}={v.value} [{v.min},{v.max}] {v.description}" for v in knobs.values())
        tools.append({
            "type": "function",
            "function": {
                "name": s.type,
                "description": desc,
                "parameters": {"type": "object", "properties": props,
                               "required": [k for k in props]},
            },
        })
    return tools


# Throughput/timing fields that add noise without informing the babysitting decision.
_METRIC_NOISE = frozenset({
    "eval_runtime", "eval_samples_per_second", "eval_steps_per_second",
    "train_runtime", "train_samples_per_second", "train_steps_per_second", "total_flos",
})


def _clean_metric(m: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in m.items() if k not in _METRIC_NOISE}


def _downsample(rows: list[dict], max_points: int = 50) -> list[dict]:
    """Evenly subsample a series down to max_points, always keeping first and last."""
    n = len(rows)
    if n <= max_points:
        return list(rows)
    step = (n - 1) / (max_points - 1)
    idxs = sorted({round(i * step) for i in range(max_points)})
    return [rows[i] for i in idxs]


_AUX_SIGNALS = ("reward", "kl", "entropy")


def _metric_sections(history: list[dict], goal: Goal | None,
                     watch: list[str] | tuple[str, ...] = ()) -> list[tuple[str, list[dict], str]]:
    """Pick the (label, rows, key) series worth showing for this goal.

    Always surfaces the training loss (general health signal); additionally surfaces the
    goal's own metric series (reward / accuracy / eval_loss / ...) so the agent sees what
    it is actually scored on, plus common RL health signals (reward/kl/entropy) when
    present -- regardless of direction."""
    sections: list[tuple[str, list[dict], str]] = []
    shown: set[str] = set()

    def add(label: str, key: str) -> None:
        if key in shown:
            return
        rows = [h for h in history if key in h]
        if rows:
            sections.append((label, rows, key))
            shown.add(key)

    add("Train loss", "loss")
    # Effective per-step LR (after any scheduler); differs from the base `lr` knob.
    add("Effective LR (scheduler-applied)", "learning_rate")
    metric = goal.metric if goal else None
    if metric:
        add(f"Goal metric '{metric}' ({goal.direction})", metric)
    for key in watch:
        add(key, key)
    for key in _AUX_SIGNALS:
        add(key.capitalize(), key)
    return sections


def _fmt_series(rows: list[dict], key: str) -> str:
    pts = [f"{r.get('step', '?')}:{round(r[key], 6)}" for r in rows if key in r]
    return ", ".join(pts) if pts else "(none yet)"


def _fmt_action_value(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _fmt_actions(actions: list[dict]) -> str:
    if not actions:
        return "  (none yet)"
    out, pending_knobs = [], []

    def flush_knobs() -> None:
        if not pending_knobs:
            return
        step, status, items = pending_knobs[0][0], pending_knobs[0][1], []
        while pending_knobs:
            s, st, item = pending_knobs.pop(0)
            if s != step or st != status or len(items) >= 10:
                out.append(f"  step {step}: set_knob " + ", ".join(items) + f" [{status}]")
                step, status, items = s, st, []
            items.append(item)
        if items:
            out.append(f"  step {step}: set_knob " + ", ".join(items) + f" [{status}]")

    for a in actions:
        payload = ", ".join(f"{k}={v}" for k, v in (a.get("payload") or {}).items())
        status = "ok" if a.get("ok", True) else "failed"
        if a.get("type") == "set_knob":
            p = a.get("payload") or {}
            item = f"{p.get('name')}={_fmt_action_value(p.get('value'))}"
            pending_knobs.append((a.get("step", "?"), status, item))
            continue
        flush_knobs()
        out.append(f"  step {a.get('step', '?')}: {a.get('type')}({payload}) [{status}]")
    flush_knobs()
    return "\n".join(out)


def _describe_goal(goal: Goal | None) -> str:
    if goal is None:
        return "No specific goal; keep the training run healthy."
    verb = "minimize" if goal.direction == "min" else "maximize"
    desc = f"{verb.capitalize()} the metric '{goal.metric}' ({goal.name})"
    if goal.target is not None:
        desc += f", aiming to reach a target of {goal.target}"
    return desc + "."


def _round_budget_line(obj: Any, *, phase: str = "during") -> str:
    agent_total = int(getattr(obj, "agent_rounds_total", 0) or 0)
    agent_idx = int(getattr(obj, "agent_round_index", 0) or 0)
    agent_after = int(getattr(obj, "agent_rounds_after_this", 0) or 0)
    if not agent_total:
        return ""
    if agent_idx:
        return f"Budget: agent round {agent_idx}/{agent_total}; {agent_after} after this."
    if phase == "reflect":
        return f"Budget: baseline done; {agent_total} agent rounds remain."
    return f"Budget: baseline; {agent_total} agent rounds after this."


def _format_tool_calls(calls: list[tuple[str, dict]]) -> str:
    if not calls:
        return "(none)"
    return json.dumps([{"name": n, "arguments": a} for n, a in calls], indent=2, default=str)


class LLMAgent:
    """Tool-calling babysitter. Knobs+actions become the model's tool schema (auto-synced)."""

    INPUT_COST_PER_MILLION = 5.0
    OUTPUT_COST_PER_MILLION = 30.0

    # Actions the LLM agent is never allowed to take. (save_checkpoint/evaluate stay
    # available: they are wired through StepControl and referenced by the run contexts.)
    # reset_module is destructive (wipes trained parameters) and stays human-only.
    # set_agent/configure_agent/set_context would let the agent disable or rewire
    # itself; blocking them also keeps the tool schema (sent every call) small.
    BLOCKED_ACTIONS = frozenset({"load_checkpoint", "pause", "resume", "note", "reset_module",
                                 "set_agent", "configure_agent", "set_context"})

    def __init__(self, client: LLMClient | None = None, model: str = "gpt-4o-mini",
                 every: int = 50, name: str = "llm"):
        self.client = client or OpenAIClient(model=model)
        self.every = every
        self.name = name
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        # Observability hook `(event_type, payload) -> None`; the session wires it to its
        # event bus on attach() so every LLM prompt/response surfaces in the frontend.
        self.on_event: Any = None

    def describe(self) -> dict:
        """Extended /state.agent view (§3.2): cadence + the client's non-secret config.
        The API key is never included — only `api_key_set`."""
        view = {"every": self.every}
        client = self.client
        if hasattr(client, "describe"):
            view.update(client.describe())
        else:  # a custom client without the config surface: report what we can
            view.update({"provider": None, "model": getattr(client, "model", None),
                         "base_url": None, "reasoning_effort": None, "api_key_set": False})
        return view

    def configure(self, *, every: int | None = None, provider: str | None = None,
                  model: str | None = None, base_url: str | None = None,
                  api_key: str | None = None, reasoning_effort: str | None = None) -> dict:
        """Apply a runtime `configure_agent` action (§3.8). Only provided keys change;
        `api_key` is write-only. Returns the redacted `describe()` (no key)."""
        if every is not None:
            try:
                self.every = max(1, int(every))
            except (TypeError, ValueError):
                pass
        client_cfg = {k: v for k, v in (
            ("provider", provider), ("model", model), ("base_url", base_url),
            ("reasoning_effort", reasoning_effort)) if v is not None}
        if api_key:  # write-only; empty means "keep the existing key"
            client_cfg["api_key"] = api_key
        if client_cfg and hasattr(self.client, "configure"):
            self.client.configure(**client_cfg)
        return self.describe()

    def _complete(self, phase: str, system: str, user: str,
                  tools: list[dict] | None = None) -> tuple[str, list[tuple[str, dict]]]:
        tools_repr = json.dumps(tools, indent=2) if tools else "(none)"
        logger.info(
            "[%s] %s — agentic prompt\n"
            "======== system ========\n%s\n"
            "======== user ========\n%s\n"
            "======== tools ========\n%s",
            self.name, phase, system, user, tools_repr,
        )
        text, calls, usage = _unpack_completion(self.client.complete(system, user, tools))
        self._log_usage(phase, usage)
        logger.info(
            "[%s] %s — agent response\n"
            "======== text ========\n%s\n"
            "======== tool_calls ========\n%s",
            self.name, phase, text or "(empty)", _format_tool_calls(calls),
        )
        if self.on_event is not None:
            try:
                self.on_event("agent_call", {
                    "agent": self.name, "phase": phase, "system": system, "user": user,
                    "response": text or "",
                    "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                    "usage": usage or {},
                })
            except Exception:
                logger.exception("[%s] on_event hook failed for agent_call", self.name)
        return text, calls

    def _log_usage(self, phase: str, usage: dict) -> None:
        if not usage:
            logger.info("[%s] %s — token usage unavailable", self.name, phase)
            return
        in_tokens = int(usage.get("input_tokens") or 0)
        out_tokens = int(usage.get("output_tokens") or 0)
        req_cost = (in_tokens * self.INPUT_COST_PER_MILLION
                    + out_tokens * self.OUTPUT_COST_PER_MILLION) / 1_000_000
        self.input_tokens += in_tokens
        self.output_tokens += out_tokens
        self.cost_usd += req_cost
        logger.info(
            "[%s] %s — token usage: request input=%d output=%d total=%d cost=$%.6f | "
            "cumulative input=%d output=%d cost=$%.6f",
            self.name, phase, in_tokens, out_tokens, usage.get("total_tokens", in_tokens + out_tokens),
            req_cost, self.input_tokens, self.output_tokens, self.cost_usd,
        )

    def act(self, obs: Observation) -> list[Action]:
        allowed = [s for s in obs.available_actions if s.type not in self.BLOCKED_ACTIONS]
        tools = _tools_from_actions(allowed, obs.knobs)
        system = ("You babysit a training run toward its goal. Watch the metrics and the "
                  "prior-round history, and adjust knobs or issue control actions (via tool "
                  "calls) when the trend suggests a change will help. You may emit several "
                  "tool calls at once, or none at all -- if the run is healthy and on track, "
                  "make no call and let it continue unchanged. Use the budget to pace "
                  "exploration.")
        user = self._render(obs)
        phase = f"act (round={obs.round}, status={obs.status})"
        _, calls = self._complete(phase, system, user, tools)
        calls = [(n, a) for n, a in calls if n not in self.BLOCKED_ACTIONS]
        return [Action(type=name, payload=args, source=self.name) for name, args in calls]

    def plan(self, ctx: Any) -> Plan:
        goal = ctx.goal
        context = getattr(ctx, "context", "") or ""
        knobs = getattr(ctx, "knobs", None) or []
        knob_desc = "; ".join(
            f"{k.name}={k.value} [{k.min},{k.max}] {k.description}".strip() for k in knobs
        ) or "(registered when the round starts; infer names from the run background)"
        system = (
            "You choose the starting hyperparameter config for a training round. Read the "
            "prior rounds' configs and scores to see what helped, and build on the best "
            "result so far -- keep the settings that work and change the ones that don't. "
            "If the score has stalled, make a bolder change; if a setting is already good, "
            "leave it as is. Use the budget to pace exploration. Set the tunable knobs by "
            "their exact names; you may also include any extra config keys described in the "
            "run background."
        )
        bg = f"Run background:\n{context.strip()}\n\n" if context else ""
        budget = _round_budget_line(ctx, phase="plan")
        budget = f"{budget}\n" if budget else ""
        user = (f"{bg}Goal: {goal.name} ({goal.direction} {goal.metric}).\n"
                f"{budget}"
                f"Tunable knobs: {knob_desc}\n"
                f"Prior rounds:\n{ctx.memory.summarize(limit=_MEMORY_LIMIT)}\n"
                'Reply with JSON: {"config": {name: value}, "strategy": "..."}.')
        text, _ = self._complete(f"plan (round={ctx.round})", system, user, None)
        return self._parse_plan(text)

    def reflect(self, ctx: Any, trajectory: list[dict], score: float) -> str:
        series = "\n".join(
            f"{label} (step:{key}): {_fmt_series(_downsample(rows), key)}"
            for label, rows, key in _metric_sections(trajectory, ctx.goal, getattr(ctx, "watch", [])))
        actions = _fmt_actions(getattr(ctx, "actions", None) or [])
        context = getattr(ctx, "context", "") or ""
        bg = f"Run background:\n{context.strip()}\n" if context else ""
        budget = _round_budget_line(ctx, phase="reflect")
        budget = f"{budget}\n" if budget else ""
        system = (
            "You reflect on a finished training round and write a concise lesson. "
            "If a config performs about the same as an earlier round, don't just retry it; "
            "suggest a genuinely different thing to try next rather than the same plan again."
        )
        user = (f"{bg}Goal: {_describe_goal(ctx.goal)} Final score={score}.\n"
                f"{budget}"
                f"{series}\n"
                f"Actions taken this round:\n{actions}\n"
                f"Prior rounds (reflections):\n{ctx.memory.summarize(limit=_MEMORY_LIMIT)}\n"
                "Write one or two sentences of actionable lessons for the next round, "
                "and a specific new variation to try.")
        text, _ = self._complete(f"reflect (round={ctx.round}, score={score})", system, user, None)
        return text.strip()

    def _render(self, obs: Observation) -> str:
        knobs = ", ".join(f"{k}={v.value}" for k, v in obs.knobs.items()) or "(none)"
        history = obs.metrics_history or obs.recent_metrics
        lines = []
        if obs.context:
            lines += ["Run background:", obs.context.strip(), ""]
        lines += [
            f"Goal: {_describe_goal(obs.goal)}",
            f"Status: {obs.status} | Round: {obs.round}",
            f"Knobs: {knobs}",
        ]
        if budget := _round_budget_line(obs, phase="act"):
            lines.append(budget)
        for label, rows, key in _metric_sections(history, obs.goal, obs.watch):
            ds = _downsample(rows)
            lines.append(f"{label} (step:{key}, {len(ds)} pts): {_fmt_series(ds, key)}")
        mod_actions = [s.type for s in obs.available_actions
                       if "module_name" in s.payload_keys and s.type not in self.BLOCKED_ACTIONS]
        if obs.model_modules and mod_actions:
            names = ", ".join(obs.model_modules[:64])
            lines.append(f"Modules (valid module_name for {', '.join(mod_actions)}): {names}")
        lines += ["Actions taken this round:", _fmt_actions(obs.actions_taken)]
        if obs.memory_summary:
            lines += ["Prior rounds (reflections):", obs.memory_summary]
        return "\n".join(lines)

    def _parse_plan(self, text: str) -> Plan:
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
            return Plan(config=data.get("config", {}), strategy=data.get("strategy", ""))
        except (ValueError, json.JSONDecodeError):
            logger.warning("plan: could not parse a JSON config from the model output; "
                           "falling back to an empty plan (round keeps its default config).")
            return Plan()
