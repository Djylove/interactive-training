"""Persistent, append-only cross-round memory (plan §3.6)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _num(v: Any) -> str:
    """Compact number rendering (e.g. 8e-06, 0.35, 1) to save prompt tokens."""
    return f"{v:g}" if isinstance(v, float) else str(v)


class Memory:
    def __init__(self, path: str | None = "memory.jsonl"):
        self.path = Path(path) if path else None
        # Sibling file holding ONLY the current best round (config + actions + reflection),
        # overwritten each round: e.g. bert_imdb_memory.jsonl -> bert_imdb_metrics.jsonl.
        self.metrics_path = self._metrics_path(self.path)
        self.rounds: list[dict] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self.rounds.append(json.loads(line))

    @staticmethod
    def _metrics_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        name = path.name.replace("memory", "metrics")
        if name == path.name:  # path didn't contain "memory"
            name = f"{path.stem}_metrics{path.suffix}"
        return path.with_name(name)

    def add_round(self, round_idx: int, plan: Any, score: float,
                  reflection: str, extra: dict | None = None) -> None:
        entry = {
            "round": round_idx,
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
            "score": score,
            "reflection": reflection,
        }
        for k, v in (extra or {}).items():  # best_step / actions (when present)
            if v not in (None, [], {}):
                entry[k] = v
        self.rounds.append(entry)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")

    def summarize(self, limit: int | None = None) -> str:
        """Token-efficient prior-round digest: peak score, the decisions taken (knob
        changes + control actions), and the reflection -- so the agent can tie each lesson
        to what it did, without anchoring on a noisy per-round config snapshot."""
        if not self.rounds:
            return "No prior rounds."
        rounds = self.rounds[-limit:] if limit else self.rounds
        return "\n".join(self._fmt_round(r) for r in rounds)

    @staticmethod
    def _fmt_decision(d: dict) -> str:
        if "knob" in d:
            return f"s{d['step']}:{d['knob']}={_num(d['value'])}"
        return f"s{d['step']}:{d['action']}"

    @classmethod
    def _fmt_round(cls, r: dict) -> str:
        tag = "Round {} (baseline)".format(r["round"]) if r.get("baseline") else "Round {}".format(r["round"])
        peak = f"score={r['score']:.4g}"
        if r.get("best_step") is not None:
            peak += f"@step{r['best_step']}"
        parts = [f"{tag}: {peak}"]
        config = r.get("config")
        if config:  # the round's starting hyperparameters -> lets the agent read the trend
            parts.append("config={" + ", ".join(f"{k}={_num(v)}" for k, v in config.items()) + "}")
        actions = r.get("actions") or r.get("schedule")  # "schedule" = legacy knob-only key
        if actions:
            parts.append("actions=[" + ", ".join(cls._fmt_decision(d) for d in actions) + "]")
        parts.append(f"lesson={r['reflection']}")
        return "; ".join(parts)

    def best(self, direction: str = "min") -> dict | None:
        if not self.rounds:
            return None
        key = lambda r: r["score"]
        return min(self.rounds, key=key) if direction == "min" else max(self.rounds, key=key)

    def update_best(self, direction: str = "min") -> dict | None:
        best = self.best(direction)
        if best is not None and self.metrics_path:
            self.metrics_path.write_text(json.dumps(best, indent=2) + "\n")
        return best
