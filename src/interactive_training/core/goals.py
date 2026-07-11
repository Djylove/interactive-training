"""Goal/Objective: turns a metric history into one comparable score (plan §3.4)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Goal(BaseModel):
    name: str
    metric: str
    direction: Literal["min", "max"] = "min"
    target: float | None = None

    def values(self, history: list[dict]) -> list[float]:
        return [float(h[self.metric]) for h in history if self.metric in h]

    def score(self, history: list[dict]) -> float:
        vals = self.values(history)
        if not vals:
            return float("inf") if self.direction == "min" else float("-inf")
        return min(vals) if self.direction == "min" else max(vals)

    def better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "min" else a > b

    def is_satisfied(self, score: float) -> bool:
        if self.target is None:
            return False
        return score <= self.target if self.direction == "min" else score >= self.target


def ValidationLoss(metric: str = "val_loss", target: float | None = None) -> Goal:
    return Goal(name="validation_loss", metric=metric, direction="min", target=target)


def AverageReward(metric: str = "reward", target: float | None = None) -> Goal:
    return Goal(name="average_reward", metric=metric, direction="max", target=target)


def Accuracy(metric: str = "accuracy", target: float | None = None) -> Goal:
    return Goal(name="accuracy", metric=metric, direction="max", target=target)
