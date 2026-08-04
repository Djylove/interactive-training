"""Validation and aggregation for XPolicyLab trial results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from interactive_training.integrations.xpolicylab.contracts import (
    EvaluationSummary,
    TrialResult,
)


def load_trial_results(
    path: str | Path,
    *,
    evaluation_id: str | None = None,
    checkpoint_id: str | None = None,
    max_bytes: int = 16 * 1024 * 1024,
) -> list[TrialResult]:
    """Load a bounded JSONL spool produced by an XPolicyLab environment client."""
    source = Path(path)
    if source.is_symlink():
        raise ValueError("trial result JSONL cannot be a symlink")
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"trial result JSONL not found: {source}")
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"trial result JSONL exceeds {max_bytes} bytes")

    results: list[TrialResult] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                trial = TrialResult.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid trial result at {source}:{line_number}: {exc}"
                ) from exc
            if evaluation_id is not None and trial.evaluation_id != evaluation_id:
                raise ValueError(
                    f"trial result at line {line_number} has unexpected evaluation_id"
                )
            if checkpoint_id is not None and trial.checkpoint_id != checkpoint_id:
                raise ValueError(
                    f"trial result at line {line_number} has unexpected checkpoint_id"
                )
            results.append(trial)
    return results


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class TrialAggregator:
    def __init__(
        self, evaluation_id: str, checkpoint_id: str, *, minimum_valid_trials: int = 1
    ):
        if minimum_valid_trials < 1:
            raise ValueError("minimum_valid_trials must be positive")
        self.evaluation_id = evaluation_id
        self.checkpoint_id = checkpoint_id
        self.minimum_valid_trials = minimum_valid_trials
        self._trials: list[TrialResult] = []
        self._trial_ids: set[str] = set()
        self._evidence_type: str | None = None

    def add(self, trial: TrialResult | dict) -> TrialResult:
        trial = (
            trial
            if isinstance(trial, TrialResult)
            else TrialResult.model_validate(trial)
        )
        if trial.evaluation_id != self.evaluation_id:
            raise ValueError("trial evaluation_id does not match aggregator")
        if trial.checkpoint_id != self.checkpoint_id:
            raise ValueError("trial checkpoint_id does not match aggregator")
        if trial.trial_id in self._trial_ids:
            raise ValueError(f"duplicate trial_id: {trial.trial_id}")
        if (
            self._evidence_type is not None
            and trial.evidence_type != self._evidence_type
        ):
            raise ValueError("evaluation cannot mix trial evidence types")
        self._evidence_type = trial.evidence_type
        self._trial_ids.add(trial.trial_id)
        self._trials.append(trial)
        return trial

    @property
    def trials(self) -> tuple[TrialResult, ...]:
        return tuple(self._trials)

    def summarize(self) -> EvaluationSummary:
        valid = [
            trial for trial in self._trials if trial.outcome in {"success", "failure"}
        ]
        task_valid = [trial for trial in valid if trial.evidence_type == "task_outcome"]
        replay_valid = [
            trial for trial in valid if trial.evidence_type == "replay_validation"
        ]
        invalid = len(self._trials) - len(valid)
        task_values: dict[str, list[float]] = defaultdict(list)
        for trial in task_valid:
            assert trial.success is not None
            task_values[trial.task].append(trial.success)

        task_rates = {
            task: mean(values) for task, values in sorted(task_values.items())
        }
        conclusive = (
            len(task_valid) >= self.minimum_valid_trials
            or len(replay_valid) >= self.minimum_valid_trials
        )
        metrics: dict[str, float] = {
            "invalid_trial_rate": invalid / len(self._trials) if self._trials else 0.0,
        }
        if task_valid:
            metrics["success_rate"] = mean([trial.success for trial in task_valid])
            metrics["worst_task_success_rate"] = min(task_rates.values())
        if replay_valid:
            metrics["replay_pass_rate"] = mean(
                [trial.success for trial in replay_valid]
            )
            metrics["replay_trial_count"] = float(len(replay_valid))

        steps = [
            float(trial.episode_steps)
            for trial in valid
            if trial.episode_steps is not None
        ]
        if steps:
            metrics["episode_steps_mean"] = mean(steps)

        metric_values: dict[str, list[float]] = defaultdict(list)
        for trial in valid:
            for name, value in trial.metrics.items():
                metric_values[name].append(value)
        for name, values in sorted(metric_values.items()):
            if name == "inference_latency_ms":
                metrics["inference_latency_p95_ms"] = _percentile(values, 0.95)
            else:
                metrics[f"{name}_mean"] = mean(values)

        return EvaluationSummary(
            evaluation_id=self.evaluation_id,
            checkpoint_id=self.checkpoint_id,
            evidence_type=self._evidence_type,
            status="completed" if conclusive else "inconclusive",
            total_trials=len(self._trials),
            valid_trials=len(valid),
            invalid_trials=invalid,
            metrics=metrics,
            task_success_rates=task_rates,
        )
