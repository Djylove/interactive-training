"""One-round train/evaluate orchestration built on the versioned bridge."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from interactive_training.integrations.xpolicylab.contracts import (
    ArtifactManifest,
    EvaluationSummary,
    ExperimentSpec,
    TrialResult,
)
from interactive_training.integrations.xpolicylab.reporter import TrialAggregator


class EmbodiedRoundResult(BaseModel):
    spec: ExperimentSpec
    artifact: ArtifactManifest
    evaluation: EvaluationSummary | None = None


Evaluator = Callable[
    [ExperimentSpec, ArtifactManifest, str, str],
    Iterable[TrialResult | dict],
]


class XPolicyExperiment:
    def __init__(self, session: Any, runner: Any):
        self.session = session
        self.runner = runner

    def run_round(
        self,
        spec: ExperimentSpec,
        evaluator: Evaluator,
        *,
        gpu_ids: str = "0",
        timeout_seconds: int | None = None,
        dataset_manifest_path: str | Path | None = None,
        record_memory: bool = True,
    ) -> EmbodiedRoundResult:
        owns_session = not self.session.started
        self.session.start()
        try:
            artifact = self.runner.prepare(
                spec,
                gpu_ids=gpu_ids,
                timeout_seconds=timeout_seconds,
                dataset_manifest_path=dataset_manifest_path,
            )
            if artifact.status != "completed" or artifact.checkpoint is None:
                return EmbodiedRoundResult(spec=spec, artifact=artifact)

            checkpoint = artifact.checkpoint
            self.session.checkpoint_saved(
                checkpoint.path, checkpoint.step, tag=spec.experiment_id
            )
            evaluation_id = f"{spec.experiment_id}-round-{spec.round}"
            expected_trials = len(spec.evaluation.tasks) * spec.evaluation.repeats
            aggregator = TrialAggregator(
                evaluation_id,
                checkpoint.sha256,
                minimum_valid_trials=expected_trials,
            )
            for trial in evaluator(spec, artifact, evaluation_id, checkpoint.sha256):
                aggregator.add(trial)
            summary = aggregator.summarize()
            session_metrics = summary.session_metrics()
            self.session.report_eval(session_metrics)
            result = EmbodiedRoundResult(
                spec=spec, artifact=artifact, evaluation=summary
            )

            score_metric = (
                self.session.goal.metric
                if self.session.goal is not None
                else "success_rate"
            )
            if (
                record_memory
                and self.session.memory is not None
                and summary.status == "completed"
                and score_metric in session_metrics
            ):
                score = (
                    self.session.goal.score(self.session.history)
                    if self.session.goal is not None
                    else session_metrics["success_rate"]
                )
                self.session.memory.add_round(
                    spec.round,
                    spec.train,
                    score,
                    "",
                    extra={"embodied": result.model_dump(mode="json")},
                )
                direction = (
                    self.session.goal.direction
                    if self.session.goal is not None
                    else "max"
                )
                self.session.memory.update_best(direction)
            return result
        finally:
            if owns_session:
                self.session.end()
