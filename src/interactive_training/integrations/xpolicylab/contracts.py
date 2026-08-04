"""Versioned, dependency-light contracts for the XPolicyLab bridge."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "xpolicy_interactive.v1"
Identifier = str
Outcome = Literal["success", "failure", "aborted", "timeout", "invalid"]
EvidenceType = Literal["task_outcome", "protocol_debug", "replay_validation"]
RunStatus = Literal["completed", "failed", "timeout", "aborted"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VersionedModel(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION


class TrainSpec(BaseModel):
    enabled: bool = True
    max_steps: int | None = Field(default=None, ge=1)
    learning_rate: float | None = Field(default=None, gt=0)
    dataset_mix: dict[str, float] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("learning_rate")
    @classmethod
    def finite_learning_rate(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("learning_rate must be finite")
        return value

    @field_validator("dataset_mix")
    @classmethod
    def valid_dataset_mix(cls, value: dict[str, float]) -> dict[str, float]:
        for name, weight in value.items():
            if not name or not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    "dataset_mix requires non-empty names and finite "
                    "non-negative weights"
                )
        if value and sum(value.values()) <= 0:
            raise ValueError("dataset_mix must contain a positive weight")
        return value


class EvaluationSpec(BaseModel):
    environment: Literal["debug", "replay", "sim", "robot_shadow", "robot_enforce"] = (
        "debug"
    )
    tasks: list[str] = Field(min_length=1)
    repeats: int = Field(default=1, ge=1, le=1000)
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env")
    @classmethod
    def valid_environment(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not key
            or "\x00" in key
            or "=" in key
            or "\x00" in setting
            for key, setting in value.items()
        ):
            raise ValueError("evaluation environment names and values must be safe strings")
        return value

    @field_validator("tasks")
    @classmethod
    def unique_tasks(cls, value: list[str]) -> list[str]:
        clean = [task.strip() for task in value]
        if any(not task for task in clean):
            raise ValueError("evaluation tasks must be non-empty")
        if any(
            task in {".", ".."} or "/" in task or "\\" in task or "\x00" in task
            for task in clean
        ):
            raise ValueError("evaluation tasks cannot contain path separators")
        if len(set(clean)) != len(clean):
            raise ValueError("evaluation tasks must be unique")
        return clean


class ExperimentSpec(VersionedModel):
    experiment_id: Identifier = Field(min_length=1, max_length=128)
    round: int = Field(ge=0)
    policy: str = Field(min_length=1, max_length=128)
    checkpoint_name: str = Field(default="interactive", min_length=1, max_length=128)
    bench_name: str = Field(min_length=1, max_length=128)
    env_cfg_type: str = Field(min_length=1, max_length=128)
    action_type: str = Field(min_length=1, max_length=128)
    seed: int = Field(ge=0)
    mode: Literal["local_smoke", "remote_train"] = "local_smoke"
    train: TrainSpec = Field(default_factory=TrainSpec)
    evaluation: EvaluationSpec
    parent_checkpoint: str | None = None
    dataset_manifest_id: str | None = None

    @field_validator(
        "experiment_id",
        "policy",
        "checkpoint_name",
        "bench_name",
        "env_cfg_type",
        "action_type",
        "dataset_manifest_id",
    )
    @classmethod
    def safe_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
            raise ValueError("identifiers cannot contain path separators")
        return value


class ArtifactFile(BaseModel):
    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetEpisodeManifest(BaseModel):
    """Robot-agnostic episode identity and training eligibility."""

    episode_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    source_schema: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    file_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[ArtifactFile] = Field(min_length=1)
    task_id: str
    task_instruction: str
    task_outcome: Literal["success", "failure", "aborted", "unknown"]
    outcome_confirmed_by_operator: bool
    termination_reason: str
    recording_saved_successfully: bool
    provenance: dict[str, str | int | float | bool | None]
    statistics: dict[str, float]
    profile_data: dict[str, object]
    filters: list[dict[str, object]]
    warnings: list[str]
    exclusion_reasons: list[str]
    requires_filtering: bool
    train_eligible_after_filters: bool

    @model_validator(mode="after")
    def eligibility_is_consistent(self):
        if self.train_eligible_after_filters == bool(self.exclusion_reasons):
            raise ValueError("eligibility must match exclusion reasons")
        if any(
            not name or not math.isfinite(float(value))
            for name, value in self.statistics.items()
        ):
            raise ValueError("dataset statistics must be named and finite")
        return self


class DatasetSummary(BaseModel):
    episode_count: int = Field(ge=1)
    eligible_episode_count: int = Field(ge=0)
    excluded_episode_count: int = Field(ge=0)
    requires_filtering_episode_count: int = Field(ge=0)
    task_outcome_counts: dict[str, int]
    statistics: dict[str, float]

    @field_validator("task_outcome_counts")
    @classmethod
    def valid_outcome_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or count < 0 for key, count in value.items()):
            raise ValueError("task outcome counts must be non-negative")
        return value

    @field_validator("statistics")
    @classmethod
    def finite_statistics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(
            not name or not math.isfinite(float(metric))
            for name, metric in value.items()
        ):
            raise ValueError("dataset statistics must be named and finite")
        return {name: float(metric) for name, metric in value.items()}


class DatasetManifest(BaseModel):
    schema_version: Literal["xpolicy_dataset.v1"]
    dataset_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_root: str = Field(min_length=1)
    created_at: datetime
    profile_id: str = Field(min_length=1)
    source_revisions: dict[str, str | None]
    profile_config: dict[str, object]
    summary: DatasetSummary
    episodes: list[DatasetEpisodeManifest] = Field(min_length=1)

    @model_validator(mode="after")
    def summary_matches_episodes(self):
        eligible = sum(ep.train_eligible_after_filters for ep in self.episodes)
        expected = {
            "episode_count": len(self.episodes),
            "eligible_episode_count": eligible,
            "excluded_episode_count": len(self.episodes) - eligible,
            "requires_filtering_episode_count": sum(
                ep.requires_filtering for ep in self.episodes
            ),
        }
        for field_name, value in expected.items():
            if getattr(self.summary, field_name) != value:
                raise ValueError(f"dataset summary mismatch: {field_name}")
        if len({episode.episode_id for episode in self.episodes}) != len(self.episodes):
            raise ValueError("dataset episode ids must be unique")
        if any(episode.profile_id != self.profile_id for episode in self.episodes):
            raise ValueError("dataset episode profile ids must match the dataset")
        return self


class SourceRevision(BaseModel):
    interactive_training_commit: str | None = None
    xpolicylab_commit: str | None = None
    policy: str


class CheckpointArtifact(BaseModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step: int = Field(default=0, ge=0)
    files: list[ArtifactFile] = Field(default_factory=list)


class DatasetBinding(BaseModel):
    dataset_id: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_id: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    eligible_episode_count: int = Field(ge=1)


class ArtifactManifest(VersionedModel):
    experiment_id: str
    round: int = Field(ge=0)
    status: RunStatus
    source: SourceRevision
    checkpoint: CheckpointArtifact | None = None
    dataset: DatasetBinding | None = None
    logs: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    returncode: int | None = None
    error: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def completed_has_checkpoint(self):
        if self.status == "completed" and self.checkpoint is None:
            raise ValueError("completed training requires a checkpoint")
        return self


class TrialResult(VersionedModel):
    evaluation_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    seed: int = Field(ge=0)
    repeat_index: int = Field(ge=0)
    evidence_type: EvidenceType = "task_outcome"
    outcome: Outcome
    success: float | None = Field(default=None, ge=0.0, le=1.0)
    episode_steps: int | None = Field(default=None, ge=0)
    termination_reason: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str | None] = Field(default_factory=dict)

    @field_validator("metrics")
    @classmethod
    def finite_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        for name, metric in value.items():
            if not name or isinstance(metric, bool) or not math.isfinite(float(metric)):
                raise ValueError(
                    "metrics require non-empty names and finite numeric values"
                )
        return {name: float(metric) for name, metric in value.items()}

    @model_validator(mode="after")
    def outcome_matches_success(self):
        if self.outcome == "success" and self.success != 1.0:
            raise ValueError("success outcome requires success=1.0")
        if self.outcome == "failure" and self.success != 0.0:
            raise ValueError("failure outcome requires success=0.0")
        if self.evidence_type == "protocol_debug" and (
            self.outcome != "invalid" or self.success is not None
        ):
            raise ValueError("protocol_debug evidence requires invalid outcome")
        if self.evidence_type == "replay_validation" and self.outcome not in {
            "success",
            "failure",
        }:
            raise ValueError(
                "replay_validation evidence requires success or failure outcome"
            )
        return self


class EvaluationSummary(VersionedModel):
    evaluation_id: str
    checkpoint_id: str
    evidence_type: EvidenceType | None = None
    status: Literal["completed", "inconclusive"]
    total_trials: int = Field(ge=0)
    valid_trials: int = Field(ge=0)
    invalid_trials: int = Field(ge=0)
    metrics: dict[str, float]
    task_success_rates: dict[str, float] = Field(default_factory=dict)

    @field_validator("metrics", "task_success_rates")
    @classmethod
    def finite_summary_metrics(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(float(metric)) for metric in value.values()):
            raise ValueError("summary metrics must be finite")
        return {name: float(metric) for name, metric in value.items()}

    def session_metrics(self) -> dict[str, float]:
        result = dict(self.metrics)
        result["evaluation_conclusive"] = float(self.status == "completed")
        if self.status == "completed":
            result.update(
                {
                    f"task_success_rate/{task}": rate
                    for task, rate in self.task_success_rates.items()
                }
            )
        else:
            # Do not let a Goal/Agent interpret partial evidence as a valid score.
            result.pop("success_rate", None)
            result.pop("worst_task_success_rate", None)
        return result


class EmbodiedRoundRecord(BaseModel):
    spec: ExperimentSpec
    artifact: ArtifactManifest
    evaluation: EvaluationSummary | None = None
