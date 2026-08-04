"""Validation for the GR3 AnyGrasp LeRobot v3 training profile."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from interactive_training.integrations.xpolicylab.contracts import DatasetManifest

GR3_ANYGRASP_PROFILE_ID = "xpolicylab.gr3_anygrasp_lerobot_v3"


class Gr3AnygraspProfile(BaseModel):
    robot_type: Literal["gr3qnexo"]
    state_dim: Literal[33]
    action_dim: Literal[37]
    fps: Literal[30]
    camera_key: Literal["observation.images.top"]
    camera_views: Literal[1]
    dataset_root: str = Field(min_length=1)
    clips_file: str = Field(min_length=1)
    clip_limit: int = Field(ge=1)
    available_clips_before_limit: int | None = Field(default=None, ge=1)
    selected_frames: int = Field(ge=1)
    batch_count: int = Field(ge=1)
    selected_task_count: int | None = Field(default=None, ge=1)
    selected_task_ids: list[str] = Field(default_factory=list)
    split: Literal["all", "train", "validation", "heldout"] = "all"
    split_strategy: Literal["manifest_prefix", "task_id_sha256"] = "manifest_prefix"
    split_seed: int | None = None
    heldout_fraction: float | None = Field(default=None, gt=0, lt=1)
    validation_fraction: float | None = Field(default=None, ge=0, lt=1)
    heldout_task_ids_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    validation_task_ids_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_media_access: Literal["read_only_in_place"]

    @model_validator(mode="after")
    def split_metadata_is_consistent(self):
        if self.selected_task_ids:
            if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
                raise ValueError("selected_task_ids must be unique")
            if self.selected_task_count != len(self.selected_task_ids):
                raise ValueError("selected_task_count must match selected_task_ids")
            if self.available_clips_before_limit is None or self.available_clips_before_limit < self.clip_limit:
                raise ValueError("available clip count must cover clip_limit")
            if (
                self.split_strategy != "task_id_sha256"
                or self.split_seed is None
                or self.heldout_fraction is None
                or self.heldout_task_ids_sha256 is None
            ):
                raise ValueError("task split metadata is incomplete")
            if self.validation_fraction is not None and self.validation_fraction > 0:
                if self.validation_task_ids_sha256 is None:
                    raise ValueError("validation task split metadata is incomplete")
                if self.heldout_fraction + self.validation_fraction >= 1:
                    raise ValueError("task split must leave training tasks")
        return self


def validate_gr3_anygrasp_manifest(manifest: DatasetManifest) -> DatasetManifest:
    if manifest.profile_id != GR3_ANYGRASP_PROFILE_ID:
        raise ValueError(f"expected profile_id={GR3_ANYGRASP_PROFILE_ID}")
    if len(manifest.episodes) != 1:
        raise ValueError("GR3 AnyGrasp profile requires one selection binding")
    episode = manifest.episodes[0]
    if episode.source_schema != "lerobot_v3_gr3qnexo_top":
        raise ValueError("GR3 AnyGrasp requires the LeRobot v3 gr3qnexo top-camera schema")
    profile = Gr3AnygraspProfile.model_validate(episode.profile_data)
    expected_statistics: dict[str, int] = {
        "clips": profile.clip_limit,
        "frames": profile.selected_frames,
        "batches": profile.batch_count,
    }
    if profile.selected_task_count is not None:
        expected_statistics["tasks"] = profile.selected_task_count
    for name, expected in expected_statistics.items():
        if episode.statistics.get(name) != expected:
            raise ValueError(f"GR3 AnyGrasp episode statistic mismatch: {name}")
        if manifest.summary.statistics.get(name) != expected:
            raise ValueError(f"GR3 AnyGrasp dataset statistic mismatch: {name}")
    return manifest
