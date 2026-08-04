"""Validation of the GR3 DAgger v2 payload inside a generic manifest."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from interactive_training.integrations.xpolicylab.contracts import DatasetManifest

GR3_DAGGER_PROFILE_ID = "xpolicylab.gr3_dagger_v2"


class Gr3Alignment(BaseModel):
    limit_ms: float = Field(gt=0)
    canonical_state_max_ms_after_filter: float = Field(ge=0)
    expert_max_ms_after_filter: float = Field(ge=0)
    raw_camera_state_max_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def finite_metrics(self):
        if any(
            not math.isfinite(value)
            for value in (
                self.limit_ms,
                self.canonical_state_max_ms_after_filter,
                self.expert_max_ms_after_filter,
                self.raw_camera_state_max_ms,
            )
        ):
            raise ValueError("GR3 alignment metrics must be finite")
        return self


class Gr3DaggerEpisodeProfile(BaseModel):
    """Fields that are meaningful only for the GR3 DAgger v2 recorder."""

    state_dim: Literal[33]
    action_dim: Literal[37]
    camera_frames: int = Field(ge=0)
    trainable_camera_frames: int = Field(ge=0)
    trailing_audit_frames: int = Field(ge=0)
    valid_label_rows: int = Field(ge=0)
    label_rows: int = Field(ge=0)
    selected_action_source_counts: dict[str, int]
    control_mode_counts: dict[str, int]
    alignment: Gr3Alignment

    @field_validator("selected_action_source_counts", "control_mode_counts")
    @classmethod
    def valid_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or count < 0 for key, count in value.items()):
            raise ValueError("GR3 category counts must be non-negative")
        return value

    @model_validator(mode="after")
    def consistent_frame_and_label_counts(self):
        if (
            self.trainable_camera_frames + self.trailing_audit_frames
            != self.camera_frames
        ):
            raise ValueError(
                "GR3 trainable and trailing frames must total camera frames"
            )
        if self.valid_label_rows > self.label_rows:
            raise ValueError("GR3 valid labels cannot exceed label rows")
        return self


def validate_gr3_dagger_manifest(manifest: DatasetManifest) -> DatasetManifest:
    """Apply GR3-only schema and aggregate checks to a generic manifest."""

    if manifest.profile_id != GR3_DAGGER_PROFILE_ID:
        raise ValueError(
            f"expected profile_id={GR3_DAGGER_PROFILE_ID}, got {manifest.profile_id}"
        )
    profiles: list[Gr3DaggerEpisodeProfile] = []
    for episode in manifest.episodes:
        if episode.source_schema != "gr3_dagger_v2":
            raise ValueError("GR3 profile requires gr3_dagger_v2 episode sources")
        profile = Gr3DaggerEpisodeProfile.model_validate(episode.profile_data)
        expected_statistics = {
            "camera_frames": profile.camera_frames,
            "trainable_camera_frames": profile.trainable_camera_frames,
            "valid_label_rows": profile.valid_label_rows,
        }
        for name, expected in expected_statistics.items():
            if episode.statistics.get(name) != expected:
                raise ValueError(f"GR3 episode statistic mismatch: {name}")
        profiles.append(profile)

    aggregate_statistics = {
        "camera_frames": sum(profile.camera_frames for profile in profiles),
        "trainable_camera_frames": sum(
            profile.trainable_camera_frames for profile in profiles
        ),
        "valid_label_rows": sum(profile.valid_label_rows for profile in profiles),
        "intervention_count": sum(
            int(episode.statistics.get("intervention_count", 0))
            for episode in manifest.episodes
        ),
    }
    for name, expected in aggregate_statistics.items():
        if manifest.summary.statistics.get(name) != expected:
            raise ValueError(f"GR3 dataset statistic mismatch: {name}")
    return manifest
