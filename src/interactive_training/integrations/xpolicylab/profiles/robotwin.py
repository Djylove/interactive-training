"""Validation for TurboVLA RoboTwin clean dataset manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from interactive_training.integrations.xpolicylab.contracts import DatasetManifest

ROBOTWIN_CLEAN_PROFILE_ID = "turbovla.robotwin_clean_v1"
ROBOTWIN_SOURCE_SCHEMA = "lerobot_v2_robotwin_clean"
ROBOTWIN_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


class RoboTwinCleanTaskProfile(BaseModel):
    task_name: str = Field(min_length=1)
    robot_type: Literal["aloha"]
    lerobot_version: str = Field(pattern=r"^v2")
    state_dim: Literal[14]
    action_dim: Literal[14]
    horizon: Literal[50]
    fps: int = Field(gt=0)
    trajectory_count: int = Field(gt=0)
    frame_count: int = Field(gt=0)
    video_count: int = Field(gt=0)
    camera_keys: list[str]

    @model_validator(mode="after")
    def validate_camera_and_counts(self):
        if self.camera_keys != ROBOTWIN_CAMERA_KEYS:
            raise ValueError("RoboTwin clean profile requires the canonical cameras")
        if self.video_count != self.trajectory_count * len(self.camera_keys):
            raise ValueError("RoboTwin clean video count must be 3 per trajectory")
        if self.frame_count < self.trajectory_count:
            raise ValueError("RoboTwin clean frame count is inconsistent")
        return self


def validate_robotwin_clean_manifest(manifest: DatasetManifest) -> DatasetManifest:
    if manifest.profile_id != ROBOTWIN_CLEAN_PROFILE_ID:
        raise ValueError(
            f"expected profile_id={ROBOTWIN_CLEAN_PROFILE_ID}, got {manifest.profile_id}"
        )
    profiles = []
    task_names = set()
    for episode in manifest.episodes:
        if episode.source_schema != ROBOTWIN_SOURCE_SCHEMA:
            raise ValueError("RoboTwin clean profile requires LeRobot v2 sources")
        profile = RoboTwinCleanTaskProfile.model_validate(episode.profile_data)
        if profile.task_name != episode.task_id or profile.task_name in task_names:
            raise ValueError("RoboTwin task identity is missing or duplicated")
        task_names.add(profile.task_name)
        expected = {
            "trajectory_count": profile.trajectory_count,
            "frame_count": profile.frame_count,
            "video_count": profile.video_count,
        }
        for name, value in expected.items():
            if episode.statistics.get(name) != value:
                raise ValueError(f"RoboTwin task statistic mismatch: {name}")
        profiles.append(profile)

    aggregates = {
        "trajectory_count": sum(profile.trajectory_count for profile in profiles),
        "frame_count": sum(profile.frame_count for profile in profiles),
        "video_count": sum(profile.video_count for profile in profiles),
    }
    for name, value in aggregates.items():
        if manifest.summary.statistics.get(name) != value:
            raise ValueError(f"RoboTwin dataset statistic mismatch: {name}")
    return manifest
