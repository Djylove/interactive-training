#!/usr/bin/env python3
"""Prepare session-isolated GR3 tomato train/validation manifests.

The source LeRobot v3 dataset stores all episodes in one parquet/video batch.
This script creates reference-only clip selections; it never copies or rewrites
the source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROFILE_ID = "xpolicylab.gr3_anygrasp_lerobot_v3"
SOURCE_SCHEMA = "lerobot_v3_gr3qnexo_top"
PROMPT = "Pick the tomato from the table and place it inside the basket on the left."
BATCH_PATH = "task_001656/batch_000000"
VIDEO_PATH = f"{BATCH_PATH}/videos/observation.images.top/chunk-000/file-000.mp4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact(workspace_root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(workspace_root)),
        "size": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _validate_source(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[int, str]]:
    batch_root = dataset_root / BATCH_PATH
    info_path = batch_root / "meta/info.json"
    episodes_path = batch_root / "meta/episodes/chunk-000/file-000.parquet"
    selected_path = dataset_root / "selected_episodes.json"
    data_path = batch_root / "data/chunk-000/file-000.parquet"
    video_path = dataset_root / VIDEO_PATH
    for path in (info_path, episodes_path, selected_path, data_path, video_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    expected = {
        "robot_type": "gr3qnexo",
        "fps": 30,
        "state_shape": [33],
        "action_shape": [37],
        "camera_dtype": "video",
    }
    actual = {
        "robot_type": info.get("robot_type"),
        "fps": int(info.get("fps", 0)),
        "state_shape": features.get("observation.state", {}).get("shape"),
        "action_shape": features.get("action", {}).get("shape"),
        "camera_dtype": features.get("observation.images.top", {}).get("dtype"),
    }
    if actual != expected:
        raise ValueError(f"unsupported GR3 tomato schema: {actual!r}")

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    source_session_by_episode = {
        int(item["task_episode_index"]): str(item["source_session_id"])
        for item in selected["episodes"]
    }
    table = pq.read_table(
        episodes_path,
        columns=[
            "episode_index",
            "length",
            "dataset_from_index",
            "dataset_to_index",
            "expand_task",
        ],
    )
    rows = table.to_pylist()
    if len(rows) != int(selected["selected_count"]):
        raise ValueError("episode metadata and selected_episodes.json disagree")
    if set(source_session_by_episode) != {int(row["episode_index"]) for row in rows}:
        raise ValueError("source session mapping does not cover every episode")
    for row in rows:
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        if end - start != int(row["length"]):
            raise ValueError(f"invalid episode interval: {row['episode_index']}")
        if str(row["expand_task"]).strip() != PROMPT:
            raise ValueError(f"unexpected prompt in episode {row['episode_index']}")
    return rows, source_session_by_episode


def _choose_validation_session(
    session_counts: dict[str, int], validation_fraction: float
) -> str:
    target = sum(session_counts.values()) * validation_fraction
    return min(
        session_counts,
        key=lambda session: (
            abs(session_counts[session] - target),
            hashlib.sha256(session.encode()).hexdigest(),
        ),
    )


def _write_selection(
    output_root: Path,
    split: str,
    rows: list[dict[str, Any]],
    source_session_by_episode: dict[int, str],
) -> tuple[Path, Path, int]:
    selection_root = output_root / split
    manifest_dir = selection_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=False)
    clip_rows: list[dict[str, Any]] = []
    for row in rows:
        episode_index = int(row["episode_index"])
        clip_rows.append(
            {
                "clip_id": f"gr3-tomato-episode-{episode_index:06d}",
                "task_id": "1656",
                "batch_path": BATCH_PATH,
                "episode_index": episode_index,
                "subtask": PROMPT,
                "prompt": PROMPT,
                "clip_frame_start": 0,
                "video_frame_start": int(row["dataset_from_index"]),
                "num_clip_frames": int(row["length"]),
                "source_video": VIDEO_PATH,
                "source_session_id": source_session_by_episode[episode_index],
            }
        )
    clips_path = manifest_dir / "clips.parquet"
    pq.write_table(pa.Table.from_pylist(clip_rows), clips_path, compression="zstd")
    frames = sum(int(row["num_clip_frames"]) for row in clip_rows)
    report = {
        "schema_version": "gr3_tomato_selection.v1",
        "split": split,
        "clips": len(clip_rows),
        "frames": frames,
        "sessions": sorted({row["source_session_id"] for row in clip_rows}),
        "selection": "complete_episode_reference_only",
        "quality_gate": "operator_confirmed_verified_dataset",
        "prompt": PROMPT,
    }
    report_path = manifest_dir / "selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return clips_path, report_path, frames


def _build_manifest(
    workspace_root: Path,
    dataset_root: Path,
    clips_path: Path,
    report_path: Path,
    split: str,
    split_sessions: list[str],
    validation_session: str,
    frames: int,
    validation_fraction: float,
) -> dict[str, Any]:
    clips = pq.read_table(clips_path)
    clip_count = len(clips)
    source_files = [
        clips_path,
        report_path,
        dataset_root / "selected_episodes.json",
        dataset_root / BATCH_PATH / "meta/info.json",
        dataset_root / BATCH_PATH / "meta/episodes/chunk-000/file-000.parquet",
    ]
    files = sorted(
        (_artifact(workspace_root, path) for path in source_files),
        key=lambda item: item["path"],
    )
    profile_data = {
        "robot_type": "gr3qnexo",
        "state_dim": 33,
        "action_dim": 37,
        "model_state_dim": 31,
        "model_action_dim": 31,
        "model_axis_selection": "first_31",
        "fps": 30,
        "camera_key": "observation.images.top",
        "camera_views": 1,
        "dataset_root": str(dataset_root),
        "clips_file": str(clips_path.relative_to(workspace_root)),
        "clip_limit": clip_count,
        "available_clips_before_limit": clip_count,
        "selected_frames": frames,
        "batch_count": 1,
        "selected_task_count": None,
        "selected_task_ids": [],
        "split": split,
        "split_strategy": "manifest_prefix",
        "split_seed": 20260805,
        "validation_fraction": validation_fraction,
        "selected_source_sessions": split_sessions,
        "validation_source_session_sha256": hashlib.sha256(
            validation_session.encode()
        ).hexdigest(),
        "raw_media_access": "read_only_in_place",
        "instruction_source": "prompt",
        "sample_stride": 1,
        "quality_gate": "operator_confirmed_verified_dataset",
    }
    episode = {
        "episode_id": f"gr3-tomato-{split}-{clip_count}-episodes",
        "path": str(workspace_root),
        "source_schema": SOURCE_SCHEMA,
        "profile_id": PROFILE_ID,
        "file_manifest_sha256": _stable_digest(files),
        "files": files,
        "task_id": "gr3-tomato-grasp",
        "task_instruction": PROMPT,
        "task_outcome": "unknown",
        "outcome_confirmed_by_operator": False,
        "termination_reason": "curated_offline_dataset",
        "recording_saved_successfully": True,
        "provenance": {
            "dataset": "grasp_710",
            "selection": split,
            "selection_strategy": "source_session_holdout",
        },
        "statistics": {
            "clips": clip_count,
            "frames": frames,
            "batches": 1,
            "tasks": 1,
        },
        "profile_data": profile_data,
        "filters": [],
        "warnings": ["source_video_and_action_parquet_are_read_only_in_place"],
        "exclusion_reasons": [],
        "requires_filtering": False,
        "train_eligible_after_filters": True,
    }
    episode["episode_manifest_sha256"] = _stable_digest(episode)
    dataset_digest = _stable_digest(
        [
            {
                "episode_id": episode["episode_id"],
                "episode_manifest_sha256": episode["episode_manifest_sha256"],
            }
        ]
    )
    return {
        "schema_version": "xpolicy_dataset.v1",
        "dataset_id": f"gr3-tomato-{split}-{clip_count}-{dataset_digest[:12]}",
        "dataset_name": f"GR3 tomato grasp {split} session split",
        "dataset_sha256": dataset_digest,
        "source_root": str(workspace_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile_id": PROFILE_ID,
        "source_revisions": {
            "xpolicylab_commit": None,
            "dataset_revision": "grasp_710",
        },
        "profile_config": profile_data,
        "summary": {
            "episode_count": 1,
            "eligible_episode_count": 1,
            "excluded_episode_count": 0,
            "requires_filtering_episode_count": 0,
            "task_outcome_counts": {"unknown": 1},
            "statistics": {
                "clips": clip_count,
                "frames": frames,
                "batches": 1,
                "tasks": 1,
            },
        },
        "episodes": [episode],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    workspace_root = args.workspace_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("validation-fraction must be between 0 and 0.5")
    if output_root.exists():
        parser.error(f"refusing to overwrite output root: {output_root}")
    if not dataset_root.is_relative_to(workspace_root):
        parser.error("dataset-root must be inside workspace-root")
    if not output_root.is_relative_to(workspace_root):
        parser.error("output-root must be inside workspace-root")

    rows, source_session_by_episode = _validate_source(dataset_root)
    session_counts: dict[str, int] = {}
    for session in source_session_by_episode.values():
        session_counts[session] = session_counts.get(session, 0) + 1
    if len(session_counts) < 2:
        raise ValueError("session-isolated split requires at least two source sessions")
    validation_session = _choose_validation_session(
        session_counts, args.validation_fraction
    )
    split_rows = {
        "train": [
            row
            for row in rows
            if source_session_by_episode[int(row["episode_index"])] != validation_session
        ],
        "validation": [
            row
            for row in rows
            if source_session_by_episode[int(row["episode_index"])] == validation_session
        ],
    }
    output_root.mkdir(parents=True)
    manifests: dict[str, str] = {}
    split_summary: dict[str, Any] = {}
    for split, selected_rows in split_rows.items():
        clips_path, report_path, frames = _write_selection(
            output_root, split, selected_rows, source_session_by_episode
        )
        sessions = sorted(
            {source_session_by_episode[int(row["episode_index"])] for row in selected_rows}
        )
        manifest = _build_manifest(
            workspace_root,
            dataset_root,
            clips_path,
            report_path,
            split,
            sessions,
            validation_session,
            frames,
            args.validation_fraction,
        )
        manifest_path = output_root / f"{split}.xpolicy-dataset.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifests[split] = str(manifest_path)
        split_summary[split] = {
            "episodes": len(selected_rows),
            "frames": frames,
            "sessions": sessions,
            "dataset_id": manifest["dataset_id"],
        }

    summary = {
        "schema_version": "gr3_tomato_training_candidate.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_root),
        "quality_gate": "operator_confirmed_verified_dataset",
        "split_strategy": "source_session_holdout_closest_to_fraction",
        "validation_fraction_requested": args.validation_fraction,
        "validation_session": validation_session,
        "session_episode_counts": session_counts,
        "model_contract": {
            "state": "first_31_of_canonical_33",
            "action": "first_31_of_canonical_37",
            "horizon": 50,
            "frequency_hz": 30,
            "prompt": PROMPT,
        },
        "normalization": "fit_on_train_split_only_and_reuse_for_validation",
        "splits": split_summary,
        "manifests": manifests,
    }
    summary_path = output_root / "training_candidate.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
