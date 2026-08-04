#!/usr/bin/env python3
"""Convert one successful RoboTwin HDF5 episode to LeRobot v2.1.

This intentionally writes a standalone single-task dataset.  It never mutates the
published RoboTwin clean dataset, which keeps gate-overfit evidence separate from
general evaluation data.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}
MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


def _json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _decode_images(dataset: h5py.Dataset) -> list[np.ndarray]:
    frames = []
    for encoded in dataset:
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("RoboTwin HDF5 contains an invalid JPEG frame")
        frames.append(image)
    return frames


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "medium"}
        for image in frames:
            if image.shape[:2] != (height, width):
                raise ValueError("camera resolution changes within an episode")
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _numeric_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def convert(source: Path, output: Path, instruction: str, fps: int) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"missing regular RoboTwin HDF5: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not instruction.strip():
        raise ValueError("instruction cannot be empty")
    if fps < 1:
        raise ValueError("fps must be positive")

    with h5py.File(source, "r") as episode:
        joints = np.asarray(episode["joint_action/vector"], dtype=np.float32)
        if joints.ndim != 2 or joints.shape[1] != 14:
            raise ValueError(f"expected joint_action/vector [T,14], got {joints.shape}")
        images = {
            key: _decode_images(episode[f"observation/{source_key}/rgb"])
            for key, source_key in CAMERAS.items()
        }
    length = joints.shape[0]
    if any(len(camera_frames) != length for camera_frames in images.values()):
        raise ValueError("camera/action frame counts differ")
    height, width = images["cam_high"][0].shape[:2]

    try:
        data_dir = output / "data" / "chunk-000"
        meta_dir = output / "meta"
        data_dir.mkdir(parents=True)
        meta_dir.mkdir(parents=True)

        # RoboTwin's public converter derives both state and action from the
        # collected joint trajectory when no separate measured qpos is present.
        flat = pa.array(joints.reshape(-1), type=pa.float32())
        state = pa.FixedSizeListArray.from_arrays(flat, 14)
        action = pa.FixedSizeListArray.from_arrays(
            pa.array(joints.reshape(-1), type=pa.float32()), 14
        )
        table = pa.table(
            {
                "observation.state": state,
                "action": action,
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / fps),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.zeros(length), type=pa.int64()),
                "index": pa.array(np.arange(length), type=pa.int64()),
                "task_index": pa.array(np.zeros(length), type=pa.int64()),
            }
        )
        pq.write_table(table, data_dir / "episode_000000.parquet")

        for camera, frames in images.items():
            _write_video(
                output
                / "videos"
                / "chunk-000"
                / f"observation.images.{camera}"
                / "episode_000000.mp4",
                frames,
                fps,
            )

        video_features = {}
        for camera in CAMERAS:
            video_features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": [3, height, width],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.height": height,
                    "video.width": width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }
        scalar_features = {
            key: {"dtype": dtype, "shape": [1], "names": None}
            for key, dtype in {
                "timestamp": "float32",
                "frame_index": "int64",
                "episode_index": "int64",
                "index": "int64",
                "task_index": "int64",
            }.items()
        }
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": [MOTORS],
            },
            "action": {"dtype": "float32", "shape": [14], "names": [MOTORS]},
            **video_features,
            **scalar_features,
        }
        _json(
            meta_dir / "info.json",
            {
                "codebase_version": "v2.1",
                "robot_type": "aloha",
                "total_episodes": 1,
                "total_frames": length,
                "total_tasks": 1,
                "total_videos": 3,
                "total_chunks": 1,
                "chunks_size": 1000,
                "fps": fps,
                "splits": {"train": "0:1"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": features,
            },
        )
        _json(
            meta_dir / "modality.json",
            {
                "action": {
                    "left_joints": {"start": 0, "end": 6, "original_key": "action"},
                    "left_gripper": {"start": 6, "end": 7, "original_key": "action"},
                    "right_joints": {"start": 7, "end": 13, "original_key": "action"},
                    "right_gripper": {"start": 13, "end": 14, "original_key": "action"},
                },
                "state": {
                    "left_joints": {"start": 0, "end": 6, "original_key": "observation.state"},
                    "left_gripper": {"start": 6, "end": 7, "original_key": "observation.state"},
                    "right_joints": {"start": 7, "end": 13, "original_key": "observation.state"},
                    "right_gripper": {"start": 13, "end": 14, "original_key": "observation.state"},
                },
                "video": {
                    camera: {"original_key": f"observation.images.{camera}"}
                    for camera in CAMERAS
                },
                "annotation": {
                    "human.action.task_description": {"original_key": "task_index"}
                },
            },
        )
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "tasks": [instruction], "length": length})
            + "\n",
            encoding="utf-8",
        )
        (meta_dir / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": instruction}) + "\n",
            encoding="utf-8",
        )
        episode_stats = {
            "episode_index": 0,
            "stats": {
                "observation.state": _numeric_stats(joints),
                "action": _numeric_stats(joints),
            },
        }
        (meta_dir / "episodes_stats.jsonl").write_text(
            json.dumps(episode_stats) + "\n", encoding="utf-8"
        )
        _json(
            meta_dir / "conversion.json",
            {
                "source_hdf5": str(source),
                "instruction": instruction,
                "fps": fps,
                "state_source": "joint_action/vector",
                "action_source": "joint_action/vector",
                "camera_source_map": CAMERAS,
            },
        )
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    convert(args.source, args.output, args.instruction, args.fps)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
