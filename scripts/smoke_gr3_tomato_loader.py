#!/usr/bin/env python3
"""Exercise the real TurboVLA GR3 loader and freeze train-only statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from turbovla.data.gr3_anygrasp import Gr3AnygraspDataset


def _inspect(dataset: Gr3AnygraspDataset) -> list[dict[str, object]]:
    indices = sorted({0, len(dataset) // 2, len(dataset) - 1})
    observations = []
    for index in indices:
        item = dataset[index]
        expected = {
            "image": (dataset.image_size, dataset.image_size, 3),
            "state": (31,),
            "action": (dataset.horizon, 31),
            "action_mask": (dataset.horizon,),
        }
        for key, shape in expected.items():
            value = np.asarray(item[key])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {key} at sample {index}: {value.shape}")
        observations.append(
            {
                "sample_index": index,
                "metadata": item["metadata"],
                "image_shape": list(item["image"].shape),
                "state_shape": list(item["state"].shape),
                "action_shape": list(item["action"].shape),
                "valid_action_steps": int(item["action_mask"].sum()),
                "prompt": item["lang"],
            }
        )
    return observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--normalization-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    for output in (args.normalization_output, args.report_output):
        if output.exists():
            parser.error(f"refusing to overwrite output: {output}")

    train = Gr3AnygraspDataset(
        args.train_manifest,
        horizon=args.horizon,
        image_size=args.image_size,
        batch_cache_size=1,
        decode_threads=1,
        model_state_dim=31,
        model_action_dim=31,
    )
    train.preload_batches()
    validation = Gr3AnygraspDataset(
        args.validation_manifest,
        horizon=args.horizon,
        image_size=args.image_size,
        stats=train.stats,
        batch_cache_size=1,
        decode_threads=1,
        model_state_dim=31,
        model_action_dim=31,
    )
    validation.preload_batches()
    # Source-session isolation is asserted from the immutable manifest profile.
    train_profile = train.manifest["episodes"][0]["profile_data"]
    validation_profile = validation.manifest["episodes"][0]["profile_data"]
    train_source_sessions = set(train_profile["selected_source_sessions"])
    validation_source_sessions = set(validation_profile["selected_source_sessions"])
    if train_source_sessions & validation_source_sessions:
        raise ValueError("train and validation source sessions overlap")

    dataset_id = train.manifest["dataset_id"]
    normalization = {
        "schema_version": "gr3_normalization.v1",
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_split": "train",
        "fit_sample_count": len(train),
        "normalization": train.stats.to_dict(),
        "diagnostics": train.normalization_diagnostics(),
    }
    args.normalization_output.parent.mkdir(parents=True, exist_ok=True)
    args.normalization_output.write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    normalization_sha256 = hashlib.sha256(args.normalization_output.read_bytes()).hexdigest()
    report = {
        "schema_version": "gr3_tomato_loader_smoke.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "model_state_dim": 31,
        "model_action_dim": 31,
        "horizon": args.horizon,
        "train": {
            "dataset_id": dataset_id,
            "samples": len(train),
            "source_sessions": sorted(train_source_sessions),
            "observations": _inspect(train),
        },
        "validation": {
            "dataset_id": validation.manifest["dataset_id"],
            "samples": len(validation),
            "source_sessions": sorted(validation_source_sessions),
            "observations": _inspect(validation),
        },
        "normalization": {
            "path": str(args.normalization_output.resolve()),
            "sha256": normalization_sha256,
            "fit_split": "train_only",
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
