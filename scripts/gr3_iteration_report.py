"""Create an auditable promotion report from staged GR3 offline evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise ValueError(f"evaluation did not pass: {path}")
    return report


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(raw_path)


def _improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-candidate", action="append", type=_named_path, required=True)
    parser.add_argument("--milestone", action="append", type=_named_path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    screens = {name: _load(path) for name, path in args.screen_candidate}
    milestones = {name: _load(path) for name, path in args.milestone}
    if len(screens) != len(args.screen_candidate) or len(milestones) != len(args.milestone):
        parser.error("candidate and milestone names must be unique")
    validation_reports = [*screens.values(), *milestones.values()]
    validation_dataset_ids = {report["dataset_id"] for report in validation_reports}
    validation_index_digests = {
        report["evaluation_index_sha256"] for report in validation_reports
    }
    if len(validation_dataset_ids) != 1 or len(validation_index_digests) != 1:
        raise ValueError("validation reports do not use the same dataset and sample indices")

    screen_winner = min(screens, key=lambda name: screens[name]["normalized_l1"])
    promoted_milestone = min(
        milestones,
        key=lambda name: milestones[name]["normalized_l1"],
    )
    screen_l1 = float(screens[screen_winner]["normalized_l1"])
    promoted_l1 = float(milestones[promoted_milestone]["normalized_l1"])
    heldout = _load(args.heldout)
    heldout_l1 = float(heldout["normalized_l1"])
    promoted = milestones[promoted_milestone]
    payload = {
        "schema_version": "gr3_anygrasp_iteration.v1",
        "status": "promoted",
        "selection_metric": "normalized_l1_min",
        "validation_dataset_id": next(iter(validation_dataset_ids)),
        "validation_index_sha256": next(iter(validation_index_digests)),
        "screen_candidates": {
            name: float(report["normalized_l1"]) for name, report in screens.items()
        },
        "screen_winner": screen_winner,
        "milestones": {
            name: float(report["normalized_l1"]) for name, report in milestones.items()
        },
        "promoted_milestone": promoted_milestone,
        "promoted_checkpoint": promoted["checkpoint"],
        "promoted_model_state_checkpoint": promoted.get("model_state_checkpoint"),
        "validation_improvement_from_screen_percent": _improvement(screen_l1, promoted_l1),
        "sealed_heldout": {
            "dataset_id": heldout["dataset_id"],
            "evaluation_index_sha256": heldout["evaluation_index_sha256"],
            "normalized_l1": heldout_l1,
            "generalization_gap_percent": 100.0 * (heldout_l1 - promoted_l1) / promoted_l1,
            "used_for_selection": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
