"""Launch a manifest-bound TurboVLA RoboTwin baseline through Interactive Training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interactive_training.integrations.xpolicylab import (
    EvaluationSpec,
    ExperimentSpec,
    RunnerPolicy,
    TrainSpec,
    XPolicyExperimentRunner,
    load_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpolicylab-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--turbovla-root", type=Path, required=True)
    parser.add_argument("--turbovla-python", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--experiment-id", default="turbovla-robotwin-train-smoke")
    parser.add_argument("--checkpoint-name", default="beat-block-hammer-smoke")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/turbovla-robotwin-train")
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.dataset_manifest)
    if manifest.profile_id != "turbovla.robotwin_clean_v1":
        parser.error("RoboTwin training requires turbovla.robotwin_clean_v1")
    turbovla_python = str(args.turbovla_python.expanduser().absolute())
    allowed_env = {
        "TURBOVLA_ROOT",
        "TURBOVLA_PYTHON",
        "TURBOVLA_DINOV3_PATH",
        "TURBOVLA_BERT_PATH",
        "TURBOVLA_INIT_CHECKPOINT",
        "TURBOVLA_BATCH_SIZE",
        "TURBOVLA_MAX_STEPS",
        "TURBOVLA_LEARNING_RATE",
    }
    runner = XPolicyExperimentRunner(
        args.xpolicylab_root,
        RunnerPolicy(
            allowed_policies=frozenset({"TurboVLA"}),
            allowed_scripts=frozenset({"train.sh"}),
            allowed_env=frozenset(allowed_env),
            train_env_map={
                "max_steps": "TURBOVLA_MAX_STEPS",
                "learning_rate": "TURBOVLA_LEARNING_RATE",
            },
            max_timeout_seconds=24 * 60 * 60,
        ),
        log_root=args.output_dir.resolve() / "process-logs",
    )
    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        round=0,
        policy="TurboVLA",
        checkpoint_name=args.checkpoint_name,
        bench_name="RoboTwin",
        env_cfg_type="robotwin",
        action_type="bimanual14",
        seed=0,
        dataset_manifest_id=manifest.dataset_id,
        train=TrainSpec(
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            env={
                "TURBOVLA_ROOT": str(args.turbovla_root.resolve()),
                "TURBOVLA_PYTHON": turbovla_python,
                "TURBOVLA_DINOV3_PATH": str(args.dinov3_path.resolve()),
                "TURBOVLA_BERT_PATH": str(args.bert_path.resolve()),
                "TURBOVLA_INIT_CHECKPOINT": str(
                    args.pretrained_checkpoint.resolve()
                ),
                "TURBOVLA_BATCH_SIZE": str(args.batch_size),
            },
        ),
        # Training-only entry point. Simulation is a separate audited round.
        evaluation=EvaluationSpec(
            environment="sim", tasks=["beat_block_hammer"], repeats=1
        ),
    )
    validated = runner.validate_training_dataset(spec, args.dataset_manifest)
    assert validated is not None
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "dataset_id": validated.dataset_id,
                    "profile_id": validated.profile_id,
                    "tasks": [episode.task_id for episode in validated.episodes],
                    "trajectory_count": validated.summary.statistics.get(
                        "trajectory_count"
                    ),
                    "training_launched": False,
                },
                indent=2,
            )
        )
        return

    artifact = runner.train(
        spec,
        gpu_ids=args.gpu_id,
        dataset_manifest_path=args.dataset_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "training-artifact.json"
    result_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"TurboVLA RoboTwin training artifact: {result_path.resolve()}")
    if artifact.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
