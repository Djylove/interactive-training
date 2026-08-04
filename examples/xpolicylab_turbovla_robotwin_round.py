"""Run a manifest-bound TurboVLA train/evaluate round through XPolicyLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from interactive_training import TrainingSession
from interactive_training.core import Goal
from interactive_training.integrations.xpolicylab import (
    EvaluationSpec,
    ExperimentSpec,
    RunnerPolicy,
    TrainSpec,
    XPolicyExperiment,
    XPolicyExperimentRunner,
    load_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpolicylab-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--turbovla-root", type=Path, required=True)
    parser.add_argument("--turbovla-python", type=Path, required=True)
    parser.add_argument("--robotwin-python", type=Path, required=True)
    parser.add_argument("--dinov3-path", type=Path, required=True)
    parser.add_argument("--bert-path", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-execution",
        choices=(
            "temporal_ensemble",
            "temporal_ensemble_oldest_binary",
            "open_loop_50",
        ),
        default="temporal_ensemble",
    )
    parser.add_argument(
        "--instruction-type",
        choices=("seen", "unseen"),
        default="unseen",
    )
    parser.add_argument(
        "--fixed-instruction",
        help="Optional deterministic instruction for a controlled evaluation gate",
    )
    parser.add_argument("--experiment-id", default="turbovla-robotwin-round")
    parser.add_argument("--checkpoint-name", default="beat-block-hammer-smoke")
    parser.add_argument("--reuse-checkpoint", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=24 * 60 * 60)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/turbovla-robotwin-round")
    )
    args = parser.parse_args()

    manifest = load_dataset_manifest(args.dataset_manifest)
    if manifest.profile_id != "turbovla.robotwin_clean_v1":
        parser.error("RoboTwin requires turbovla.robotwin_clean_v1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            allowed_env=frozenset(allowed_env),
            allowed_evaluation_env=frozenset(
                {
                    "TURBOVLA_ACTION_ENSEMBLE",
                    "TURBOVLA_BINARY_ACTION_SOURCE",
                    "TURBOVLA_EXEC_HORIZON",
                    "ROBOTWIN_INSTRUCTION_TYPE",
                    "ROBOTWIN_FIXED_INSTRUCTION",
                }
            ),
            train_env_map={
                "max_steps": "TURBOVLA_MAX_STEPS",
                "learning_rate": "TURBOVLA_LEARNING_RATE",
            },
            max_timeout_seconds=args.timeout_seconds,
        ),
        log_root=output_dir / "process-logs",
    )
    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        round=0,
        policy="TurboVLA",
        checkpoint_name=args.checkpoint_name,
        bench_name="RoboTwin",
        env_cfg_type="robotwin",
        action_type="bimanual14",
        seed=args.seed,
        dataset_manifest_id=manifest.dataset_id,
        train=TrainSpec(
            enabled=not args.reuse_checkpoint,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            env={
                "TURBOVLA_ROOT": str(args.turbovla_root.resolve()),
                "TURBOVLA_PYTHON": str(args.turbovla_python.expanduser().absolute()),
                "TURBOVLA_DINOV3_PATH": str(args.dinov3_path.resolve()),
                "TURBOVLA_BERT_PATH": str(args.bert_path.resolve()),
                "TURBOVLA_INIT_CHECKPOINT": str(args.pretrained_checkpoint.resolve()),
                "TURBOVLA_BATCH_SIZE": str(args.batch_size),
            },
        ),
        evaluation=EvaluationSpec(
            environment="sim",
            tasks=["beat_block_hammer"],
            repeats=1,
            env={
                "TURBOVLA_ACTION_ENSEMBLE": (
                    "false" if args.action_execution == "open_loop_50" else "true"
                ),
                "TURBOVLA_BINARY_ACTION_SOURCE": (
                    "oldest"
                    if args.action_execution == "temporal_ensemble_oldest_binary"
                    else "ensemble"
                ),
                "TURBOVLA_EXEC_HORIZON": (
                    "1"
                    if args.action_execution == "temporal_ensemble"
                    else "50"
                ),
                "ROBOTWIN_INSTRUCTION_TYPE": args.instruction_type,
                **(
                    {"ROBOTWIN_FIXED_INSTRUCTION": args.fixed_instruction}
                    if args.fixed_instruction
                    else {}
                ),
            },
        ),
    )
    session = TrainingSession(
        goal=Goal(name="robotwin_success", metric="success_rate", direction="max"),
        memory=str(output_dir / "memory.jsonl"),
    )

    def evaluate(spec, artifact, evaluation_id, checkpoint_id):
        return runner.evaluate_sim(
            spec,
            artifact,
            evaluation_id,
            checkpoint_id,
            policy_env=str(args.turbovla_python.expanduser().absolute()),
            evaluation_env=str(args.robotwin_python.expanduser().absolute()),
            policy_gpu_id=args.gpu_id,
            environment_gpu_id=args.gpu_id,
            timeout_seconds=args.timeout_seconds,
        )

    result = XPolicyExperiment(session, runner).run_round(
        spec,
        evaluate,
        gpu_ids=args.gpu_id,
        timeout_seconds=args.timeout_seconds,
        dataset_manifest_path=args.dataset_manifest,
    )
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"TurboVLA RoboTwin round: {result_path}")
    if result.artifact.status != "completed":
        raise SystemExit(1)
    if result.evaluation is None or result.evaluation.status != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
