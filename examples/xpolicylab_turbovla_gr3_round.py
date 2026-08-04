"""Run a TurboVLA GR3 train -> recorded-replay round under Interactive Training."""

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
    parser.add_argument("--dinov3-path", type=Path)
    parser.add_argument("--bert-path", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--episode", type=Path, action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--replay-start", type=int, default=0)
    parser.add_argument("--replay-frame-count", type=int, default=2)
    parser.add_argument("--replay-stride", type=int, default=100)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="Register and replay an existing XPolicyLab checkpoint without training.",
    )
    parser.add_argument("--experiment-id", default="turbovla-gr3-smoke")
    parser.add_argument("--checkpoint-name", default="dagger-smoke")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/turbovla-gr3"))
    args = parser.parse_args()

    # Do not resolve the Python executable: venv/bin/python is commonly a
    # symlink, and resolving it bypasses the virtual environment at runtime.
    turbovla_python = str(args.turbovla_python.expanduser().absolute())

    manifest = load_dataset_manifest(args.dataset_manifest)
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
                    "TURBOVLA_ROOT",
                    "TURBOVLA_PYTHON",
                    "TURBOVLA_DINOV3_PATH",
                    "TURBOVLA_BERT_PATH",
                }
            ),
            inherited_env=frozenset(
                {
                    "PATH",
                    "HOME",
                    "LANG",
                    "LC_ALL",
                    "TMPDIR",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "LIBRARY_PATH",
                    "CUDA_HOME",
                    "CUDA_PATH",
                    "CUDA_TOOLKIT_ROOT",
                    "PPU_SDK",
                    "TORCH_HOME",
                }
            ),
            train_env_map={
                "max_steps": "TURBOVLA_MAX_STEPS",
                "learning_rate": "TURBOVLA_LEARNING_RATE",
            },
            max_timeout_seconds=24 * 60 * 60,
        ),
        log_root=args.output_dir.resolve() / "process-logs",
    )
    train_env = {
        "TURBOVLA_ROOT": str(args.turbovla_root.resolve()),
        "TURBOVLA_PYTHON": turbovla_python,
        "TURBOVLA_BATCH_SIZE": str(args.batch_size),
    }
    if args.dinov3_path is not None:
        train_env["TURBOVLA_DINOV3_PATH"] = str(args.dinov3_path.resolve())
    if args.bert_path is not None:
        train_env["TURBOVLA_BERT_PATH"] = str(args.bert_path.resolve())
    if args.init_checkpoint is not None:
        train_env["TURBOVLA_INIT_CHECKPOINT"] = str(args.init_checkpoint.resolve())
    evaluation_env = {
        "TURBOVLA_ROOT": str(args.turbovla_root.resolve()),
        "TURBOVLA_PYTHON": turbovla_python,
    }
    if args.dinov3_path is not None:
        evaluation_env["TURBOVLA_DINOV3_PATH"] = str(args.dinov3_path.resolve())
    if args.bert_path is not None:
        evaluation_env["TURBOVLA_BERT_PATH"] = str(args.bert_path.resolve())
    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        round=0,
        policy="TurboVLA",
        checkpoint_name=args.checkpoint_name,
        bench_name="GR3",
        env_cfg_type="gr3",
        action_type="canonical37",
        seed=0,
        dataset_manifest_id=manifest.dataset_id,
        train=TrainSpec(
            enabled=not args.reuse_checkpoint,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            env=train_env,
        ),
        evaluation=EvaluationSpec(
            environment="replay",
            tasks=["recorded_gr3"],
            repeats=max(1, len(args.episode)),
            env=evaluation_env,
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
                    "eligible_episodes": validated.summary.eligible_episode_count,
                    "max_steps": spec.train.max_steps,
                    "learning_rate": spec.train.learning_rate,
                    "training_launched": False,
                },
                indent=2,
            )
        )
        return
    if not args.reuse_checkpoint and (
        args.dinov3_path is None or args.bert_path is None
    ):
        parser.error("training requires --dinov3-path and --bert-path")
    if not args.episode:
        parser.error("train+replay requires at least one --episode")

    session = TrainingSession(
        goal=Goal(
            name="TurboVLA GR3 replay gate", metric="replay_pass_rate", direction="max"
        ),
        memory=str(args.output_dir.resolve() / "memory.jsonl"),
    )

    def evaluate(spec, artifact, evaluation_id, checkpoint_id):
        return runner.evaluate_replay(
            spec,
            artifact,
            evaluation_id,
            checkpoint_id,
            episodes={"recorded_gr3": args.episode},
            policy_env=turbovla_python,
            evaluation_env=turbovla_python,
            start=args.replay_start,
            frame_count=args.replay_frame_count,
            stride=args.replay_stride,
            policy_gpu_id=args.gpu_id,
        )

    result = XPolicyExperiment(session, runner).run_round(
        spec,
        evaluate,
        gpu_ids=args.gpu_id,
        dataset_manifest_path=args.dataset_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"TurboVLA GR3 round: {result_path.resolve()}")


if __name__ == "__main__":
    main()
