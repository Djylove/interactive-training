"""Launch a manifest-bound TurboVLA GR3 training job through Interactive Training."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decode-threads", type=int, default=1)
    parser.add_argument("--batch-cache-size", type=int, default=2)
    parser.add_argument("--preload-batches", action="store_true")
    parser.add_argument("--normalization-json", type=Path)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=72 * 60 * 60)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--experiment-id", default="turbovla-gr3-anygrasp-train")
    parser.add_argument("--checkpoint-name", default="anygrasp-smoke")
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.gradient_accumulation_steps < 1
        or args.save_every < 1
        or args.timeout_seconds < 1
    ):
        parser.error("gradient accumulation, save interval, and timeout must be positive")

    manifest = load_dataset_manifest(args.dataset_manifest)
    allowed_env = {
        "TURBOVLA_ROOT",
        "TURBOVLA_PYTHON",
        "TURBOVLA_DINOV3_PATH",
        "TURBOVLA_BERT_PATH",
        "TURBOVLA_INIT_CHECKPOINT",
        "TURBOVLA_BATCH_SIZE",
        "TURBOVLA_IMAGE_SIZE",
        "TURBOVLA_HORIZON",
        "TURBOVLA_MAX_STEPS",
        "TURBOVLA_LEARNING_RATE",
        "TURBOVLA_NUM_WORKERS",
        "TURBOVLA_DECODE_THREADS",
        "TURBOVLA_BATCH_CACHE_SIZE",
        "TURBOVLA_PRELOAD_BATCHES",
        "TURBOVLA_NORMALIZATION_JSON",
        "TURBOVLA_GRAD_ACCUM_STEPS",
        "TURBOVLA_SAVE_EVERY",
    }
    runner = XPolicyExperimentRunner(
        args.xpolicylab_root,
        RunnerPolicy(
            allowed_policies=frozenset({"TurboVLA"}),
            allowed_scripts=frozenset({"train.sh"}),
            allowed_env=frozenset(allowed_env),
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
            max_timeout_seconds=args.timeout_seconds,
        ),
        log_root=args.log_root.resolve(),
    )
    train_env = {
        "TURBOVLA_ROOT": str(args.turbovla_root.resolve()),
        "TURBOVLA_PYTHON": str(args.turbovla_python.expanduser().absolute()),
        "TURBOVLA_DINOV3_PATH": str(args.dinov3_path.resolve()),
        "TURBOVLA_BERT_PATH": str(args.bert_path.resolve()),
        "TURBOVLA_BATCH_SIZE": str(args.batch_size),
        "TURBOVLA_IMAGE_SIZE": str(args.image_size),
        "TURBOVLA_HORIZON": str(args.horizon),
        "TURBOVLA_NUM_WORKERS": str(args.num_workers),
        "TURBOVLA_DECODE_THREADS": str(args.decode_threads),
        "TURBOVLA_BATCH_CACHE_SIZE": str(args.batch_cache_size),
        "TURBOVLA_PRELOAD_BATCHES": "1" if args.preload_batches else "0",
        "TURBOVLA_GRAD_ACCUM_STEPS": str(args.gradient_accumulation_steps),
        "TURBOVLA_SAVE_EVERY": str(args.save_every),
    }
    if args.init_checkpoint is not None:
        train_env["TURBOVLA_INIT_CHECKPOINT"] = str(args.init_checkpoint.resolve())
    if args.normalization_json is not None:
        train_env["TURBOVLA_NORMALIZATION_JSON"] = str(
            args.normalization_json.resolve()
        )
    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        round=0,
        policy="TurboVLA",
        checkpoint_name=args.checkpoint_name,
        bench_name="GR3",
        env_cfg_type="gr3",
        action_type="model33_canonical37",
        seed=0,
        mode="remote_train",
        dataset_manifest_id=manifest.dataset_id,
        train=TrainSpec(
            enabled=True,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            env=train_env,
        ),
        evaluation=EvaluationSpec(environment="replay", tasks=["not_run_in_train_only"]),
    )
    artifact = runner.train(
        spec,
        gpu_ids=args.gpu_id,
        dataset_manifest_path=args.dataset_manifest,
    )
    print(artifact.model_dump_json(indent=2))
    if artifact.status != "completed":
        raise RuntimeError(f"training failed: {artifact.error}; logs={artifact.logs}")


if __name__ == "__main__":
    main()
