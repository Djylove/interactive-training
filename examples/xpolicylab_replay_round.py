"""Validate an existing checkpoint against recorded GR3 episodes."""

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
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpolicylab-root", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--policy-env", required=True)
    parser.add_argument("--evaluation-env", required=True)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--task", default="recorded_gr3")
    parser.add_argument("--bench-name", default="RoboDojo")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=2)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--policy-gpu-id", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/xpolicylab-replay")
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = XPolicyExperimentRunner(
        args.xpolicylab_root,
        RunnerPolicy(
            allowed_policies=frozenset({args.policy}),
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            max_timeout_seconds=args.timeout_seconds,
        ),
        log_root=output_dir / "process-logs",
    )
    spec = ExperimentSpec(
        experiment_id="xpolicy-recorded-replay",
        round=0,
        policy=args.policy,
        checkpoint_name=args.checkpoint_name,
        bench_name=args.bench_name,
        env_cfg_type=args.env_cfg_type,
        action_type=args.action_type,
        seed=args.seed,
        train=TrainSpec(enabled=False),
        evaluation=EvaluationSpec(
            environment="replay",
            tasks=[args.task],
            repeats=len(args.episode),
        ),
    )
    session = TrainingSession(
        goal=Goal(name="replay gate", metric="replay_pass_rate", direction="max"),
        memory=str(output_dir / "memory.jsonl"),
    )

    def evaluate(spec, artifact, evaluation_id, checkpoint_id):
        return runner.evaluate_replay(
            spec,
            artifact,
            evaluation_id,
            checkpoint_id,
            episodes={args.task: args.episode},
            policy_env=args.policy_env,
            evaluation_env=args.evaluation_env,
            start=args.start,
            frame_count=args.frame_count,
            stride=args.stride,
            policy_gpu_id=args.policy_gpu_id,
            timeout_seconds=args.timeout_seconds,
        )

    result = XPolicyExperiment(session, runner).run_round(spec, evaluate)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Replay round completed: {result_path}")
    if result.evaluation is not None:
        print(
            f"Gate status: {result.evaluation.status}; "
            f"pass rate: {result.evaluation.metrics.get('replay_pass_rate')}"
        )
    print("Replay pass rate is contract evidence, not task success rate.")


if __name__ == "__main__":
    main()
