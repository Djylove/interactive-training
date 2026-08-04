"""Run one simulator-backed XPolicyLab/RoboDojo experiment round."""

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
    parser.add_argument("--policy-env", required=True)
    parser.add_argument("--evaluation-env", required=True)
    parser.add_argument("--policy", default="demo_policy")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--checkpoint-name", default="sim-smoke")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--action-type", default="joint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--policy-gpu-id", default="0")
    parser.add_argument("--environment-gpu-id", default="0")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/xpolicylab-sim")
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = XPolicyExperimentRunner(
        args.xpolicylab_root,
        RunnerPolicy(
            allowed_policies=frozenset({args.policy}),
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            allow_empty_checkpoints_for=(
                frozenset({"demo_policy"})
                if args.policy == "demo_policy"
                else frozenset()
            ),
            max_timeout_seconds=args.timeout_seconds,
        ),
        log_root=output_dir / "process-logs",
    )
    spec = ExperimentSpec(
        experiment_id="xpolicy-robodojo-sim",
        round=0,
        policy=args.policy,
        checkpoint_name=args.checkpoint_name,
        bench_name="RoboDojo",
        env_cfg_type=args.env_cfg_type,
        action_type=args.action_type,
        seed=args.seed,
        train=TrainSpec(max_steps=1),
        evaluation=EvaluationSpec(
            environment="sim",
            tasks=args.task or ["stack_bowls"],
            repeats=args.repeats,
        ),
    )
    session = TrainingSession(
        goal=Goal(name="success", metric="success_rate", direction="max"),
        memory=str(output_dir / "memory.jsonl"),
    )

    def evaluate(spec, artifact, evaluation_id, checkpoint_id):
        return runner.evaluate_sim(
            spec,
            artifact,
            evaluation_id,
            checkpoint_id,
            policy_env=args.policy_env,
            evaluation_env=args.evaluation_env,
            policy_gpu_id=args.policy_gpu_id,
            environment_gpu_id=args.environment_gpu_id,
            timeout_seconds=args.timeout_seconds,
        )

    result = XPolicyExperiment(session, runner).run_round(spec, evaluate)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Simulator round completed: {result_path}")
    if result.evaluation is not None:
        print(
            f"Evaluation status: {result.evaluation.status}; "
            f"valid trials: {result.evaluation.valid_trials}"
        )


if __name__ == "__main__":
    main()
