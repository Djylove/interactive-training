"""Run XPolicyLab's real no-simulator debug protocol through the integration.

The debug environment validates observation/action wiring and always reports invalid
benchmark outcomes. A successful run is therefore expected to be inconclusive.
"""

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
    parser.add_argument(
        "--policy-env", required=True, help="XPolicyLab policy conda env"
    )
    parser.add_argument(
        "--evaluation-env", required=True, help="debug client conda env"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/xpolicylab-debug")
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = XPolicyExperimentRunner(
        args.xpolicylab_root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            allow_empty_checkpoints_for=frozenset({"demo_policy"}),
            max_timeout_seconds=1200,
        ),
        log_root=output_dir / "process-logs",
    )
    spec = ExperimentSpec(
        experiment_id="xpolicy-real-debug",
        round=0,
        policy="demo_policy",
        checkpoint_name="real-debug",
        bench_name="RoboDojo",
        env_cfg_type="arx_x5",
        action_type="joint",
        seed=0,
        train=TrainSpec(max_steps=1),
        evaluation=EvaluationSpec(
            environment="debug",
            tasks=["stack_bowls"],
            repeats=args.repeats,
        ),
    )
    session = TrainingSession(
        goal=Goal(name="success", metric="success_rate", direction="max"),
        memory=str(output_dir / "memory.jsonl"),
    )

    def evaluate(spec, artifact, evaluation_id, checkpoint_id):
        return runner.evaluate_debug(
            spec,
            artifact,
            evaluation_id,
            checkpoint_id,
            policy_env=args.policy_env,
            evaluation_env=args.evaluation_env,
        )

    result = XPolicyExperiment(session, runner).run_round(spec, evaluate)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")
    print(f"Debug round completed: {result_path}")
    if result.evaluation is not None:
        print(f"Evaluation status: {result.evaluation.status}")
    print("Debug outcomes are protocol evidence, not benchmark scores.")


if __name__ == "__main__":
    main()
