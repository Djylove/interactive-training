"""No-GPU contract smoke test for the XPolicyLab integration.

This uses XPolicyLab's demo_policy training stub, then emits clearly labeled
synthetic trials. It validates orchestration and provenance only; it is not a model
evaluation and must not be used as benchmark evidence.
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
    TrialResult,
    XPolicyExperiment,
    XPolicyExperimentRunner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xpolicylab-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/xpolicylab-contract-smoke")
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = ExperimentSpec(
        experiment_id="xpolicy-contract-smoke",
        round=0,
        policy="demo_policy",
        checkpoint_name="contract-smoke",
        bench_name="RoboDojo",
        env_cfg_type="arx_x5",
        action_type="joint",
        seed=0,
        train=TrainSpec(max_steps=1),
        evaluation=EvaluationSpec(
            environment="debug",
            tasks=["synthetic_contract_task"],
            repeats=2,
        ),
    )
    runner = XPolicyExperimentRunner(
        args.xpolicylab_root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allow_empty_checkpoints_for=frozenset({"demo_policy"}),
            max_timeout_seconds=60,
        ),
        log_root=output_dir / "process-logs",
    )
    session = TrainingSession(
        goal=Goal(name="success", metric="success_rate", direction="max"),
        memory=str(output_dir / "memory.jsonl"),
    )

    def synthetic_evaluator(spec, artifact, evaluation_id, checkpoint_id):
        del artifact
        for repeat_index in range(spec.evaluation.repeats):
            success = float(repeat_index == 0)
            yield TrialResult(
                evaluation_id=evaluation_id,
                checkpoint_id=checkpoint_id,
                trial_id=f"synthetic-{repeat_index}",
                task=spec.evaluation.tasks[0],
                seed=spec.seed,
                repeat_index=repeat_index,
                outcome="success" if success else "failure",
                success=success,
                episode_steps=1,
                termination_reason="synthetic_contract_smoke",
                metrics={"inference_latency_ms": 1.0 + repeat_index},
            )

    result = XPolicyExperiment(session, runner).run_round(spec, synthetic_evaluator)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2) + "\n")
    print(f"Contract smoke completed: {result_path}")
    print("The trial results are synthetic and are not benchmark evidence.")


if __name__ == "__main__":
    main()
