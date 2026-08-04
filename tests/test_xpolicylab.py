import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from interactive_training.core import Goal, TrainingSession
from interactive_training.integrations.xpolicylab import (
    ControlPolicy,
    DatasetManifest,
    EvaluationSpec,
    ExperimentSpec,
    PromotionStage,
    RunnerPolicy,
    TrainSpec,
    TrialAggregator,
    TrialResult,
    XPolicyExperiment,
    XPolicyExperimentRunner,
    load_dataset_manifest,
    validate_gr3_anygrasp_manifest,
    validate_gr3_dagger_manifest,
    validate_robotwin_clean_manifest,
)


def _spec(**updates):
    data = {
        "experiment_id": "smoke-001",
        "round": 0,
        "policy": "demo_policy",
        "checkpoint_name": "tiny",
        "bench_name": "RoboDojo",
        "env_cfg_type": "arx_x5",
        "action_type": "joint",
        "seed": 0,
        "train": TrainSpec(max_steps=2),
        "evaluation": EvaluationSpec(
            environment="debug", tasks=["stack_bowls"], repeats=2
        ),
    }
    data.update(updates)
    return ExperimentSpec(**data)


def _trial(
    evaluation_id,
    checkpoint_id,
    suffix,
    *,
    outcome,
    success,
    task="stack_bowls",
    **metrics,
):
    return TrialResult(
        evaluation_id=evaluation_id,
        checkpoint_id=checkpoint_id,
        trial_id=f"trial-{suffix}",
        task=task,
        seed=0,
        repeat_index=int(suffix),
        outcome=outcome,
        success=success,
        episode_steps=10 + int(suffix),
        metrics=metrics,
    )


def _fake_xpolicy_repo(tmp_path: Path, *, exit_code=0, write_checkpoint=True) -> Path:
    root = tmp_path / "XPolicyLab"
    policy = root / "policy" / "demo_policy"
    policy.mkdir(parents=True)
    script = policy / "train.sh"
    checkpoint_write = (
        'printf "fake checkpoint\\n" > "$checkpoint/model.bin"\n'
        if write_checkpoint
        else ""
    )
    script.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        f"if [ {exit_code} -ne 0 ]; then exit {exit_code}; fi\n"
        'checkpoint="checkpoints/$1-$2-$3-$4-$5"\n'
        'mkdir -p "$checkpoint"\n'
        + checkpoint_write
        + 'if [ -n "${XPOLICYLAB_DATASET_ID:-}" ]; then\n'
        + '  printf "%s\\n" "$XPOLICYLAB_DATASET_ID" > '
        '"$checkpoint/dataset-id.txt"\n' + "fi\n"
    )
    (policy / "eval.sh").write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        'task="$2"\n'
        "index=0\n"
        'while [ "$index" -lt "$XPOLICYLAB_EVAL_EPISODES" ]; do\n'
        '  if [ "$EVAL_ENV_TYPE" = "replay" ]; then\n'
        '    outcome_fields=\'"outcome":"success","success":1.0\'\n'
        '    evidence_field=\'"evidence_type":"replay_validation",\'\n'
        '  elif [ "$EVAL_ENV_TYPE" = "sim" ] && [ $((index % 2)) -eq 0 ]; then\n'
        '    outcome_fields=\'"outcome":"success","success":1.0\'\n'
        "    evidence_field=''\n"
        '  elif [ "$EVAL_ENV_TYPE" = "sim" ]; then\n'
        '    outcome_fields=\'"outcome":"failure","success":0.0\'\n'
        "    evidence_field=''\n"
        "  else\n"
        '    outcome_fields=\'"outcome":"invalid"\'\n'
        '    evidence_field=\'"evidence_type":"protocol_debug",\'\n'
        "  fi\n"
        "  printf "
        '\'{"schema_version":"xpolicy_interactive.v1",'
        '"evaluation_id":"%s","checkpoint_id":"%s",'
        '"trial_id":"%s-repeat-%s","task":"%s",'
        '"seed":0,"repeat_index":%s,%s%s,'
        '"episode_steps":5,"termination_reason":"debug_protocol_complete",'
        '"metrics":{"inference_latency_ms":1.5},"artifacts":{}}\\n\' '
        '"$XPOLICYLAB_EVALUATION_ID" "$XPOLICYLAB_CHECKPOINT_ID" '
        '"${XPOLICYLAB_TRIAL_ID:-$XPOLICYLAB_EVALUATION_ID-$task}" '
        '"$index" "$task" "$index" "$evidence_field" "$outcome_fields" '
        '>> "$XPOLICYLAB_TRIAL_RESULT_JSONL"\n'
        "  index=$((index + 1))\n"
        "done\n"
    )
    return root


def _dataset_manifest_payload(profile_id="vendor.robot_dataset_v1"):
    digest = "a" * 64
    episode = {
        "episode_id": "episode-1",
        "path": "/recordings/episode-1",
        "source_schema": "vendor_robot_v1",
        "profile_id": profile_id,
        "file_manifest_sha256": digest,
        "episode_manifest_sha256": "b" * 64,
        "files": [{"path": "metadata.json", "size": 2, "sha256": digest}],
        "task_id": "pick",
        "task_instruction": "Pick the object",
        "task_outcome": "success",
        "outcome_confirmed_by_operator": True,
        "termination_reason": "operator_finish",
        "recording_saved_successfully": True,
        "provenance": {"recorder": "vendor"},
        "statistics": {"sample_count": 10},
        "profile_data": {"vendor_axis_count": 7},
        "filters": [],
        "warnings": [],
        "exclusion_reasons": [],
        "requires_filtering": False,
        "train_eligible_after_filters": True,
    }
    return {
        "schema_version": "xpolicy_dataset.v1",
        "dataset_id": "dataset-1",
        "dataset_name": "dataset",
        "dataset_sha256": digest,
        "source_root": "/recordings",
        "created_at": "2026-07-31T00:00:00Z",
        "profile_id": profile_id,
        "source_revisions": {"recorder_commit": None},
        "profile_config": {},
        "summary": {
            "episode_count": 1,
            "eligible_episode_count": 1,
            "excluded_episode_count": 0,
            "requires_filtering_episode_count": 0,
            "task_outcome_counts": {"success": 1},
            "statistics": {"sample_count": 10},
        },
        "episodes": [episode],
    }


def _gr3_dataset_manifest_payload():
    payload = _dataset_manifest_payload("xpolicylab.gr3_dagger_v2")
    episode = payload["episodes"][0]
    episode["source_schema"] = "gr3_dagger_v2"
    episode["statistics"] = {
        "camera_frames": 12,
        "trainable_camera_frames": 10,
        "valid_label_rows": 8,
        "intervention_count": 1,
    }
    episode["profile_data"] = {
        "state_dim": 33,
        "action_dim": 37,
        "camera_frames": 12,
        "trainable_camera_frames": 10,
        "trailing_audit_frames": 2,
        "valid_label_rows": 8,
        "label_rows": 9,
        "selected_action_source_counts": {"EXPERT": 6, "MODEL": 3},
        "control_mode_counts": {"EXPERT": 6, "MODEL": 3},
        "alignment": {
            "limit_ms": 20.0,
            "canonical_state_max_ms_after_filter": 4.0,
            "expert_max_ms_after_filter": 5.0,
            "raw_camera_state_max_ms": 1000.0,
        },
    }
    payload["summary"]["statistics"] = {
        "camera_frames": 12,
        "trainable_camera_frames": 10,
        "valid_label_rows": 8,
        "intervention_count": 1,
    }
    return payload


def _robotwin_dataset_manifest_payload():
    payload = _dataset_manifest_payload("turbovla.robotwin_clean_v1")
    episode = payload["episodes"][0]
    episode["source_schema"] = "lerobot_v2_robotwin_clean"
    episode["task_id"] = "beat_block_hammer"
    episode["statistics"] = {
        "trajectory_count": 50,
        "frame_count": 5682,
        "video_count": 150,
    }
    episode["profile_data"] = {
        "task_name": "beat_block_hammer",
        "robot_type": "aloha",
        "lerobot_version": "v2.1",
        "state_dim": 14,
        "action_dim": 14,
        "horizon": 50,
        "fps": 15,
        "trajectory_count": 50,
        "frame_count": 5682,
        "video_count": 150,
        "camera_keys": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ],
    }
    payload["summary"]["statistics"] = dict(episode["statistics"])
    return payload


def _gr3_anygrasp_dataset_manifest_payload():
    payload = _dataset_manifest_payload("xpolicylab.gr3_anygrasp_lerobot_v3")
    episode = payload["episodes"][0]
    episode["source_schema"] = "lerobot_v3_gr3qnexo_top"
    episode["statistics"] = {"clips": 32, "frames": 2592, "batches": 14}
    episode["profile_data"] = {
        "robot_type": "gr3qnexo",
        "state_dim": 33,
        "action_dim": 37,
        "fps": 30,
        "camera_key": "observation.images.top",
        "camera_views": 1,
        "dataset_root": "/mnt/workspace/jmy/dataset/anygrasp_v2",
        "clips_file": "Data/outputs/selection/manifest/clips.parquet",
        "clip_limit": 32,
        "selected_frames": 2592,
        "batch_count": 14,
        "raw_media_access": "read_only_in_place",
    }
    payload["summary"]["statistics"] = dict(episode["statistics"])
    return payload


def _refresh_dataset_digests(payload):
    def digest(value):
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    identities = []
    for episode in payload["episodes"]:
        episode["file_manifest_sha256"] = digest(episode["files"])
        episode_payload = dict(episode)
        episode_payload.pop("episode_manifest_sha256", None)
        episode["episode_manifest_sha256"] = digest(episode_payload)
        identities.append(
            {
                "episode_id": episode["episode_id"],
                "episode_manifest_sha256": episode["episode_manifest_sha256"],
            }
        )
    payload["dataset_sha256"] = digest(identities)


def test_dataset_manifest_core_is_robot_agnostic():
    manifest = DatasetManifest.model_validate(_dataset_manifest_payload())
    assert manifest.episodes[0].profile_data == {"vendor_axis_count": 7}
    with pytest.raises(ValueError, match="expected profile_id"):
        validate_gr3_dagger_manifest(manifest)


def test_gr3_profile_validation_is_explicit_and_robot_specific():
    payload = _gr3_dataset_manifest_payload()
    episode = payload["episodes"][0]
    manifest = DatasetManifest.model_validate(payload)
    assert validate_gr3_dagger_manifest(manifest) is manifest

    episode["profile_data"]["action_dim"] = 36
    invalid = DatasetManifest.model_validate(payload)
    with pytest.raises(ValidationError):
        validate_gr3_dagger_manifest(invalid)


def test_robotwin_profile_validation_is_explicit_and_separate_from_gr3():
    payload = _robotwin_dataset_manifest_payload()
    manifest = DatasetManifest.model_validate(payload)
    assert validate_robotwin_clean_manifest(manifest) is manifest

    payload["episodes"][0]["profile_data"]["camera_keys"].reverse()
    invalid = DatasetManifest.model_validate(payload)
    with pytest.raises(ValidationError):
        validate_robotwin_clean_manifest(invalid)


def test_gr3_anygrasp_profile_uses_one_top_camera_without_dagger_semantics():
    payload = _gr3_anygrasp_dataset_manifest_payload()
    manifest = DatasetManifest.model_validate(payload)
    assert validate_gr3_anygrasp_manifest(manifest) is manifest

    payload["episodes"][0]["profile_data"]["camera_views"] = 3
    invalid = DatasetManifest.model_validate(payload)
    with pytest.raises(ValidationError):
        validate_gr3_anygrasp_manifest(invalid)


def test_runner_binds_an_explicit_validated_dataset_snapshot(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(allowed_policies=frozenset({"demo_policy"})),
        log_root=tmp_path / "logs",
    )
    payload = _gr3_dataset_manifest_payload()
    episode_dir = tmp_path / "recordings" / "episode-1"
    episode_dir.mkdir(parents=True)
    metadata = episode_dir / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    episode = payload["episodes"][0]
    episode["path"] = str(episode_dir)
    episode["files"] = [
        {
            "path": "metadata.json",
            "size": metadata.stat().st_size,
            "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        }
    ]
    _refresh_dataset_digests(payload)
    manifest_path = tmp_path / "gr3-dataset-manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    spec = _spec(dataset_manifest_id=payload["dataset_id"])

    with pytest.raises(ValueError, match="requires dataset_manifest_path"):
        runner.train(spec)
    with pytest.raises(ValueError, match="does not match"):
        runner.train(
            _spec(dataset_manifest_id="different-dataset"),
            dataset_manifest_path=manifest_path,
        )

    artifact = runner.train(spec, dataset_manifest_path=manifest_path)
    assert artifact.status == "completed"
    assert artifact.dataset is not None
    assert artifact.dataset.dataset_id == payload["dataset_id"]
    snapshot = Path(artifact.dataset.manifest_path)
    assert snapshot.is_file() and snapshot in map(Path, artifact.logs)
    assert artifact.checkpoint is not None
    bound_id = Path(artifact.checkpoint.path, "dataset-id.txt").read_text().strip()
    assert bound_id == payload["dataset_id"]


def test_dataset_loader_rejects_manifest_content_tampering(tmp_path):
    payload = _gr3_dataset_manifest_payload()
    _refresh_dataset_digests(payload)
    payload["episodes"][0]["warnings"].append("changed_after_digest")
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest digest mismatch"):
        load_dataset_manifest(path)


def test_contract_rejects_unknown_version_and_path_identifiers():
    with pytest.raises(ValidationError):
        _spec(schema_version="xpolicy_interactive.v2")
    with pytest.raises(ValidationError):
        _spec(policy="../unsafe")
    with pytest.raises(ValidationError):
        _spec(
            evaluation=EvaluationSpec(environment="sim", tasks=["../unsafe"], repeats=1)
        )
    with pytest.raises(ValidationError):
        TrialResult(
            evaluation_id="e",
            checkpoint_id="c",
            trial_id="t",
            task="task",
            seed=0,
            repeat_index=0,
            outcome="success",
            success=0.0,
        )


def test_trial_aggregation_excludes_invalid_outcomes_and_reports_tasks():
    aggregator = TrialAggregator("eval", "checkpoint")
    aggregator.add(
        _trial(
            "eval",
            "checkpoint",
            "0",
            outcome="success",
            success=1.0,
            inference_latency_ms=10.0,
            takeover_count=0,
        )
    )
    aggregator.add(
        _trial(
            "eval",
            "checkpoint",
            "1",
            outcome="failure",
            success=0.0,
            inference_latency_ms=20.0,
            takeover_count=1,
        )
    )
    aggregator.add(
        TrialResult(
            evaluation_id="eval",
            checkpoint_id="checkpoint",
            trial_id="trial-timeout",
            task="stack_bowls",
            seed=0,
            repeat_index=2,
            outcome="timeout",
        )
    )
    summary = aggregator.summarize()
    assert summary.total_trials == 3
    assert summary.valid_trials == 2 and summary.invalid_trials == 1
    assert summary.status == "completed"
    assert summary.metrics["success_rate"] == 0.5
    assert summary.metrics["inference_latency_p95_ms"] == 19.5
    assert summary.task_success_rates == {"stack_bowls": 0.5}
    assert summary.session_metrics()["task_success_rate/stack_bowls"] == 0.5
    with pytest.raises(ValueError, match="duplicate"):
        aggregator.add(
            _trial("eval", "checkpoint", "0", outcome="success", success=1.0)
        )


def test_inconclusive_evaluation_does_not_publish_success_score():
    aggregator = TrialAggregator("eval", "checkpoint", minimum_valid_trials=2)
    aggregator.add(_trial("eval", "checkpoint", "0", outcome="success", success=1.0))
    summary = aggregator.summarize()
    assert summary.status == "inconclusive"
    assert summary.metrics["success_rate"] == 1.0  # retained as partial evidence
    assert "success_rate" not in summary.session_metrics()
    assert summary.session_metrics()["evaluation_conclusive"] == 0.0


def test_replay_validation_reports_gate_rate_without_task_success():
    aggregator = TrialAggregator("replay-eval", "checkpoint", minimum_valid_trials=2)
    for index, passed in enumerate((True, False)):
        aggregator.add(
            TrialResult(
                evaluation_id="replay-eval",
                checkpoint_id="checkpoint",
                trial_id=f"replay-{index}",
                task="recorded_gr3",
                seed=0,
                repeat_index=index,
                evidence_type="replay_validation",
                outcome="success" if passed else "failure",
                success=1.0 if passed else 0.0,
                metrics={"inference_latency_ms": 10.0 + index},
            )
        )
    summary = aggregator.summarize()
    assert summary.status == "completed"
    assert summary.evidence_type == "replay_validation"
    assert summary.metrics["replay_pass_rate"] == 0.5
    assert "success_rate" not in summary.metrics
    assert summary.task_success_rates == {}
    assert summary.session_metrics()["evaluation_conclusive"] == 1.0


def test_aggregation_rejects_mixed_evidence_semantics():
    aggregator = TrialAggregator("eval", "checkpoint")
    aggregator.add(_trial("eval", "checkpoint", "0", outcome="success", success=1.0))
    with pytest.raises(ValueError, match="mix"):
        aggregator.add(
            TrialResult(
                evaluation_id="eval",
                checkpoint_id="checkpoint",
                trial_id="replay-1",
                task="recorded_gr3",
                seed=0,
                repeat_index=1,
                evidence_type="replay_validation",
                outcome="success",
                success=1.0,
            )
        )


def test_control_policy_blocks_agent_shadow_and_all_safety_changes():
    policy = ControlPolicy(
        training_knobs={"learning_rate"},
        shadow_knobs={"filter_alpha"},
        safety_knobs={"joint_limit"},
    )
    assert policy.authorize_knob("learning_rate", "agent:planner", "sim")
    assert not policy.authorize_knob("filter_alpha", "agent:planner", "robot_shadow")
    assert policy.authorize_knob("filter_alpha", "human:web", "robot_shadow")
    assert not policy.authorize_knob("joint_limit", "human:web", "debug")
    assert ControlPolicy.can_promote(
        PromotionStage.TRAINED, PromotionStage.OFFLINE_VALIDATED
    )
    assert not ControlPolicy.can_promote(
        PromotionStage.TRAINED, PromotionStage.SIMULATOR_APPROVED
    )


def test_runner_rejects_unlisted_policy_and_environment(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(allowed_policies=frozenset({"demo_policy"})),
        log_root=tmp_path / "logs",
    )
    with pytest.raises(PermissionError, match="policy"):
        runner.train(_spec(policy="other"))
    with pytest.raises(PermissionError, match="environment"):
        runner.train(_spec(train=TrainSpec(max_steps=2, env={"SECRET": "value"})))
    with pytest.raises(PermissionError, match="evaluation environment"):
        runner._evaluation_environment(
            _spec(
                evaluation=EvaluationSpec(
                    environment="sim",
                    tasks=["stack_bowls"],
                    env={"SECRET": "value"},
                )
            )
        )

    bounded_runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allowed_evaluation_env=frozenset({"ACTION_EXECUTION"}),
        ),
        log_root=tmp_path / "bounded-eval-logs",
    )
    bounded_env = bounded_runner._evaluation_environment(
        _spec(
            evaluation=EvaluationSpec(
                environment="sim",
                tasks=["stack_bowls"],
                env={"ACTION_EXECUTION": "open_loop_50"},
            )
        )
    )
    assert bounded_env["ACTION_EXECUTION"] == "open_loop_50"


def test_empty_stub_checkpoint_requires_explicit_policy_exception(tmp_path):
    root = _fake_xpolicy_repo(tmp_path, write_checkpoint=False)
    strict = XPolicyExperimentRunner(
        root,
        RunnerPolicy(allowed_policies=frozenset({"demo_policy"})),
        log_root=tmp_path / "strict-logs",
    )
    assert strict.train(_spec()).status == "failed"

    stub = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allow_empty_checkpoints_for=frozenset({"demo_policy"}),
        ),
        log_root=tmp_path / "stub-logs",
    )
    artifact = stub.train(_spec())
    assert artifact.status == "completed"
    assert artifact.checkpoint is not None and artifact.checkpoint.files == []


def test_prepare_registers_existing_checkpoint_without_training(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(allowed_policies=frozenset({"demo_policy"})),
        log_root=tmp_path / "logs",
    )
    spec = _spec(train=TrainSpec(enabled=False))
    checkpoint = runner.expected_checkpoint_path(spec)
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.bin").write_text("existing checkpoint")

    artifact = runner.prepare(spec)
    assert artifact.status == "completed"
    assert artifact.checkpoint is not None
    assert artifact.command == [] and artifact.logs == []


def test_no_gpu_fake_training_evaluation_roundtrip(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}), max_timeout_seconds=10
        ),
        log_root=tmp_path / "logs",
    )
    memory_path = tmp_path / "embodied_memory.jsonl"
    session = TrainingSession(
        goal=Goal(name="success", metric="success_rate", direction="max"),
        memory=str(memory_path),
    )

    def evaluate(spec, artifact, evaluation_id, checkpoint_id):
        assert artifact.checkpoint is not None
        assert Path(artifact.checkpoint.path).is_dir()
        return [
            _trial(
                evaluation_id,
                checkpoint_id,
                "0",
                outcome="success",
                success=1.0,
                inference_latency_ms=12.0,
            ),
            _trial(
                evaluation_id,
                checkpoint_id,
                "1",
                outcome="failure",
                success=0.0,
                inference_latency_ms=18.0,
            ),
        ]

    result = XPolicyExperiment(session, runner).run_round(_spec(), evaluate)
    assert result.artifact.status == "completed"
    assert result.artifact.checkpoint is not None
    assert len(result.artifact.checkpoint.sha256) == 64
    assert result.evaluation is not None
    assert result.evaluation.status == "completed"
    assert result.evaluation.metrics["success_rate"] == 0.5
    assert session.history[-1]["success_rate"] == 0.5
    assert session.state.status == "ended"
    assert len(session.state.checkpoints.list()) == 1
    record = json.loads(memory_path.read_text().splitlines()[0])
    assert record["score"] == 0.5
    assert record["embodied"]["artifact"]["status"] == "completed"
    assert all(Path(path).is_file() for path in result.artifact.logs)


def test_real_debug_jsonl_runner_bridge_marks_results_inconclusive(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            max_timeout_seconds=10,
        ),
        log_root=tmp_path / "logs",
    )
    spec = _spec()
    artifact = runner.train(spec)
    assert artifact.checkpoint is not None

    trials = runner.evaluate_debug(
        spec,
        artifact,
        "real-debug-eval",
        artifact.checkpoint.sha256,
        policy_env="fake-policy-env",
        evaluation_env="fake-eval-env",
    )
    assert len(trials) == 2
    assert {trial.outcome for trial in trials} == {"invalid"}
    assert {trial.termination_reason for trial in trials} == {"debug_protocol_complete"}
    aggregator = TrialAggregator(
        "real-debug-eval",
        artifact.checkpoint.sha256,
        minimum_valid_trials=2,
    )
    for trial in trials:
        aggregator.add(trial)
    assert aggregator.summarize().status == "inconclusive"


def test_sim_jsonl_runner_bridge_returns_valid_trials(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            max_timeout_seconds=10,
        ),
        log_root=tmp_path / "logs",
    )
    spec = _spec(
        evaluation=EvaluationSpec(environment="sim", tasks=["stack_bowls"], repeats=2)
    )
    artifact = runner.train(spec)
    assert artifact.checkpoint is not None

    trials = runner.evaluate_sim(
        spec,
        artifact,
        "real-sim-eval",
        artifact.checkpoint.sha256,
        policy_env="fake-policy-env",
        evaluation_env="fake-sim-env",
    )
    assert [trial.outcome for trial in trials] == ["success", "failure"]
    assert [trial.success for trial in trials] == [1.0, 0.0]
    assert len(artifact.logs) == 4


def test_recorded_replay_runner_bridge_uses_distinct_gate_metric(tmp_path):
    root = _fake_xpolicy_repo(tmp_path)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}),
            allowed_scripts=frozenset({"train.sh", "eval.sh"}),
            max_timeout_seconds=10,
        ),
        log_root=tmp_path / "logs",
    )
    spec = _spec(
        evaluation=EvaluationSpec(
            environment="replay", tasks=["stack_bowls"], repeats=2
        )
    )
    artifact = runner.train(spec)
    assert artifact.checkpoint is not None
    episode_paths = []
    for index in range(2):
        episode = tmp_path / f"episode-{index}"
        episode.mkdir()
        (episode / "schema.json").write_text("{}")
        (episode / "metadata.json").write_text("{}")
        episode_paths.append(episode)

    trials = runner.evaluate_replay(
        spec,
        artifact,
        "replay-eval",
        artifact.checkpoint.sha256,
        episodes={"stack_bowls": episode_paths},
        policy_env="fake-policy-env",
        evaluation_env="fake-replay-env",
    )
    assert len(trials) == 2
    assert {trial.evidence_type for trial in trials} == {"replay_validation"}
    aggregator = TrialAggregator(
        "replay-eval", artifact.checkpoint.sha256, minimum_valid_trials=2
    )
    for trial in trials:
        aggregator.add(trial)
    summary = aggregator.summarize()
    assert summary.status == "completed"
    assert summary.metrics["replay_pass_rate"] == 1.0
    assert "success_rate" not in summary.metrics


def test_failed_training_does_not_run_evaluator(tmp_path):
    root = _fake_xpolicy_repo(tmp_path, exit_code=3)
    runner = XPolicyExperimentRunner(
        root,
        RunnerPolicy(
            allowed_policies=frozenset({"demo_policy"}), max_timeout_seconds=10
        ),
        log_root=tmp_path / "logs",
    )
    called = False

    def evaluate(*args):
        nonlocal called
        called = True
        return []

    result = XPolicyExperiment(TrainingSession(), runner).run_round(_spec(), evaluate)
    assert result.artifact.status == "failed"
    assert result.artifact.returncode == 3
    assert result.evaluation is None
    assert not called


def test_turbovla_example_preserves_virtualenv_python_symlink():
    example = (
        Path(__file__).parents[1] / "examples" / "xpolicylab_turbovla_gr3_round.py"
    ).read_text()
    assert "args.turbovla_python.expanduser().absolute()" in example
    assert "args.turbovla_python.resolve()" not in example
    assert "evaluation_env=turbovla_python" in example


def test_turbovla_robotwin_round_propagates_requested_seed():
    example = (
        Path(__file__).parents[1]
        / "examples"
        / "xpolicylab_turbovla_robotwin_round.py"
    ).read_text()
    assert 'parser.add_argument("--seed", type=int, default=0)' in example
    assert "seed=args.seed" in example
    assert '"--action-execution"' in example
    assert '"TURBOVLA_ACTION_ENSEMBLE"' in example
    assert '"--instruction-type"' in example
    assert '"ROBOTWIN_INSTRUCTION_TYPE"' in example
    assert '"--fixed-instruction"' in example
    assert '"ROBOTWIN_FIXED_INSTRUCTION"' in example
