"""Restricted subprocess runner for XPolicyLab's standard policy lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from interactive_training.integrations.xpolicylab.contracts import (
    ArtifactFile,
    ArtifactManifest,
    CheckpointArtifact,
    DatasetBinding,
    DatasetManifest,
    ExperimentSpec,
    SourceRevision,
    TrialResult,
)
from interactive_training.integrations.xpolicylab.datasets import (
    load_dataset_manifest,
    validate_dataset_manifest_profile,
    verify_dataset_files,
)
from interactive_training.integrations.xpolicylab.reporter import load_trial_results

_GPU_IDS = re.compile(r"^[0-9]+(?:,[0-9]+)*$")


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_artifact(
    path: Path, step: int = 0, *, allow_empty: bool = False
) -> CheckpointArtifact:
    if path.is_symlink():
        raise ValueError("checkpoint root cannot be a symlink")
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    paths = (
        [path]
        if path.is_file()
        else sorted(item for item in path.rglob("*") if item.is_file())
    )
    files: list[ArtifactFile] = []
    for item in paths:
        if item.is_symlink():
            raise ValueError(f"checkpoint cannot contain symlinks: {item}")
        relative = item.name if path.is_file() else item.relative_to(path).as_posix()
        files.append(
            ArtifactFile(
                path=relative, size=item.stat().st_size, sha256=_sha256_file(item)
            )
        )
    if not files and not allow_empty:
        raise ValueError(f"checkpoint contains no files: {path}")

    encoded = json.dumps(
        [entry.model_dump() for entry in files], sort_keys=True, separators=(",", ":")
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return CheckpointArtifact(path=str(path), sha256=digest, step=step, files=files)


@dataclass(frozen=True)
class RunnerPolicy:
    allowed_policies: frozenset[str]
    allowed_scripts: frozenset[str] = frozenset({"train.sh"})
    allowed_env: frozenset[str] = frozenset()
    allowed_evaluation_env: frozenset[str] = frozenset()
    allow_empty_checkpoints_for: frozenset[str] = frozenset()
    inherited_env: frozenset[str] = frozenset(
        {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"}
    )
    train_env_map: dict[str, str] = field(default_factory=dict)
    max_timeout_seconds: int = 3600

    def __post_init__(self):
        if self.max_timeout_seconds < 1:
            raise ValueError("max_timeout_seconds must be positive")
        mapped = set(self.train_env_map.values())
        if not mapped <= set(self.allowed_env):
            raise ValueError("train_env_map values must be included in allowed_env")
        if not self.allow_empty_checkpoints_for <= self.allowed_policies:
            raise ValueError("empty checkpoint exceptions must name allowed policies")


class XPolicyExperimentRunner:
    def __init__(
        self,
        repo_root: str | Path,
        policy: RunnerPolicy,
        *,
        log_root: str | Path | None = None,
        interactive_training_root: str | Path | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise FileNotFoundError(f"XPolicyLab root not found: {self.repo_root}")
        self.policy = policy
        self.log_root = (
            Path(log_root).resolve()
            if log_root
            else Path.cwd() / ".interactive_training"
        )
        self.interactive_training_root = (
            Path(interactive_training_root).resolve()
            if interactive_training_root
            else Path(__file__).resolve().parents[4]
        )

    def _script(self, policy_name: str, script_name: str) -> Path:
        if policy_name not in self.policy.allowed_policies:
            raise PermissionError(f"policy is not allowed: {policy_name}")
        if script_name not in self.policy.allowed_scripts:
            raise PermissionError(f"script is not allowed: {script_name}")
        policy_root = (self.repo_root / "policy" / policy_name).resolve()
        script = (policy_root / script_name).resolve()
        if policy_root not in script.parents or not script.is_file():
            raise FileNotFoundError(f"allowed script not found: {script}")
        return script

    def _environment(self, spec: ExperimentSpec, gpu_ids: str) -> dict[str, str]:
        if not _GPU_IDS.fullmatch(gpu_ids):
            raise ValueError(
                "gpu_ids must be a comma-separated list of numeric device ids"
            )
        unknown = set(spec.train.env) - set(self.policy.allowed_env)
        if unknown:
            raise PermissionError(
                f"training environment keys are not allowed: {sorted(unknown)}"
            )
        env = {
            key: os.environ[key]
            for key in self.policy.inherited_env
            if key in os.environ
        }
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        env.update(spec.train.env)
        values = {
            "max_steps": spec.train.max_steps,
            "learning_rate": spec.train.learning_rate,
        }
        for field_name, env_name in self.policy.train_env_map.items():
            value = values.get(field_name)
            if value is not None:
                env[env_name] = str(value)
        return env

    def _base_environment(self) -> dict[str, str]:
        return {
            key: os.environ[key]
            for key in self.policy.inherited_env
            if key in os.environ
        }

    def _evaluation_environment(self, spec: ExperimentSpec) -> dict[str, str]:
        unknown = set(spec.evaluation.env) - set(self.policy.allowed_evaluation_env)
        if unknown:
            raise PermissionError(
                f"evaluation environment keys are not allowed: {sorted(unknown)}"
            )
        env = self._base_environment()
        env.update(spec.evaluation.env)
        return env

    def validate_training_dataset(
        self,
        spec: ExperimentSpec,
        dataset_manifest_path: str | Path | None,
    ) -> DatasetManifest | None:
        if spec.dataset_manifest_id is None:
            if dataset_manifest_path is not None:
                raise ValueError(
                    "dataset_manifest_path requires spec.dataset_manifest_id"
                )
            return None
        if dataset_manifest_path is None:
            raise ValueError("spec.dataset_manifest_id requires dataset_manifest_path")
        manifest = load_dataset_manifest(dataset_manifest_path)
        if manifest.dataset_id != spec.dataset_manifest_id:
            raise ValueError(
                "dataset manifest id does not match spec.dataset_manifest_id"
            )
        validate_dataset_manifest_profile(manifest)
        if manifest.summary.eligible_episode_count < 1:
            raise ValueError("dataset manifest contains no eligible episodes")
        verify_dataset_files(manifest)
        return manifest

    @staticmethod
    def _snapshot_dataset(run_dir: Path, manifest: DatasetManifest) -> DatasetBinding:
        path = run_dir / "dataset-manifest.json"
        if path.exists():
            raise FileExistsError(f"refusing to replace dataset snapshot: {path}")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(manifest.model_dump_json(indent=2))
            stream.write("\n")
        return DatasetBinding(
            dataset_id=manifest.dataset_id,
            dataset_sha256=manifest.dataset_sha256,
            profile_id=manifest.profile_id,
            manifest_path=str(path.resolve()),
            eligible_episode_count=manifest.summary.eligible_episode_count,
        )

    @staticmethod
    def _dataset_environment(binding: DatasetBinding) -> dict[str, str]:
        return {
            "XPOLICYLAB_DATASET_MANIFEST": binding.manifest_path,
            "XPOLICYLAB_DATASET_ID": binding.dataset_id,
            "XPOLICYLAB_DATASET_SHA256": binding.dataset_sha256,
            "XPOLICYLAB_DATASET_PROFILE": binding.profile_id,
            "XPOLICYLAB_DATASET_ELIGIBLE_EPISODES": str(binding.eligible_episode_count),
        }

    def expected_checkpoint_path(self, spec: ExperimentSpec) -> Path:
        name = "-".join(
            [
                spec.bench_name,
                spec.checkpoint_name,
                spec.env_cfg_type,
                spec.action_type,
                str(spec.seed),
            ]
        )
        return self.repo_root / "policy" / spec.policy / "checkpoints" / name

    def train(
        self,
        spec: ExperimentSpec,
        *,
        gpu_ids: str = "0",
        timeout_seconds: int | None = None,
        dataset_manifest_path: str | Path | None = None,
    ) -> ArtifactManifest:
        if not spec.train.enabled:
            raise ValueError("train() requires spec.train.enabled=true")
        timeout = timeout_seconds or self.policy.max_timeout_seconds
        if timeout < 1 or timeout > self.policy.max_timeout_seconds:
            raise ValueError("timeout exceeds RunnerPolicy limit")
        dataset_manifest = self.validate_training_dataset(spec, dataset_manifest_path)
        script = self._script(spec.policy, "train.sh")
        command = [
            "/bin/bash",
            str(script),
            spec.bench_name,
            spec.checkpoint_name,
            spec.env_cfg_type,
            spec.action_type,
            str(spec.seed),
            gpu_ids,
        ]
        env = self._environment(spec, gpu_ids)
        run_dir = self.log_root / spec.experiment_id / f"round-{spec.round}"
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset = (
            self._snapshot_dataset(run_dir, dataset_manifest)
            if dataset_manifest is not None
            else None
        )
        if dataset is not None:
            env.update(self._dataset_environment(dataset))
        stdout_path = run_dir / "train.stdout.log"
        stderr_path = run_dir / "train.stderr.log"
        started = datetime.now(timezone.utc)
        status = "failed"
        returncode: int | None = None
        error: str | None = None

        try:
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                result = subprocess.run(
                    command,
                    cwd=script.parent,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
            returncode = result.returncode
            status = "completed" if returncode == 0 else "failed"
            if returncode != 0:
                error = f"training command exited with status {returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
            error = f"training command exceeded {timeout} seconds"
        except OSError as exc:
            status = "failed"
            error = str(exc)

        checkpoint = None
        if status == "completed":
            try:
                checkpoint = checkpoint_artifact(
                    self.expected_checkpoint_path(spec),
                    step=spec.train.max_steps or 0,
                    allow_empty=spec.policy in self.policy.allow_empty_checkpoints_for,
                )
            except (OSError, ValueError) as exc:
                status = "failed"
                error = str(exc)

        return ArtifactManifest(
            experiment_id=spec.experiment_id,
            round=spec.round,
            status=status,
            source=SourceRevision(
                interactive_training_commit=_git_commit(self.interactive_training_root),
                xpolicylab_commit=_git_commit(self.repo_root),
                policy=spec.policy,
            ),
            checkpoint=checkpoint,
            dataset=dataset,
            logs=[
                *([dataset.manifest_path] if dataset is not None else []),
                str(stdout_path),
                str(stderr_path),
            ],
            command=command,
            returncode=returncode,
            error=error,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    def prepare(
        self,
        spec: ExperimentSpec,
        *,
        gpu_ids: str = "0",
        timeout_seconds: int | None = None,
        dataset_manifest_path: str | Path | None = None,
    ) -> ArtifactManifest:
        """Train or register an existing checkpoint according to the spec."""
        if spec.train.enabled:
            return self.train(
                spec,
                gpu_ids=gpu_ids,
                timeout_seconds=timeout_seconds,
                dataset_manifest_path=dataset_manifest_path,
            )

        started = datetime.now(timezone.utc)
        dataset_manifest = self.validate_training_dataset(spec, dataset_manifest_path)
        run_dir = self.log_root / spec.experiment_id / f"round-{spec.round}"
        run_dir.mkdir(parents=True, exist_ok=True)
        dataset = (
            self._snapshot_dataset(run_dir, dataset_manifest)
            if dataset_manifest is not None
            else None
        )
        checkpoint = None
        error = None
        status = "completed"
        try:
            checkpoint = checkpoint_artifact(
                self.expected_checkpoint_path(spec),
                allow_empty=spec.policy in self.policy.allow_empty_checkpoints_for,
            )
        except (OSError, ValueError) as exc:
            status = "failed"
            error = str(exc)
        return ArtifactManifest(
            experiment_id=spec.experiment_id,
            round=spec.round,
            status=status,
            source=SourceRevision(
                interactive_training_commit=_git_commit(self.interactive_training_root),
                xpolicylab_commit=_git_commit(self.repo_root),
                policy=spec.policy,
            ),
            checkpoint=checkpoint,
            dataset=dataset,
            logs=[dataset.manifest_path] if dataset is not None else [],
            command=[],
            returncode=None,
            error=error,
            started_at=started,
            finished_at=datetime.now(timezone.utc),
        )

    def evaluate_debug(
        self,
        spec: ExperimentSpec,
        artifact: ArtifactManifest,
        evaluation_id: str,
        checkpoint_id: str,
        *,
        policy_env: str,
        evaluation_env: str,
        policy_gpu_id: str = "0",
        environment_gpu_id: str = "0",
        timeout_seconds: int | None = None,
    ) -> list[TrialResult]:
        """Run XPolicyLab's real debug client and consume its versioned JSONL."""
        if spec.evaluation.environment != "debug":
            raise ValueError("evaluate_debug requires evaluation.environment='debug'")
        if artifact.status != "completed" or artifact.checkpoint is None:
            raise ValueError(
                "debug evaluation requires a completed checkpoint artifact"
            )
        if checkpoint_id != artifact.checkpoint.sha256:
            raise ValueError("checkpoint_id does not match artifact")
        if (
            not policy_env
            or not evaluation_env
            or "\x00" in policy_env + evaluation_env
        ):
            raise ValueError("policy and evaluation environments must be non-empty")
        if not _GPU_IDS.fullmatch(policy_gpu_id) or not _GPU_IDS.fullmatch(
            environment_gpu_id
        ):
            raise ValueError("GPU ids must be numeric")
        timeout = timeout_seconds or self.policy.max_timeout_seconds
        if timeout < 1 or timeout > self.policy.max_timeout_seconds:
            raise ValueError("timeout exceeds RunnerPolicy limit")

        script = self._script(spec.policy, "eval.sh")
        run_dir = self.log_root / spec.experiment_id / f"round-{spec.round}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "debug-trials.jsonl"
        if result_path.exists():
            raise FileExistsError(
                f"refusing to append to existing trial spool: {result_path}"
            )

        for task_index, task in enumerate(spec.evaluation.tasks):
            stdout_path = run_dir / f"eval-{task_index}.stdout.log"
            stderr_path = run_dir / f"eval-{task_index}.stderr.log"
            command = [
                "/bin/bash",
                str(script),
                spec.bench_name,
                task,
                spec.checkpoint_name,
                spec.env_cfg_type,
                spec.action_type,
                str(spec.seed),
                policy_gpu_id,
                environment_gpu_id,
                policy_env,
                evaluation_env,
            ]
            env = self._evaluation_environment(spec)
            env.update(
                {
                    "EVAL_ENV_TYPE": "debug",
                    "XPOLICYLAB_CHECKPOINT_ID": checkpoint_id,
                    "XPOLICYLAB_EVALUATION_ID": evaluation_id,
                    "XPOLICYLAB_TRIAL_ID": f"{evaluation_id}-{task}",
                    "XPOLICYLAB_TRIAL_RESULT_JSONL": str(result_path),
                    "XPOLICYLAB_EVAL_EPISODES": str(spec.evaluation.repeats),
                }
            )
            try:
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    process = subprocess.run(
                        command,
                        cwd=script.parent,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout,
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"debug evaluation for task {task!r} exceeded {timeout} seconds"
                ) from exc
            artifact.logs.extend([str(stdout_path), str(stderr_path)])
            if process.returncode != 0:
                raise RuntimeError(
                    f"debug evaluation for task {task!r} exited with status "
                    f"{process.returncode}; "
                    f"see {stderr_path}"
                )

        return load_trial_results(
            result_path,
            evaluation_id=evaluation_id,
            checkpoint_id=checkpoint_id,
        )

    def evaluate_sim(
        self,
        spec: ExperimentSpec,
        artifact: ArtifactManifest,
        evaluation_id: str,
        checkpoint_id: str,
        *,
        policy_env: str,
        evaluation_env: str,
        policy_gpu_id: str = "0",
        environment_gpu_id: str = "0",
        timeout_seconds: int | None = None,
    ) -> list[TrialResult]:
        """Run RoboDojo and consume its validated episode result bridge."""
        if spec.evaluation.environment != "sim":
            raise ValueError("evaluate_sim requires evaluation.environment='sim'")
        if artifact.status != "completed" or artifact.checkpoint is None:
            raise ValueError("sim evaluation requires a completed checkpoint artifact")
        if checkpoint_id != artifact.checkpoint.sha256:
            raise ValueError("checkpoint_id does not match artifact")
        if (
            not policy_env
            or not evaluation_env
            or "\x00" in policy_env + evaluation_env
        ):
            raise ValueError("policy and evaluation environments must be non-empty")
        if not _GPU_IDS.fullmatch(policy_gpu_id) or not _GPU_IDS.fullmatch(
            environment_gpu_id
        ):
            raise ValueError("GPU ids must be numeric")
        timeout = timeout_seconds or self.policy.max_timeout_seconds
        if timeout < 1 or timeout > self.policy.max_timeout_seconds:
            raise ValueError("timeout exceeds RunnerPolicy limit")

        script = self._script(spec.policy, "eval.sh")
        run_dir = self.log_root / spec.experiment_id / f"round-{spec.round}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "sim-trials.jsonl"
        if result_path.exists():
            raise FileExistsError(
                f"refusing to append to existing trial spool: {result_path}"
            )

        for task_index, task in enumerate(spec.evaluation.tasks):
            stdout_path = run_dir / f"sim-eval-{task_index}.stdout.log"
            stderr_path = run_dir / f"sim-eval-{task_index}.stderr.log"
            command = [
                "/bin/bash",
                str(script),
                spec.bench_name,
                task,
                spec.checkpoint_name,
                spec.env_cfg_type,
                spec.action_type,
                str(spec.seed),
                policy_gpu_id,
                environment_gpu_id,
                policy_env,
                evaluation_env,
            ]
            run_identity = hashlib.sha256(
                f"{evaluation_id}\0{task_index}\0{task}".encode()
            ).hexdigest()[:16]
            env = self._evaluation_environment(spec)
            env.update(
                {
                    "EVAL_ENV_TYPE": "sim",
                    "EVAL_NUM": str(spec.evaluation.repeats),
                    "ROBODOJO_RUN_ID": f"interactive-{run_identity}",
                    "XPOLICYLAB_CHECKPOINT_ID": checkpoint_id,
                    "XPOLICYLAB_EVALUATION_ID": evaluation_id,
                    "XPOLICYLAB_EVAL_EPISODES": str(spec.evaluation.repeats),
                    "XPOLICYLAB_TRIAL_RESULT_JSONL": str(result_path),
                }
            )
            try:
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    process = subprocess.run(
                        command,
                        cwd=script.parent,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout,
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"sim evaluation for task {task!r} exceeded {timeout} seconds"
                ) from exc
            artifact.logs.extend([str(stdout_path), str(stderr_path)])
            if process.returncode != 0:
                raise RuntimeError(
                    f"sim evaluation for task {task!r} exited with status "
                    f"{process.returncode}; see {stderr_path}"
                )

        trials = load_trial_results(
            result_path,
            evaluation_id=evaluation_id,
            checkpoint_id=checkpoint_id,
        )
        expected_trials = len(spec.evaluation.tasks) * spec.evaluation.repeats
        if len(trials) > expected_trials:
            raise ValueError(
                f"sim evaluation produced {len(trials)} trials; "
                f"expected at most {expected_trials}"
            )
        return trials

    def evaluate_replay(
        self,
        spec: ExperimentSpec,
        artifact: ArtifactManifest,
        evaluation_id: str,
        checkpoint_id: str,
        *,
        episodes: Mapping[str, Sequence[str | Path]],
        policy_env: str,
        evaluation_env: str,
        start: int = 0,
        frame_count: int = 2,
        stride: int = 100,
        policy_gpu_id: str = "0",
        environment_gpu_id: str = "0",
        timeout_seconds: int | None = None,
    ) -> list[TrialResult]:
        """Replay recorded GR3 episodes as an offline contract gate."""
        if spec.evaluation.environment != "replay":
            raise ValueError("evaluate_replay requires evaluation.environment='replay'")
        if artifact.status != "completed" or artifact.checkpoint is None:
            raise ValueError(
                "replay evaluation requires a completed checkpoint artifact"
            )
        if checkpoint_id != artifact.checkpoint.sha256:
            raise ValueError("checkpoint_id does not match artifact")
        if (
            not policy_env
            or not evaluation_env
            or "\x00" in policy_env + evaluation_env
        ):
            raise ValueError("policy and evaluation environments must be non-empty")
        if start < 0 or frame_count < 1 or stride < 1:
            raise ValueError("replay start must be non-negative; count/stride positive")
        if not _GPU_IDS.fullmatch(policy_gpu_id) or not _GPU_IDS.fullmatch(
            environment_gpu_id
        ):
            raise ValueError("GPU ids must be numeric")
        if set(episodes) != set(spec.evaluation.tasks):
            raise ValueError(
                "replay episode mapping must exactly match evaluation tasks"
            )
        timeout = timeout_seconds or self.policy.max_timeout_seconds
        if timeout < 1 or timeout > self.policy.max_timeout_seconds:
            raise ValueError("timeout exceeds RunnerPolicy limit")

        validated_episodes: dict[str, list[Path]] = {}
        for task in spec.evaluation.tasks:
            task_episodes = episodes[task]
            if (
                isinstance(task_episodes, (str, bytes))
                or len(task_episodes) != spec.evaluation.repeats
            ):
                raise ValueError(
                    f"replay task {task!r} requires exactly "
                    f"{spec.evaluation.repeats} episode directories"
                )
            validated_episodes[task] = []
            for raw_path in task_episodes:
                path = Path(raw_path).expanduser()
                if path.is_symlink():
                    raise ValueError(f"replay episode cannot be a symlink: {path}")
                path = path.resolve()
                if not path.is_dir():
                    raise FileNotFoundError(f"replay episode not found: {path}")
                for required_name in ("schema.json", "metadata.json"):
                    required = path / required_name
                    if required.is_symlink() or not required.is_file():
                        raise FileNotFoundError(
                            f"replay episode is missing regular {required_name}: {path}"
                        )
                validated_episodes[task].append(path)

        script = self._script(spec.policy, "eval.sh")
        run_dir = self.log_root / spec.experiment_id / f"round-{spec.round}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_path = run_dir / "replay-trials.jsonl"
        if result_path.exists():
            raise FileExistsError(
                f"refusing to append to existing trial spool: {result_path}"
            )

        for task_index, task in enumerate(spec.evaluation.tasks):
            manifest_path = run_dir / f"replay-episodes-{task_index}.json"
            if manifest_path.exists():
                raise FileExistsError(
                    f"refusing to replace replay manifest: {manifest_path}"
                )
            manifest = [
                {
                    "episode_dir": str(path),
                    "task": task,
                    "repeat_index": repeat_index,
                    "trial_id": f"{evaluation_id}-{task}-repeat-{repeat_index}",
                }
                for repeat_index, path in enumerate(validated_episodes[task])
            ]
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            stdout_path = run_dir / f"replay-eval-{task_index}.stdout.log"
            stderr_path = run_dir / f"replay-eval-{task_index}.stderr.log"
            command = [
                "/bin/bash",
                str(script),
                spec.bench_name,
                task,
                spec.checkpoint_name,
                spec.env_cfg_type,
                spec.action_type,
                str(spec.seed),
                policy_gpu_id,
                environment_gpu_id,
                policy_env,
                evaluation_env,
            ]
            env = self._evaluation_environment(spec)
            env.update(
                {
                    "EVAL_ENV_TYPE": "replay",
                    "XPOLICYLAB_CHECKPOINT_ID": checkpoint_id,
                    "XPOLICYLAB_EVALUATION_ID": evaluation_id,
                    "XPOLICYLAB_EVAL_EPISODES": str(spec.evaluation.repeats),
                    "XPOLICYLAB_REPLAY_EPISODE_MANIFEST": str(manifest_path),
                    "XPOLICYLAB_REPLAY_START": str(start),
                    "XPOLICYLAB_REPLAY_COUNT": str(frame_count),
                    "XPOLICYLAB_REPLAY_STRIDE": str(stride),
                    "XPOLICYLAB_TRIAL_RESULT_JSONL": str(result_path),
                }
            )
            try:
                with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                    process = subprocess.run(
                        command,
                        cwd=script.parent,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout,
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"replay evaluation for task {task!r} exceeded {timeout} seconds"
                ) from exc
            artifact.logs.extend(
                [str(manifest_path), str(stdout_path), str(stderr_path)]
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"replay evaluation for task {task!r} exited with status "
                    f"{process.returncode}; see {stderr_path}"
                )

        trials = load_trial_results(
            result_path,
            evaluation_id=evaluation_id,
            checkpoint_id=checkpoint_id,
        )
        expected_trials = len(spec.evaluation.tasks) * spec.evaluation.repeats
        if len(trials) > expected_trials:
            raise ValueError(
                f"replay evaluation produced {len(trials)} trials; "
                f"expected at most {expected_trials}"
            )
        if any(trial.evidence_type != "replay_validation" for trial in trials):
            raise ValueError("replay evaluation returned non-replay evidence")
        return trials
