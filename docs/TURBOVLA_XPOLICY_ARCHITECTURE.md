# TurboVLA × XPolicyLab × Interactive Training

Status: official RoboTwin single-task training and XPolicyLab simulator round
validated locally.

## Architecture decision

The three repositories remain separate runtimes but form one product boundary:

```text
Interactive Training (control plane)
  ExperimentSpec / dataset manifest / bounded knobs / round memory
                  |
                  v
XPolicyLab (embodied execution and evaluation framework)
  policy/TurboVLA train + serve / benchmark adapters / result contract
                  |
                  v
TurboVLA (policy implementation)
  StarVLA model / official data loader / optimizer / inference server
                  |
                  v
RoboTwin simulator now; GR3 replay/shadow/robot later
```

XPolicyLab is the user-facing framework. TurboVLA is an XPolicyLab policy
plugin, not a second experiment system. Interactive Training is the mandatory
training controller; direct unbound training is rejected by
`policy/TurboVLA/train.sh`.

Do not merge the Python environments. Interactive Training, TurboVLA, and
RoboTwin have incompatible heavy dependencies and run as separate processes.
Their integration contracts are versioned JSON manifests, checkpoint artifacts,
trial JSONL, and the existing policy-server protocol.

## Separate robot profiles

RoboTwin and GR3 are deliberately distinct:

| Profile | Cameras | State/action | Purpose |
| --- | --- | --- | --- |
| `turbovla.robotwin_clean_v1` | head + two wrists | 14D / 14D, horizon 50 | official TurboVLA training and RoboTwin simulation |
| `xpolicylab.gr3_dagger_v2` | recorded GR3 top camera | raw 33D/37D; TurboVLA joint 31D/31D, horizon 50 | GR3 DAgger replay, then GR3-specific deployment gates |

Neither data nor benchmark evidence is silently transferred between profiles.
In particular, RoboTwin success is not GR3 deployment evidence.

## Current local evidence

The official converted `beat_block_hammer` subset contains 50 trajectories and
5,682 frames. XPolicyLab hashes it into an immutable manifest; Interactive
Training verifies the profile and every source file before launch.

The published RoboTwin checkpoint belongs to TurboVLA's legacy
`GroundingDINODiT` runtime. An early current-runtime smoke used a permissive
`strict=False` load and therefore did not establish released-checkpoint
initialization: the old and current parameter namespaces did not match. Both
training runtimes now use strict full-model loading so this failure mode is
rejected instead of being reported as a successful load.

XPolicyLab now selects the legacy runtime for the published checkpoint and its
derived fine-tunes. Interactive Training completed strict-load training through
the full Interactive Training -> XPolicyLab -> TurboVLA path and saved raw and
EMA artifacts. A raw candidate trained on one successful seed-100000 expert
trajectory then produced two deterministic 1/1 RoboTwin successes with
identical action traces. The same artifact failed on seed 200000. The local
results in `TURBOVLA_ROBOTWIN_LOCAL_RESULTS.md` therefore establish architecture
and experiment-loop validity, but not cross-scene model quality.

This workstation does not currently expose an `nvcc` toolkit for compiling
CuRobo. The local smoke therefore uses RoboTwin's documented MPLib screw-planner
alternative for its expert seed feasibility check. Policy rollout and RoboTwin
success checking are unchanged. Formal comparisons with published CuRobo-based
settings must use an environment with the complete official planner stack.

## Supported round

`examples/xpolicylab_turbovla_robotwin_round.py` is the combined entry point.
Without `--reuse-checkpoint` it trains first; with that flag it registers the
named XPolicyLab checkpoint and proceeds directly to simulation. Both modes
still validate and snapshot the dataset manifest.

`--action-execution temporal_ensemble` is the default benchmark path.
`--action-execution open_loop_50` is an explicit diagnostic A/B. The selected
mode is stored in `EvaluationSpec.env`; the runner only forwards allowlisted
evaluation knobs, so it cannot become an unrecorded shell-environment override.
`temporal_ensemble_oldest_binary` is also available for diagnosing aligned
binary actions. It is experimental and must not be used as a promotion
benchmark because it produced an incorrect right-gripper closure locally.
`--fixed-instruction` is an audited control for deterministic gates. It is only
forwarded as the allowlisted `ROBOTWIN_FIXED_INSTRUCTION` evaluation variable;
normal benchmark runs should omit it and retain seen/unseen instruction
sampling.

```bash
PYTHONPATH=src .venv/bin/python \
  examples/xpolicylab_turbovla_robotwin_round.py \
  --xpolicylab-root /home/ubuntu/xpolicylabdagger \
  --dataset-manifest runs/turbovla-robotwin/beat-block-hammer-manifest-v2.json \
  --turbovla-root /home/ubuntu/TurboVLA \
  --turbovla-python /home/ubuntu/TurboVLA/.venv/bin/python \
  --robotwin-python /home/ubuntu/RoboTwin/.venv/bin/python \
  --dinov3-path /home/ubuntu/TurboVLA/pretrained/dinov3-vitl16 \
  --bert-path /home/ubuntu/TurboVLA/pretrained/bert-base-uncased \
  --pretrained-checkpoint \
    /home/ubuntu/TurboVLA/pretrained/TurboVLA/checkpoints/robotwin/steps_55000_ema_model.safetensors \
  --max-steps 1 \
  --action-execution temporal_ensemble \
  --checkpoint-name beat-block-hammer-upstream-fixed2 \
  --reuse-checkpoint
```

The simulator bridge writes one authoritative RoboTwin episode as one
`task_outcome` trial. Interactive Training rejects missing or excess trials and
records the checkpoint digest, dataset binding, simulator evidence, aggregate,
and round memory.

## Development gates

1. One-task local training and one-episode simulator smoke. **Complete.**
2. Released checkpoint baseline over three seeds for `beat_block_hammer`.
   **Complete; 0/3 successes.**
3. Single-task incremental fine-tuning and same-seed comparison. **Complete;
   the controlled raw candidate succeeded twice at seed 100000.**
4. Per-step rollout diagnostics and expert-distribution comparison. **Complete;
   temporal ensemble stalls the learned phase; raw open-loop reproduced it.**
5. Held-out seed gate. **Started; seed 200000 failed, so generalization is not
   yet established.**
6. Multi-seed interactive data collection/retraining rounds.
7. Multi-task Clean50 benchmark and failure taxonomy.
8. GR3 recorded replay and shadow mode using the GR3-only profile.
9. Human-approved GR3 enforce after safety and latency qualification.

The RTX 5090 is sufficient for frozen-encoder single-task tuning and inference.
The successful gate used batch 16 with about 4.0 GiB observed training memory;
its 1,000 steps were data-decoding bound rather than GPU-memory bound.
Multi-GPU compute is therefore not the next bottleneck for this narrow
experiment. A full multi-task or unfrozen reproduction should move to remote
compute after the multi-seed data and evaluation definitions are fixed.
