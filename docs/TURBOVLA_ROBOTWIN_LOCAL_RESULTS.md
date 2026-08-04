# TurboVLA RoboTwin local baseline and fine-tuning results

Date: 2026-08-01

## Scope

This report covers the first controlled local experiment for
`beat_block_hammer`. Every training and evaluation run was launched by
Interactive Training, executed by the XPolicyLab TurboVLA plugin, and recorded
as an `xpolicy_interactive.v1` round.

The released-baseline manifest is
`runs/turbovla-robotwin/beat-block-hammer-manifest-v2.json` with profile
`turbovla.robotwin_clean_v1`. It binds the official converted subset of 50
trajectories and 5,682 frames. A later controlled success gate uses the same
profile with a separate one-trajectory manifest,
`runs/turbovla-robotwin/beat-block-hammer-seed100000-gate-manifest.json`.
GR3 data and the GR3 DAgger profile are not used in either experiment.

## Checkpoint compatibility correction

The published RoboTwin weights use the legacy `GroundingDINODiT` model and
parameter namespace. They are not compatible with the repository's current
runtime. An earlier current-runtime smoke used `strict=False`; it could finish
without loading matching released parameters and is excluded from model-quality
evidence.

The integration now routes published RoboTwin weights and their fine-tunes to a
detached legacy runtime at TurboVLA commit `8015c7c`. Full-model training loads
are strict in both the legacy and current runtimes. A namespace mismatch now
fails the run.

## Results

| Policy artifact | Training | Evaluation seeds | Valid trials | Successes |
| --- | ---: | --- | ---: | ---: |
| Published RoboTwin checkpoint | none | 0, 1, 2 | 3/3 | 0/3 |
| Strict-load fine-tune | 100 steps, batch 1 | 0 | 1/1 | 0/1 |
| Strict-load fine-tune EMA | 500 steps, batch 1 | 0, 1, 2 | 3/3 | 0/3 |
| 50-demo fine-tune EMA | 1,000 steps, batch 16 | 0 | 1/1 | 0/1 |
| Seed-100000 expert fine-tune EMA | 1,000 steps, batch 16 | 0 | 1/1 | 0/1 |
| Seed-100000 expert fine-tune raw | 1,000 steps, batch 16 | 0, repeated | 2/2 | 2/2 |
| Same raw artifact, held-out scene | no additional training | 1 | 1/1 | 0/1 |

The released baseline used checkpoint ID
`0c74ff1f13c25979cb7940147a7cafbb3c910e7d5df3661d73a264dcd192b09a`.
The three-seed 500-step comparison used one immutable deployment bundle with
checkpoint ID
`905fbaa4e565044aaf97a9056dee1e028721dac7af1eee2bbe02a7f361c9439d`.
The deployed EMA model file itself has SHA-256
`ba4136e082b2668d5d1937db40d2766e362c36c6d4ca4652c40d76358f6ceefb`.

For the 100-step run, action loss was 0.394644 on step 1, 0.210450 on average,
and 0.074677 over the last 10 steps. For the 500-step run it was 0.394644 on
step 1, 0.077065 on average, and 0.033263 over the last 100 steps. The lower
training objective did not produce a successful task rollout.

The 500-step training artifact was produced in 42.8 seconds on the local RTX
5090. An observed steady-state training sample used approximately 2.8 GiB of
GPU memory. Policy and simulator inference together used approximately 5.3 GiB
in the observed sample. These are point observations, not peak-memory claims.

## Controlled success gate

RoboTwin's expert planner produced one successful `beat_block_hammer` episode
at simulator seed 100000. The episode contains 165 frames, three camera views,
and 14D joint/gripper actions. It was converted without modifying the public
50-demo dataset by `scripts/convert_robotwin_hdf5_to_lerobot.py`, then hashed
into dataset ID
`robotwin-beat-block-hammer-seed100000-gate-c39625069a43`.

The combined Interactive Training -> XPolicyLab -> TurboVLA round trained the
released model for 1,000 steps with batch 16. Final action loss was 0.007595.
The EMA artifact still missed the hammer. Its first 50 open-loop left-arm joint
RMSE against the expert was 0.134230. The raw training artifact reduced that
number to 0.039055 and succeeded at simulator step 147.

The raw artifact was evaluated twice with seed 0 (RoboTwin seed 100000), fixed
instruction `Pick up the hammer with the left arm and beat the block.`, and
`open_loop_50`. Both trials succeeded at step 147. Their complete per-step
action traces have the identical SHA-256
`c78b991ccac66e0d099549cbcbc8059151ac0bced95d3242b63cbdd1dd26931e`.
The immutable XPolicyLab checkpoint ID is
`0cc3610b05149fe142f7acdf3745a45430c7b802204734cd1d6f4f5fe91e8923`.

The same raw weights, without additional training, failed on evaluation seed 1
(RoboTwin seed 200000), 0/1. This is therefore a deterministic same-scene
closed-loop integration success, not evidence of cross-scene generalization or
a benchmark-level success rate.

## Interpretation

The integration and evidence contracts are working: all reported simulator
trials completed, none were invalid, and repeated evaluations used stable
checkpoint identities. The controlled gate proves that the complete training,
registration, serving, action execution, simulation, and task-outcome path can
produce and deterministically reproduce a real RoboTwin success. The held-out
failure confines the next problem to data coverage, policy generalization, and
execution strategy rather than launch plumbing.

Training loss alone is not a promotion signal. Interactive Training must gate
future checkpoints on simulator task success and retain the released baseline
for comparison. The two repeated same-seed successes are adequate for the
architecture development gate but not for model promotion or a publishable
success-rate estimate.

The local simulator uses the MPLib screw-planner fallback because this machine
does not expose `nvcc` for the official CuRobo stack. This keeps the A/B
environment consistent, but results must not be presented as an official
CuRobo reproduction.

## Rollout diagnostics

The XPolicyLab evaluator now binds a per-step
`turbovla.robotwin_action_trace.v1` JSONL artifact to each RoboTwin trial. The
bridge rejects missing, malformed, non-contiguous, non-finite, or non-14D
traces. Interactive Training aggregates continuous-output saturation, action to
state distance, and left/right gripper open rates alongside task success.

On seed 0, both the published checkpoint and the 500-step EMA kept both
grippers open for all 400 simulator steps under temporal ensemble. With the
same checkpoints, seed, and simulator state, open-loop 50-step chunk execution
did execute the predicted left-gripper closure:

| Policy | Execution | Mean action/state L2 | Continuous saturation | Left open | Right open | Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Published | temporal ensemble | 0.015927 | 0.000008 | 100% | 100% | 0/1 |
| Published | ensemble + oldest binary | 0.031148 | 0.000900 | 26.25% | 76.75% | 0/1 |
| 500-step EMA | temporal ensemble | 0.010089 | 0.078104 | 100% | 100% | 0/1 |
| Published | open-loop 50 | 0.083606 | 0.000417 | 32.25% | 100% | 0/1 |
| 500-step EMA | open-loop 50 | 0.030627 | 0.083542 | 79% | 100% | 0/1 |

The converted expert data does not have this behavior. Across 5,682 frames its
left and right grippers are open 77.49% and 73.03% of the time. In the first 10
episodes, one gripper begins closing near frame 57 or 58. Expert continuous
action/state L2 has mean 0.045175, p90 0.140042, and maximum 0.179278.

The 500-step policy's action/state distance averaged 0.248994 over the first 10
rollout steps but only about 0.00044 over the last 100 steps. It approaches a
nearly static behavior without initiating a grasp. Fine-tuning also increased
continuous normalized-output saturation to 7.81%. This points to phase/action
learning or rollout alignment, not insufficient local training throughput.

The future chunks already contained a left-gripper close prediction in 97.25%
of published-checkpoint temporal-ensemble steps and 99.75% of 500-step steps.
Temporal ensemble nevertheless selected an open left gripper at every step.
Open-loop execution closed it first at step 62 for the published model and step
88 for the fine-tuned model. Neither rollout succeeded, so temporal ensemble is
a confirmed action-execution defect for this case but not the only cause of
task failure. The local 500-step fine-tune degraded the published open-loop
behavior and must not be promoted.

An experimental temporal-ensemble mode that selects the oldest aligned binary
prediction closed the left gripper at step 60, but also closed the right gripper
at step 116 and still failed. It demonstrates that aligned binary selection is
mechanically effective, while also showing that an unconditional oldest-action
rule is too aggressive. It remains a diagnostic mode and is not the default.

For the seed-100000 fine-tune, temporal ensemble exposed the same phase defect:
the expert closes the left gripper from frame 92 onward, and the model's future
chunk contained a close prediction in 386 of 400 temporal-ensemble steps, but
the selected chunk-head action remained open in every step. `open_loop_50`
allowed the learned phase to advance. EMA open-loop closed the gripper but
missed the hammer; raw open-loop reproduced the expert closely enough to
succeed. The successful gate therefore uses open-loop execution explicitly and
does not validate temporal ensemble as a promotion default.

## Compute decision and next gate

Multi-GPU training is not required for further frozen-encoder, single-task
debugging: 500 local steps are already inexpensive and success, not throughput,
is the blocker. Remote multi-GPU compute becomes useful for Clean50-scale data,
larger effective batches, multi-task sampling, or unfreezing encoders.

Before requesting cloud compute, the next phase should:

1. collect successful expert or DAgger corrections on multiple RoboTwin seeds;
2. train a mixed multi-seed candidate and require held-out-seed success;
3. add held-out action reconstruction metrics split into joints and grippers;
4. replace the phase-stalling temporal ensemble with a guarded stage-aware
   execution rule and compare it against explicit open-loop execution;
5. only then scale the validated recipe to multi-GPU training.

## Evidence locations

- Released baseline: `runs/turbovla-robotwin/released-baseline/`
- 100-step round: `runs/turbovla-robotwin/legacy-ft100/seed0/`
- 500-step training round: `runs/turbovla-robotwin/legacy-ft500/seed0/`
- 500-step immutable three-seed evaluation: seed 0 under
  `runs/turbovla-robotwin/legacy-ft500-eval/seed0/`, seeds 1 and 2 under
  `runs/turbovla-robotwin/legacy-ft500/`
- Published rollout trace:
  `runs/turbovla-robotwin/released-trace/seed0/`
- 500-step rollout trace with diagnostic metrics:
  `runs/turbovla-robotwin/legacy-ft500-trace-v2/seed0/`
- Published and 500-step open-loop A/B:
  `runs/turbovla-robotwin/released-openloop50/seed0/` and
  `runs/turbovla-robotwin/legacy-ft500-openloop50/seed0/`
- Experimental oldest-aligned binary execution:
  `runs/turbovla-robotwin/released-ensemble-oldest/seed0/`
- Seed-100000 expert source and converted dataset:
  `runs/robotwin-gate-expert/beat_block_hammer/gate_clean/` and
  `runs/robotwin-gate-lerobot/Clean/beat_block_hammer/`
- Seed-100000 EMA temporal/open-loop failures:
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16/seed0/` and
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-openloop/seed0/`
- Reproducible raw successes:
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-raw-openloop/seed0/` and
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-raw-openloop-repeat2/seed0/`
- Held-out seed-200000 failure:
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-raw-openloop/seed1/`
