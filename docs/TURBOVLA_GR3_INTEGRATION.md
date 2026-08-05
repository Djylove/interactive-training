# TurboVLA × Interactive Training × XPolicyLab × GR3

Status: local train, recorded replay, and promoted-checkpoint GR3 deployment
interface validated; real-robot action execution is not enabled.

## Decision

TurboVLA is a good local experimental VLA because its shared architecture is
configurable and substantially smaller than the current PI0.5/GR00T candidates.
All training is controlled by Interactive Training. TurboVLA is a model runtime,
not an independent orchestration path.

```text
Interactive Training
  ExperimentSpec + Goal + bounded knobs + DatasetBinding
        |
        v
XPolicyExperimentRunner
  manifest/profile/file SHA-256 validation
        |
        v
XPolicyLab policy/TurboVLA/train.sh
  refuses calls without the Interactive Training dataset identity
        |
        v
TurboVLA experiments.gr3.train
  one-view / joint-state-31 / learned-joint-action-31 GR3 model
  deployment adapter pads six zero base axes to canonical action-37
        |
        v
checkpoint ArtifactManifest -> replay -> simulation -> shadow -> enforce
```

## Why the released checkpoint cannot drive GR3 directly

The released RoboTwin recipe uses three cameras, a 14D state, 14D bimanual
absolute joint actions, and a 50-step chunk. The GR3 AnyGrasp dataset has one
top camera, a canonical 33D state, and a heterogeneous 37D action:
31 absolute joint/hand positions plus height/pitch/base velocity fields and one
absolute base yaw field.

The raw GR3 dataset retains its canonical 33D state and external 37D action.
After the 2026-08-05 real-robot audit, the learned model contract was reduced to
the 31 named joints on both sides. The reader drops the unused `base_height` and
`base_pitch` observations before normalization, and drops all six base command
axes from the learned loss. The inference adapter appends six zeros before
returning the canonical 37D chunk. This keeps the recorder/runtime contract
stable without feeding untrained world-frame base values into the policy.
Shape-compatible vision, text, projection, and interaction tensors may be loaded
from a released TurboVLA checkpoint; incompatible state/action/view tensors are
not silently reshaped. A released RoboTwin success score is not GR3 evidence.

## GR3 data contract

The primary profile is `xpolicylab.gr3_anygrasp_lerobot_v3`. The reader follows
the existing LeRobot v3 dataset rather than converting it to the old DAgger
recorder format:

- reads `observation.images.top`, `observation.state`, and `action` in place;
- verifies `robot_type=gr3qnexo`, 30 FPS, state 33D, and action 37D per batch;
- uses the curated clip manifest's absolute video-frame index and checks that it
  matches the parquet episode/frame index;
- uses each clip's `subtask`, falling back to its full prompt;
- builds future action chunks only inside the same curated clip and masks the
  unavailable tail;
- software-decodes the AV1 top-camera stream, center-crops it, and resizes it;
- normalizes state by mean/std and action by the selected frames' 1%/99% interval.

The raw 33D/37D schema above is a storage contract. TurboVLA receives and learns
only the first 31 named joint axes.

## 2026-08-05 real-robot contract correction

The first full-epoch 33D checkpoint passed offline reconstruction gates but
failed the attended real-robot trial: both arms lifted and did not approach the
tomato. Joint-order auditing found no permutation. A read-only Aurora sample
instead exposed a state-domain bug: the dataset's final two state axes had mean
zero and standard deviation `1e-4`, while deployment supplied world-frame
`base_pos_W.z=0.814436` and `base_pitch=0.011261`. Their normalized values were
approximately `8144` and `112.6`, dominating the state tokens.

The runtime now uses `gr3_joint_v1.state` (`31D`) for TurboVLA, while other GR3
models may continue using the canonical 33D observation. The trained checkpoint
was structurally projected from 33D/33D to 31D/31D by slicing only the two state
input axes and two action output axes; it is a deployment correction, not new
training. The projected checkpoint is:

```text
/home/ubuntu/xpolicylabdagger/policy/TurboVLA/checkpoints/GR3-deploy-v4-round-a-joint31-e1-lr1e4-s13200-gr3-joint31_canonical37-0/model_final.pt
SHA-256: 0145f79dca8bff3411c117f63320efd13f53b55c92500e277df9db7f16945b58
```

A local WebSocket shadow smoke accepted `31D`, returned finite `50x37`, kept
axes 31–36 at zero, and measured warm latencies of `38.61` and `22.56 ms`.
No robot command was issued. The next cloud round must train the native 31D/31D
head rather than relying permanently on checkpoint projection.

GR3 has no wrist cameras in this dataset. The model profile therefore has one
view; it must not duplicate the top image to imitate a three-camera robot.

`xpolicylab.gr3_dagger_v2` is retained only as a legacy compatibility path for
old recordings. DAgger intervention and expert-safe-label semantics do not
apply to AnyGrasp.

## Training control and audit

`ExperimentSpec.dataset_manifest_id` and the provided manifest path must match.
The runner recomputes manifest digests and verifies every manifested selection
index and batch metadata SHA-256 before launching the process. Raw video and
action parquet remain read-only in their existing dataset root and are checked
for schema/index consistency when loaded; they are not duplicated into the
project. The runner creates a per-round immutable manifest snapshot and records
`DatasetBinding` in the checkpoint artifact.

`policy/TurboVLA/train.sh` requires the following runner-provided values:

- `XPOLICYLAB_DATASET_MANIFEST`, ID, digest, and profile;
- `TURBOVLA_MAX_STEPS` and `TURBOVLA_LEARNING_RATE`, mapped from `TrainSpec`;
- allowlisted local paths for TurboVLA, DINOv3, BERT, and optional initialization;
- explicitly bounded batch size and visible GPU IDs.

Direct invocation without the Interactive Training binding fails before Python
training starts.

## Evaluation and deployment gates

1. CPU data preflight: dimensions, timestamps, masks, normalization.
2. One-GPU overfit smoke: 10–100 steps, batch 1–2, frozen text and vision
   encoders; verify decreasing finite loss and checkpoint reproducibility.
3. Recorded replay: raw 50×37 finite output through XPolicyLab's WebSocket
   contracts. Replay pass rate is protocol evidence, not task success.
4. Simulation: add an actual GR3 simulator adapter. RoboTwin's 14D robot and
   RoboDojo's other embodiments cannot substitute for GR3 dynamics.
5. Robot shadow: inference only; compare against expert and executed actions.
6. Robot enforce: separate human approval. Output must pass the existing GR3
   router, joint/base limits, filter/QP/Mink, watchdog, and takeover path.

No policy process may write directly to Aurora.

## Local versus cloud compute

Inference should fit comfortably on the RTX 5090 once the assets are available.
The first smoke run freezes DINOv3 and BERT and uses batch 1–2, so it should also
fit locally. Full unfreezing, large batches, multi-task data, or a statistically
meaningful training run may need remote GPUs, but cloud compute is not justified
until the local overfit smoke has passed.

The local development environment is `/home/ubuntu/TurboVLA/.venv`. The
released RoboTwin checkpoint, DINOv3 ViT-L, and BERT base assets are installed
under `/home/ubuntu/TurboVLA/pretrained`. GroundingDINO initialization is
optional and has not been installed.

## Validated local smoke

On 2026-08-01, Interactive Training launched GR3 runs through XPolicyLab using
the manifest-bound episode. The 10-step batch-1 run took about 10 seconds and
reduced loss from `0.664165` to `0.645011` (about 2.9%). It produced a 1.74 GB
checkpoint; 199 released tensors were shape-compatible and 679 were
deliberately skipped. GPU spot measurements during the frozen DINOv3/BERT run
were below 3 GB for the TurboVLA process.

Recorded replay passed the WebSocket/action contract with a 50×37 output,
`replay_pass_rate=1.0`, and 131.0 ms P95 inference latency for the small replay
sample. This validates plumbing and local feasibility only. Ten optimization
steps and two replayed frames are not model-quality evidence.

## Validated cloud PPU single-card smoke

On 2026-08-03, the deployed TurboVLA current runtime completed a two-step
single-card train smoke on one `PPU-ZW810E` through its CUDA-compatible PyTorch
API. The test exercised the real TurboVLA text encoder, vision encoder,
vision-language interaction, ACT action head, MSE loss, backward pass, AdamW
update, checkpoint save, and checkpoint reload. It used BF16 autocast where
configured. This first synthetic device test used three tiny image tensors and
therefore did not validate the actual one-camera GR3 data contract.

To avoid treating a weight download as device validation, this first PPU test
used tiny randomly initialized BERT and DINOv3 models and froze both backbones.
It is therefore a kernel/runtime compatibility test, not a quality benchmark.
The two losses were `0.401533` and `0.331395`; gradient norms were finite, a
tracked trainable parameter changed by `0.004003`, and checkpoint round-trip
validation passed. The first step took 13.8 seconds while PPU kernels were
compiled; the warm second step took 24 ms. Peak framework-reported reserved
memory was about 28 MiB for this deliberately tiny model.

The remote report and checkpoint are stored below the isolated deployment root:

```text
/mnt/workspace/jiayu/autogoal/runs/ppu-turbovla-single-card/20260803T042037Z/
```

Reproduction from the remote deployment root:

```bash
cd /mnt/workspace/jiayu/autogoal
source ./activate_ppu.sh
python interactive-training/scripts/ppu_turbovla_single_card_smoke.py \
  --workspace /mnt/workspace/jiayu/autogoal \
  --steps 2
```

## Validated AnyGrasp real-data PPU smoke

On 2026-08-03, a second PPU smoke used the real AnyGrasp data through the full
`Interactive Training -> XPolicyLab -> TurboVLA` launch path. A reference-only
manifest bound the first 32 curated clips: 2,592 frames across 14 source batches.
The manifest was 6.5 KB; no source video or action parquet was copied.

Preflight decoded the real AV1 top-camera frame and produced `224x224x3`, 33D
state, and `50x37` action-chunk tensors. The two-step training test used the tiny
random backbones, a 32-pixel input, and a four-step action chunk. It completed
with finite losses `0.472793` and `0.590810` on different shuffled samples and
registered an 80,386,798-byte checkpoint. Two shuffled steps are compatibility
evidence only; the loss values are not a convergence claim.

The registered checkpoint was then loaded through the neutral
`gr3_top_rgb_v1` runtime with a real `480x832` top-camera frame and raw 33D
state. Inference returned a finite `4x37` action chunk, confirming that the new
checkpoint no longer depends on a DAgger dataset identity.

```text
dataset: gr3-anygrasp-top-32-f22046f86ea3
profile: xpolicylab.gr3_anygrasp_lerobot_v3
checkpoint: /mnt/workspace/jiayu/autogoal/xpolicylabdagger/policy/TurboVLA/checkpoints/GR3-anygrasp32-tiny-smoke-gr3-canonical37-0
logs: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-top-32/interactive-logs/
```

The next PPU gate must replace the tiny random backbones with released
BERT/DINOv3 assets and run a longer held-out-aware experiment. Neither PPU smoke
report is evidence of convergence or robot readiness.

## Validated released-backbone PPU smoke

On 2026-08-03, the same manifest-bound AnyGrasp selection completed a two-step
PPU train using released BERT base, DINOv3 ViT-L, and the released TurboVLA
RoboTwin checkpoint as shape-compatible initialization. Interactive Training
and XPolicyLab registered a 1,735,925,722-byte checkpoint with return code zero.

The initialization loader accepted 199 compatible tensors and deliberately
skipped 679 incompatible tensors, primarily because GR3 uses one view and a
33D/37D state-action contract instead of RoboTwin's three-view 14D/14D head.
Loss was finite and changed from `0.630241` to `0.525800`. The steps used
different shuffled samples, so this is not a controlled convergence estimate.

The registered checkpoint was loaded through `gr3_top_rgb_v1` and ran inference
on a real `480x832` AnyGrasp frame plus raw 33D state. It returned a finite
`4x37` action chunk. The PPU was empty after both training and inference.

```text
checkpoint: /mnt/workspace/jiayu/autogoal/xpolicylabdagger/policy/TurboVLA/checkpoints/GR3-anygrasp32-official-smoke-gr3-canonical37-0
logs: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-official/interactive-logs/
inference report: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-official/inference-smoke.json
```

This released-backbone smoke was followed by the task-disjoint pilot below.

## Validated task-disjoint 100-step PPU pilot

On 2026-08-03, the AnyGrasp curated selection was split by `task_id`, before
training and normalization, using a deterministic SHA-256 assignment with seed
zero. The complete reference-only manifests contain 78 train tasks (1,026
clips, 83,106 frames) and 19 held-out tasks (209 clips, 16,929 frames). Their
task sets are disjoint and their union covers all 97 task IDs present in the
curated selection. A 256-clip pilot subset of the train partition contains
20,736 frames across 47 tasks. No source media or parquet was copied or changed.

Interactive Training launched a single-card PPU run using released BERT base,
DINOv3 ViT-L, and shape-compatible TurboVLA initialization. Both backbones were
frozen; the action horizon was 50, batch size was 2, learning rate was `5e-5`,
and training stopped at 100 optimizer steps. The sampled training loss was
finite throughout and changed from `0.554684` at step 1 to `0.221258` at step
100. Because these are different shuffled batches, the change is an operational
training signal rather than a controlled before/after quality metric.

The checkpoint was registered with step, manifest identity, and SHA-256. Its
`model_final.pt` is 1,735,972,826 bytes. The offline evaluator then used the
checkpoint's train-only normalization and deterministic task-round-robin sample
selection. Fifty train samples covered all 47 pilot tasks; fifty held-out
samples covered all 19 unseen tasks.

| partition | tasks | zero-step L1 | 100-step L1 | improvement | warm P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pilot train | 47/47 | 0.512545 | 0.308595 | 39.79% | 32.60 ms |
| task held-out | 19/19 | 0.502926 | 0.349810 | 30.44% | 31.99 ms |

The zero-step baseline uses the same config, train-only normalization, model
seed, sample indices, and the same 199 shape-compatible released tensors, but no
optimizer update. The 100-step pilot improves held-out L1 by 30.44%, while the
post-training held-out error remains 13.36% above pilot-train error. Cold-start
inference was about 2.6 seconds because PPU kernels were initialized; it is
reported separately from warm latency. All predictions were finite with shape
`50x37`. This establishes a reproducible task-generalization baseline, not task
success or robot readiness. Offline action L1 cannot replace simulation,
shadow, or robot evaluation.

```text
split manifests: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-task-split/
checkpoint: /mnt/workspace/jiayu/autogoal/xpolicylabdagger/policy/TurboVLA/checkpoints/GR3-anygrasp-task-pilot256-s100-gr3-canonical37-0
artifact/logs: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-task-pilot256-s100/
train report: .../train-eval-stratified.json
held-out report: .../heldout-eval-stratified.json
zero-step reports: .../train-eval-zero-step.json, .../heldout-eval-zero-step.json
```

The next optimization gate therefore introduced a separate validation task
partition. Hyperparameters and training duration must be selected on validation;
held-out tasks are opened only once after promotion.

## Validated three-way staged iteration

On 2026-08-03, the deterministic task split was extended to train, validation,
and sealed held-out partitions while preserving the previous 19 held-out task
IDs exactly. The complete partitions contain 68 train tasks (903 clips, 73,143
frames), 10 validation tasks (123 clips, 9,963 frames), and 19 held-out tasks
(209 clips, 16,929 frames). A 256-clip train pilot contains 20,736 frames across
43 tasks. All three task sets are disjoint and their union is the curated
selection's 97 tasks.

Two equal-budget 100-step candidates differed only in learning rate. Fixed
task-round-robin validation used 50 samples, five from each validation task:

| candidate | validation normalized L1 |
| --- | ---: |
| `lr=5e-5`, step 100 | 0.327461 |
| `lr=1e-4`, step 100 | 0.324633 |

The `1e-4` candidate won by 0.86% and was retrained from the same released
initialization for 1,000 steps. Validation continued to improve to 0.241974 at
step 500 and 0.187984 at step 1,000. The final checkpoint therefore improved
42.09% over its 100-step screen and was promoted. The run took about 59 minutes
on one PPU and was registered with all 100-step milestone weights and SHA-256
digests.

The sealed held-out partition was then evaluated once. Fifty samples covered
all 19 unseen tasks and produced normalized L1 0.196614, only 4.59% above
validation. Warm inference mean/P95 were 28.65/31.53 ms; all `50x37`
predictions were finite. Held-out was not used for candidate selection.

An attempted four-worker AV1 DataLoader failed before the first optimization
step because the remote filesystem does not support Python multiprocessing's
resource-sharing socket (`OSError 95`). It was terminated without deleting the
audit trail and the successful run used the already validated single-worker
loader. This is a throughput limitation, not a model or dataset failure.

```text
three-way manifests: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-three-way-v1/
iteration evidence: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-iteration-v1/
promotion record: .../iteration-summary.json
checkpoint: /mnt/workspace/jiayu/autogoal/xpolicylabdagger/policy/TurboVLA/checkpoints/GR3-anygrasp-v1-lr1e4-s1000-w0-gr3-canonical37-0/model_final.pt
checkpoint SHA-256: bd91480cbc41bd41a68c751985e866b162d5c83ca4124ec4bd6dbf00200f23bb
```

This is offline action-prediction evidence. The promoted model must still pass
recorded replay, simulation/digital-twin evaluation, and robot shadow safety
gates before any enforce-mode deployment.

## Validated promoted-checkpoint AnyGrasp replay

The promoted 1,000-step checkpoint passed XPolicyLab's recorded replay gate on
2026-08-03. The previous replay reader accepted only legacy DAgger camera
parquet, so an explicit `gr3_anygrasp_reference_v1` episode was added. It stores
only dataset identity, deterministic sample indices, and source paths below the
isolated project root; AV1 video and state parquet remain read-only under
`/mnt/workspace/jmy` and are not copied.

The reference contains one real frame from each of ten validation tasks. Each
frame was decoded at its original `480x832` resolution, paired with its raw 33D
state and per-clip instruction, sent through XPolicyLab's WebSocket policy
server, and returned as a finite `50x37` GR3 action chunk. The result was one
valid `replay_validation` trial, ten of ten frames, `replay_pass_rate=1.0`, and
zero invalid trials.

Cold PPU/model startup was 2.579 seconds. After warm-up, mean inference latency
was 35.56 ms and P95 was 39.44 ms. The original aggregate latency of about 290
ms included the cold first frame, so cold and warm metrics are now reported
separately. A second full replay produced the same ten per-frame action SHA-256
prefixes in the same order, providing deterministic protocol evidence.

```text
reference episode: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-promoted-replay/reference-episode/
audited result: /mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-promoted-replay/interactive-round-audit/result.json
result SHA-256: 388de9ad5e98a62a3e6b7a68143faff04f57eddecf1e041b3fc0e00c95c634ec
checkpoint artifact ID: e72aed7132e2a622026ab419e882834000a254fe1d47daf2273c52ce06647402
```

Replay does not execute actions and does not establish physical safety or task
success. The raw action chunks still require GR3 limits, filtering/QP/Mink,
watchdog, takeover, shadow comparison, and explicit enforce authorization.

## Validated local GR3 deployment interface

On 2026-08-03, the promoted checkpoint was downloaded from the isolated PPU
workspace to the local XPolicyLab policy directory and verified against the
promotion SHA-256. The already installed frozen assets under
`/home/ubuntu/TurboVLA/pretrained` were reused after their DINOv3-L and BERT
file hashes matched the cloud copies; they were not downloaded a second time.

TurboVLA was added to the common `scripts/gr3.sh serve/dagger/replay` launcher.
It uses the same mature `gr3-policy-dagger.yml` graph as GR00T, FastWAM, and
PI0.5: camera, QNexo, robot adaptor, XPolicyLab GR3 bridge, output pipeline,
DAgger router, recorder, and teleop remain model-independent. The policy server
is the only new runtime process; no Aurora or robot-control implementation was
added or changed.

The XPolicyLab-owned GR3 Python environment and Rust recorder were rebuilt
locally. Twenty model/launcher/transport tests and 24 tests for the graph's
camera, QNexo, router, and robot-adaptor components passed. A local RTX 5090
PolicyServer then loaded the exact promoted checkpoint and received three
synthetic contract-only requests through the standard WebSocket client. All
three outputs were finite `50x37` chunks with the canonical GR3 action schema
and identical hashes. Cold latency was 617.90 ms; the two warm requests were
15.26 and 13.32 ms. GPU memory after warm inference was about 1.93 GiB.

The synthetic requests validate deployment plumbing only, not model quality or
robot behavior. The GR3 hardware graph was not started in this validation.

```text
local checkpoint: /home/ubuntu/xpolicylabdagger/policy/TurboVLA/checkpoints/GR3-anygrasp-v1-lr1e4-s1000-w0-gr3-canonical37-0/model_final.pt
checkpoint SHA-256: bd91480cbc41bd41a68c751985e866b162d5c83ca4124ec4bd6dbf00200f23bb
local endpoint: ws://127.0.0.1:19000
native output: 50 x 37 at 30 Hz
execution queue default: 16 steps, refill trigger at 4 remaining steps
```

Deployment commands:

```bash
cd /home/ubuntu/xpolicylabdagger
runtime/dagger_gr3/scripts/setup_local.sh  # first deployment only
scripts/gr3.sh serve turbovla
scripts/gr3.sh dagger turbovla --dry-run
```

Removing `--dry-run` starts the actuation-capable mature GR3 graph and therefore
requires an operator-attended robot session and separate explicit approval.

## Validated local real-hardware shadow preflight

On 2026-08-04, the promoted TurboVLA checkpoint passed an operator-attended,
write-disabled GR3 shadow preflight. The real OAK camera, Aurora state reader,
QNexo exoskeleton, footswitch, TurboVLA WebSocket server, safety guard, and
deployment probe were run together through `gr3-model-exo-shadow.yml`. The
graph hard-disabled Aurora writes with `GR3_ENABLE_WRITE=0`, limited inference
to one action, and did not run the lifecycle node. The successful probe marker
was:

```text
GR3_MODEL_EXO_SHADOW_OK {"driver":"hardware_write_disabled","operator":"exoskeleton_step","safe_action_count":1}
```

The OAK camera produced stable `640x400` frames from serial
`19443010E1003A2E00`; its raw USB image is inverted and the production graph's
`ROTATE=180` setting corrects it. Aurora returned all 31 joints for robot 115.
After moving QNexo and the footswitch onto separate USB controllers, a 20-second
QNexo read test observed 9,642 updates at 482.09 Hz with a maximum sample age
of 2.061 ms and no disconnect. The left-blue exoskeleton step event and all
three foot pedals were independently observed. The footswitch mapping was
left=`left_save`, middle key code 57, and right=`right_takeover`.

After graph shutdown, 30 read-only Aurora samples showed a maximum joint span
of `7.13e-06 rad`; lifecycle remained `Default / Default / Joystick`, and no
Dora, teleoperation, or robot-writer process remained. Thirty-seven relevant
GR3 launcher, lifecycle, operator, driver, safety, shadow, and result-bridge
tests passed after the local hardware dependencies were installed. The
operator then disconnected the robot, exoskeleton, and footswitch, so their
subsequent offline state is expected and is not a runtime failure. The local
TurboVLA policy server was also stopped normally to release its GPU allocation.

This validates device access, event routing, inference plumbing, single-action
safety filtering, and clean shutdown. It does not validate grasp success and
does not authorize robot actuation. The standalone validation graph is no longer
the operational deployment path. Robot sessions use the latest `dagger-gr3`
robot runtime copied into `runtime/dagger_gr3/graphs/gr3-policy-dagger.yml`, with
only the policy client replaced by XPolicyLab's WebSocket bridge. The first
TurboVLA gate explicitly sets `ACTION_CHUNK_SIZE=1`,
`INFERENCE_TRIGGER_STEPS=0`, `MAX_JOINT_STEP_RAD=0.03`, and `FREEZE_WAIST=1`;
chunk length and prefetch increase only after operator-attended single-step
validation.

## Commands

Build a reference-only AnyGrasp manifest inside the isolated deployment:

```bash
PYTHONPATH=xpolicylabdagger python \
  -m policy.TurboVLA.anygrasp_dataset_manifest \
  --workspace-root /mnt/workspace/jmy \
  --dataset-root /mnt/workspace/jmy/dataset/anygrasp_v2 \
  --selection-root \
    /mnt/workspace/jmy/Data/outputs/anygrasp_v2_dense_100k_20260714_81f \
  --clip-limit 32 \
  --output runs/gr3-anygrasp-top-32/dataset-manifest.json
```

Launch manifest-bound training with
`examples/xpolicylab_turbovla_gr3_train.py`, providing the released DINOv3 and
BERT paths. This train-only entry still uses `XPolicyExperimentRunner.train`;
it does not bypass Interactive Training.

Recorded replay and robot evaluation remain separate gates after training.
