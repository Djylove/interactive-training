# Interactive Training × XPolicyLab 具身训练闭环开发设计

状态：Phase 1 开发中  
目标版本：MVP v0.1  
本地开发设备：NVIDIA GeForce RTX 5090，32 GB 显存  
相关仓库：

- Interactive Training：训练控制、指标、动作、Agent 与跨轮记忆；
- XPolicyLab：策略适配、训练入口、PolicyServer、仿真/真机评测与 DAgger 记录。

当前实现状态（2026-07-31）：

- 已实现 Phase 0 契约、受限 runner、评测聚合、权限/晋级规则和单轮实验编排；
- 已实现不依赖 GPU 的 fake/stub 闭环测试；
- `examples/xpolicylab_contract_smoke.py` 可配合 XPolicyLab `demo_policy` 验证合约；
- 已实现 XPolicyLab debug PolicyServer 的真实 `trial_end` + JSONL bridge；
- 已实现 RoboDojo `_result.json` 到 `TrialResult` 的仿真结果桥接及 `evaluate_sim()`；
- 已实现已有 checkpoint 登记、GR3 recorded replay 门禁及独立 `replay_pass_rate`；
- AHA_WAM Trainer、DAgger 和真机路径尚未接入；本机尚未执行 Isaac Sim 实测。

合约 smoke 命令：

```bash
python -m examples.xpolicylab_contract_smoke \
  --xpolicylab-root /path/to/XPolicyLab
```

该示例生成的 trial 是明确标注的 synthetic 数据，只验证编排和产物追踪，不能作为
模型效果或 benchmark 证据。

真实 debug WebSocket 闭环：

```bash
python -m examples.xpolicylab_debug_round \
  --xpolicylab-root /path/to/XPolicyLab \
  --policy-env <demo-policy-conda-env> \
  --evaluation-env <debug-client-conda-env>
```

该路径会启动 XPolicyLab `demo_policy` PolicyServer，运行无 simulator 的 debug
episode，发送 `trial_end` 并由 Interactive Training 读取 JSONL。debug outcome 固定为
`invalid`，预期汇总状态是 `inconclusive`。

RoboDojo 仿真闭环（会启动仿真，请在环境准备好后执行）：

```bash
python -m examples.xpolicylab_sim_round \
  --xpolicylab-root /path/to/XPolicyLab \
  --policy-env <policy-conda-env> \
  --evaluation-env <robodojo-conda-env> \
  --policy demo_policy \
  --task stack_bowls \
  --repeats 2
```

Controller 会把 `repeats` 映射为 RoboDojo 的 `EVAL_NUM`。仿真正常结束后，XPolicyLab
读取 RoboDojo 权威 `_result.json`，核对 `eval_time`、逐 episode 明细与
`success_rate`，再发送 `trial_end` 并写入本轮 JSONL。RoboDojo 判定为 unstable 的布局
不会出现在正式明细中；有效样本不足时聚合状态为 `inconclusive`，不会写入成功率目标。

已有 checkpoint 的 GR3 recorded replay（不会启动机器人或 Isaac Sim）：

```bash
python -m examples.xpolicylab_replay_round \
  --xpolicylab-root /path/to/XPolicyLab \
  --policy Pi_05 \
  --checkpoint-name <existing-run-name> \
  --policy-env <pi05-policy-env> \
  --evaluation-env <replay-env-with-pyarrow> \
  --episode /path/to/episode_000000019 \
  --task recorded_gr3
```

该路径使用 `TrainSpec(enabled=False)`，只登记并计算已有 checkpoint 清单摘要，不会
调用 `train.sh`。每个录制 episode 会检查 observation 解码、PolicyServer 推理、动作
有限性、`ActionChunk` schema 稳定性和延迟，并发送
`evidence_type: replay_validation`。通过率写入 `replay_pass_rate`；它可以作为离线晋级
门禁，但不会生成或覆盖 `success_rate`。当前机器未挂载文档中记录的 GR3 episode，
因此本轮只完成合成契约测试，没有宣称真实 checkpoint replay 已复验。

## 1. 目标

在不耦合两个仓库运行环境的前提下，形成以下可审计闭环：

```text
实验配置
  -> XPolicyLab 训练入口
  -> checkpoint
  -> PolicyServer
  -> 仿真/真机评测
  -> 标准评测结果
  -> Interactive Training 记忆与决策
  -> 下一轮配置或数据策略
```

MVP 首先证明控制协议、产物追踪和评测回传正确，不以本地完成大型 VLA/WAM
全参数训练为验收条件。

## 2. 非目标

v0.1 不做以下工作：

- 在 RTX 5090 上复现大型 VLA/WAM 的正式训练结果；
- 把 XPolicyLab 和各 policy 的依赖合并到 Interactive Training 环境；
- 用 Interactive Training 替代 Accelerate、DeepSpeed、FSDP 或原生 Trainer；
- 让 Agent 直接控制真机、安全限位、急停或 Mandatory Safety Guard；
- 一次性适配 XPolicyLab 中的全部 policy；
- 建设集群调度、GPU 租赁、多租户权限或通用模型注册平台；
- 在第一阶段重构 GR3 DAgger runtime。

## 3. 设计原则

### 3.1 控制面与执行面分离

Interactive Training 是控制面，保存实验意图、参数边界、指标和决策历史。
XPolicyLab 是执行面，负责训练脚本、模型服务、环境协议、动作语义和机器人安全。

两个项目通过版本化 JSON 产物和本机 HTTP/进程协议连接，不互相导入对方的大型依赖。

### 3.2 训练、部署质量和安全参数分层

| 参数层级 | 示例 | Agent 权限 |
| --- | --- | --- |
| 训练参数 | 学习率、loss 权重、数据混合、LoRA 配置 | 可在明确上下界内修改 |
| 部署质量参数 | chunk overlap、滤波 alpha、QP 权重、推理频率 | v0.1 只建议或 shadow |
| 强安全参数 | 关节/速度限制、watchdog、急停、最终安全检查 | 永不开放 |

### 3.3 先离线、再仿真、后真机

任何新控制动作都依次经过：

```text
schema/unit test -> debug client -> recorded replay -> simulator -> shadow -> human approval -> enforce
```

### 3.4 不把一次评测当作训练改进证据

每个 checkpoint 至少保留 seed、任务、场景、重复次数和失败原因。结果不足时标为
`inconclusive`，不得仅用单次 success/failure 驱动 Agent 更新配置。

## 4. 总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│ Interactive Training Experiment Controller                 │
│ knobs / actions / goals / journal / Agent / artifact index │
└──────────────┬───────────────────────────────▲──────────────┘
               │ ExperimentSpec                │ EvaluationResult
               ▼                               │
┌──────────────────────────┐        ┌──────────┴──────────────┐
│ XPolicy ExperimentRunner │        │ Evaluation Reporter     │
│ argv/env/process/artifact│        │ aggregate + validate    │
└──────────────┬───────────┘        └──────────▲──────────────┘
               │                               │
               ▼                               │ trial_end
┌──────────────────────────┐   WS   ┌──────────┴──────────────┐
│ train.sh -> checkpoint   │ ----> │ PolicyServer + env      │
│ policy-specific env      │       │ debug/sim/replay/robot  │
└──────────────────────────┘       └─────────────────────────┘
```

### 4.1 进程所有权

- Experiment Controller 由 Interactive Training 进程持有并在一轮训练和评测之间保持存活；
- ExperimentRunner 只用参数数组启动受允许的 XPolicyLab 入口，不拼接 shell 字符串；
- policy 继续运行在自己的 conda/uv 环境；
- simulator/robot client 继续运行在环境侧依赖中；
- PolicyServer 继续使用 XPolicyLab WebSocket 协议；
- Reporter 只接受结构化结果，不解析自由格式控制台文本作为正式接口。

## 5. 核心数据契约

所有契约初始版本使用 `xpolicy_interactive.v1`，未知主版本必须拒绝。

### 5.1 ExperimentSpec

```json
{
  "schema_version": "xpolicy_interactive.v1",
  "experiment_id": "exp-20260731-001",
  "round": 1,
  "policy": "AHA_WAM",
  "bench_name": "RoboDojo",
  "env_cfg_type": "arx_x5",
  "action_type": "joint",
  "seed": 1,
  "mode": "local_smoke",
  "train": {
    "enabled": true,
    "max_steps": 20,
    "learning_rate": 0.00001,
    "dataset_mix": {}
  },
  "evaluation": {
    "environment": "debug",
    "tasks": ["stack_bowls"],
    "repeats": 2
  },
  "parent_checkpoint": null
}
```

### 5.2 ArtifactManifest

```json
{
  "schema_version": "xpolicy_interactive.v1",
  "experiment_id": "exp-20260731-001",
  "round": 1,
  "status": "completed",
  "source": {
    "interactive_training_commit": "<sha>",
    "xpolicylab_commit": "<sha>",
    "policy": "AHA_WAM"
  },
  "checkpoint": {
    "path": "<absolute-or-artifact-uri>",
    "sha256": "<sha256>",
    "step": 20
  },
  "logs": [],
  "started_at": "<iso8601>",
  "finished_at": "<iso8601>"
}
```

大型分片 checkpoint 不要求对整个目录做单文件 hash。应生成稳定排序的文件清单，
记录每个文件的相对路径、大小和 SHA-256，再对清单本身计算摘要。

### 5.3 TrialResult

```json
{
  "schema_version": "xpolicy_interactive.v1",
  "evaluation_id": "eval-001",
  "checkpoint_id": "ckpt-001",
  "trial_id": "stack_bowls-seed1-repeat0",
  "task": "stack_bowls",
  "seed": 1,
  "repeat_index": 0,
  "outcome": "success",
  "success": 1.0,
  "episode_steps": 183,
  "termination_reason": "task_success",
  "metrics": {
    "inference_latency_p50_ms": 65.1,
    "inference_latency_p95_ms": 74.2,
    "takeover_count": 0,
    "collision_count": 0,
    "safety_hold_count": 0
  },
  "artifacts": {
    "video": null,
    "trajectory": null
  }
}
```

`outcome` 至少支持：`success`、`failure`、`aborted`、`timeout`、`invalid`。
只有 `success` 和 `failure` 默认进入成功率分母，其余状态单独统计。

### 5.4 EvaluationSummary

正式交给 Goal/Agent 的是聚合结果：

- `success_rate`；
- `valid_trials`、`invalid_trials`；
- `task_success_rate/<task>`；
- `worst_task_success_rate`；
- `takeover_rate`；
- `collision_rate`；
- `safety_hold_rate`；
- `inference_latency_p95_ms`；
- 可选置信区间。

主目标建议使用 `success_rate` 或 `worst_task_success_rate`，延迟与安全指标作为硬约束，
而不是全部压成一个难以解释的加权分数。

## 6. Interactive Training 侧组件

建议新增可选模块，核心包不依赖 XPolicyLab：

```text
src/interactive_training/integrations/xpolicylab/
├── __init__.py
├── contracts.py       # Pydantic 契约
├── runner.py          # 受限子进程与产物管理
├── reporter.py        # trial 聚合和指标上报
├── experiment.py      # train/deploy/evaluate 生命周期
└── policies.py        # 权限、边界和晋级规则
```

### 6.1 XPolicyExperimentRunner

最小接口：

```python
class XPolicyExperimentRunner:
    def train(self, spec: ExperimentSpec) -> ArtifactManifest: ...
    def start_policy_server(self, spec, artifact) -> PolicyHandle: ...
    def evaluate(self, spec, handle) -> EvaluationSummary: ...
    def stop(self, handle) -> None: ...
```

要求：

- policy、脚本入口和环境必须来自显式 allowlist；
- 使用 `subprocess` 参数数组，禁止拼接任意命令；
- 保存退出码、启动参数、环境变量白名单、stdout/stderr 路径和超时原因；
- 只向子进程传必要环境变量，不传 API key 等无关凭据；
- 超时后先温和终止，再有界等待，最后记录强制终止；
- 不自动删除失败产物。

### 6.2 指标回传

v0.1 不新增公网可访问的通用 `/metrics` API。推荐两种方式：

1. 同一 Controller 进程内由 Reporter 调用 `session.report_eval(summary)`；
2. 跨环境时，使用 loopback-only 的受限 bridge，把 `TrialResult` 写入本轮 spool
   目录，由 Controller 校验并消费。

不得把当前无认证的 Interactive Training HTTP transport 直接暴露到外网或机器人网络。

### 6.3 多轮语义

Interactive Training 当前 `run_rounds()` 更适合每轮从相同初始化开始的对照实验。
具身模型训练昂贵，MVP 采用自定义实验循环，支持两种模式：

- `independent`：每轮从同一基础 checkpoint 开始，用于公平超参对比；
- `continual`：从上一轮最佳 checkpoint 继续，用于 DAgger 数据增量闭环。

日志必须明确保存 `parent_checkpoint`，不得混淆两种结果。

## 7. XPolicyLab 侧接入点

### 7.1 外层训练编排（MVP）

优先复用：

```text
policy/<POLICY>/train.sh
policy/<POLICY>/eval.sh
policy/<POLICY>/deploy.yml
setup_policy_server.py
```

第一阶段只控制脚本已公开的参数和经审核的环境变量，不修改上游 Trainer。

### 7.2 PolicyServer 评测生命周期

XPolicyLab 已支持 `prepare_case`、`reset`、`infer` 和 `trial_end`。环境侧应在每个
trial 结束时发送完整 `TrialResult`，PolicyServer 的 `on_trial_end` 只作模型状态清理，
不作为 success 的事实来源。

### 7.3 深度 Trainer 接入（MVP 之后）

AHA_WAM 可在真实 optimizer update 后显式调用 `session.step()`。不使用 optimizer
monkeypatch，原因包括 Accelerate、DeepSpeed、梯度累积和 scheduler 包装。

分布式约束：

- 所有 rank 在同一个 optimizer step 到达控制点；
- 只有 rank 0 启动 transport 和 Agent；
- knob 在所有 rank 注册，决策由 rank 0 广播；
- 动态 LR 必须同步更新包装后的 optimizer/scheduler；
- save/load 由原 Trainer 实现，Interactive Training 只发出请求和登记结果；
- 在 DeepSpeed 路径验证完成前，load checkpoint 不开放给 Agent。

## 8. 5090 本地开发策略

当前设备约 32 GB 显存。本地目标是高反馈、低成本验证，不承诺容纳 AHA_WAM、
LingBot VLA 等大型模型的正式训练和完整评测。

### 8.1 本地允许的工作负载

- 纯 CPU 契约、runner、权限和产物测试；
- XPolicyLab `demo_policy` debug evaluation；
- recorded observation/action replay；
- 小模型或 Diffusion Policy 的短步 smoke train；
- 冻结大部分参数的 LoRA/adapter smoke test（仅在实际显存探测通过后）；
- 单环境、少 episode 仿真；
- 已存在 checkpoint 的单次/短序列推理；
- 使用假 checkpoint 和假 PolicyServer 验证失败恢复。

### 8.2 默认禁止的本地任务

- 大型 WAM/VLA 全参数训练；
- 为验证协议而下载数百 GB 权重或数据；
- 未设步数、时间和磁盘上限的训练；
- 多个大型 policy/server 同时驻留 GPU；
- 未验证余量时同时运行 simulator 与大模型推理；
- 自动从失败 checkpoint 反复重试。

### 8.3 资源预算

本地 profile 建议默认值：

```yaml
profile: local_5090_smoke
limits:
  gpu_count: 1
  gpu_memory_soft_limit_gib: 28
  wall_time_minutes: 30
  train_max_steps: 20
  eval_tasks: 1
  eval_repeats: 2
  concurrent_policy_servers: 1
  artifact_disk_gib: 20
```

保留约 4 GB 显存余量只是初始软限制，不是安全保证。运行前后记录
`nvidia-smi`、峰值显存、主机内存和磁盘余量；OOM 只降级 workload，不自动改用
CPU 跑大型模型。

### 8.4 远程正式运行

远程 runner 与本地使用同一 `ExperimentSpec` 和 `ArtifactManifest`，仅替换执行后端。
正式训练至少要求：

- policy README 声明的 GPU/驱动/CUDA 环境；
- 固定代码 commit、基础 checkpoint 和数据版本；
- 可恢复的分片 checkpoint；
- 训练与评测日志回传；
- GPU-hour、token/cost（如有）和失败状态记账；
- checkpoint 晋级后才允许进入仿真或真机队列。

## 9. Agent 权限与晋级门禁

v0.1 默认权限：

| 动作 | 人类 | Agent |
| --- | ---: | ---: |
| 修改有界训练 knob | 是 | 是 |
| 启动 local smoke | 是 | 可建议，Controller 执行 |
| 选择远程正式训练 | 是 | 否 |
| checkpoint 晋级到仿真 | 是 | 否 |
| 修改 shadow 参数 | 是 | 仅建议 |
| 进入真机 enforce | 是，需审批 | 否 |
| 修改 Mandatory Safety Guard | 否 | 否 |

由于当前 Interactive Training 对自定义 action 没有通用 RBAC，具身集成必须在
Controller 层再次校验 action type、source、参数范围和当前阶段。不能只依赖 Agent
自身的固定 blocklist。

checkpoint 晋级建议：

```text
trained
  -> offline_validated
  -> debug_validated
  -> simulator_candidate
  -> simulator_approved
  -> robot_shadow_candidate
  -> robot_shadow_approved
  -> robot_enforce_candidate
```

任何阶段失败都保留原因和产物，不隐式跳级。

## 10. DAgger 数据闭环

Continual 模式每轮产物还应包含 `DatasetManifest`：

- 原始数据集版本；
- 本轮新增 episode 和有效标签数；
- success/failure/aborted/takeover 分布；
- `selected_action_source` 和 `label_valid` 统计；
- 数据过滤规则；
- 新旧数据混合比例；
- 生成数据集摘要的代码 commit。

`DatasetManifest` 的外层只定义通用身份、文件哈希、来源、统计量和训练可用性。
机器人的向量维度、流名称、时间对齐及过滤规则放在显式 `profile_id` 对应的专属
payload 和校验器中。例如 GR3 使用 `xpolicylab.gr3_dagger_v2`；其他机器人不得
隐式继承 GR3 的 33/37 维布局或终止后相机尾帧规则。

训练绑定是显式的：`ExperimentSpec.dataset_manifest_id` 与调用方提供的 manifest
路径必须同时存在且 ID 一致。Runner 先执行对应 profile 校验并确认至少一个
episode 在过滤后可训练，再将清单以只写一次的快照放入 round 目录。训练脚本只
通过 `XPOLICYLAB_DATASET_MANIFEST`、`XPOLICYLAB_DATASET_ID`、数据摘要哈希、
profile ID 和有效 episode 数读取这次绑定；未知 profile 或不匹配的 ID 在进程
启动前失败。Runner 同时重算清单层级摘要，并逐文件核对 regular file、大小和
SHA-256，防止生成 manifest 后数据被替换。checkpoint 的 `ArtifactManifest.dataset`
保留同一绑定以供追溯。

Agent 可以建议：

- 下一轮优先采集的任务；
- failure/takeover episode 的采样权重；
- 新旧数据比例；
- 是否继续训练或停止。

Agent 不得决定标签有效性、绕过人工确认或修改机器人安全控制。

## 11. 实施阶段

### Phase 0：文档与契约

- 固定本设计；
- 定义四类 Pydantic model；
- 明确 XPolicyLab commit 和首个 policy；
- 给出本地与远程 profile。

验收：契约 JSON round-trip、未知版本拒绝、非法 outcome/路径/指标拒绝。

### Phase 1：无 GPU 闭环

- 实现受限 runner；
- 使用 `demo_policy` 和 debug client；
- 产生假 checkpoint manifest；
- 完成 trial result 聚合；
- 将 summary 写入 Interactive Training memory。

验收：一次完整闭环可重复运行，无 GPU、无 simulator、无 LLM API key。

### Phase 2：5090 小规模验证

- 选择可在本机运行的小 policy 或短步训练；
- 记录显存、时间和产物；
- 使用 recorded replay 或单任务少 episode 仿真；
- 验证失败、超时、OOM 和中断恢复。

验收：在资源预算内完成至少两个独立配置的训练/评测对照。

### Phase 3：AHA_WAM 远程训练接入

- 在 AHA_WAM Trainer 插入显式控制点；
- 验证 Accelerate/DeepSpeed rank 同步；
- 接入原生 save/eval；
- 将 checkpoint 放入 XPolicyLab PolicyServer 评测。

验收：短远程训练中可以安全修改一个训练 knob，并完成 checkpoint 到评测的追踪。

### Phase 4：GR3 DAgger 数据回流

- 标准化 episode result 和 DatasetManifest；
- 先使用 recorded replay；
- 仿真验证 continual 模式；
- 真机只读指标和 shadow 建议。

验收：从一个已记录 DAgger episode 生成可追踪数据版本并启动下一轮训练。

TurboVLA 的 GR3 实验采用同一闭环，但训练只能由 Interactive Training Runner
启动。具体的数据、模型、回放与真机门禁设计见
[`TURBOVLA_GR3_INTEGRATION.md`](TURBOVLA_GR3_INTEGRATION.md)。

## 12. 测试计划

### 单元测试

- 契约版本、类型、数值有限性与 outcome 校验；
- 多任务聚合和 invalid trial 分母；
- argv/env allowlist；
- checkpoint 清单摘要；
- action source 和参数边界；
- local profile 资源上限。

### 集成测试

- fake train -> fake checkpoint -> debug PolicyServer -> trial_end -> summary；
- PolicyServer 启动失败；
- 环境 client 超时或断连；
- 不完整 trial result；
- 训练进程被中断后保留 manifest；
- Controller 重启后从 spool 恢复未消费结果；
- 两个 round 的 parent checkpoint 和数据版本正确关联。

### GPU smoke test

- 单 GPU 峰值显存记录；
- OOM 后结果为失败而不是误报完成；
- 不残留 PolicyServer 或训练进程；
- checkpoint 不完整时禁止晋级。

### 真机前强制测试

- recorded replay；
- action schema、字段、单位、语义和频率一致；
- NaN/Inf、过期状态、generation mismatch 均进入 HOLD；
- shadow 输出不能进入机器人唯一写入路径；
- Agent 不能修改安全参数或触发 enforce。

## 13. 主要风险

| 风险 | 对策 |
| --- | --- |
| policy 训练入口差异大 | MVP 只做外层统一，深度 hook 按 policy 实现 |
| 5090 显存不足 | debug/replay/小模型优先，正式训练远程执行 |
| 评测噪声误导 Agent | 多 seed/repeat、置信信息、`inconclusive` 状态 |
| DeepSpeed 动态参数不同步 | 显式控制点、全 rank 注册、逐后端测试 |
| 自定义 action 越权 | Controller 二次 allowlist/RBAC，不依赖 Agent blocklist |
| checkpoint 与结果错配 | commit、配置、数据和 checkpoint digest 全链路记录 |
| 真机动作风险 | Agent 永不控制 Safety Guard，shadow 与审批门禁 |
| HTTP 控制面暴露 | 默认 loopback，跨机通过受认证的专用桥接层 |
| 大量产物耗尽磁盘 | 每轮预算、清单和保留策略；不自动删除失败证据 |

## 14. MVP 决策

首个实现建议采用：

- 控制端：Interactive Training 自定义实验循环；
- 执行端：XPolicyLab `demo_policy`；
- 评测端：debug client；
- 训练：fake 或 CPU tiny train；
- Agent：先使用 scripted agent，不调用外部 LLM；
- 目标：验证协议与可追踪性，而不是模型效果。

完成后，本地 GPU MVP 优先选择较小 policy/DP，而不是直接以 AHA_WAM 或 LingBot
VLA 作为 RTX 5090 首测。AHA_WAM 保留为远程多 GPU 深度 Trainer 接入的第一个目标，
因为它已有明确的 Accelerate/DeepSpeed 训练循环和 XPolicyLab 部署适配。

## 15. 待确认事项

- 第一个本地 GPU policy 选择 DP、其他小模型，还是自建 tiny embodied policy；
- 远程执行后端是 SSH、Slurm 还是现有集群平台；
- RoboDojo 是否能在同一台 5090 上与目标 policy 同时运行；
- TrialResult 由通用 env client 统一发送，还是先由各 benchmark adapter 生成；
- 正式实验主目标使用平均成功率还是 worst-task 成功率；
- DAgger 数据目录和长期 artifact store 的位置与保留周期。
