# 具身模型交互训练与部署测评项目总结

更新时间：2026-08-01

## 1. 项目概述

本项目以 XPolicyLab 为具身模型统一框架，将 Interactive Training、模型
运行时、仿真环境、数据回放和未来真机部署组织成一个可审计的迭代闭环。
当前主要实验模型为轻量级 VLA——TurboVLA，主要仿真环境为 RoboTwin，
同时保留独立的 GR3 DAgger 数据和部署路径。

项目解决的核心问题不是单独“训练一个模型”，而是建立以下公共流程：

```text
数据采集/数据回流
       |
       v
Interactive Training：目标、数据快照、参数、轮次和历史
       |
       v
XPolicyLab：训练启动、checkpoint 注册、部署、仿真/回放/真机门禁
       |
       v
TurboVLA 或其他模型运行时：训练与推理
       |
       v
RoboTwin / recorded replay / GR3 shadow
       |
       v
任务结果、动作轨迹、性能指标回传 Interactive Training
       |
       +--------------------> 下一轮数据、参数或模型选择
```

截至目前，以上流程已经完成“人或 Agent 决策、系统受控执行”的闭环验证。
它能够根据训练和仿真反馈开展下一轮实验，但尚不是无人值守的自动超参搜索、
自动 DAgger 采集和自动模型晋级系统。

## 2. 当前总体状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 数据 manifest 与文件哈希绑定 | 已完成 | 训练前重新校验数据 ID、profile 和源文件 SHA-256 |
| Interactive Training 训练控制 | 已完成 | 训练步数、学习率、batch、数据和环境参数受白名单约束 |
| XPolicyLab TurboVLA 插件 | 已完成 | 支持训练、checkpoint 注册、评测和结果回传 |
| RoboTwin 单任务训练—仿真闭环 | 已完成 | `beat_block_hammer` 同一场景确定性成功 2/2 |
| RoboTwin 跨 seed 泛化 | 未完成 | 未见 seed 200000 为 0/1 |
| GR3 数据预检与 recorded replay | 已完成 | 协议回放通过，但不是任务成功证据 |
| GR3 仿真 | 未完成 | 仍需真正匹配 GR3 动力学和动作空间的环境适配器 |
| GR3 真机 shadow/enforce | 未启用 | 必须经过独立安全、时延和人工批准门禁 |
| 自动多轮调参 | 部分完成 | 轮次、指标、历史和参数契约已具备，自动决策策略待开发 |
| 云端多卡训练 | 待接入 | 本地单卡链路已验证，多卡启动和调度尚未验收 |

当前结论是：项目架构已经证明能够产生、记录并复现真实仿真任务成功；
模型质量仍处于单任务、单场景开发阶段，不能据此宣称通用 RoboTwin 或 GR3
能力。

## 3. 项目公共能力

### 3.1 统一实验契约

所有具身实验使用版本化的 `xpolicy_interactive.v1` 契约，核心对象包括：

- `ExperimentSpec`：模型、benchmark、任务、seed、训练参数和评测配置；
- `DatasetManifest`：数据 ID、profile、源文件清单、过滤状态和 SHA-256；
- `ArtifactManifest`：训练进程、checkpoint、配置、数据绑定和运行状态；
- `TrialResult`：单次任务结果、证据类型、终止原因、指标和证据文件；
- `EvaluationSummary`：有效/无效 trial、成功率、最差任务表现和诊断聚合。

这些契约使模型训练、仿真和回放不依赖同一个 Python 环境，也让每次结果都
可以追溯到确定的数据、代码、参数和 checkpoint。

### 3.2 数据与 checkpoint 可追溯

- 训练只能使用已验证并快照化的数据 manifest；
- 训练脚本拒绝缺少 Interactive Training 数据身份的直接调用；
- checkpoint 注册时记录文件大小和 SHA-256；
- 评测结果绑定 checkpoint ID 和 dataset ID；
- 重复实验可以验证是否使用了同一个不可变模型包。

### 3.3 分层评测与部署门禁

项目将证据强度分为：

1. 数据预检；
2. 短训练/过拟合 smoke；
3. recorded replay 协议验证；
4. 仿真任务成功；
5. 真机 shadow；
6. 人工批准的真机 enforce。

低等级证据不能替代高等级证据。例如 recorded replay 通过只能证明输入输出
协议可用，不能写成机器人任务成功。

### 3.4 运行时隔离

Interactive Training、XPolicyLab、TurboVLA 和 RoboTwin 保持各自 Python
环境，通过 JSON、JSONL、checkpoint 文件和策略服务协议连接。这样可以避免
PyTorch、仿真器和模型依赖互相污染，也方便未来替换模型或迁移云端。

### 3.5 诊断反馈

除任务成功率外，当前 RoboTwin 链路还记录：

- 每步 14D 动作及连续性校验；
- 模型动作与机器人状态的 L2 距离；
- 连续输出饱和率；
- 左右夹爪开合比例；
- 预测 action chunk 的范围和实际选择动作；
- 推理/仿真证据路径和终止原因。

这些指标用于判断问题位于优化、数据、动作执行、语言、仿真还是模型泛化，
避免只根据训练 loss 盲目增加训练步数。

## 4. 各模块功能

### 4.1 Interactive Training：控制面

代码位置：`src/interactive_training/` 及
`src/interactive_training/integrations/xpolicylab/`。

主要职责：

- 定义训练目标、实验轮次和可调整参数；
- 验证数据 manifest 和机器人 profile；
- 通过 `RunnerPolicy` 限制可运行的 policy、脚本和环境变量；
- 启动训练或注册已有 checkpoint；
- 接收 XPolicyLab 的 trial 并聚合为评测指标；
- 将每轮配置、得分和完整具身证据写入 memory；
- 支持人工或 Agent 根据历史选择下一轮参数。

内部组件：

| 组件 | 功能 |
| --- | --- |
| `contracts.py` | 版本化实验、数据、artifact 和 trial 数据模型 |
| `datasets.py` | manifest 读取、摘要与文件完整性校验 |
| `profiles/robotwin.py` | RoboTwin 三相机、14D 动作数据规则 |
| `profiles/gr3.py` | GR3 DAgger 33D 状态、37D 动作数据规则 |
| `policies.py` | Agent、shadow、安全和晋级权限边界 |
| `runner.py` | XPolicyLab 训练、回放和仿真进程管理 |
| `reporter.py` | trial 校验、证据语义和指标聚合 |
| `experiment.py` | 一轮训练—评测—memory 回写流程 |

### 4.2 XPolicyLab：统一执行与部署测评框架

项目位置：`/home/ubuntu/xpolicylabdagger`。

主要职责：

- 作为模型、benchmark、机器人和部署方式之间的统一插件框架；
- 接受 Interactive Training 下发的受控实验配置；
- 启动模型训练并形成标准 checkpoint 目录；
- 启动策略服务和仿真/回放客户端；
- 管理 policy GPU 与环境 GPU；
- 输出标准 trial JSONL、动作轨迹及日志；
- 为未来真机 shadow/enforce 复用现有安全路由和 watchdog。

TurboVLA 插件位于 `policy/TurboVLA/`：

| 文件 | 功能 |
| --- | --- |
| `train.sh` | 校验 Interactive Training 绑定并路由 GR3/RoboTwin 训练 |
| `eval.sh` | 选择 checkpoint、启动策略服务和评测环境 |
| `model.py` | TurboVLA 的 XPolicyLab 模型接口 |
| `robotwin_dataset_manifest.py` | RoboTwin LeRobot 数据扫描与 manifest 生成 |
| `register_robotwin_checkpoint.py` | 外部、EMA 或 raw checkpoint 的可审计注册 |
| `robotwin_trial_report.py` | RoboTwin 结果与逐步动作轨迹转为标准 trial |

### 4.3 TurboVLA：轻量级 VLA 模型运行时

项目位置：`/home/ubuntu/TurboVLA`；发布的 RoboTwin 权重训练兼容运行时位于
`/home/ubuntu/TurboVLA-legacy`。

主要职责：

- DINOv3 视觉编码、BERT 文本编码和视觉语言融合；
- ACT 风格 50 步 action chunk 预测；
- RoboTwin 三相机、14D 双臂动作训练与推理；
- GR3 单相机、33D 状态、37D 动作专用适配；
- 2026-08-04 起，TurboVLA 内部改为 33D 状态、33D 学习动作头；部署适配器补四个
  零值底盘平面动作，继续兼容 XPolicyLab/GR3 外部 37D 协议；
- raw/EMA checkpoint 保存和策略服务。

发布的 RoboTwin checkpoint 使用 legacy `GroundingDINODiT` 参数命名，不能
直接加载到当前 runtime。项目已经改为严格加载并自动路由 legacy runtime，
禁止用 `strict=False` 将参数不匹配误报为成功初始化。

### 4.4 RoboTwin：仿真 benchmark

项目位置：`/home/ubuntu/RoboTwin`。

主要职责：

- 生成任务场景和可行 seed；
- 使用专家规划器采集成功轨迹；
- 加载 XPolicyLab 策略进行闭环仿真；
- 保存视频、任务成功结果和 episode 日志；
- 支持 seen/unseen 指令以及受审计的固定指令实验。

本地环境缺少编译官方 CuRobo 所需的 `nvcc`，当前统一使用 MPLib screw
planner 进行专家可行性检查。该结果不能表述为官方 CuRobo 配置复现。

### 4.5 GR3 DAgger 数据与部署路径

数据位置：`/home/ubuntu/xpolicylabdagger/data/gr3`。

GR3 使用独立的 `xpolicylab.gr3_dagger_v2` profile：

- 顶部相机；
- 33D 规范状态；
- 37D 异构动作；
- 30 Hz、50 步 action chunk；
- expert-safe 动作和时间戳对齐。

GR3 数据不能当作通用机器人数据，也不能使用 RoboTwin 14D checkpoint
直接驱动。当前只完成短训练和 recorded replay，尚未启用真机动作写入。

### 4.6 数据转换与证据工具

`scripts/convert_robotwin_hdf5_to_lerobot.py` 将成功的 RoboTwin HDF5 专家
轨迹转换为独立的 LeRobot v2.1 数据集，保留三路视频、14D 状态/动作、指令
和来源元数据，并拒绝覆盖已有目录。公开 50-demo 数据不会被修改。

## 5. 已完成的闭环流程

一次标准训练评测轮次如下：

1. Interactive Training 创建 `ExperimentSpec`；
2. 加载并重新校验数据 manifest；
3. XPolicyLab 调用目标模型的 `train.sh`；
4. 模型保存 raw、EMA、配置和数据统计；
5. XPolicyLab 注册 checkpoint 并计算不可变 ID；
6. XPolicyLab 部署策略到 RoboTwin 或 recorded replay；
7. 环境产生 trial、动作 trace、视频和日志；
8. Interactive Training 聚合成功率和诊断指标；
9. 本轮完整证据写入 `result.json`、`metrics.jsonl` 和 memory；
10. 人或 Agent 根据失败类型配置下一轮。

目前第 10 步是受控的人/Agent 决策。自动超参搜索、自动数据采集和自动晋级
仍属于后续功能。

## 6. 实验记录

### 6.1 GR3 数据、训练和回放

| 实验 | 配置 | 结果 | 结论 |
| --- | --- | --- | --- |
| GR3 数据预检 | 1 条 DAgger episode | profile、维度和时间对齐通过 | 可用于 pipeline smoke，不代表数据充分 |
| TurboVLA GR3 短训练 | 10 steps，batch 1 | loss `0.664165 -> 0.645011` | 训练链路和 checkpoint 生成可用 |
| 云端 GR3 AnyGrasp 正式骨干 smoke | PPU 单卡，BERT + DINOv3-L + TurboVLA 初始化，2 steps | loss `0.630241 -> 0.525800`，真实帧推理输出有限 `4x37` 动作 | 证明 AnyGrasp 原地只读训练闭环；不代表模型收敛 |
| 云端 GR3 AnyGrasp task-disjoint pilot | PPU 单卡，256 clips/20,736 frames/47 train tasks，100 steps | step 1/100 loss `0.554684 -> 0.221258`；train L1 `0.512545 -> 0.308595`，19 个未见 task 的 held-out L1 `0.502926 -> 0.349810` | 相对零步初始化，train/held-out 分别改善 39.79%/30.44%；建立可复现的任务级泛化基线 |
| GR3 三分数据分级迭代 | 68/10/19 train/validation/held-out tasks；pilot 256 clips；`1e-4`，1,000 steps | validation L1 `0.324633@100 -> 0.241974@500 -> 0.187984@1000`；sealed held-out `0.196614` | 1000-step 比 100-step 改善 42.09%；held-out 泛化差距 4.59%，最终 checkpoint 晋级 |
| GR3 deployment-v4 33D full epoch | 5,427 clips、105,207 有效样本，完整 prompt，`batch=8`、`1e-4`、13,200 steps | 2026-08-04 21:48 完成；最终 loss `0.075209`；19-task held-out 的 104 个分层样本 L1 `0.131373`，热态推理均值/P95 `41.33/46.01 ms` | 33D 学习头完成约 1.004 epoch；离线门禁通过，下一证据为本地回放和人工监护真机成功率 |
| 晋级模型 AnyGrasp recorded replay | 10 个 validation tasks 各 1 个真实 AV1 帧，重复 2 次 | 两次均为 10/10、pass rate 1.0；动作 SHA 逐帧一致；预热平均/P95 `35.56/39.44 ms` | XPolicyLab runtime、WebSocket、33D 输入和 50×37 输出合约通过；不代表抓取成功或真机安全 |
| GR3 recorded replay | 2 帧，50×37 输出 | replay pass rate 1.0，P95 约 131 ms | WebSocket 和动作协议通过 |

正式骨干训练加载了 199 个与发布模型形状兼容的张量，并明确跳过 679 个不兼容
张量。AnyGrasp task split 使用 SHA-256 确定性分配：完整 train 为 78 tasks、
1,026 clips、83,106 frames；held-out 为 19 tasks、209 clips、16,929 frames，
二者 task ID 无交集。离线评估按 task 分层抽取 50 帧，覆盖全部 47 个 pilot
train tasks 和全部 19 个 held-out tasks。零步对照使用相同模型 seed、训练集统计、
抽样索引和发布初始化，只是不执行优化。100 steps 后 held-out L1 改善 30.44%，
但仍比 train 高 13.36%。预热后 PPU 推理平均约 29 ms，P95 约 32 ms。该结果是
动作重建基线，不是抓取成功率或真机安全结论。

后续迭代增加独立 validation，完整数据重新划分为 68 个 train tasks、10 个
validation tasks 和原样封存的 19 个 held-out tasks。学习率筛选只使用 validation：
`1e-4` 的 100-step L1 为 `0.324633`，略优于 `5e-5` 的 `0.327461`。胜出配置
从官方初始化训练 1,000 steps，validation L1 降至 `0.187984`，sealed held-out
为 `0.196614`，且 held-out 未参与选择。晋级记录位于远端
`runs/gr3-anygrasp-iteration-v1/iteration-summary.json`。

deployment-v4 将训练覆盖扩大到 5,427 clips，并使用与真机一致的完整任务 prompt。
模型内部采用 33D 状态和 33D 学习动作，推理适配器补四个零值平面底盘动作后
继续输出外部 `50x37` 合约。旧 37D checkpoint 的前 33 维权重迁移为
`876 loaded / 0 skipped`。新旧 held-out L1 的动作维度和抽样协议不同，因此
`0.131373` 只作为晋级信号，不作为严格同比的模型提升百分比。

晋级 checkpoint 随后通过新的 AnyGrasp 引用式 replay：源 AV1 和 parquet 仍在
`/mnt/workspace/jmy` 原地只读，项目目录只保存数据身份和 10 个确定性样本索引。
两次 replay 的十组动作 SHA 完全一致。冷启动约 2.58 秒，预热推理平均 35.56 ms、
P95 39.44 ms；旧聚合的约 290 ms 包含首帧冷启动，现已拆分报告。

### 6.2 RoboTwin 发布模型基线

任务：`beat_block_hammer`。

| 模型/训练 | seed | 执行方式 | 成功 |
| --- | --- | --- | ---: |
| 发布 checkpoint | 0、1、2 | temporal ensemble | 0/3 |
| 100 steps，batch 1 EMA | 0 | temporal ensemble | 0/1 |
| 500 steps，batch 1 EMA | 0、1、2 | temporal ensemble | 0/3 |
| 1,000 steps，batch 16，50 demos EMA | 0 | open-loop | 0/1 |

这些实验确认了严格加载、训练、注册和评测链路，但降低训练 loss 并未自动产生
任务成功。

### 6.3 动作执行诊断

逐步 trace 显示，发布模型和早期微调模型的未来 action chunk 已包含夹爪闭合，
但 temporal ensemble 每次重规划只执行块首动作，导致夹爪长期保持打开。增加
`open_loop_50` 后夹爪能够闭合，但发布模型和 50-demo EMA 仍未完成任务，说明
执行策略是问题之一，而不是唯一问题。

实验性的 oldest-binary 规则能够强制左夹爪闭合，但也错误关闭右夹爪，仍然
失败，因此只保留为诊断模式。

### 6.4 成功专家轨迹与目标场景训练

RoboTwin 专家规划器在 seed 100000 上成功采集：

- 165 帧；
- 3 路 320×240 相机；
- 14D 双臂关节/夹爪动作；
- 左臂抓锤并敲击方块；
- 转换后 dataset ID：
  `robotwin-beat-block-hammer-seed100000-gate-c39625069a43`。

使用 released checkpoint 初始化，batch 16 训练 1,000 steps，最终 action loss
为 `0.007595`。训练与评测均通过 Interactive Training -> XPolicyLab ->
TurboVLA 入口完成。

### 6.5 成功门槛实验

| checkpoint | 执行方式 | seed | 结果 | 关键观察 |
| --- | --- | --- | --- | --- |
| 目标场景 EMA | temporal ensemble | 100000 | 0/1 | 400 步夹爪均未执行闭合 |
| 目标场景 EMA | open-loop 50 | 100000 | 0/1 | 能闭合，但空间轨迹偏离锤柄 |
| 目标场景 raw | open-loop 50 | 100000 | 1/1 | 第 147 步成功 |
| 同一 raw 重复评测 | open-loop 50 | 100000 | 1/1 | 第 147 步成功，trace 完全一致 |
| 同一 raw、未见场景 | open-loop 50 | 200000 | 0/1 | 400 步失败，未建立泛化 |

EMA 的首 50 步左臂关节 RMSE 为 `0.134230`，raw 降至 `0.039055`。两次成功
动作 trace 的 SHA-256 均为：

```text
c78b991ccac66e0d099549cbcbc8059151ac0bced95d3242b63cbdd1dd26931e
```

成功 checkpoint ID：

```text
0cc3610b05149fe142f7acdf3745a45430c7b802204734cd1d6f4f5fe91e8923
```

这一结果证明端到端架构可以训练、部署并确定性复现真实 RoboTwin 成功；它是
同场景闭环门槛，不是通用 benchmark 成绩。

## 7. Interactive Training 如何针对反馈迭代

本轮实验体现了四类针对性决策：

1. loss 下降但任务失败：不继续盲目增加训练步数；
2. future chunk 有闭合但实际动作不闭合：调整动作执行策略；
3. EMA 轨迹误差高于 raw：将 raw 注册为独立候选并进行 A/B；
4. 同 seed 成功但未见 seed 失败：下一轮增加多 seed 数据，而不是继续对单一
   seed 过拟合。

下一阶段应把上述人工/Agent 推理固化为自动规则和候选搜索策略。

## 8. 当前限制与风险

- RoboTwin 成功只覆盖一个任务和一个训练场景；
- 未见 seed 已经证明当前模型缺少跨场景泛化；
- temporal ensemble 存在阶段停滞，成功配置依赖显式 open-loop；
- EMA 衰减 0.999 在短目标场景训练中保留过多旧权重；
- 本地专家规划器与官方 CuRobo 设置不同；
- GR3 AnyGrasp 已有 100,035 帧的 curated 数据、三分 task 划分、零步基线和
  1,000-step pilot，但仍只训练了 256 clips，尚无完整 train 训练或实际抓取成功率；
- GR3 没有可替代真实机器人动力学的有效仿真结果；
- 真机 enforce 未授权，也不应由训练进程直接写 Aurora；
- 云端多卡、分布式 checkpoint 和失败恢复尚未验证；
- 自动调参、自动 DAgger 回流和模型晋级仍需实现。

## 9. 更换其他模型的接入方式

整体框架不绑定 TurboVLA。新的 VLA、ACT、Diffusion Policy 或其他策略模型
需要实现：

1. XPolicyLab policy 插件；
2. 受 Interactive Training 约束的训练入口；
3. 模型专用数据 profile 和 manifest 校验；
4. checkpoint 注册、配置和统计文件；
5. 策略服务或标准推理适配器；
6. benchmark/机器人动作转换；
7. 标准 `TrialResult` 和必要诊断指标。

WAM 也可以复用控制面和执行框架，但需要新的评价语义，例如未来视频/状态预测
误差、规划可用性、闭环收益和不确定性，不能直接用 VLA 动作成功率代替。

## 10. 后续计划

### Phase A：多 seed RoboTwin 数据闭环

- 采集多个成功 seed 的专家轨迹；
- 对失败 rollout 增加 DAgger/人工修正；
- 构建 train/validation/held-out seed 划分；
- 训练混合多 seed 候选；
- 要求未见 seed 出现可重复成功后才能晋级。

### Phase B：动作执行与 EMA 修正

- 增加 joints/gripper 分阶段 reconstruction 指标；
- 对比 raw、不同 EMA decay 和延迟启动 EMA；
- 开发 stage-aware temporal ensemble；
- 将 open-loop 长度、重规划频率和二值夹爪策略纳入受控搜索；
- 保留动作饱和、错误夹爪和静态停滞的否决门槛。

### Phase C：Interactive Training 自动化

- 根据失败类型自动选择参数类别；
- 支持候选 checkpoint 并行 A/B；
- 实现成功率优先的晋级和回滚；
- 未见 seed 失败时自动创建数据采集任务；
- 将训练、仿真、数据和资源成本纳入多目标选择。

### Phase D：云端多卡训练

云算力提供后需要补齐：

- Slurm、SSH 或容器调度后端；
- DDP/DeepSpeed 多卡启动与拓扑记录；
- 共享数据、模型缓存和 artifact 存储；
- 分布式 checkpoint 合并、校验和恢复；
- 多 seed 并行仿真和故障隔离；
- 云端结果回传本地 Interactive Training memory。

远程算力优先用于多任务、解冻视觉/语言编码器、大 effective batch 和多候选
搜索，不用于替代尚未定义清楚的数据与评测门槛。

### Phase E：多模型公共 benchmark

- 已按独立环境安装 LIBERO、SimplerEnv 与 RoboDojo；LIBERO 和 SimplerEnv 已通过
  环境级 smoke；RoboDojo 的运行栈、39 GiB Assets、54 个任务与配置引用已通过
  静态安装核验，等待系统驱动从 595.80 降至 Isaac Sim 5.1 验证过的 580 系列后
  完成 RTX 运行验收；
- 安装版本、兼容补丁、运行命令与验收边界见
  [`BENCHMARK_INSTALLATION.md`](BENCHMARK_INSTALLATION.md)；
- 接入至少一个不同架构的轻量 VLA；
- 用同一数据快照和 seed 集比较模型；
- 分离模型能力、执行策略和仿真环境影响；
- 为 WAM 建立独立但兼容的预测/规划评价 profile；
- 最终形成 XPolicyLab 内统一模型排行榜与晋级记录。

### Phase F：GR3 部署推进

- 已将云端晋级 TurboVLA checkpoint 下载到本地并核对 promotion SHA-256；复用
  本地已存在且与云端哈希一致的 DINOv3-L/BERT 资产；
- 已把 TurboVLA 接入统一 `gr3.sh serve/dagger/replay` 入口和成熟
  `gr3-policy-dagger.yml` 真机图；本地 5090 WebSocket 合约测试输出有限
  `50x37`，两次热态推理为 `15.26/13.32 ms`；
- 2026-08-04 已完成真实 OAK、Aurora、QNexo、脚踏、TurboVLA、安全节点和
  probe 的联合 shadow：硬件写入被强制关闭，左蓝单步事件触发后获得
  `GR3_MODEL_EXO_SHADOW_OK`，机器人关节最大观测漂移仅 `7.13e-06 rad`，生命周期
  保持 `Default / Default / Joystick`；
- 外骨骼与脚踏重插到不同 USB 控制器后，外骨骼连续 20 秒得到 9,642 次更新，
  平均 `482.09 Hz`、最大样本年龄 `2.061 ms`，无掉线；三只脚踏与左蓝按钮均已
  独立实测，相关 GR3/结果桥接回归为 37/37 通过；测试结束后机器人、外骨骼与
  脚踏均由操作员主动断开，后续离线状态属于预期；本地策略服务已正常停止并
  释放其 GPU 占用；
- 当前结论仅覆盖设备、协议、安全过滤和无写入链路，不代表 AnyGrasp 抓取成功；
  下一门禁是清场、急停确认和明确人工授权后的单次真机动作，连续执行仍禁用；
- 旧 1,000-step 模型已完成真机执行链路验证，但抓取效果较差；因此扩大
  deployment-v4 数据并完成 13,200-step 33D full-epoch 训练。最终 checkpoint
  SHA-256 为 `6279698d0a59362a2be9da40418cc0251f89c4cc5740fd44928a72d1b9d4fc6a`；
- 2026-08-04 再次以 `/home/ubuntu/dagger-gr3` 最新工作树为基线同步：robot-adaptor、
  QNexo、DAgger router、DepthAI、recorder、farther-core 与 schema 的共享文件已逐字
  一致；正式真机入口固定为 `gr3-policy-dagger.yml`，不再使用独立单步图作为运行
  入口；
- 最新参考语义恢复为右脚踏按住接管/松开回交，并默认 `FREEZE_WAIST=1`；当前
  TurboVLA 真机参数与成熟 GR3 链路一致：chunk `16`、prefetch `4`、相机与动作
  `30 Hz`、机器人状态 `60 Hz`、最大关节步长 `0.15 rad`；
- 同步了新版 GR3 episode validator：兼容 Dora typed parameters，并报告和剔除
  canonical robot state 结束后的尾部相机审计帧，禁止与陈旧状态配对训练；融合
  runtime 回归 26/26、XPolicyLab GR3/结果桥接回归 30/30 通过；
- 已完成固定 validation 上的学习率/步数筛选；后续不得用本轮 held-out 反向调参；
- 扩大 clip 覆盖时保留新的 task-disjoint 最终测试集，避免复用已查看的 held-out；
- 比较是否解冻骨干、学习率调度和更大 effective batch；
- 按动作组报告误差，并补充实际抓取/放置成功率；
- 后续如增加 DAgger 数据，必须使用 GR3 专用 profile，不能与 AnyGrasp 语义混合；
- 建立 GR3 专用仿真或数字孪生；
- 完成 shadow 模式的专家/模型/实际动作比较；
- 验证推理时延、限位、QP/Mink、watchdog 和接管；
- 只有通过安全审查并得到人工授权后才进入 enforce。

## 11. 建议验收门槛

| 阶段 | 最低门槛 |
| --- | --- |
| 数据 | manifest 完整、无排除项、状态/动作/相机覆盖满足 profile |
| 训练 | 严格 checkpoint 加载、有限 loss、无 NaN/OOM、artifact 可复现 |
| 回放 | 输出有限且维度正确，trial 语义为 `replay_validation` |
| 仿真开发门槛 | 同一 seed 至少重复成功且动作 trace/证据完整 |
| 模型晋级 | 多个 held-out seed 重复成功，优于不可变 baseline |
| 多任务晋级 | 报告平均成功率、最差任务成功率和无效 trial 比例 |
| 真机 shadow | 时延、动作误差、限位和接管全部通过 |
| 真机 enforce | 独立安全审批与人工授权 |

## 12. 资源与算力结论

本地 RTX 5090（32 GB）足以完成：

- frozen-encoder 单任务训练；
- batch 16 的目标场景训练；
- TurboVLA 策略推理；
- RoboTwin 单任务仿真；
- GR3 recorded replay。

成功门槛训练的观察显存约 4.0 GB，主要瓶颈是三路视频解码而不是显存。云端
单卡 PPU 已完成 GR3 AnyGrasp 官方骨干的三分数据筛选与 1,000-step pilot，单次
约 59 分钟。多进程 AV1 预取受远端 filesystem IPC 限制，因此当前主要吞吐瓶颈
仍是单 worker 随机视频解码。多卡用于完整训练前，应先解决解码缓存/本地暂存和
新的 sealed test 设计，避免增加设备却继续等待数据。

## 13. 关键文档与证据

- 总体闭环设计：[`XPOLICYLAB_EMBODIED_LOOP.md`](XPOLICYLAB_EMBODIED_LOOP.md)
- TurboVLA/XPolicyLab 架构：
  [`TURBOVLA_XPOLICY_ARCHITECTURE.md`](TURBOVLA_XPOLICY_ARCHITECTURE.md)
- RoboTwin 本地实验报告：
  [`TURBOVLA_ROBOTWIN_LOCAL_RESULTS.md`](TURBOVLA_ROBOTWIN_LOCAL_RESULTS.md)
- GR3 集成说明：[`TURBOVLA_GR3_INTEGRATION.md`](TURBOVLA_GR3_INTEGRATION.md)
- 公共 benchmark 安装记录：
  [`BENCHMARK_INSTALLATION.md`](BENCHMARK_INSTALLATION.md)
- 首次成功结果：
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-raw-openloop/seed0/result.json`
- 重复成功结果：
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-raw-openloop-repeat2/seed0/result.json`
- 未见 seed 结果：
  `runs/turbovla-robotwin/seed100000-gate-ft1000-b16-raw-openloop/seed1/result.json`
- GR3 训练/回放结果：`runs/turbovla-gr3-overfit-10/result.json`
- 云端 GR3 task split、100-step checkpoint 与分层评估：
  `/mnt/workspace/jiayu/autogoal/runs/gr3-anygrasp-task-pilot256-s100/`

## 14. 总结

项目已经从文档设计推进到可运行、可审计、可复现的具身训练—部署—评测闭环。
Interactive Training 负责目标、参数、数据和历史，XPolicyLab 负责模型接入、
部署和评测，TurboVLA 提供当前轻量 VLA 实现，RoboTwin 和 GR3 分别承担仿真
验证与机器人专用数据/部署路径。

`beat_block_hammer` 的两次确定性成功证明总体架构合理；未见 seed 失败则明确
指出下一阶段重点不是继续证明链路可运行，而是扩大数据覆盖、修正动作执行、
建立自动迭代和验证跨场景泛化。云端算力将服务于这一扩展阶段。
