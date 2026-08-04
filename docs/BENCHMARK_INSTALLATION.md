# 公共具身 Benchmark 安装与接入记录

更新日期：2026-08-01

本机采用“控制面与运行环境分离”的安装方式：Interactive Training 保持训练迭代、
实验记忆和晋级控制；XPolicyLab 负责策略服务及测评协议；每个 benchmark 使用自己的
Python 环境，避免 MuJoCo、SAPIEN、Isaac Sim 和 CUDA 依赖互相覆盖。

## 安装清单

| Benchmark | 安装目录 | Python 环境 | 固定提交 | 当前验收状态 |
| --- | --- | --- | --- | --- |
| LIBERO | `/home/ubuntu/benchmarks/LIBERO` | `.venv`（Python 3.10.20） | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 环境 `reset/step/close` 通过 |
| SimplerEnv | `/home/ubuntu/benchmarks/SimplerEnv` | `.venv`（Python 3.10.20） | `06accaca93535902d408da4855f21cece12bceb7` | 环境 `reset/step/close` 通过 |
| RoboDojo | `/home/ubuntu/benchmarks/RoboDojo` | Conda `RoboDojo`（Python 3.11.15） | `b8e5eed9fc11fc3da8e4b142fc60acf22036efa5` | 代码、运行栈和 Assets 已安装；RTX 启动受本机 595.80 驱动阻塞 |

没有下载策略 checkpoint 或 benchmark 演示数据。训练数据与模型权重必须作为独立、
有版本的 artifact 管理，不能混入 benchmark 安装目录。

## LIBERO

LIBERO 使用官方 `robosuite==1.4.0`。为了兼容当前 RTX 5090 软件栈，保留新版
PyTorch 2.13.0+cu130，但固定 `mujoco==2.3.7`；无上限解析到 MuJoCo 3.x 会导致 robosuite 访问
已经移除的 `MjData.qM`。PyTorch 2.6 以后默认只加载 weights，因此对仓库内受信任的
LIBERO init-state 文件显式使用 `weights_only=False`。

官方仓库的 `setup.py` 使用 `find_packages()`，但源码实际采用外层 `libero` namespace，
会产生只有在仓库根目录才能 import 的空 editable 安装。本机改为
`find_namespace_packages(include=["libero", "libero.*"])`；已经从
`/home/ubuntu/interactive-training` 外部工作目录复验导入与环境创建，避免 adapter
依赖隐式当前目录。

运行前设置：

```bash
export LIBERO_CONFIG_PATH=/home/ubuntu/benchmarks/.config/libero
export MUJOCO_GL=egl
source /home/ubuntu/benchmarks/LIBERO/.venv/bin/activate
```

已验证 `libero_spatial` 的第一个任务：任务套件与 init-state 可读取，离屏环境可以
`reset`、恢复初始状态、执行一步 7 维动作并返回 128×128 RGB 图像。

## SimplerEnv

SimplerEnv 与其官方 `ManiSkill2_real2sim` 子模块安装在同一隔离环境，固定
`numpy==1.24.4` 和 `setuptools<81`。后者用于兼容 SAPIEN 2.2.2 对
`pkg_resources` 的依赖。

本机同时存在多个 Vulkan ICD。SAPIEN 自动选择时可能落到不支持所需扩展的软件设备，
运行前必须显式选择 NVIDIA ICD：

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
source /home/ubuntu/benchmarks/SimplerEnv/.venv/bin/activate
```

已验证 `google_robot_pick_coke_can`：环境可创建、`reset`、执行随机动作、返回 5 元组并
正常关闭；观测包含 `agent`、`camera_param`、`extra`、`image`，图像包含 base 与
overhead 两路相机。

## RoboDojo

RoboDojo 使用官方原生安装器和独立 Conda `RoboDojo` 环境，已安装 Python
3.11.15、PyTorch 2.7.0+cu128、Isaac Sim 5.1.0、Isaac Lab 0.54.3 和 CuRobo。仓库包含它自己的
XPolicyLab 子模块，但正式接入时仍以 `/home/ubuntu/xpolicylabdagger` 为项目主框架，
不把 RoboDojo 内的副本当成新的控制面。

官方安装器使用 `git submodule update --remote`，所以除了记录 RoboDojo superproject
提交，还必须记录安装后的实际子模块提交：

| 子模块 | 实际提交 |
| --- | --- |
| `XPolicyLab` | `72b3a52f2b069220088ef824ade5890885858379` |
| `third_party/IsaacLab` | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| `third_party/curobo` | `895c6517243f8cb091c73c018c8167192d39599a` |

激活环境：

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate RoboDojo
export OMNI_KIT_ACCEPT_EULA=YES
```

`OMNI_KIT_ACCEPT_EULA=YES` 与 RoboDojo 官方安装器一致，表示接受 NVIDIA Omniverse
EULA；批处理或 XPolicyLab 子进程若不继承该变量，Isaac Sim 会在非交互环境中提示
输入并提前退出。

RoboDojo 的 Assets 是仿真运行必需项，演示数据不是。Assets、策略 checkpoint 和
LeRobot/HDF5 数据要分别记录；不能因为代码可导入就宣称完成 benchmark 环境验证。

官方 Assets 安装器已完成：`Assets` 链接到
`/home/ubuntu/benchmarks/RoboDojo/.cache/robodojo_assets_repo/Assets`，包含 15,365 个
文件，展开后的逻辑体积约 39 GiB，Git LFS 对象占用约 27 GiB。仅限 `Assets/**` 的
LFS pull 已复验，工作树中没有残留 LFS 指针；`x5` 和 `franka` 的两个 CuRobo 配置
也已由模板成功生成。

官方静态安装验证摘要保存在
`/home/ubuntu/benchmarks/RoboDojo/install_verify.json`：14 项通过、3 项因未提供待测
policy 而跳过、0 项失败；54/54 个任务均有实现和配置，环境配置引用的仿真、场景、
机器人与相机文件均存在。该结果证明代码、依赖和资产完整，不替代下面的 RTX
环境启动验收。

### RTX 5090 驱动阻塞

以下检查已经通过：

- PyTorch CUDA 可见并识别 `NVIDIA GeForce RTX 5090`；
- `isaacsim`、`isaaclab` 与 CuRobo 直接导入；
- Isaac Sim 5.1 Compatibility Checker 判定 GPU、595.80 驱动、VRAM、CPU、RAM、
  存储和 Ubuntu 22.04 均满足静态最低要求。

但真实无头 RTX 应用在 `librtx.scenedb.plugin.so` 初始化时段错误。该堆栈与
[NVIDIA 官方 Isaac Sim issue #568](https://github.com/isaac-sim/IsaacSim/issues/568)
的 595.x 案例一致；NVIDIA 维护者说明 Isaac Sim 5.1
经过验证的 Linux 驱动为 580.65.06，并由案例提交者确认降级后恢复。社区提供的
Vulkan Profiles 属性兼容层在本机可越过 SceneDB 崩溃点，但随后仍在 595.80 的
`libnvidia-gpucomp.so` 着色器编译阶段 SIGILL，因此没有把它纳入正式启动路径。

结论：RoboDojo 的 Python/CUDA 运行栈已安装，但在把系统驱动降到 580.65.06 并重启
复验前，不得标记为“环境安装完成”，也不得生成模型 benchmark 分数。驱动变更需要
管理员权限与计划内重启，不能由普通用户环境安装器隐式执行。

另有一个上游 Python 元数据冲突：`isaacsim-kernel 5.1.0.0` 固定
`fastapi==0.115.7`（要求 `starlette<0.46`），而当前 Isaac Lab 0.54.3 元数据固定
`starlette==0.49.1`。RoboDojo 官方安装器明确固定 `starlette==0.45.3`，本机保留
该运行组合以满足 Isaac Sim；因此 `pip check` 会报告这一条 Isaac Lab 元数据冲突，
不能通过强升 Starlette 消除，否则会直接破坏 Isaac Sim 的精确依赖。

## XPolicyLab 映射

三个 benchmark 都作为 XPolicyLab 的 evaluation backend，而不是直接嵌入
Interactive Training 进程：

```text
Interactive Training ExperimentSpec
  -> XPolicyLab PolicyServer
  -> LIBERO / SimplerEnv / RoboDojo env adapter
  -> episode evidence + success/invalid/failure
  -> XPolicyLab TrialResult
  -> Interactive Training memory 与下一轮参数/数据动作
```

LIBERO 和 SimplerEnv 下一阶段需要新增独立 adapter，将任务名、seed、最大步数、动作
空间、成功判据和视频证据映射到现有 `TrialResult`。RoboDojo 已有 `_result.json`
桥接，完成 Isaac Sim 与 Assets 验证后即可复用现有 `evaluate_sim()`。

## 验收边界

- “代码安装”：仓库和依赖存在，可导入；
- “环境安装”：能够创建环境并至少完成一次 `reset/step/close`；
- “模型 benchmark”：策略通过 XPolicyLab 执行多个固定 seed，并保存权威成功判据；
- “可比较结果”：同一任务、seed、最大步数、动作执行策略和版本锁定后，才能比较模型。

当前安装阶段不把随机动作 smoke test 当作模型效果，也不把某个机器人或某条 DAgger
数据的结果外推成通用 benchmark 成绩。

## 官方来源

- [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [SimplerEnv 官方仓库](https://github.com/simpler-env/SimplerEnv)
- [RoboDojo 安装与 Assets 指南](https://robodojo-benchmark.com/doc/usage/install-and-download/)
- [Isaac Sim 5.1 下载与版本说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)
