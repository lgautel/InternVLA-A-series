# InternVLA-A1.5 + GeoPredict 3D 关键点轨迹融合版在 RoboTwin 2.0 上的评估实施方案与操作手册（V2）

> **版本说明**：V2 手册在意外删除后依据 V1、7 份执行 LOG 与 agent 评估会话交叉重建。预检（§6）、静态测试 v4（§2.2）、命令模板（§7/§8）与 LOG0807192 系列实际执行脚本一致。
>
> **V2 改进说明**：本手册在 V1（`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000.md`）基础上做了以下改进：
> 1. **新增代码修改章节**（第 1 节）：发现并修复了 4 个代码问题，其中 1 个严重 bug 会导致评估完全无法运行
> 2. **新增自动化测试章节**（第 2 节）：为每个代码修改设计了静态验证 v4（16 项）和集成测试用例
> 3. **修正错误描述**：V1 中 3 处错误地声称 `check_success` 排序 bug 已修复，实际该 bug 仍存在。V2 已修正
> 4. **工作流调整**：先修改代码 → 再测试修改 → 然后评估 checkpoint
>
> 本手册详细说明如何在本机（GCP Blackwell 实例）上使用 [RoboTwin 2.0](https://robotwin-platform.github.io/) 仿真平台的 `stack_bowls_three` 任务评估 **InternVLA-A1.5 + GeoPredict 3D 关键点轨迹预测器融合版**（以下简称 **itvlaGp**）的 checkpoint 权重。
>
> **与基线版的核心区别**：itvlaGp 在标准 InternVLA-A1.5 的 2-path（VLM prefix + action expert）架构基础上增加了第三条路径——**关键点专家（keypoint expert）**，形成 **3-path Mixture-of-Transformers (MoT)** 架构。推理时从 SAPIEN 机器人的 14 个关节链接实时提取 3D 关键点，通过 TrackEncoder（移植自 [GeoPredict](https://arxiv.org/abs/2501.01787)）编码历史轨迹，再由关键点专家处理后其 K/V 缓存供动作专家在 flow matching 去噪循环中注意力查询。
>
> **本机配置**：2× NVIDIA RTX PRO 6000 Blackwell Server Edition（96 GB VRAM × 2）、CUDA 12.8.2、Ubuntu on GCP。RoboTwin 2.0 源码在 `/home/luogang/share/zwy/Projects/RoboTwin/`；**需按 §3.11 创建** `third_party/RoboTwin` 符号链接后方可评估。
>
> **参考文档**：
> - **执行日志（Part B 详情见各 LOG）**：
>   - [`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md) — 环境搭建 + 4 项修复 + 冒烟
>   - [`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md) — 010k 重跑 63%/22%
>   - [`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192_10k.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192_10k.md) — 080719_2 run 010k 58%/8%
>   - 更多步数见 `reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192*.md` 系列
> - V1 操作手册：[`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000.md)
> - 基线版评估手册：[`/home/luogang/SRC/Robot/InternVLA-A-series/b/d/p/reprd_rbtwn_stackb3_eval6000.md`](../../../InternVLA-A-series/b/d/p/reprd_rbtwn_stackb3_eval6000.md)
> - itvlaGp 架构设计：[`itrnVLA15_GeoP_3dtrj_3cn2.md`](itrnVLA15_GeoP_3dtrj_3cn2.md)
> - ALOHA 双臂适配：[`itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md`](itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md)
> - 微调实施手册：[`itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2.md`](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2.md)
> - Phase 2 训练日志：[`itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2.md`](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2.md)

---

## 目录

- [Part A：评估实施方案与操作手册](#part-a评估实施方案与操作手册)
  - [0. 评估方案概览](#0-评估方案概览)
  - [1. 代码修改（评估前必须完成）](#1-代码修改评估前必须完成)
  - [2. 代码修改验证测试](#2-代码修改验证测试)
  - [3. 环境准备](#3-环境准备)
  - [4. itvlaGp 3-path 推理架构深度解读](#4-itvlagp-3-path-推理架构深度解读)
  - [5. 评估代码解读（inference.py 中的关键点支持）](#5-评估代码解读inferencepy-中的关键点支持)
  - [6. 全流程验证检查表](#6-全流程验证检查表)
  - [7. 冒烟测试（2 episode 快速验证）](#7-冒烟测试2-episode-快速验证)
  - [8. 正式评估执行](#8-正式评估执行)
  - [9. 结果汇总与对比分析](#9-结果汇总与对比分析)
  - [10. 关键约束与注意事项](#10-关键约束与注意事项)
  - [11. 故障排除手册](#11-故障排除手册)
  - [12. 附录](#12-附录)
- [Part B：执行记录](#part-b执行记录)
  - [时间线 / 操作日志](#时间线--操作日志)
  - [问题记录](#问题记录报错--根因--修复--验证)
  - [最终结果](#最终结果)

---

## Part A：评估实施方案与操作手册

### 0. 评估方案概览

#### 0.1 评估目标

在 RoboTwin 2.0 的 `stack_bowls_three` 任务上评估 itvlaGp（InternVLA-A1.5 + GeoPredict 3D 关键点轨迹预测器融合版）在 Phase 2 训练 10000 步后的 checkpoint，与基线版 InternVLA-A1.5（无 GeoPredict）对比，验证 3-path MoT 关键点融合架构是否提升操作成功率。

#### 0.2 评估配置一览

| 维度 | 值 |
|------|------|
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp/`（`b0728GeoP` 分支） |
| **Conda 环境** | `itvlaGp`（需新建） |
| **Checkpoint（step 010000）** | 见 §0.2.1；默认经 `outputs-ext` 引用 `itvlaGp080719_2/outputs/.../010000/pretrained_model` |
| **模型大小** | 5.9 GB（model.safetensors） |
| **架构** | 3-path MoT：VLM Prefix + Keypoint Expert + Action Expert |
| **关键点关节数** | 14（ALOHA 双臂，每臂 7：6 个链接 + 1 个腕部相机） |
| **推理后端** | `standard`（**必须**，optimized 不支持 3-path 关键点推理） |
| **动作模式** | `abs`（**必须**，与训练一致） |
| **RoboTwin 任务** | `stack_bowls_three`（任务索引 **46**） |
| **评测配置** | `demo_clean`（Easy）+ `demo_randomized`（Hard） |
| **每配置 episode 数** | 100 |
| **dtype** | bfloat16 |
| **GPU** | 每个评测使用 1 块 GPU（可双 GPU 并行跑两个配置） |
| **步数上限** | 1200 步/episode（`_eval_step_limit.yml`） |
| **预估时间** | 每配置 ~2.5-3 小时，双 GPU 并行共 ~3 小时 |
| **RoboTwin 平台** | `/home/luogang/share/zwy/Projects/RoboTwin/`（**需**符号链接至 `third_party/RoboTwin`，见 §3.11） |

#### 0.2.1 Checkpoint 与 outputs 路径

本机 `itvlaGp/outputs/` 通常为空；checkpoint 存放在同级目录 `itvlaGp080719_2/outputs/`。恢复后已创建快捷链接：

```bash
# 已存在（恢复时创建）
ls -la /home/luogang/SRC/Robot/itvlaGp/outputs-ext
# outputs-ext -> /home/luogang/SRC/Robot/itvlaGp080719_2/outputs
```

**默认 step-010000 checkpoint**（080719_2 训练 run，LOG0807192 系列使用）：

```bash
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model
```

> **历史路径说明**：早期 LOG1/LOG2 使用 `outputs/internvla_a1_5/p2/checkpoints/010000/`（旧 p2 训练目录），该路径在本机已不存在。对比历史结果时注意 checkpoint 来源差异（见 §9.3）。

评估其他步数时，将 `checkpoints/010000` 替换为 `checkpoints/006000` 等，并相应修改 `CKPT_RUN` 或 run 目录名。

#### 0.3 V2 工作流总览

```mermaid
graph LR
    subgraph "Phase 1: 代码修改"
        M1["修改 1: inference.py<br/>check_success bug (CRITICAL)"]
        M2["修改 2: eval.sh<br/>CONDA_ENV"]
        M3["修改 3: eval.sh<br/>RESIZE_SIZE"]
        M4["修改 4: requirements.txt<br/>scipy"]
    end
    
    subgraph "Phase 2: 测试验证"
        T1["静态源码验证<br/>(grep/python, 无需GPU)"]
        T2["集成冒烟测试<br/>(2 episode, 需GPU)"]
    end
    
    subgraph "Phase 3: 环境准备"
        E1["conda env + deps"]
        E2["15 项检查表"]
    end
    
    subgraph "Phase 4: 评估执行"
        R1["2-ep 冒烟测试"]
        R2["100-ep demo_clean"]
        R3["100-ep demo_randomized"]
    end
    
    M1 --> M2 --> M3 --> M4
    M4 --> T1 --> T2
    T2 --> E1 --> E2
    E2 --> R1 --> R2
    R1 --> R3
```

#### 0.4 与基线版评估的关键差异

| 差异点 | 基线版 InternVLA-A1.5 | itvlaGp（GeoPredict 融合版） |
|--------|----------------------|---------------------------|
| 代码库 | `InternVLA-A-series` | `itvlaGp` |
| Conda 环境 | `ivla15` | `itvlaGp` |
| 模型架构 | 2-path MoT（prefix + action） | 3-path MoT（prefix + **keypoint** + action） |
| 推理时 suffix | action suffix 仅 | kpt suffix（29 token）+ action suffix |
| K/V 缓存 | prefix → denoise loop | prefix → keypoint → denoise loop |
| 推理后端 | standard 或 optimized | **仅 standard**（optimized 不支持 3-path） |
| SAPIEN 关键点提取 | 无 | 有（`get_keypoints_aloha()` 提取 14 个关节 3D 坐标） |
| 滚动历史缓冲区 | 无 | `his_kpts[1000, 14, 3]` + `his_len` |
| Checkpoint config | `enable_keypoint_predictor: false` | `enable_keypoint_predictor: true` |
| 额外模块 | 无 | TrackEncoder（~3M params）、kpt_state_proj、keypoint_embedding、keypoint_out_proj |

#### 0.5 RoboTwin 2.0 `stack_bowls_three` 任务简介

- **任务描述**：将 3 个碗依次堆叠到指定目标位置
- **机器人**：ALOHA-AgileX 双臂，14 DOF 关节空间控制
- **相机**：3 个视角（head_camera 俯视、left/right_camera 腕部）
- **成功条件**：3 个碗的 XY 位置对齐（容差 0.04m）、Z 高度分别在目标值 ±0.02m 内、双爪打开
- **步数上限**：1200 步
- **评测配置**：
  - `demo_clean`（Easy）：无域随机化
  - `demo_randomized`（Hard）：物体姿态、光照、桌面纹理/高度、杂物、语言指令 5 轴随机化

---

### 1. 代码修改（评估前必须完成）

> **重要**：代码审查发现 itvlaGp 的评估代码存在 4 个问题，其中 1 个严重 bug 会导致评估完全无法运行。必须在开始评估前修复。
>
> V1 手册中 3 处错误地声称 `check_success` 排序 bug 已修复（第 601 行、第 753 行、第 1065-1078 行），实际该 bug 仍存在于 itvlaGp 代码库中。以下修改已通过与 InternVLA-A-series 已修复版本的对比验证。

#### 1.1 修改一：inference.py check_success 排序 bug 修复（CRITICAL）

**严重程度**：🔴 CRITICAL — 不修复则评估完全无法运行，每个 episode 必崩溃

**文件**：`evaluation/RoboTwin/inference.py`

**Bug 描述**：

在 seed 验证循环中，`maybe_close_env(task_env)` 在 `play_once()` 之后、`check_success()` 之前被调用，会执行 `_base_task.py` 的 `close_env()`，将 `self.robot` 设为 `None`。随后调用 `task_env.check_success()` 时，该方法内部访问 `self.robot.is_left_gripper_open()`，触发 `AttributeError`。

**崩溃链**（bug 版）：

```
inference.py  maybe_close_env(task_env)   # close_env() → robot = None
    ↓
inference.py  task_env.check_success()   # 在 try/except 之外
    ↓
stack_bowls_three.py  self.is_left_gripper_open()
    ↓
AttributeError: 'NoneType' object has no attribute 'is_left_gripper_open'
```

**影响**：这个 `AttributeError` 发生在 try/except 块之外，不会被捕获。评估在第一个 seed 验证成功后立即崩溃，**零个 episode 会被评估**。

**参考修复**：InternVLA-A-series 版本已修复此 bug。

**修改前**（seed 验证段，bug 版在 try/except **之后**调用 `check_success()`）：

```python
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed_value, is_test=True, **task_args)
            episode_info = task_env.play_once()
            maybe_close_env(task_env)
        except unstable_error as exc:
            logging.warning("Skipping unstable seed for task=%s seed=%s: %s", task_name, seed_value, exc)
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue
        except Exception:
            logging.error("Expert rollout failed for task=%s seed=%s", task_name, seed_value)
            logging.error(traceback.format_exc())
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue

        if task_env.plan_success and task_env.check_success():
            successful_seed_count += 1
        else:
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue
```

**修改后**：

```python
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed_value, is_test=True, **task_args)
            episode_info = task_env.play_once()
            # check_success() requires self.robot, which close_env() sets to None,
            # so the expert result must be captured before closing the env.
            expert_success = bool(task_env.plan_success and task_env.check_success())
            maybe_close_env(task_env)
        except unstable_error as exc:
            logging.warning("Skipping unstable seed for task=%s seed=%s: %s", task_name, seed_value, exc)
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue
        except Exception:
            logging.error("Expert rollout failed for task=%s seed=%s", task_name, seed_value)
            logging.error(traceback.format_exc())
            maybe_close_env(task_env)
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue

        if not expert_success:
            seed_cursor += 1
            task_args["render_freq"] = render_freq
            continue
        successful_seed_count += 1
```

**修改要点**：

| 行为 | 修改前 | 修改后 |
|------|--------|--------|
| `check_success()` 调用时机 | try/except 之后（`self.robot` 已为 None） | try 块内、`close_env()` 之前（`self.robot` 有效） |
| 结果存储 | 直接调用，无存储 | 捕获到 `expert_success` 布尔变量 |
| 后续判断 | `if task_env.plan_success and task_env.check_success():` | `if not expert_success:` |
| `close_env()` 位置 | try 块内、`play_once()` 之后（bug 版） | try 块内、`expert_success` 赋值之后（L393） |

**`expert_success` 为何不需要在 try 前初始化**：两个 `except` 分支都以 `continue` 结束，因此只有 try 正常完成时才会到达 `if not expert_success:`，此时 `expert_success` 必已赋值。

**风险评估**：🟢 LOW — 修改与 InternVLA-A-series 已验证版本语义完全一致，且仅影响 seed 验证阶段的控制流。

---

#### 1.2 修改二：eval.sh CONDA_ENV 修复（MEDIUM）

**严重程度**：🟡 MEDIUM — 通过 eval.sh 启动评估时会失败（`conda activate internvla_a1_5` 找不到环境），但可绕过

**文件**：`evaluation/RoboTwin/eval.sh` 第 7 行

**问题**：硬编码 `CONDA_ENV=internvla_a1_5`，该名称是从 InternVLA-A-series 代码库复制过来的，本机不存在此环境。itvlaGp 的 conda 环境名为 `itvlaGp`。

**修改前**：

```bash
CONDA_ENV=internvla_a1_5
```

**修改后**：

```bash
CONDA_ENV=${CONDA_ENV:-itvlaGp}
```

**设计说明**：使用 `${VAR:-default}` 模式，与同文件中 `RESIZE_SIZE`、`ACTION_MODE`、`INFER_HORIZON` 等变量的风格一致。允许通过 `CONDA_ENV=myenv bash eval.sh` 覆盖，默认为 `itvlaGp`。

**风险评估**：🟢 NONE — 标准 bash 变量默认值语法，无回归风险。

---

#### 1.3 修改三：eval.sh CLI 传参修复（LOW）

**严重程度**：🟢 LOW — 当前默认值（224）恰好匹配 inference.py 默认值，功能正确。但如果用户设置 `RESIZE_SIZE=384` 期望生效，会被静默忽略

**文件**：`evaluation/RoboTwin/eval.sh`

**问题**：shell 变量已定义（如 `RESIZE_SIZE`），但 python 命令未全部传递给 `inference.py`。当前已修复 `--resize-size`，但 **`--dtype` 和 `--num-episodes` 仍未传递**——通过 eval.sh 运行会使用 inference.py 默认的 `float32`（非手册要求的 `bfloat16`）。

**已完成的修改**（`--resize-size`）：

```bash
python ../../evaluation/RoboTwin/inference.py \
  --ckpt-path "${PRETRAINED_CKPT}" \
  --video-dir "${OUTPUT_PATH}" \
  --task-config "${TASK_CONFIG}" \
  --task-idx "${TASK_IDX}" \
  --action-mode "${ACTION_MODE}" \
  --infer-horizon "${INFER_HORIZON}" \
  --inference-backend "${INFERENCE_BACKEND}" \
```

**修改后**：

```bash
python ../../evaluation/RoboTwin/inference.py \
  --ckpt-path "${PRETRAINED_CKPT}" \
  --video-dir "${OUTPUT_PATH}" \
  --task-config "${TASK_CONFIG}" \
  --task-idx "${TASK_IDX}" \
  --resize-size "${RESIZE_SIZE}" \
  --action-mode "${ACTION_MODE}" \
  --infer-horizon "${INFER_HORIZON}" \
  --inference-backend "${INFERENCE_BACKEND}" \
```

**修改要点**：在 `--task-idx` 之后插入 `--resize-size "${RESIZE_SIZE}" \` 一行。

**仍缺传参**（若需完善 eval.sh，可追加）：

```bash
  --dtype "${DTYPE:-bfloat16}" \
  --num-episodes "${NUM_EPISODES:-100}" \
```

> **推荐**：本手册以 **直接调用 `inference.py`** 为主路径（§7/§8），可精确控制 `--dtype bfloat16` 等参数。eval.sh 仅作便捷包装。

**风险评估**：🟢 NONE — `--resize-size` 传参不改变现有默认行为。

---

#### 1.4 修改四：requirements.txt 添加 scipy（LOW）

**严重程度**：🟢 LOW — `pip install -e ".[all]"` 已通过 `pyproject.toml` 安装 scipy，但如果仅通过 `pip install -r evaluation/RoboTwin/requirements.txt` 设置环境则会缺少 scipy

**文件**：`evaluation/RoboTwin/requirements.txt`

**问题**：`get_keypoints_aloha()` 在 `inference.py` 第 67 行执行 `from scipy.spatial.transform import Rotation`，但 `requirements.txt` 未列出 scipy。

**修改前**：

```
sapien==3.0.0b1
mplib==0.2.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
azure==4.0.0
azure-ai-inference
pyglet<2
moviepy
tyro
h5py
```

**修改后**：

```
sapien==3.0.0b1
mplib==0.2.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
azure==4.0.0
azure-ai-inference
pyglet<2
moviepy
tyro
h5py
scipy
```

**设计说明**：不加版本约束，因为主版本约束已在 `pyproject.toml` 第 61 行定义（`scipy>=1.12.0,<1.16.0`）。此 requirements.txt 是 RoboTwin 评估的补充依赖列表。

**风险评估**：🟢 NONE — 仅添加依赖声明，不影响代码行为。

---

#### 1.5 修改实施步骤

所有修改在 conda 环境 `itvlaGp` 中进行（如尚未创建，先完成第 3 节环境准备）：

```bash
cd /home/luogang/SRC/Robot/itvlaGp
```

**步骤 1：修复 inference.py（修改一）**

用编辑器打开 `evaluation/RoboTwin/inference.py`，定位到第 387-410 行（seed 验证循环中的 try/except 块），将其替换为上述修改后的代码。

或使用 `sed` 进行等效修改（注意：推荐使用编辑器以避免行号偏移问题）：

```bash
# 验证修改前的代码确实是 bug 版本
grep -n "expert_success\|maybe_close_env\|check_success" evaluation/RoboTwin/inference.py | head -10
# 预期：expert_success 与 check_success 在 L392 附近，maybe_close_env 紧随其后（L393）
```

**步骤 2：修复 eval.sh（修改二 + 三）**

```bash
# 修改二：CONDA_ENV
sed -i 's/^CONDA_ENV=internvla_a1_5$/CONDA_ENV=${CONDA_ENV:-itvlaGp}/' evaluation/RoboTwin/eval.sh

# 修改三：添加 --resize-size
sed -i '/--task-idx "${TASK_IDX}"/a\  --resize-size "${RESIZE_SIZE}" \\' evaluation/RoboTwin/eval.sh
```

**步骤 3：修复 requirements.txt（修改四）**

```bash
echo "scipy" >> evaluation/RoboTwin/requirements.txt
```

**步骤 4：验证所有修改**

运行第 2 节的测试套件。

---

### 2. 代码修改验证测试

> 本节提供两级测试：**静态源码验证**（无需 GPU/SAPIEN，可在任何环境运行）和**集成冒烟测试**（需要完整评估环境 + GPU）。

#### 2.1 静态测试 v4（16 项）

> 静态测试验证 **V2 四项代码修复** 是否已正确应用（非 checkpoint 权重加载测试）。LOG0807192 系列均记录 **16/16 PASS**。
>
> **注意**：旧版 regex 会误匹配 `maybe_close_env()` 内的 try 块（L340）导致误报 FAIL。v4 使用 `play_once()` 定位 seed 验证段，并跳过注释行（L390 注释同时含 `check_success` 和 `close_env` 字样）。

**一键运行**（在 `${REPO_ROOT}` 下）：

```bash
cd /home/luogang/SRC/Robot/itvlaGp
python3 << 'STATIC_TEST_EOF'
import re, sys

results = []

def check(name, condition, msg=""):
    results.append((name, condition, msg))
    status = "PASS" if condition else "FAIL"
    detail = f" — {msg}" if msg else ""
    print(f"  [{status}] {name}{detail}")

print("=" * 60)
print("Static Code Modification Tests (v4)")
print("=" * 60)

with open("evaluation/RoboTwin/inference.py") as f:
    src = f.read()

print("\n--- Test 1: inference.py check_success ordering ---")
idx = src.find("episode_info = task_env.play_once()")
check("T1.1: play_once try block found", idx > 0)
if idx > 0:
    chunk = src[idx:idx+800]
    lines = chunk.split("\n")
    cs_line = ce_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "check_success" in line and cs_line is None:
            cs_line = i
        if "close_env" in line and ce_line is None:
            ce_line = i
    check("T1.2: check_success inside try", cs_line is not None)
    check("T1.3: close_env inside try", ce_line is not None)
    if cs_line is not None and ce_line is not None:
        check("T1.4: check_success BEFORE close_env", cs_line < ce_line,
              f"check_success at rel-line {cs_line}, close_env at rel-line {ce_line}")

check("T1.5: expert_success variable exists", "expert_success" in src)
bare_check = bool(re.search(r'^\s{8}if task_env\.plan_success and task_env\.check_success',
                            src, re.MULTILINE))
check("T1.6: no bare check_success after try/except", not bare_check)

print("\n--- Test 2: eval.sh CONDA_ENV ---")
with open("evaluation/RoboTwin/eval.sh") as f:
    eval_sh = f.read()
check("T2.1: CONDA_ENV uses overridable default", "CONDA_ENV=${CONDA_ENV:-itvlaGp}" in eval_sh)
check("T2.2: no hardcoded internvla_a1_5", "CONDA_ENV=internvla_a1_5" not in eval_sh)

print("\n--- Test 3: requirements.txt scipy ---")
with open("evaluation/RoboTwin/requirements.txt") as f:
    reqs = f.read()
check("T3.1: scipy in requirements.txt", "scipy" in reqs)

print("\n--- Test 4: eval.sh CLI pass-through ---")
check("T4.1: --resize-size in eval.sh", "--resize-size" in eval_sh)
var_to_flag = {
    "RESIZE_SIZE": "resize-size",
    "ACTION_MODE": "action-mode",
    "INFER_HORIZON": "infer-horizon",
    "INFERENCE_BACKEND": "inference-backend",
    "TASK_CONFIG": "task-config",
    "TASK_IDX": "task-idx",
}
for var, flag in var_to_flag.items():
    check(f"T4.2: {var} -> --{flag}", f"--{flag}" in eval_sh)

print("\n" + "=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"Results: {passed} passed, {failed} failed, {len(results)} total")
if failed > 0:
    sys.exit(1)
else:
    print("\nAll static tests PASSED")
STATIC_TEST_EOF
```

**各项说明**（共 16 项）：

| 组 | 测试项 | 预期 | 失败含义 |
|----|--------|------|---------|
| T1.1–T1.4 | seed 验证段 `check_success` 顺序 | `check_success` 在 `close_env` 之前 | §1.1 修复未应用 |
| T1.5 | `expert_success` 变量 | 存在 | §1.1 修复未应用 |
| T1.6 | 无裸 `check_success` 调用 | 通过 | 旧版 bug 模式残留 |
| T2.1–T2.2 | `eval.sh` CONDA_ENV | `${CONDA_ENV:-itvlaGp}` | §1.2 修复未应用 |
| T3.1 | `requirements.txt` 含 scipy | 通过 | §1.4 修复未应用 |
| T4.1–T4.2 | `eval.sh` CLI 传参 | 含 `--resize-size` 等 6 个 flag | §1.3 修复未应用 |

> 若静态测试失败，按 §1.5 完成四项代码修复后重新运行。修复完成前请勿启动正式评估。

**辅助 grep 检查**（可选）：

```bash
grep -n "task_env.check_success\|expert_success" evaluation/RoboTwin/inference.py
# 预期：expert_success 在 L392 附近；不应有 try/except 之后的裸 check_success 调用
```

#### 2.2 静态测试结果判定

| 测试类别 | 必须通过 | 阻断评估 |
|---------|---------|---------|
| T1.1–T1.6（check_success 修复） | ✅ 全部（6 项） | 是 — 不通过则评估必崩 |
| T2.1–T2.2（CONDA_ENV） | ✅ 全部（2 项） | 否 — 可绕过 eval.sh 直接调用 inference.py |
| T3.1（scipy） | ✅（1 项） | 否 — `pip install -e ".[all]"` 通常已安装 |
| T4.1–T4.2（eval.sh CLI） | ✅ 全部（7 项） | 否 — 直接调用 inference.py 可绕过 |
| **合计** | **16/16 PASS** | |

#### 2.3 集成冒烟测试

> 此测试需要完整的评估环境（conda `itvlaGp` + SAPIEN + GPU）。在完成第 3 节环境准备后运行。

**测试 1C：2 episode 端到端测试（修改一验证）**

此测试验证 `check_success` bug 修复后评估可以正常运行，不再崩溃：

```bash
conda activate itvlaGp
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model

cd ${REPO_ROOT}/third_party/RoboTwin

python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir "/tmp/code_fix_test" \
  --task-config demo_clean \
  --task-idx 46 \
  --action-mode abs \
  --infer-horizon 20 \
  --inference-backend standard \
  --num-episodes 2 \
  --dtype bfloat16 \
  2>&1 | tee /tmp/code_fix_test.log

echo ""
echo "========== Test Results =========="
echo "Exit code: $?"

# 检查致命错误
if grep -qi "AttributeError\|NoneType\|is_left_gripper_open" /tmp/code_fix_test.log; then
    echo "FAIL: AttributeError detected — check_success bug NOT fixed"
else
    echo "PASS: No AttributeError — check_success bug is fixed"
fi

# 检查是否产生了结果
VIDEOS=$(ls /tmp/code_fix_test/*.mp4 2>/dev/null | wc -l)
echo "Video files produced: ${VIDEOS}"
if [ "${VIDEOS}" -ge 2 ]; then
    echo "PASS: At least 2 episodes completed"
else
    echo "FAIL: Expected at least 2 episode videos"
fi

# 检查 success rate 输出
if grep -q "Success rate:" /tmp/code_fix_test.log; then
    echo "PASS: Success rate reported"
    grep "Success rate:" /tmp/code_fix_test.log
else
    echo "INFO: No explicit Success rate line (check video filenames for success_/failure_ prefix)"
fi
```

**预期结果**：

| 检查项 | 预期 |
|--------|------|
| 退出码 | 0 |
| AttributeError | 无 |
| 视频文件 | ≥ 2 个 .mp4 |
| 运行时间 | 2-5 分钟 |

**测试 3B：scipy 可导入（修改四验证）**

```bash
conda activate itvlaGp
python3 -c "
from scipy.spatial.transform import Rotation
print('PASS: scipy importable')

with open('evaluation/RoboTwin/requirements.txt') as f:
    assert 'scipy' in f.read(), 'FAIL: scipy not in requirements.txt'
print('PASS: scipy in requirements.txt')
"
```

#### 2.4 集成测试判定

| 测试 | 必须通过 | 阻断评估 |
|------|---------|---------|
| 1C（2 episode 端到端） | ✅ | 是 — 必须在正式评估前通过 |
| 3B（scipy 可导入） | ✅ | 否 |

---

### 3. 环境准备

> 所有操作在 conda 环境 `itvlaGp` 中进行。

#### 3.1 创建 conda 环境

```bash
conda create -n itvlaGp python=3.10 -y
conda activate itvlaGp
```

> **为什么用 Python 3.10？** 虽然 CLAUDE.md 模板写的是 3.11，但本机验证通过的参考环境 `ivla15` 使用 3.10，且 RoboTwin 的 SAPIEN 3.0.0b1 和 mplib 0.2.1 目标版本为 3.10。`pyproject.toml` 要求 `>=3.10`。

#### 3.2 安装 PyTorch（CUDA 12.8）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(f'torch={torch.__version__}, CUDA={torch.version.cuda}, GPU={torch.cuda.get_device_name(0)}')"
# 预期: torch=2.11.0+cu128, CUDA=12.8, GPU=NVIDIA RTX PRO 6000 Blackwell Server Edition
```

#### 3.3 安装 itvlaGp 包（可编辑模式）

```bash
cd /home/luogang/SRC/Robot/itvlaGp
pip install -e ".[all]"
```

这会安装 `pyproject.toml` 中的全部依赖：datasets、diffusers、accelerate、draccus、omegaconf、einops、wandb、imageio、scipy、flash-linear-attention、safetensors 等。

验证：

```bash
python -c "from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config; print('InternVLA-A1.5 config OK')"
python -c "from lerobot.policies.internvla_a1_5.keypoints import TrackEncoder; print('TrackEncoder OK')"
```

#### 3.4 锁定 transformers==5.2.0 并打 Qwen3.5 补丁

**关键**：transformers 必须精确为 5.2.0。更高版本（如 5.14.1）会导致 `create_causal_mask()` API 不兼容，补丁后的 `modeling_qwen3_5.py` 无法工作。这在基线版评估中已确认为 Problem #8。

```bash
pip install "transformers==5.2.0"

# 找到 transformers 安装位置
TRANSFORMERS_DIR=$(python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")
echo "Transformers at: ${TRANSFORMERS_DIR}"

# 复制 Qwen3.5 补丁（核心：支持 knowledge insulation、3-path MoT 等自定义 attention）
cp -r src/lerobot/policies/internvla_a1_5/transformers_replace/models/* ${TRANSFORMERS_DIR}/models/

# 顺便复制 pi0/pi05 的补丁（避免间接 import 报错）
cp -r src/lerobot/policies/pi0/transformers_replace/models/* ${TRANSFORMERS_DIR}/models/ 2>/dev/null || true
cp -r src/lerobot/policies/pi05/transformers_replace/models/* ${TRANSFORMERS_DIR}/models/ 2>/dev/null || true
```

验证：

```bash
python -c "from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration; print('Qwen3.5 patch OK')"
python -c "import transformers; assert transformers.__version__ == '5.2.0', f'Need 5.2.0, got {transformers.__version__}'"
```

> **为什么需要补丁？** itvlaGp 的 `modeling_internvla_a1_5.py` 导入了 `from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration`，并使用了自定义 attention 修改以支持 knowledge insulation、foresight token 和 3-path MoT 的 cross-path K/V 拼接。这些修改不在上游 transformers 中。

#### 3.5 编译安装 flash-attn

**关键**：必须使用 `--no-build-isolation --no-cache-dir` 从源码编译，确保 CUDA kernel 与当前 torch ABI 匹配。使用缓存的 wheel 会导致 `undefined symbol` 运行时错误。编译时间约 15-100 分钟（取决于 MAX_JOBS）。

```bash
export CUDA_HOME=/usr/local/cuda-12.8
MAX_JOBS=16 pip install flash-attn --no-build-isolation --no-cache-dir
```

验证：

```bash
python -c "import flash_attn; print(f'flash-attn {flash_attn.__version__}')"
# 预期: flash-attn 2.8.3.post1
```

> **如果编译失败**：
> - 检查 `nvcc --version` 是否为 12.8
> - 检查 `CUDA_HOME` 是否设置为 `/usr/local/cuda-12.8`
> - 尝试 `MAX_JOBS=4` 降低并行编译任务数减少内存占用
> - 清理旧缓存：`pip cache purge`

#### 3.6 安装 flash-linear-attention 和 causal-conv1d

Qwen3.5 补丁中的 Gated DeltaNet 线性注意力层需要这两个包。

```bash
pip install flash-linear-attention==0.5.2 causal-conv1d==1.6.1 --no-build-isolation
```

> `pyproject.toml` 约束为 `flash-linear-attention>=0.2.0,<1.0.0`；本机 LOG 验证使用 0.5.2，以 `pip show flash-linear-attention` 实际版本为准。

验证：

```bash
python -c "import fla; print('flash-linear-attention OK')"
python -c "import causal_conv1d; print('causal-conv1d OK')"
```

#### 3.7 安装 RoboTwin 评估依赖

```bash
cd /home/luogang/SRC/Robot/itvlaGp
pip install -r evaluation/RoboTwin/requirements.txt
```

该文件安装的关键包（修改四已添加 scipy）：

| 包 | 版本 | 用途 |
|----|------|------|
| `sapien` | 3.0.0b1 | SAPIEN 物理引擎（GPU 渲染 + 碰撞） |
| `mplib` | 0.2.1 | 运动规划库（expert policy 使用） |
| `trimesh` | 4.4.3 | 3D mesh 处理 |
| `open3d` | 0.18.0 | 3D 数据处理 |
| `imageio` | 2.34.2 | 视频录制 |
| `h5py` | - | HDF5 数据格式 |
| `moviepy` | - | 视频处理 |
| `tyro` | - | CLI 参数解析 |
| `scipy` | - | 空间变换（关键点提取需要 `scipy.spatial.transform.Rotation`） |

额外安装 gymnasium（RoboTwin 的 `_base_task.py` 需要但未列入 requirements.txt）：

```bash
pip install gymnasium
```

#### 3.8 安装 CuRobo（Blackwell sm_120）

CuRobo 是 NVIDIA 的 CUDA 加速运动规划库，仅用于评估中的 **seed 验证阶段**（运行 expert policy 验证 seed 是否可行）。Blackwell GPU（RTX PRO 6000）使用 sm_120 计算能力。

> **重要**：`cuobjdump` 检查 .so 是否含 sm_120 **不可靠**（LOG 曾显示有 sm_120 但运行时仍失败）。Blackwell 上建议直接重编译，并用下方**运行时 kinematics 验证**确认。

**推荐：直接重编译 sm_120**（约 10–30 分钟）：

```bash
cd /home/luogang/share/zwy/Projects/RoboTwin/envs/curobo
rm -f src/curobo/curobolib/*.so
rm -rf build
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 pip install -e . --no-build-isolation --no-cache-dir --force-reinstall --no-deps
```

**安装运行时依赖**（`--no-deps` 会跳过，LOG 证实必须手动安装）：

```bash
pip install setuptools_scm warp-lang yourdfpy
```

> `warp-lang` 的 import 名为 `warp`；CuRobo 的 URDF 解析依赖 `yourdfpy`；版本号获取依赖 `setuptools_scm`。

验证（import + 运行时 kinematics）：

```bash
cd /home/luogang/SRC/Robot/itvlaGp
python -c "import curobo; print('CuRobo import OK')"
python -c "from curobo.curobolib import kinematics; print('CuRobo kinematics OK')"
python -c "from curobo.types.math import Pose; print('CuRobo Pose OK')"
```

> **如果 kinematics 报 `no kernel image is available for execution on the device`**：按上方重编译步骤操作，然后重装 `setuptools_scm warp-lang yourdfpy`。

> **注意**：CuRobo 以可编辑模式安装在共享源码目录。重编译 .so 会影响所有使用该目录的 conda 环境。

#### 3.9 安装 ffmpeg

```bash
conda install -c conda-forge "ffmpeg>=7" -y
```

验证：

```bash
ffmpeg -version 2>/dev/null | head -1
# 预期: ffmpeg version 7.x 或更高
```

#### 3.10 打 SAPIEN 和 mplib 补丁

RoboTwin 需要对 SAPIEN 的 URDF 加载器和 mplib 的运动规划器做两个补丁：

```bash
conda activate itvlaGp

# ---- 补丁 1：SAPIEN urdf_loader.py — 添加 encoding="utf-8" ----
# 原因：URDF 文件中含有非 ASCII 字符时，默认编码会导致读取错误
SAPIEN_LOCATION=$(pip show sapien | grep 'Location' | awk '{print $2}')/sapien
URDF_LOADER=${SAPIEN_LOCATION}/wrapper/urdf_loader.py
if [ -f "${URDF_LOADER}" ]; then
    sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "${URDF_LOADER}"
    echo "✓ SAPIEN urdf_loader.py patched"
else
    echo "WARNING: urdf_loader.py not found at ${URDF_LOADER}"
fi

# ---- 补丁 2：mplib planner.py — 移除 'or collide' 条件 ----
# 原因：移除运动规划中过于保守的碰撞检测短路条件，否则 expert policy 在某些配置下会放弃规划
MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib
PLANNER=${MPLIB_LOCATION}/planner.py
if [ -f "${PLANNER}" ]; then
    sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${PLANNER}"
    echo "✓ mplib planner.py patched"
else
    echo "WARNING: planner.py not found at ${PLANNER}"
fi
```

验证补丁是否生效：

```bash
# 检查 SAPIEN 补丁
grep 'encoding="utf-8"' ${URDF_LOADER} && echo "SAPIEN patch confirmed" || echo "SAPIEN patch MISSING"

# 检查 mplib 补丁（应该不再包含 'or collide'）
grep 'or collide' ${PLANNER} && echo "mplib patch MISSING" || echo "mplib patch confirmed"
```

#### 3.11 创建 RoboTwin 符号链接

`inference.py` 在代码中硬编码了 `ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "RoboTwin"`（第 20-21 行），因此必须在 `third_party/` 下提供 RoboTwin。本机已在 `/home/luogang/share/zwy/Projects/RoboTwin/` 有完整安装（含 50 个任务环境、3D 资产、已编译的 CuRobo），用符号链接即可：

```bash
cd /home/luogang/SRC/Robot/itvlaGp
mkdir -p third_party
ln -sfn /home/luogang/share/zwy/Projects/RoboTwin third_party/RoboTwin
```

验证：

```bash
ls third_party/RoboTwin/envs/__init__.py && echo "RoboTwin link OK"
ls third_party/RoboTwin/envs/stack_bowls_three.py && echo "stack_bowls_three task OK"
ls third_party/RoboTwin/assets/embodiments/aloha-agilex/ && echo "ALOHA assets OK"
```

#### 3.12 环境变量配置

创建激活脚本便于每次评估前快速配置：

```bash
cat > /home/luogang/SRC/Robot/itvlaGp/activate_itvlaGp.sh << 'ACTIVATE_EOF'
#!/usr/bin/env bash
# itvlaGp 评估环境激活脚本
conda activate itvlaGp
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export HF_HOME="${HOME}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "itvlaGp evaluation environment activated."
echo "  REPO_ROOT=${REPO_ROOT}"
echo "  CKPT=${CKPT}"
echo "  PYTHONPATH includes src/ and third_party/RoboTwin"
echo "  CUDA_HOME=${CUDA_HOME}"
ACTIVATE_EOF
chmod +x /home/luogang/SRC/Robot/itvlaGp/activate_itvlaGp.sh
```

使用方式：

```bash
source /home/luogang/SRC/Robot/itvlaGp/activate_itvlaGp.sh
```

---

### 4. itvlaGp 3-path 推理架构深度解读

> 本节深入分析 itvlaGp 在推理时的数据流和模型组件交互，帮助理解评估过程中各模块的角色。

#### 4.1 三条路径的结构

itvlaGp 在标准 Qwen3.5-2B VLM 骨干上附加了两个独立的 Expert 模块，共形成三条并行路径：

```mermaid
graph TD
    subgraph "Path 1: VLM Prefix (Qwen3.5-2B)"
        V_IN["vision tokens + language tokens"] --> V_ATTN["Self-Attention<br/>(prefix only)"]
        V_ATTN --> V_OUT["prefix K/V cache"]
    end
    
    subgraph "Path 2: Keypoint Expert (hidden=1024)"
        K_IN["state(1) + history_track(14) + query(14)<br/>= 29 tokens"] --> K_ATTN["Cross-Path Attention<br/>[prefix K/V, kpt K/V]"]
        K_ATTN --> K_OUT["keypoint K/V cache"]
    end
    
    subgraph "Path 3: Action Expert (hidden=1024)"
        A_IN["state(1) + learnable(50) + noisy_action(50)<br/>= 101 tokens"] --> A_ATTN["Cross-Path Attention<br/>[prefix K/V, kpt K/V, action K/V]"]
        A_ATTN --> A_OUT["predicted velocity field"]
    end
    
    V_OUT --> K_ATTN
    V_OUT --> A_ATTN
    K_OUT --> A_ATTN
```

**注意力规则**（定义于 `compute_layer_complete_3path`，`modeling_internvla_a1_5.py:343-553`）：

| 路径 | Queries 能看到的 K/V | Knowledge Insulation |
|------|---------------------|---------------------|
| VLM Prefix | 仅 prefix 自身 | — |
| Keypoint Expert | prefix（可选 detach）+ kpt 自身 | `knowledge_insulation_kpt=True` → detach prefix K/V |
| Action Expert | prefix（可选 detach）+ kpt（可选 detach）+ action 自身 | `knowledge_insulation=True` → detach prefix K/V |

#### 4.2 关键点 suffix 的构建

关键点 suffix 由 `embed_kpt_suffix()` 方法（`modeling_internvla_a1_5.py:1562-1617`）构建，总共 29 个 token（对于 J=14 关节）：

```
kpt_suffix = [state_token(1), history_track_tokens(14), query_tokens(14)]
```

1. **state_token**（1 个）：机器人关节状态经 `kpt_state_proj(Linear(32→1024))` 投影
2. **history_track_tokens**（14 个）：`TrackEncoder` 对 `his_kpts[H, 14, 3]` 历史轨迹编码，每个关节产生 1 个 1024 维 token
3. **query_tokens**（14 个）：`keypoint_embedding(Embedding(14, 1024))` 的可学习权重，用于查询当前和未来的 3D 关键点位置

#### 4.3 TrackEncoder 架构

`TrackEncoder`（`keypoints.py:244-313`）是从 GeoPredict 移植的核心编码器，将 3D 点轨迹历史编码为固定维度的 per-joint token：

```
输入: his_kpts [B, H, J, 3]  （H 帧历史，J=14 关节，3D 坐标）
  ↓
PointPatchEmbedding: 1D 卷积时间分片 → [B, num_patches, J, 256]
  ↓
CrossAttentionBlock: 可学习 queries 对分片的 cross-attention（附加时间位置编码）
  ↓
linear_transform: Sequential(Linear, ReLU, Dropout, Linear) → [B, J, 512]
  ↓
final_norm: LayerNorm
  ↓
track_fusion_layer: Linear(512→1024) → [B, J, 1024]
  ↓
输出: [B, J*1, 1024]  （J 个 per-joint tokens）
```

> **注意**：GeoPredict 原始的 `track_fusion_layer` 输出维度是 2048，本移植版改为 1024 以匹配 keypoint expert 的 hidden_size。因此权重加载时跳过 `track_fusion_layer`，使用随机初始化 + Phase 1 训练。

#### 4.4 推理数据流（sample_actions）

推理入口是 `InternVLAA15.sample_actions()`（`modeling_internvla_a1_5.py:1274-1398`）。完整数据流：

```
Step 1: 嵌入 VLM prefix（vision + language tokens）→ use_cache=True → prefix K/V cache
  |
Step 2: 构建 kpt suffix（embed_kpt_suffix）
  |→ kpt_state_proj(state) → 1 token
  |→ TrackEncoder(his_kpts, his_len) → 14 tokens
  |→ keypoint_embedding.weight → 14 tokens
  |→ 合并为 29 tokens
  |
Step 3: 前向计算 kpt suffix → inputs_embeds=[None, kpt_embs, None]
  |→ 3-path forward：仅计算 keypoint expert，prefix 从 cache 读取
  |→ kpt K/V cache 追加到 past_key_values
  |→ 此步仅执行 **1 次/env step**
  |
Step 4: Flow Matching 去噪循环（10 步）
  |→ 每步：构建 action suffix（state + learnable + noisy actions = 101 tokens）
  |→ inputs_embeds=[None, None, suffix_embs]
  |→ 3-path forward：仅计算 action expert，prefix+kpt 从 cache 读取
  |→ 读取 outputs_embeds[2][-chunk_size:]
  |→ action_out_proj → velocity field
  |→ 更新 action = action + dt * velocity（Euler 积分）
  |
Step 5: 返回 action chunk [B, chunk_size, action_dim]
```

**性能关键点**：关键点 cache 在每个 env step 只计算 **1 次**（29 tokens），之后在 10 次去噪迭代中复用。这保证了关键点融合不会显著增加推理延迟。

#### 4.5 Optimized 后端为何不支持关键点

`InternVLAA15Optimized`（`modeling_internvla_a1_5_optimized.py`）继承自 `InternVLAA15`，但其 `sample_actions()` 方法（第 422 行）的参数签名中 **没有** `his_kpts` 和 `his_len` 参数。它仅使用 2-path `[prefix, action]` 前向传播，不支持关键点 suffix 的 K/V 缓存构建。使用 optimized 后端会 **静默忽略** 关键点数据，使 keypoint expert 形同虚设，导致评估结果无意义。

**因此，评估 itvlaGp checkpoint 时必须使用 `--inference-backend standard`。**

---

### 5. 评估代码解读（inference.py 中的关键点支持）

> 完整代码在 `evaluation/RoboTwin/inference.py`（527 行）。`require_robotwin()` 报错文案提及 `git submodule`，但本手册使用 **symlink** 方式（§3.11）提供 RoboTwin。本节重点解读 GeoPredict 关键点融合相关部分。

#### 5.1 关键点链接定义

```python
# inference.py:45-48
ALOHA_KEYPOINT_LINKS = [
    "fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6", "left_camera",
    "fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6", "right_camera",
]
```

14 个关节链接，每臂 7 个（6 个 ARX5 连杆 + 1 个腕部相机）。与训练数据中的 `util_scripts/generate_aloha_keypoints.py` 使用的链接名称完全一致。

#### 5.2 关键点提取函数

```python
# inference.py:51-77
def get_keypoints_aloha(robot_entity, footprint_pose=None):
    """Extract 14 keypoint 3D positions (footprint-relative) from SAPIEN aloha robot."""
    if footprint_pose is None:
        fp_link = robot_entity.find_link_by_name("footprint")
        footprint_pose = fp_link.get_pose()

    fp_pos = np.asarray(footprint_pose.p, dtype=np.float64)
    q = footprint_pose.q
    from scipy.spatial.transform import Rotation
    fp_rot_inv = Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv().as_matrix()

    keypoints = np.zeros((14, 3), dtype=np.float32)
    for i, link_name in enumerate(ALOHA_KEYPOINT_LINKS):
        link = robot_entity.find_link_by_name(link_name)
        world_pos = np.asarray(link.get_pose().p, dtype=np.float64)
        keypoints[i] = (fp_rot_inv @ (world_pos - fp_pos)).astype(np.float32)

    return keypoints, footprint_pose
```

**核心逻辑**：
1. 获取 `footprint` 链接的 pose（固定底座，首次调用后缓存）
2. 计算 footprint 坐标系的逆旋转矩阵
3. 对每个关键点链接：获取其世界坐标 → 转为 footprint 相对坐标
4. 返回 `[14, 3]` 的关键点数组

> **坐标系约定**：footprint-relative，与训练数据（使用 pinocchio FK 从 `arx5_description_isaac.urdf` 计算）一致。

#### 5.3 推理循环中的关键点集成

```python
# inference.py:426-460（在 infer_once 函数内）
use_kpt = getattr(config, "enable_keypoint_predictor", False)
if use_kpt:
    J = getattr(config, "num_keypoint_joints", 14)
    H = getattr(config, "keypoint_history_max_len", 1000)
    his_kpts = np.zeros((H, J, 3), dtype=np.float32)
    his_len = 0
    footprint_pose = None

while task_env.take_action_cnt < task_env.step_lim:
    observation = task_env.get_obs()
    sample = build_sample(observation, task_env.get_instruction(), dtype)
    sample = input_transforms(sample)
    # ... image recording ...

    if use_kpt:
        robot_entity = task_env.robot.left_entity
        kpt_t, footprint_pose = get_keypoints_aloha(robot_entity, footprint_pose)
        if his_len < H:
            his_kpts[his_len] = kpt_t
        else:
            his_kpts = np.roll(his_kpts, -1, axis=0)
            his_kpts[-1] = kpt_t
        his_len = min(his_len + 1, H)

    if not action_plan:
        batch = to_policy_batch(sample, device, dtype)
        if use_kpt:
            batch["observation.his_kpts"] = torch.from_numpy(his_kpts).unsqueeze(0).to(device=device, dtype=dtype)
            batch["observation.his_len"] = torch.tensor([his_len], dtype=torch.long, device=device)
        with torch.no_grad():
            action_pred = policy.predict_action_chunk(batch)
        # ... action post-processing ...
```

**数据流**：
1. **初始化**：创建 `his_kpts[1000, 14, 3]` 滚动缓冲区和 `his_len` 计数器
2. **每个 env step**：
   - 从 SAPIEN 机器人提取当前 14 个关节的 3D 坐标
   - 追加到滚动缓冲区（超过 1000 帧则滚动丢弃最旧帧）
   - 递增 `his_len`（上限 1000）
3. **每次策略预测**（每 `infer_horizon` 步触发一次）：
   - 将 `his_kpts` 和 `his_len` 打包为 tensor 加入 batch
   - 传递给 `policy.predict_action_chunk(batch)`
   - 模型内部调用 `sample_actions()` 执行 3-path 推理

#### 5.4 动作后处理

动作后处理与基线版完全一致：

1. `compact_reordered_dual_arm_actions()`：将 16 维重排序动作压缩回 14 维（跳过 indices 6, 14, 15）
2. `unnormalize_fn`：使用 `stats.json` 中的 `aloha.action.mean/std` 反标准化
3. 如果 `action_mode=delta`（本次评估不使用），加上当前关节角
4. gripper 值（indices 6, 13）裁剪到 [0, 1]

#### 5.5 关键注意事项

| 注意事项 | 说明 |
|---------|------|
| `robot_entity` 必须有效 | 关键点提取使用 `task_env.robot.left_entity`，如果 `close_env()` 已被调用（会将 `robot` 设为 None），将导致 AttributeError。此 bug 已在第 1.1 节修复：`expert_success = bool(task_env.plan_success and task_env.check_success())` 在 `maybe_close_env()` 之前捕获结果。 |
| `his_len` 的含义 | 当 `his_len < H` 时，仅 `his_kpts[:his_len]` 是有效数据，后面是零填充。TrackEncoder 使用 `his_len` 来忽略零填充帧。 |
| 首帧关键点 | 第一个 env step 的 `his_len=1`，只有 1 帧历史。TrackEncoder 的 PointPatchEmbedding 使用 stride=4 的 1D 卷积，因此 <4 帧时实际使用的 patch 数为 1（zero-padded）。 |
| `footprint_pose` 缓存 | ALOHA 底座固定不动，footprint pose 首次提取后缓存，避免重复查询。 |
| dtype 一致性 | `his_kpts` 以 float32 提取，转为 batch tensor 时转换为 dtype（bfloat16）。 |

---

### 6. 全流程验证检查表

> 在正式评估前，执行以下 15 项预检确保环境、checkpoint 和 V2 代码修复就绪。LOG0807192 系列评估均执行此脚本，要求 **15/15 PASS**。

```bash
conda activate itvlaGp
cd /home/luogang/SRC/Robot/itvlaGp
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.8

echo "=== itvlaGp 15-item preflight ==="

echo -n "[1/15] Conda env: "
conda info --envs | grep -q "itvlaGp" && echo "OK" || echo "MISSING"

echo -n "[2/15] PyTorch: "
python -c "import torch; assert torch.cuda.is_available(); print(f'OK (torch={torch.__version__}, CUDA={torch.version.cuda})')" 2>/dev/null || echo "FAIL"

echo -n "[3/15] transformers: "
python -c "import transformers; assert transformers.__version__ == '5.2.0'; print('OK (5.2.0)')" 2>/dev/null || echo "FAIL"

echo -n "[4/15] Qwen3.5 patch: "
python -c "from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "[5/15] flash-attn: "
python -c "import flash_attn; print(f'OK ({flash_attn.__version__})')" 2>/dev/null || echo "FAIL"

echo -n "[6/15] flash-linear-attention: "
python -c "import fla; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "[7/15] SAPIEN: "
python -c "import sapien; print(f'OK ({sapien.__version__})')" 2>/dev/null || echo "FAIL"

echo -n "[8/15] CuRobo: "
python -c "import curobo; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "[9/15] scipy: "
python -c "from scipy.spatial.transform import Rotation; print('OK')" 2>/dev/null || echo "FAIL"

echo -n "[10/15] RoboTwin link: "
ls third_party/RoboTwin/envs/__init__.py >/dev/null 2>&1 && echo "OK" || echo "FAIL — run §3.11 symlink step"

echo -n "[11/15] stack_bowls_three: "
ls third_party/RoboTwin/envs/stack_bowls_three.py >/dev/null 2>&1 && echo "OK" || echo "FAIL"

echo -n "[12/15] Checkpoint: "
ls "${CKPT}/config.json" "${CKPT}/model.safetensors" "${CKPT}/stats.json" >/dev/null 2>&1 && echo "OK" || echo "FAIL"

echo -n "[13/15] Config check: "
python -c "
import json
c = json.load(open('${CKPT}/config.json'))
assert c.get('enable_keypoint_predictor') == True
assert c.get('num_keypoint_joints') == 14
print(f'OK (kpt=True, J=14; action_loss_only={c.get(\"action_loss_only\")} in config, inference.py L270 forces True at load)')
" 2>/dev/null || echo "FAIL"

echo -n "[14/15] check_success fix: "
python3 -c "
src = open('evaluation/RoboTwin/inference.py').read()
idx = src.find('episode_info = task_env.play_once()')
assert idx > 0
chunk = src[idx:idx+800]
lines = chunk.split('\n')
cs = ce = None
for i, line in enumerate(lines):
    s = line.strip()
    if s.startswith('#'):
        continue
    if 'check_success' in line and cs is None:
        cs = i
    if 'close_env' in line and ce is None:
        ce = i
assert cs is not None and ce is not None and cs < ce
assert 'expert_success' in src
print('OK (check_success before close_env, expert_success present)')
" 2>/dev/null || echo "FAIL — run Section 1 fixes first!"

echo -n "[15/15] Disk space: "
df -h / | awk 'NR==2{print "OK (" $4 " free)"}'
```

**各项说明**：

| # | 检查项 | 失败时的处理 |
|---|--------|------------|
| 1 | conda 环境 `itvlaGp` 存在 | §3.1 创建环境 |
| 2 | PyTorch + CUDA 可用 | 检查 GPU 驱动、`nvidia-smi` |
| 3 | transformers **== 5.2.0** | §11.4 降级并重新打补丁 |
| 4 | Qwen3.5 patch 可 import | §3.4 复制补丁 |
| 5 | flash-attn 可 import | §11.1 ABI 问题 |
| 6 | flash-linear-attention (`fla`) | §3.6 |
| 7 | SAPIEN 可 import | §3.7 |
| 8 | CuRobo 可 import | §3.8、§11.2 |
| 9 | scipy 可 import | §1.4 或 `pip install scipy` |
| 10 | RoboTwin symlink 可用 | §3.11 创建符号链接 |
| 11 | `stack_bowls_three.py` 存在 | 确认 RoboTwin 安装完整 |
| 12 | Checkpoint 三文件齐全 | 检查 `CKPT` 路径；见 §11.12 `.gstmp` 重命名 |
| 13 | `enable_keypoint_predictor=true`, `num_keypoint_joints=14` | 确认是 itvlaGp 3-path checkpoint；`action_loss_only` 在 config 中可为 `false` |
| 14 | V2 修复 #1：`expert_success` + `check_success` 在 `close_env` 前 | §1.1 |
| 15 | 磁盘剩余空间充足 | §10.6、§11.13 |

**全部 15 项通过后方可进行冒烟测试。** 如果第 14 项失败，必须先完成第 1 节的代码修改。

---

### 7. 冒烟测试（2 episode 快速验证）

#### 7.1 运行冒烟测试

```bash
conda activate itvlaGp
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

mkdir -p ${REPO_ROOT}/outputs/logs

cd ${REPO_ROOT}/third_party/RoboTwin

python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir "${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_010k/smoke/demo_clean/stack_bowls_three" \
  --task-config demo_clean \
  --task-idx 46 \
  --action-mode abs \
  --infer-horizon 20 \
  --inference-backend standard \
  --num-episodes 2 \
  --dtype bfloat16 \
  2>&1 | tee ${REPO_ROOT}/outputs/logs/smoke_itvlaGp_p2_010k.log
```

#### 7.2 验证冒烟测试结果

```bash
echo "=== 冒烟测试验证 ==="

# 检查退出码
echo "Exit code: $?"

# 检查日志中是否有错误
grep -i -E "error|exception|traceback|his_kpts|left_entity" ${REPO_ROOT}/outputs/logs/smoke_itvlaGp_p2_010k.log

# 检查 GPU 使用情况
nvidia-smi
```

**预期结果**：

| 检查项 | 预期 |
|--------|------|
| 退出码 | 0 |
| 视频文件 | 2 个 .mp4（success_1.mp4 或 failure_1.mp4 + success_2.mp4 或 failure_2.mp4） |
| 错误日志 | 无 Error/Exception/Traceback |
| GPU 显存 | 模型加载约 12-15 GB（bfloat16，无 WAN） |
| 运行时间 | 2-5 分钟 |

#### 7.3 冒烟测试排错

如果冒烟测试失败，按以下顺序排查：

1. **`AttributeError: 'NoneType' object has no attribute 'is_left_gripper_open'`**：第 1.1 节的 check_success 修复未生效。重新按第 1.5 节步骤 1 执行修改，并运行第 2.1 节 v4 静态测试验证
2. **段错误 / SAPIEN 崩溃**：检查 EGL 渲染（`python -c "import sapien; sapien.Scene()"`），确认无头服务器环境支持 GPU 渲染
3. **CuRobo sm_120 错误**：如果进程卡住（CPU 高、GPU 0%），检查日志文件中是否有 "no kernel image" 错误。参见 [11.2 CuRobo sm_120 问题](#112-curobo-sm_120-缺失)
4. **模型加载失败**：检查 config.json 中的 `type` 字段是否为 `internvla_a1_5`，以及 `enable_keypoint_predictor` 是否为 `true`
5. **transformers 版本不匹配**：检查 `python -c "import transformers; print(transformers.__version__)"` 是否为 5.2.0

---

### 8. 正式评估执行

> 冒烟测试通过后，执行正式的 100 episode 评估。双 GPU 可并行运行 demo_clean 和 demo_randomized。
>
> **后台运行建议**：在 Cursor Shell 或某些 IDE 终端中，裸 `nohup ... &` 会随 shell 退出被 SIGHUP 终止（见 [`2LOG2.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md) Problem #1）。推荐：
> - **tmux/screen** 会话中前台运行；或
> - SSH 直连终端中使用 `nohup ... & disown`；或
> - Cursor Shell 的持久化后台模式（`block_until_ms=0`）
>
> 以下示例使用 `nohup`（仅适用于 SSH 直连且 shell 不会提前退出的场景）。

#### 8.1 demo_clean 评估（GPU 0）

```bash
# 终端 1（建议在 tmux 内）
conda activate itvlaGp
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

OUT=${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_010k
mkdir -p ${REPO_ROOT}/outputs/logs

cd ${REPO_ROOT}/third_party/RoboTwin

nohup python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir "${OUT}/robotwin/demo_clean/stack_bowls_three" \
  --task-config demo_clean \
  --task-idx 46 \
  --action-mode abs \
  --infer-horizon 20 \
  --inference-backend standard \
  --num-episodes 100 \
  --dtype bfloat16 \
  > ${REPO_ROOT}/outputs/logs/eval_itvlaGp_p2_010k_demo_clean.log 2>&1 &

echo "demo_clean PID: $!"
```

#### 8.2 demo_randomized 评估（GPU 1）

```bash
# 终端 2（建议在 tmux 内）
conda activate itvlaGp
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp
export CKPT_RUN=2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=1

OUT=${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_010k
mkdir -p ${REPO_ROOT}/outputs/logs

cd ${REPO_ROOT}/third_party/RoboTwin

nohup python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir "${OUT}/robotwin/demo_randomized/stack_bowls_three" \
  --task-config demo_randomized \
  --task-idx 46 \
  --action-mode abs \
  --infer-horizon 20 \
  --inference-backend standard \
  --num-episodes 100 \
  --dtype bfloat16 \
  > ${REPO_ROOT}/outputs/logs/eval_itvlaGp_p2_010k_demo_randomized.log 2>&1 &

echo "demo_randomized PID: $!"
```

#### 8.3 进度监控

```bash
# 终端 3：实时监控进度
watch -n 60 '
echo "===== $(date) ====="
echo ""
echo "--- demo_clean ---"
S=$(ls /home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three/success_*.mp4 2>/dev/null | wc -l)
F=$(ls /home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three/failure_*.mp4 2>/dev/null | wc -l)
T=$((S+F))
echo "  $T/100 episodes done, $S successes, $F failures"
if [ $T -gt 0 ]; then echo "  Success rate: $(echo "scale=1; $S*100/$T" | bc)%"; fi
echo ""
echo "--- demo_randomized ---"
S=$(ls /home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_randomized/stack_bowls_three/success_*.mp4 2>/dev/null | wc -l)
F=$(ls /home/luogang/SRC/Robot/itvlaGp/outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_randomized/stack_bowls_three/failure_*.mp4 2>/dev/null | wc -l)
T=$((S+F))
echo "  $T/100 episodes done, $S successes, $F failures"
if [ $T -gt 0 ]; then echo "  Success rate: $(echo "scale=1; $S*100/$T" | bc)%"; fi
echo ""
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
echo ""
echo "--- Disk ---"
df -h / | tail -1
'
```

也可以 tail 日志文件查看详细进度：

```bash
tail -f /home/luogang/SRC/Robot/itvlaGp/outputs/logs/eval_itvlaGp_p2_010k_demo_clean.log
```

#### 8.4 时间估算

| 配置 | 预估时间 | 说明 |
|------|---------|------|
| demo_clean | ~150 分钟（2.5h） | seed 验证通过率高，扫描种子数少 |
| demo_randomized | ~170 分钟（2.8h） | 域随机化导致更多 seed 失败，需扫描更多种子 |
| **双 GPU 并行总时间** | **~170 分钟（2.8h）** | 受限于较慢的 demo_randomized |

> **注意**：itvlaGp 的 3-path 推理（多了 29 个 kpt token 的 K/V 缓存构建）理论上比基线版稍慢，但关键点缓存仅每 env step 计算一次，且 kpt suffix 只有 29 tokens（远小于 prefix 的数百 tokens），因此延迟增加极小（<5%）。

#### 8.5 中断恢复

**重要**：`inference.py` 的 `main()` 函数在启动时会 `shutil.rmtree(args.video_dir)`（L517–518），**删除指定输出目录下的所有内容**。

因此，如果评估中断（如机器重启、OOM、手动 kill），**不能直接重启**——已完成的 episode 视频会被删除。

**应对方案**：

1. **中断前备份**：
   ```bash
   cp -r outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three \
         outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three_backup
   ```

2. **从中断处继续**：无直接支持。可以修改 `inference.py` 注释掉 `shutil.rmtree` 行（L518），并手动设置 `--seed` 和 `--num-episodes` 参数来跳过已完成的 episode。但这需要小心处理 seed 序列的一致性。

3. **推荐做法**：在 **tmux/screen** 中运行评估进程，确保 shell 退出不影响任务。SSH 直连时可用 `nohup ... & disown`。评估前/中用 `df -h /` 监控磁盘（见 §10.6）。如需中断，先备份输出目录。

---

### 9. 结果汇总与对比分析

#### 9.1 使用结果汇总脚本

```bash
cd /home/luogang/SRC/Robot/itvlaGp
python util_scripts/robotwin_result_stats.py outputs/robotwin/itvlaGp_p2_010k
```

输出文件：`outputs/robotwin/itvlaGp_p2_010k/results_robotwin.csv`

#### 9.2 手动计数（备用）

```bash
echo "=== itvlaGp Phase2 Step 10000 Evaluation Results ==="
echo ""
echo "--- demo_clean ---"
S_C=$(ls outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three/success_*.mp4 2>/dev/null | wc -l)
F_C=$(ls outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three/failure_*.mp4 2>/dev/null | wc -l)
T_C=$((S_C+F_C))
echo "  Successes: ${S_C}"
echo "  Failures:  ${F_C}"
echo "  Total:     ${T_C}"
echo "  Rate:      $(echo "scale=1; ${S_C}*100/${T_C}" | bc)%"
echo ""
echo "--- demo_randomized ---"
S_R=$(ls outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_randomized/stack_bowls_three/success_*.mp4 2>/dev/null | wc -l)
F_R=$(ls outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_randomized/stack_bowls_three/failure_*.mp4 2>/dev/null | wc -l)
T_R=$((S_R+F_R))
echo "  Successes: ${S_R}"
echo "  Failures:  ${F_R}"
echo "  Total:     ${T_R}"
echo "  Rate:      $(echo "scale=1; ${S_R}*100/${T_R}" | bc)%"
```

#### 9.3 与基线版对比

| 模型 | Checkpoint 来源 | 训练步数 | demo_clean | demo_randomized | 备注 |
|------|----------------|---------|------------|-----------------|------|
| InternVLA-A1.5 base (rbtwn2/10000) | InternVLA-A-series | 10000 | 71.0% (71/100) | 54.0% (54/100) | 2-path，无关键点 |
| GR00T 0.8B | — | — | 62.0% | 57.0% | 基线参考 |
| itvlaGp (LOG1, 旧 p2 路径) | `itvlaGp/outputs/.../p2/010000` | 10000 | 64.0% | 16.0% | 见 [`2LOG.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md) |
| itvlaGp (LOG2 rerun) | 同上 | 10000 | 63.0% | 22.0% | 见 [`2LOG2.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md) |
| itvlaGp (LOG0807192_10k) | `itvlaGp080719_2/outputs/.../010000` | 10000 | 58.0% | 8.0% | 见 [`2LOG0807192_10k.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192_10k.md) |
| itvlaGp (LOG0807192) | 080719_2 run | 015000 | 65.0% | 11.0% | 015k 在 demo_clean 上最佳 |
| itvlaGp (LOG0807192_20k) | 080719_2 run | 020000 | 57.0% | 11.0% | — |

> **注意**：不同 LOG 使用的 checkpoint 路径/训练 run 不同，成功率不宜直接横向对比绝对数值，但可观察域随机化（demo_randomized）相对 demo_clean 的降幅模式。

**分析维度**：

1. **成功率提升**：关键点融合是否提高了 stack_bowls_three 的成功率？
2. **demo_clean vs demo_randomized**：关键点融合在域随机化条件下是否更鲁棒？（关键点是 3D 几何特征，理论上对视觉干扰更不敏感）
3. **失败模式分析**：检查 failure 视频，关注：
   - 是否抓取位置更准确（关键点提供了精确的 3D 位姿信息）
   - 是否堆叠对齐更好
   - 是否出现新的失败模式（如关键点噪声导致的抖动）

#### 9.4 评估其他 checkpoint 步数

080719_2 训练 run 有每 1000 步的 checkpoint（001000 到 010000）：

```bash
ls ${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/
# 001000  002000  003000  004000  005000  006000  007000  008000  009000  010000
```

如需评估其他步数（例如 step 6000）：

```bash
export CKPT=${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/006000/pretrained_model
OUT=${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_006k

# 先验证 checkpoint 存在
ls ${CKPT}/config.json ${CKPT}/model.safetensors ${CKPT}/stats.json

# 然后用与 8.1/8.2 相同的 inference.py 命令，替换 CKPT 和 OUT 路径
```

---

### 10. 关键约束与注意事项

#### 10.1 必须使用 standard 推理后端

| 参数 | 要求 | 原因 |
|------|------|------|
| `--inference-backend` | `standard` | `InternVLAA15Optimized` 的 `sample_actions()` 方法（`modeling_internvla_a1_5_optimized.py:422`）不接受 `his_kpts`/`his_len` 参数，仅支持 2-path `[prefix, action]` 前向传播。使用 optimized 会 **静默忽略** 关键点数据。 |

> **如何验证在推理时确实使用了 3-path？** 检查日志中是否有 keypoint 相关的 CUDA 活动。或在 `inference.py` 的 `use_kpt` 分支添加临时打印：
> ```python
> if use_kpt and his_len == 1:
>     print(f"[DEBUG] 3-path inference active: J={J}, H={H}, his_len={his_len}")
> ```

#### 10.2 必须使用 abs 动作模式

Checkpoint 的 `train_config.json` 确认 `action_mode: "abs"`。使用 `delta` 模式会产生无意义的动作输出。

#### 10.3 task_idx 必须为 46

`stack_bowls_three` 在 `TASK_NAMES` 列表（`inference.py:80-131`）中的索引为 **46**。`eval.sh` 的默认值是 44（`stack_blocks_three`），不要使用默认值。

#### 10.4 eval.sh 的使用方式（修改后）

修改二已将 `eval.sh` 的 `CONDA_ENV` 改为 `${CONDA_ENV:-itvlaGp}`，修改三已添加 `--resize-size` 传参。**但 eval.sh 仍未传递 `--dtype`**，直接用它运行会得到 `float32` 而非 `bfloat16`。

**不推荐**直接依赖 eval.sh；若必须使用，请确认已添加 `--dtype bfloat16`：

```bash
bash evaluation/RoboTwin/eval.sh \
  "${CKPT}" \
  outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three \
  demo_clean \
  46
```

由于 eval.sh 会 `cd third_party/RoboTwin` 后使用相对路径，符号链接解析可能导致路径错误（参见 [11.7](#117-符号链接路径解析问题)），**仍推荐直接调用 `inference.py`**（§7/§8）。

#### 10.5 video-dir 会被自动删除

`inference.py` 的 `main()` 函数（L517–518）：

```python
if args.video_dir.exists():
    shutil.rmtree(args.video_dir)
```

**始终为不同的评估使用不同的 video-dir 路径**。`demo_clean` 和 `demo_randomized` 使用不同的子目录。

#### 10.6 磁盘空间

评估前和评估中应检查磁盘：`df -h /`。每 100 episode 约产生 100–200 MB 视频 + 日志（总计通常 <1 GB），但长时间运行或 pip cache 积累可能导致磁盘满（LOG2 曾 100% 满导致日志写入失败）。

**建议**：
- 评估前确认至少 **10 GB** 可用空间
- 评估期间在 §8.3 进度监控中包含 `df -h /`
- 空间不足时：`pip cache purge`（可释放数十 GB）

---

### 11. 故障排除手册

#### 11.1 flash-attn ABI 不匹配

**症状**：`ImportError: ... undefined symbol: _ZN3c104cuda...`

**根因**：使用了与当前 torch 版本不匹配的 flash-attn 缓存 wheel。

**修复**：
```bash
pip uninstall flash-attn -y
pip cache purge
MAX_JOBS=16 pip install flash-attn --no-build-isolation --no-cache-dir
```

#### 11.2 CuRobo sm_120 缺失

**症状**：评估卡住（CPU 高占用、GPU 0% 利用率），日志无输出或显示 `CUDA error: no kernel image is available for execution on the device`。

**根因**：CuRobo 的 CUDA kernel 未编译 sm_120（Blackwell 架构）。这通常表现为 seed 验证阶段的 expert rollout 永远无法完成。

**诊断**：
```bash
# 运行时 kinematics 验证（比 cuobjdump 更可靠）
python -c "from curobo.curobolib import kinematics; print('kinematics OK')"

# 如果进程卡住超过 5 分钟，检查日志
grep -i "kernel\|No module named" outputs/logs/smoke_itvlaGp_p2_010k.log
```

**修复**：
```bash
cd /home/luogang/share/zwy/Projects/RoboTwin/envs/curobo
rm -f src/curobo/curobolib/*.so
rm -rf build
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 pip install -e . --no-build-isolation --no-cache-dir --force-reinstall --no-deps
pip install setuptools_scm warp-lang yourdfpy
```

> 注意：这会修改共享目录中的 .so 文件，影响所有使用该 CuRobo 源码的环境。

#### 11.2a CuRobo 运行时依赖缺失

**症状**：`ModuleNotFoundError: No module named 'setuptools_scm'` / `'warp'` / `'yourdfpy'`

**根因**：CuRobo 以 `--no-deps` 安装，跳过了上述依赖。

**修复**：`pip install setuptools_scm warp-lang yourdfpy`（见 §3.8）

#### 11.3 check_success 排序 bug

**症状**：`AttributeError: 'NoneType' object has no attribute 'is_left_gripper_open'` 或关键点提取时 `robot_entity` 为 None。

**根因**：`close_env()` 将 `self.robot` 设为 None（`_base_task.py` 第 648 行），如果 `check_success()` 在 `close_env()` 之后调用则报错。`check_success()` 内部调用 `self.is_left_gripper_open()` / `self.is_right_gripper_open()`，这些方法访问 `self.robot`（`_base_task.py` 第 768-772 行）。

**状态**：已在第 1.1 节修复。修复方式：在 `maybe_close_env()` 之前捕获结果到 `expert_success` 变量，随后在 try/except 块之外检查该变量。验证修复：

```bash
grep -n "expert_success\|check_success\|close_env" evaluation/RoboTwin/inference.py
# 预期输出（行号可能略有偏差）：
#   expert_success = bool(task_env.plan_success and task_env.check_success())
#   maybe_close_env(task_env)    # 在 expert_success 赋值之后
#   if not expert_success:
```

#### 11.4 transformers 版本漂移

**症状**：`TypeError: create_causal_mask() got an unexpected keyword argument 'cache_position'`

**根因**：安装其他包时 transformers 被升级到 5.2.0 以上版本。

**修复**：
```bash
pip install "transformers==5.2.0"
# 重新打补丁
TRANSFORMERS_DIR=$(python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")
cp -r src/lerobot/policies/internvla_a1_5/transformers_replace/models/* ${TRANSFORMERS_DIR}/models/
```

**预防**：在安装其他包时使用 `--no-deps` 避免自动升级 transformers。

#### 11.5 gymnasium 缺失

**症状**：`ModuleNotFoundError: No module named 'gymnasium'`

**修复**：`pip install gymnasium`

#### 11.6 ffmpeg libx264 不匹配

**症状**：`ffmpeg: error while loading shared libraries: libx264.so.138`

**修复**：`conda install -c conda-forge "ffmpeg>=7" -y`

#### 11.7 符号链接路径解析问题

**症状**：`python: can't open file '../../evaluation/RoboTwin/inference.py'`

**根因**：`cd third_party/RoboTwin` 后，shell 可能解析符号链接为物理路径 `/home/luogang/share/zwy/Projects/RoboTwin/`，导致相对路径 `../../` 不再指向 `itvlaGp` 仓库根目录。

**修复**：使用绝对路径调用 `inference.py`（如本手册所有命令所示）：
```bash
python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py ...
```

#### 11.8 僵尸进程

**症状**：`nvidia-smi` 显示异常的 GPU 显存占用，或前一次评估的进程仍在运行。

**诊断**：
```bash
ps aux | grep inference.py | grep -v grep
```

**修复**：
```bash
kill <PID>
# 或强制终止
kill -9 <PID>
```

**预防**：在启动新评估前，始终检查是否有遗留进程。

#### 11.8a nohup / Cursor Shell 进程被 SIGHUP 终止

**症状**：启动后 PID 很快消失；日志仅数行，无 mp4 产出。

**根因**：Cursor Shell 或 IDE 终端退出时，裸 `nohup ... &` 子进程收到 SIGHUP 被终止（LOG2 Problem #1）。

**修复**：改用 tmux/screen 会话，或 SSH 直连终端 + `nohup ... & disown`，或 Cursor 持久化后台模式。

#### 11.8b 磁盘空间不足

**症状**：`write failed: no space left on device`；根分区 100% 满。

**根因**：pip cache、评估日志、mp4 视频持续写入。

**修复**：`pip cache purge`（可释放数十 GB）；清理旧评估输出；评估前/中监控 `df -h /`（见 §10.6）。

#### 11.8c 禁止 `| tail` 管道吞掉 seed 验证错误

**症状**：评估看似在运行但 seed 验证 silently 失败，难以定位根因。

**根因**：`| tail -N` 会丢弃 expert rollout 阶段的早期错误输出。

**修复**：使用 `tee` 将完整日志写入文件（如 §7/§8 示例），再用 `grep`/`tail -f` 查看日志文件，而非管道截断 stdout。

#### 11.9 seed 验证异常缓慢

**症状**：进度正常但非常慢（特别是 demo_randomized）。

**原因**：这是正常行为。RoboTwin 对每个 seed 先运行 expert policy 验证成功后才交给被测 policy。在 `demo_randomized` 模式下，域随机化导致更多 seed 的 expert rollout 失败或不稳定，需要扫描更多 seed 才能找到 100 个有效 seed。参考基线评估中：demo_clean 扫描 182 个 seed 找到 100 个有效的，demo_randomized 扫描 207 个。

**诊断**：看日志中 `current seed` 的值增长速度。如果每隔几分钟完成一个 episode 但 seed 号跳跃较大，这是正常的。

#### 11.10 模型加载 OOM

**症状**：`torch.cuda.OutOfMemoryError`

**不太可能发生**：模型仅 5.9 GB（bfloat16），RTX PRO 6000 有 96 GB VRAM。但如果同一 GPU 上有其他进程占用显存：

**修复**：
```bash
# 检查 GPU 显存使用
nvidia-smi

# 确保 CUDA_VISIBLE_DEVICES 设置正确，且无其他进程占用
kill <占用GPU的PID>
```

#### 11.12 Checkpoint 权重文件 `.gstmp` 未重命名

**症状**：预检 [12] FAIL；`model.safetensors` 不存在但有 `model.safetensors_.gstmp`。

**修复**：确认下载/保存完成后重命名：`mv model.safetensors_.gstmp model.safetensors`

---

### 12. 附录

#### 12.1 Checkpoint 目录结构

```
${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model/
├── config.json          # 模型配置（3606 bytes）
│   ├── type: "internvla_a1_5"
│   ├── enable_keypoint_predictor: true
│   ├── num_keypoint_joints: 14
│   ├── action_loss_only: false（训练配置；inference.py L270 加载时强制 True）
│   ├── inference_backend: "standard"
│   ├── knowledge_insulation: true
│   ├── knowledge_insulation_kpt: true
│   └── ...
├── model.safetensors    # 模型权重（5.9 GB）
├── stats.json           # 标准化统计量（11269 bytes）
│   └── aloha:
│       ├── observation.state: {mean, std, min, max}  # dim=14
│       └── action: {mean, std, min, max}             # dim=14
└── train_config.json    # 训练配置（12795 bytes）
    ├── dataset: "robotwin/stack_bowls_three_kpt"
    ├── batch_size: 16
    ├── steps: 10000
    └── resume: true（从 step 2000 恢复）
```

#### 12.2 训练历史简述

itvlaGp 的训练分为两个阶段：

**Phase 1（关键点专家预热）**：400 步，`train_expert_only=true`，冻结 action expert（`action_expert_lr_scale=0.1`），集中训练 keypoint expert + TrackEncoder。kpt_cur loss 从 0.54 降至 0.001（99.8% 下降）。

**Phase 2（联合训练）**：10000 步（从 Phase 1 step 300 的 checkpoint 继续），解冻 action expert（`action_expert_lr_scale=1.0`），联合训练 action + keypoint。中途 step 3000 因磁盘满中断，从 step 2000 恢复继续训练到 10000。action loss 从 0.091 降至 0.003（96.7% 下降），kpt_cur 保持在 0.0006-0.0011 范围。

Phase 2 LOG 推荐 step 5000 checkpoint 为最优。本次评估使用 step 10000 checkpoint。

#### 12.3 TASK_NAMES 索引参考

```
 0: adjust_bottle          25: place_can_basket
 1: beat_block_hammer      26: place_cans_plasticbox
 2: blocks_ranking_rgb     27: place_container_plate
 3: blocks_ranking_size    28: place_dual_shoes
 4: click_alarmclock       29: place_empty_cup
 5: click_bell             30: place_fan
 6: dump_bin_bigbin        31: place_mouse_pad
 7: grab_roller            32: place_object_basket
 8: handover_block         33: place_object_scale
 9: handover_mic           34: place_object_stand
10: hanging_mug            35: place_phone_stand
11: lift_pot               36: place_shoe
12: move_can_pot           37: press_stapler
13: move_pillbottle_pad    38: put_bottles_dustbin
14: move_playingcard_away  39: put_object_cabinet
15: move_stapler_pad       40: rotate_qrcode
16: open_laptop ⚠️         41: scan_object
17: open_microwave         42: shake_bottle
18: pick_diverse_bottles   43: shake_bottle_horizontally
19: pick_dual_bottles      44: stack_blocks_three
20: place_a2b_left         45: stack_blocks_two
21: place_a2b_right        46: stack_bowls_three ← 本次目标
22: place_bread_basket     47: stack_bowls_two
23: place_bread_skillet    48: stamp_seal
24: place_burger_fries     49: turn_switch
```

> ⚠️ `open_laptop`（index 16）存在已知的 `arm_tag` bug，评估全 benchmark 时应跳过。

#### 12.4 代码修改汇总（V2 新增）

| # | 文件 | 严重程度 | 修改内容 | 对应章节 |
|---|------|---------|---------|---------|
| 1 | `evaluation/RoboTwin/inference.py` | 🔴 CRITICAL | `check_success()` 移至 `close_env()` 之前 | [1.1](#11-修改一inferencepy-check_success-排序-bug-修复critical) |
| 2 | `evaluation/RoboTwin/eval.sh` | 🟡 MEDIUM | `CONDA_ENV` 改为可覆盖默认值 | [1.2](#12-修改二evalsh-conda_env-修复medium) |
| 3 | `evaluation/RoboTwin/eval.sh` | 🟢 LOW | 添加 `--resize-size` 传参 | [1.3](#13-修改三evalsh-resize_size-传参修复low) |
| 4 | `evaluation/RoboTwin/requirements.txt` | 🟢 LOW | 添加 `scipy` | [1.4](#14-修改四requirementstxt-添加-scipylow) |

#### 12.5 已验证无需修改的组件

| 组件 | 文件 | 验证结论 |
|------|------|---------|
| 3-path MoT 推理 | `modeling_internvla_a1_5.py` | `sample_actions()` 完整支持 `his_kpts`/`his_len`，3-path forward 正确 |
| TrackEncoder | `keypoints.py` | 正确处理可变长度历史（包括 `his_len=0/1` 边界情况） |
| 模型配置 | `configuration_internvla_a1_5.py` | 28 个关键点相关配置字段完整 |
| Policy 工厂 | `factory.py` | `get_policy_class("internvla_a1_5")` 正确解析 |
| 权重加载 | `configs/policies.py` | `from_pretrained()` 正确处理所有字段 |
| 数据变换 | `transform_internvla_a1_5.py` | Chat processor 正确 |
| 关键点提取 | `inference.py:get_keypoints_aloha()` | `task_env.robot.left_entity` 访问模式正确 |
| ALOHA 关键点链接 | `inference.py:ALOHA_KEYPOINT_LINKS` | 14 个链接名称在 URDF 中全部存在 |
| Checkpoint 配置 | `config.json` | `enable_keypoint_predictor=true`, `num_keypoint_joints=14`；`action_loss_only` 在 config 中可为 false |

#### 12.6 关键文件路径速查

| 文件 | 路径 | 作用 |
|------|------|------|
| 评估入口脚本 | `evaluation/RoboTwin/eval.sh` | Shell 包装（缺 `--dtype`；推荐直接调用 inference.py） |
| 推理脚本 | `evaluation/RoboTwin/inference.py` | 完整评估逻辑，含 GeoPredict 关键点支持 |
| 模型定义 | `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | 3-path MoT 模型 |
| 关键点模块 | `src/lerobot/policies/internvla_a1_5/keypoints.py` | TrackEncoder（移植自 GeoPredict） |
| 优化推理后端 | `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5_optimized.py` | **不支持关键点**，不可用 |
| 模型配置 | `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` | 含 28 个关键点相关配置字段 |
| 数据变换 | `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` | Extract3DKeypointTransformFn 等 |
| Checkpoint | `${REPO_ROOT}/outputs-ext/internvla_a1_5/${CKPT_RUN}/checkpoints/010000/pretrained_model/` | 待评估权重 |
| 结果统计工具 | `util_scripts/robotwin_result_stats.py` | CSV 结果汇总 |
| RoboTwin 平台 | `third_party/RoboTwin/`（符号链接） | 50 个任务环境 |
| RoboTwin 评估依赖 | `evaluation/RoboTwin/requirements.txt` | SAPIEN、mplib、scipy 等 |

#### 12.7 快速命令参考

```bash
# 激活环境
source /home/luogang/SRC/Robot/itvlaGp/activate_itvlaGp.sh

# 运行静态代码修改测试 v4（完整脚本见 §2.1）

# 2-episode 冒烟测试
cd ${REPO_ROOT}/third_party/RoboTwin && \
python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir ${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_010k/smoke/demo_clean/stack_bowls_three \
  --task-config demo_clean --task-idx 46 --action-mode abs --infer-horizon 20 \
  --inference-backend standard --num-episodes 2 --dtype bfloat16

# 100-episode 正式评估（demo_clean, GPU 0）
CUDA_VISIBLE_DEVICES=0 python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir ${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three \
  --task-config demo_clean --task-idx 46 --action-mode abs --infer-horizon 20 \
  --inference-backend standard --num-episodes 100 --dtype bfloat16

# 100-episode 正式评估（demo_randomized, GPU 1）
CUDA_VISIBLE_DEVICES=1 python -u ${REPO_ROOT}/evaluation/RoboTwin/inference.py \
  --ckpt-path "${CKPT}" \
  --video-dir ${REPO_ROOT}/outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_randomized/stack_bowls_three \
  --task-config demo_randomized --task-idx 46 --action-mode abs --infer-horizon 20 \
  --inference-backend standard --num-episodes 100 --dtype bfloat16

# 结果汇总
python util_scripts/robotwin_result_stats.py outputs/robotwin/itvlaGp_p2_010k

# 手动计数
ls outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three/success_*.mp4 | wc -l
ls outputs/robotwin/itvlaGp_p2_010k/robotwin/demo_clean/stack_bowls_three/failure_*.mp4 | wc -l
```

---

## Part B：执行记录

> **说明**：历次完整执行日志见文首链接的 LOG 系列文件。本节摘要已完成的评估；**新 run 可在下方空白行追加**。

### 时间线 / 操作日志

| 时间 | 操作 | 结果 | 详见 |
|------|------|------|------|
| 2026-08-05 | 环境搭建 + V2 四项代码修复 + 静态测试 16/16 + 冒烟 2/2 | PASS | [`2LOG.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md) |
| 2026-08-07 | 010k 重跑正式评估（demo_clean + demo_randomized 各 100 ep） | 63%/22% | [`2LOG2.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md) |
| 2026-08-09 | 080719_2 run 010k 评估 | 58%/8% | [`2LOG0807192_10k.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192_10k.md) |
| | （新 run 记录于此） | | |

### 问题记录（报错 → 根因 → 修复 → 验证）

| # | 问题描述 | 根因 | 修复方案 | 详见 |
|---|---------|------|---------|------|
| 1 | transformers 5.14.1 与 Qwen3.5 补丁不兼容 | pip 安装最新版 | 锁定 5.2.0 + 重打补丁 | [`2LOG.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md) #1 |
| 2 | 静态测试 regex 误报 FAIL | 匹配到错误的 try 块 | 改用 v4（play_once + 跳过注释行） | [`2LOG.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md) #2 |
| 3–7 | CuRobo import/CUDA 失败 | 缺依赖 + 无 sm_120 | 安装 setuptools_scm/warp-lang/yourdfpy + 重编译 | [`2LOG.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md) #3–7 |
| 8 | nohup 后台进程被 SIGHUP 终止 | Cursor Shell 退出 | 持久化后台 shell / tmux | [`2LOG2.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md) #1 |
| 9 | 磁盘 100% 满 | pip cache + 评估日志/mp4 | `pip cache purge` | [`2LOG2.md`](reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md) #2 |

### 最终结果（历史摘要）

| 模型 | Checkpoint | demo_clean | demo_randomized | LOG |
|------|-----------|------------|-----------------|-----|
| InternVLA-A1.5 base | step 10000 | 71.0% | 54.0% | 基线手册 |
| itvlaGp (旧 p2 路径) | 010k | 64.0% / 63.0% | 16.0% / 22.0% | LOG1 / LOG2 |
| itvlaGp (080719_2) | 010k | 58.0% | 8.0% | LOG0807192_10k |
| itvlaGp (080719_2) | 015k | 65.0% | 11.0% | LOG0807192 |
| itvlaGp (080719_2) | 020k | 57.0% | 11.0% | LOG0807192_20k |

**结论**：V2 四项代码修复 + 15 项预检 + 16 项静态测试流程已验证可行。itvlaGp 在 demo_clean 上与 base 仍有差距；demo_randomized 泛化是主要瓶颈。详细分析见 §9.3 与各 LOG 文件。
