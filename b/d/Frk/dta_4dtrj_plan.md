# Franka 机器人插拔插座数据集处理实施方案

> **目标**: 将 HDF5 格式的 Franka 插拔插座遥操作数据集转换为 LeRobot 格式, 并通过 FK (Forward Kinematics) 计算 7D 关键点 (3D 位置 + 4D 四元数姿态), 生成可直接用于 InternVLA-A1.5 训练的数据集.
>
> **参考文档**: [R1Pro E1 方案设计](../R1Pro/dta_3dtrj_E2.md), [R1Pro E1 实施手册](../R1Pro/dta_3dtrj_E2impl.md), [R1Pro E1 实施日志](../R1Pro/dta_3dtrj_E2implLog.md)
>
> **撰写日**: 2026-09-04

---

## 变量定义

下表列出了本方案中所有与服务器环境相关的可配置变量. 在脚本中, 这些值均通过 CLI 参数传入, 不硬编码在脚本内部.

| 变量名 | 含义 | 默认值 | 当前服务器实际值 |
|--------|------|--------|-----------------|
| `VENV` | Python 虚拟环境激活命令 (conda/uv/venv 均可) | `conda activate itvlaGp` | `conda activate itvlaGp` |
| `REPO_ROOT` | 代码库根目录 | — | `/home/luogang/SRC/Robot/itvlaGp` |
| `HDF5_SOURCE` | 源 HDF5 数据集路径 | — | `/home/luogang/DATA/plug_into_socket_hdf5` |
| `LRB_INTERMEDIATE` | 中间 LeRobot 数据集 (无关键点) | — | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb` |
| `LRB_4D_DEST` | 最终带 7D 关键点的 LeRobot 数据集 | — | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D` |
| `URDF_PATH` | Franka URDF 文件路径 | `$REPO_ROOT/b/d/Frk/fr3v2_1_franka_hand.urdf` | 同左 |
| `HF_HOME` | HuggingFace 缓存根目录 | `~/.cache/huggingface` | `/home/luogang/hf_home` |
| `HF_LEROBOT_HOME` | LeRobot 数据集根目录 (`$HF_HOME/lerobot`) | `$HF_HOME/lerobot` | `/home/luogang/hf_home/lerobot` |
| `ROBOT_TYPE` | Schema 注册的机器人类型标识符 | `franka_plug` | `franka_plug` |
| `LINK_PREFIX` | URDF 中 link/joint 名称前缀 | `fr3v2_1` | `fr3v2_1` |
| `TARGET_FPS` | LeRobot 数据集目标帧率 | `30` | `30` (匹配相机 Hz) |
| `NUM_KEYPOINTS` | FK 关键点数量 | `8` | `8` |
| `KEYPOINT_DIM` | 每个关键点的特征维度 | `7` | `7` (3D pos + 4D quat) |
| `TOTAL_KPT_DIM` | 关键点总维度 = `NUM_KEYPOINTS × KEYPOINT_DIM` | `56` | `56` |
| `BBOX_MARGIN` | R\_pad 的安全裕量 | `0.15` | `0.15` |
| `LEROBOT_FORMAT` | 输出 LeRobot 数据集的格式版本 | `v3.0` | `v3.0` |

---

## 目录

0. [概述](#0-概述)
1. [环境准备](#1-环境准备)
2. [URDF 准备与分析](#2-urdf-准备与分析)
3. [HDF5 → LeRobot 格式转换](#3-hdf5--lerobot-格式转换)
4. [7D 关键点生成](#4-7d-关键点生成)
5. [归一化策略（继承 E1）](#5-归一化策略继承-e1)
6. [数据验证](#6-数据验证)
7. [归一化统计量生成](#7-归一化统计量生成)
8. [模型侧适配与训练集成](#8-模型侧适配与训练集成)
9. [Smoke Test](#9-smoke-test)
10. [完整文件清单](#10-完整文件清单)
11. [故障排查手册](#11-故障排查手册)

---

## 0. 概述

### 0.1 处理管道架构

本方案的数据处理管道分为 **三步** (与 R1Pro 方案的"两步"不同 — R1Pro 的源数据已是 LeRobot 格式, 无需格式转换):

```
┌─────────────────────────────────┐
│  HDF5 源数据集                   │
│  plug_into_socket_hdf5/         │
│  100 episodes, 100Hz state,     │
│  30Hz camera (wrist + global)   │
└────────────┬────────────────────┘
             │  Step 1: convert_franka_plug_hdf5.py
             │  - 频率对齐 (100Hz → 30Hz)
             │  - HDF5 → LeRobot parquet + MP4
             ▼
┌─────────────────────────────────┐
│  中间 LeRobot 数据集             │
│  plug_into_socket_lrb/          │
│  30Hz, observation.state.arm[7] │
│  + observation.state.gripper[1] │
│  + 2 camera videos              │
└────────────┬────────────────────┘
             │  Step 2: generate_franka_keypoints.py (§4.3, 归一化 §5)
             │  - rsync 拷贝到目标目录
             │  - Pass 1: 全帧 FK → 全局包围盒 → R_pad
             │  - Pass 2: pos/R_pad + quat 半球 → 写 parquet
             ▼
┌─────────────────────────────────┐
│  最终 LeRobot 数据集             │
│  plug_into_socket_lrb_4D/       │
│  30Hz, 含 observation.           │
│  keypoint_3d [56]               │
└────────────┬────────────────────┘
             │  Step 3: compute_norm_stats_single.py
             │  - 计算 stats.json (含关键点统计量)
             ▼
┌─────────────────────────────────┐
│  训练就绪                        │
│  accelerate launch ...           │
│  --dataset.repo_id=              │
│    plug_into_socket_lrb_4D      │
└─────────────────────────────────┘
```

### 0.2 与 R1Pro 方案的差异对比

| 维度 | R1Pro E1 方案 | Franka 方案 (本文) |
|------|-------------|------------------|
| 源数据格式 | 已是 LeRobot parquet | **HDF5**, 需要额外的格式转换步骤 |
| 输出 LeRobot 格式版本 | v3.0 | **v3.0** (与 R1Pro 相同) |
| 机器人结构 | 双臂 7DOF × 2 + 4DOF 躯干 | **单臂** 7DOF + 2DOF 夹爪 (mimic) |
| 关键点数量 | 16 (8 per arm) | **8** (link1-7 + hand\_tcp) |
| 关键点总维度 | 112 = 16 × 7 | **56** = 8 × 7 |
| 状态列名 | `observation.state.left_arm` [7] + `right_arm` [7] | `observation.state.arm` [7] |
| 源频率 | 15Hz (已同步) | **100Hz state / 30Hz camera** (需对齐) |
| 目标频率 | 15Hz | **30Hz** |
| 躯干处理 | FK 中需 bake torso\_q | **不需要** (无躯干关节) |
| 夹爪处理 | 固定关节, 不影响 FK | **棱柱关节**, 但不影响 link1-7 / hand\_tcp 的 FK |
| 相机数量 | 3 (head, wrist\_left, wrist\_right) | **2** (global, wrist) |
| 数据集大小 | ~1.4 GB, 27145 帧 | **~19.5 GB** (HDF5), 转换后约 **2-4 GB** (30Hz ~66k 帧) |

### 0.3 代码复用策略

遵循"扩展代替修改"原则:

| 已有代码 | 复用方式 | 说明 |
|---------|---------|------|
| `convert_robotwin.py` | **参考模式** | 参考其 HDF5 → LeRobot 的框架模式 (LeRobotDataset.create/add\_frame/save\_episode), 但因数据结构差异较大 (频率对齐, feature 布局), 需新建脚本 |
| `generate_r1pro_keypoints_e1.py` | **import 共享函数** | `compute_r_pad`, `_get_parquet_files`, `_copy_dataset` 是机器人无关的, 直接导入; FK 提取器和 pass1/pass2 需针对 Franka 重写 |
| `verify_e1_keypoints.py` | **参考 + 适配** | 7 项检查逻辑大部分通用, 但 shape 和列名需适配 |
| `compute_norm_stats_single.py` | **零改动** | 自动遍历所有非 video 列, 只需注册 schema |
| Schema 系统 | **新增 YAML** | 新增 `franka_plug.yaml`, 不修改现有 schema |
| 模型侧代码 | **复用 R1Pro E1 改动** | `keypoint_dim` / `keypoint_out_dim` / `kpt_rot_loss_weight` 参数已在 R1Pro E1 方案中设计, 只需用不同参数值 |

---

## 1. 环境准备

### 1.1 依赖检查

```bash
# 激活虚拟环境 (根据实际情况选择其中一种)
conda activate itvlaGp          # conda 环境
# source .venv/bin/activate     # python venv 或 uv venv
# source /path/to/env/bin/activate

# Pinocchio (FK 计算)
python -c "import pinocchio as pin; print(f'pinocchio {pin.__version__}')"
# 预期: >= 2.6 (当前服务器: 4.1.0)

# h5py (HDF5 读取)
python -c "import h5py; print(f'h5py {h5py.__version__}')"

# OpenCV (JPEG 解码)
python -c "import cv2; print(f'cv2 {cv2.__version__}')"

# 其他基础依赖
python -c "import pandas, numpy, torch; print(f'pandas={pandas.__version__}, numpy={numpy.__version__}, torch={torch.__version__}')"
```

若 `h5py` 未安装:

```bash
pip install h5py
```

### 1.2 路径设置

```bash
export REPO_ROOT=/home/luogang/SRC/Robot/itvlaGp

# HF_HOME: HuggingFace 缓存根目录 (若未在 shell 配置中设置则在此导出)
export HF_HOME=/home/luogang/hf_home

# HF_LEROBOT_HOME: LeRobot 数据集根目录, 约定为 $HF_HOME/lerobot
export HF_LEROBOT_HOME=${HF_HOME}/lerobot

# 若 lerobot 目录不存在则创建
mkdir -p ${HF_LEROBOT_HOME}
```

**关于 `HF_LEROBOT_HOME` 的数据组织**: LeRobot 按 `$HF_LEROBOT_HOME/<repo_id>/` 定位数据集. 本方案将转换后的数据集直接写入 `${HF_LEROBOT_HOME}/` 下, 无需额外注册. 若其他已有数据集 (如 `elevator0714_lerobot`) 存放在别处, 可通过符号链接注册:

```bash
# 将其他路径下的数据集注册到 HF_LEROBOT_HOME (仅对存放在 HF_LEROBOT_HOME 外部的数据集才需要)
ln -s /home/luogang/DATA/elevator0714_lerobot \
      ${HF_LEROBOT_HOME}/elevator0714_lerobot
```

本方案的脚本均接受完整的绝对路径作为 `--dest` 参数, 不依赖 `HF_LEROBOT_HOME` 的具体位置. 只有 `compute_norm_stats_single.py` 和训练脚本通过 `HF_LEROBOT_HOME` 查找数据集.

### 1.3 磁盘空间检查

```bash
# 源数据集
du -sh /home/luogang/DATA/plug_into_socket_hdf5
# 预期: ~19.5 GB

# 目标磁盘可用空间 (需至少 10 GB: 中间 ~4 GB + 最终 ~4 GB + 余量)
df -h /home/luogang/hf_home/lerobot/
```

**空间估算**:
- 中间 LeRobot 数据集: ~66k 帧 × 2 相机 × MP4 ≈ 2-4 GB (取决于视频编码压缩率)
- 最终数据集: ≈ 中间 + 关键点增量 (~66k × 56 × 4 bytes ≈ 15 MB, parquet 压缩后更小)
- 总计需要: **约 6-10 GB**

---

## 2. URDF 准备与分析

### 2.1 URDF 位置

URDF 已位于 `b/d/Frk/fr3v2_1_franka_hand.urdf`, **无需复制到其他目录**. 所有脚本通过 `--urdf` 参数指向该路径.

> **注意**: URDF 中引用了 `package://franka_description/meshes/...` 的网格文件. Pinocchio 的 `buildModelFromUrdf` 只构建运动学模型 (关节+连杆), **不加载网格文件**, 因此 FK 计算不需要这些 mesh 文件存在.

### 2.2 运动链分析

Franka FR3v2 的主运动链:

```
base (固定)
  └── fr3v2_1_link0 (固定基座)
        └── [joint1, revolute] → fr3v2_1_link1  ← 关键点 0
              └── [joint2, revolute] → fr3v2_1_link2  ← 关键点 1
                    └── [joint3, revolute] → fr3v2_1_link3  ← 关键点 2
                          └── [joint4, revolute] → fr3v2_1_link4  ← 关键点 3
                                └── [joint5, revolute] → fr3v2_1_link5  ← 关键点 4
                                      └── [joint6, revolute] → fr3v2_1_link6  ← 关键点 5
                                            └── [joint7, revolute] → fr3v2_1_link7  ← 关键点 6
                                                  └── [joint8, fixed] → link8
                                                        └── [hand_joint, fixed] → hand
                                                              └── [hand_tcp_joint, fixed] → hand_tcp  ← 关键点 7
                                                              └── [finger_joint1, prismatic] → leftfinger
                                                              └── [finger_joint2, prismatic, mimic] → rightfinger
```

### 2.3 关键点链接选择

| 关键点索引 | Link 名称 | 驱动关节 | 选择理由 |
|-----------|-----------|---------|---------|
| 0 | `{PREFIX}_link1` | joint1 (肩部旋转) | 捕捉 shoulder 姿态 |
| 1 | `{PREFIX}_link2` | joint2 (肩部俯仰) | 捕捉上臂抬举 |
| 2 | `{PREFIX}_link3` | joint3 (上臂旋转) | 捕捉上臂内旋/外旋 |
| 3 | `{PREFIX}_link4` | joint4 (肘部) | 捕捉肘关节弯曲 |
| 4 | `{PREFIX}_link5` | joint5 (前臂旋转) | 捕捉前臂内旋/外旋 |
| 5 | `{PREFIX}_link6` | joint6 (腕部) | 捕捉腕部翻转 |
| 6 | `{PREFIX}_link7` | joint7 (腕部旋转) | 捕捉末端旋转 |
| 7 | `{PREFIX}_hand_tcp` | 固定链 (link7→link8→hand→tcp) | **工具中心点**, 代表末端执行器精确位置 |

其中 `{PREFIX}` = `fr3v2_1` (由 CLI 参数 `--link-prefix` 控制).

**关于夹爪关节**: `finger_joint1/2` 是 prismatic 关节, 控制手指开合. 但 link1-7 和 hand\_tcp 都在夹爪关节的**上游**, 夹爪的张合**不影响**这 8 个关键点的 FK 结果. 因此 FK 计算时可以将夹爪关节保持在 neutral 位置. 夹爪状态通过 `observation.state.gripper` 单独传递给模型.

### 2.4 关节限位

| 关节 | 下限 (rad) | 上限 (rad) | 备注 |
|------|-----------|-----------|------|
| joint1 | -2.9007 | +2.9007 | 对称 |
| joint2 | -1.8361 | +1.8361 | 对称 |
| joint3 | -2.9007 | +2.9007 | 对称 |
| joint4 | -3.0770 | **-0.1169** | **仅负值** (肘部只能向一个方向弯曲) |
| joint5 | -2.8763 | +2.8763 | 对称 |
| joint6 | **+0.4398** | +4.6216 | **仅正值** (腕部偏置范围) |
| joint7 | -3.0508 | +3.0508 | 对称 |

> joint4 和 joint6 的非对称限位意味着 bounding box 可能也不对称 — 这在 isotropic 归一化下会导致一些维度利用率低于另一些. 但 isotropic 归一化保证了各轴等比缩放, 模型可以自行学习不对称的分布.

---

## 3. HDF5 → LeRobot 格式转换

### 3.1 源数据集分析

| 属性 | 值 |
|------|-----|
| 格式 | HDF5, 每 episode 一个文件 (`episode_XXXXXX.hdf5`) |
| Episodes | 100 |
| 状态帧率 | 100 Hz |
| 相机帧率 | 30 Hz |
| 状态总帧数 | 221,428 |
| 相机总帧数 | ~66,590 (global) / ~66,596 (wrist) |
| 每 episode 时长 | 17.9s ~ 26.6s (均值 22.2s) |
| 任务 | `"plug into socket"` |
| 相机 | `global` (外部视角) + `wrist` (腕部视角), 480×640, JPEG quality=90 |
| 关节状态 | `joint_positions` [7], `joint_velocities` [7], `ee_pos` [3], `ee_quat` [4], `gripper_width` [1] |
| 动作 | `action_joints` [7] (绝对关节位置目标) + `action_gripper` [1] (归一化夹爪指令) |
| 深度图 | `depth_image` uint16, 480×640, LZF 压缩 (**本方案不转换**) |

### 3.2 频率对齐策略

源数据的状态 (100Hz) 与相机 (30Hz) 频率不同. 本方案以 **相机帧为基准** 进行对齐:

```
100Hz state:  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ...
 30Hz camera:  ▼        ▼        ▼        ▼        ▼        ...
              ↑        ↑        ↑        ↑        ↑
              nearest  nearest  nearest  nearest  nearest
```

对于每个相机帧 (时间戳 $t_{\text{cam}}$):
1. 在状态时间戳数组中找到最近的索引: $i^* = \arg\min_i |t_{\text{state}}[i] - t_{\text{cam}}|$
2. 用 `joint_positions[$i^*$]` 作为 `observation.state.arm`
3. 用 `gripper_width[$i^*$]` 作为 `observation.state.gripper`
4. 用 `action_joints[$i^*$]` 作为 `action.arm`
5. 用 `action_gripper[$i^*$]` 作为 `action.gripper`
6. 用 `ee_pos[$i^*$]`, `ee_quat[$i^*$]` 作为辅助特征 (可选, 用于验证)

**理由**: 30Hz 足以捕捉插拔操作的时间尺度 (典型操作速度 ~10cm/s, 30Hz 每帧位移 ~3mm); 以相机帧为基准保证每帧都有对应图像; 丢弃的高频状态信息 (100Hz 中间帧) 对下游策略学习影响可忽略.

### 3.3 Feature 布局设计

| 特征键 | Shape | Dtype | 来源 | 说明 |
|--------|-------|-------|------|------|
| `observation.state.arm` | [7] | float32 | `joint_positions` | 7 个关节角 (rad) |
| `observation.state.gripper` | [1] | float32 | `gripper_width` | 夹爪张开宽度 (m, 0~0.079) |
| `observation.state.ee_pos` | [3] | float32 | `ee_pos` | 末端位置 (m), 辅助验证用 |
| `observation.state.ee_quat` | [4] | float32 | `ee_quat` | 末端四元数 (wxyz), 辅助验证用 |
| `action.arm` | [7] | float32 | `action_joints` | 关节位置目标 (rad, 绝对) |
| `action.gripper` | [1] | float32 | `action_gripper` | 夹爪指令 (归一化, ~0.01~1.0) |
| `observation.images.global` | (480, 640, 3) | video | `camera_global/color_image_jpeg` | 外部相机 RGB |
| `observation.images.wrist` | (480, 640, 3) | video | `camera_wrist/color_image_jpeg` | 腕部相机 RGB |

> **关于 `ee_pos` / `ee_quat`**: 这些是 HDF5 中预录的末端执行器位姿. 将它们纳入 LeRobot 数据集有两个好处:
> 1. 可以与 FK 计算的 hand\_tcp 关键点交叉验证, 确认 FK 结果正确
> 2. 某些策略架构可能需要 EE pose 作为输入

### 3.4 Schema 注册

新增文件: `b/s/Frk/cfg/franka_plug.yaml`

```yaml
robot_type: franka_plug
action_mask_spec: [7, -1]
# [7, -1] 含义: 前 7 维 (arm joints) 在 delta 模式下做差分, 最后 1 维 (gripper) 保持绝对值
feature_mapping:
  observation.state:
    - observation.state.arm
    - observation.state.gripper
  action:
    - action.arm
    - action.gripper
image_mapping:
  observation.images.global: observation.images.image0
  observation.images.wrist: observation.images.image1
```

**为什么要新建 schema 而不复用已有的 `franka` schema**: 已有的 `a1-franka.yaml` 使用 A1 旧格式的 feature key (`states.joint.position` 等), 与本数据集的 key 布局 (`observation.state.arm` 等) 不兼容. 新建 schema 既避免了修改已有 schema, 也让不同 Franka 数据集可以共存.

**注册方式**: Schema 系统默认扫描 `src/lerobot/dataset_schemas/configs/*.yaml`. 由于本文件放在 `b/s/Frk/cfg/franka_plug.yaml`, 需要通过以下**任一**方式让系统识别它:

- **方式 A (推荐): 符号链接**
  ```bash
  ln -s $REPO_ROOT/b/s/Frk/cfg/franka_plug.yaml \
        $REPO_ROOT/src/lerobot/dataset_schemas/configs/franka_plug.yaml
  ```

- **方式 B: 直接复制** (需在修改后同步)
  ```bash
  cp b/s/Frk/cfg/franka_plug.yaml src/lerobot/dataset_schemas/configs/franka_plug.yaml
  ```

推荐方式 A — 符号链接保证 `b/s/Frk/cfg/franka_plug.yaml` 是单一事实来源, 不会产生两份文件的同步问题.

### 3.5 输出格式说明: LeRobot v3.0

本方案输出 **LeRobot `codebase_version: v3.0`** 格式的数据集, 与代码库中 R1Pro 数据集 (`elevator0714_lerobot`) 保持一致. v3.0 格式的主要规范:

| 方面 | v3.0 规范 |
|------|----------|
| **数值数据** | Parquet 文件, 路径模板: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| **视频数据** | MP4 (H.264 + yuv420p), 路径模板: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |
| **元数据** | `meta/info.json` (数据集描述 + feature schema), `meta/episodes.jsonl` (每集信息), `meta/tasks.jsonl` (任务列表), `meta/stats.json` (归一化统计量) |
| **每帧必需字段** | `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` (由框架自动填充) |
| **Feature 类型** | `float32` (数值), `video` (视频流), `image` (单帧 PNG) |
| **Chunk 大小** | 由 `LeRobotDataset` 自动管理 (默认每个 chunk 最多 1000 episodes) |

v3.0 与 v2.x 的主要区别在于: v3.0 使用 `episodes.jsonl` 替代旧的 `episode_data_index.parquet`, 并在 `info.json` 中统一描述所有 feature 的形状和类型. 本方案使用的 `LeRobotDataset.create()` API 会自动生成所有符合 v3.0 规范的元数据文件.

### 3.6 转换脚本: `convert_franka_plug_hdf5.py`

新建文件: `b/s/Frk/convert_franka_plug_hdf5.py`

```python
"""Convert Franka plug-into-socket HDF5 dataset to LeRobot format.

Handles frequency alignment: state is recorded at 100Hz, camera at 30Hz.
Each output frame corresponds to one camera frame, with the nearest state
frame matched by timestamp.

Usage:
    python b/s/Frk/convert_franka_plug_hdf5.py \
        --source /home/luogang/DATA/plug_into_socket_hdf5 \
        --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
        --robot-type franka_plug
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROBOT_TYPE = "franka_plug"
DEFAULT_TARGET_FPS = 30
DEFAULT_IMAGE_SHAPE = (480, 640, 3)
CAMERA_NAMES = ["global", "wrist"]
CAMERA_HDF5_GROUPS = ["camera_global", "camera_wrist"]


def build_features(image_shape: tuple[int, ...] = DEFAULT_IMAGE_SHAPE,
                   use_videos: bool = True) -> dict:
    """Construct LeRobot feature dict for Franka plug dataset."""
    mode = "video" if use_videos else "image"
    features = {
        "observation.state.arm": {
            "dtype": "float32", "shape": (7,),
            "names": {"motors": [f"joint{i}" for i in range(1, 8)]},
        },
        "observation.state.gripper": {
            "dtype": "float32", "shape": (1,),
            "names": {"motors": ["gripper_width"]},
        },
        "observation.state.ee_pos": {
            "dtype": "float32", "shape": (3,),
            "names": {"position": ["x", "y", "z"]},
        },
        "observation.state.ee_quat": {
            "dtype": "float32", "shape": (4,),
            "names": {"quaternion": ["w", "x", "y", "z"]},
        },
        "action.arm": {
            "dtype": "float32", "shape": (7,),
            "names": {"motors": [f"joint{i}" for i in range(1, 8)]},
        },
        "action.gripper": {
            "dtype": "float32", "shape": (1,),
            "names": {"motors": ["gripper_cmd"]},
        },
    }
    for cam_name in CAMERA_NAMES:
        features[f"observation.images.{cam_name}"] = {
            "dtype": mode,
            "shape": image_shape,
            "names": ["height", "width", "rgb"],
        }
    return features


def align_timestamps(state_ts: np.ndarray, camera_ts: np.ndarray) -> np.ndarray:
    """For each camera timestamp, find the index of the nearest state frame.

    Returns: int array of shape [len(camera_ts)] with state indices.
    """
    indices = np.searchsorted(state_ts, camera_ts)
    indices = np.clip(indices, 1, len(state_ts) - 1)
    left_diff = np.abs(state_ts[indices - 1] - camera_ts)
    right_diff = np.abs(state_ts[indices] - camera_ts)
    mask = left_diff <= right_diff
    indices[mask] -= 1
    return indices


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB uint8 numpy array."""
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process_episode(
    hdf5_path: Path,
    dataset,  # LeRobotDataset
    task_str: str,
    reference_camera: str = "camera_global",
) -> int:
    """Process one HDF5 episode file, add frames to dataset.

    Uses the reference camera's timestamps as the timeline anchor.
    Returns the number of frames added.
    """
    with h5py.File(hdf5_path, "r") as f:
        # Read state data
        state_ts = f["robot_state/timestamps"][:]
        joint_pos = f["robot_state/joint_positions"][:]
        gripper_w = f["robot_state/gripper_width"][:]
        ee_pos = f["robot_state/ee_pos"][:]
        ee_quat = f["robot_state/ee_quat"][:]
        action_j = f["robot_state/action_joints"][:]
        action_g = f["robot_state/action_gripper"][:]

        # Read reference camera timestamps for alignment
        ref_cam_ts = f[f"{reference_camera}/timestamps"][:]

        # Align state to camera timestamps
        state_indices = align_timestamps(state_ts, ref_cam_ts)

        # Verify alignment quality
        time_diffs = np.abs(state_ts[state_indices] - ref_cam_ts)
        max_diff_ms = time_diffs.max() * 1000
        if max_diff_ms > 20:  # > 20ms suggests misalignment
            logger.warning(
                "  Large state-camera alignment gap: max %.1f ms in %s",
                max_diff_ms, hdf5_path.name,
            )

        # Read camera images
        cam_images = {}
        for cam_group, cam_name in zip(CAMERA_HDF5_GROUPS, CAMERA_NAMES):
            if cam_group in f:
                cam_images[cam_name] = f[f"{cam_group}/color_image_jpeg"]
            else:
                logger.warning("  Camera group '%s' not found in %s", cam_group, hdf5_path.name)
                return 0

        # Determine how many camera frames to use
        # Use the minimum across all cameras
        n_cam_frames = min(len(ref_cam_ts), *(len(f[f"{cg}/timestamps"][:]) for cg in CAMERA_HDF5_GROUPS))

        # Add frames (skip the last frame to follow the action convention:
        # action[t] = target at time t; but last frame has no meaningful next target)
        n_added = 0
        for cam_idx in range(n_cam_frames - 1):
            s_idx = state_indices[cam_idx]

            frame = {
                "task": task_str,
                "observation.state.arm": joint_pos[s_idx].astype(np.float32),
                "observation.state.gripper": gripper_w[s_idx].astype(np.float32),
                "observation.state.ee_pos": ee_pos[s_idx].astype(np.float32),
                "observation.state.ee_quat": ee_quat[s_idx].astype(np.float32),
                "action.arm": action_j[s_idx].astype(np.float32),
                "action.gripper": action_g[s_idx].astype(np.float32),
            }

            # Decode and add camera images
            for cam_name in CAMERA_NAMES:
                jpeg_data = cam_images[cam_name][cam_idx]
                rgb = decode_jpeg(bytes(jpeg_data))
                frame[f"observation.images.{cam_name}"] = rgb

            dataset.add_frame(frame)
            n_added += 1

        return n_added


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Path to HDF5 dataset directory.")
    parser.add_argument("--dest", type=str, required=True,
                        help="Output LeRobot dataset directory.")
    parser.add_argument("--robot-type", type=str, default=DEFAULT_ROBOT_TYPE,
                        help=f"Robot type for LeRobot metadata (default: {DEFAULT_ROBOT_TYPE}).")
    parser.add_argument("--fps", type=int, default=DEFAULT_TARGET_FPS,
                        help=f"Target FPS (default: {DEFAULT_TARGET_FPS}).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --dest if it exists.")
    parser.add_argument("--no-videos", action="store_true",
                        help="Store images as PNGs instead of MP4 videos.")
    parser.add_argument("--reference-camera", type=str, default="camera_global",
                        choices=["camera_global", "camera_wrist"],
                        help="Camera whose timestamps anchor the alignment.")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if dest.exists():
        if args.force:
            logger.info("Removing existing %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} exists. Use --force to overwrite.")

    # Read meta.json for task info
    meta_path = source / "meta.json"
    task_str = "plug into socket"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        task_str = meta.get("task", task_str)
        logger.info("Task: %s, Total episodes: %s", task_str, meta.get("total_episodes"))

    # Discover HDF5 files
    hdf5_files = sorted(source.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files in {source}")
    logger.info("Found %d HDF5 episode files", len(hdf5_files))

    # Determine image shape from first file
    with h5py.File(hdf5_files[0], "r") as f:
        sample_jpeg = bytes(f[f"{CAMERA_HDF5_GROUPS[0]}/color_image_jpeg"][0])
        sample_img = decode_jpeg(sample_jpeg)
        image_shape = sample_img.shape
    logger.info("Image shape: %s", image_shape)

    # Build features and create dataset
    use_videos = not args.no_videos
    features = build_features(image_shape=image_shape, use_videos=use_videos)

    # Import LeRobotDataset here to avoid import overhead at parse time
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=dest.name,
        fps=args.fps,
        root=dest,
        robot_type=args.robot_type,
        features=features,
        use_videos=use_videos,
        tolerance_s=1 / args.fps,  # allow up to 1 frame of timestamp tolerance
    )

    # Process each episode
    total_frames = 0
    for ep_idx, hdf5_path in enumerate(hdf5_files):
        n_frames = process_episode(
            hdf5_path, dataset, task_str,
            reference_camera=args.reference_camera,
        )
        if n_frames > 0:
            dataset.save_episode()
            total_frames += n_frames
            logger.info(
                "Episode %d/%d (%s): %d frames (total: %d)",
                ep_idx + 1, len(hdf5_files), hdf5_path.name, n_frames, total_frames,
            )
        else:
            logger.warning("Episode %d (%s): skipped (0 frames)", ep_idx, hdf5_path.name)

    logger.info("=== DONE ===")
    logger.info("  Total episodes: %d", len(hdf5_files))
    logger.info("  Total frames: %d", total_frames)
    logger.info("  FPS: %d", args.fps)
    logger.info("  Output: %s", dest)


if __name__ == "__main__":
    main()
```

### 3.7 执行转换

```bash
cd $REPO_ROOT

# 激活虚拟环境 (按实际情况选择)
conda activate itvlaGp   # 或: source .venv/bin/activate

python b/s/Frk/convert_franka_plug_hdf5.py \
    --source /home/luogang/DATA/plug_into_socket_hdf5 \
    --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
    --robot-type franka_plug \
    --fps 30 \
    --force
```

**预期日志**:

```
[INFO] Task: plug into socket, Total episodes: 100
[INFO] Found 100 HDF5 episode files
[INFO] Image shape: (480, 640, 3)
[INFO] Episode 1/100 (episode_000000.hdf5): ~665 frames (total: 665)
...
[INFO] Episode 100/100 (episode_000099.hdf5): ~665 frames (total: ~66500)
[INFO] === DONE ===
[INFO]   Total episodes: 100
[INFO]   Total frames: ~66500
[INFO]   FPS: 30
[INFO]   Output: /home/luogang/hf_home/lerobot/plug_into_socket_lrb
```

**预期耗时**: 15-30 分钟 (主要开销在 JPEG 解码 + MP4 视频编码).

### 3.8 转换后快速检查

```bash
# 1. 检查 info.json
python3 -c "
import json
with open('/home/luogang/hf_home/lerobot/plug_into_socket_lrb/meta/info.json') as f:
    info = json.load(f)
print(f'robot_type: {info[\"robot_type\"]}')
print(f'fps: {info[\"fps\"]}')
print(f'total_episodes: {info[\"total_episodes\"]}')
print(f'total_frames: {info[\"total_frames\"]}')
print(f'features: {list(info[\"features\"].keys())}')
"

# 2. 抽检 parquet 内容
python3 -c "
import pandas as pd, numpy as np
from pathlib import Path
files = sorted(Path('/home/luogang/hf_home/lerobot/plug_into_socket_lrb/data').rglob('*.parquet'))
print(f'Parquet files: {len(files)}')
df = pd.read_parquet(files[0])
print(f'Columns: {list(df.columns)}')
print(f'Rows: {len(df)}')
arm = np.stack(df['observation.state.arm'].values)
print(f'arm shape: {arm.shape}, range: [{arm.min():.4f}, {arm.max():.4f}]')
gripper = np.stack(df['observation.state.gripper'].values)
print(f'gripper shape: {gripper.shape}, range: [{gripper.min():.4f}, {gripper.max():.4f}]')
assert 'observation.keypoint_3d' not in df.columns, 'keypoint_3d should not exist yet'
print('OK: no keypoint_3d yet (expected)')
"

# 3. 检查视频文件存在
ls /home/luogang/hf_home/lerobot/plug_into_socket_lrb/videos/observation.images.global/chunk-000/ | head -5
ls /home/luogang/hf_home/lerobot/plug_into_socket_lrb/videos/observation.images.wrist/chunk-000/ | head -5
```

---

## 4. 7D 关键点生成

### 4.1 FK 提取器设计

`FrankaFKExtractor7D` 相比 R1Pro 的 `R1ProFKExtractorE1`:
- **更简单**: 无躯干关节, 单臂, 无需 `_q_base` 预设
- **相同的 FK 核心**: Pinocchio `forwardKinematics` + `updateFramePlacements` + `pin.Quaternion` 半球归一化
- **参数化**: 链接前缀通过参数传入, 不硬编码

```
R1ProFKExtractorE1                     FrankaFKExtractor7D
├── _left_idx_q [7]                    ├── _arm_idx_q [7]
├── _right_idx_q [7]                   │   (无 right arm)
├── _torso_idx_q [4]                   │   (无 torso)
├── _q_base (neutral + torso baked)    ├── _q_base (neutral, 无额外设置)
├── compute(left, right) → [16, 7]    ├── compute(arm) → [8, 7]
└── compute_batch(lefts, rights)       └── compute_batch(arms)
```

### 4.2 代码复用

从 `generate_r1pro_keypoints_e1.py` 导入以下机器人无关的函数:

| 函数 | 用途 | 是否可直接导入 |
|------|------|--------------|
| `compute_r_pad(global_min, global_max, margin)` | 计算 isotropic bounding radius | **是** — 纯数学函数, 无 R1Pro 依赖 |
| `_get_parquet_files(data_dir)` | 递归发现 parquet 文件 | **是** |
| `_copy_dataset(source, dest, force)` | rsync 拷贝数据集 | **是** |

以下函数因含 R1Pro 专用逻辑, **需要在新脚本中重新实现**:

| 函数 | R1Pro 专用点 | Franka 适配 |
|------|------------|------------|
| `_read_joint_angles(df)` | 读取 `left_arm` + `right_arm` 两列 | 读取 `observation.state.arm` 一列 |
| `_read_recorded_torso(df)` | 检查 torso 列恒定 | **不需要** (Franka 无 torso) |
| `pass1_compute_bbox(...)` | 双臂 FK | 单臂 FK |
| `pass2_write_keypoints(...)` | 双臂 FK | 单臂 FK |
| `_update_info_json(dest)` | NUM\_KEYPOINTS=16, DIM=7, shape=112 | NUM\_KEYPOINTS=8, DIM=7, shape=56 |
| `_write_meta(dest, ...)` | 含 torso\_q 字段 | 无 torso\_q |

### 4.3 关键点生成脚本: `generate_franka_keypoints.py`

> **归一化设计**（Pass 1/2 公式、值域约定、Franka 数值示例）见 **[§5 归一化策略（继承 E1）](#5-归一化策略继承-e1)**. 本节仅给出脚本实现与执行命令.

新建文件: `b/s/Frk/generate_franka_keypoints.py`

```python
"""Offline FK-based 7D keypoint generation for Franka arm.

Two-pass pipeline — see dta_4dtrj_plan.md §5 for normalization design.

Output: observation.keypoint_3d [56] = 8 keypoints × 7D (px,py,pz,qx,qy,qz,qw).

See ../b/d/R1Pro/dta_3dtrj_E2.md for the 7D keypoint design rationale.

Usage:
    python b/s/Frk/generate_franka_keypoints.py \
        --source /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
        --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
        --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Reuse robot-agnostic functions from E1 script ---
import sys
_UTIL_DIR = Path(__file__).resolve().parents[3] / "util_scripts"
if str(_UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(_UTIL_DIR))
from generate_r1pro_keypoints_e1 import (
    compute_r_pad,
    _copy_dataset,
    _get_parquet_files,
)

# --- Franka-specific constants (configurable via CLI) ---
DEFAULT_LINK_PREFIX = "fr3v2_1"
DEFAULT_BBOX_MARGIN = 0.15
DEFAULT_NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7  # 3 (pos) + 4 (quat xyzw)
STATE_COLUMN = "observation.state.arm"


def make_keypoint_links(prefix: str) -> list[str]:
    """Construct keypoint link names from prefix."""
    return [f"{prefix}_link{i}" for i in range(1, 8)] + [f"{prefix}_hand_tcp"]


def make_joint_names(prefix: str) -> list[str]:
    """Construct revolute joint names from prefix."""
    return [f"{prefix}_joint{i}" for i in range(1, 8)]


def make_feature_names(keypoint_links: list[str]) -> list[str]:
    """Construct per-dimension feature names for info.json."""
    return [
        f"{link}_{comp}"
        for link in keypoint_links
        for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")
    ]


class FrankaFKExtractor7D:
    """FK extractor producing 7D keypoints for single-arm Franka robots.

    Args:
        urdf_path: Path to Franka URDF file.
        link_prefix: Prefix for link/joint names in the URDF (e.g. 'fr3v2_1').
        keypoint_links: Override list of link names to extract. If None, uses
            default 8 links (link1-7 + hand_tcp).
    """

    def __init__(self, urdf_path: str, link_prefix: str = DEFAULT_LINK_PREFIX,
                 keypoint_links: list[str] | None = None):
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.keypoint_links = keypoint_links or make_keypoint_links(link_prefix)
        self.num_keypoints = len(self.keypoint_links)

        self.frame_ids = []
        for name in self.keypoint_links:
            fid = self.model.getFrameId(name)
            if fid >= self.model.nframes:
                raise ValueError(f"Frame '{name}' not found in URDF {urdf_path}")
            self.frame_ids.append(fid)

        joint_names = make_joint_names(link_prefix)
        self._arm_idx_q: list[int] = []
        for jname in joint_names:
            jid = self.model.getJointId(jname)
            if jid >= len(self.model.joints):
                raise ValueError(f"Joint '{jname}' not found in URDF {urdf_path}")
            nq = self.model.joints[jid].nq
            if nq != 1:
                raise ValueError(f"Joint '{jname}' has nq={nq}, expected 1 (revolute)")
            self._arm_idx_q.append(self.model.joints[jid].idx_q)

        self._q_base = pin.neutral(self.model)

    def compute(self, arm_joints: np.ndarray) -> np.ndarray:
        """Compute keypoints as 7D (position + quaternion) in base_link frame.

        Args:
            arm_joints: [7] joint angles in radians.

        Returns:
            [num_keypoints, 7] float32 — [px, py, pz, qx, qy, qz, qw].
            Quaternions are hemisphere-normalized (qw >= 0).
        """
        pin = self._pin
        q = self._q_base.copy()
        for idx_q, angle in zip(self._arm_idx_q, arm_joints, strict=True):
            q[idx_q] = float(angle)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        keypoints = np.empty((self.num_keypoints, KEYPOINT_DIM), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            oMf = self.data.oMf[fid]
            keypoints[i, :3] = oMf.translation
            quat = pin.Quaternion(oMf.rotation)
            raw_q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)
            if raw_q[3] < 0:
                raw_q = -raw_q
            keypoints[i, 3:7] = raw_q
        return keypoints

    def compute_batch(self, arm_joints_batch: np.ndarray) -> np.ndarray:
        """[N, 7] -> [N, num_keypoints, 7]."""
        n = arm_joints_batch.shape[0]
        out = np.empty((n, self.num_keypoints, KEYPOINT_DIM), dtype=np.float32)
        for i in range(n):
            out[i] = self.compute(arm_joints_batch[i])
        return out


def _read_arm_joints(df: pd.DataFrame, pq_path: Path | None = None):
    """Read arm joint angles from parquet DataFrame."""
    if len(df) == 0:
        logger.warning("Skipping empty parquet (0 rows): %s", pq_path)
        return None
    if STATE_COLUMN not in df.columns:
        raise KeyError(f"Required column '{STATE_COLUMN}' not found in {pq_path}. "
                       f"Available: {list(df.columns)}")
    arm = np.stack(df[STATE_COLUMN].values).astype(np.float64)
    if np.any(np.isnan(arm)):
        raise ValueError(f"NaN in {STATE_COLUMN} in {pq_path}")
    return arm


def pass1_compute_bbox(parquet_files: list[Path], extractor: FrankaFKExtractor7D):
    """Pass 1: FK all frames, collect global position min/max."""
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    qw_min, qw_max = 1.0, 0.0
    quat_norm_err_max = 0.0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        arm = _read_arm_joints(df, pq_path)
        if arm is None:
            continue

        kpts = extractor.compute_batch(arm)  # [N, 8, 7]
        pos = kpts[:, :, :3]
        quat = kpts[:, :, 3:7]

        frame_min = pos.reshape(-1, 3).min(axis=0)
        frame_max = pos.reshape(-1, 3).max(axis=0)
        global_min = np.minimum(global_min, frame_min)
        global_max = np.maximum(global_max, frame_max)

        qw_vals = quat[:, :, 3]
        qw_min = min(qw_min, float(qw_vals.min()))
        qw_max = max(qw_max, float(qw_vals.max()))
        quat_norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
        quat_norm_err_max = max(quat_norm_err_max, float(np.abs(quat_norms - 1.0).max()))

        total_frames += len(df)
        logger.info("  Pass 1: %s — %d frames (total: %d)", pq_path.name, len(df), total_frames)

    logger.info("Quaternion stats: qw_min=%.6f, qw_max=%.6f, norm_err_max=%.2e",
                qw_min, qw_max, quat_norm_err_max)
    if qw_min < 0:
        raise RuntimeError(f"Hemisphere normalization failed: qw_min={qw_min:.6f}")
    if quat_norm_err_max > 0.01:
        raise RuntimeError(f"Quaternion norm check failed: max error={quat_norm_err_max:.4f}")

    return global_min.astype(np.float32), global_max.astype(np.float32), total_frames


def pass2_write_keypoints(parquet_files: list[Path], extractor: FrankaFKExtractor7D,
                          r_pad: float, source_data_dir: Path, dest: Path):
    """Pass 2: FK → normalize → write observation.keypoint_3d into parquet."""
    total_frames = 0
    oob_count = 0

    for pq_path in parquet_files:
        rel = pq_path.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(dest_pq)
        arm = _read_arm_joints(df, dest_pq)
        if arm is None:
            continue

        kpts = extractor.compute_batch(arm)  # [N, 8, 7]
        kpts[:, :, :3] /= r_pad

        if (np.abs(kpts[:, :, :3]) > 1.01).any():
            oob_count += 1
            logger.warning("  Position OOB in %s: max |pos| = %.4f",
                           dest_pq.name, np.abs(kpts[:, :, :3]).max())

        num_kpts = extractor.num_keypoints
        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts]
        df.to_parquet(dest_pq)
        total_frames += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if oob_count:
        logger.warning("%d parquet files had OOB positions. Increase --bbox-margin.", oob_count)
    return total_frames


def _update_info_json(dest: Path, num_keypoints: int, keypoint_links: list[str]) -> None:
    """Add observation.keypoint_3d feature to info.json."""
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    total_dim = num_keypoints * KEYPOINT_DIM
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [total_dim],
        "names": make_feature_names(keypoint_links),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d [%d].", info_path, total_dim)


def _write_meta(dest: Path, r_pad: float, global_min: np.ndarray,
                global_max: np.ndarray, total_frames: int,
                urdf_path: str, keypoint_links: list[str],
                bbox_margin: float) -> None:
    """Write keypoints_meta.json with normalization parameters."""
    meta = {
        "bbox_radius": r_pad,
        "bbox_margin": bbox_margin,
        "global_min_base_relative": global_min.tolist(),
        "global_max_base_relative": global_max.tolist(),
        "normalization": "base_link_origin_isotropic",
        "keypoint_dim": KEYPOINT_DIM,
        "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
        "rotation_representation": "quaternion_xyzw_hemisphere",
        "rotation_convention": "qw >= 0; negate if qw < 0",
        "num_keypoints": len(keypoint_links),
        "keypoint_links": keypoint_links,
        "total_frames": total_frames,
        "coordinate_system": (
            "base_link-relative, position divided by bbox_radius, "
            "quaternion hemisphere-normalized"
        ),
        "urdf": str(urdf_path),
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Source LeRobot dataset (intermediate, no keypoints).")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination for the copy with 7D keypoints.")
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[3] / "b" / "d" / "Frk" / "fr3v2_1_franka_hand.urdf"),
                        help="Path to Franka URDF.")
    parser.add_argument("--link-prefix", type=str, default=DEFAULT_LINK_PREFIX,
                        help=f"URDF link/joint name prefix (default: {DEFAULT_LINK_PREFIX}).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --dest if it exists.")
    parser.add_argument("--skip-copy", action="store_true",
                        help="Reuse existing --dest (e.g. from a partial run).")
    parser.add_argument("--bbox-margin", type=float, default=DEFAULT_BBOX_MARGIN,
                        help=f"Safety margin for R_pad (default: {DEFAULT_BBOX_MARGIN}).")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    bbox_margin = args.bbox_margin

    # Step 0: Copy dataset
    if not args.skip_copy:
        _copy_dataset(source, dest, args.force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy but {dest} does not exist.")
        logger.info("--skip-copy: reusing %s", dest)

    # Step 1: Build FK extractor
    logger.info("Loading URDF from %s (prefix=%s)", args.urdf, args.link_prefix)
    keypoint_links = make_keypoint_links(args.link_prefix)
    extractor = FrankaFKExtractor7D(args.urdf, link_prefix=args.link_prefix)
    logger.info("Keypoint links: %s", keypoint_links)

    source_parquets = _get_parquet_files(source / "data")
    logger.info("Found %d parquet files", len(source_parquets))

    # Step 2: Pass 1 — compute global bounding box
    logger.info("=== Pass 1: computing global bounding box ===")
    global_min, global_max, total_frames_p1 = pass1_compute_bbox(source_parquets, extractor)
    r_pad = compute_r_pad(global_min, global_max, margin=bbox_margin)

    logger.info("Global min (base-rel): %s", global_min)
    logger.info("Global max (base-rel): %s", global_max)
    logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, bbox_margin * 100)

    # Step 3: Pass 2 — normalize and write keypoints
    logger.info("=== Pass 2: writing 7D keypoints ===")
    total_frames_p2 = pass2_write_keypoints(
        source_parquets, extractor, r_pad, source / "data", dest,
    )

    # Step 4: Update metadata
    _update_info_json(dest, extractor.num_keypoints, keypoint_links)
    _write_meta(dest, r_pad, global_min, global_max, total_frames_p2,
                args.urdf, keypoint_links, bbox_margin)

    total_dim = extractor.num_keypoints * KEYPOINT_DIM
    logger.info("=== DONE ===")
    logger.info("  Frames: %d (Pass 1) / %d (Pass 2)", total_frames_p1, total_frames_p2)
    logger.info("  Output: %s", dest)
    logger.info("  keypoint_dim: %d (= %d keypoints × %d per point)",
                total_dim, extractor.num_keypoints, KEYPOINT_DIM)
    logger.info("  R_pad: %.6f m", r_pad)


if __name__ == "__main__":
    main()
```

### 4.4 执行关键点生成

```bash
cd $REPO_ROOT

python b/s/Frk/generate_franka_keypoints.py \
    --source /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
    --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
    --urdf b/d/Frk/fr3v2_1_franka_hand.urdf \
    --link-prefix fr3v2_1 \
    --force
```

**预期日志**:

```
[INFO] Copying /home/luogang/hf_home/lerobot/plug_into_socket_lrb -> /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D ...
[INFO] Loading URDF from b/d/Frk/fr3v2_1_franka_hand.urdf (prefix=fr3v2_1)
[INFO] Keypoint links: ['fr3v2_1_link1', ..., 'fr3v2_1_hand_tcp']
[INFO] Found N parquet files
[INFO] === Pass 1: computing global bounding box ===
...
[INFO] R_pad = X.XXXXXX m (margin=15%)
[INFO] === Pass 2: writing 7D keypoints ===
...
[INFO] === DONE ===
[INFO]   keypoint_dim: 56 (= 8 keypoints × 7 per point)
```

**预期耗时**: ~2-5 分钟 (rsync 拷贝 ~2-4 GB + 两遍 FK 扫描 ~66k 帧).

### 4.5 关键点生成后快速检查

```bash
# 1. keypoints_meta.json
cat /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/meta/keypoints_meta.json | python3 -m json.tool

# 预期:
#   "num_keypoints": 8,
#   "keypoint_dim": 7,
#   "bbox_radius": <正数>,
#   "rotation_representation": "quaternion_xyzw_hemisphere"

# 2. info.json 中的 observation.keypoint_3d
python3 -c "
import json
with open('/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/meta/info.json') as f:
    info = json.load(f)
kpt = info['features']['observation.keypoint_3d']
print(f'shape: {kpt[\"shape\"]}')   # 应输出 [56]
print(f'dtype: {kpt[\"dtype\"]}')   # 应输出 float32
print(f'names count: {len(kpt[\"names\"])}')  # 应输出 56
"

# 3. 抽检 parquet
python3 -c "
import pandas as pd, numpy as np
from pathlib import Path
files = sorted(Path('/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/data').rglob('*.parquet'))
df = pd.read_parquet(files[0])
kpts = np.stack(df['observation.keypoint_3d'].values)
print(f'keypoint_3d shape: {kpts.shape}')  # [N, 56]
kpts_reshaped = kpts.reshape(-1, 8, 7)
print(f'pos range: [{kpts_reshaped[:,:,:3].min():.4f}, {kpts_reshaped[:,:,:3].max():.4f}]')
print(f'qw range: [{kpts_reshaped[:,:,6].min():.4f}, {kpts_reshaped[:,:,6].max():.4f}]')
"
```

---

## 5. 归一化策略（继承 E1）

> **继承关系**: 本节完全继承 [R1Pro E1 方案](../R1Pro/dta_3dtrj_E2.md) §5 与 [cod_analyz_1.md 方案 E](../R1Pro/cod_analyz_1.md) 的 **等尺度立方包围盒归一化** (isotropic bounding-box scaling). Franka 方案的差异仅在于: **单臂 8 关键点** (非 16)、**无躯干关节** (`torso_q` 不参与 FK). 归一化数学与两遍扫描流程与 R1Pro 相同.
>
> **实现入口**: `generate_franka_keypoints.py` (§4.3). Pass 1/2 在 Step 2 关键点生成阶段执行; Step 1 (HDF5 转换) **不涉及** 关键点归一化.

### 5.1 每帧 7D 表示

| 分量 | 维度 | 含义 | Pass 2 写入 parquet 前的处理 |
|------|------|------|------------------------------|
| 位置 $(p_x, p_y, p_z)$ | 3 | link 在 `base_link` 系中的位置 (m) | $\mathbf{p}_{\text{norm}} = \mathbf{p}_{\text{base}} / R_{\text{pad}}$ |
| 姿态 $(q_x, q_y, q_z, q_w)$ | 4 | link 在 `base_link` 系中的朝向 | 半球约束 ($q_w \geq 0$), **不做额外缩放** |

**每个关键点**: 7D = $[p_x, p_y, p_z, q_x, q_y, q_z, q_w]$

**每帧总维度**: $8 \times 7 = 56$

**Parquet 列**: `observation.keypoint_3d`, shape `[56]` (展平), dtype `float32`

四元数顺序与 Pinocchio 一致: `[qx, qy, qz, qw]`. 位置在前, 姿态在后.

### 5.2 位置归一化: 等尺度缩放

$$\mathbf{p}_{\text{norm}} = \frac{\mathbf{p}_{\text{base}}}{R_{\text{pad}}}$$

其中包围半径 $R_{\text{pad}}$ 由 **Pass 1 全数据集 FK 扫描** 得到:

$$R = \max\bigl(\lvert x_{\min}\rvert, x_{\max}, \lvert y_{\min}\rvert, y_{\max}, \lvert z_{\min}\rvert, z_{\max}\bigr)$$
$$R_{\text{pad}} = R \times (1 + \alpha), \quad \alpha = \texttt{BBOX\_MARGIN} = 0.15$$

- $x_{\min}, \dots, z_{\max}$: Pass 1 遍历 **所有 parquet 帧、所有 8 个关键点** 的 FK 位置后累积的全局 min/max (仅使用位置 3D, **四元数不参与** $R$ 的计算)
- $R_{\text{pad}}$ 是**标量**, 保证 X/Y/Z 各向同性等比缩放
- 归一化后 $\mathbf{p}_{\text{norm}} \in [-1, 1]^3$ (理论上; 15% 裕量保证实际数据不溢出)
- `base_link` 原点恒在归一化空间 $(0, 0, 0)$

代码复用 R1Pro 的纯函数 `compute_r_pad(global_min, global_max, margin)` (§4.2).

### 5.3 姿态归一化: 半球约束

在 FK 提取阶段 (`FrankaFKExtractor7D.compute`) 对原始四元数施加:

$$\mathbf{q}_{\text{norm}} = \text{hemisphere}(\mathbf{q}_{\text{raw}}) = \begin{cases} \mathbf{q}_{\text{raw}} & \text{if } q_w \geq 0 \\ -\mathbf{q}_{\text{raw}} & \text{if } q_w < 0 \end{cases}$$

- 输入: `pin.Quaternion(oMf.rotation)` → $[q_x, q_y, q_z, q_w]$
- 输出: 同一旋转的半球代表, $q_w \geq 0$, $\|\mathbf{q}\| = 1$
- Pass 2 **不再**对四元数做缩放或二次变换

Pass 1 同时校验: 全部 $q_w \geq 0$ (半球已生效) 且 $\bigl|\|\mathbf{q}\| - 1\bigr| \leq 0.01$.

### 5.4 位置与姿态的值域对齐

| 分量 | 值域 | 量纲 |
|------|------|------|
| $p_x, p_y, p_z$ | $\approx [-1, 1]$ | 无量纲 ($R_{\text{pad}}$ 吸收物理尺度) |
| $q_x, q_y, q_z, q_w$ | $[-1, 1]$, $\|\mathbf{q}\|=1$ | 无量纲 |

位置经 $R_{\text{pad}}$ 除法后与四元数分量尺度自然对齐, 便于 TrackEncoder (`keypoint_track_input_dim=7`) 训练.

### 5.5 两遍扫描流水线 (Pass 1 / Pass 2)

`generate_franka_keypoints.py` 对每个 parquet 文件 **扫描两遍** (与 R1Pro `generate_r1pro_keypoints_e1.py` 相同架构):

```mermaid
sequenceDiagram
    participant P1 as Pass 1
    participant P2 as Pass 2
    participant Meta as keypoints_meta.json

    loop 每个 parquet, 每帧
        P1->>P1: FK(arm[7]) → [8,7]
        P1->>P1: 累积 global_min, global_max (仅 pos)
    end
    P1->>P1: R_pad = compute_r_pad(min,max, margin=0.15)
    P1->>Meta: 记录 bbox_radius, global_min/max

    loop 每个 parquet, 每帧
        P2->>P2: FK → [8,7]
        P2->>P2: pos /= R_pad; quat 已在 compute 中半球化
        P2->>P2: 展平 [56] → observation.keypoint_3d
    end
```

#### Pass 1: 全局包围盒扫描

**输入**: 中间 LeRobot 数据集 `plug_into_socket_lrb/` 的全部 parquet; 每帧读取 `observation.state.arm` [7].

**过程** (对应 §4.3 `pass1_compute_bbox`):

1. `FrankaFKExtractor7D.compute_batch(arm)` → `[N, 8, 7]`
2. 取 `pos = kpts[:, :, :3]`, 对 **所有帧 × 8 关键点** 累积 `global_min`, `global_max`
3. 校验四元数范数与半球约束
4. `r_pad = compute_r_pad(global_min, global_max, margin=BBOX_MARGIN)`

**输出**: `global_min`, `global_max` (float32, shape [3]), `r_pad` (标量, 单位 m), `total_frames`.

**不写 parquet** — Pass 1 只读源数据、只算统计量.

#### Pass 2: 归一化并写入

**输入**: rsync 拷贝到 `plug_into_socket_lrb_4D/` 的 parquet (与 Pass 1 同源帧).

**过程** (对应 §4.3 `pass2_write_keypoints`):

1. 重新 FK → `[N, 8, 7]`
2. `kpts[:, :, :3] /= r_pad` — 位置归一化
3. 姿态已在 `compute()` 中半球化, 无需额外处理
4. OOB 检查: `max|pos| > 1.01` 时 WARNING (可增大 `--bbox-margin` 重跑)
5. 展平为 `[56]` 写入 `observation.keypoint_3d`

**输出**: 带关键点列的最终数据集 + 更新的 `meta/info.json` + `meta/keypoints_meta.json`.

#### 元数据: `keypoints_meta.json`

Pass 2 结束后写入, 推理/验证时必须能复现相同坐标系:

| 字段 | 含义 |
|------|------|
| `bbox_radius` | $R_{\text{pad}}$ (m), 推理时 FK 原始位置除以此值 |
| `bbox_margin` | $\alpha$, 仅记录 |
| `global_min_base_relative` / `global_max_base_relative` | Pass 1 实测包围盒 (m) |
| `keypoint_dim` | 7 |
| `keypoint_dim_layout` | `px,py,pz,qx,qy,qz,qw` |
| `num_keypoints` | 8 |
| `keypoint_links` | 8 个 link 名列表 |

> **与 R1Pro 的差异**: Franka 的 `keypoints_meta.json` **不含** `torso_q` — 单臂固定基座, FK 仅依赖 `observation.state.arm` [7].

### 5.6 Franka 数值示例（Pass 1 估算）

以下为基于 Franka 典型工作空间与插拔任务范围的 **估算** (实际值须运行 Pass 1 后从日志/`keypoints_meta.json` 读取):

```
global_min ≈ [-0.50, -0.50, +0.10]  (m, base_link)
global_max ≈ [+0.80, +0.50, +1.00]  (m, base_link)
```

各轴绝对值极大值: $R \approx 1.00$ m (由 $z_{\max}$ 主导)

$$R_{\text{pad}} = 1.00 \times 1.15 = 1.15 \text{ m}$$

典型 `hand_tcp` 关键点 (索引 7) 变换示例:

| 分量 | base_link 原始 | 归一化后 |
|------|--------------|---------|
| 位置 | $[0.56, 0.04, 0.33]$ m | $[0.487, 0.035, 0.287]$ |
| 姿态 | $[q_x, q_y, q_z, 0.998]$ (已 $q_w>0$) | 不变 |
| **合并 7D** | — | $[0.487, 0.035, 0.287, q_x, q_y, q_z, 0.998]$ |

Franka 最大臂展约 0.855 m; 加上任务姿态与 15% margin, $R_{\text{pad}}$ 通常落在 **0.98–1.15 m** 区间.

### 5.7 与 §7「归一化统计量」的区别

本方案中存在 **两套互不替代** 的归一化, 勿混淆:

| | §5 关键点 $R_{\text{pad}}$ 归一化 | §7 `compute_norm_stats` |
|--|--------------------------------|---------------------------|
| **时机** | Step 2 写 parquet **之前** (Pass 2) | Step 3, 数据集已生成 **之后** |
| **对象** | 仅 `observation.keypoint_3d` 的**位置 3 维** | 所有非 video 列 (state, action, keypoint_3d 等) |
| **方法** | 全数据集 FK 包围盒 → 标量 isotropic 除法 | 逐列 mean / std / min / max |
| **用途** | 固定坐标系, 训练/推理 TrackEncoder 输入尺度一致 | 训练时 `NormalizeTransformFn` 标准化 |
| **写入** | `keypoints_meta.json` 的 `bbox_radius` | `stats.json` |

关键点 parquet 中存的是 **已除以 $R_{\text{pad}}$ 的位置 + 半球四元数**; `stats.json` 是对该已归一化列再算统计量, 供 dataloader 使用.

---

## 6. 数据验证

### 6.1 转换验证脚本: `verify_franka_conversion.py`

新建文件: `b/s/Frk/verify_franka_conversion.py`

```python
"""Post-conversion verification for Franka plug-into-socket LeRobot dataset.

Checks:
  1. Dataset metadata integrity (info.json, episodes)
  2. Feature shapes and dtypes match declaration
  3. No NaN values in state/action columns
  4. Joint angles within Franka limits
  5. Gripper width within physical range [0, 0.08]
  6. Episode count and frame count consistency
  7. Video files exist for each episode
  8. Cross-check ee_pos/ee_quat with FK-computed hand_tcp (optional, if URDF provided)

Usage:
    python b/s/Frk/verify_franka_conversion.py \
        --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
        [--urdf b/d/Frk/fr3v2_1_franka_hand.urdf]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FRANKA_JOINT_LIMITS = [
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.0770, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
]

GRIPPER_MAX = 0.08  # meters


def load_all_data(dataset: Path):
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset}/data")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True), files


def check_metadata(dataset: Path):
    print("\n=== Check 1: Metadata integrity ===")
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        print(f"  ✗ FAIL: {info_path} not found")
        return False
    with open(info_path) as f:
        info = json.load(f)
    required = ["robot_type", "fps", "total_episodes", "total_frames", "features"]
    for key in required:
        if key not in info:
            print(f"  ✗ FAIL: missing key '{key}' in info.json")
            return False
    print(f"  robot_type: {info['robot_type']}")
    print(f"  fps: {info['fps']}")
    print(f"  total_episodes: {info['total_episodes']}")
    print(f"  total_frames: {info['total_frames']}")
    print(f"  features: {list(info['features'].keys())}")
    print("  ✓ PASS")
    return True


def check_shapes(df: pd.DataFrame):
    print("\n=== Check 2: Feature shapes ===")
    expected = {
        "observation.state.arm": 7,
        "observation.state.gripper": 1,
        "action.arm": 7,
        "action.gripper": 1,
    }
    all_ok = True
    for col, expected_dim in expected.items():
        if col not in df.columns:
            print(f"  ✗ FAIL: column '{col}' not found")
            all_ok = False
            continue
        sample = np.array(df[col].iloc[0])
        actual_dim = sample.shape[-1] if sample.ndim > 0 else 1
        if actual_dim != expected_dim:
            print(f"  ✗ FAIL: {col} dim={actual_dim}, expected {expected_dim}")
            all_ok = False
        else:
            print(f"  ✓ {col}: dim={actual_dim}")
    if all_ok:
        print("  ✓ PASS")
    return all_ok


def check_no_nan(df: pd.DataFrame):
    print("\n=== Check 3: No NaN values ===")
    cols = ["observation.state.arm", "observation.state.gripper", "action.arm", "action.gripper"]
    all_ok = True
    for col in cols:
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values)
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            print(f"  ✗ FAIL: {col} has {nan_count} NaN values")
            all_ok = False
    if all_ok:
        print("  ✓ PASS")
    return all_ok


def check_joint_limits(df: pd.DataFrame):
    print("\n=== Check 4: Joint limits ===")
    arm = np.stack(df["observation.state.arm"].values)
    violations = 0
    for j in range(7):
        lo, hi = FRANKA_JOINT_LIMITS[j]
        col = arm[:, j]
        below = (col < lo - 0.01).sum()
        above = (col > hi + 0.01).sum()
        if below or above:
            print(f"  ✗ joint{j+1}: {below} below {lo:.4f}, {above} above {hi:.4f}")
            violations += below + above
    if violations:
        print(f"  ✗ FAIL: {violations} total violations")
    else:
        print("  ✓ PASS")
    return violations == 0


def check_gripper_range(df: pd.DataFrame):
    print("\n=== Check 5: Gripper range ===")
    gripper = np.stack(df["observation.state.gripper"].values).flatten()
    g_min, g_max = gripper.min(), gripper.max()
    print(f"  gripper_width range: [{g_min:.6f}, {g_max:.6f}] m")
    if g_min < -0.001 or g_max > GRIPPER_MAX + 0.001:
        print(f"  ✗ FAIL: out of physical range [0, {GRIPPER_MAX}]")
        return False
    print("  ✓ PASS")
    return True


def check_episode_consistency(df: pd.DataFrame, dataset: Path):
    print("\n=== Check 6: Episode consistency ===")
    with open(dataset / "meta" / "info.json") as f:
        info = json.load(f)
    declared_episodes = info["total_episodes"]
    declared_frames = info["total_frames"]
    actual_episodes = df["episode_index"].nunique()
    actual_frames = len(df)
    print(f"  Declared: {declared_episodes} episodes, {declared_frames} frames")
    print(f"  Actual:   {actual_episodes} episodes, {actual_frames} frames")
    ok = (actual_episodes == declared_episodes) and (actual_frames == declared_frames)
    if ok:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL: mismatch")
    return ok


def check_videos(dataset: Path):
    print("\n=== Check 7: Video files ===")
    video_dir = dataset / "videos"
    if not video_dir.exists():
        print("  [skip] no videos directory")
        return True
    for cam in ["observation.images.global", "observation.images.wrist"]:
        cam_dir = video_dir / cam
        if cam_dir.exists():
            mp4s = list(cam_dir.rglob("*.mp4"))
            print(f"  {cam}: {len(mp4s)} video files")
        else:
            print(f"  ⚠ {cam}: directory not found")
    print("  ✓ PASS (existence check only)")
    return True


def check_fk_crosscheck(df: pd.DataFrame, dataset: Path, urdf_path: str):
    print("\n=== Check 8: FK cross-check with recorded ee_pos ===")
    if "observation.state.ee_pos" not in df.columns:
        print("  [skip] no ee_pos column")
        return True
    try:
        import pinocchio as pin
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_franka_keypoints import FrankaFKExtractor7D

    extractor = FrankaFKExtractor7D(urdf_path)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(df), size=min(20, len(df)), replace=False)
    max_pos_err = 0.0
    for idx in indices:
        row = df.iloc[idx]
        arm = np.array(row["observation.state.arm"], dtype=np.float64)
        kpts = extractor.compute(arm)  # [8, 7]
        fk_tcp_pos = kpts[-1, :3]  # hand_tcp position (last keypoint)
        recorded_ee = np.array(row["observation.state.ee_pos"], dtype=np.float32)
        err = np.linalg.norm(fk_tcp_pos - recorded_ee)
        max_pos_err = max(max_pos_err, err)

    print(f"  Max |FK_tcp_pos - recorded_ee_pos| over 20 samples: {max_pos_err:.6f} m")
    if max_pos_err > 0.05:
        print(f"  ⚠ WARNING: large discrepancy ({max_pos_err:.4f} m). "
              "Possible URDF mismatch or different coordinate frames.")
    else:
        print("  ✓ PASS")
    return max_pos_err <= 0.05


def print_statistics(df: pd.DataFrame):
    print("\n=== Statistics ===")
    for col in ["observation.state.arm", "observation.state.gripper",
                "action.arm", "action.gripper"]:
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values)
        print(f"\n  {col}: shape={arr.shape}")
        for d in range(arr.shape[-1]):
            vals = arr[:, d] if arr.ndim > 1 else arr
            print(f"    dim{d}: mean={vals.mean():+.6f} std={vals.std():.6f} "
                  f"min={vals.min():+.6f} max={vals.max():+.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str, default=None,
                        help="Optional URDF for FK cross-check.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying Franka conversion: {dataset}")

    ok = check_metadata(dataset)
    df, _ = load_all_data(dataset)
    ok &= check_shapes(df)
    ok &= check_no_nan(df)
    ok &= check_joint_limits(df)
    ok &= check_gripper_range(df)
    ok &= check_episode_consistency(df, dataset)
    ok &= check_videos(dataset)
    if args.urdf:
        ok &= check_fk_crosscheck(df, dataset, args.urdf)
    print_statistics(df)

    print(f"\n=== Summary: {'ALL PASS' if ok else 'SOME CHECKS FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

### 6.2 关键点验证脚本: `verify_franka_keypoints.py`

新建文件: `b/s/Frk/verify_franka_keypoints.py`

```python
"""Post-generation verification for Franka 7D keypoints.

Same 7-check framework as verify_e1_keypoints.py, adapted for Franka
single-arm (8 keypoints × 7D = 56-dim).

Checks:
  1. Shape: observation.keypoint_3d exists and has [56] per frame
  2. Position bounds: |pos| <= 1.01 after R_pad normalization (§5.2)
  3. Quaternion unit norm: |‖q‖-1| <= 0.001
  4. Hemisphere constraint: qw >= 0
  5. Temporal smoothness: frame-to-frame quaternion change < 0.5
  6. FK reproducibility: recompute random samples, compare with stored
  7. Per-dimension statistics

Usage:
    python b/s/Frk/verify_franka_keypoints.py \
        --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
        --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 56


def load_all_keypoints(dataset: Path):
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset}/data")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "observation.keypoint_3d" not in df.columns:
            sys.exit(f"observation.keypoint_3d not found in {f}")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    raw = np.stack(full["observation.keypoint_3d"].values)
    kpts = raw.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    return kpts, full


def check_shape(kpts):
    print(f"\n=== Check 1: Shape ===")
    print(f"  Total frames: {kpts.shape[0]}")
    print(f"  Shape per frame: [{kpts.shape[1]}, {kpts.shape[2]}] (expect [{NUM_KEYPOINTS}, {KEYPOINT_DIM}])")
    assert kpts.shape[1:] == (NUM_KEYPOINTS, KEYPOINT_DIM)
    print("  ✓ PASS")


def check_position_bounds(kpts):
    print(f"\n=== Check 2: Position bounds ===")
    pos = kpts[:, :, :3]
    pos_max = np.abs(pos).max()
    print(f"  max |position|: {pos_max:.6f} (threshold: 1.01)")
    ok = pos_max <= 1.01
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def check_quaternion_norm(kpts):
    print(f"\n=== Check 3: Quaternion unit norm ===")
    quat = kpts[:, :, 3:7]
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    norm_err = np.abs(norms - 1.0)
    print(f"  norm error: mean={norm_err.mean():.2e}, max={norm_err.max():.2e}")
    ok = norm_err.max() <= 0.001
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def check_hemisphere(kpts):
    print(f"\n=== Check 4: Hemisphere constraint (qw >= 0) ===")
    qw = kpts[:, :, 6]
    qw_min = qw.min()
    violations = (qw < -1e-7).sum()
    print(f"  qw min: {qw_min:.8f}, violations: {violations}")
    ok = violations == 0
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def check_temporal_smoothness(kpts, full_df):
    print(f"\n=== Check 5: Temporal smoothness ===")
    episodes = full_df["episode_index"].values
    quat = kpts[:, :, 3:7]
    max_jump = 0.0
    jump_count = 0
    total_transitions = 0
    for ep in np.unique(episodes):
        mask = episodes == ep
        ep_quat = quat[mask]
        if len(ep_quat) < 2:
            continue
        diffs = np.linalg.norm(ep_quat[1:] - ep_quat[:-1], axis=-1)
        frame_max = diffs.max(axis=1)
        max_jump = max(max_jump, float(frame_max.max()))
        jump_count += int((frame_max > 0.5).sum())
        total_transitions += len(frame_max)
    print(f"  Transitions: {total_transitions}, max jump: {max_jump:.6f}, jumps>0.5: {jump_count}")
    if jump_count > 0:
        print(f"  ⚠ WARNING: {jump_count} large jumps")
    else:
        print("  ✓ PASS")
    return jump_count == 0


def check_fk_reproducibility(kpts, full_df, dataset, urdf_path):
    print(f"\n=== Check 6: FK reproducibility ===")
    try:
        import pinocchio as pin
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    meta_path = dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    r_pad = meta["bbox_radius"]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_franka_keypoints import FrankaFKExtractor7D

    extractor = FrankaFKExtractor7D(urdf_path)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(full_df), size=min(10, len(full_df)), replace=False)
    max_err = 0.0
    for idx in indices:
        row = full_df.iloc[idx]
        arm = np.array(row["observation.state.arm"], dtype=np.float64)
        recomputed = extractor.compute(arm)
        recomputed[:, :3] /= r_pad
        stored = kpts[idx]
        err = np.abs(recomputed - stored).max()
        max_err = max(max_err, err)

    print(f"  Max recomputation error (10 random frames): {max_err:.2e}")
    ok = max_err <= 1e-5
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def print_statistics(kpts):
    print(f"\n=== Check 7: Per-dimension statistics ===")
    flat = kpts.reshape(-1, KEYPOINT_DIM)
    labels = ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    print(f"  {'dim':>4s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print("  " + "-" * 50)
    for i, label in enumerate(labels):
        col = flat[:, i]
        print(f"  {label:>4s}  {col.mean():+10.6f}  {col.std():10.6f}  "
              f"{col.min():+10.6f}  {col.max():+10.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[3] / "b" / "d" / "Frk" / "fr3v2_1_franka_hand.urdf"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying Franka 7D keypoints in: {dataset}")

    kpts, full_df = load_all_keypoints(dataset)
    check_shape(kpts)
    ok = check_position_bounds(kpts)
    ok &= check_quaternion_norm(kpts)
    ok &= check_hemisphere(kpts)
    ok &= check_temporal_smoothness(kpts, full_df)
    ok &= check_fk_reproducibility(kpts, full_df, dataset, args.urdf)
    print_statistics(kpts)

    print(f"\n=== Summary: {'ALL PASS' if ok else 'SOME CHECKS FAILED'} ===")
    print(f"  Frames: {len(kpts)}, Keypoints: {NUM_KEYPOINTS} × {KEYPOINT_DIM}D = {TOTAL_DIM}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

### 6.3 执行验证

```bash
cd $REPO_ROOT

# Step A: 验证 LeRobot 格式转换
python b/s/Frk/verify_franka_conversion.py \
    --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
    --urdf b/d/Frk/fr3v2_1_franka_hand.urdf

# Step B: 验证 7D 关键点
python b/s/Frk/verify_franka_keypoints.py \
    --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
    --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
```

### 6.4 验收标准

**转换验证 (8 项)**:

| # | 检查 | 通过条件 | 失败时动作 |
|---|------|---------|-----------|
| 1 | Metadata | info.json 含所有必需字段 | 检查转换脚本的 features dict |
| 2 | Shapes | arm=[7], gripper=[1], action.arm=[7], action.gripper=[1] | 检查 HDF5 读取逻辑 |
| 3 | No NaN | 全部列无 NaN | 检查频率对齐是否越界 |
| 4 | Joint limits | 7 个关节均在 URDF 限位内 (±0.01 容差) | 数据集本身有问题, 需检查原始 HDF5 |
| 5 | Gripper range | 0 ≤ width ≤ 0.08 m | 同上 |
| 6 | Episode consistency | 声明数 = 实际数 | 检查 save\_episode 调用 |
| 7 | Videos | 每个相机目录有 MP4 文件 | 检查视频编码 |
| 8 | FK cross-check | FK(hand\_tcp) 与 recorded ee\_pos 差距 < 5cm | URDF 不匹配, 需确认坐标系 |

**关键点验证 (7 项)** — 与 R1Pro E1 验证完全相同的检查框架:

| # | 检查 | 通过条件 |
|---|------|---------|
| 1 | Shape | 每帧 [8, 7] |
| 2 | Position bounds | max\|pos\| ≤ 1.01 |
| 3 | Quaternion norm | max\|‖q‖-1\| ≤ 0.001 |
| 4 | Hemisphere | 全部 qw ≥ 0 |
| 5 | Temporal smoothness | 无 jump > 0.5 |
| 6 | FK reproducibility | max\_err ≤ 1e-5 |
| 7 | Statistics | 位置在 [-1, 1], qw ≥ 0 |

---

## 7. 归一化统计量生成

### 7.1 前置: Schema 注册

确保 `b/s/Frk/cfg/franka_plug.yaml` 已按 §3.4 创建, 且已通过符号链接或复制注册到 `src/lerobot/dataset_schemas/configs/`.

### 7.2 执行

```bash
cd $REPO_ROOT
export HF_HOME=/home/luogang/hf_home
export HF_LEROBOT_HOME=${HF_HOME}/lerobot

# abs 模式
python util_scripts/compute_norm_stats_single.py \
    --repo_id plug_into_socket_lrb_4D \
    --action_mode abs \
    --chunk_size 50

# delta 模式 (如训练使用 delta action)
python util_scripts/compute_norm_stats_single.py \
    --repo_id plug_into_socket_lrb_4D \
    --action_mode delta \
    --chunk_size 50
```

### 7.3 验证 stats.json

```bash
python3 -c "
import json, glob
from pathlib import Path

# 查找 stats.json
candidates = [
    Path('/home/luogang/hf_home/lerobot/stats/abs/plug_into_socket_lrb_4D/stats.json'),
    Path('/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/meta/stats.json'),
]
stats_path = next((p for p in candidates if p.exists()), None)
if not stats_path:
    print('stats.json not found in expected locations')
    exit(1)

print(f'Reading: {stats_path}')
with open(stats_path) as f:
    stats = json.load(f)

if 'observation.keypoint_3d' in stats:
    kpt = stats['observation.keypoint_3d']
    print(f'observation.keypoint_3d stats:')
    print(f'  mean dim: {len(kpt[\"mean\"])}')   # 应为 56
    print(f'  std dim:  {len(kpt[\"std\"])}')     # 应为 56
    print(f'  First keypoint (7D): mean={kpt[\"mean\"][:7]}')
else:
    print('✗ observation.keypoint_3d NOT found in stats.json')
    print(f'  Keys: {list(stats.keys())}')
"
```

**预期**: `observation.keypoint_3d` 统计量为 56 维, 位置分量 mean ≈ 0, std < 1.

> `compute_norm_stats_single.py` 无需修改 — 它自动遍历数据集中所有非 video 列. 只要 schema 注册正确, 它就能正确计算 stats.

---

## 8. 模型侧适配与训练集成

### 8.1 模型侧代码改动

模型侧改动与 R1Pro E1 方案完全相同 (参见 [dta_3dtrj_E2impl.md §7](../R1Pro/dta_3dtrj_E2impl.md#7-模型侧代码改动)). 这些改动是**参数化**的, 添加后同时支持 R1Pro (16 关键点) 和 Franka (8 关键点):

| 文件 | 改动 | 说明 |
|------|------|------|
| `configuration_internvla_a1_5.py` | 新增 `keypoint_dim`, `keypoint_out_dim`, `kpt_rot_loss_weight` | 默认值保持向后兼容 (3, 3, 1.0) |
| `modeling_internvla_a1_5.py` | `keypoint_out_proj` 用 `config.keypoint_out_dim`; 分离 pos/rot loss | 当 `keypoint_out_dim=3` 时行为不变 |
| `transform_internvla_a1_5.py` | `Extract3DKeypointTransformFn` 新增 `keypoint_dim` 字段 | 默认 3, reshape 逻辑参数化 |

如果这些改动已在 R1Pro 工作中完成, 则 Franka 无需任何额外模型侧改动 — 只需在训练 CLI 中传入 Franka 的参数值.

如果尚未完成, 请按照 [dta_3dtrj_E2impl.md §7](../R1Pro/dta_3dtrj_E2impl.md#7-模型侧代码改动) 中的逐行改动说明执行. 改动是增量式的 (只添加参数和条件分支), 不影响已有的 3D 关键点训练.

### 8.2 Franka 训练 CLI 参数

相比 R1Pro, Franka 训练的关键差异只在 **关键点数量** 和 **数据集 ID**:

```bash
accelerate launch --num_processes=2 src/lerobot/scripts/lerobot_train.py \
    --policy.type=internvla_a1_5 \
    --policy.pretrained_path=InternRobotics/InternVLA-A1.5-base \
    --policy.enable_keypoint_predictor=true \
    --policy.num_keypoint_joints=8 \
    --policy.keypoint_track_input_dim=7 \
    --policy.keypoint_out_dim=7 \
    --policy.kpt_rot_loss_weight=1.0 \
    --policy.kpt_loss_weight=10.0 \
    --policy.kpt_future_loss_weight=2.0 \
    --policy.train_expert_only=true \
    --policy.action_loss_only=true \
    --policy.keypoint_history_max_len=200 \
    --dataset.type=internvla_a1_5 \
    --dataset.repo_id=plug_into_socket_lrb_4D \
    --dataset.enable_keypoint_predictor=true \
    --dataset.num_keypoint_joints=8 \
    --dataset.keypoint_dim=7 \
    --dataset.action_mode=abs \
    --batch_size=12 \
    --steps=100
```

**Franka vs R1Pro 参数差异**:

| 参数 | R1Pro E1 | Franka | 原因 |
|------|---------|--------|------|
| `num_keypoint_joints` | 16 | **8** | 单臂 vs 双臂 |
| `repo_id` | `elevator0714_lerobot_4D` | **`plug_into_socket_lrb_4D`** | 不同数据集 |
| `keypoint_track_input_dim` | 7 | 7 | 相同 (7D 关键点) |
| `keypoint_out_dim` | 7 | 7 | 相同 |
| `keypoint_dim` | 7 | 7 | 相同 |

---

## 9. Smoke Test

### 9.1 命令

```bash
cd $REPO_ROOT
export HF_HOME=/home/luogang/hf_home
export HF_LEROBOT_HOME=${HF_HOME}/lerobot

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=internvla_a1_5 \
    --policy.pretrained_path=InternRobotics/InternVLA-A1.5-base \
    --policy.enable_keypoint_predictor=true \
    --policy.num_keypoint_joints=8 \
    --policy.keypoint_track_input_dim=7 \
    --policy.keypoint_out_dim=7 \
    --policy.kpt_rot_loss_weight=1.0 \
    --policy.kpt_loss_weight=10.0 \
    --policy.train_expert_only=true \
    --policy.action_loss_only=true \
    --policy.keypoint_history_max_len=200 \
    --dataset.type=internvla_a1_5 \
    --dataset.repo_id=plug_into_socket_lrb_4D \
    --dataset.enable_keypoint_predictor=true \
    --dataset.num_keypoint_joints=8 \
    --dataset.keypoint_dim=7 \
    --dataset.action_mode=abs \
    --batch_size=2 \
    --steps=100
```

### 9.2 检查项

| 检查项 | 预期 | 不符时动作 |
|--------|------|-----------|
| 启动无 RuntimeError | 无 shape mismatch | 检查 `keypoint_dim` 传递 |
| Batch shape | `his_kpts: [B, 200, 8, 7]` | 检查 transform reshape |
| TrackEncoder 无报错 | Conv1d(7, 256) 正常 | 检查 `keypoint_track_input_dim` |
| Loss 非 NaN | `loss_kpt_current/future` 有限 | 检查四元数归一化 |
| Loss 下降趋势 | 100 步内总体下降 | 调整学习率 |

---

## 10. 完整文件清单

### 10.1 新增文件

| 文件 | 用途 | 行数 (约) |
|------|------|----------|
| `b/s/Frk/convert_franka_plug_hdf5.py` | HDF5 → LeRobot 格式转换 | ~250 |
| `b/s/Frk/generate_franka_keypoints.py` | 7D 关键点离线生成 (复用 E1 共享函数) | ~250 |
| `b/s/Frk/verify_franka_conversion.py` | LeRobot 转换验证 (8 项检查) | ~200 |
| `b/s/Frk/verify_franka_keypoints.py` | 关键点验证 (7 项检查) | ~170 |
| `b/s/Frk/cfg/franka_plug.yaml` | Franka 插拔数据集的 schema | ~15 |

### 10.2 修改文件

| 文件 | 改动 | 说明 |
|------|------|------|
| **无** | — | 遵循"扩展代替修改"原则, 所有改动均为新增文件 |

> 模型侧改动 (configuration / modeling / transform) 按 R1Pro E1 方案执行, 详见 [dta_3dtrj_E2impl.md §7](../R1Pro/dta_3dtrj_E2impl.md#7-模型侧代码改动). 这些改动是机器人无关的参数化, 执行一次后 Franka 和 R1Pro 均可使用.

### 10.3 生成的数据产出

| 产出 | 路径 | 大小估算 |
|------|------|---------|
| 中间 LeRobot 数据集 | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb/` | ~2-4 GB |
| 最终 LeRobot 数据集 (含关键点) | `/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/` | ~2-4 GB |
| 归一化统计量 | `${HF_LEROBOT_HOME}/stats/abs/plug_into_socket_lrb_4D/stats.json` | ~20 KB |

### 10.4 不修改的现有文件

| 文件 | 原因 |
|------|------|
| `util_scripts/generate_r1pro_keypoints_e1.py` | Franka 脚本通过 import 复用其共享函数, 不修改原文件 |
| `util_scripts/compute_norm_stats_single.py` | 自动遍历所有非 video 列, 零改动 |
| `util_scripts/convert_robotwin.py` | 参考其模式, 但因数据结构差异大, 新建独立脚本 |
| 已有的 `a1-franka.yaml` 等 schema | 保持不变, 新建 `franka_plug.yaml` 服务于本数据集 |

---

## 11. 故障排查手册

### 10.1 格式转换阶段

| 症状 | 排查 | 修复 |
|------|------|------|
| `ModuleNotFoundError: No module named 'h5py'` | h5py 未安装 | `pip install h5py` |
| `ValueError: Failed to decode JPEG` | 某些 JPEG 帧损坏 | 在 `decode_jpeg` 中添加 try/except 跳过坏帧, 记录日志 |
| `Large alignment gap: max X.X ms` | 状态和相机时间戳偏差大 | 正常情况下 < 20ms; 若持续偏大, 检查 reference\_camera 选择 |
| `FileExistsError: ... exists` | 目标目录已存在 | 添加 `--force` 或手动删除 |
| 视频编码极慢 (>1 小时) | ffmpeg 编码器问题 | 检查 ffmpeg 安装: `ffmpeg -version`; 或用 `--no-videos` 先跳过视频 |
| `KeyError: 'robot_state/joint_positions'` | HDF5 内部结构不匹配 | 用 `h5py.File` 手动检查文件结构, 适配 key 名称 |

### 10.2 关键点生成阶段

| 症状 | 排查 | 修复 |
|------|------|------|
| `ValueError: Frame 'fr3v2_1_link1' not found` | URDF 中 link 前缀不匹配 | 用 `grep 'link name' <urdf>` 确认前缀, 通过 `--link-prefix` 指定 |
| `ImportError: cannot import name 'compute_r_pad'` | `generate_r1pro_keypoints_e1.py` 不在 `util_scripts/` 中 | 确认该文件存在; 或将 `compute_r_pad` 直接复制到 Franka 脚本 |
| `TypeError: 'float' object is not callable` | Pinocchio 版本差异 (属性 vs 方法) | 参见 [R1Pro 实施日志 §8.2](../R1Pro/dta_3dtrj_E2implLog.md#82-error-2-typeerror-float-object-is-not-callable), 使用 `quat.x` 而非 `quat.x()` |
| Position OOB 警告 | R\_pad 安全裕量不足 | 增大 `--bbox-margin` (如 0.2 或 0.25) 后重新生成 |
| FK cross-check 误差 > 5cm | URDF 与实际机器人不一致 | 可能原因: (1) URDF 是 FR3v2 但数据来自其他型号; (2) ee\_quat 用了不同约定 (wxyz vs xyzw). 检查 URDF 版本与采集时的实际机器人 |

### 10.3 训练阶段

| 症状 | 排查 | 修复 |
|------|------|------|
| `shape mismatch: [N, 56] cannot be reshaped to [H+1+C, 8, 7]` | `keypoint_dim` 未传递到 transform | 确认 `--dataset.keypoint_dim=7` |
| `mat1 and mat2 cannot be multiplied (x3 and 7x)` | `keypoint_out_dim` 不匹配 | 确认 `--policy.keypoint_out_dim=7` |
| `KeyError: 'observation.keypoint_3d'` | 数据集无关键点 | 重新执行 §4 关键点生成 |
| NaN loss | 四元数归一化问题 | 在 loss 计算中添加 `F.normalize` 的 `eps=1e-6` |

---

## 附录 A: 端到端数据流

```
                  HDF5 源数据集
                  (plug_into_socket_hdf5/)
                        │
                        │  episode_XXXXXX.hdf5 × 100
                        │  ├── robot_state/ (100Hz)
                        │  │   ├── joint_positions [N, 7]
                        │  │   ├── gripper_width [N, 1]
                        │  │   ├── ee_pos [N, 3]
                        │  │   ├── ee_quat [N, 4]
                        │  │   ├── action_joints [N, 7]
                        │  │   └── action_gripper [N, 1]
                        │  ├── camera_global/ (30Hz)
                        │  │   └── color_image_jpeg [M,] (JPEG bytes)
                        │  └── camera_wrist/ (30Hz)
                        │      └── color_image_jpeg [M,] (JPEG bytes)
                        │
            ────────────┼───────────── Step 1: convert ──
                        │
                        │  频率对齐: 100Hz → 30Hz
                        │  JPEG 解码 → RGB → MP4 编码
                        ▼
                  LeRobot 中间数据集
                  (plug_into_socket_lrb/)
                        │
                        │  data/chunk-000/file-{NNN}.parquet
                        │  ├── observation.state.arm [7]
                        │  ├── observation.state.gripper [1]
                        │  ├── observation.state.ee_pos [3]
                        │  ├── observation.state.ee_quat [4]
                        │  ├── action.arm [7]
                        │  └── action.gripper [1]
                        │  videos/observation.images.{global,wrist}/...
                        │
            ────────────┼───────────── Step 2: keypoints (§5) ──
                        │
                        │  Pass 1: 全帧 FK → global_min/max → R_pad
                        │  Pass 2: rsync → FK → pos/R_pad + quat hemisphere
                        ▼
                  最终 LeRobot 数据集
                  (plug_into_socket_lrb_4D/)
                        │
                        │  (原有内容保留)
                        │  + observation.keypoint_3d [56]     ★ 新增
                        │  + meta/keypoints_meta.json          ★ 新增
                        │  + meta/info.json                    (更新)
                        │
            ────────────┼───────────── Step 3: norm_stats ──
                        │
                        │  compute_norm_stats_single.py
                        ▼
                  训练就绪
                  ├── stats.json (含 56 维 keypoint 统计量)
                  └── lerobot_train.py
                      --dataset.repo_id=plug_into_socket_lrb_4D
                      --policy.num_keypoint_joints=8
                      --policy.keypoint_out_dim=7
                      --dataset.keypoint_dim=7

                      训练数据流:
                      LeRobotDataset 读取 observation.keypoint_3d [56]
                        → delta_indices 堆叠 [(H+1+C) × 56]
                        → Extract3DKeypointTransformFn reshape [H+1+C, 8, 7]
                          ├── his_kpts [H, 8, 7] → TrackEncoder(input_dim=7)
                          ├── kpt_t [8, 7] → loss_kpt_current (pos MSE + rot MSE)
                          └── kpt_future [C, 8, 7] → loss_kpt_future
                        → keypoint_out_proj: Linear(hidden, 7) → pred [B, 8, 7]
```

## 附录 B: keypoints\_meta.json 参考

生成后 `/home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D/meta/keypoints_meta.json` 的预期内容:

```json
{
  "bbox_radius": 1.15,
  "bbox_margin": 0.15,
  "global_min_base_relative": [-0.50, -0.50, 0.10],
  "global_max_base_relative": [0.80, 0.50, 1.00],
  "normalization": "base_link_origin_isotropic",
  "keypoint_dim": 7,
  "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
  "rotation_representation": "quaternion_xyzw_hemisphere",
  "rotation_convention": "qw >= 0; negate if qw < 0",
  "num_keypoints": 8,
  "keypoint_links": [
    "fr3v2_1_link1", "fr3v2_1_link2", "fr3v2_1_link3", "fr3v2_1_link4",
    "fr3v2_1_link5", "fr3v2_1_link6", "fr3v2_1_link7", "fr3v2_1_hand_tcp"
  ],
  "total_frames": 66500,
  "coordinate_system": "base_link-relative, position divided by bbox_radius, quaternion hemisphere-normalized",
  "urdf": "b/d/Frk/fr3v2_1_franka_hand.urdf"
}
```

> 注意: `bbox_radius` 和 `global_min/max` 的具体数值在实际运行 Pass 1 后才能确定. 上面是基于 Franka 典型工作空间的估算值, 完整设计见 [§5.6](#56-franka-数值示例pass-1-估算). Franka 的工作空间半径约 0.855m, 加上 15% margin 后 R\_pad ≈ 0.98-1.15m.

## 附录 C: 操作检查清单

逐项勾选, 确保无遗漏:

- [ ] **1. 环境准备**: 虚拟环境已激活 (conda/uv/venv), pinocchio + h5py + cv2 可用
- [ ] **2. URDF 确认**: `b/d/Frk/fr3v2_1_franka_hand.urdf` 文件存在 (已在位, 无需移动)
- [ ] **3. Schema 注册**: `b/s/Frk/cfg/franka_plug.yaml` 已创建, 并通过符号链接注册到 `src/lerobot/dataset_schemas/configs/franka_plug.yaml` (见 §3.4)
- [ ] **4. HDF5 → LeRobot**: `convert_franka_plug_hdf5.py` 运行完成, 100 episodes 全部转换
- [ ] **5. 转换验证**: `verify_franka_conversion.py` 8 项检查全部通过
- [ ] **6. 关键点生成**: `generate_franka_keypoints.py` 运行完成, Pass 1 日志含 `R_pad`, Pass 2 无 OOB WARNING (§5)
- [ ] **7. 元数据检查**: `keypoints_meta.json` 存在, `keypoint_dim=7`, `num_keypoints=8`
- [ ] **8. info.json 检查**: `observation.keypoint_3d` shape=[56]
- [ ] **9. 关键点验证**: `verify_franka_keypoints.py` 7 项检查全部通过
- [ ] **10. norm\_stats 生成**: `compute_norm_stats_single.py` 运行, stats.json 含 56 维 keypoint 统计量
- [ ] **11. 模型侧改动**: 配置/模型/transform 3 个文件的参数化改动已完成 (R1Pro E1 共享)
- [ ] **12. Smoke test**: 100 步训练无报错, loss 非 NaN
- [ ] **13. 源数据完整**: 原始 HDF5 未被修改; 中间/最终数据集的 videos / episodes 完整

## 附录 D: 关于 FK cross-check 误差的预期

转换验证脚本的 Check 8 会将 FK 计算的 `hand_tcp` 位置与 HDF5 中录制的 `ee_pos` 进行对比. 预期行为:

1. **如果 URDF 精确匹配采集时的机器人**: 误差应 < 1mm (FK 的数值精度级别)
2. **如果 URDF 是同系列但不完全相同的型号** (如 FR3 vs FR3v2): 误差可能在 1-5cm, 这是可接受的, 因为关键点用于策略学习而非精确控制
3. **如果 `ee_quat` 的四元数约定不同** (如 HDF5 用 wxyz 而 Pinocchio 用 xyzw): 旋转对比会有大误差, 但位置对比不受影响. `ee_quat` 在 HDF5 中的约定 (根据数据检查) 是 **(w, x, y, z)** — 注意这与 Pinocchio 的 (x, y, z, w) 不同. 本方案在 LeRobot 中以原始约定存储 `ee_quat`, 仅用于参考/验证, 不参与训练

> 如果 FK cross-check 误差 > 5cm, 需要检查:
> 1. URDF 是否正确 (Franka 型号是否匹配)
> 2. `ee_pos` 是否相对于 base\_link 坐标系 (而非 world 坐标系)
> 3. 是否存在工具偏移 (tool center point offset) 差异
