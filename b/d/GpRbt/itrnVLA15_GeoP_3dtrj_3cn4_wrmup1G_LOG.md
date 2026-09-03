# 方案 A 落地实施日志（kptsim 体素坐标 → Keypoint Expert Warmup）

> 对应方案: [`itrnVLA15_GeoP_3dtrj_3cn4_wrmup.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup.md) §12–§14「方案 A 落地实施补充」  
> 实施日期: 2026-08-11  
> 环境: conda `itvlaGp`，GPU 0（NVIDIA RTX PRO 6000 Blackwell）

---

## 总览

| 阶段 | 状态 |
|:---|:---:|
| 数据注入 `inject_kptsim_keypoints.py` | ✅ |
| Layer 1 静态验收（6 项） | ✅ |
| LeRobot v2.1 → v3.0 转换 | ✅ |
| Layer 2 Dataset 加载 | ✅ |
| 权重下载（InternVLA base + GeoPredict） | ✅ |
| 视频解码修复（PyAV direct） | ✅ |
| Layer 3 单卡 Smoke 100 step | ✅ 收敛 |
| v30 → lrbv30 持久化 + 自包含复验 | ✅ 收敛 |
| 正式 Warmup 400 step | ⏸ 未跑（Smoke 已满足收敛判据） |

---

## Phase 0 — 前置检查

### 已有资产（无需重新生成）

| 资产 | 路径 | 说明 |
|:---|:---|:---|
| kptsim 关键点 GT | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim/` | 50 ep，SAPIEN 提取（2026-08-10） |
| LeRobot 主数据 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three/` | v2.1，50 ep |
| 归一化 stats 源 | `/home/luogang/SRC/Robot/GeoPredict/ckpts/robotwin_norm_stats.json` | 14 维 z-score |
| 注入脚本 | `util_scripts/inject_kptsim_keypoints.py` | 文档 §12.4 已落地 |
| factory.py None 防御 | `src/lerobot/datasets/factory.py` L419-420 | 文档 §13.1 **已存在** |

---

## Phase 1 — 数据注入（方案 A：体素坐标原样）

### 操作

```bash
cd /home/luogang/SRC/Robot/itvlaGp
conda activate itvlaGp

python util_scripts/inject_kptsim_keypoints.py \
  --source /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three \
  --kptsim_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim \
  --dest /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrb \
  --norm_stats_path /home/luogang/SRC/Robot/GeoPredict/ckpts/robotwin_norm_stats.json \
  --coord_mode voxel \
  --force
```

### 结果

- 50 episodes，23,550 frames，耗时 ~1.4s
- XYZ 范围: min `[0.405, 0.365, 0.253]`，max `[1.195, 1.235, 0.747]`
- 输出: `stack_bowls_three_kptsim_lrb/`（含 `observation.keypoint_3d`、`meta/stats.json`、`norm_stat.json`）

### 文件变更

| 路径 | 操作 | 原因 |
|:---|:---|:---|
| `.../stack_bowls_three_kptsim_lrb/` | **新增**（rsync 复制 + 注入） | 自包含 LeRobot 数据集 |
| `.../stack_bowls_three_kptsim_lrb/data/chunk-000/episode_*.parquet` | 修改 | 新增列 `observation.keypoint_3d [42]` |
| `.../stack_bowls_three_kptsim_lrb/meta/info.json` | 修改 | 声明 feature + `keypoint_coord_mode=voxel` |
| `.../stack_bowls_three_kptsim_lrb/meta/stats.json` | 新增 | 键名重映射后的归一化统计 |
| `.../stack_bowls_three_kptsim_lrb/norm_stat.json` | 新增 | 与 stats.json 同内容，供 CLI 引用 |
| `.../stack_bowls_three_kptsim_lrb/meta/keypoints_meta.json` | 新增 | kptsim 溯源 |

---

## Phase 2 — Layer 1 静态验收

一次性 Python 脚本运行 6 项检查（文档 §14.2），**全部 PASS**：

| Check | 内容 | 结果 |
|:---:|:---|:---:|
| 1 | info.json feature 声明（float32, shape=[42], coord_mode=voxel） | PASS |
| 2 | 50/50 episode 行数对齐 + npy 值 decimal=6 精确匹配 | PASS |
| 3 | 值域在 `[0, 1.6]^3` 内 | PASS |
| 4 | norm_stat.json 键名为 `observation.state`/`action`（非 state/actions） | PASS |
| 5 | meta/keypoints_meta.json 溯源（K=14, fl_eef_tcp） | PASS |
| 6 | 原列完整，state dim=14 | PASS |

---

## Phase 3 — LeRobot v2.1 → v3.0 转换

### Error 1: BackwardCompatibilityError（Layer 2 首次失败）

- **现象**: `LeRobotDataset('robotwin/stack_bowls_three_kptsim')` 报 v2.1 不兼容 v3.0
- **根因**: 注入输出 `codebase_version: v2.1`；itvlaGp 训练管线要求 v3.0
- **Fix**: 运行官方转换脚本

```bash
cd /home/luogang/SRC/Robot/itvlaGp
export HF_LEROBOT_HOME=/home/luogang/.cache/huggingface/lerobot

# symlink v2.1 注入数据到 HF 路径
ln -sfn /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrb \
  ${HF_LEROBOT_HOME}/robotwin/stack_bowls_three_kptsim

python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id=robotwin/stack_bowls_three_kptsim \
  --root=/home/luogang/.cache/huggingface/lerobot \
  --push-to-hub=false \
  --force-conversion
```

- **输出**: `${HF_LEROBOT_HOME}/robotwin/stack_bowls_three_kptsim_v30/`
  - parquet 合并为 `data/chunk-000/file-000.parquet`（23550 行，含 `observation.keypoint_3d`）
  - `codebase_version: v3.0`
  - `meta/stats.json` 由 episodes_stats.jsonl 重新聚合（含 observation.state/action）

### 后续 symlink 更新

```bash
ln -sfn /home/luogang/.cache/huggingface/lerobot/robotwin/stack_bowls_three_kptsim_v30 \
  ${HF_LEROBOT_HOME}/robotwin/stack_bowls_three_kptsim

# 复制溯源文件到 v30 目录
cp .../stack_bowls_three_kptsim_lrb/meta/keypoints_meta.json \
   .../stack_bowls_three_kptsim_v30/meta/
cp .../stack_bowls_three_kptsim_lrb/norm_stat.json \
   .../stack_bowls_three_kptsim_v30/
```

### Layer 2 复测 — PASS

```
num_episodes: 50  num_frames: 23550
stats keys: ['observation.state', 'action', ...]
keypoint_3d shape: torch.Size([42])
```

> **注**: 首次 Layer 2 加载时出现 torchcodec/libnvrtc 警告与 zero-video fallback，不影响 parquet/keypoint 字段验证。

---

## Phase 4 — 模型权重下载

本机原先无 InternVLA-A1.5-base 与 GeoPredict_robocasa.pth，从 HuggingFace 下载：

```bash
mkdir -p /home/luogang/SRC/Robot/itvlaGp/ckpts

python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
hf_hub_download("Jingjing0601/GeoPredict-Robocasa", "GeoPredict_robocasa.pth",
                local_dir="/home/luogang/SRC/Robot/itvlaGp/ckpts")
snapshot_download("InternRobotics/InternVLA-A1.5-base",
                  local_dir="/home/luogang/SRC/Robot/itvlaGp/ckpts/InternVLA-A1.5-base")
PY
```

| 文件 | 大小 | 路径 |
|:---|:---:|:---|
| GeoPredict_robocasa.pth | 6.1G | `ckpts/GeoPredict_robocasa.pth` |
| InternVLA-A1.5-base | 5.1G | `ckpts/InternVLA-A1.5-base/model.safetensors` |

---

## Phase 5 — Layer 3 Smoke 训练

### Error 2: torchvision.io.VideoReader 不存在（首次训练失败）

- **现象**: `--dataset.video_backend=pyav` 仍报 `AttributeError: module 'torchvision.io' has no attribute 'VideoReader'`
- **根因**: torchvision **0.26.0** 移除了 `VideoReader`；原 `decode_video_frames_torchvision` 依赖该 API
- **连带现象**: 视频 decode fallback 产生错误 shape `(5, 640, 3, 480)` 零张量 → Qwen image processor 报 `Unable to infer channel dimension format`
- **Fix**: 在 `src/lerobot/datasets/video_utils.py` 新增 `decode_video_frames_pyav()`，用 PyAV 直接解码 AV1 mp4；`backend=="pyav"` 时走新路径

> torchcodec 加载失败（`libcudart.so.13` / `libnvrtc.so.13`）是 Layer 2 的另一条警告线；完整根因、GPU 修复步骤见 [附录 A](#附录-a--torchcodec-视频解码问题方案与建议)。Smoke 阶段临时用 pyav；**正式长训应尽量切回 torchcodec GPU 解码**。

**修改文件**: `src/lerobot/datasets/video_utils.py`

- 新增函数 `decode_video_frames_pyav`（seek + decode + 最近邻时间戳匹配 + NHWC→NCHW）
- `decode_video_frames()` 中 `pyav` 分支改为调用新函数（`video_reader` 仍走旧 torchvision 路径）

**验证 PyAV 单独解码**:

```bash
python -c "
import av, torch
from pathlib import Path
p = Path('.../stack_bowls_three_kptsim_v30/videos/.../file-000.mp4')
# ... decode 0-2s → 31 frames, shape [480,640,3] OK
"
```

### Smoke 训练命令（100 step，成功）

```bash
cd /home/luogang/SRC/Robot/itvlaGp
export HF_LEROBOT_HOME=/home/luogang/.cache/huggingface/lerobot
export WANDB_MODE=offline

PRETRAINED_PATH=/home/luogang/SRC/Robot/itvlaGp/ckpts/InternVLA-A1.5-base
GEOPREDICT_CKPT=/home/luogang/SRC/Robot/itvlaGp/ckpts/GeoPredict_robocasa.pth
NORM_STATS=/home/luogang/.cache/huggingface/lerobot/robotwin/stack_bowls_three_kptsim_v30/norm_stat.json

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir=outputs/internvla_a1_5/smoke_kptsim_voxel_100step_v2 \
  --policy.type=internvla_a1_5 \
  --policy.pretrained_path="${PRETRAINED_PATH}" \
  --policy.train_expert_only=true \
  --policy.action_loss_only=true \
  --policy.enable_vqa_loss=false \
  --policy.enable_keypoint_predictor=true \
  --policy.num_keypoint_joints=14 \
  --policy.kpt_loss_weight=10.0 \
  --policy.kpt_future_loss_weight=2.0 \
  --policy.init_kpt_expert_from_action=true \
  --policy.geopredict_checkpoint_path="${GEOPREDICT_CKPT}" \
  --dataset.repo_id=robotwin/stack_bowls_three_kptsim \
  --dataset.enable_keypoint_predictor=true \
  --dataset.num_keypoint_joints=14 \
  --dataset.tokenize_state=true \
  --dataset.use_external_stats=true \
  --dataset.external_stats_path="${NORM_STATS}" \
  --dataset.video_backend=pyav \
  --batch_size=2 --steps=100 --log_freq=10 \
  --wandb.enable=false
```

完整日志: `outputs/smoke_kptsim_voxel_100step_v2.log`

### 初始化验证

```
load_geopredict_track_encoder_weights: loaded 26 keys, skipped 2 (track_fusion_layer)
Trainable params: ~927M / Total: ~3B (VLM frozen)
```

### 训练轨迹（kpt 收敛 — 核心验收）

| Step | loss | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 14.750 | **0.4033** | 0.5110 | 0.248 | 418.3 |
| 20 | 6.311 | 0.1358 | 0.2270 | 0.207 | 171.7 |
| 30 | 3.385 | 0.0452 | 0.1242 | 0.224 | 93.2 |
| 50 | 2.008 | 0.0146 | 0.0693 | 0.237 | 55.3 |
| 70 | 1.397 | 0.0070 | 0.0492 | 0.172 | 39.9 |
| **100** | **1.147** | **0.0042** | **0.0382** | **0.170** | **29.8** |

**收敛判据（文档 §14.5）**:

| 判据 | 预期 | 实测 |
|:---|:---|:---:|
| step 10 `loss_kpt_current > 0` | > 0 | ✅ 0.4033 |
| step 50 明显低于 step 10 | ↓ | ✅ 0.0146 vs 0.4033 |
| 无 NaN/OOM | 正常完成 | ✅ exit 0 |
| TrackEncoder init | loaded N keys | ✅ 26 keys |

Checkpoint: `outputs/internvla_a1_5/smoke_kptsim_voxel_100step_v2/checkpoints/000100/pretrained_model`

---

## 关键路径汇总

| 用途 | 路径 |
|:---|:---|
| kptsim GT（只读） | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim/` |
| 注入 v2.1 数据 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrb/` |
| 训练用 v3.0 数据 | `/home/luogang/.cache/huggingface/lerobot/robotwin/stack_bowls_three_kptsim_v30/` |
| HF symlink（训练 repo_id） | `${HF_LEROBOT_HOME}/robotwin/stack_bowls_three_kptsim` → v30 |
| InternVLA-A1.5-base | `/home/luogang/SRC/Robot/itvlaGp/ckpts/InternVLA-A1.5-base/` |
| GeoPredict TrackEncoder | `/home/luogang/SRC/Robot/itvlaGp/ckpts/GeoPredict_robocasa.pth` |
| norm_stat（训练 CLI） | `.../stack_bowls_three_kptsim_v30/norm_stat.json` |
| Smoke 输出 | `outputs/internvla_a1_5/smoke_kptsim_voxel_100step_v2/` |

---

## 代码变更清单

| 文件 | 变更 | 必要性 |
|:---|:---|:---:|
| `util_scripts/inject_kptsim_keypoints.py` | 已存在，本次直接运行 | 数据注入 |
| `src/lerobot/datasets/factory.py` | L419-420 None 防御（改动前已存在） | 防御性 |
| `src/lerobot/datasets/video_utils.py` | **新增** `decode_video_frames_pyav` | **必需**（torchvision 0.26 兼容） |

> **建议后续**: 将 v3.0 转换步骤写入 `inject_kptsim_keypoints.py` 的 `--to-v30` 选项，或更新 wrmup 文档 §14.6 执行顺序。

---

## 未执行项 / 后续建议

1. **正式 Warmup 400 step（8×GPU）**: Smoke 100 step 已显示 kpt loss 饱和趋势；可按文档 §7 启动全量 Warmup。
2. ~~**v30 数据持久化**~~: ✅ 已完成 → `stack_bowls_three_kptsim_lrbv30/`，自包含复验 PASS（见 Phase 6）。
3. **torchcodec GPU 解码修复（建议）**: 当前用 PyAV direct 绕过；正式长训前建议按 [附录 A](#附录-a--torchcodec-视频解码问题方案与建议) 换 `0.11.1+cu128` wheel，尽量启用 torchcodec GPU 解码。
4. **factory.py root 语义**: `--dataset.root` 直接指向数据集根目录时与 `find_info_json_path_for_repo` 不一致；自包含训练建议 `HF_LEROBOT_HOME=父目录` + `repo_id=目录名`。
5. **推理对齐**: 部署前按 wrmup 文档 §10 更新 `evaluation/RoboTwin/inference.py` 运行时关键点提取（体素坐标 + fl_eef_tcp）。

---

## 结论

**方案 A（kptsim 体素坐标原样注入）已端到端跑通**：数据注入 → v3.0 转换 → Dataset 加载 → 单卡 100 step 训练，`loss_kpt_current` 从 0.40 降至 0.004，**稳定收敛，无 NaN**。训练代码除 **video_utils.py PyAV 兼容补丁** 外无需改动 keypoint 相关模块。

---

## Phase 6 — v30 持久化 + 自包含目录单卡复验

> 目标：将 HF cache 中的 v30 复制到 share 盘，**仅依赖** `stack_bowls_three_kptsim_lrbv30/` 及其内 `norm_stat.json` 完成单卡 smoke 训练至 kpt loss 收敛。

### 6.1 复制 v30 → lrbv30

```bash
SRC=/home/luogang/.cache/huggingface/lerobot/robotwin/stack_bowls_three_kptsim_v30
DEST=/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrbv30

rsync -a "$SRC/" "$DEST/"
```

| 检查项 | 结果 |
|:---|:---:|
| 总大小 | 163M |
| 内部 symlink | 无 |
| `data/chunk-000/file-000.parquet` | ✅ |
| `meta/info.json` (v3.0, voxel) | ✅ |
| `norm_stat.json` | ✅ |
| `meta/keypoints_meta.json` | ✅ |

### 6.2 Dataset 加载复验（仅 root 指向 lrbv30）

```python
LeRobotDataset(repo_id="robotwin/stack_bowls_three_kptsim",
               root="/home/luogang/.../stack_bowls_three_kptsim_lrbv30")
# → 50 ep, 23550 frames, keypoint_3d [42] OK
```

> 注：直接 `root=lrbv30` 时 LeRobotDataset 可加载；但训练管线 `factory.py` 的 `find_info_json_path_for_repo` 会拼 `root/repo_id/meta/info.json`，与 LeRobotDataset 语义不一致（后者 root 即数据集根目录）。见 §6.3。

### 6.3 训练命令调试（3 次尝试）

| 尝试 | 配置 | 结果 | 原因 |
|:---:|:---|:---:|:---|
| 1 | `--dataset.root=lrbv30`（无 type） | ❌ | draccus 缺 `dataset.type` |
| 2 | `+ dataset.type` + `root=lrbv30` | ❌ | `factory.py` 找 `lrbv30/robotwin/.../info.json` |
| 3 | `HF_LEROBOT_HOME=RoboTwin-Clean` + `repo_id=stack_bowls_three_kptsim_lrbv30` | ✅ | 与 LeRobot 路径约定一致 |

**最终可用命令**（不依赖 HF cache、不依赖 v2.1/lrb/kptsim npy）：

```bash
cd /home/luogang/SRC/Robot/itvlaGp
conda activate itvlaGp

export HF_LEROBOT_HOME=/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean
DATA_ROOT="${HF_LEROBOT_HOME}/stack_bowls_three_kptsim_lrbv30"
NORM_STATS="${DATA_ROOT}/norm_stat.json"   # 数据集内 stats，非 GeoPredict 外部文件

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir=outputs/internvla_a1_5/smoke_lrbv30_selfcontained_100step \
  --policy.type=internvla_a1_5 \
  --policy.push_to_hub=false \
  --policy.pretrained_path=/home/luogang/SRC/Robot/itvlaGp/ckpts/InternVLA-A1.5-base \
  --policy.train_expert_only=true \
  --policy.action_loss_only=true \
  --policy.enable_vqa_loss=false \
  --policy.enable_keypoint_predictor=true \
  --policy.num_keypoint_joints=14 \
  --policy.kpt_loss_weight=10.0 \
  --policy.kpt_future_loss_weight=2.0 \
  --policy.init_kpt_expert_from_action=true \
  --policy.geopredict_checkpoint_path=/home/luogang/SRC/Robot/itvlaGp/ckpts/GeoPredict_robocasa.pth \
  --dataset.type=internvla_a1_5 \
  --dataset.repo_id=stack_bowls_three_kptsim_lrbv30 \
  --dataset.enable_keypoint_predictor=true \
  --dataset.num_keypoint_joints=14 \
  --dataset.tokenize_state=true \
  --dataset.use_external_stats=true \
  --dataset.external_stats_path="${NORM_STATS}" \
  --dataset.video_backend=pyav \
  --batch_size=2 --steps=100 --log_freq=10 \
  --wandb.enable=false
```

完整日志: `outputs/smoke_lrbv30_selfcontained_100step.log`

**数据依赖说明**：

- 训练数据：**仅** `stack_bowls_three_kptsim_lrbv30/`（parquet + video + meta + norm_stat）
- 归一化 stats：**仅** `lrbv30/norm_stat.json`（非 GeoPredict `robotwin_norm_stats.json`）
- 路径约定：`HF_LEROBOT_HOME` 设为 lrbv30 的**父目录**，`repo_id=stack_bowls_three_kptsim_lrbv30`
- 模型权重仍来自 `itvlaGp/ckpts/`（与数据无关）

### 6.4 训练结果 — PASS（稳定收敛）

| Step | loss | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 22.978 | **0.6110** | 0.6933 | 0.300 | 512.3 |
| 20 | 5.146 | 0.0792 | 0.1505 | 0.135 | 130.2 |
| 30 | 4.629 | 0.0650 | 0.1107 | 0.177 | 107.9 |
| 50 | 3.428 | 0.0304 | 0.0735 | 0.165 | 68.8 |
| 70 | 2.702 | 0.0174 | 0.0595 | 0.134 | 54.1 |
| **100** | **2.985** | **0.0138** | **0.0523** | **0.180** | **49.6** |

- TrackEncoder init: loaded 26 keys ✅
- exit code: 0，无 NaN/OOM ✅
- Checkpoint: `outputs/internvla_a1_5/smoke_lrbv30_selfcontained_100step/checkpoints/000100/pretrained_model`

与 Phase 5（HF cache v30）对比：step-100 `loss_kpt_cur` 0.0138 vs 0.0042，均呈单调下降并饱和，**自包含目录训练有效**。

### 6.5 关键路径更新

| 用途 | 路径 |
|:---|:---|
| **自包含训练数据（推荐）** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrbv30/` |
| norm_stat（训练 CLI） | `.../stack_bowls_three_kptsim_lrbv30/norm_stat.json` |
| Smoke 输出（lrbv30 复验） | `outputs/internvla_a1_5/smoke_lrbv30_selfcontained_100step/` |

---

## 附录 A — torchcodec 视频解码：问题、方案与建议

> **优先级提醒**：LeRobot 默认优先 `torchcodec`（`get_safe_default_codec()`）。**正式 Warmup / 长训时，应尽量修复 torchcodec 并启用 GPU 解码**（`--dataset.video_backend=torchcodec` 或省略 backend 走默认），比 PyAV CPU 解码吞吐更高、更不易成为 dataloader 瓶颈。  
> 本次 Smoke 因环境未就绪临时使用 `--dataset.video_backend=pyav`；**不能代表最终生产配置**。  
> 同类问题在本仓库 LIBERO 复现中亦有记录（[`reprd_liberop_cam_rb.md`](p/reprd_liberop_cam_rb.md) #6）。

### A.1 当前 itvlaGp 环境快照

| 组件 | 版本 / 状态 |
|:---|:---|
| Python | 3.10 |
| torch | 2.11.0+cu128（CUDA 12.8） |
| torchvision | 0.26.0+cu128 |
| torchcodec（当前） | 0.15.0（PyPI 默认 **CUDA 13** wheel）→ **import 失败** |
| ffmpeg | 8.1.2（conda-forge，shared libs ✅） |
| pyav | 15.1.0 ✅ |

### A.2 可能出现的症状

| 现象 | 含义 |
|:---|:---|
| `Could not load libtorchcodec` | 底层 `.so` 动态库加载失败 |
| `libcudart.so.13: cannot open shared object file` | torchcodec wheel 链 CUDA 13，环境只有 cu12 |
| `libnvrtc.so.13: cannot open shared object file` | 同上（Blackwell + cu128 环境常见） |
| `libavutil.so.59/58/57: cannot open shared object file` | FFmpeg SONAME 与 wheel 不匹配（次要；本机 ffmpeg 8 通常 OK） |
| 日志大量 `[video_decode_error]` + `using_zeros` | 解码静默失败 → **全黑帧喂给 VLM**（极隐蔽的数据损坏） |
| `torchvision.io` 无 `VideoReader` | torchvision 0.26 移除 API；旧 pyav 路径也会挂 |

> **危险**：`lerobot_dataset.py` 对 decode 失败会 fallback 为零张量并继续训练，loss 可能仍下降，但视觉分支学到的是退化解。长训前务必 `grep -c video_decode_error train.log` 确认为 0。

### A.3 根因（三层）

1. **wheel CUDA 变体不匹配（主因）**  
   `pip install torchcodec` 默认装 **链 CUDA 13** 的 wheel（`libcudart.so.13` / `libnvrtc.so.13`），与本机 `torch 2.11.0+cu128`（CUDA 12.8 工具链）不一致。  
   版本号 0.15 ↔ torch 2.11 虽在[官方兼容表](https://github.com/meta-pytorch/torchcodec?tab=readme-ov-file#installing-torchcodec)内，但 **PyPI 默认 wheel ≠ cu128 wheel**。

2. **LD_LIBRARY_PATH 未覆盖 FFmpeg / nvrtc**  
   即使 wheel 正确，也需让 loader 找到 `$CONDA_PREFIX/lib`（FFmpeg）和 `nvidia/cuda_nvrtc/lib`（`libnvrtc.so.12`）。

3. **torchvision 0.26 移除 VideoReader（次因，已补丁）**  
   旧代码里 `backend=pyav` 仍走 torchvision → 已在 `video_utils.py` 新增 `decode_video_frames_pyav()` 直接解码。

### A.4 方案对比

| 方案 | GPU 解码 | 改动范围 | 推荐场景 |
|:---|:---:|:---|:---|
| **A. `0.11.1+cu128` wheel（首选）** | ✅ | 仅重装 torchcodec | **要 GPU 解码 + 不破坏现有环境** |
| B. `0.15.0+cpu` wheel | ❌ CPU | 仅重装 torchcodec | 只求 import 通过、可接受 CPU 解码 |
| C. PyAV direct（当前 Smoke） | ❌ CPU | 无（已有补丁） | 临时 smoke / torchcodec 未修前 |
| D. 硬装 CUDA 13 runtime | ✅? | 全局 CUDA 库 | ❌ 易与 cu12 包冲突，不推荐 |
| E. 降级 torch 以配 PyPI wheel | — | 整个训练栈 | ❌ 破坏 InternVLA-A1.5 环境 |

> **说明**：PyTorch [cu128 index](https://download.pytorch.org/whl/cu128/torchcodec/) 目前最高到 `0.11.1+cu128`，尚无 `0.12–0.15+cu128`。在 torch 2.11+cu128 下要用 **GPU 解码**，现实选择是 **0.11.1+cu128**（官方表：0.11 ↔ 2.11）。

### A.5 推荐修复：GPU cu128 wheel（方案 A）

**只替换 torchcodec，不动 torch / flash-attn / transformers。**

已预检：`torchcodec-0.11.1+cu128` 的 `libtorchcodec_core8.so` 依赖 `libcudart.so.12`、`libnvrtc.so.12`、`libavutil.so.60`，在本机 `itvlaGp` + 正确 `LD_LIBRARY_PATH` 下均可解析。

```bash
conda activate itvlaGp

# 可选：记录当前版本以便回滚
pip show torchcodec > /tmp/torchcodec_before.txt

# 只重装为 cu128 GPU wheel（勿用 PyPI 默认源）
pip install --force-reinstall "torchcodec==0.11.1" \
  --index-url https://download.pytorch.org/whl/cu128

# 运行时库路径（建议写入 launch 脚本）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH:-}
```

**验证 import**：

```bash
python -c "import torchcodec; print('torchcodec OK')"
```

**验证真实解码（非全零）**：

```python
from torchcodec.decoders import VideoDecoder
p = "/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/stack_bowls_three_kptsim_lrbv30/videos/observation.images.cam_high/chunk-000/file-000.mp4"
d = VideoDecoder(p)
f = d.get_frames_played_at(seconds=[0.0])
print(f.data.shape, float(f.data.float().mean()))  # mean 应 >> 0
```

**训练配置**：去掉 `--dataset.video_backend=pyav`，或显式：

```bash
--dataset.video_backend=torchcodec
```

**长训前检查**：

```bash
grep -c video_decode_error outputs/your_train.log   # 应为 0
grep -c using_zeros outputs/your_train.log           # 应为 0
```

**回滚**：

```bash
pip install --force-reinstall torchcodec==0.15.0   # 回到 PyPI 默认（会再次 broken）
# 或继续 --dataset.video_backend=pyav
```

### A.6 备选：CPU torchcodec 0.15（方案 B）

若暂不接受 torchcodec 0.11.1 小版本回退，可装 CPU wheel（**训练仍用 GPU，仅视频解码在 CPU**）：

```bash
pip install --force-reinstall "torchcodec==0.15.0" \
  --index-url https://download.pytorch.org/whl/cpu
```

参考：[`reprd_liberop_cam_rb.md`](p/reprd_liberop_cam_rb.md) #6（`torch 2.10 + torchcodec 0.10+cpu` 成功案例）。

### A.7 当前临时方案：PyAV direct（方案 C）

Phase 5 已在 [`video_utils.py`](../../src/lerobot/datasets/video_utils.py) 新增 `decode_video_frames_pyav()`；Smoke 命令使用：

```bash
--dataset.video_backend=pyav
```

- ✅ 不依赖 torchcodec，Smoke 已跑通  
- ⚠️ CPU 解码，长训可能成为瓶颈；**正式 Warmup 前仍建议切回 torchcodec GPU**

### A.8 不建议的操作

- 为凑 PyPI `torchcodec 0.15` 硬装 **CUDA 13 runtime**（与现有 `nvidia-*-cu12` 冲突风险高）
- 为 torchcodec **降级 torch**（破坏 InternVLA-A1.5 依赖链）
- 不检查日志直接长训（零帧 fallback 极隐蔽）

### A.9 launch 脚本建议（防复发）

```bash
# 安装时指定 index，避免 pip 默认 cu13 wheel
pip install "torchcodec==0.11.1" --index-url https://download.pytorch.org/whl/cu128

# 每 job 导出（可与现有 launch 脚本合并）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:\
${LD_LIBRARY_PATH:-}

# 训练默认 backend（优先 torchcodec GPU）
# --dataset.video_backend=torchcodec
```

---

*日志版本: wrmup1G-v1.2 | 2026-08-11*
