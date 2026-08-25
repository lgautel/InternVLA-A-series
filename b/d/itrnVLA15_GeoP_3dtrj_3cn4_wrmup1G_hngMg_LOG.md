# 方案 A 落地实施日志 — `hanging_mug`（1G 正式 Warmup 400 step）

> 对应手册: [`itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_hngMg.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_hngMg.md)  
> 实施日期: 2026-08-25  
> 环境: conda `itvlaGp`，**GPU 1**（NVIDIA RTX PRO 6000 Blackwell，97 GB）  
> 并行任务: `scan_object` 在 GPU 0 同时执行，见 [`itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_scnObj_LOG.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_scnObj_LOG.md)

---

## 总览

| 阶段 | 状态 |
|:---|:---:|
| Phase 0 前置检查 | ✅ |
| Phase 1 norm stats + 注入 | ✅ |
| Phase 2 Layer 1 静态验收（6 项） | ✅ |
| Phase 3 v2.1→v3.0 + lrbv30 | ✅ |
| Phase 4 权重（复用已有 ckpt） | ✅ |
| Phase 5 正式 Warmup 400 step（GPU 1） | ✅ 收敛 |
| Layer 2 Dataset 加载 | ✅ |

---

## Phase 0 — 前置检查

### 环境快照

| 项 | 值 |
|:---|:---|
| conda | `itvlaGp`（Python 3.10.20） |
| GPU | 1 × RTX PRO 6000 Blackwell，97887 MiB |
| 训练时段 | 2026-08-25 08:26–08:34 UTC+8 |
| `HF_LEROBOT_HOME` | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean` |

### 已有资产

| 资产 | 路径 | 说明 |
|:---|:---|:---|
| kptsim GT | `.../hanging_mug_kptsim/` | 50 ep，SAPIEN FK（2026-08-25，见 GeoPredict LOG） |
| LeRobot 主数据 | `.../hanging_mug/` | v2.1，50 ep，**16889 frames** |
| 注入脚本 | `util_scripts/inject_kptsim_keypoints.py` | 方案 A voxel |
| InternVLA base | `itvlaGp/ckpts/InternVLA-A1.5-base/` | 5.1G，已存在 |
| GeoPredict | `itvlaGp/ckpts/GeoPredict_robocasa.pth` | 6.1G，已存在 |
| task_idx | **10** | RoboTwin `hanging_mug` |

### 任务专属参数（不可复用 stack_bowls_three / scan_object）

| 参数 | 值 |
|:---|:---|
| `coord_offset` | `[-0.772, -1.050, 0.478]` |
| norm stats | `GeoPredict/ckpts/robotwin_norm_stats_hanging_mug.json` |
| repo_id | `hanging_mug_kptsim_lrbv30` |

---

## Phase 1 — norm stats + 注入

### 1.1 计算任务专属 norm stats

```bash
cd /home/luogang/SRC/Robot/GeoPredict
python tools/compute_robotwin_norm_stats.py \
  --dataset_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug \
  --output ./ckpts/robotwin_norm_stats_hanging_mug.json
```

- 输出: `GeoPredict/ckpts/robotwin_norm_stats_hanging_mug.json`（3.0K，2026-08-25 08:25）

### 1.2 方案 A 注入（体素坐标原样）

```bash
cd /home/luogang/SRC/Robot/itvlaGp
conda activate itvlaGp

python util_scripts/inject_kptsim_keypoints.py \
  --source /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug \
  --kptsim_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim \
  --dest /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrb \
  --norm_stats_path /home/luogang/SRC/Robot/GeoPredict/ckpts/robotwin_norm_stats_hanging_mug.json \
  --coord_mode voxel \
  --force
```

### 结果

| 项 | 值 |
|:---|:---|
| episodes | 50 |
| frames | **16889** |
| XYZ min | `[0.422, 0.392, 0.185]` |
| XYZ max | `[1.178, 1.208, 0.815]` |
| 产物 | `hanging_mug_kptsim_lrb/`（v2.1） |

### 文件变更

| 路径 | 操作 |
|:---|:---|
| `.../hanging_mug_kptsim_lrb/` | **新增** |
| `.../hanging_mug_kptsim_lrb/data/chunk-000/episode_*.parquet` | 新增列 `observation.keypoint_3d [42]` |
| `.../hanging_mug_kptsim_lrb/meta/info.json` | `keypoint_coord_mode=voxel` |
| `.../hanging_mug_kptsim_lrb/norm_stat.json` | 新增 |
| `.../hanging_mug_kptsim_lrb/meta/keypoints_meta.json` | 新增（含 coord_offset） |

---

## Phase 2 — Layer 1 静态验收

运行手册 §5 六项检查，**全部 PASS**（`validate_all` 无 ep42 类 caveat）：

| Check | 内容 | 结果 |
|:---:|:---|:---:|
| 1 | info.json feature（float32, [42], voxel） | PASS |
| 2 | 50/50 episode 行数对齐 + npy decimal=6 匹配 | PASS |
| 3 | 值域在 `[0, 1.6]^3` 内 | PASS |
| 4 | norm_stat.json 键名 `observation.state`/`action` | PASS |
| 5 | keypoints_meta.json 溯源（K=14, fl_eef_tcp） | PASS |
| 6 | 原列完整，state dim=14 | PASS |

---

## Phase 3 — v2.1→v3.0 + lrbv30

```bash
export HF_LEROBOT_HOME=~/.cache/huggingface/lerobot
ln -sfn /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrb \
  ${HF_LEROBOT_HOME}/robotwin/hanging_mug_kptsim

cd /home/luogang/SRC/Robot/itvlaGp
python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id=robotwin/hanging_mug_kptsim \
  --root=${HF_LEROBOT_HOME} \
  --push-to-hub=false \
  --force-conversion

cp .../hanging_mug_kptsim_lrb/meta/keypoints_meta.json \
   .../hanging_mug_kptsim_v30/meta/
cp .../hanging_mug_kptsim_lrb/norm_stat.json \
   .../hanging_mug_kptsim_v30/

rsync -a ${HF_LEROBOT_HOME}/robotwin/hanging_mug_kptsim_v30/ \
  /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrbv30/
```

### 结果

| 检查项 | 结果 |
|:---|:---:|
| 总大小 | 121M |
| `codebase_version` | v3.0 |
| `total_frames` | 16889 |
| `norm_stat.json` | ✅ |
| `meta/keypoints_meta.json` | ✅ |

### Layer 2 Dataset 加载 — PASS

```python
# HF_LEROBOT_HOME=RoboTwin-Clean, repo_id=hanging_mug_kptsim_lrbv30
# num_episodes=50, num_frames=16889, keypoint_3d shape=(42,)
```

> Layer 2 测试时 torchcodec 报 `libnvrtc.so.13` 缺失；训练显式指定 `--dataset.video_backend=pyav` 规避。

---

## Phase 4 — 模型权重

复用已有权重，未重新下载：

| 文件 | 大小 | 路径 |
|:---|:---:|:---|
| GeoPredict_robocasa.pth | 6.1G | `itvlaGp/ckpts/GeoPredict_robocasa.pth` |
| InternVLA-A1.5-base | 5.1G | `itvlaGp/ckpts/InternVLA-A1.5-base/model.safetensors` |

---

## Phase 5 — 正式 Warmup 400 step（GPU 1）

### 超参

| 参数 | 值 |
|:---|:---|
| `CUDA_VISIBLE_DEVICES` | **1** |
| `batch_size` | **16**（未 OOM） |
| `steps` | **400** |
| `save_freq` / `log_freq` | 100 / 10 |
| `scheduler_warmup_steps` | 50 |
| `scheduler_decay_steps` | 400 |
| `video_backend` | `pyav` |
| Loss 权重 | action=2.0, kpt=10.0, kpt_future=2.0 |
| 关键 flag | `enable_keypoint_predictor` policy+dataset 双 true；`init_kpt_expert_from_action=true`；`action_loss_only=true` |

### 训练命令

```bash
cd /home/luogang/SRC/Robot/itvlaGp
conda activate itvlaGp

export HF_LEROBOT_HOME=/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean
export WANDB_MODE=offline USE_LIBUV=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

PRETRAINED_PATH=/home/luogang/SRC/Robot/itvlaGp/ckpts/InternVLA-A1.5-base
GEOPREDICT_CKPT=/home/luogang/SRC/Robot/itvlaGp/ckpts/GeoPredict_robocasa.pth
CLEAN=/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean

CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes=1 \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir=outputs/internvla_a1_5/warmup_hanging_mug_kptsim_400step \
  --policy.type=internvla_a1_5 \
  --policy.push_to_hub=false \
  --policy.dtype=bfloat16 \
  --policy.optimizer_lr=5e-5 \
  --policy.scheduler_warmup_steps=50 \
  --policy.scheduler_decay_steps=400 \
  --policy.scheduler_decay_lr=5e-6 \
  --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B \
  --policy.pretrained_path="${PRETRAINED_PATH}" \
  --policy.train_expert_only=true \
  --policy.action_loss_only=true \
  --policy.enable_vqa_loss=false \
  --policy.tokenize_state=true \
  --policy.freeze_learnable_tokens=true \
  --policy.enable_keypoint_predictor=true \
  --policy.num_keypoint_joints=14 \
  --policy.action_loss_weight=2.0 \
  --policy.kpt_loss_weight=10.0 \
  --policy.kpt_future_loss_weight=2.0 \
  --policy.knowledge_insulation=true \
  --policy.knowledge_insulation_kpt=true \
  --policy.kpt_to_action_detach=false \
  --policy.action_expert_lr_scale=0.04 \
  --policy.kpt_expert_lr_scale=1.0 \
  --policy.track_encoder_lr_scale=1.0 \
  --policy.init_kpt_expert_from_action=true \
  --policy.geopredict_checkpoint_path="${GEOPREDICT_CKPT}" \
  --dataset.type=internvla_a1_5 \
  --dataset.repo_id=hanging_mug_kptsim_lrbv30 \
  --dataset.enable_keypoint_predictor=true \
  --dataset.num_keypoint_joints=14 \
  --dataset.action_mode=abs \
  --dataset.tokenize_state=true \
  --dataset.use_fast_action_tokens=true \
  --dataset.use_external_stats=true \
  --dataset.external_stats_path="${CLEAN}/hanging_mug_kptsim_lrbv30/norm_stat.json" \
  --dataset.video_backend=pyav \
  --seed=42 --batch_size=16 --steps=400 --save_freq=100 --log_freq=10 --num_workers=8 \
  --wandb.enable=false \
  > outputs/internvla_a1_5/warmup_hanging_mug_kptsim_400step.log 2>&1
```

完整日志: `outputs/internvla_a1_5/warmup_hanging_mug_kptsim_400step.log`

### 初始化验证

```
post_init_keypoint_weights: initialized keypoint_expert from action_expert weights.
load_geopredict_keypoint_weights: loaded 26 TrackEncoder keys (skipped 2, e.g. track_fusion_layer)
Trainable params: 927M / Total: 3B
num_frames=16889, num_episodes=50, effective batch_size=16
```

### 训练轨迹（每 10 step）

| Step | loss | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 20.662 | **0.5639** | 0.7244 | 0.267 | 550.2 |
| 20 | 6.013 | 0.1339 | 0.2092 | 0.244 | 188.1 |
| 30 | 3.704 | 0.0467 | 0.1400 | 0.218 | 93.4 |
| 40 | 2.874 | 0.0308 | 0.1073 | 0.210 | 67.8 |
| 50 | 2.110 | 0.0160 | 0.0792 | 0.182 | 48.4 |
| 60 | 1.658 | 0.0099 | 0.0587 | 0.192 | 44.2 |
| 70 | 1.270 | 0.0067 | 0.0431 | 0.170 | 34.2 |
| 80 | 0.990 | 0.0053 | 0.0307 | 0.162 | 28.5 |
| 90 | 0.813 | 0.0051 | 0.0234 | 0.147 | 27.1 |
| 100 | 0.664 | 0.0048 | 0.0175 | 0.133 | 24.1 |
| 110 | 0.596 | 0.0044 | 0.0133 | 0.143 | 22.1 |
| 120 | 0.552 | 0.0041 | 0.0105 | 0.150 | 22.7 |
| 130 | 0.486 | 0.0035 | 0.0089 | 0.136 | 18.9 |
| 140 | 0.426 | 0.0035 | 0.0079 | 0.117 | 17.0 |
| 150 | 0.427 | 0.0037 | 0.0072 | 0.123 | 17.2 |
| 160 | 0.389 | 0.0031 | 0.0061 | 0.118 | 14.1 |
| 170 | 0.409 | 0.0034 | 0.0058 | 0.129 | 15.1 |
| 180 | 0.385 | 0.0033 | 0.0058 | 0.118 | 14.4 |
| 190 | 0.356 | 0.0028 | 0.0049 | 0.115 | 13.1 |
| 200 | 0.381 | 0.0031 | 0.0052 | 0.123 | 13.8 |
| 210 | 0.348 | 0.0026 | 0.0044 | 0.117 | 11.1 |
| 220 | 0.343 | 0.0026 | 0.0045 | 0.114 | 10.9 |
| 230 | 0.372 | 0.0029 | 0.0046 | 0.126 | 11.4 |
| 240 | 0.327 | 0.0026 | 0.0042 | 0.109 | 10.0 |
| 250 | 0.353 | 0.0028 | 0.0042 | 0.121 | 11.5 |
| 260 | 0.319 | 0.0023 | 0.0038 | 0.110 | 8.0 |
| 270 | 0.329 | 0.0023 | 0.0036 | 0.117 | 9.4 |
| 280 | 0.335 | 0.0026 | 0.0042 | 0.113 | 9.1 |
| 290 | 0.332 | 0.0025 | 0.0037 | 0.116 | 8.5 |
| **300** | **0.284** | **0.0024** | **0.0033** | **0.097** | **8.2** |
| 310 | 0.340 | 0.0023 | 0.0038 | 0.121 | 7.9 |
| 320 | 0.335 | 0.0023 | 0.0033 | 0.123 | 7.5 |
| 330 | 0.352 | 0.0023 | 0.0038 | 0.126 | 7.2 |
| 340 | 0.338 | 0.0021 | 0.0035 | 0.124 | 6.5 |
| 350 | 0.322 | 0.0023 | 0.0038 | 0.112 | 5.7 |
| 360 | 0.331 | 0.0026 | 0.0039 | 0.113 | 6.2 |
| 370 | 0.313 | 0.0023 | 0.0037 | 0.109 | 6.5 |
| **380** | **0.296** | **0.0021** | **0.0032** | **0.105** | **6.1** |
| 390 | 0.287 | 0.0021 | 0.0031 | 0.102 | 5.6 |
| 400 | 0.311 | 0.0022 | 0.0034 | 0.110 | 5.4 |

### 验收判据

| 判据 | 预期 | 实测 |
|:---|:---|:---:|
| TrackEncoder init | loaded ~26 keys | ✅ 26 keys |
| step 10 `loss_kpt_current > 0` | > 0 | ✅ 0.5639 |
| kpt loss 明显下降 | step 300 << step 10 | ✅ 0.0024 vs 0.5639 |
| `video_decode_error` | 0 | ✅ 0 |
| `using_zeros` | 0 | ✅ 0 |
| 无 NaN/OOM | 正常完成 | ✅ MUG_EXIT:0 |
| 训练耗时 | — | ~4m50s（纯训练步） |

### Checkpoint

| Step | 路径 | 备注 |
|:---:|:---|:---|
| 100 | `outputs/.../checkpoints/000100/pretrained_model` | |
| 200 | `outputs/.../checkpoints/000200/pretrained_model` | |
| **300** | `outputs/.../checkpoints/000300/pretrained_model` | **推荐 Phase 2 起点**（loss 最低区间） |
| 380 | — | `loss_kpt_cur` 最低 0.0021 |
| 400 | `outputs/.../checkpoints/000400/pretrained_model` | 最终 checkpoint |

---

## 错误与 Fix

| 现象 | 根因 | 修复 |
|:---|:---|:---|
| Layer 2 torchcodec 警告 | `libnvrtc.so.13` 缺失 | 训练用 `--dataset.video_backend=pyav` |

本次训练 **无 OOM**（BS=16），**无数据兼容性错误**，与 `scan_object` 并行执行互不干扰（独立 `repo_id` 与 GPU）。

---

## 关键路径汇总

| 用途 | 路径 |
|:---|:---|
| kptsim GT（只读） | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim/` |
| 注入 v2.1 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrb/` |
| **训练用 v3.0（自包含）** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrbv30/` |
| norm_stat（训练 CLI） | `.../hanging_mug_kptsim_lrbv30/norm_stat.json` |
| 原始 norm stats | `GeoPredict/ckpts/robotwin_norm_stats_hanging_mug.json` |
| InternVLA-A1.5-base | `itvlaGp/ckpts/InternVLA-A1.5-base/` |
| GeoPredict TrackEncoder | `itvlaGp/ckpts/GeoPredict_robocasa.pth` |
| 训练输出 | `outputs/internvla_a1_5/warmup_hanging_mug_kptsim_400step/` |
| 训练日志 | `outputs/internvla_a1_5/warmup_hanging_mug_kptsim_400step.log` |
| **推荐 checkpoint** | `outputs/.../checkpoints/000300/pretrained_model` |

---

## 结论

**`hanging_mug` 方案 A 端到端跑通**：数据注入 → v3.0 → 单卡 400 step 正式 Warmup，`loss_kpt_current` 从 0.56 降至 ~0.002 并饱和，**稳定收敛，无 NaN/OOM，视频解码零错误**。推荐以 **step 300** checkpoint 作为 Phase 2 微调起点。

---

*日志版本: wrmup1G-hngMg-v1.0 | 2026-08-25*
