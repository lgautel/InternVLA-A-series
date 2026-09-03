# 方案 A 落地实施日志 — `scan_object`（1G 正式 Warmup 400 step）

> 对应手册: [`itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_scnObj.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_scnObj.md)  
> 实施日期: 2026-08-25  
> 环境: conda `itvlaGp`，**GPU 0**（NVIDIA RTX PRO 6000 Blackwell，97 GB）  
> 并行任务: `hanging_mug` 在 GPU 1 同时执行，见 [`itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_hngMg_LOG.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_hngMg_LOG.md)

---

## 总览

| 阶段 | 状态 |
|:---|:---:|
| Phase 0 前置检查 | ✅ |
| Phase 1 norm stats + 注入 | ✅ |
| Phase 2 Layer 1 静态验收（6 项） | ✅ |
| Phase 3 v2.1→v3.0 + lrbv30 | ✅ |
| Phase 4 权重（复用已有 ckpt） | ✅ |
| Phase 5 正式 Warmup 400 step（GPU 0） | ✅ 收敛 |
| Layer 2 Dataset 加载 | ✅ |

---

## Phase 0 — 前置检查

### 环境快照

| 项 | 值 |
|:---|:---|
| conda | `itvlaGp`（Python 3.10.20） |
| GPU | 0 × RTX PRO 6000 Blackwell，97887 MiB |
| 训练时段 | 2026-08-25 08:26–08:34 UTC+8 |
| `HF_LEROBOT_HOME` | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean` |

### 已有资产

| 资产 | 路径 | 说明 |
|:---|:---|:---|
| kptsim GT | `.../scan_object_kptsim/` | 50 ep，SAPIEN FK（2026-08-25，见 GeoPredict LOG） |
| LeRobot 主数据 | `.../scan_object/` | v2.1，50 ep，**8463 frames** |
| 注入脚本 | `util_scripts/inject_kptsim_keypoints.py` | 方案 A voxel |
| InternVLA base | `itvlaGp/ckpts/InternVLA-A1.5-base/` | 5.1G，已存在 |
| GeoPredict | `itvlaGp/ckpts/GeoPredict_robocasa.pth` | 6.1G，已存在 |
| task_idx | **41** | RoboTwin `scan_object` |

### 任务专属参数（不可复用 stack_bowls_three）

| 参数 | 值 |
|:---|:---|
| `coord_offset` | `[-0.675, -1.035, 0.622]` |
| norm stats | `GeoPredict/ckpts/robotwin_norm_stats_scan_object.json` |
| repo_id | `scan_object_kptsim_lrbv30` |

---

## Phase 1 — norm stats + 注入

### 1.1 计算任务专属 norm stats

```bash
cd /home/luogang/SRC/Robot/GeoPredict
python tools/compute_robotwin_norm_stats.py \
  --dataset_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object \
  --output ./ckpts/robotwin_norm_stats_scan_object.json
```

- 输出: `GeoPredict/ckpts/robotwin_norm_stats_scan_object.json`（2.9K，2026-08-25 08:25）

### 1.2 方案 A 注入（体素坐标原样）

```bash
cd /home/luogang/SRC/Robot/itvlaGp
conda activate itvlaGp

python util_scripts/inject_kptsim_keypoints.py \
  --source /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object \
  --kptsim_dir /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim \
  --dest /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrb \
  --norm_stats_path /home/luogang/SRC/Robot/GeoPredict/ckpts/robotwin_norm_stats_scan_object.json \
  --coord_mode voxel \
  --force
```

### 结果

| 项 | 值 |
|:---|:---|
| episodes | 50 |
| frames | **8463** |
| XYZ min | `[0.323, 0.376, 0.157]` |
| XYZ max | `[1.277, 1.224, 0.843]` |
| 产物 | `scan_object_kptsim_lrb/`（v2.1） |

### 文件变更

| 路径 | 操作 |
|:---|:---|
| `.../scan_object_kptsim_lrb/` | **新增** |
| `.../scan_object_kptsim_lrb/data/chunk-000/episode_*.parquet` | 新增列 `observation.keypoint_3d [42]` |
| `.../scan_object_kptsim_lrb/meta/info.json` | `keypoint_coord_mode=voxel` |
| `.../scan_object_kptsim_lrb/norm_stat.json` | 新增 |
| `.../scan_object_kptsim_lrb/meta/keypoints_meta.json` | 新增（含 coord_offset） |

---

## Phase 2 — Layer 1 静态验收

运行手册 §5 六项检查，**全部 PASS**：

| Check | 内容 | 结果 |
|:---:|:---|:---:|
| 1 | info.json feature（float32, [42], voxel） | PASS |
| 2 | 50/50 episode 行数对齐 + npy decimal=6 匹配 | PASS |
| 3 | 值域在 `[0, 1.6]^3` 内 | PASS |
| 4 | norm_stat.json 键名 `observation.state`/`action` | PASS |
| 5 | keypoints_meta.json 溯源（K=14, fl_eef_tcp） | PASS |
| 6 | 原列完整，state dim=14 | PASS |

> **ep42 caveat**（手册附录 C）：episode 42 右 TCP 相邻帧位移 0.125 m，为演示轨迹特性，非 FK 错误；注入验收不以 5 cm 阈值判失败，数据仍 PASS。

---

## Phase 3 — v2.1→v3.0 + lrbv30

```bash
export HF_LEROBOT_HOME=~/.cache/huggingface/lerobot
ln -sfn /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrb \
  ${HF_LEROBOT_HOME}/robotwin/scan_object_kptsim

cd /home/luogang/SRC/Robot/itvlaGp
python src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
  --repo-id=robotwin/scan_object_kptsim \
  --root=${HF_LEROBOT_HOME} \
  --push-to-hub=false \
  --force-conversion

# 复制溯源文件
cp .../scan_object_kptsim_lrb/meta/keypoints_meta.json \
   .../scan_object_kptsim_v30/meta/
cp .../scan_object_kptsim_lrb/norm_stat.json \
   .../scan_object_kptsim_v30/

# 持久化到 share 盘
rsync -a ${HF_LEROBOT_HOME}/robotwin/scan_object_kptsim_v30/ \
  /home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/
```

### 结果

| 检查项 | 结果 |
|:---|:---:|
| 总大小 | 75M |
| `codebase_version` | v3.0 |
| `total_frames` | 8463 |
| `norm_stat.json` | ✅ |
| `meta/keypoints_meta.json` | ✅ |

### Layer 2 Dataset 加载 — PASS

```python
# HF_LEROBOT_HOME=RoboTwin-Clean, repo_id=scan_object_kptsim_lrbv30
# num_episodes=50, num_frames=8463, keypoint_3d shape=(42,)
```

> Layer 2 测试时 torchcodec 报 `libnvrtc.so.13` 缺失（环境已知问题）；训练显式指定 `--dataset.video_backend=pyav` 规避。

---

## Phase 4 — 模型权重

复用已有权重，未重新下载：

| 文件 | 大小 | 路径 |
|:---|:---:|:---|
| GeoPredict_robocasa.pth | 6.1G | `itvlaGp/ckpts/GeoPredict_robocasa.pth` |
| InternVLA-A1.5-base | 5.1G | `itvlaGp/ckpts/InternVLA-A1.5-base/model.safetensors` |

---

## Phase 5 — 正式 Warmup 400 step（GPU 0）

### 超参

| 参数 | 值 |
|:---|:---|
| `CUDA_VISIBLE_DEVICES` | **0** |
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

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  src/lerobot/scripts/lerobot_train.py \
  --output_dir=outputs/internvla_a1_5/warmup_scan_object_kptsim_400step \
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
  --dataset.repo_id=scan_object_kptsim_lrbv30 \
  --dataset.enable_keypoint_predictor=true \
  --dataset.num_keypoint_joints=14 \
  --dataset.action_mode=abs \
  --dataset.tokenize_state=true \
  --dataset.use_fast_action_tokens=true \
  --dataset.use_external_stats=true \
  --dataset.external_stats_path="${CLEAN}/scan_object_kptsim_lrbv30/norm_stat.json" \
  --dataset.video_backend=pyav \
  --seed=42 --batch_size=16 --steps=400 --save_freq=100 --log_freq=10 --num_workers=8 \
  --wandb.enable=false \
  > outputs/internvla_a1_5/warmup_scan_object_kptsim_400step.log 2>&1
```

完整日志: `outputs/internvla_a1_5/warmup_scan_object_kptsim_400step.log`

### 初始化验证

```
post_init_keypoint_weights: initialized keypoint_expert from action_expert weights.
load_geopredict_keypoint_weights: loaded 26 TrackEncoder keys (skipped 2, e.g. track_fusion_layer)
Trainable params: 927M / Total: 3B
num_frames=8463, num_episodes=50, effective batch_size=16
```

### 训练轨迹（每 10 step）

| Step | loss | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 19.546 | **0.5191** | 0.6941 | 0.236 | 532.4 |
| 20 | 5.725 | 0.1296 | 0.1988 | 0.226 | 181.7 |
| 30 | 3.734 | 0.0470 | 0.1392 | 0.240 | 95.3 |
| 40 | 2.905 | 0.0323 | 0.1068 | 0.223 | 67.2 |
| 50 | 2.105 | 0.0158 | 0.0776 | 0.198 | 46.4 |
| 60 | 1.560 | 0.0089 | 0.0560 | 0.176 | 38.5 |
| 70 | 1.221 | 0.0063 | 0.0408 | 0.171 | 33.4 |
| 80 | 0.983 | 0.0056 | 0.0290 | 0.173 | 27.5 |
| 90 | 0.766 | 0.0044 | 0.0208 | 0.153 | 24.8 |
| 100 | 0.682 | 0.0043 | 0.0157 | 0.163 | 24.0 |
| 110 | 0.601 | 0.0039 | 0.0117 | 0.164 | 20.3 |
| 120 | 0.542 | 0.0042 | 0.0097 | 0.153 | 20.2 |
| 130 | 0.489 | 0.0037 | 0.0077 | 0.148 | 19.6 |
| 140 | 0.439 | 0.0037 | 0.0068 | 0.133 | 17.0 |
| 150 | 0.419 | 0.0032 | 0.0058 | 0.135 | 15.6 |
| 160 | 0.406 | 0.0032 | 0.0054 | 0.133 | 16.2 |
| 170 | 0.391 | 0.0029 | 0.0046 | 0.135 | 13.3 |
| 180 | 0.364 | 0.0028 | 0.0044 | 0.124 | 13.8 |
| 190 | 0.367 | 0.0025 | 0.0041 | 0.130 | 12.6 |
| 200 | 0.438 | 0.0034 | 0.0043 | 0.159 | 11.7 |
| 210 | 0.371 | 0.0025 | 0.0038 | 0.135 | 11.4 |
| 220 | 0.342 | 0.0023 | 0.0035 | 0.124 | 11.4 |
| 230 | 0.380 | 0.0023 | 0.0035 | 0.144 | 12.4 |
| 240 | 0.349 | 0.0026 | 0.0033 | 0.128 | 10.3 |
| 250 | 0.349 | 0.0026 | 0.0034 | 0.128 | 10.2 |
| 260 | 0.347 | 0.0026 | 0.0035 | 0.125 | 11.0 |
| 270 | 0.328 | 0.0021 | 0.0030 | 0.124 | 8.8 |
| 280 | 0.368 | 0.0019 | 0.0029 | 0.146 | 8.9 |
| 290 | 0.335 | 0.0022 | 0.0030 | 0.127 | 9.7 |
| **300** | **0.341** | **0.0023** | **0.0031** | **0.128** | **9.0** |
| 310 | 0.339 | 0.0021 | 0.0028 | 0.131 | 6.2 |
| 320 | 0.323 | 0.0021 | 0.0027 | 0.124 | 6.7 |
| 330 | 0.346 | 0.0024 | 0.0028 | 0.133 | 6.6 |
| 340 | 0.358 | 0.0022 | 0.0031 | 0.137 | 7.4 |
| 350 | 0.297 | 0.0017 | 0.0024 | 0.116 | 5.6 |
| 360 | 0.371 | 0.0020 | 0.0028 | 0.147 | 6.2 |
| **370** | **0.327** | **0.0016** | **0.0023** | **0.132** | **5.5** |
| 380 | 0.298 | 0.0018 | 0.0024 | 0.116 | 5.8 |
| 390 | 0.302 | 0.0017 | 0.0023 | 0.119 | 5.3 |
| 400 | 0.340 | 0.0019 | 0.0025 | 0.136 | 5.4 |

### 验收判据

| 判据 | 预期 | 实测 |
|:---|:---|:---:|
| TrackEncoder init | loaded ~26 keys | ✅ 26 keys |
| step 10 `loss_kpt_current > 0` | > 0 | ✅ 0.5191 |
| kpt loss 明显下降 | step 300 << step 10 | ✅ 0.0023 vs 0.5191 |
| `video_decode_error` | 0 | ✅ 0 |
| `using_zeros` | 0 | ✅ 0 |
| 无 NaN/OOM | 正常完成 | ✅ SCAN_EXIT:0 |
| 训练耗时 | — | ~4m42s（纯训练步） |

### Checkpoint

| Step | 路径 | 备注 |
|:---:|:---|:---|
| 100 | `outputs/.../checkpoints/000100/pretrained_model` | |
| 200 | `outputs/.../checkpoints/000200/pretrained_model` | |
| **300** | `outputs/.../checkpoints/000300/pretrained_model` | **推荐 Phase 2 起点**（与 stack_bowls 经验一致） |
| 370 | — | `loss_kpt_cur` 最低 0.0016 |
| 400 | `outputs/.../checkpoints/000400/pretrained_model` | 最终 checkpoint |

---

## 错误与 Fix

| 现象 | 根因 | 修复 |
|:---|:---|:---|
| Layer 2 torchcodec 警告 | `libnvrtc.so.13` 缺失，PyPI wheel 链 CUDA 13 | 训练用 `--dataset.video_backend=pyav` |
| ep42 TCP 大步长 | 演示轨迹特性，非 FK 错误 | 注入验收仍 PASS，见手册附录 C |

本次训练 **无 OOM**（BS=16 可直接使用），**无 BackwardCompatibilityError**（v3.0 转换一次成功）。

---

## 关键路径汇总

| 用途 | 路径 |
|:---|:---|
| kptsim GT（只读） | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim/` |
| 注入 v2.1 | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrb/` |
| **训练用 v3.0（自包含）** | `/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/` |
| norm_stat（训练 CLI） | `.../scan_object_kptsim_lrbv30/norm_stat.json` |
| 原始 norm stats | `GeoPredict/ckpts/robotwin_norm_stats_scan_object.json` |
| InternVLA-A1.5-base | `itvlaGp/ckpts/InternVLA-A1.5-base/` |
| GeoPredict TrackEncoder | `itvlaGp/ckpts/GeoPredict_robocasa.pth` |
| 训练输出 | `outputs/internvla_a1_5/warmup_scan_object_kptsim_400step/` |
| 训练日志 | `outputs/internvla_a1_5/warmup_scan_object_kptsim_400step.log` |
| **推荐 checkpoint** | `outputs/.../checkpoints/000300/pretrained_model` |

---

## 结论

**`scan_object` 方案 A 端到端跑通**：数据注入 → v3.0 → 单卡 400 step 正式 Warmup，`loss_kpt_current` 从 0.52 降至 ~0.002 并饱和，**稳定收敛，无 NaN/OOM，视频解码零错误**。推荐以 **step 300** checkpoint 作为 Phase 2 微调起点。

---

*日志版本: wrmup1G-scnObj-v1.0 | 2026-08-25*
