# Phase 1 正式训练日志 — Kpt Expert 预热

> 基于 [itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2.md](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2.md) §4.1 和 [预探索阶段日志](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG.md) 的推荐超参。

---

## 1. 训练配置

### 推荐超参（来源：预探索阶段 LOG）

| 参数 | 值 | 依据 |
|------|------|------|
| GPU | 8×H200 (140 GB each) | 用满所有 GPU |
| batch_size (per GPU) | 16 | P1-1: 81 GB/卡, 余量充足; BS=32 OOM |
| 有效 BS | 128 (16×8) | 23550 frames / 128 ≈ 184 steps/epoch |
| train_expert_only | true | VLM 冻结, 节省 ~24 GB optimizer state |
| action_loss_only | true | 不加载 WAN |
| action_loss_weight | 5.0 | kpt:action = 2:1 |
| kpt_loss_weight | 10.0 | P1-2: 不敏感, 10.0 grad_norm 适中 |
| action_expert_lr_scale | 0.1 | P1-3: 不敏感, 0.1 安全默认 |
| kpt_expert_lr_scale | 1.0 | 手册默认 |
| optimizer_lr | 5e-5 | 手册默认 |
| scheduler_warmup_steps | 50 | 探索验证有效 (~12.5% of 400 steps) |
| scheduler_decay_steps | 400 | = total steps |
| scheduler_decay_lr | 5e-6 | 手册默认 |
| steps | 400 | P1-4: kpt 在 200-300 步饱和 |
| save_freq | 100 | 覆盖 200/300/400 步 checkpoint |
| log_freq | 10 | 细粒度监控 |
| wandb.enable | true (offline) | 正式训练监控 |
| seed | 42 | 可复现 |
| GeoPredict ckpt | `/mnt/r/CKPT/geopredict/GeoPredict_robocasa.pth` | 小写路径 |

### 与手册 §4.1 脚本的差异

| 项 | 手册默认 | 本次设置 | 原因 |
|---|---|---|---|
| BATCH_SIZE | 8 | **16** | 预探索确认 H200 支持 BS=16 |
| STEPS | 5000 | **400** | 预探索确认 kpt 在 200-300 步饱和 |
| SAVE_FREQ | 1000 | **100** | 保存 200/300/400 步 checkpoint |
| LOG_FREQ | 50 | **10** | 细粒度监控 |
| scheduler_warmup_steps | 500 | **50** | 按比例缩小 (12.5%) |
| scheduler_decay_steps | 5000 | **400** | = total steps |
| GEOPREDICT_CKPT | `/mnt/r/CKPT/GeoPredict/...` | `/mnt/r/CKPT/geopredict/...` | 实际路径为小写 |

---

## 2. 训练执行

**日期**: 2026-08-04
**墙钟时间**: ~7 分钟 (含模型加载 + 400 步训练 + 4 次 checkpoint 保存)

```
JOB_NAME: 2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs
OUTPUT_DIR: outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs
```

### 训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py \
  --policy.type=internvla_a1_5 \
  --policy.pretrained_path=/mnt/r/CKPT/InternVLA-A1.5-base \
  --policy.train_expert_only=true \
  --policy.action_loss_only=true \
  --policy.enable_keypoint_predictor=true --policy.num_keypoint_joints=14 \
  --policy.action_loss_weight=5.0 --policy.kpt_loss_weight=10.0 \
  --policy.action_expert_lr_scale=0.1 --policy.kpt_expert_lr_scale=1.0 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=50 \
  --policy.scheduler_decay_steps=400 --policy.scheduler_decay_lr=5e-6 \
  --policy.init_kpt_expert_from_action=true \
  --policy.geopredict_checkpoint_path=/mnt/r/CKPT/geopredict/GeoPredict_robocasa.pth \
  --policy.knowledge_insulation=true --policy.knowledge_insulation_kpt=true \
  --policy.enable_vqa_loss=true --policy.tokenize_state=true \
  --policy.freeze_learnable_tokens=true --policy.num_learnable_tokens=50 \
  --policy.dtype=bfloat16 --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B \
  --dataset.type=internvla_a1_5 --dataset.repo_id=robotwin/stack_bowls_three_kpt \
  --dataset.enable_keypoint_predictor=true --dataset.num_keypoint_joints=14 \
  --dataset.action_mode=abs --dataset.use_external_stats=true \
  --dataset.external_stats_path=/mnt/r/CKPT/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json \
  --dataset.tokenize_state=true --dataset.use_fast_action_tokens=true \
  --seed=42 --batch_size=16 --steps=400 --save_freq=100 --log_freq=10 \
  --wandb.enable=true --wandb.project=internvla_a1_5 --wandb.mode=offline
```

### 初始化验证

- GeoPredict TrackEncoder: ✅ loaded 26 keys from `/mnt/r/CKPT/geopredict/GeoPredict_robocasa.pth`
- Trainable params: 927M / Total: 3B (VLM 2B frozen, WAN 0)
- Knowledge insulation: True
- Effective batch size: 16 × 8 = 128

---

## 3. 训练轨迹

| Step | loss | action | kpt_cur | kpt_fut | grad_norm | lr | epoch | iters/s |
|------|------|--------|---------|---------|-----------|------|-------|---------|
| 10 | 16.886 | 0.277 | 0.5438 | 0.5337 | 311.86 | 6.4e-6 | 0.05 | 0.55 |
| 20 | 9.170 | 0.264 | 0.0785 | 0.2345 | 120.76 | 1.6e-5 | 0.11 | 0.79 |
| 50 | 6.469 | 0.177 | 0.0080 | 0.0794 | 25.23 | 4.5e-5 | 0.27 | 1.05 |
| 100 | 5.566 | 0.118 | **0.0028** | 0.0174 | 14.73 | 4.4e-5 | 0.54 | 1.27 |
| 150 | 5.331 | 0.105 | **0.0016** | 0.0060 | 6.60 | 3.7e-5 | 0.82 | 1.44 |
| **200** | **5.383** | **0.103** | **0.0013** | **0.0045** | **6.03** | 2.8e-5 | 1.09 | 1.33 |
| 250 | 5.319 | 0.095 | **0.0011** | 0.0038 | 3.81 | 2.0e-5 | 1.36 | 1.37 |
| **300** | **5.382** | **0.095** | **0.0011** | **0.0037** | **3.51** | 1.2e-5 | 1.63 | 1.57 |
| 350 | 5.331 | 0.088 | **0.0010** | 0.0030 | 2.76 | 7.0e-6 | 1.90 | 1.65 |
| **400** | **5.329** | **0.089** | **0.0010** | **0.0032** | **3.13** | 5.0e-6 | 2.17 | 1.60 |

### 收敛分析

**kpt_cur 收敛轨迹**:
```
step  10: 0.5438  ████████████████████████████████████████████████████████ (100%)
step  50: 0.0080  █ (1.5%)
step 100: 0.0028  ▎ (0.5%)
step 200: 0.0013  ▏ (0.24%)  ← 基本饱和
step 300: 0.0011  ▏ (0.20%)  ← 完全饱和
step 400: 0.0010  ▏ (0.18%)  ← 终态
```

- **kpt_cur**: 0.5438 → 0.0010 (降幅 **99.82%**)，step 200 后变化 < 0.0003
- **kpt_fut**: 0.5337 → 0.0032 (降幅 **99.40%**)，step 200 后变化 < 0.0013
- **action**: 0.277 → 0.089 (降幅 **67.9%**)，action expert LR 低 (0.1×) 但随 kpt 特征改善自然下降
- **grad_norm**: 312 → 3.1，step 200 后稳定在 3-6 范围

---

## 4. Checkpoint 验证

| Step | 路径 | 大小 | kpt_cur | kpt_fut | action | grad_norm |
|------|------|------|---------|---------|--------|-----------|
| 100 | `checkpoints/000100/pretrained_model` | 5.9G | 0.0028 | 0.0174 | 0.118 | 14.73 |
| **200** | **`checkpoints/000200/pretrained_model`** | **5.9G** | **0.0013** | **0.0045** | **0.103** | **6.03** |
| **300** | **`checkpoints/000300/pretrained_model`** | **5.9G** | **0.0011** | **0.0037** | **0.095** | **3.51** |
| **400** | **`checkpoints/000400/pretrained_model`** | **5.9G** | **0.0010** | **0.0032** | **0.089** | **3.13** |
| last | `checkpoints/last/pretrained_model` + `training_state` | — | = step 400 | — | — | — |

WandB offline log: `wandb/offline-run-20260804_054226-9yrgzvmt`

所有 checkpoint 完整路径前缀:
```
outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/
```

---

## 5. 推荐 Checkpoint

### 推荐: **Step 300**

```
outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model
```

### 为什么推荐 Step 300 而不是 Step 400

| 考虑因素 | Step 200 | Step 300 ⭐ | Step 400 |
|----------|----------|------------|----------|
| kpt_cur | 0.0013 | **0.0011** | 0.0010 |
| kpt_fut | 0.0045 | **0.0037** | 0.0032 |
| action | 0.103 | **0.095** | 0.089 |
| grad_norm | 6.03 | **3.51** | 3.13 |
| LR | 2.8e-5 | **1.2e-5** | 5.0e-6 |
| kpt 饱和度 | 接近饱和 | **完全饱和** | 完全饱和 |

**推荐 Step 300 的 3 个理由**:

1. **kpt 已完全饱和**: kpt_cur 从 step 200 (0.0013) 到 step 300 (0.0011) 的降幅仅 0.0002，step 300 到 400 (0.0010) 降幅仅 0.0001。step 300 是 kpt 收敛的 "甜蜜点"——loss 已充分下降，梯度稳定。

2. **LR 处于合适位置**: step 300 的 LR=1.2e-5 已经较低但尚未触底。step 400 的 LR=5.0e-6 是 `scheduler_decay_lr` 的底部值，此时模型参数更新极小。Phase 2 将从 Phase 1 checkpoint 重新创建优化器并从头 warmup——如果 Phase 1 已经在极低 LR 区间过度拟合了微小的 noise pattern，这些 pattern 可能在 Phase 2 的 warmup 阶段被扰动，反而不如 step 300 稳定。

3. **防止轻微过拟合**: 23550 帧 / 128 eff_BS ≈ 184 steps/epoch。step 300 ≈ 1.63 epoch, step 400 ≈ 2.17 epoch。Phase 1 的目标仅是 kpt 预热，不需要将 action loss 压到极致（那是 Phase 2 的任务）。step 300 在 ~1.6 epoch 处结束训练，kpt 已饱和但未在小数据集上多次重复，降低过拟合风险。

> **备选**: 如果 Phase 2 实验中发现 step 300 checkpoint 的 kpt 起始值偏高，可改用 step 400。但基于预探索和正式训练结果，step 300 和 step 400 的差异极小 (kpt_cur 0.0011 vs 0.0010)，实际影响可以忽略。

### Phase 2 使用此 checkpoint 的关键配置

```bash
--policy.pretrained_path=outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model
--policy.init_kpt_expert_from_action=false   # ⚠️ 绝不能设为 true, 否则覆盖 Phase 1 训练成果
# 不设 --policy.geopredict_checkpoint_path   # ⚠️ track encoder 已在 checkpoint 中
--policy.action_loss_weight=10.0
--policy.kpt_loss_weight=2.5
--policy.action_expert_lr_scale=1.0
```

