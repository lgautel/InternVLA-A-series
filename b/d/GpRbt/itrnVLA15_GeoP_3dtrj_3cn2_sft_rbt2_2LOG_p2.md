# Phase 2 正式训练日志 — Action 训练

> 基于 [itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2.md](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2.md) §4.2、[预探索阶段日志](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG.md) 的推荐超参、和 [Phase 1 日志](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p1.md) 推荐的 Step 300 checkpoint。

---

## 1. 训练配置

### Phase 1 checkpoint 来源

Phase 1 推荐 checkpoint（Step 300, 详见 LOG_p1 §5）:
```
outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model
```

Phase 1 Step 300 终态指标: kpt_cur=0.0011, kpt_fut=0.0037, action=0.095, grad_norm=3.51

### Phase 2 超参（来源：预探索决策矩阵）

| 参数 | 值 | 依据 |
|------|------|------|
| GPU | 8×H200 (140 GB each) | 用满所有 GPU |
| batch_size (per GPU) | 16 | Phase 1 验证: 81 GB/卡 |
| 有效 BS | 128 (16×8) | 23550 frames / 128 ≈ 184 steps/epoch |
| train_expert_only | true | VLM 冻结 |
| action_loss_only | true | P2-3 验证: WAN 不推荐 |
| **action_loss_weight** | **10.0** | Phase 2 action 主导 (= 4× kpt) |
| **kpt_loss_weight** | **2.5** | Phase 2 kpt 维持性训练 |
| **action_expert_lr_scale** | **1.0** | Phase 2 action expert LR 恢复正常 |
| kpt_expert_lr_scale | 1.0 | 同 action |
| optimizer_lr | 5e-5 | 手册默认 |
| scheduler_warmup_steps | 1000 | 手册默认 (10% of steps) |
| scheduler_decay_steps | 10000 | = total steps |
| scheduler_decay_lr | 5e-6 | 手册默认 |
| steps | 10000 | 手册默认 |
| save_freq | 1000 | 用户要求 |
| log_freq | 50 | 手册默认 |
| wandb.enable | true (offline) | 正式训练监控 |
| seed | 42 | 可复现 |
| **init_kpt_expert_from_action** | **false** ⚠️ | 保护 Phase 1 kpt expert |
| **geopredict_checkpoint_path** | **不设置** ⚠️ | track encoder 已在 Phase 1 ckpt 中 |

### Phase 2 vs Phase 1 关键差异

| 配置项 | Phase 1 | Phase 2 |
|--------|---------|---------|
| pretrained_path | InternVLA-A1.5-base | **Phase 1 Step 300 checkpoint** |
| action_loss_weight | 5.0 | **10.0** |
| kpt_loss_weight | 10.0 | **2.5** |
| action_expert_lr_scale | 0.1 | **1.0** |
| init_kpt_expert_from_action | true | **false** ⚠️ |
| geopredict_checkpoint_path | 设置 | **不设置** ⚠️ |
| steps | 400 | **10000** |
| scheduler_warmup_steps | 50 | **1000** |

---

## 2. 训练执行

**日期**: 2026-08-04
**总墙钟时间**: ~3 小时 20 分钟 (含两次启动)
- 首次训练: 06:05 ~ 06:42 (steps 0-2000, 因磁盘满中断)
- 磁盘清理: 移动旧 runs 至 `/mnt/r/`，释放 ~378 GB
- 恢复训练: 07:47 ~ 09:25 (steps 2001-10000)

```
JOB_NAME: 2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs
OUTPUT_DIR: outputs/internvla_a1_5/2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs
```

### 训练命令

**首次启动 (steps 0-2000)**:
```bash
HF_HOME=/mnt/r/CKPT/hf_home \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py \
  --policy.type=internvla_a1_5 \
  --policy.pretrained_path=outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model \
  --policy.train_expert_only=true \
  --policy.action_loss_only=true \
  --policy.enable_keypoint_predictor=true --policy.num_keypoint_joints=14 \
  --policy.action_loss_weight=10.0 --policy.kpt_loss_weight=2.5 \
  --policy.action_expert_lr_scale=1.0 --policy.kpt_expert_lr_scale=1.0 \
  --policy.optimizer_lr=5e-5 --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=10000 --policy.scheduler_decay_lr=5e-6 \
  --policy.init_kpt_expert_from_action=false \
  --policy.knowledge_insulation=true --policy.knowledge_insulation_kpt=true \
  --policy.enable_vqa_loss=true --policy.tokenize_state=true \
  --policy.freeze_learnable_tokens=true --policy.num_learnable_tokens=50 \
  --policy.dtype=bfloat16 --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B \
  --dataset.type=internvla_a1_5 --dataset.repo_id=robotwin/stack_bowls_three_kpt \
  --dataset.enable_keypoint_predictor=true --dataset.num_keypoint_joints=14 \
  --dataset.action_mode=abs --dataset.use_external_stats=true \
  --dataset.external_stats_path=/mnt/r/CKPT/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json \
  --dataset.tokenize_state=true --dataset.use_fast_action_tokens=true \
  --seed=42 --batch_size=16 --steps=10000 --save_freq=1000 --log_freq=50 \
  --wandb.enable=true --wandb.project=internvla_a1_5 --wandb.mode=offline
```

**恢复训练 (steps 2001-10000)**:
```bash
HF_HOME=/mnt/r/CKPT/hf_home \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu --num_processes=8 \
  src/lerobot/scripts/lerobot_train.py \
  --resume=true \
  --config_path=outputs/internvla_a1_5/2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs/checkpoints/002000/pretrained_model/train_config.json
```

### 中断与恢复记录

**中断原因**: step 3000 checkpoint 保存时磁盘空间不足 (`No space left on device`)。根盘 969G 中有 433G 被 `outputs/internvla_a1_5/` 占用（含历史训练 runs）。

**修复步骤**:
1. 将 4 个旧 run 移至 `/mnt/r/CKPT/b1k2026/internvla_a15_outputs_archive/`:
   - `a15_libero4suite_100k_20260728_151045` (301G)
   - `a15_robotwin_stackb3_10k_bs16_20260731_093918` (61G)
   - `smoketest2_1785246394` (16G)
   - `smoketest_1785246073` (220K)
2. 清理失败的 step 3000 空 checkpoint 目录
3. 释放 ~378G 空间（588G/969G → 61%）

**恢复验证**: 从 step 2000 checkpoint 恢复后，首条 log 的指标与中断前完全一致:
- `loss_action=0.010`, `kpt_cur=0.0009`, `lr=4.6e-05` ✅

### 初始化验证

- Phase 1 checkpoint 已加载: kpt expert 保留了 Phase 1 训练成果
- `init_kpt_expert_from_action=false`: ✅ 不覆盖 kpt expert
- `geopredict_checkpoint_path` 未设置: ✅ track encoder 已在 Phase 1 ckpt 中
- Trainable params: 927M / Total: 3B (VLM 2B frozen, WAN 0)
- 首步 kpt_cur=0.0010, kpt_fut=0.0034（与 Phase 1 Step 300 终态 0.0011/0.0037 一致）

---

## 3. 训练轨迹

### 完整训练数据

| Step | loss | action | kpt_cur | kpt_fut | grad_norm | lr | epoch |
|------|------|--------|---------|---------|-----------|------|-------|
| 50 | 5.658 | 0.091 | 0.0010 | 0.0034 | 5.15 | 1.3e-6 | 0.27 |
| 100 | 5.638 | 0.088 | 0.0010 | 0.0034 | 4.81 | 3.8e-6 | 0.54 |
| 200 | 5.631 | 0.083 | 0.0010 | 0.0033 | 5.04 | 8.8e-6 | 1.09 |
| 500 | 5.257 | 0.050 | 0.0010 | 0.0030 | 4.87 | 2.4e-5 | 2.72 |
| **1000** | **4.973** | **0.022** | **0.0011** | **0.0027** | **4.19** | 4.9e-5 | 5.44 |
| **2000** | **4.851** | **0.010** | **0.0009** | **0.0021** | **2.70** | 4.6e-5 | 10.87 |
| **3000** | **4.822** | **0.006** | **0.0009** | **0.0021** | **2.58** | 4.1e-5 | 16.31 |
| **4000** | **4.823** | **0.005** | **0.0008** | **0.0018** | **1.74** | 3.5e-5 | 21.74 |
| **5000** | **4.820** | **0.004** | **0.0007** | **0.0019** | **1.58** | 2.8e-5 | 27.18 |
| **6000** | **4.802** | **0.004** | **0.0007** | **0.0019** | **1.44** | 2.1e-5 | 32.61 |
| **7000** | **4.775** | **0.003** | **0.0008** | **0.0020** | **1.14** | 1.4e-5 | 38.05 |
| **8000** | **4.792** | **0.003** | **0.0008** | **0.0020** | **1.01** | 9.4e-6 | 43.48 |
| **9000** | **4.787** | **0.003** | **0.0006** | **0.0016** | **0.96** | 6.2e-6 | 48.92 |
| **10000** | **4.819** | **0.003** | **0.0007** | **0.0019** | **1.00** | 5.0e-6 | 54.35 |

### 收敛分析

**action loss 收敛轨迹**:
```
step    50: 0.091  █████████████████████████████████████████████████ (100%)
step   500: 0.050  ███████████████████████████ (55%)
step  1000: 0.022  ████████████ (24%)
step  2000: 0.010  █████ (11%)
step  3000: 0.006  ███ (7%)
step  4000: 0.005  ██▌ (5.5%)
step  5000: 0.004  ██ (4.4%)   ← 进入平台期
step  7000: 0.003  █▌ (3.3%)
step 10000: 0.003  █▌ (3.3%)   ← 完全饱和
```

**四阶段收敛特征**:

| 阶段 | Steps | action 变化 | grad_norm | 特征 |
|------|-------|-------------|-----------|------|
| 快速下降期 | 0-1000 | 0.091→0.022 (-76%) | 5.2→4.2 | Warmup + 急速学习 |
| 主学习期 | 1000-3000 | 0.022→0.006 (-73%) | 4.2→2.6 | 峰值 LR 区间, 主要收敛 |
| 缓慢收敛期 | 3000-5000 | 0.006→0.004 (-33%) | 2.6→1.6 | 增益递减 |
| 饱和平台期 | 5000-10000 | 0.004→0.003 (-25%) | 1.6→1.0 | 完全饱和, LR 衰减至底 |

**kpt 稳定性**: kpt_cur 全程保持在 0.0006-0.0011 范围内，与 Phase 1 终态 (0.0011) 一致或略优。Phase 2 的 kpt 维持性训练 (kpt_loss_weight=2.5) 成功保护了 Phase 1 的 kpt 训练成果。

**grad_norm 下降**: 从 5.15 稳步降至 1.0, 表明模型在训练后期已接近局部最优，参数更新幅度极小。

---

## 4. Checkpoint 验证

| Step | 路径 | 大小 | action | kpt_cur | kpt_fut | grad_norm |
|------|------|------|--------|---------|---------|-----------|
| 1000 | `checkpoints/001000/pretrained_model` | 9.4G | 0.022 | 0.0011 | 0.0027 | 4.19 |
| 2000 | `checkpoints/002000/pretrained_model` | 9.4G | 0.010 | 0.0009 | 0.0021 | 2.70 |
| 3000 | `checkpoints/003000/pretrained_model` | 9.4G | 0.006 | 0.0009 | 0.0021 | 2.58 |
| 4000 | `checkpoints/004000/pretrained_model` | 9.4G | 0.005 | 0.0008 | 0.0018 | 1.74 |
| **5000** | **`checkpoints/005000/pretrained_model`** | **9.4G** | **0.004** | **0.0007** | **0.0019** | **1.58** |
| 6000 | `checkpoints/006000/pretrained_model` | 9.4G | 0.004 | 0.0007 | 0.0019 | 1.44 |
| **7000** | **`checkpoints/007000/pretrained_model`** | **9.4G** | **0.003** | **0.0008** | **0.0020** | **1.14** |
| 8000 | `checkpoints/008000/pretrained_model` | 9.4G | 0.003 | 0.0008 | 0.0020 | 1.01 |
| 9000 | `checkpoints/009000/pretrained_model` | 9.4G | 0.003 | 0.0006 | 0.0016 | 0.96 |
| **10000** | **`checkpoints/010000/pretrained_model`** | **9.4G** | **0.003** | **0.0007** | **0.0019** | **1.00** |
| last | `checkpoints/last` → `010000` | — | = step 10000 | — | — | — |

所有 checkpoint 完整路径前缀:
```
outputs/internvla_a1_5/2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs/
```

---

## 5. 推荐 Checkpoint

### 推荐: **Step 5000**

```
outputs/internvla_a1_5/2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs/checkpoints/005000/pretrained_model
```

### 为什么推荐 Step 5000

| 考虑因素 | Step 3000 | Step 5000 ⭐ | Step 7000 | Step 10000 |
|----------|----------|-------------|----------|-----------|
| action | 0.006 | **0.004** | 0.003 | 0.003 |
| kpt_cur | 0.0009 | **0.0007** | 0.0008 | 0.0007 |
| grad_norm | 2.58 | **1.58** | 1.14 | 1.00 |
| LR | 4.1e-5 | **2.8e-5** | 1.4e-5 | 5.0e-6 |
| epoch | 16.3 | **27.2** | 38.1 | 54.4 |
| action 收敛阶段 | 主学习期末 | **缓慢收敛→平台边界** | 饱和平台 | 完全饱和 |

**推荐 Step 5000 的 3 个理由**:

1. **Action loss 已进入平台期**: 从 step 5000 (0.004) 到 step 10000 (0.003) 的绝对降幅仅 0.001，而 step 1000→5000 降幅为 0.018。Step 5000 已捕获 ~96% 的 action 收敛增益，后续 5000 步仅贡献 ~4% 增益。

2. **过拟合风险**: 50 episodes × 471 frames/episode = 23550 帧。在 128 effective BS 下, step 5000 = ~27 epoch。step 10000 = ~54 epoch——在仅 50 条轨迹的小数据集上，54 epoch 过拟合风险显著。step 5000 的 ~27 epoch 在充分学习与泛化之间取得平衡。

3. **LR 处于合理区间**: step 5000 的 LR=2.8e-5 (peak 5e-5 的 56%)，仍有足够学习能力。step 10000 的 LR=5.0e-6 (decay 底部) 意味着模型在极低 LR 下反复训练同一小数据集，可能记住 noise pattern。

> **备选 1**: Step 7000 (action=0.003, 更低 loss 但 38 epoch)。如果实际评估中 step 5000 欠拟合，可用 step 7000。
> **备选 2**: Step 3000 (action=0.006, 16 epoch)。如果 step 5000 在新场景上过拟合，可回退到更保守的 step 3000。

### 推荐 checkpoint 的评估使用方式

```python
# 配置
config.pretrained_path = "outputs/internvla_a1_5/2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs/checkpoints/005000/pretrained_model"
config.inference_backend = "optimized"
config.action_loss_only = True
```

---

## 6. 训练过程分析

### Action Loss 分析

Phase 2 的 action loss 从 0.091 (step 50) 降至 0.003 (step 10000)，降幅 **96.7%**。与 Phase 1 对比:
- Phase 1 终态 action=0.095 (action_expert_lr_scale=0.1, LR 极低)
- Phase 2 起步 action=0.091 (step 50, warmup 刚开始)
- Phase 2 收敛速度快: 1000 步内 action 从 0.091→0.022 (-76%)

这说明 Phase 1 的 kpt expert 预热成功提供了有用的 3D 特征表示，使 Phase 2 的 action expert 能快速收敛。

### Kpt 稳定性分析

Phase 2 的核心设计目标之一是在 action 训练期间**保护** Phase 1 的 kpt 训练成果。结果:

| 指标 | Phase 1 终态 | Phase 2 起步 | Phase 2 终态 | 变化 |
|------|-------------|-------------|-------------|------|
| kpt_cur | 0.0011 | 0.0010 | 0.0007 | ↓0.0004 (改善) |
| kpt_fut | 0.0037 | 0.0034 | 0.0019 | ↓0.0018 (改善) |

kpt loss 不仅未退化，反而有轻微改善——这得益于:
1. `kpt_loss_weight=2.5` 提供了持续的 kpt 监督信号
2. Action expert 的改善间接提升了 kpt 特征质量（通过 cross-attention 反馈）

### Grad Norm 趋势

从 5.15 (step 50) 单调下降至 1.0 (step 10000)，无异常峰值或不稳定现象。这与 Phase 1 的初始 grad_norm 311 形成对比——Phase 2 的模型参数已处于良好初始化状态，无需大幅调整。

### 训练速度

- 平均: ~1.55 iters/s (128 effective BS)
- 每步: ~0.65s
- 每 1000 步: ~12 min (含 checkpoint 保存)
- 全程 10000 步纯训练: ~97 min

---

## 7. 关键路径汇总

| 项目 | 路径 |
|------|------|
| Phase 1 checkpoint (来源) | `outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model` |
| Phase 2 output dir | `outputs/internvla_a1_5/2026_08_04_06_05_16-internvla_a1_5-geop-phase2-action-train-stackb3-abs/` |
| **推荐 checkpoint** | **`<output_dir>/checkpoints/005000/pretrained_model`** |
| 备选 checkpoint 1 | `<output_dir>/checkpoints/007000/pretrained_model` |
| 备选 checkpoint 2 | `<output_dir>/checkpoints/003000/pretrained_model` |
| 数据集 | `/mnt/r/CKPT/hf_home/lerobot/robotwin/stack_bowls_three_kpt/` |
| 历史训练 runs (已归档) | `/mnt/r/CKPT/b1k2026/internvla_a15_outputs_archive/` |

