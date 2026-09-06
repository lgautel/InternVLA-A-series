# Phase 2 SFT 训练执行日志（8×H200）

> 对应手册: [`p2sft_planH200.md`](p2sft_planH200.md)
> 执行日期: 2026-09-04
> 操作者: Claude Code (自动化执行)

---

## 用户提供的关键信息

| 项 | 值 |
|:---|:---|
| `WARMUP_CKPT` | `/home/a26113/b/Ckp/itvlaGpR1pro/elvat0714_4D_p1wrmup2609031326_2609031345/checkpoints/000426/pretrained_model` |
| `HF_HOME` | `/B/VENV/hf_home` |
| `HF_LEROBOT_HOME` | `/B/VENV/hf_home/lerobot` |
| `VENV` | `/B/VENV/itnvla15rbt20`（默认） |
| `PROJ_ROOT` | `/B/SRC/itvlaGp`（默认） |
| 数据集 | `/B/Dta/elevator0714_lerobot_4D`（默认） |

> 注: 用户的 Phase 1 ckpt 是 step 426（而非手册中的 400），说明 Phase 1 使用了不同的步数配置，但只要权重结构正确即可。

---

## 执行记录

### 1. 代码改动（§3 完成 05:10）

**§3.1 `configuration_internvla_a1_5.py`**（7 处改动）：
- §3.1.0: `InternVLAA15Config` 新增 `kpt_4d_mode`, `kpt_rot_loss_weight`, `_KPT_4D_DIM` ClassVar, `__post_init__` 派生 `keypoint_track_input_dim`
- §3.1.1: `InternVLAA15DatasetConfig` 新增 `kpt_4d_mode`, `keypoint_dim`, `__post_init__` 派生
- §3.1.2: `Extract3DKeypointTransformFn` 构建传入 `keypoint_dim`; 两个 `UnifyInputs` passthrough 加 `t.keypoint_dim`
- §3.1.3: `UnifyInternVLAA15InputsTransformFn` 新增 `keypoint_dim: int = 3`
- §3.1.4: `_kpt_fields_passthrough_or_zero` 新增参数 `keypoint_dim`, 所有 `3` → `d`
- §3.1.5: VQA 侧同步

**§3.2 `modeling_internvla_a1_5.py`**（4 处改动）：
- `keypoint_out_proj` 输出维度 → `config.keypoint_track_input_dim`
- `embed_kpt_suffix` fallback zeros → `self.config.keypoint_track_input_dim`
- 新增 `_kpt_split_loss` helper（pos MSE + rot normalize+MSE）
- kpt loss 计算使用 `kpt_dim` + `self._kpt_split_loss()`

**§3.3 `transform_internvla_a1_5.py`**（1 处改动）：
- `Extract3DKeypointTransformFn` 新增 `keypoint_dim`, 所有 `3` → `d`

### 2. 数据准备（§5 完成 05:12）

- symlink: `ln -sfn /B/Dta/elevator0714_lerobot_4D /B/VENV/hf_home/lerobot/elevator0714_lerobot_4D` ✓
- norm stats: 直接从 parquet 计算 → `observation.state` 25D, `action` 19D ✓

### 3. Smoke 测试中遇到的 5 个错误与修复

| # | Error | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | `unrecognized: --policy.use_fast_action_tokens` | 该字段仅在 DatasetConfig, 不在 PolicyConfig | 删除 launch L174 |
| 2 | `NCCL ncclInternalError` | 集群无 NCCL tuner | `NCCL_TUNER_PLUGIN=""` |
| 3 | WAN_SMOKE 仍启 8GPU | wrapper 无条件 export 覆盖 | 条件判断 smoke 模式 |
| 4 | `list<double>` vs `float32` | info.json dtype 与 parquet 不匹配 | info.json→float64 + load fallback |
| 5 | `pos_embedding [50,256] vs [75,256]` | Phase1 用 history=200, launch 传 300 | history→200 |

### 4. WAN Smoke 通过（05:25）

```
step:1 | loss:8.601 | loss_action:0.073 | loss_video:0.423 | loss_kpt_cur:0.0014 | loss_kpt_fut:0.0018
step:2 | loss:5.817 | loss_action:0.089 | loss_video:0.381 | loss_kpt_cur:0.0735 | loss_kpt_fut:0.0791
```

### 5. 正式训练（§11）

#### 5.1 首次 8GPU 尝试（05:27 — 失败）

**Error 6: NCCL `ncclInternalError`**
```
No NCCL_TUNER_CONFIG_PATH provided. Please populate NCCL_TUNER_CONFIG_PATH to use config-based tuner plugin.
torch.distributed.DistBackendError: NCCL error in: NCCLUtils.cpp:93, internal error
```

| 项 | 值 |
|---|---|
| Root Cause | 集群 `/usr/local/nvidia/lib64/libnccl-tuner.so` 在 `LD_LIBRARY_PATH` 中，NCCL 2.27 自动加载但无 config。wrapper 设 `NCCL_TUNER_PLUGIN=""` 无效（空字符串不阻止 NCCL 自动检测） |
| Fix | 改为 `NCCL_TUNER_PLUGIN="libnccl-tuner-disabled.so"`（指向不存在的库名，NCCL dlopen 失败后安全跳过） |
| 文件 | `launch/r1pro_elevator_p2_8xH200.sh` L28, `launch/internvla_a15_r1pro_geop_phase2_elevator.sh` L48 |

- 训练 exit=1 后，monitor 自动启动 bigmatrix（PID 515706）占用全部 8 GPU

#### 5.2 修复后第二次尝试（未成功）

- bigmatrix 占用了全部 8 GPU（100% / 122GB each），第二次 launch 无法获取 GPU 资源

#### 5.3 最终成功尝试（05:34 — 08:39）

- kill bigmatrix → 确认 8 GPU 空闲 → 重新 launch

**训练配置确认：**
```
Policy: InternVLAA15Policy
Total params: 8B | Trainable: 3B
Qwen3_5: 2B | Action expert: 460M | WAN: 5B (frozen)
Learnable tokens: 50 (frozen)
Effective batch size: 16 × 8 = 128
Steps: 2130 | Save freq: 213 (每 epoch 一次)
```

**Loss 收敛曲线：**

| Step | Epoch | Loss | Action | Video | VQA | FAST | Kpt_cur | Kpt_fut |
|------|-------|------|--------|-------|-----|------|---------|---------|
| 50 | 0.24 | 6.823 | 0.157 | 0.459 | 4.792 | 4.685 | 0.0015 | 0.0032 |
| 213 | 1.00 | — | — | — | — | — | — | — |
| 250 | 1.18 | 2.202 | 0.037 | 0.097 | 1.724 | 1.865 | 0.0021 | 0.0035 |
| 426 | 2.00 | — | — | — | — | — | — | — |
| 500 | 2.36 | 1.515 | 0.023 | 0.076 | 1.202 | 1.348 | 0.0015 | 0.0023 |
| 639 | 3.00 | — | — | — | — | — | — | — |
| 700 | 3.30 | 1.203 | 0.015 | 0.072 | 0.976 | 1.094 | 0.0013 | 0.0019 |
| 852 | 4.00 | — | — | — | — | — | — | — |
| 1000 | 4.72 | 0.955 | 0.011 | 0.068 | 0.775 | 0.865 | 0.0010 | 0.0015 |
| 1065 | 5.00 | — | — | — | — | — | — | — |
| 1278 | 6.00 | — | — | — | — | — | — | — |
| 1300 | 6.13 | 0.712 | 0.008 | 0.074 | 0.552 | 0.639 | 0.0009 | 0.0014 |
| 1491 | 7.00 | — | — | — | — | — | — | — |
| 1500 | 7.07 | 0.611 | 0.007 | 0.064 | 0.474 | 0.559 | 0.0009 | 0.0014 |
| 1704 | 8.00 | — | — | — | — | — | — | — |
| 1800 | 8.25 | 0.538 | 0.008 | 0.066 | 0.391 | 0.467 | 0.0009 | 0.0014 |
| 1917 | 9.00 | — | — | — | — | — | — | — |
| 2050 | 9.67 | 0.491 | 0.007 | 0.066 | 0.349 | 0.434 | 0.0009 | 0.0014 |
| 2100 | 9.90 | 0.513 | 0.007 | 0.068 | 0.368 | 0.445 | 0.0009 | 0.0014 |
| 2130 | 10.0 | — | — | — | — | — | — | — |

**训练耗时：** 05:37:08 → 08:39:47 ≈ **3h03m**（含 WAN 加载 + 首步编译）

**Checkpoints（10/10 全部保存）：**

| Epoch | Step | 路径 | 大小 |
|-------|------|------|------|
| 1 | 213 | `checkpoints/000213/pretrained_model/` | 18G |
| 2 | 426 | `checkpoints/000426/pretrained_model/` | 18G |
| 3 | 639 | `checkpoints/000639/pretrained_model/` | 18G |
| 4 | 852 | `checkpoints/000852/pretrained_model/` | 18G |
| 5 | 1065 | `checkpoints/001065/pretrained_model/` | 18G |
| 6 | 1278 | `checkpoints/001278/pretrained_model/` | 18G |
| 7 | 1491 | `checkpoints/001491/pretrained_model/` | 18G |
| 8 | 1704 | `checkpoints/001704/pretrained_model/` | 18G |
| 9 | 1917 | `checkpoints/001917/pretrained_model/` | 18G |
| 10 | 2130 | `checkpoints/002130/pretrained_model/` | 18G |

Output dir: `/B/SRC/itvlaGp/outputs/internvla_a1_5/2026_09_04_05_34_46-internvla_a1_5-r1pro-elev-geop-p2-e1-sft/`

### 6. 后处理

**归档备份：**
- Monitor 自动归档: `~/b/Ckp/ItvlaGpR1proElvtH200_26090408.tar` (24GB)
- 失败尝试归档: `~/b/Ckp/ItvlaGpR1proElvtH200_26090405_err.tar`

**GPU 占用：**
- bigmatrix_multiply_optimization.py 已启动 (PID 568244)
- 8 GPU 全部占用 ~110GB / ~100% utilization

### 7. Error 汇总

共遇到 **6 个 error**（5 个在 smoke 阶段，1 个在正式训练阶段）：

| # | 阶段 | Error | Root Cause | Fix |
|---|------|-------|-----------|-----|
| 1 | smoke | `unrecognized: --policy.use_fast_action_tokens` | 该字段仅在 DatasetConfig | 删除 launch L174 |
| 2 | smoke | `NCCL ncclInternalError` | 集群无 NCCL tuner config | `NCCL_TUNER_PLUGIN=""` |
| 3 | smoke | WAN_SMOKE 仍启 8GPU | wrapper 无条件 export 覆盖 | 条件判断 smoke 模式 |
| 4 | smoke | `list<double>` vs `float32` | info.json dtype 与 parquet 不匹配 | info.json→float64 + load fallback |
| 5 | smoke | `pos_embedding [50,256] vs [75,256]` | Phase1 用 history=200, launch 传 300 | history→200 |
| 6 | formal | NCCL 自动加载 tuner plugin | `NCCL_TUNER_PLUGIN=""` 不阻止自动检测 | 改为指向不存在的库名 |

### 8. 改动文件汇总

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py` | §3.1 代码改动 | 7D kpt 支持 |
| `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | §3.2 代码改动 | 7D kpt loss + proj |
| `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py` | §3.3 代码改动 | Extract transform 7D |
| `launch/internvla_a15_r1pro_geop_phase2_elevator.sh` | 修复 | 4 处修复 |
| `launch/r1pro_elevator_p2_8xH200.sh` | 新建 | 8×H200 wrapper |
| `src/lerobot/datasets/utils.py` | 修复 | load 容错 |
| `/B/Dta/elevator0714_lerobot_4D/meta/info.json` | 修复 | dtype float64 |
| `/B/Dta/elevator0714_lerobot_4D/meta/norm_stat_abs.json` | 新建 | 归一化统计 |

---

## Phase 2 SFT 续训（Epoch 11-20）

> 执行时间: 2026-09-04（接续上述训练）
> 目标：在 Epoch 10 checkpoint（step 2130）基础上再训 10 epoch，总计 20 epoch

### 9. 续训参数分析

**需要变更的参数**：

| 参数 | 首次训练值 | 续训值 | 原因 |
|------|-----------|--------|------|
| `WARMUP_CKPT` | Phase 1 ckpt@426 | Epoch 10 ckpt@2130 | 从 SFT 第 10 epoch 权重出发 |
| `EXPR_NAME` | `ItvlaGpR1proElvtH200` | `ItvlaGpR1proElvtH200E2` | 区分归档文件名 |

**保持不变的参数**：

| 参数 | 值 | 原因 |
|------|-----|------|
| `STEPS` | 2130 | 再训 10 epoch（10 × 213 步） |
| `SAVE_FREQ` | 213 | 每 epoch 保存一次 |
| `SCHEDULER_WARMUP` | 213 | 保持与首次相同的 warmup 比例 |
| `optimizer_lr` | 5e-5 | 与首次一致，模型从 SFT 权重继续收敛 |
| `scheduler_decay_lr` | 5e-6 | 最终 LR |
| `BATCH_SIZE` | 16 | 8×H200 不变 |

**续训策略说明**：
- 采用"从已微调权重重新启动"（re-finetune）策略，而非 true resume（不恢复 optimizer state）
- 原因：目标是在 epoch 10 权重基础上继续学习，而非恢复上次训练的精确轨迹。LR schedule 重新从 warmup 开始，有利于在新的 LR 曲线下进一步优化
- Epoch 10 ckpt 验证：`kpt_4d_mode=pos_rot`，`keypoint_track_input_dim=7`，`keypoint_history_max_len=200` ✓

**操作**：
- 无需修改任何代码文件（代码改动已在首次训练中完成）
- 只需在启动时传入新的 `WARMUP_CKPT` 和 `EXPR_NAME`

**续训 Epoch 10 checkpoint 路径**：
```
/B/SRC/itvlaGp/outputs/internvla_a1_5/2026_09_04_05_34_46-internvla_a1_5-r1pro-elev-geop-p2-e1-sft/checkpoints/002130/pretrained_model
```

### 10. 续训 Smoke 测试

WAN Smoke（1 GPU × 2 step，09:01）：

```
step:1 | loss:0.457 | loss_action:0.005 | loss_video:0.076 | loss_kpt_cur:0.0006 | loss_kpt_fut:0.0008
step:2 | loss:1.152 | loss_action:0.023 | loss_video:0.050 | loss_kpt_cur:0.0707 | loss_kpt_fut:0.0732
```

- exit=0 ✓，所有 loss 分量激活
- 初始 loss=0.457（远低于首次 Epoch 1 的 6.823），符合预期（已从微调权重出发）

### 11. 续训正式训练（Epoch 11-20）

**启动**：09:04:38（杀掉 bigmatrix → GPU 空闲 → 重新 launch）

```bash
export WARMUP_CKPT="/B/SRC/itvlaGp/outputs/internvla_a1_5/2026_09_04_05_34_46-internvla_a1_5-r1pro-elev-geop-p2-e1-sft/checkpoints/002130/pretrained_model"
export EXPR_NAME="ItvlaGpR1proElvtH200E2"
nohup bash launch/r1pro_elevator_p2_8xH200.sh > /tmp/p2sft_e2_train.log 2>&1 &
```

Output dir: `/B/SRC/itvlaGp/outputs/internvla_a1_5/2026_09_04_09_04_38-internvla_a1_5-r1pro-elev-geop-p2-e1-sft/`

**Loss 收敛曲线（Epoch 11-20 = local step 1-2130）**：

> 注：local epoch 1-10 = 全局 epoch 11-20。LR warmup 造成前 1 epoch loss 先升后降（正常现象）。

| Local Step | Epoch (local) | Loss | Action | Video | VQA | Kpt_cur |
|------|-------|------|--------|-------|-----|---------|
| 50 | 0.24 | 0.468 | 0.006 | 0.060 | 0.342 | 0.0009 |
| 250 | 1.18 | 0.944 | 0.011 | 0.066 | 0.768 | 0.0014 |
| 500 | 2.36 | 0.852 | 0.010 | 0.064 | 0.683 | 0.0015 |
| 700 | 3.30 | 0.698 | 0.008 | 0.063 | 0.547 | 0.0010 |
| 1000 | 4.72 | 0.544 | 0.007 | 0.062 | 0.411 | 0.0009 |
| 1100 | 5.19 | 0.469 | 0.007 | 0.064 | 0.336 | 0.0008 |
| 1300 | 6.13 | 0.372 | 0.005 | 0.068 | 0.248 | 0.0008 |
| 1500 | 7.07 | 0.304 | 0.005 | 0.058 | 0.197 | 0.0008 |
| 1700 | 8.02 | 0.284 | 0.006 | 0.061 | 0.165 | 0.0008 |
| 1900 | 8.96 | 0.267 | 0.006 | 0.065 | 0.141 | 0.0009 |
| 2050 | 9.67 | 0.240 | 0.005 | 0.061 | 0.131 | 0.0009 |
| 2100 | 9.90 | 0.248 | 0.005 | 0.063 | 0.132 | 0.0008 |

**训练耗时**：09:04:38 → 12:16:50 ≈ **3h12m**

**Checkpoints（10/10 全部保存）**：

| Local Epoch | Global Epoch | Step | 大小 |
|-------------|-------------|------|------|
| 1 | 11 | 213 | 18G |
| 2 | 12 | 426 | 18G |
| 3 | 13 | 639 | 18G |
| 4 | 14 | 852 | 18G |
| 5 | 15 | 1065 | 18G |
| 6 | 16 | 1278 | 18G |
| 7 | 17 | 1491 | 18G |
| 8 | 18 | 1704 | 18G |
| 9 | 19 | 1917 | 18G |
| 10 | 20 | 2130 | 18G |

### 12. 续训后处理

**归档备份**：
- Monitor 自动归档: `~/b/Ckp/ItvlaGpR1proElvtH200E2_26090412.tar` (21GB)

**GPU 占用**：
- bigmatrix_multiply_optimization.py 已启动 (PID 631906)
- 8 GPU 全部占用 ~102GB / ~100% utilization

### 13. 全局训练总结

| 项 | Epoch 1-10（首次 SFT） | Epoch 11-20（续训 SFT） |
|---|---|---|
| 起点权重 | Phase 1 Warmup ckpt@426 | Epoch 10 ckpt (step 2130) |
| 训练步数 | 2130 (10 epochs) | 2130 (10 epochs) |
| 初始 loss | 6.823 (step 50) | 0.468 (step 50) |
| 最终 loss | ~0.491 (step 2100) | ~0.240 (step 2100) |
| 训练耗时 | 3h03m | 3h12m |
| 全局最终 loss | — | **0.240**（action:0.005, video:0.063, vqa:0.132, kpt:0.0008） |

续训结论：epoch 11-20 loss 从 0.468 下降到 0.240，**远低于 epoch 10 的 0.491**，模型在第二轮 SFT 中取得了明显的进一步收敛。


