# Phase 2 正式训练计划 (080718) — Action + Video + Kpt (aw10, vw/kw 0.1, 10k)

> **Part A**：可执行微调计划（本文 §0–§6）  
> **Part B**：执行记录（本文 §7–§11，训练过程中填写）
>
> **引用链**：[LOG_p2.md](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2.md) → [LOG_p2_0805.md](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2_0805.md) → [LOG_p2_0807.md](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2_0807.md) → **本文**

---

## 0. 摘要与变更对照

### 0.1 目标

在 `/tmp/itnvla15rbt20/` 中，从 **Phase 1 Step 300 checkpoint** 启动 GeoP Phase 2 微调：Action + WAN Video + 3D Kpt，`stack_bowls_three_kpt`，8×H200，10000 steps。

### 0.2 相对 0807 的变更（loss 权重）

| 参数 | 0807 | **080718** |
|------|------|------------|
| `action_loss_weight` | 50.0 | **10.0** |
| `video_loss_weight` | 1.0 | **0.1** |
| `kpt_loss_weight` | 1.0 | **0.1** |
| `kpt_future_loss_weight` | 1.0 | **0.1** |
| `action_loss_only` | false | **false**（不变，WAN 启用） |
| `steps` | 10000 | 10000（不变） |
| venv | itnvla15rbt20 | 不变 |

### 0.3 相对 0807 保持不变

| 参数 | 值 |
|------|-----|
| `pretrained_path` | Phase 1 Step 300 |
| `lambda_vqa` | 1.0（默认，未改） |
| `batch_size` | 16（有效 BS=128） |
| `save_freq` | 2500 |
| `log_freq` | 50 |
| `optimizer_lr` | 5e-5 |
| `scheduler_warmup_steps` | 1000 |
| `scheduler_decay_steps` | 10000 |
| `scheduler_decay_lr` | 5e-6 |
| `train_expert_only` | true |
| `init_kpt_expert_from_action` | false |

### 0.4 总 loss 公式

```python
loss = (
    10.0   * loss_action          # action_loss_weight
    + 1.0  * loss_vlm             # lambda_vqa (default)
    + 0.1  * loss_video           # video_loss_weight
    + 0.1  * (loss_kpt_cur + 0.1 * loss_kpt_fut)  # kpt weights
)
```

有效权重比（action : video : kpt_cur）：**100 : 1 : 0.01**（相对 0807 的 50:1:1，video/kpt 监督大幅减弱）。

---

## 1. 完整训练配置

### Phase 1 checkpoint

```
outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model
```

### 080718 超参表

| 参数 | 值 |
|------|-----|
| GPU | 8×H200 |
| batch_size | 16 |
| **action_loss_weight** | **10.0** |
| **video_loss_weight** | **0.1** |
| **kpt_loss_weight** | **0.1** |
| **kpt_future_loss_weight** | **0.1** |
| **action_loss_only** | **false** |
| steps | 10000 |
| save_freq | 2500 |
| 其余 | 同 0807 |

---

## 2. 环境与路径

| 用途 | 路径 |
|------|------|
| venv | `/tmp/itnvla15rbt20/` |
| WAN | `/tmp/hf_home/hub/Wan2.2-TI2V-5B/` |
| 数据集 | `/tmp/hf_home/lerobot/robotwin/stack_bowls_three_kpt` |
| stats | `/tmp/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json` |
| **正式脚本** | `launch/internvla_a15_geop_phase2_finetune_stackb3_080718.sh` |
| **WAN smoke** | `launch/internvla_a15_geop_phase2_wan_smoke_080718.sh` |
| **训练日志** | `outputs/internvla_a1_5/train_080718_geop_phase2.log` |

### 环境变量

```bash
export HF_HOME=/tmp/hf_home HF_LEROBOT_HOME=/tmp/hf_home/lerobot
export VENV_ROOT=/tmp/itnvla15rbt20
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:${VENV_ROOT}/lib/pulseaudio"
export USE_LIBUV=0 WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=36501
```

---

## 3. 改良项与风险

| 风险 | 说明 |
|------|------|
| video/kpt 权重降至 0.1 | 直接 video/kpt 梯度弱，action 主导更强；kpt 主要靠间接监督 |
| 与 0805 对比 | 0805 用 aw10/vw1/kw1；080718 用 aw10/vw0.1/kw0.1，更偏 action-only 行为 |
| grad_norm | 预期低于 0807 aw50（~28@step50），训练应更稳定 |
| MASTER_PORT | 36501（避 0807 的 36500） |

---

## 4. Checklist

```bash
export VENV_ROOT=/tmp/itnvla15rbt20
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:${VENV_ROOT}/lib/pulseaudio"
${VENV_ROOT}/bin/python -c "import torch,torchcodec,lerobot; print('cuda', torch.cuda.device_count())"
ls /tmp/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
ls /tmp/hf_home/lerobot/robotwin/stack_bowls_three_kpt/meta/info.json
ls outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model/model.safetensors
pgrep -af lerobot_train || echo "no train procs"
```

---

## 5. 参考基线

| Run | action@10k | video@10k | grad_norm@10k |
|-----|------------|-----------|---------------|
| 0805 (aw10,vw1,kw1,20k) | 0.002 | 0.088 | 0.875 |
| 0807 (aw50,vw1,kw1,10k) | 0.002 | 0.117 | 4.338 |

---

## 6. 执行流程

1. Preflight checklist
2. WAN smoke（1 GPU, 2 step, 同 080718 loss 权重）
3. 8 GPU 正式训练 10000 steps
4. 记录 Part B

### 正式训练命令

```bash
cd /tmp/SRC/InternVLA-A-series
nohup bash launch/internvla_a15_geop_phase2_finetune_stackb3_080718.sh \
  > outputs/internvla_a1_5/train_080718_geop_phase2.log 2>&1 &
```

---

## Part B：执行记录

---

## 7. 时间线 / 操作日志

| 时间 (UTC+8) | 操作 | 结果 |
|---|---|---|
| 2026-08-07 10:47 | Preflight | ✅ venv/cuda8/WAN/dataset 均就绪，无残留训练进程 |
| 2026-08-07 10:48–10:50 | WAN smoke (1 GPU, 2 step) | ✅ loss_video=0.517/0.571, loss_action=0.073/0.057 |
| 2026-08-07 10:50 | 正式训练启动 (PID 117556) | ✅ 8×H200, MASTER_PORT=36501 |
| 2026-08-07 ~12:17 | **用户中止：计划写错，手动 kill 全部相关进程** | ❌ 已终止于 step ~5000 |

**中止原因**：`LOG_p2_080718.md` 微调计划配置有误，用户要求停止并清理全部相关进程。

**最后记录 step** (~5000): loss=4.808, action=0.004, video=0.130, kpt_cur=0.0002, grad_norm=1.461

**Launch 脚本**:
- `launch/internvla_a15_geop_phase2_finetune_stackb3_080718.sh`
- `launch/internvla_a15_geop_phase2_wan_smoke_080718.sh`

**日志**: `outputs/internvla_a1_5/train_080718_geop_phase2.log`

---

## 8. 问题记录

### #1: 计划配置错误 — 训练已中止

- **时间**: 2026-08-07 ~12:17 UTC+8
- **操作**: `pkill -TERM/-KILL` 终止所有 `080718` / `geop-phase2-aw10-vw0.1` / `lerobot_train` 相关进程
- **结果**: GPU 进程全部释放，无残留 `lerobot_train`
- **进度**: 中止于 step ~5000（checkpoint 002500 可能已保存）

---

## 9. 训练轨迹

| Step | loss | action | video | kpt_cur | kpt_fut | grad_norm |
|------|------|--------|-------|---------|---------|-----------|
| 50 | 5.800 | 0.092 | 0.953 | 0.0010 | 0.0033 | 5.344 |
| 100 | 5.731 | 0.087 | 0.837 | 0.0010 | 0.0035 | 4.893 |
| 150 | 5.743 | 0.091 | 0.629 | 0.0009 | 0.0036 | 5.337 |
| ~5000 | 4.808 | 0.004 | 0.130 | 0.0002 | 0.0037 | 1.461 |
| 10000 | — | — | — | — | — | **已中止** |

---

## 10. Checkpoint 验证

**OUTPUT_DIR**:

```
outputs/internvla_a1_5/2026_08_07_10_50_34-internvla_a1_5-geop-phase2-aw10-vw0.1-kw0.1-stackb3-abs-10k
```

预期 checkpoint: 002500 / 005000 / 007500 / 010000（save_freq=2500）

---

## 11. 最终结果

**状态**: ❌ 已中止（计划配置错误，未完成 10000 steps）

**OUTPUT_DIR**（部分 checkpoint 可能保留）:

```
outputs/internvla_a1_5/2026_08_07_10_50_34-internvla_a1_5-geop-phase2-aw10-vw0.1-kw0.1-stackb3-abs-10k
```

待用户提供修正后的计划后重新启动。
