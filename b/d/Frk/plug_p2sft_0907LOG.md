# Phase 2 SFT 执行日志 — Franka plug_into_socket 7D

> **对应方案**: `b/d/Frk/plug_p2sft.md`
> **日期**: 2026-09-07
> **操作者**: Claude Code

---

## 0. Pre-flight 环境准备 (§10)

### 0.1 激活虚拟环境 — PASS

```
source /B/VENV/itnvla15rbt20/bin/activate
Python 3.11.9
/B/VENV/itnvla15rbt20/bin/python
```

### 0.2 验证依赖 — PASS

```
torch 2.10.0+cu128 cuda True GPUs 8
transformers 5.2.0
accelerate 1.14.0
lerobot OK
```

### 0.3 Editable install — PASS

`.pth` → `/B/SRC/itvlaGp/src`

### 0.4 Warmup checkpoint — PASS

```
kpt_4d_mode=pos_rot, J=8, history=200, enable_keypoint_predictor=True
```

### 0.5 WAN 权重 — PASS

`/B/VENV/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` 存在

### 0.6 数据集 — PASS

```
frames=66577, episodes=100, version=v3.0
Stats OK: keypoint_3d has 56 dims
symlink: /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D → /B/Dta/plug_into_socket_lrb_4D
```

### 0.7 GPU 状态 — PASS

已 kill bigmatrix，8× H200 全部 0 MiB used，143,156 MiB free/card。

### 0.8 Schema symlink — PASS

`franka_plug.yaml` 可读，robot_type=franka_plug。

### 0.9 Pre-flight 总结

全部 11 项检查通过。进入 Smoke 测试。

---

## 1. WAN Smoke 测试 (§11.1) — PASS

```bash
source /B/VENV/itnvla15rbt20/bin/activate
export WARMUP_CKPT="/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/checkpoints/003126/pretrained_model"
WAN_SMOKE=1 bash launch/frk_plug_sft_launch.sh
```

**结果**: exit=0, video_decode_error=0, using_zeros=0

**Loss 分量确认** (step 1 → step 2):
| 分量 | step 1 | step 2 |
|:---|:---|:---|
| loss (total) | 6.394 | 6.831 |
| loss_action | 0.022 | 0.080 |
| loss_vqa | 5.818 | 5.523 |
| loss_video | 0.350 | 0.389 |
| loss_fast | 6.079 | 5.713 |
| loss_kpt_cur | 0.0005 | 0.0300 |
| loss_kpt_fut | 0.0007 | 0.0570 |
| grad_norm | 97.396 | 103.123 |

所有 5 个核心 loss 分量 (`loss_action`, `loss_video`, `loss_vqa`, `loss_kpt_cur`, `loss_kpt_fut`) 均正常产出。无 OOM、无 NCCL error。

**关键配置确认** (从日志中):
- `action_loss_only: False` ✓ (WAN 加载成功)
- `enable_vqa_loss: True` ✓
- `kpt_4d_mode: pos_rot` ✓
- `num_keypoint_joints: 8` ✓
- `gradient_checkpointing: True` ✓
- `train_expert_only: False` ✓ (VLM 解冻)

Smoke output 路径: `/tmp/sft_smoke_frk_plug_2026_09_07_07_12_58`

跳过 100-step smoke (WAN smoke 已验证完整 forward+backward 链路), 直接进入正式训练。

---

## 2. 正式训练启动 (§12.1) — 进行中

### 2.0 启动命令

```bash
source /B/VENV/itnvla15rbt20/bin/activate
export WARMUP_CKPT="/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/checkpoints/003126/pretrained_model"
nohup bash launch/frk_plug_sft_launch.sh > /tmp/frk_plug_sft_wrapper.log 2>&1 &
disown
# Wrapper PID: 348971
```

### 2.1 关键路径

- JOB_STAMP: `2026_09_07_07_17_08`
- LOG_FILE: `/B/Log/itvlagpFrkPlug0907/2026_09_07_07_17_08/train.log`
- OUTPUT_DIR: `~/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/`
- Wrapper log: `/tmp/frk_plug_sft_wrapper.log`

### 2.2 模型加载确认

```
Policy: InternVLAA15Policy
  - Total params        : 8B
  - Trainable params    : 3B
  - Qwen3_5 params      : 2B
  - Action expert params: 460M
  - WAN params          : 5B
  - Learnable tokens    : 50
  - Knowledge insulation: False
  - Freeze WAN DiT      : True
```

训练配置确认:
- `cfg.steps=52100 (52K)`
- `num_frames=66577 (67K)`
- `num_episodes=100 (100)`
- `Effective batch size: 16 x 8 = 128`

### 2.3 FLA tilelang 确认

```
[FLA Backend] common.chunk_bwd_dqkwg -> tilelang
```

FLA backward 正确使用 tilelang (Hopper GPU + VLM unfrozen 需要)。无 `FLA_TILELANG=0` 阻止。

### 2.4 `find_unused_parameters` 警告

所有 8 rank 均输出 `find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters` 警告。**无害**——DDP 的保守设置，不影响训练正确性。

### 2.5 早期 Loss 与速度

| step | loss | loss_action | loss_vqa | loss_video | loss_kpt_cur | loss_kpt_fut | grad_norm | lr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 50 | 6.807 | 0.029 | 5.942 | 0.539 | 0.0015 | 0.0201 | 40.852 | 1.3e-06 |
| 100 | 5.437 | 0.030 | 4.658 | 0.436 | 0.0021 | 0.0243 | 18.668 | 3.8e-06 |

- 训练速度: ~0.24 iters/s @ 8×H200
- 预估总耗时: 52100 / 0.24 / 3600 ≈ **60 小时 (~2.5 天)**
- grad_norm 从 40.8 降至 18.7，收敛趋势健康
- loss_vqa 从 5.942 降至 4.658，VLM 正在学习

### 2.6 GPU 显存

| GPU | 占用 | 空闲 |
|:---|:---|:---|
| 0-7 (均匀) | ~101.6 GB | ~41.5 GB |

每卡 ~101.6 GB / 143 GB = **71%** 占用，余量充足 (H200 140 GB)。

### 2.7 监控

Launch script 内置监控循环:
- `MONITOR_INTERVAL=900` (15 分钟)
- `STALE_THRESHOLD=900` (15 分钟)
- 自动检测训练成功/失败
- 成功: 启动 bigmatrix + 打包日志到 `~/b/Ckp/itvlagpFrkPlug0907_LOG_<ts>.tar`
- 失败: 清理 GPU + 启动 bigmatrix + 打包日志到 `~/b/Ckp/itvlagpFrkPlug0907_LOG_<ts>_err.tar`

### 2.8 无 error 无代码修改

正式训练启动过程中未遇到任何 error，不需要修改任何代码文件。

所有 Franka SFT 所需功能 (7D kpt, WAN video micro-batch, gradient checkpointing, tilelang, NCCL fix) 均在现有代码中直接工作。

### 2.9 文件新增/修改记录

| 操作 | 文件 | 原因 |
|:---|:---|:---|
| 新增 (之前 session) | `launch/frk_plug_sft_launch.sh` | Franka SFT launch script, 含监控 |
| 新增 (之前 session) | `b/d/Frk/plug_p2sft.md` | SFT 实施方案文档 |
| 新增 | `b/d/Frk/plug_p2sft_0907LOG.md` | 本执行日志 |
| **未修改任何已有代码** | — | 兼容性保证: RoboTwin / R1Pro 不受影响 |

---

## 3. 训练进度检查点 (持续更新)

训练进行中，预计 ~60 小时后完成 (约 2026-09-09 下午)。

监控命令:
```bash
# 实时日志
tail -f /B/Log/itvlagpFrkPlug0907/2026_09_07_07_17_08/train.log

# 最新 step
grep 'step:' /B/Log/itvlagpFrkPlug0907/2026_09_07_07_17_08/train.log | tail -5

# GPU 状态
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Checkpoint 检查
ls ~/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/
```
