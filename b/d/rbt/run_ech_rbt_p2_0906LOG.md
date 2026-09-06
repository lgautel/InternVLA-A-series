# Phase 2 SFT 微调训练执行日志 — 2026-09-06

> **实施方案**: [run_ech_rbt_p2.md](run_ech_rbt_p2.md)
>
> **任务列表**: `pick_dual_bottles`, `rotate_qrcode`
>
> **执行人**: Claude Code (自动化)
>
> **开始时间**: 2026-09-06

---

## 1. 环境与前置条件

### 1.1 Preflight 检查结果

| 检查项 | 结果 |
|--------|------|
| GPU | 8×NVIDIA H200 (143771 MiB each), 可用 |
| VENV | `/B/VENV/itnvla15rbt20/bin/activate` 存在 |
| WAN2.2 VAE | `/B/VENV/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` 存在 |
| GeoPredict | `/B/VENV/hf_home/ckpts/GeoPredict_robocasa.pth` 存在 |
| A1.5-base | `/B/VENV/hf_home/ckpts/InternVLA-A1.5-base/config.json` 存在 |

### 1.2 任务数据

| 任务 | total_frames | total_episodes | 数据路径 |
|------|-------------|----------------|---------|
| pick_dual_bottles | 6129 | 50 | `/B/Dta/RoboTwin-Clean/pick_dual_bottles_lrb3_kptsim/` |
| rotate_qrcode | 7724 | 50 | `/B/Dta/RoboTwin-Clean/rotate_qrcode_lrb3_kptsim/` |

每个任务数据目录均包含 `meta/info.json`, `norm_stat.json`, `meta/keypoints_meta.json`。

### 1.3 Warmup Checkpoints

| 任务 | Warmup Checkpoint 路径 |
|------|----------------------|
| pick_dual_bottles | `/home/a26113/b/Ckp/ItvlaGpRbt0905/pick_dual_bottles/warmup/2026_09_06_07_17_35-internvla_a1_5-geop-kpt-warmup-pick_dual_bottles/checkpoints/000288/pretrained_model/` |
| rotate_qrcode | `/home/a26113/b/Ckp/ItvlaGpRbt0905/rotate_qrcode/warmup/2026_09_06_07_48_17-internvla_a1_5-geop-kpt-warmup-rotate_qrcode/checkpoints/000366/pretrained_model/` |

### 1.4 步数计算 (EBS=8×16×1=128)

| 任务 | frames | steps/epoch | total_steps | save_freq | save@steps | warmup_steps |
|------|--------|-------------|-------------|-----------|------------|-------------|
| pick_dual_bottles | 6129 | 48 | 3648 | 960 | 960,1920,2880,3648 | 364 |
| rotate_qrcode | 7724 | 61 | 4636 | 1220 | 1220,2440,3660,4636 | 463 |

## 2. 代码变更

### 2.1 修改 `b/s/rbt/compute_sft_steps.py`

**目的**: 支持 `--save-every-epochs` 参数, 允许自定义 checkpoint 保存间隔 (默认 E/4, 本次使用 20)。

**修改内容**:
1. `checkpoint_epochs(total_epochs, save_every=0)`: 新增 `save_every` 参数
2. `quarter_epoch_save_freq(steps, spe, epochs, save_every=0)`: 新增 `save_every` 参数
3. `compute_schedule(..., save_every=0)`: 新增 `save_every` 参数, 透传到上游函数
4. `main()`: 新增 `--save-every-epochs` CLI 参数, 透传到 `compute_schedule()`

**向后兼容**: 所有新增参数默认为 0, 等效于原来的 `E/4` 行为。

**验证**: 两个任务的步数计算结果已确认 (见 §1.4)。

## 3. 训练执行

### 3.1 环境准备

- 杀掉 GPU 上的 bigmatrix 占位进程 (PID 49860): `kill -9 49860`
- 创建目录:
  - `~/b/Ckp/ItvlaGpRbtSft/pick_dual_bottles/sft/`
  - `~/b/Ckp/ItvlaGpRbtSft/rotate_qrcode/sft/`
  - `/B/Log/ItvlaGpRbtSft/pick_dual_bottles/sft/`
  - `/B/Log/ItvlaGpRbtSft/rotate_qrcode/sft/`

### 3.2 Task 1: pick_dual_bottles — Smoke Test

**时间**: 08:54 — 08:55

**命令**: 1 GPU, 1 step smoke test (见下方环境变量)

**Error 1: FLA tilelang 必需**

```
RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg (see #640). Please install tilelang: `pip install tilelang`
```

**根因分析**: P2 SFT 解冻 VLM (train_expert_only=false), 梯度流经 FLA 层的 backward 函数 `chunk_bwd_dqkwg`。FLA 0.5.0 在 Triton >= 3.4.0 + Hopper GPU (H200, compute_cap=9.0) 上拒绝使用纯 Triton backend (已知产生错误结果), 要求使用 tilelang 作为替代。P1 warmup 中 VLM 冻结不触发此路径。

**Fix**: 移除 `FLA_TILELANG=0` 环境变量。tilelang 0.1.13 已安装, nvcc (CUDA 13.0) 可用。

**验证**: 重新运行 smoke test, 成功通过。

```
post_check: video_decode_error=0 using_zeros=0 exit=0
INFO 2026-09-06 08:54:57 ot_train.py:387 Checkpoint saved at: /tmp/sft_smoke_pick_dual_bottles/checkpoints/000001
```

### 3.3 Task 1: pick_dual_bottles — 正式训练 Attempt 1

**时间**: 08:55 — 08:56 (失败)

**JOB_STAMP**: `2026_09_06_08_55_45`

**Error 2: NCCL Internal Error**

```
torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:93,
internal error - please report this issue to the NCCL developers, NCCL version 2.27.5
ncclInternalError: Internal check failed.
Last error:
No NCCL_TUNER_CONFIG_PATH provided. Please populate NCCL_TUNER_CONFIG_PATH to use config-based tuner plugin.
```

崩溃点: `lerobot_train.py:185` → `accelerator.wait_for_everyone()` → NCCL barrier

**根因分析**: NCCL 2.27.5 加载了系统自带的 tuner plugin, 但缺少 `NCCL_TUNER_CONFIG_PATH` 配置。P1 warmup launch 脚本设置了 `NCCL_TUNER_PLUGIN=/dev/null` (禁用 tuner), 但 P2 launch 脚本没有。首次启动 8 GPU DDP 时 NCCL barrier 就失败。

**Fix**: 在 launch 环境变量中添加 `NCCL_TUNER_PLUGIN=/dev/null`。

### 3.4 Task 1: pick_dual_bottles — 正式训练 Attempt 2 (成功启动)

**时间**: 08:58 开始

**JOB_STAMP**: `2026_09_06_08_58_13`

**关键环境变量**:
```bash
SMOKE=0
STEPS=3648
SAVE_FREQ=960
PROC_PER_NODE=8
BATCH_SIZE=16
SCHEDULER_WARMUP=364
VIDEO_MICRO_BATCH_SIZE=1
NCCL_TUNER_PLUGIN=/dev/null          # Fix for Error 2
# FLA_TILELANG 不设置 (允许 tilelang)  # Fix for Error 1
TRITON_CACHE_DIR=/tmp/triton_cache_a26113
```

**关键路径**:
- OUTPUT_DIR: `~/b/Ckp/ItvlaGpRbtSft/pick_dual_bottles/sft/2026_09_06_08_58_13-internvla_a1_5-geop-kpt-sft-pick_dual_bottles`
- LOG_FILE: `/B/Log/ItvlaGpRbtSft/pick_dual_bottles/sft/sft_2026_09_06_08_58_13.log`
- WARMUP_CKPT: `/home/a26113/b/Ckp/ItvlaGpRbt0905/pick_dual_bottles/warmup/2026_09_06_07_17_35-internvla_a1_5-geop-kpt-warmup-pick_dual_bottles/checkpoints/000288/pretrained_model/`

**训练进程**: PID 69056

**初始训练指标** (step 50-100):

| step | epoch | loss | loss_action | loss_vqa | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr | iters/s |
|------|-------|------|-------------|----------|------------|-------------|-------------|------|-----|---------|
| 50 | 1.04 | 6.119 | 0.125 | 3.943 | 0.911 | 0.0035 | 0.0046 | 19.303 | 3.6e-06 | 0.19 |
| 100 | 2.09 | 3.639 | 0.070 | 2.443 | 0.482 | 0.0033 | 0.0048 | 9.781 | 1.0e-05 | 0.24 |

**速度**: ~0.24 iters/s (4.1s/step), 预计 ETA ~4 小时

**GPU 显存**: 8 张 H200 均 ~103 GB / 143 GB

**FLA Backend**: 确认使用 tilelang: `[FLA Backend] common.chunk_bwd_dqkwg -> tilelang`

---

