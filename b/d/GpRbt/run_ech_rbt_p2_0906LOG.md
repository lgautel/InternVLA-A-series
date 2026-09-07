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

**训练指标汇总** (每 ~10 epochs):

| step | epoch | loss | loss_action | loss_vqa | loss_video | loss_fast | loss_kpt_cur | loss_kpt_fut | grdn | lr |
|------|-------|------|-------------|----------|------------|-----------|-------------|-------------|------|-----|
| 50 | 1.04 | 6.119 | 0.125 | 3.943 | 0.911 | 3.994 | 0.0035 | 0.0046 | 19.3 | 3.6e-06 |
| 250 | 5.22 | 1.379 | 0.014 | 1.003 | 0.228 | 1.098 | 0.0029 | 0.0037 | 15.1 | 3.1e-05 |
| 500 | 10.44 | 0.965 | 0.009 | 0.688 | 0.180 | 0.748 | 0.0030 | 0.0038 | 15.8 | 4.8e-05 |
| 750 | 15.66 | 0.708 | 0.006 | 0.477 | 0.159 | 0.528 | 0.0035 | 0.0042 | 14.5 | 4.6e-05 |
| 1100 | 21.93 | 0.455 | 0.004 | 0.264 | 0.142 | 0.294 | 0.0033 | 0.0038 | 9.1 | 4.2e-05 |
| 1450 | 30.28 | 0.323 | 0.004 | 0.140 | 0.138 | 0.157 | 0.0026 | 0.0032 | 6.9 | 3.5e-05 |
| 1950 | 39.68 | 0.247 | 0.004 | 0.065 | 0.133 | 0.073 | 0.0026 | 0.0032 | 5.0 | 2.7e-05 |
| 2600 | 54.30 | 0.164 | 0.002 | 0.017 | 0.123 | 0.019 | 0.0024 | 0.0029 | 2.4 | 1.4e-05 |
| 3200 | 66.83 | 0.156 | 0.002 | 0.009 | 0.123 | 0.010 | 0.0023 | 0.0029 | 1.6 | 6.8e-06 |
| 3600 | 75.18 | 0.157 | 0.002 | 0.007 | 0.125 | 0.008 | 0.0025 | 0.0031 | 1.4 | 5.0e-06 |

**Checkpoints 保存**:

| step | epoch | 时间 | 路径 |
|------|-------|------|------|
| 960 | 20 | 10:12 | `.../checkpoints/000960` |
| 1920 | 40 | 11:26 | `.../checkpoints/001920` |
| 2880 | 60 | 12:41 | `.../checkpoints/002880` |
| 3648 | 76 | 13:42 | `.../checkpoints/003648` (final) |

**训练结束**: 13:45:32 UTC — `End of training`, `post_check: video_decode_error=0 using_zeros=0 exit=0`

**总训练时间**: 4h47m (08:58 → 13:45)

**最终 loss 分解**: loss=0.157, 其中 loss_video=0.125 占主导 (79.6%), loss_action=0.002, loss_vqa=0.007, loss_fast=0.008, loss_kpt_cur=0.0025, loss_kpt_fut=0.0031

**`latest` 符号链接**: 已创建, 指向 `checkpoints/003648/pretrained_model`

**Wandb**: 已移至 `/B/Log/ItvlaGpRbtSft/pick_dual_bottles/sft/`

### 3.5 Task 2: rotate_qrcode — 启动尝试 (3次失败)

**Error 3: PROJ_ROOT 未设置**

首次尝试 (13:47) 失败: `cd: /tmp/SRC/InternVLA-A-series: No such file or directory`

**根因**: 启动脚本默认 `PROJ_ROOT=/tmp/SRC/InternVLA-A-series` (line 25), 而实际路径为 `/B/SRC/itvlaGp`。pick_dual_bottles 在前一个 session 中已设置了此变量, 但此次未传递。

**Fix**: 显式传递 `PROJ_ROOT=/B/SRC/itvlaGp`。

**Error 4: VENV_ROOT 未设置**

第二次尝试 (13:50) 失败: `/tmp/itnvla15rbt20/bin/python: No such file or directory`

**根因**: 启动脚本默认 `VENV_ROOT=/tmp/itnvla15rbt20` (line 24), 而实际 venv 在 `/B/VENV/itnvla15rbt20/`。

**Fix**: 显式传递 `VENV_ROOT=/B/VENV/itnvla15rbt20`。

**Error 5: Output 目录已存在**

第三次尝试 (13:53) 失败: `FileExistsError: Output directory ... already exists and resume is False`

**根因**: 在启动前用 `mkdir -p` 预创建了 OUTPUT_DIR, 但训练脚本 `validate()` 检查输出目录不应已存在 (防止覆盖)。

**Fix**: 不预创建 OUTPUT_DIR, 让训练脚本自行创建。

**Error 6: HF_LEROBOT_HOME 未设置**

第四次尝试 (13:54) 失败: `FileNotFoundError: .../hf_home/lerobot/rotate_qrcode_lrb3_kptsim/meta/info.json`

**根因**: 启动脚本默认 `HF_LEROBOT_HOME=${VENV_ROOT}/var/datasets`, 但数据实际在 `/B/Dta/RoboTwin-Clean/`。

**Fix**: 显式传递 `HF_LEROBOT_HOME=/B/Dta/RoboTwin-Clean`。

### 3.6 Task 2: rotate_qrcode — 正式训练 (成功启动)

**时间**: 13:58 开始

**JOB_STAMP**: `2026_09_06_13_58_17`

**关键环境变量**:
```bash
VENV_ROOT=/B/VENV/itnvla15rbt20
PROJ_ROOT=/B/SRC/itvlaGp
HF_LEROBOT_HOME=/B/Dta/RoboTwin-Clean
SMOKE=0
STEPS=4636
SAVE_FREQ=1220
PROC_PER_NODE=8
BATCH_SIZE=16
SCHEDULER_WARMUP=463
VIDEO_MICRO_BATCH_SIZE=1
NCCL_TUNER_PLUGIN=/dev/null          # Fix for Error 2
# FLA_TILELANG 不设置 (允许 tilelang)  # Fix for Error 1
TRITON_CACHE_DIR=/tmp/triton_cache_a26113
```

**关键路径**:
- OUTPUT_DIR: `~/b/Ckp/ItvlaGpRbtSft/rotate_qrcode/sft/2026_09_06_13_58_17-internvla_a1_5-geop-kpt-sft-rotate_qrcode`
- LOG_FILE: `/B/Log/ItvlaGpRbtSft/rotate_qrcode/sft/sft_2026_09_06_13_58_17.log`
- WARMUP_CKPT: `/home/a26113/b/Ckp/ItvlaGpRbt0905/rotate_qrcode/warmup/2026_09_06_07_48_17-internvla_a1_5-geop-kpt-warmup-rotate_qrcode/checkpoints/000366/pretrained_model/`

**训练进程**: PID 141666

**初始训练指标**:

| step | epoch | loss | loss_action | loss_vqa | loss_video | loss_fast | grdn | lr | iters/s |
|------|-------|------|-------------|----------|------------|-----------|------|-----|---------|
| 50 | 0.83 | 6.111 | 0.082 | 4.458 | 0.831 | 4.579 | 24.977 | 2.9e-06 | 0.20 |

**GPU 显存**: 8 张 H200 均 ~102 GB / 143 GB

**FLA Backend**: 确认使用 tilelang

**训练指标汇总** (关键节点):

| step | epoch | loss | loss_action | loss_vqa | loss_video | loss_fast | loss_kpt_cur | loss_kpt_fut | grdn | lr |
|------|-------|------|-------------|----------|------------|-----------|-------------|-------------|------|-----|
| 50 | 0.83 | 6.111 | 0.082 | 4.458 | 0.831 | 4.579 | 0.0010 | 0.0016 | 25.0 | 2.9e-06 |
| 300 | 4.97 | 0.978 | 0.012 | 0.696 | 0.159 | 0.783 | 0.0008 | 0.0010 | 11.8 | 3.0e-05 |
| 800 | 13.26 | 0.528 | 0.005 | 0.348 | 0.123 | 0.394 | 0.0008 | 0.0009 | 5.9 | 4.7e-05 |
| 1200 | 19.89 | 0.378 | 0.004 | 0.217 | 0.115 | 0.245 | 0.0012 | 0.0013 | 4.7 | 4.3e-05 |
| 1900 | 31.49 | 0.237 | 0.004 | 0.087 | 0.106 | 0.096 | 0.0005 | 0.0005 | 3.4 | 3.4e-05 |
| 2550 | 42.26 | 0.154 | 0.002 | 0.026 | 0.105 | 0.028 | 0.0004 | 0.0005 | 2.0 | 2.4e-05 |
| 3300 | 54.69 | 0.121 | 0.001 | 0.005 | 0.100 | 0.005 | 0.0004 | 0.0004 | 1.2 | 1.4e-05 |
| 4100 | 67.94 | 0.112 | 0.001 | 0.003 | 0.097 | 0.003 | 0.0004 | 0.0004 | 0.9 | 6.6e-06 |
| 4600 | 76.23 | 0.112 | 0.001 | 0.003 | 0.097 | 0.003 | 0.0004 | 0.0005 | 0.8 | 5.0e-06 |

**Checkpoints 保存**:

| step | epoch | 时间 | 路径 |
|------|-------|------|------|
| 1220 | 20 | 15:33 | `.../checkpoints/001220` |
| 2440 | 40 | 17:08 | `.../checkpoints/002440` |
| 3660 | 60 | 18:42 | `.../checkpoints/003660` |
| 4636 | 76 | 19:58 | `.../checkpoints/004636` (final) |

**训练结束**: 20:02:28 UTC — `End of training`, `post_check: video_decode_error=0 using_zeros=0 exit=0`

**总训练时间**: 6h04m (13:58 → 20:02)

**最终 loss 分解**: loss=0.112, 其中 loss_video=0.097 占主导 (86.6%), loss_action=0.001, loss_vqa=0.003, loss_fast=0.003, loss_kpt_cur=0.0004, loss_kpt_fut=0.0005

**`latest` 符号链接**: 已创建, 指向 `checkpoints/004636/pretrained_model`

**Wandb**: 已移至 `/B/Log/ItvlaGpRbtSft/rotate_qrcode/sft/`

## 4. 后处理

### 4.1 Bigmatrix GPU 占位

已启动 `bigmatrix_multiply_optimization.py` (PID 214025), 占用 8×H200 ~70% VRAM。

### 4.2 日志打包

所有训练日志 (包括 wandb) 已复制到 `~/b/Ckp/ItvlaGpRbtSft/<task>/sft/logs/`。

### 4.3 最终 Checkpoint 概览

| 任务 | Checkpoint 根路径 | 保存的步数 | 最终 loss |
|------|------------------|-----------|-----------|
| pick_dual_bottles | `~/b/Ckp/ItvlaGpRbtSft/pick_dual_bottles/sft/2026_09_06_08_58_13-...` | 960, 1920, 2880, 3648 | 0.157 |
| rotate_qrcode | `~/b/Ckp/ItvlaGpRbtSft/rotate_qrcode/sft/2026_09_06_13_58_17-...` | 1220, 2440, 3660, 4636 | 0.112 |

## 5. 错误汇总

| # | 错误 | 根因 | Fix | 影响 |
|---|------|------|-----|------|
| 1 | FLA tilelang required | P2 解冻 VLM, FLA backward 需要 tilelang | 移除 `FLA_TILELANG=0` | pick_dual_bottles smoke test |
| 2 | NCCL Internal Error | NCCL tuner plugin 缺少配置 | 添加 `NCCL_TUNER_PLUGIN=/dev/null` | pick_dual_bottles attempt 1 |
| 3 | PROJ_ROOT 未设置 | 默认路径 `/tmp/SRC/InternVLA-A-series` 不存在 | 传递 `PROJ_ROOT=/B/SRC/itvlaGp` | rotate_qrcode attempt 1 |
| 4 | VENV_ROOT 未设置 | 默认路径 `/tmp/itnvla15rbt20` 不存在 | 传递 `VENV_ROOT=/B/VENV/itnvla15rbt20` | rotate_qrcode attempt 2 |
| 5 | Output 目录已存在 | `mkdir -p` 预创建了 OUTPUT_DIR | 不预创建, 让训练脚本自建 | rotate_qrcode attempt 3 |
| 6 | HF_LEROBOT_HOME 未设置 | 默认路径不含数据集 | 传递 `HF_LEROBOT_HOME=/B/Dta/RoboTwin-Clean` | rotate_qrcode attempt 4 |

## 6. 完成状态

**全部任务完成** ✓

- [x] pick_dual_bottles SFT: 3648 steps, 76 epochs, 4h47m, loss 6.119 → 0.157
- [x] rotate_qrcode SFT: 4636 steps, 76 epochs, 6h04m, loss 6.111 → 0.112
- [x] Bigmatrix GPU 占位启动
- [x] 日志打包到 ~/b/Ckp/

---

