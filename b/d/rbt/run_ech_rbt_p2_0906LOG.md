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

- 杀掉 GPU 上的 bigmatrix 占位进程 (PID 49860)
- 创建 EXPR_NAME=ItvlaGpRbtSft 对应的日志/checkpoint 目录
- 激活虚拟环境并设置环境变量

---

