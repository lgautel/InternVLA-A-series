# Phase 1 Warmup 执行日志 — 2026-09-06

> 任务列表: `pick_dual_bottles`, `rotate_qrcode`
> 基于: `b/d/rbt/run_ech_rbt_p1.md`（含监控功能的新版本）

---

## Step 0: 环境检查

**时间**: 2026-09-06 07:00

### 0.1 Python 环境
- VENV 路径: `/B/VENV/itnvla15rbt20`
- Python: `/B/VENV/itnvla15rbt20/bin/python`

### 0.2 模型权重
- HF_HOME: `/B/VENV/hf_home` (注意: 不是 `${VENV_ROOT}/var/hf_home`，该路径不存在)
- InternVLA-A1.5-base: `/B/VENV/hf_home/ckpts/InternVLA-A1.5-base` — 存在 ✓
- GeoPredict: `/B/VENV/hf_home/ckpts/GeoPredict_robocasa.pth` — 存在 ✓

### 0.3 数据集
- `pick_dual_bottles_lrb3_kptsim`: 6129 frames, 50 episodes, has keypoint_3d ✓
- `rotate_qrcode_lrb3_kptsim`: 7724 frames, 50 episodes, has keypoint_3d ✓
- norm_stat.json: 两者均存在 ✓

### 0.4 关键修复: editable install 路径

**问题**: venv 的 editable install 指向旧仓库 `/B/SRC/InternVLA-A-series/src`，该版本 **没有** keypoint predictor 的配置字段（`enable_keypoint_predictor`, `num_keypoint_joints`, `kpt_loss_weight` 等），导致之前训练时报 `unrecognized arguments` 错误。

**根因**: `.pth` 文件 `/B/VENV/itnvla15rbt20/lib/python3.11/site-packages/__editable__.internvla_a1_5-1.0.0.pth` 内容为 `../../../../../SRC/InternVLA-A-series/src`，指向不含 GeoPredict/keypoint 功能的旧代码。

**修复**: 将 `.pth` 文件内容改为 `/B/SRC/itvlaGp/src`（当前仓库，含完整 keypoint 支持）。备份原文件为 `.bak.20260906070057`。

**验证**: 所有 13 个 keypoint 相关配置字段均 OK:
```
enable_keypoint_predictor: OK
num_keypoint_joints: OK
action_loss_weight: OK
kpt_loss_weight: OK
kpt_future_loss_weight: OK
knowledge_insulation_kpt: OK
kpt_to_action_detach: OK
freeze_keypoint_modules: OK
action_expert_lr_scale: OK
kpt_expert_lr_scale: OK
track_encoder_lr_scale: OK
init_kpt_expert_from_action: OK
geopredict_checkpoint_path: OK
```

### 0.5 GPU 状态
- 8× GPU 全部被 PID 394 (`matrixkernle_optimmiz.py`) 占用，100% 利用率
- 需要在训练前清理

---

## Step 1: 部署配置和脚本

**时间**: 2026-09-06 07:02

### 1.1 config_p1.env
创建 `b/s/rbt/config_p1.env`，关键路径:
- `HF_HOME=/B/VENV/hf_home` (而非默认的 `${VENV_ROOT}/var/hf_home`)
- `PRETRAINED_PATH=${HF_HOME}/ckpts/InternVLA-A1.5-base`
- `GEOPREDICT_CKPT=${HF_HOME}/ckpts/GeoPredict_robocasa.pth`
- `MONITOR_INTERVAL=900` (15 分钟)

### 1.2 run_warmup_p1.sh
从 `b/d/rbt/run_ech_rbt_p1.md` §10 Step 4 提取，部署到 `b/s/rbt/run_warmup_p1.sh`，672 行，含监控功能。

### 1.3 Dry-run 验证
```
pick_dual_bottles: frames=6129  steps/epoch=48  total=288  save_freq=48
rotate_qrcode:    frames=7724  steps/epoch=61  total=366  save_freq=61
```

---

## Step 2: 清理 GPU 并启动训练

**时间**: 2026-09-06 07:05 ~ 07:17

### 2.1 终止 bigmatrix 占用 GPU

清理 PID 394 (`matrixkernle_optimmiz.py`)，释放 8 块 GPU。

### 2.2 NCCL 调优插件问题

**问题**: 8-GPU 训练启动时报 `ncclInternalError: Internal check failed. Last error: No NCCL_TUNER_CONFIG_PATH provided.`

**根因**: 机器上 GCP NCCL 调优插件 (`/usr/local/nvidia/lib64/libnccl-tuner.so`, `GcpTunerPlugin_v2`) 自动加载，但缺少其所需的配置文件。

**排查过程**:
1. `NCCL_TUNER_PLUGIN=""` → 无效，插件仍被自动加载
2. `NCCL_TUNER_CONFIG_PATH={}` → SIGSEGV
3. `NCCL_TUNER_PLUGIN="/dev/null"` → 成功跳过插件加载
4. 验证: 2-GPU barrier 测试通过, 8-GPU barrier 测试通过

**修复**: 在 launch 脚本第 23 行添加:
```bash
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-/dev/null}"
```

### 2.3 Launch 脚本参数化修改

`launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh`:
- Line 108: `--policy.scheduler_warmup_steps=50` → `"${SCHEDULER_WARMUP_STEPS:-50}"`
- Line 110: `--policy.scheduler_decay_lr=5e-6` → `"${SCHEDULER_DECAY_LR:-5e-6}"`

### 2.4 Smoke test

单 GPU smoke test 通过 (100 steps, exit=0)。

### 2.5 启动编排器

```bash
nohup bash b/s/rbt/run_warmup_p1.sh \
  --config b/s/rbt/config_p1.env \
  --tasks pick_dual_bottles,rotate_qrcode \
  --keep-going \
  > /tmp/warmup_p1_orchestrator.log 2>&1 &
```
编排器 PID: 29425

---

## Step 3: pick_dual_bottles Warmup 训练

**时间**: 2026-09-06 07:17 ~ 07:40

### 3.1 训练参数

| 参数 | 值 |
|------|-----|
| 数据集 | `pick_dual_bottles_lrb3_kptsim` |
| 帧数 | 6129 |
| EBS | 128 (8 GPU × 16 BS) |
| steps/epoch | 48 |
| epochs | 6 |
| 总步数 | 288 |
| save_freq | 48 |
| warmup_steps | 48 |
| lr | 5e-5 (action expert 0.04x, kpt expert 1.0x) |
| MASTER_PORT | 36201 |

### 3.2 训练过程

- 07:17:35 — 训练开始
- 07:21:43 — Step 10, loss=13.81, kpt_cur=0.9216, kpt_fut=0.5455
- 07:22:37 — Checkpoint 000048 (epoch 1)
- 07:25:38 — Checkpoint 000096 (epoch 2)
- 07:28:40 — Checkpoint 000144 (epoch 3)
- 07:31:42 — Checkpoint 000192 (epoch 4)
- 07:34:42 — Checkpoint 000240 (epoch 5)
- 07:37:43 — **Checkpoint 000288 (epoch 6, final)**
- 07:40:05 — `End of training`

**post_check**: `video_decode_error=0 using_zeros=0 exit=0` ✓

### 3.3 Loss 轨迹

| Step | Epoch | loss | action | kpt_cur | kpt_fut | grad_norm |
|------|-------|------|--------|---------|---------|-----------|
| 10 | 0.21 | 13.81 | 0.219 | 0.9216 | 0.5455 | 576 |
| 60 | 1.25 | 1.465 | 0.163 | 0.0075 | 0.0532 | 29 |
| 120 | 2.51 | 0.521 | 0.142 | 0.0043 | 0.0097 | 11 |
| 180 | 3.76 | 0.381 | 0.120 | 0.0036 | 0.0052 | 5.4 |
| 240 | 5.01 | 0.375 | 0.128 | 0.0035 | 0.0042 | 3.5 |
| 280 | 5.85 | 0.347 | 0.116 | 0.0034 | 0.0040 | 2.8 |

**观察**: Loss 快速下降，keypoint loss (kpt_cur, kpt_fut) 在 epoch 2 后趋于平稳，action loss 持续下降到 ~0.12。gradient norm 从 576 降至 2.8，训练稳定。

### 3.4 Checkpoint

- 最终 checkpoint: `~/b/Ckp/ItvlaGpRbt0905/pick_dual_bottles/warmup/2026_09_06_07_17_35-internvla_a1_5-geop-kpt-warmup-pick_dual_bottles/checkpoints/000288/pretrained_model/`
- model.safetensors: ~6.3 GB
- `latest` 符号链接: ✓
- wandb 已移至 `/B/Log/ItvlaGpRbt0905/pick_dual_bottles/warmup/`

### 3.5 编排器监控问题

训练完成后，编排器的 `monitor_training()` 进入了 `sleep 900` 监控循环，需要等待两个 15 分钟周期 (共 ~30 分钟) 才能检测到训练已完成。原因: 监控函数设计要求 "GPU 连续空闲 MONITOR_INTERVAL (900s) 才判定训练结束"，且每次检查间隔也是 900s。

**决策**: 手动终止编排器 (PID 29425)，接管后续流程，避免 25+ 分钟的无意义等待。pick_dual_bottles 的最终化 (latest symlink, wandb 移动) 和 rotate_qrcode 的启动均手动执行。

---

## Step 4: rotate_qrcode Warmup 训练

**时间**: 2026-09-06 07:45 ~ 08:09

### 4.1 首次启动失败

**时间**: 07:45

**错误**: `FileExistsError: Output directory ... already exists and resume is False.`

**根因**: 手动接管后，用 `mkdir -p` 预创建了 output_dir，触发了 `lerobot_train.py` 的 output_dir 存在性检查。

**修复**: 删除预创建的目录，重新生成 JOB_NAME 后再启动。

### 4.2 成功启动

**时间**: 07:48:17

### 4.3 训练参数

| 参数 | 值 |
|------|-----|
| 数据集 | `rotate_qrcode_lrb3_kptsim` |
| 帧数 | 7724 |
| EBS | 128 (8 GPU × 16 BS) |
| steps/epoch | 61 |
| epochs | 6 |
| 总步数 | 366 |
| save_freq | 61 |
| warmup_steps | 61 |
| lr | 5e-5 |
| MASTER_PORT | 36202 |

### 4.4 训练过程

- 07:48:17 — 训练开始
- 07:50:15 — Step 10, loss=22.68, kpt_cur=0.6129, kpt_fut=0.8048
- 07:50:47 — Checkpoint 000061 (epoch 1)
- 07:53:57 — Checkpoint 000122 (epoch 2)
- 07:57:09 — Checkpoint 000183 (epoch 3)
- 08:00:18 — Checkpoint 000244 (epoch 4)
- 08:03:43 — Checkpoint 000305 (epoch 5)
- 08:06:54 — **Checkpoint 000366 (epoch 6, final)**
- 08:09 — 训练进程退出

**注意**: 退出时有 wandb 离线模式的 cleanup 错误 (`OSError: Stale file handle`, `ConnectionResetError: Connection lost`)，但均为 wandb 离线同步的非关键错误，不影响 checkpoint 完整性。

### 4.5 Loss 轨迹

| Step | Epoch | loss | action | kpt_cur | kpt_fut | grad_norm |
|------|-------|------|--------|---------|---------|-----------|
| 10 | 0.17 | 22.68 | 0.226 | 0.6129 | 0.8048 | 608 |
| 60 | 0.99 | 1.562 | 0.136 | 0.0068 | 0.0611 | 35 |
| 120 | 1.99 | 0.434 | 0.112 | 0.0027 | 0.0092 | 19 |
| 180 | 2.98 | 0.228 | 0.079 | 0.0013 | 0.0029 | 7.7 |
| 240 | 3.98 | 0.211 | 0.079 | 0.0011 | 0.0020 | 4.8 |
| 300 | 4.97 | 0.201 | 0.079 | 0.0010 | 0.0017 | 2.8 |
| 360 | 5.97 | 0.205 | 0.086 | 0.0008 | 0.0013 | 2.6 |

**观察**: rotate_qrcode 初始 loss 更高 (22.68 vs pick_dual_bottles 的 13.81)，但 6 epoch 后 loss 下降到 0.205，低于 pick_dual_bottles 的 0.347。Keypoint loss 也更低 (kpt_cur=0.0008 vs 0.0034)。总体训练非常顺利。

### 4.6 Checkpoint

- 最终 checkpoint: `~/b/Ckp/ItvlaGpRbt0905/rotate_qrcode/warmup/2026_09_06_07_48_17-internvla_a1_5-geop-kpt-warmup-rotate_qrcode/checkpoints/000366/pretrained_model/`
- model.safetensors: 6,308,534,276 bytes (~6.3 GB)
- `latest` 符号链接: ✓
- wandb 已移至 `/B/Log/ItvlaGpRbt0905/rotate_qrcode/warmup/`

---

## Step 5: Post-training 操作

**时间**: 2026-09-06 08:11

### 5.1 启动 bigmatrix GPU 占用

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python3 b/d/rbt/bigmatrix_multiply_optimization.py > /tmp/bigmatrix.log 2>&1 &
```
- PID: 49860
- 8 块 GPU 全部 100% 利用率 ✓
- 每块 GPU 分配 102.37 GiB (73.8% VRAM)
- matmul: 32768×32768 bf16+cudagraph

### 5.2 日志打包

```
~/b/Ckp/ItvlaGpRbt0905/ItvlaGpRbt0905_warmup_logs_20260906_081130.tar.gz (148K)
```
包含:
- `/B/Log/ItvlaGpRbt0905/pick_dual_bottles/warmup/` (训练日志 + wandb)
- `/B/Log/ItvlaGpRbt0905/rotate_qrcode/warmup/` (训练日志 + wandb)

执行日志也拷贝到 `~/b/Ckp/ItvlaGpRbt0905/run_ech_rbt_p1_0906LOG.md`。

---

## 总结

### 结果

| 任务 | 状态 | 总步数 | 最终 loss | action loss | kpt_cur | kpt_fut | 训练时长 |
|------|------|--------|-----------|-------------|---------|---------|----------|
| pick_dual_bottles | ✅ 成功 | 288 | 0.347 | 0.116 | 0.0034 | 0.0040 | ~23 min |
| rotate_qrcode | ✅ 成功 | 366 | 0.205 | 0.086 | 0.0008 | 0.0013 | ~19 min |

### 遇到的问题及修复

1. **Editable install 指向旧仓库** → 修改 `.pth` 文件指向当前仓库
2. **HF_HOME 路径不匹配** → 使用 `/B/VENV/hf_home` 而非默认路径
3. **NCCL 调优插件缺配置** → `NCCL_TUNER_PLUGIN="/dev/null"` 跳过插件
4. **Launch 脚本参数硬编码** → 参数化为环境变量
5. **Output dir 预创建导致 FileExistsError** → 不预创建，让训练脚本自行创建
6. **编排器监控等待过久 (30 min)** → 手动接管避免浪费时间

### 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `__editable__.internvla_a1_5-1.0.0.pth` | 修改 | 指向 `/B/SRC/itvlaGp/src` |
| `launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh` L22-23 | 新增 | NCCL_TUNER_PLUGIN 设置 |
| `launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh` L108 | 修改 | scheduler_warmup_steps 参数化 |
| `launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh` L110 | 修改 | scheduler_decay_lr 参数化 |
| `b/s/rbt/config_p1.env` | 新建 | 机器特定配置 |
| `b/s/rbt/run_warmup_p1.sh` | 新建 | 编排脚本 (672 行) |

### Checkpoint 路径

```
~/b/Ckp/ItvlaGpRbt0905/
├── pick_dual_bottles/warmup/latest → .../checkpoints/000288/pretrained_model/
├── rotate_qrcode/warmup/latest    → .../checkpoints/000366/pretrained_model/
├── ItvlaGpRbt0905_warmup_logs_20260906_081130.tar.gz
└── run_ech_rbt_p1_0906LOG.md
```

### 后续

- bigmatrix GPU 占用已启动 (PID 49860)，8 块 GPU 100% 利用率
- Phase 2 正式训练可基于上述 warmup checkpoint 继续
