# CubInBx Phase 2 SFT 训练日志

> **方案文档**: [`cubbx_sft1.md`](cubbx_sft1.md)
> **EXPR_NAME**: `4dwvlaFrkCubBx0924`
> **启动时间**: 2026-09-25
> **Warmup Checkpoint**: `/home/a26113/b/Ckp/4dwvlaFrkCubBx0924/2026_09_25_01_12_18-internvla_a1_5-frk3-cubbx-warmup/checkpoints/001156/pretrained_model`

---

## 1. Pre-flight 环境检查

按照 [`cubbx_sft1.md`](cubbx_sft1.md) §13 执行全部 12 项检查。

| # | 检查项 | 结果 | 备注 |
|:---:|:---|:---|:---|
| 13.1 | Python 版本 | ✅ 3.11.9 | |
| 13.2 | torch/transformers/lerobot | ✅ torch 2.10.0+cu128, transformers 5.2.0, lerobot OK | 8 GPUs |
| 13.3 | Editable install | ✅ | |
| 13.4 | GPU 状态 | ✅ 0 processes | 先 kill 了 bigmatrix (PID 3859806) |
| 13.5 | Warmup checkpoint | ✅ model.safetensors 5.9G | config 确认: kpt_4d_mode=pos_rot, 8 joints, tokenize_state=True |
| 13.6 | 数据集 info | ✅ 36953 frames, 56 episodes, v3.0, franka_cubinbx | |
| 13.7 | Stats | ✅ state:15, action:8, kpt:56 | |
| 13.8 | HF symlink | ✅ → /B/Dta/put_cube_into_box_lrb3_4D | |
| 13.9 | Schema | ✅ franka_cubinbx, action_mask_spec [7, -1] | |
| 13.10 | WAN 权重 | ✅ Wan2.2_VAE.pth 存在 | |
| 13.11 | FAST tokenizer | ✅ tokenizer.json 存在 | |
| 13.12 | 代码修改 | ✅ 全部 6 项 patch 就位 | ceiling div, bfloat16, FAST offline, p_schedule, begin_sample, RandomBlackout |

**Pre-flight: 全部 12 项通过。**

---

## 2. Smoke 测试

### Errors encountered during smoke setup

**Error 1**: `accelerate: command not found`
- **根因**: venv 的 `bin/` 中无 `accelerate` 入口脚本，但 accelerate 包已安装（`accelerate.commands.launch` 可用）
- **Fix**: 改 `accelerate launch` 为 `python -m accelerate.commands.launch`（与 warmup 脚本一致）
- **文件**: `b/s/Frk3/frk3_cubbx_sft_launch.sh` L220, `b/d/Frk3/ds/cubinbx/cubbx_sft1.md` L585, Appendix A

**Error 2**: `unrecognized arguments: --dataset.data_transforms.inputs.extract_3d_keypoint.*`
- **根因**: draccus CLI 不支持 `data_transforms.inputs.extract_3d_keypoint` 这种深层嵌套路径。warmup 脚本只传 `--dataset.keypoint_history_max_len=90`，extract_3d_keypoint transform 自动从 dataset config 读取。
- **Fix**: 删除 3 行无效 CLI args，改为 `--dataset.keypoint_history_max_len=92`
- **文件**: `b/s/Frk3/frk3_cubbx_sft_launch.sh`

**Error 3**: `ValueError: 'policy.repo_id' argument missing`
- **根因**: `TrainPipelineConfig.validate()` 要求 `policy.repo_id`（即使 `push_to_hub=false`）
- **Fix**: 增加 `--policy.repo_id=lerobot_lab/internvla_a1_5 --policy.push_to_hub=false`
- **文件**: `b/s/Frk3/frk3_cubbx_sft_launch.sh`

**Error 4**: `ValueError: Transform 'color_jitter' is not valid`
- **根因**: `make_transform_from_config()` 使用 PascalCase (`ColorJitter`, `SharpnessJitter`, `RandomAffine`, `RandomBlackout`)，SFT 脚本 TFS_CONFIG 误用了 snake_case。此外 RandomBlackout 参数是 `noise_scale` 而非 `noise_std`/`p`。
- **Fix**: TFS_CONFIG 改为 PascalCase + 正确参数名
- **文件**: `b/s/Frk3/frk3_cubbx_sft_launch.sh`

**Error 5**: `video_decode_error` / `using_zeros` — torchcodec 加载失败
- **根因**: LD_LIBRARY_PATH 设置不完整。SFT 脚本只有 `/usr/lib/x86_64-linux-gnu:/usr/local/npp/lib`，缺少 torch/nvidia 库路径，导致 libstdc++ ABI 不匹配 (`CXXABI_1.3.15` not found)。
- **Fix**: 使用与 warmup 脚本一致的完整 LD_LIBRARY_PATH（含 torch/lib、nvidia cuda_runtime/cuda_nvrtc/npp）
- **文件**: `b/s/Frk3/frk3_cubbx_sft_launch.sh`

### 2.1 Smoke 模式 1: Action-only (1 GPU, 10 steps)

- 命令: `SMOKE=1 bash b/s/Frk3/frk3_cubbx_sft_launch.sh`
- 结果: **✅ 通过** (exit=0)
- Checkpoint 保存于: `/tmp/sft_smoke_frk3_cubbx_2026_09_25_16_07_29/checkpoints/000010/`
- 无 `video_decode_error`、无 `using_zeros`
- `action_loss_only=true`, `enable_vqa_loss=false` 确认

### 2.2 Smoke 模式 2: WAN_SMOKE (8 GPUs, 2 steps)

- 命令: `WAN_SMOKE=1 bash b/s/Frk3/frk3_cubbx_sft_launch.sh`
- 结果: **✅ 通过** (exit=0)
- Checkpoint 保存于: `/tmp/sft_smoke_frk3_cubbx_2026_09_25_16_13_57/checkpoints/000002/`
- `action_loss_only=false` (完整 WAN model loaded)
- 无 OOM、无 NCCL error
- `log_freq=50` > 2 steps，所以无 loss 打印（预期行为）

**Smoke 测试: 全部通过。**

---

## 3. 正式训练

### 3.1 启动信息

- **启动时间**: 2026-09-25 16:18:45 UTC
- **PID**: 190033 (wrapper), 190039 (accelerate), 190153-190160 (8× GPU workers)
- **Output dir**: `~/b/Ckp/4dwvlaFrkCubBx0924/2026_09_25_16_18_45-internvla_a1_5-frk3-cubbx-sft/`
- **Log file**: `/B/Log/4dwvlaFrkCubBx0924/2026_09_25_16_18_45/train.log`
- **GPU**: 8× H200, ~60 GB/card (总 VRAM 143 GB/card)
- **EBS**: 128 (16×8)
- **预计总步数**: 5780 (20 epochs)
- **预计总时间**: ~7 小时
- **Checkpoint 保存点**: 578, 1156, 1734, 2312, 2890, 3468, 4046, 4624, 5202, 5780
- **p_schedule 切换点**: epoch 2 (step 578), epoch 4 (step 1156), epoch 6 (step 1734), epoch 8 (step 2312)

### 3.2 训练进展

| Step | Epoch | loss | loss_action | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr | speed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.17 | 1.379 | 0.063 | 0.701 | 0.0142 | 0.0250 | 10.673 | 4.6e-06 | 0.22 it/s |
| 100 | 0.35 | 0.811 | 0.044 | 0.328 | 0.0094 | 0.0232 | 6.358 | 1.3e-05 | 0.23 it/s |
| 150 | 0.52 | 0.539 | 0.031 | 0.184 | 0.0068 | 0.0257 | 5.373 | 2.2e-05 | 0.23 it/s |
| 200 | 0.69 | 0.418 | 0.024 | 0.130 | 0.0086 | 0.0282 | 4.467 | 3.0e-05 | 0.23 it/s |
| 250 | 0.87 | 0.373 | 0.021 | 0.115 | 0.0079 | 0.0276 | 4.116 | 3.9e-05 | 0.23 it/s |
| 300 | 1.04 | 0.351 | 0.019 | 0.107 | 0.0076 | 0.0292 | 4.109 | 4.7e-05 | 0.23 it/s |
| 350 | 1.21 | 0.330 | 0.018 | 0.100 | 0.0073 | 0.0287 | 4.211 | 5.0e-05 | 0.23 it/s |
| 400 | 1.39 | 0.303 | 0.015 | 0.097 | 0.0086 | 0.0300 | 3.721 | 5.0e-05 | 0.23 it/s |
| 450 | 1.56 | 0.287 | 0.014 | 0.092 | 0.0081 | 0.0287 | 3.436 | 4.9e-05 | 0.23 it/s |
| 500 | 1.73 | 0.263 | 0.013 | 0.088 | 0.0062 | 0.0256 | 3.397 | 4.9e-05 | 0.23 it/s |
| 550 | 1.91 | 0.239 | 0.012 | 0.076 | 0.0042 | 0.0240 | 3.196 | 4.9e-05 | 0.23 it/s |

### 3.3 Milestone: Step 578 (Epoch 2) — Checkpoint #1 + p_schedule Switch #1

- **时间**: 2026-09-25 17:03:48 UTC
- **Checkpoint**: ✅ 已保存 `~/b/Ckp/.../checkpoints/000578/` (18 GB, 含 pretrained_model + training_state)
- **p_schedule 切换**: ✅ `p schedule: 0.20 → 0.40 (epoch 2, interval 2, schedule idx 1/4)`
- **video_decode_error**: 0
- **using_zeros**: 0
- **GPU 内存**: ~60 GB/card（稳定不变）
- **CUDA/OOM/NCCL error**: 无

**观察**:
- 所有 loss 持续下降（loss_action: 0.063→0.012, loss_video: 0.701→0.076）
- grad_norm 从 10.7 降至 3.2，训练稳定
- LR 在 step 289 完成 warmup，峰值 5.0e-05，开始 cosine 衰减
- 速度稳定 0.23 it/s
- GPU 内存 ~60 GB/card，远低于 120 GB 限制
- loss_fast = 0.000, loss_subtask = 0.000（enable_vqa_loss=false，符合预期）
- p_schedule 第 1 次切换成功: 0.20 → 0.40（预期还有 3 次: epoch 4/6/8）

**后续步骤 (step 600-1150):**

| Step | Epoch | loss | loss_action | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr | speed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 600 | 2.08 | 0.241 | 0.012 | 0.079 | 0.0069 | 0.0259 | 2.977 | 4.9e-05 | 0.23 it/s |
| 700 | 2.42 | — | — | — | — | — | — | — | — |
| 850 | 2.94 | 0.211 | 0.009 | 0.075 | 0.0057 | 0.0251 | 2.543 | 4.8e-05 | 0.23 it/s |
| 950 | 3.29 | 0.197 | 0.009 | 0.068 | 0.0076 | 0.0238 | 2.592 | 4.7e-05 | 0.23 it/s |
| 1000 | 3.46 | 0.192 | 0.008 | 0.070 | 0.0068 | 0.0228 | 2.507 | 4.7e-05 | 0.23 it/s |
| 1050 | 3.64 | 0.188 | 0.008 | 0.066 | 0.0054 | 0.0260 | 2.591 | 4.7e-05 | 0.23 it/s |
| 1100 | 3.81 | 0.189 | 0.008 | 0.067 | 0.0062 | 0.0256 | 2.372 | 4.6e-05 | 0.23 it/s |
| 1150 | 3.98 | 0.183 | 0.007 | 0.066 | 0.0063 | 0.0266 | 2.301 | 4.6e-05 | 0.24 it/s |

### 3.4 Milestone: Step 1156 (Epoch 4) — Checkpoint #2 + p_schedule Switch #2

- **时间**: 2026-09-25 17:50:45 UTC
- **Checkpoint**: ✅ 已保存 `~/b/Ckp/.../checkpoints/001156/` (18 GB, 含 pretrained_model + training_state)
- **p_schedule 切换**: ✅ `p schedule: 0.40 → 0.60 (epoch 4, interval 2, schedule idx 2/4)`
- **video_decode_error**: 0
- **using_zeros**: 0
- **GPU 内存**: ~60 GB/card（稳定）
- **CUDA/OOM/NCCL error**: 无

**观察**:
- loss 从 step 578 到 step 1150 缓慢下降: 0.241 → 0.183
- loss_action 从 0.012 降至 0.007，接近收敛
- loss_video 从 0.079 降至 0.066，趋于稳定
- grad_norm 稳定在 2.3-2.6 范围
- LR 从峰值 5.0e-05 开始 cosine 衰减，当前 4.6e-05
- 速度略有提升 0.23→0.24 it/s
- p_schedule 第 2 次切换成功: 0.40 → 0.60（预期还有 2 次: epoch 6/8）

**后续步骤 (step 1200-1700):**

| Step | Epoch | loss | loss_action | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr | speed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1300 | 4.50 | 0.172 | 0.007 | 0.058 | 0.0069 | 0.0239 | 2.452 | 4.5e-05 | 0.22 it/s |
| 1450 | 5.02 | 0.162 | 0.006 | 0.055 | 0.0072 | 0.0236 | 2.297 | 4.4e-05 | 0.23 it/s |
| 1500 | 5.20 | 0.156 | 0.006 | 0.054 | 0.0060 | 0.0241 | 2.295 | 4.3e-05 | 0.22 it/s |
| 1600 | 5.54 | 0.167 | 0.006 | 0.062 | 0.0058 | 0.0237 | 2.301 | 4.2e-05 | 0.22 it/s |
| 1650 | 5.72 | 0.153 | 0.007 | 0.054 | 0.0045 | 0.0189 | 2.353 | 4.2e-05 | 0.23 it/s |
| 1700 | 5.89 | 0.153 | 0.006 | 0.057 | 0.0055 | 0.0191 | 2.078 | 4.1e-05 | 0.22 it/s |

### 3.5 Milestone: Step 1734 (Epoch 6) — Checkpoint #3 + p_schedule Switch #3

- **时间**: 2026-09-25 18:38:44 UTC
- **Checkpoint**: ✅ 已保存 `~/b/Ckp/.../checkpoints/001734/` (18 GB)
- **p_schedule 切换**: ✅ `p schedule: 0.60 → 0.50 (epoch 6, interval 2, schedule idx 3/4)`
- **video_decode_error**: 0
- **CUDA/OOM/NCCL error**: 无

**累计状态**: 3 checkpoints (578, 1156, 1734), 3 p_schedule switches (0.20→0.40→0.60→0.50)
**下一里程碑**: step 2312 (epoch 8), checkpoint #4, 最后一次 p_schedule switch 0.50→0.30

**后续步骤 (step 1800-2300):**

| Step | Epoch | loss | loss_action | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr | speed |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1800 | 6.23 | 0.146 | 0.006 | 0.061 | 0.0038 | 0.0164 | 2.158 | 4.0e-05 | 0.22 it/s |
| 1950 | 6.75 | 0.138 | 0.005 | 0.057 | 0.0046 | 0.0185 | 1.906 | 3.9e-05 | 0.23 it/s |
| 2050 | 7.10 | 0.137 | 0.005 | 0.058 | 0.0034 | 0.0147 | 2.044 | 3.8e-05 | 0.23 it/s |
| 2150 | 7.45 | 0.138 | 0.005 | 0.057 | 0.0049 | 0.0172 | 2.020 | 3.7e-05 | 0.23 it/s |
| 2250 | 7.79 | 0.134 | 0.005 | 0.056 | 0.0044 | 0.0163 | 1.833 | 3.5e-05 | 0.23 it/s |
| 2300 | 7.97 | 0.131 | 0.005 | 0.055 | 0.0035 | 0.0158 | 1.846 | 3.5e-05 | 0.23 it/s |

### 3.6 Milestone: Step 2312 (Epoch 8) — Checkpoint #4 + p_schedule Switch #4 (最后)

- **时间**: 2026-09-25 19:26:19 UTC
- **Checkpoint**: ✅ 已保存 `~/b/Ckp/.../checkpoints/002312/`
- **p_schedule 切换**: ✅ `p schedule: 0.50 → 0.30 (epoch 8, interval 2, schedule idx 4/4)` — **全部 4 次切换完成**
- **video_decode_error**: 0
- **CUDA/OOM/NCCL error**: 无

**p_schedule 切换历史汇总**:
| # | Epoch | Step | 切换 | 时间 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 2 | 578 | 0.20 → 0.40 | 17:03:48 |
| 2 | 4 | 1156 | 0.40 → 0.60 | 17:50:39 |
| 3 | 6 | 1734 | 0.60 → 0.50 | 18:38:38 |
| 4 | 8 | 2312 | 0.50 → 0.30 | 19:26:08 |

**累计状态**: 4 checkpoints (578, 1156, 1734, 2312), 4 p_schedule switches 全部完成
**剩余**: 6 checkpoints (2890, 3468, 4046, 4624, 5202, 5780), 无更多 p_schedule 切换
**预计完成时间**: ~23:30 UTC（约 4 小时后）

**后续步骤 (step 2400-5800):**

| Step | Epoch | loss | loss_action | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2400 | 8.14 | 0.129 | 0.004 | 0.059 | 0.0039 | 0.0160 | 1.891 | 3.4e-05 |
| 2600 | 9.01 | 0.124 | 0.004 | 0.060 | 0.0046 | 0.0147 | 1.649 | 3.1e-05 |
| 2850 | 9.87 | 0.122 | 0.003 | 0.061 | 0.0029 | 0.0158 | 1.669 | 2.8e-05 |
| 3100 | 10.74 | 0.127 | 0.004 | 0.057 | 0.0048 | 0.0180 | 1.546 | 2.5e-05 |
| 3400 | 11.60 | 0.113 | 0.003 | 0.052 | 0.0038 | 0.0165 | 1.430 | 2.2e-05 |
| 3600 | 12.64 | 0.112 | 0.003 | 0.057 | 0.0031 | 0.0151 | 1.354 | 1.9e-05 |
| 3850 | 13.16 | 0.105 | 0.002 | 0.056 | 0.0034 | 0.0143 | 1.186 | 1.7e-05 |
| 4100 | 14.20 | 0.103 | 0.002 | 0.056 | 0.0028 | 0.0136 | 1.314 | 1.4e-05 |
| 4350 | 15.07 | 0.104 | 0.002 | 0.060 | 0.0048 | 0.0122 | 1.162 | 1.2e-05 |
| 4600 | 15.93 | 0.105 | 0.002 | 0.054 | 0.0033 | 0.0151 | 1.193 | 9.7e-06 |
| 4850 | 16.80 | 0.102 | 0.002 | 0.057 | 0.0031 | 0.0144 | 1.166 | 8.0e-06 |
| 5050 | 17.49 | 0.098 | 0.002 | 0.051 | 0.0031 | 0.0156 | 1.036 | 6.9e-06 |
| 5300 | 18.36 | 0.102 | 0.002 | 0.055 | 0.0032 | 0.0147 | 1.002 | 5.8e-06 |
| 5500 | 19.05 | 0.114 | 0.003 | 0.058 | 0.0042 | 0.0159 | 1.171 | 5.3e-06 |
| 5700 | 19.57 | 0.105 | 0.002 | 0.053 | 0.0037 | 0.0158 | 1.155 | 5.1e-06 |
| 5800 | 19.92 | 0.098 | 0.002 | 0.051 | 0.0049 | 0.0156 | 1.030 | 5.0e-06 |

### 3.7 Checkpoint 保存汇总

| # | Step | Epoch | 保存时间 | 大小 |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 578 | 2 | 17:03:48 | 18 GB |
| 2 | 1156 | 4 | 17:50:45 | 18 GB |
| 3 | 1734 | 6 | 18:38:44 | 18 GB |
| 4 | 2312 | 8 | 19:26:19 | 18 GB |
| 5 | 2890 | 10 | 20:13:32 | 18 GB |
| 6 | 3468 | 12 | 21:00:18 | 18 GB |
| 7 | 4046 | 14 | 21:47:06 | 18 GB |
| 8 | 4624 | 16 | 22:33:45 | 18 GB |
| 9 | 5202 | 18 | 23:20:36 | 18 GB |
| 10 | 5780 | 20 | 00:06:24 | 18 GB |

---

## 4. 训练完成

- **完成时间**: 2026-09-26 00:10:09 UTC
- **总训练时间**: 7 小时 52 分 (16:18:45 → 00:10:09)
- **Exit code**: 0
- **最终 loss**: loss=0.098, loss_action=0.002, loss_video=0.051
- **最终 grad_norm**: 1.030
- **最终 LR**: 5.0e-06

### 4.1 Post-training 操作

- **bigmatrix**: ✅ 已启动 (PID 858304, ~104 GB/card)
- **日志归档**: ✅ `~/b/Ckp/4dwvlaFrkCubBx0924_sft_LOG_20260926_001015.tar` (184 KB)

---

## 5. 验收结果

### 5.1 A-Level 验收 (必须全部通过)

| # | 检查项 | 结果 | 详情 |
|:---:|:---|:---:|:---|
| A1 | Exit code = 0 | ✅ PASS | |
| A2 | 最终 checkpoint 存在 | ✅ PASS | 005780 (18 GB, pretrained_model + training_state) |
| A3 | 无 CUDA errors | ✅ PASS | 0 |
| A4 | 无 video_decode_error | ✅ PASS | 0 |
| A5 | 无 NaN/Inf grad_norm | ✅ PASS | 所有 grad_norm 范围: 1.0-10.7 |
| A6 | 全部 10 个 checkpoint | ✅ PASS | 578, 1156, 1734, 2312, 2890, 3468, 4046, 4624, 5202, 5780 |

**A-Level: 6/6 通过 ✅**

### 5.2 B-Level 验收 (建议通过)

| # | 检查项 | 结果 | 详情 |
|:---:|:---|:---:|:---|
| B1 | final loss_action < 0.1 | ✅ PASS | 0.002 |
| B2 | 训练时间 < 10h | ✅ PASS | ~7:52 |
| B3 | 4 次 p_schedule 切换 | ✅ PASS | 0.20→0.40→0.60→0.50→0.30 |
| B4 | GPU 内存 < 120 GB | ✅ PASS | ~60 GB/card |

**B-Level: 4/4 通过 ✅**

### 5.3 训练曲线总结

- **loss_action**: 0.063 (step 50) → 0.002 (step 5780), 下降 97%
- **loss_video**: 0.701 (step 50) → 0.051 (step 5780), 下降 93%
- **loss_kpt_cur**: 0.0142 → 0.0049
- **loss_kpt_fut**: 0.0250 → 0.0156
- **grad_norm**: 10.7 → 1.0, 训练全程稳定
- **速度**: 0.22-0.24 it/s, 稳定无退化
- **GPU 内存**: ~60 GB/card, 全程稳定
- **zero video_decode_error**, **zero using_zeros**, **zero CUDA/OOM/NCCL errors**

