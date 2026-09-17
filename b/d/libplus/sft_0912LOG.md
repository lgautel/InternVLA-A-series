# Libero-Plus OpVLA SFT 实施操作日志（2026-09-12）

> **方案**: [`sft.md`](sft.md)  
> **EXPR_NAME**: `4dwvlaOpvlaLibplusKpt0911`  
> **日期**: 2026-09-12  
> **环境**: `/B/VENV/itnvla15rbt20/`，`HF_HOME=/B/VENV/hf_home`  
> **上下文**: 9/11 训练因监控误杀 + FileExistsError 死循环失败（详见 `sft_0911LOG.md`）。  
>            9/12 已修复 stall 检测（step-based）、fresh start OUTPUT_DIR、LOG_FREQ 等问题，本次重新执行完整流程。

---

## 0. 任务目标

按 `sft.md` 附录 B Checklist 执行完整训练流程：前置检查 → 单元测试 → 监控测试 → WAN Smoke → SMOKE 100 步 → 正式训练 53450 步（50 epoch），所有测试和验收通过，训练成功完成。

---

## 1. 清空 GPU（2026-09-12 启动前）

**操作**:
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

**结果**: 8× GPU memory.used = 0 MiB。无残留进程。**无需 pkill**。

---

## 2. 环境安装与前置检查（§10.1/§10.2）

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/hf_home HF_LEROBOT_HOME=${HF_HOME}/lerobot NCCL_TUNER_PLUGIN=/dev/null
cd /B/SRC/itvlaGpLibPlus
pip install -e .
```

**检查项**:

| 项目 | 结果 |
|------|------|
| GPU | 8 × H200, 143771 MiB ✅ |
| InternVLA-A1.5-base | `/B/VENV/hf_home/ckpts/InternVLA-A1.5-base/config.json` ✅ |
| WAN2.2 | `/B/VENV/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` ✅ |
| 数据集 | episodes=1693, frames=273465, fps=10, robot_type=panda ✅ |
| Schema | `image` → `image0`, `image2` → `image1` ✅ |
| PyTorch | 2.10.0+cu128, 8 GPUs ✅ |
| Transformers | 5.2.0 ✅ |
| Qwen3.5 patch | OK ✅ |
| flash-attn | 2.8.3 ✅ |
| FLA | OK ✅ |
| lerobot | `/B/SRC/itvlaGpLibPlus/src/lerobot/__init__.py` ✅ |

---

## 3. 单元测试 §12.1

### 测试 1: Schema — ✅
```
✅ panda schema OK
```

### 测试 2: Keypoint reshape 56D→8×7D — ✅
```
✅ keypoint reshape OK, samples=273465
```

### 测试 3: Transform Pipeline — ✅
```
✅ pipeline: ['NormalizeTransformFn', 'Extract3DKeypointTransformFn']
```

### 测试 4: Delta Timestamps — ✅（经 fix）

**首次失败**:
```
ImportError: cannot import name '_build_delta_timestamps' from 'lerobot.datasets.factory'
```

**根因**: `sft.md` 中测试 4 使用的函数名 `_build_delta_timestamps` 已在代码中重命名为 `resolve_delta_timestamps`。这是 sft.md 文档中的过时引用，代码侧无 bug。

**Fix**: 测试使用正确的函数名 `resolve_delta_timestamps`，并需传入 `ds_meta.robot_type='panda'` 和完整的 features（含 image keys）。

**复测**:
```
✅ delta timestamps OK, first= -20.0
```

---

## 4. 监控 & Auto-recover 测试 §12.1b — ✅

```bash
bash tests/test_monitor_stall_detection.sh
```

**结果**: 35 passed, 0 failed ✅

覆盖：stall 检测逻辑、fresh start OUTPUT_DIR、checkpoint 跨目录搜索、参数兼容性。

---

## 5. WAN Smoke §12.2 — ✅

**命令**:
```bash
WAN_SMOKE=1 bash launch/libplus_sft_launch.sh
```

**结果**:
```
step:1.0 | loss:10.992 | loss_action:0.339 | loss_kpt_cur:1.0369 | loss_kpt_fut:1.1254 | loss_video:0.215
step:2.0 | loss:28.665 | loss_action:0.168 | loss_kpt_cur:7.7332 | loss_kpt_fut:7.4946 | loss_video:0.172
post_check: video_decode_error=0 using_zeros=0 exit=0
```

| 检查项 | 结果 |
|--------|------|
| 无 Traceback | ✅ |
| 所有 loss 出现 | ✅ action/kpt_cur/kpt_fut/video/vqa/fast |
| 无 NaN/Inf | ✅ |
| video_decode_error=0 | ✅ |
| Exit code 0 | ✅ |

---

## 6. SMOKE 100 步 §12.3 — ✅

**命令**:
```bash
SMOKE=1 bash launch/libplus_sft_launch.sh
```

**结果** (~1.5 min, 1 GPU, BS=2):

| Step | loss | loss_action | loss_kpt_cur | loss_kpt_fut | loss_video |
|------|------|-------------|--------------|--------------|------------|
| 10 | 9.796 | 0.222 | 1.1247 | 1.2048 | 0.227 |
| 50 | 6.005 | 0.190 | 0.1126 | 0.1711 | 0.168 |
| 100 | 5.518 | 0.159 | 0.0872 | 0.1367 | 0.171 |

```
post_check: video_decode_error=0 using_zeros=0 exit=0
```

Loss 下降趋势正常。无 Traceback/NaN。

---

## 7. 正式训练启动（8×H200）

**启动时间**: 2026-09-12 08:52:49

**命令**:
```bash
nohup bash launch/libplus_sft_launch.sh > /tmp/libplus_sft_outer_0912.log 2>&1 &
```

**关键参数**:

| 参数 | 值 |
|------|-----|
| GPU | 8 × H200 (138 GB/GPU, ~96% 显存占用) |
| Batch size | 32/GPU × 8 GPU = 256 (EBS) |
| Steps | 53450 (50 epochs) |
| Speed | 0.21 iters/s (~4.79s/step) |
| ETA | ~70h |
| SAVE_FREQ | 5345 (每 5 epoch) |
| LOG_FREQ | 100 |
| STALE_THRESHOLD | 1800s |
| MONITOR_INTERVAL | 900s |

**关键路径**:

| 项目 | 路径 |
|------|------|
| Outer PID | 1471839 |
| Training PID | 1471852 |
| OUTPUT_DIR | `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft` |
| LOG_FILE | `/B/Log/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49/train.log` |
| Outer log | `/tmp/libplus_sft_outer_0912.log` |

### 7.1 训练进度

| 时间 | Step | loss | loss_action | loss_kpt_cur | loss_kpt_fut | loss_video | lr | grdn | ETA |
|------|------|------|-------------|--------------|--------------|------------|-----|------|-----|
| 09:03 | 100 | 8.527 | 0.272 | 0.5579 | 0.6730 | 0.148 | 2.6e-06 | 53.800 | 72:02h |
| 09:11 | 200 | 4.980 | 0.124 | 0.1017 | 0.1909 | 0.129 | 7.6e-06 | 6.799 | 70:51h |
| 09:19 | 300 | 4.237 | 0.093 | 0.0447 | 0.1080 | 0.123 | 1.3e-05 | 7.010 | 69:17h |
| 09:27 | 400 | 3.947 | 0.081 | 0.0285 | 0.0714 | 0.119 | 1.8e-05 | 6.969 | 69:35h |
| 09:35 | 500 | 3.819 | 0.075 | 0.0237 | 0.0604 | 0.116 | 2.3e-05 | 6.458 | 69:23h |
| 09:43 | 600 | 3.741 | 0.074 | 0.0204 | 0.0565 | 0.113 | 2.8e-05 | 5.858 | 69:24h |
| 09:51 | 700 | 3.695 | 0.072 | 0.0190 | 0.0556 | 0.112 | 3.3e-05 | 5.384 | 68:34h |
| 09:59 | 800 | 3.671 | 0.070 | 0.0198 | 0.0562 | 0.111 | 3.8e-05 | 5.248 | 68:47h |
| 10:07 | 900 | 3.627 | 0.070 | 0.0168 | 0.0541 | 0.109 | 4.3e-05 | 4.703 | 69:26h |
| 10:15 | **1000** | **3.608** | 0.069 | 0.0191 | 0.0553 | 0.109 | **4.8e-05** | 5.388 | 69:31h |
| 10:23 | 1100 | 3.539 | 0.066 | 0.0159 | 0.0533 | 0.107 | **5.0e-05** | 5.466 | 67:51h |

| 10:31 | 1200 | 3.493 | 0.065 | 0.0143 | 0.0541 | 0.105 | 5.0e-05 | 5.486 | 68:08h |
| 10:39 | 1300 | 3.470 | 0.064 | 0.0133 | 0.0526 | 0.104 | 5.0e-05 | 5.199 | 68:16h |
| 10:47 | 1400 | 3.424 | 0.062 | 0.0142 | 0.0526 | 0.104 | 5.0e-05 | 4.788 | 67:47h |
| 10:55 | **1500** | **3.340** | 0.059 | 0.0124 | 0.0503 | 0.103 | 5.0e-05 | 4.633 | 68:38h |

**⭐ Step 1100**: Warmup 完成 (lr=5.0e-05)，进入 epoch 1.03（第一个 epoch 完成）。data_s=0.150 为 epoch 边界 data reload 正常现象。

| 11:03 | 1600 | 3.305 | 0.057 | 0.0131 | 0.0515 | 0.099 | 5.0e-05 | 4.634 | 67:30h |
| 11:11 | 1700 | 3.218 | 0.054 | 0.0124 | 0.0491 | 0.100 | 5.0e-05 | 4.294 | 67:30h |
| 11:19 | 1800 | 3.178 | 0.052 | 0.0144 | 0.0495 | 0.098 | 5.0e-05 | 4.948 | 67:05h |
| 11:27 | 1900 | 3.122 | 0.051 | 0.0129 | 0.0502 | 0.100 | 5.0e-05 | 4.437 | 67:20h |
| 11:35 | **2000** | **3.031** | 0.046 | 0.0131 | 0.0492 | 0.097 | 5.0e-05 | 4.164 | 66:51h |

**⭐ Step 1100**: Warmup 完成 (lr=5.0e-05)，进入 epoch 1.03（第一个 epoch 完成）。

**⭐ Step 1500**: 安全通过 9/11 训练误杀点。9/11 训练在此处被 mtime-based stall detection 错误终止并陷入 FileExistsError 死循环。本次 step-based detection 在该点监控报告 "Healthy"，fix 在生产环境验证通过。

| 11:43 | 2100 | 3.004 | 0.045 | 0.0138 | 0.0499 | 0.098 | 4.9e-05 | 5.857 | 67:04h |
| 11:51 | 2200 | 2.898 | 0.042 | 0.0136 | 0.0512 | 0.097 | 4.9e-05 | 4.656 | 67:10h |
| 12:07 | 2400 | 2.757 | 0.037 | 0.0132 | 0.0477 | 0.096 | 4.9e-05 | 4.593 | 66:52h |
| 12:22 | 2600 | 2.657 | 0.033 | 0.0146 | 0.0463 | 0.093 | 4.9e-05 | 5.150 | 66:14h |
| 12:38 | 2800 | 2.575 | 0.031 | 0.0160 | 0.0451 | 0.094 | 4.9e-05 | 4.343 | 65:56h |
| 12:54 | **3000** | **2.502** | 0.029 | 0.0165 | 0.0417 | 0.094 | 4.9e-05 | 4.490 | 65:35h |

**⭐ Step 2000**: 训练已稳定运行 2h40m。loss 从 8.527 降至 3.031 (↓64.5%)。进入 lr decay 区间 (step 1000-31000)。

**⭐ Step 2200**: 完成 epoch 2。data_s=0.150 确认 epoch 边界 reload。

| 13:10 | 3200 | 2.461 | 0.029 | 0.0165 | 0.0376 | 0.093 | 4.9e-05 | 4.652 | 65:12h |
| 13:34 | 3500 | 2.287 | 0.026 | 0.0168 | 0.0382 | 0.093 | 4.9e-05 | 6.009 | 65:39h |
| 14:14 | **4000** | **2.193** | 0.024 | 0.0172 | 0.0346 | 0.089 | 4.8e-05 | 5.025 | 64:34h |

**⭐ Step 3000**: 训练已稳定运行 4h。loss 2.502 (↓70.6%)。

**⭐ Step 3200**: 完成 epoch 3 (epoch:3.00)。

| 14:46 | 4400 | 2.043 | 0.021 | 0.0171 | 0.0331 | 0.089 | 4.8e-05 | 5.500 | 63:42h |
| 15:33 | **5000** | **1.986** | 0.020 | 0.0158 | 0.0327 | 0.087 | 4.7e-05 | 5.314 | 63:19h |
| 15:57 | 5300 | 1.916 | 0.019 | 0.0164 | 0.0333 | 0.088 | 4.7e-05 | 5.239 | 62:48h |

**⭐ Step 4000**: 训练已稳定运行 5h18m (↓74.3%)。sample 突破 1M。

**⭐ Step 4300**: 完成 epoch 4 (epoch:4.03)。

**⭐ Step 5000**: loss 稳定在 ~2.0 以下。ETA 63h。

**⭐ Step 5345 (epoch 5): 第一个 checkpoint 保存成功 ✅**
- 保存时间: 16:00-16:01 UTC
- 路径: `checkpoints/005345/pretrained_model/`
- 内容: `model.safetensors` (6.3 GB), `config.json`, `stats.json`, `train_config.json`
- `training_state/` 也已保存（用于 resume）
- 总大小: 18 GB
- §12.4 Checklist: checkpoint 005345 ✅

| 16:17 | 5500 | 1.830 | 0.019 | 0.0166 | 0.0315 | 0.087 | 4.6e-05 | 5.599 | 62:29h |
| 16:57 | **6000** | **1.807** | 0.018 | 0.0174 | 0.0301 | 0.082 | 4.6e-05 | 5.827 | 62:02h |

**⭐ Step 5400**: 完成 epoch 5。data_s=0.154 确认 epoch 边界。

| 17:37 | 6500 | 1.686 | 0.017 | 0.0164 | 0.0315 | 0.082 | 4.5e-05 | 6.084 | 62:02h |
| 18:16 | 7000 | 1.668 | 0.016 | 0.0162 | 0.0315 | 0.083 | 4.4e-05 | 5.774 | 60:22h |
| 18:56 | 7500 | 1.612 | 0.015 | 0.0161 | 0.0298 | 0.082 | 4.3e-05 | 5.618 | 59:44h |
| 19:35 | **8000** | **1.546** | 0.015 | 0.0190 | 0.0288 | 0.081 | 4.3e-05 | 5.739 | 58:51h |

**⭐ Step 6000**: 训练已稳定运行 8h。loss 1.807。sample 突破 2M。

**⭐ Step 6500**: 完成 epoch 6。

**⭐ Step 7500**: 完成 epoch 7。

| 20:23 | 8600 | 1.484 | 0.014 | 0.0159 | 0.0305 | 0.080 | 4.2e-05 | 5.878 | 58:09h |
| 20:54 | 9000 | 1.427 | 0.014 | 0.0182 | 0.0284 | 0.079 | 4.1e-05 | 6.112 | 57:21h |
| 21:49 | 9700 | 1.342 | 0.013 | 0.0163 | 0.0297 | 0.081 | 3.9e-05 | 6.230 | 56:38h |
| 22:13 | **10000** | **1.322** | 0.012 | 0.0166 | 0.0292 | 0.079 | 3.9e-05 | 6.300 | 56:37h |

**⭐ Step 8000**: 训练已稳定运行 10h40m，15% 完成。loss 1.546。

**⭐ Step 8600**: 完成 epoch 8。

**⭐ Step 9700**: 完成 epoch 9。

| 22:59 | 10600 | 1.336 | 0.012 | 0.0175 | 0.0291 | 0.078 | 3.8e-05 | 6.151 | 54:55h |

**⭐ Step 10000**: 训练已稳定运行 13h17m，18.7% 完成。loss 1.322 (↓84.5%)。sample 突破 3M。lr 已衰减至 3.9e-05。

**⭐ Step 10690 (epoch 10): 第二个 checkpoint 保存成功 ✅**
- 保存时间: 23:06-23:07 UTC
- 路径: `checkpoints/010690/pretrained_model/`
- 内容: `model.safetensors` (6.3 GB), `config.json`, `stats.json`, `train_config.json`
- `training_state/` 已保存
- 总大小: 18 GB
- 累计 checkpoints: 36 GB (005345 + 010690)
- 20% 训练完成

| 23:19 | 10800 | 1.205 | 0.011 | 0.0171 | 0.0275 | 0.077 | 3.7e-05 | 6.532 | 54:42h |
| 23:34 | 11000 | 1.221 | 0.012 | 0.0148 | 0.0301 | 0.078 | 3.7e-05 | 6.405 | 54:55h |
| 00:53 | **12000** | **1.130** | 0.011 | 0.0147 | 0.0295 | 0.078 | 3.5e-05 | 6.741 | 53:27h |

**⭐ Step 10800**: 完成 epoch 10。loss 从 1.33 降至 1.205（epoch boundary 效应）。

**⭐ Step 11800**: 完成 epoch 11。

**⭐ Step 12000**: 训练已稳定运行 16h，22.4% 完成。loss 1.130 (↓86.7%)。ETA 53h。lr 衰减至 3.5e-05。所有监控连续 Healthy，训练在夜间完全稳定。

| 01:40 | **13000** | **1.044** | 0.010 | 0.0165 | 0.0289 | 0.076 | 3.2e-05 | 7.150 | 52:20h |
| 02:50 | 13500 | 1.049 | 0.010 | 0.0160 | 0.0279 | 0.076 | 3.1e-05 | 7.057 | 51:47h |

**⭐ Step 12700**: 完成 epoch 12。

**⭐ Step 13000**: 训练稳定运行 ~18h，24.3% 完成。loss 1.044 (↓87.8%)。lr 衰减至 3.2e-05。

**⭐ Step 13500** (当前): 25.3% 完成。loss 1.049，与 13000 持平。所有 8×H200 GPU 100% 利用率 (138 GB/GPU)。下一个 checkpoint 在 step 16035 (epoch 15)，预计 ~06:12 UTC Sep 13 (~3.3h 后)。

**趋势分析** (step 100→1000):
- loss: 8.527 → 3.608 (↓57.7%)，下降正常，step 600 后趋缓（符合预期）
- loss_action: 0.272 → 0.069 (↓74.6%)，动作预测快速收敛
- loss_kpt_cur: 0.5579 → 0.0191 (↓96.6%)，当前关键点已基本收敛
- loss_kpt_fut: 0.6730 → 0.0553 (↓91.8%)，未来关键点收敛良好
- loss_video: 0.148 → 0.109 (↓26.4%)，视频预测下降较缓（WAN frozen，符合预期）
- loss_vqa: 3.756 → 2.676 (↓28.7%)，VQA/language loss 稳步下降
- loss_fast: 3.778 → 2.760 (↓26.9%)，FAST token loss 跟 vqa 同步
- grdn: 53.8 → 5.4，梯度初始震荡后回落正常，step 500 后稳定在 5-7
- lr: warmup 基本完成 (2.6e-06 → 4.8e-05，峰值 5e-05 at step 1000)
- 速度: 稳定 0.21 iters/s (~4.72s/step)，ETA ~69h
- 无 NaN/Inf，无 Traceback，无 CUDA OOM
- 监控: 6 次连续 "Healthy" 检查，step-based stall detection 运行完美

### 7.2 监控检查

| 时间 | 监控结果 |
|------|----------|
| 09:07:50 | ✅ Healthy: step=100/53450 |
| 09:22:50 | ✅ Healthy: step=300/53450 |
| 09:37:50 | ✅ Healthy: step=500/53450 |
| 09:52:50 | ✅ Healthy: step=700/53450 |
| 10:07:50 | ✅ Healthy: step=900/53450 |
| 10:22:51 | ✅ Healthy: step=1000/53450 |
| 10:37:51 | ✅ Healthy: step=1200/53450 |
| 10:52:51 | ✅ Healthy: step=1400/53450 |
| ... | (中间省略，每 15min 均 Healthy) |
| 02:38:03 | ✅ Healthy: step=13300/53450 |

Step-based stall detection 正常工作：每 15 min (MONITOR_INTERVAL=900s) 检查一次，能正确识别 step 进展。从 09:07 至 02:38 共 ~17.5h 连续 Healthy，零误报。

### 7.3 Checkpoint 进度

| 预计时间 | Step | 事件 | 状态 |
|----------|------|------|------|
| 09-12 15:55 | 5345 | epoch 5 checkpoint | ✅ 已保存 |
| 09-12 23:06 | 10690 | epoch 10 checkpoint | ✅ 已保存 |
| ~09-13 06:12 | 16035 | epoch 15 checkpoint | ⏳ 待保存 |
| ~09-13 13:18 | 21380 | epoch 20 checkpoint | ⏳ |
| ~09-13 20:24 | 26725 | epoch 25 checkpoint | ⏳ |
| ~09-14 03:30 | 32070 | epoch 30 checkpoint | ⏳ |
| ~09-14 10:36 | 37415 | epoch 35 checkpoint | ⏳ |
| ~09-14 17:42 | 42760 | epoch 40 checkpoint | ⏳ |
| ~09-15 00:48 | 48105 | epoch 45 checkpoint | ⏳ |
| ~09-15 07:54 | 53450 | epoch 50 (完成) | ⏳ |

---
