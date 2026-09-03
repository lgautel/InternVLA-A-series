# GeoP Phase 2 SFT 本机实施日志

> 对应方案: [`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.md)  
> 实施日期: 2026-08-11  
> 环境: venv `/tmp/itnvla15rbt20/`，8× NVIDIA H200（~140 GB/卡）

---

## 总览

| 阶段 | 状态 | 备注 |
|:---|:---:|:---|
| Preflight | ✅ | WAN 缺失、launch 未 chmod → 已修 |
| WAN 下载 | ✅ | ~32 GB → `${HF_HOME}/hub/Wan2.2-TI2V-5B/` |
| WAN Smoke (1 GPU × 2 step) | ✅ | exit=0 |
| Smoke 100 step (1 GPU) | ✅ | 四项 loss 均 >0 |
| 8 卡正式 10k | ✅ | 04:30–08:41 UTC，~4.1 h，exit=0 |
| Checkpoint 评估 | ✅ | Open-loop @ kptsim 数据，4 ckpt 对比 |
| GCS 备份 | ✅ | → `gs://physical-ai-data-eu/VENV/tmp/itnvla0801116/` |

---

## 1. 操作日志（按时间顺序）

### 1.1 Preflight（第 1 次）

**时间**: 2026-08-11  
**操作**: 按 sft_rbt2.md §8 执行 Preflight checklist  

**结果**:

| 检查项 | 结果 |
|:---|:---:|
| torch 2.10.0+cu128, cuda=8 | ✅ |
| WAN `Wan2.2_VAE.pth` | ❌ **MISSING** → 已下载 |
| 数据 symlink | ✅ |
| Warmup ckpt@400 | ✅ |
| launch 可执行 | ❌ → `chmod +x` |
| 8×H200 空闲 | ✅ |

---

### 1.2 WAN 权重下载

**时间**: 2026-08-11  
**操作**: 按 sft_rbt2.md §6 下载 Wan2.2-TI2V-5B  
**目标**: `/tmp/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/`（~32 GB）  
**结果**: ✅

---

### 1.3 WAN Smoke（1 GPU × 2 step）

**时间**: 2026-08-11 04:19–04:24  
**命令**: `WAN_SMOKE=1 bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh`  
**日志**: `/tmp/phase2_wan_smoke_kptsim.log`  
**JOB**: `2026_08_11_04_19_42-internvla_a1_5-geop-phase2-wan-smoke-kptsim-voxel`  
**结果**: ✅ `End of training`，`post_check: video_decode_error=0 using_zeros=0 exit=0`

---

### 1.4 Smoke 100 step（1 GPU）

**时间**: 2026-08-11 04:24–04:30  
**命令**: `SMOKE=1 bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh`  
**日志**: `/tmp/phase2_smoke100_kptsim.log`  
**JOB**: `2026_08_11_04_24_38-internvla_a1_5-geop-phase2-smoke100-kptsim-voxel`  
**step@100 loss**: total=3.315, action=0.086, vqa=2.163, video=0.284, kpt_cur=0.0021, kpt_fut=0.0041  
**结果**: ✅ exit=0

---

### 1.5 8 卡正式 10k 训练

**时间**: 2026-08-11 04:30–08:41（wall ~4 h 11 min）  
**命令**: `bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh`  
**日志**: `/tmp/phase2_kptsim_8g_10k.log`  
**JOB**: `2026_08_11_04_30_29-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k`  
**吞吐**: ~0.80 it/s（BS=16×8）  
**结果**: ✅ `End of training`，`post_check: video_decode_error=0 using_zeros=0 exit=0`

**Checkpoint 保存路径**:

```
outputs/internvla_a1_5/2026_08_11_04_30_29-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/checkpoints/
├── 002500/pretrained_model
├── 005000/pretrained_model
├── 007500/pretrained_model
├── 010000/pretrained_model
└── last -> 010000
```

**各 save 点训练 loss 快照**（日志 step 对齐）:

| Step | loss | loss_action | loss_vqa | loss_video | loss_kpt_cur | loss_kpt_fut |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2500 | 0.447 | 0.004 | 0.291 | 0.110 | 0.0014 | 0.0020 |
| 5000 | 0.184 | 0.002 | 0.063 | 0.101 | 0.0011 | 0.0017 |
| 7500 | 0.118 | 0.001 | 0.010 | 0.093 | 0.0012 | 0.0018 |
| 10000 | 0.109 | 0.001 | 0.004 | 0.094 | 0.0012 | 0.0017 |

**已知非致命现象**: 从 Warmup ckpt@400 加载时 WAN DiT / `learnable_to_wan_proj` 大量 Missing keys（预期，WAN 从 hub 单独加载）。

---

## 2. 错误记录与修复

| # | 现象 | 根因 | Fix |
|:---:|:---|:---|:---|
| 1 | Preflight WAN MISSING | Phase 2 前未下载 Wan2.2-TI2V-5B | §6 下载至 venv HF_HOME |
| 2 | launch 不可执行 | 新建脚本未 chmod | `chmod +x launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh` |
| 3 | Open-loop eval `TypeError: sample_actions() got unexpected keyword argument 'his_kpts'` | GeoP 模型默认走 optimized backend，不支持 kpt 输入 | 修改 `tests/openloop_internvla_a1_5.py`：`enable_keypoint_predictor=True` 时强制 `inference_backend=standard` |

---

## 3. 关键路径汇总

| 用途 | 路径 |
|:---|:---|
| venv | `/tmp/itnvla15rbt20/` |
| 源码 | `/home/a26113/SRC/InternVLA-A-series/` |
| Warmup ckpt@400 | `outputs/internvla_a1_5/2026_08_11_03_04_19-internvla_a1_5-geop-phase1-kpt-warmup-kptsim-voxel-8g/checkpoints/000400/pretrained_model` |
| Phase 2 JOB | `outputs/internvla_a1_5/2026_08_11_04_30_29-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/` |
| Launch | `launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh` |
| WAN | `/tmp/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/` |
| GCS 备份 | `gs://physical-ai-data-eu/VENV/tmp/itnvla0801116/InternVLA-A-series/`（~154 GB） |

---

## 4. 文件变更清单

| 路径 | 操作 | 原因 |
|:---|:---|:---|
| **本文** | 新增/更新 | Phase 2 实施日志 |
| `tests/openloop_internvla_a1_5.py` | 修改 | GeoP ckpt open-loop 评估需 standard backend |

---

## 5. Checkpoint 评估与推荐

### 5.1 评估方法

因本机 `third_party/RoboTwin` 未初始化，**未跑 RoboTwin 仿真成功率**；改用与训练同分布的 **Open-loop action MSE**（`tests/openloop_internvla_a1_5.py`）作为 checkpoint 对比指标。

| 参数 | 值 |
|:---|:---|
| 数据 | `stack_bowls_three_kptsim_lrbv30`（5 episodes × 8 samples，stride=50） |
| Backend | `standard` + `action_loss_only=True`（含 kpt 分支，不加载 WAN） |
| 对比 ckpt | 002500 / 005000 / 007500 / 010000 |
| 原始 metrics | `/tmp/phase2_ckpt_eval2/{002500,005000,007500,010000}/metrics.json` |

### 5.2 Open-loop MSE 结果

| Checkpoint | Step | Avg MSE (total) | Avg MSE (joints) | Avg MSE (gripper) | 相对 2500 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **002500** | 2500 | **0.001576** | **0.001430** | 0.002457 | **基准（最优）** |
| 005000 | 5000 | 0.001868 | 0.001818 | **0.002167** | +18.5% |
| 007500 | 7500 | 0.002033 | 0.001973 | 0.002394 | +28.9% |
| 010000 | 10000 | 0.002137 | 0.002099 | 0.002362 | +35.6% |

> 数值越低越好。MSE 在 unnormalized action 空间（6 joint + gripper × 2 arm）。

### 5.3 训练 loss vs Open-loop 的背离

- **训练 weighted loss** 从 step 2500→10000 持续下降（0.45→0.11），主要由 **VQA/FAST**（0.29→0.004）和 **action×10**（0.04→0.01 加权项）驱动。
- **Open-loop action MSE** 却从 2500 起**单调变差**，说明继续全量微调时，VLM/VQA/Video 分支占据了更多容量，**损害了 action flow-matching 在演示数据上的拟合**。
- Kpt loss（cur/fut ≈ 0.001–0.002）各 ckpt 差距很小，**不是区分 action 质量的主因**。
- step 7500–10000 区间训练 loss 已平台（~0.11–0.12），Open-loop MSE 继续恶化 → **明显过拟合/多任务干扰**。

### 5.4 推荐结论

| 优先级 | Checkpoint | 路径 | 理由 |
|:---:|:---|:---|:---|
| **首选** | **@2500** | `.../checkpoints/002500/pretrained_model` | Open-loop action MSE **最低**（比 @10000 低 35.6%）；VQA/video 已初步对齐（loss_vqa≈0.29）且 action 尚未被后续多任务训练侵蚀 |
| 备选 | @5000 | `.../checkpoints/005000/pretrained_model` | VQA 进一步收敛（loss_vqa≈0.06），Open-loop MSE 仅比 @2500 高 18.5%；若更看重语言/视频辅助分支可选用 |
| 不推荐（action） | @7500 / @10000 | `007500` / `010000` | Open-loop MSE 持续恶化；@10000 虽为手册默认终点且训练 loss 最低，**action 开环精度最差** |

**部署建议**:

```bash
CKPT=/home/a26113/SRC/InternVLA-A-series/outputs/internvla_a1_5/\
2026_08_11_04_30_29-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k/\
checkpoints/002500/pretrained_model
```

后续若需 RoboTwin 仿真成功率验证，在补齐 `third_party/RoboTwin` 后按 sft_rbt2.md §13 对 **@2500 与 @5000** 做 A/B eval（task_idx=46, `stack_bowls_three`）。

---

## 6. GCS 备份

**时间**: 2026-08-11 08:44–08:48  
**命令**:

```bash
gcloud storage cp -r /home/a26113/SRC/InternVLA-A-series \
  gs://physical-ai-data-eu/VENV/tmp/itnvla0801116/
```

**结果**: ✅ exit=0，目标前缀 `gs://physical-ai-data-eu/VENV/tmp/itnvla0801116/InternVLA-A-series/`（含 outputs/checkpoints，合计 ~154 GB）。

---

*日志更新: 2026-08-11 | Phase 2 10k 完成 + ckpt 评估 + GCS 备份*
