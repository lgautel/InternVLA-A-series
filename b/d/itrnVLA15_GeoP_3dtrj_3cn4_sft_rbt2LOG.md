# GeoP Phase 2 SFT 本机实施日志

> 对应方案: [`itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.md`](itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.md)  
> 实施日期: 2026-08-11  
> 环境: venv `/tmp/itnvla15rbt20/`，8× NVIDIA H200（~140 GB/卡）

---

## 总览

| 阶段 | 状态 | 备注 |
|:---|:---:|:---|
| Preflight | 进行中 | |
| WAN 下载 | 待执行 | Preflight 发现 WAN MISSING |
| WAN Smoke (1 GPU × 2 step) | 待执行 | |
| Smoke 100 step (1 GPU) | 待执行 | |
| 8 卡正式 10k | 待执行 | |

---

## 1. 操作日志（按时间顺序）

### 1.1 Preflight（第 1 次）

**时间**: 2026-08-11  
**操作**: 按 sft_rbt2.md §8 执行 Preflight checklist  
**命令**: 见方案 §8（`torch/lerobot`、WAN、DATA、WARMUP_CKPT@400、launch、GPU）

**结果**:

| 检查项 | 结果 |
|:---|:---:|
| torch 2.10.0+cu128, cuda=8 | ✅ |
| WAN `Wan2.2_VAE.pth` | ❌ **MISSING** |
| 数据 symlink | ✅ |
| Warmup ckpt@400 | ✅ |
| launch 可执行 | ❌ 脚本存在但未 `chmod +x` |
| 8×H200 空闲 | ✅ |

**结论**: 需先下载 WAN 权重（§6），并对 launch 脚本 `chmod +x`，再进入 Smoke。

---

## 2. 错误记录与修复

（实施过程中追加）

---

## 3. 关键路径汇总

| 用途 | 路径 |
|:---|:---|
| venv | `/tmp/itnvla15rbt20/` |
| 源码 | `/tmp/SRC/InternVLA-A-series/` |
| Warmup ckpt@400 | `.../2026_08_11_03_04_19-...-geop-phase1-kpt-warmup-kptsim-voxel-8g/checkpoints/000400/pretrained_model` |
| Launch | `launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh` |
| WAN 目标 | `/tmp/itnvla15rbt20/var/hf_home/hub/Wan2.2-TI2V-5B/` |

---

## 4. 文件变更清单

| 路径 | 操作 | 原因 |
|:---|:---|:---|
| **本文** | 新增 | Phase 2 实施日志 |

---
