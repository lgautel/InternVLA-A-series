# wrmup8G 落地实施日志

> 对应方案: [`itrnVLA15_GeoP_3dtrj_3cn4_wrmup8G.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup8G.md)  
> 实施日期: 2026-08-11  
> 环境: venv `/tmp/itnvla15rbt20/`，8× NVIDIA H200（~140 GB/卡）

---

## 总览

| 阶段 | 状态 | 墙钟 |
|:---|:---:|:---|
| Phase 0 venv Bootstrap | ✅ | ~2 min |
| Phase 4 Smoke 100 step（1 GPU） | ✅ | ~4 min |
| Phase 5 8 卡 Warmup 400 step | ✅ | ~18 min |
| **推荐 checkpoint** | **step 300** | 见 §5 |

---

## 1. 操作日志（按时间顺序）

### 1.1 Phase 0 — 源码 symlink

**操作**（原因：`pip install -e` metadata 已指向 `/tmp/SRC/InternVLA-A-series`，但目录尚不存在）：

```bash
mkdir -p /tmp/SRC
ln -sfn /home/a26113/SRC/InternVLA-A-series /tmp/SRC/InternVLA-A-series
```

**结果**：`/tmp/SRC/InternVLA-A-series -> /home/a26113/SRC/InternVLA-A-series`

**文件变更**：新增 symlink `/tmp/SRC/InternVLA-A-series`（无代码复制）

---

### 1.2 Phase 0 — editable 安装

**操作**（原因：确保 venv 正确链到 `/tmp/SRC/` 源码）：

```bash
/tmp/itnvla15rbt20/bin/pip install -e /tmp/SRC/InternVLA-A-series
```

**结果**：
```
Editable project location: /tmp/SRC/InternVLA-A-series
Location: /tmp/itnvla15rbt20/lib/python3.11/site-packages
```

---

### 1.3 Phase 0 — venv 内 var/ 布局

**操作**（原因：venv 自包含；HF/权重/数据集注册均落在 venv 树内）：

```bash
VENV=/tmp/itnvla15rbt20
mkdir -p ${VENV}/var/hf_home/ckpts
mkdir -p ${VENV}/var/datasets
ln -sfn /tmp/rbt2stk3kptsim0811/stack_bowls_three_kptsim_lrbv30 \
  ${VENV}/var/datasets/stack_bowls_three_kptsim_lrbv30
```

**文件变更**：
| 路径 | 操作 |
|:---|:---|
| `/tmp/itnvla15rbt20/var/hf_home/ckpts/` | **新增**目录 |
| `/tmp/itnvla15rbt20/var/datasets/` | **新增**目录 |
| `.../var/datasets/stack_bowls_three_kptsim_lrbv30` | **新增** symlink → 外部数据 |

---

### 1.4 Phase 0 — torchcodec cu128 + nvidia-npp-cu12

**操作**（原因：PyPI 默认 0.10.0 为 CPU-only wheel，不支持 cu128 NVDEC API；cu128 wheel 依赖 `libnppicc.so.12`）：

```bash
VENV=/tmp/itnvla15rbt20
${VENV}/bin/pip install --force-reinstall "torchcodec==0.10.0" \
  --index-url https://download.pytorch.org/whl/cu128
${VENV}/bin/pip install nvidia-npp-cu12
```

**结果**：
- `torchcodec 0.10.0+cu128`（自 0.10.0 CPU wheel 升级）
- `nvidia-npp-cu12-12.4.1.87` 新装

**文件变更**：venv site-packages 内 torchcodec / nvidia-npp 包更新（无仓库文件改动）

---

### 1.5 Phase 0 — Transformers patch 复验

**操作**：检查 `${VENV}/lib/.../transformers/models/qwen3_5/modeling_qwen3_5.py`

**结果**：已存在，**无需**重新 patch

---

### 1.6 Phase 0 / Phase 2 — 权重下载

**操作**（原因：权重须落入 `${VENV}/var/hf_home/`，禁止 `$HOME/.cache`）：

```bash
export HF_HOME=/tmp/itnvla15rbt20/var/hf_home
/tmp/itnvla15rbt20/bin/python - <<'PY'
# GeoPredict_robocasa.pth, InternVLA-A1.5-base, Qwen3.5-2B
PY
```

**结果**：

| 文件 | 大小 | 路径 |
|:---|:---:|:---|
| GeoPredict_robocasa.pth | 6.1 GB | `.../var/hf_home/ckpts/GeoPredict_robocasa.pth` |
| InternVLA-A1.5-base | 5.1 GB | `.../ckpts/InternVLA-A1.5-base/model.safetensors` |
| Qwen3.5-2B | — | `.../hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c.../` |

**文件变更**：`${VENV}/var/hf_home/` 下新增 hub 缓存与 ckpts（~11 GB）

---

### 1.7 Phase 0 — 自包含验收 + Layer 2

**操作**：运行 wrmup8G 附录 C 检查 + LeRobotDataset 加载

**结果**：
```
torchcodec 0.10.0+cu128
SELF_CONTAINED OK
episodes 50 frames 23550
keypoint_3d torch.Size([42])
```

---

## 2. 错误记录与修复

### Error 1 — Smoke 首次失败：`--multi_gpu` 与单进程冲突

| 项 | 内容 |
|:---|:---|
| **阶段** | Phase 4 Smoke（第 1 次尝试） |
| **现象** | `ValueError: You need to use at least 2 processes to use --multi_gpu.` |
| **根因** | launch 脚本在 `SMOKE=1`（`NUM_PROCESSES=1`）时仍传入 `--multi_gpu` |
| **Fix** | 修改 [`launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh`](../launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh)：仅当 `NUM_PROCESSES > 1` 时追加 `--multi_gpu` |
| **验证** | Smoke 第 2 次尝试 exit 0 |

**代码变更**（`launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh`）：

```bash
LAUNCH_ARGS=()
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi
LAUNCH_ARGS+=(--num_processes=... ...)
```

---

## 3. Phase 4 — Smoke 100 step（1 GPU）

**命令**：

```bash
cd /tmp/SRC/InternVLA-A-series
SMOKE=1 LOG_FILE=/tmp/smoke_kptsim_8g_100step.log \
  bash launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh
```

**配置**：GPU0，`batch_size=2`，`steps=100`，`video_backend=torchcodec`，`wandb.enable=false`

**JOB_NAME**：`2026_08_11_03_00_20-internvla_a1_5-smoke100-kptsim-voxel`

### 3.1 初始化

```
load_geopredict_track_encoder_weights: loaded 26 keys, skipped 2 (track_fusion_layer)
```

### 3.2 训练轨迹

| Step | loss | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 20.683 | **0.5511** | 0.7335 | 0.251 | 562.1 |
| 50 | 2.845 | 0.0314 | 0.1003 | 0.262 | 88.6 |
| 100 | 1.402 | **0.0058** | 0.0498 | 0.174 | 39.3 |

### 3.3 验收

| 判据 | 结果 |
|:---|:---:|
| step 10 `loss_kpt_cur > 0` | ✅ 0.5511 |
| step 100 明显低于 step 10 | ✅ |
| `video_decode_error` | ✅ **0** |
| `using_zeros` | ✅ **0** |
| exit code | ✅ 0 |

**Checkpoint**：
```
/tmp/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_08_11_03_00_20-internvla_a1_5-smoke100-kptsim-voxel/checkpoints/000100/pretrained_model/
```

**日志**：`/tmp/smoke_kptsim_8g_100step.log`

---

## 4. Phase 5 — 8 卡 Warmup 400 step

**命令**：

```bash
cd /tmp/SRC/InternVLA-A-series
LOG_FILE=/tmp/warmup_kptsim_8g_400step.log \
  bash launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh
```

**配置**：8×H200，`batch_size=16/GPU`，有效 BS=128，`steps=400`，`num_workers=12`，`video_backend=torchcodec`

**JOB_NAME**：`2026_08_11_03_04_19-internvla_a1_5-geop-phase1-kpt-warmup-kptsim-voxel-8g`

**墙钟**：~18 min（含模型加载 + 400 step + 4 次 checkpoint 保存）

### 4.1 关键 step 轨迹

| Step | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm | epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 10 | 0.5197 | 0.7012 | 0.275 | 526.7 | 0.05 |
| 100 | 0.0031 | 0.0156 | 0.124 | 19.6 | 0.54 |
| 200 | 0.0018 | 0.0041 | 0.112 | 7.5 | 1.09 |
| **300** | **0.0016** | **0.0035** | **0.103** | **3.8** | **1.63** |
| 400 | 0.0015 | 0.0030 | 0.099 | 2.4 | 2.17 |

kpt loss 在 step 100 内快速饱和，200→400 变化 < 0.0003，与 [LOG_p1](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p1.md) 一致。

### 4.2 验收

| 判据 | 结果 |
|:---|:---:|
| 8 卡全部参与 | ✅ `CUDA_VISIBLE_DEVICES=0-7`，8 processes |
| `video_decode_error` | ✅ **0** |
| `using_zeros` | ✅ **0** |
| checkpoint 100/200/300/400 | ✅ 均保存 |
| WandB offline | ✅ 见下 |
| exit code | ✅ 0 |

**post_check**：
```
post_check: video_decode_error=0 using_zeros=0 exit=0
```

### 4.3 Checkpoint 路径

```
/tmp/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_08_11_03_04_19-internvla_a1_5-geop-phase1-kpt-warmup-kptsim-voxel-8g/checkpoints/
├── 000100/pretrained_model/
├── 000200/pretrained_model/
├── 000300/pretrained_model/   ← 推荐 Phase 2 起点
├── 000400/pretrained_model/
└── last/
```

**推荐 Phase 2 使用**：`checkpoints/000300/pretrained_model`

### 4.4 WandB offline

```
.../outputs/internvla_a1_5/2026_08_11_03_04_19-.../wandb/offline-run-20260811_030520-3duxmwhe/run-3duxmwhe.wandb
```

---

## 5. 关键路径汇总

| 用途 | 路径 |
|:---|:---|
| venv | `/tmp/itnvla15rbt20/` |
| 源码（editable） | `/tmp/SRC/InternVLA-A-series/` |
| HF 缓存 + 权重 | `/tmp/itnvla15rbt20/var/hf_home/` |
| 数据 symlink | `/tmp/itnvla15rbt20/var/datasets/stack_bowls_three_kptsim_lrbv30` |
| Launch 脚本 | `/tmp/SRC/InternVLA-A-series/launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh` |
| Smoke 日志 | `/tmp/smoke_kptsim_8g_100step.log` |
| Warmup 日志 | `/tmp/warmup_kptsim_8g_400step.log` |
| Smoke 输出 | `outputs/.../2026_08_11_03_00_20-...-smoke100-kptsim-voxel/` |
| Warmup 输出 | `outputs/.../2026_08_11_03_04_19-...-geop-phase1-kpt-warmup-kptsim-voxel-8g/` |
| **Warmup ckpt@300** | `.../checkpoints/000300/pretrained_model` |

---

## 6. 文件变更清单

| 路径 | 操作 | 原因 |
|:---|:---|:---|
| `/tmp/SRC/InternVLA-A-series` | **新增** symlink | editable 源码路径 |
| `/tmp/itnvla15rbt20/var/hf_home/` | **新增**（含 ~11 GB 权重） | venv 自包含 HF/ckpts |
| `/tmp/itnvla15rbt20/var/datasets/stack_bowls_three_kptsim_lrbv30` | **新增** symlink | HF_LEROBOT_HOME 注册 |
| venv: `torchcodec` | **升级** 0.10.0 → 0.10.0+cu128 | GPU 解码能力 |
| venv: `nvidia-npp-cu12` | **新增** | cu128 torchcodec 依赖 |
| [`launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh`](../launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh) | **修改** | Smoke 单进程不加 `--multi_gpu` |
| [`b/d/itrnVLA15_GeoP_3dtrj_3cn4_wrmup8G.md`](itrnVLA15_GeoP_3dtrj_3cn4_wrmup8G.md) | **新增** | 本机实施方案 |
| **本文** | **新增** | 实施日志 |
| `outputs/internvla_a1_5/2026_08_11_03_00_20-...` | **新增** | Smoke 训练产物 |
| `outputs/internvla_a1_5/2026_08_11_03_04_19-...` | **新增** | Warmup 训练产物 + wandb offline |

**未修改**：训练核心 Python 代码（`modeling_internvla_a1_5.py` 等）

---

## 7. Phase 2 衔接备忘

| 配置 | Warmup | Phase 2 |
|:---|:---:|:---:|
| `pretrained_path` | InternVLA-A1.5-base | **ckpt@300** |
| `init_kpt_expert_from_action` | true | **false** |
| `geopredict_checkpoint_path` | 设置 | **不设** |

Phase 2 起点：

```bash
PRETRAINED_PATH=/tmp/SRC/InternVLA-A-series/outputs/internvla_a1_5/2026_08_11_03_04_19-internvla_a1_5-geop-phase1-kpt-warmup-kptsim-voxel-8g/checkpoints/000300/pretrained_model
```

---

## 8. 结论

**wrmup8G 方案已在本机端到端跑通**：

1. Phase 0：venv 自包含 bootstrap（editable + var/hf + torchcodec cu128 + 权重）✅  
2. Smoke 100 step：kpt loss 0.55→0.006，torchcodec 解码零错误 ✅  
3. 8 卡 Warmup 400 step：kpt loss 饱和，4 个 checkpoint + WandB offline 均正常保存 ✅  

**torchcodec GPU 解码**：正式训练使用 `--dataset.video_backend=torchcodec`（cu128 wheel）；DataLoader worker 内走 CPU 高速解码路径，`video_decode_error=0`，未出现全黑帧 fallback。

---

*日志版本: wrmup8G-LOG-v1.0 | 2026-08-11*
