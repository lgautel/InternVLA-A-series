# Phase 2 正式训练日志 (0805) — Action + Video + Kpt

> 基于 [itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2.md](itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2.md)，从 Phase 1 Step 300 checkpoint 继续微调。
> 本次变更: `action_loss_only=false`（启用 WAN video loss），`kpt_loss_weight=1.0`，`steps=20000`，`save_freq=2500`。

---

## 1. 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 8×H200 |
| batch_size (per GPU) | 16 |
| 有效 BS | 128 |
| pretrained_path | Phase 1 Step 300 checkpoint |
| train_expert_only | true |
| action_loss_only | **false** (启用 video loss) |
| action_loss_weight | 10.0 |
| kpt_loss_weight | **1.0** |
| video_loss_weight | 1.0 |
| init_kpt_expert_from_action | false |
| geopredict_checkpoint_path | 不设置 |
| steps | **20000** |
| save_freq | **2500** |
| 虚拟环境 | /tmp/itrnvla15rbt2 (修复版) |

---

## 2. 时间线 / 操作日志

| 时间 (UTC+8) | 操作 | 结果 |
|---|---|---|
| 2026-08-06 12:21 | 创建数据集 symlink | `/tmp/hf_home/lerobot/robotwin/stack_bowls_three_kpt` → `/tmp/robotwin2/stack_bowls_three_kpt` ✅ |
| 2026-08-06 12:22 | 修复虚拟环境 | 原 `/tmp/itrnvla15rbt/` Python stdlib 损坏；新建 `/tmp/itrnvla15rbt2/` + 链接旧 site-packages + 修复 lerobot editable path |
| 2026-08-06 12:23 | 安装 FFmpeg (conda-forge) | 修复 torchcodec `libavutil.so` 缺失 |
| 2026-08-06 12:24 | 创建 launch 脚本 | `launch/internvla_a15_geop_phase2_finetune_stackb3_0805.sh` |
| 2026-08-06 12:24 | 下载 WAN2.2-TI2V-5B | `snapshot_download` 完成，32GB ✅ |
| 2026-08-06 12:25 | 单卡 smoke test (BS=2, 2 steps) | 首次失败（见 #5/#6/#7）；修复后第二次通过 ✅ |
| 2026-08-06 12:36 | 8 卡正式训练 第 1 次启动 | 失败：`invalid device ordinal`（见 #8）❌ |
| 2026-08-06 12:37 | 8 卡正式训练 第 2 次启动 | 失败：`wandb-core` 权限（见 #9）❌ |
| 2026-08-06 12:38 | 训练状态检查 | 无 `lerobot_train` 进程；从未到达 `step 1` |

---

## 3. 问题记录（报错 → 根因 → 修复 → 验证）

### #1: Python venv 无法启动

- **报错**: `ModuleNotFoundError: No module named 'encodings'`
- **根因**: uv venv 的 base Python 路径 `/home/physical/.local/share/uv/python/...` 在本机不存在
- **修复**: 用 `/opt/conda/bin/python3.11 -m venv /tmp/itrnvla15rbt2`，通过 `.pth` 链接旧 site-packages，新建 `internvla_local.pth` 指向 `/tmp/SRC/InternVLA-A-series/src`
- **验证**: `import torch, transformers, lerobot` 成功

### #2: lerobot 模块找不到

- **报错**: `ModuleNotFoundError: No module named 'lerobot'`
- **根因**: editable install 路径指向 `/home/physical/SRC/Robot/InternVLA-A-series/src`（旧机器路径）
- **修复**: 创建 `/tmp/itrnvla15rbt2/lib/python3.11/site-packages/internvla_local.pth` 内容为 `/tmp/SRC/InternVLA-A-series/src`
- **验证**: `lerobot 1.0.0` ✅

### #3: torchcodec FFmpeg 库缺失

- **报错**: `OSError: libavutil.so.60: cannot open shared object file`
- **根因**: 系统/venv 未安装 FFmpeg 共享库
- **修复**: `conda install -c conda-forge ffmpeg -y`；训练脚本设置 `LD_LIBRARY_PATH=/opt/conda/lib:...`
- **验证**: `import torchcodec` 成功 ✅

### #4: huggingface-cli 下载失败

- **报错**: `huggingface-cli is deprecated and no longer works`
- **根因**: 旧版 CLI 脚本已弃用；且 shebang 指向 `/mnt/r/VENV/itrnvla15rbt/bin/python3`
- **修复**: 改用 `huggingface_hub.snapshot_download()` Python API
- **验证**: WAN 32GB 下载完成，`Wan2.2_VAE.pth` 存在 ✅

### #5: CUDA/GPU 不可见（nvidia-smi / torch）

- **报错**: `NVIDIA-SMI couldn't find libnvidia-ml.so`；`torch.cuda.is_available()=False`
- **根因**: 未将 NVIDIA 驱动库目录加入 `LD_LIBRARY_PATH`
- **修复**: 在 launch 脚本中加入 `LD_LIBRARY_PATH=/usr/local/nvidia/lib64:...`
- **验证**: `nvidia-smi` 显示 8×H200；`torch.cuda.device_count()=8` ✅

### #6: triton ptxas 无执行权限

- **报错**: `PermissionError: [Errno 13] Permission denied: '.../triton/backends/nvidia/bin/ptxas'`
- **根因**: 从旧 venv 复制的 triton CUDA 工具链二进制缺少 `+x` 位
- **修复**: `chmod +x /tmp/itrnvla15rbt/lib/python3.11/site-packages/triton/backends/nvidia/bin/*`
- **验证**: 单卡 smoke test forward 通过 ✅

### #7: 缺少 policy.repo_id / push_to_hub

- **报错**: `ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.`
- **根因**: smoke test 命令未传 `--policy.push_to_hub=false` 和 `--policy.repo_id`
- **修复**: launch 脚本增加 `--policy.repo_id=lerobot_lab/internvla_a1_5 --policy.push_to_hub=false`
- **验证**: 配置校验通过 ✅

### #8: 8 卡 DDP — invalid device ordinal（第 1 次正式训练）

- **报错**: `torch.AcceleratorError: CUDA error: invalid device ordinal`（rank 2+）
- **根因**: 单卡 smoke test 后 shell 残留 `CUDA_VISIBLE_DEVICES=0`，仅 1 张 GPU 可见，但 `--num_processes=8`
- **修复**: 正式训练前显式设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 或 `unset CUDA_VISIBLE_DEVICES`
- **验证**: 第 2 次启动时 8 卡均初始化成功（见 #9 前日志），但尚未验证完整训练 ⏳

### #9: WandB wandb-core 无执行权限（第 2 次正式训练）

- **报错**: `PermissionError: [Errno 13] Permission denied: '/tmp/itrnvla15rbt/lib/python3.11/site-packages/wandb/bin/wandb-core'`
- **根因**: venv 内 `wandb-core` 等二进制从旧环境复制时无执行权限；rank0 在 `wandb.init()` 时崩溃，其余 rank 被 SIGTERM
- **修复（待执行）**: `chmod +x /tmp/itrnvla15rbt/lib/python3.11/site-packages/wandb/bin/*`；或临时 `--wandb.enable=false`
- **验证**: 待第 3 次启动 ⏳

---

## 5. Smoke Test 结果（单卡 GPU0, BS=2, 2 steps）

**命令**: 单卡 `accelerate launch --num_processes=1`，`action_loss_only=false`，从 Phase 1 Step 300 ckpt 加载。

| Step | loss | action | video | kpt_cur | kpt_fut | grad_norm |
|------|------|--------|-------|---------|---------|-----------|
| 1 | 6.710 | 0.073 | **0.517** | 0.0013 | 0.0018 | 25.185 |
| 2 | 6.411 | 0.057 | **0.455** | 0.2087 | 0.2038 | 22.038 |

- `loss_video > 0` ✅（WAN video loss 正常）
- `loss_kpt_cur/fut > 0` ✅（3D 关键点监督正常）
- 首步加载约 6 分钟（含 WAN + 模型权重）

---

## 6. 正式训练启动记录

### 第 1 次（2026-08-06 04:36:43 UTC）

```
OUTPUT_DIR=outputs/internvla_a1_5/2026_08_06_04_36_43-internvla_a1_5-geop-phase2-action-video-kptw1-stackb3-abs-20k
```

- **结果**: 启动后 ~30s 内 DDP 崩溃，未进入训练循环
- **根因**: #8 `CUDA_VISIBLE_DEVICES=0`

### 第 2 次（2026-08-06 04:37:36 UTC）

```
OUTPUT_DIR=outputs/internvla_a1_5/2026_08_06_04_37_36-internvla_a1_5-geop-phase2-action-video-kptw1-stackb3-abs-20k
LOG=outputs/internvla_a1_5/train_0805_geop_phase2.log
```

- **结果**: 8 卡初始化成功，配置打印完成，rank0 WandB 初始化时崩溃
- **根因**: #9 `wandb-core` Permission denied
- **当前状态**: 无训练进程；**从未到达 step 1**

---

## 7. 文件变更清单（更新）

| 文件 / 路径 | 操作 | 原因 |
|---|---|---|
| `/tmp/itrnvla15rbt2/` | 新增 | 修复版虚拟环境（conda Python + 旧 site-packages） |
| `/tmp/itrnvla15rbt2/lib/python3.11/site-packages/internvla_local.pth` | 新增 | 修正 lerobot editable 路径 |
| `/tmp/itrnvla15rbt2/lib/python3.11/site-packages/old_venv.pth` | 新增 | 链接旧 venv site-packages |
| `/tmp/hf_home/hub/Wan2.2-TI2V-5B/` | 新增 (~32GB) | WAN2.2 权重 |
| `/tmp/hf_home/lerobot/robotwin/stack_bowls_three_kpt` | symlink | 数据集路径映射 |
| `launch/internvla_a15_geop_phase2_finetune_stackb3_0805.sh` | 新增 | 训练启动脚本（含 NVIDIA/FFmpeg LD_LIBRARY_PATH） |
| `outputs/internvla_a1_5/train_0805_geop_phase2.log` | 新增 | 训练 stdout/stderr 日志 |
| `outputs/internvla_a1_5/smoketest_0805_geop_phase2/` | 新增 | smoke test 输出 |
| `outputs/internvla_a1_5/2026_08_06_04_36_43-...-20k/` | 新增（空） | 第 1 次失败 run |
| `outputs/internvla_a1_5/2026_08_06_04_37_36-...-20k/` | 新增（仅 wandb/） | 第 2 次失败 run |
| `b/d/itrnVLA15_GeoP_3dtrj_3cn2_sft_rbt2_2LOG_p2_0805.md` | 新增/更新 | 本执行日志 |

---

## 8. 关键路径速查

| 用途 | 路径 |
|------|------|
| 虚拟环境（修复版） | `/tmp/itrnvla15rbt2/` |
| 原虚拟环境（Python 损坏，仅 site-packages 可用） | `/tmp/itrnvla15rbt/` |
| Phase 1 checkpoint | `outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model` |
| 3D kpt 数据集 | `/tmp/robotwin2/stack_bowls_three_kpt/` |
| External stats | `/tmp/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json` |
| WAN 权重 | `/tmp/hf_home/hub/Wan2.2-TI2V-5B/` |
| 训练脚本 | `launch/internvla_a15_geop_phase2_finetune_stackb3_0805.sh` |
| 训练日志 | `outputs/internvla_a1_5/train_0805_geop_phase2.log` |

---

## 9. 当前状态与下一步

| 项目 | 状态 |
|------|------|
| 环境 / 数据 / WAN / smoke test | ✅ 就绪 |
| 8 卡正式训练 | ❌ 两次启动均失败，未进入 step 1 |
| 待修复 | #9 wandb-core 权限（或禁用 wandb）；确认 `CUDA_VISIBLE_DEVICES` 覆盖全部 8 卡 |
| 待执行 | 第 3 次启动 8 卡训练，20000 steps |


---

## 10. 第 3 次训练启动（2026-08-06 12:52 UTC+8）

### 启动前修复

| 操作 | 原因 |
|------|------|
| `chmod +x /tmp/itrnvla15rbt/lib/python3.11/site-packages/wandb/bin/*` | 修复 #9 wandb-core 权限 |
| `chmod +x /tmp/itrnvla15rbt/bin/*` | 防止其他 CLI 二进制权限问题 |
| 确认 launch 脚本含 `--policy.pretrained_path="${PRETRAINED_PATH}"` | 从 Phase 1 Step 300 ckpt 加载 |
| `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` | 避免 #8 单卡残留 |
| `MASTER_PORT=36301` | 避免端口冲突 |

### 启动命令

```bash
export HF_HOME=/tmp/hf_home HF_LEROBOT_HOME=/tmp/hf_home/lerobot
export USE_LIBUV=0 WANDB_MODE=offline
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/opt/conda/lib:/tmp/itrnvla15rbt/lib
export VENV_ROOT=/tmp/itrnvla15rbt2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_PORT=36301
cd /tmp/SRC/InternVLA-A-series
nohup bash launch/internvla_a15_geop_phase2_finetune_stackb3_0805.sh \
  > outputs/internvla_a1_5/train_0805_geop_phase2_run3.log 2>&1 &
```


### #9 修复验证 ✅

- `chmod +x wandb/bin/*` 后 WandB 初始化成功
- 日志: `Track this run --> .../2026_08_06_04_52_37-.../wandb/offline-run-...`

### 训练进展

```
OUTPUT_DIR=outputs/internvla_a1_5/2026_08_06_04_52_37-internvla_a1_5-geop-phase2-action-video-kptw1-stackb3-abs-20k
LOG=outputs/internvla_a1_5/train_0805_geop_phase2_run3.log
```

| 时间 | 事件 |
|------|------|
| 04:52:37 | 8 卡进程启动 |
| 04:53:08 | WandB offline 初始化成功 |
| 04:55:11 | 模型加载完成，开始训练 (`Start offline training`) |
| 05:01:43 | TileLang kernel JIT 编译（8 卡并行，首步延迟） |
| 05:09:07 | **step 50** 首条日志（首步 wall time ~14 min） |
| 05:10:00 | **step 100**，吞吐 ~0.95 iters/s |

**step 100 指标**: loss=6.388, action=0.088, video=0.732, kpt_cur=0.0011, kpt_fut=0.0034, grad_norm=7.806

**有效配置确认**:
- Trainable: 927M / Total: 8B
- Effective BS: 128 (16×8)
- WAN params: 5B (frozen DiT)
- action_loss_only=false, kpt_loss_weight=1.0 ✅


---

## 11. 训练完成（step 20000）

**完成时间**: 2026-08-06 10:39 UTC

| Step | action | video | kpt_cur | kpt_fut | grad_norm |
|------|--------|-------|---------|---------|-----------|
| 20000 | 0.002 | 0.088 | 0.0007 | 0.0017 | 0.875 |

**Checkpoints** (8 个):
```
outputs/internvla_a1_5/2026_08_06_04_52_37-internvla_a1_5-geop-phase2-action-video-kptw1-stackb3-abs-20k/checkpoints/
├── 002500/ ... 020000/ last -> 020000/
```

**推荐 checkpoint**: `checkpoints/020000/pretrained_model` 或参考 Phase 2 经验选 step 15000/17500 防过拟合。

---

## 12. GCS 上传

```bash
gcloud storage cp -r /tmp/SRC/InternVLA-A-series/ gs://physical-ai-data-eu/VENV/tmp/itnvla088518/
```

- **本地路径**: `/tmp/SRC/InternVLA-A-series/` (~420G)
- **目标路径**: `gs://physical-ai-data-eu/VENV/tmp/itnvla088518/InternVLA-A-series/`
- **上传日志**: `outputs/gcloud_upload_itnvla088518.log`

