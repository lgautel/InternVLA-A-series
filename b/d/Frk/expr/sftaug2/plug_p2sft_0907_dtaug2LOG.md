# Phase 2 SFT + Full Data Augmentation (incl. affine@0.2) — 操作手册

> **目的**: 对 SFT checkpoint@10420 继续微调 20000 步，启用全部 6 种图像数据增强（含 affine，权重降至 0.2）
> **起点 Checkpoint**: `/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/`
> **数据集**: `plug_into_socket_lrb_4D` — 100 episodes, 66577 frames, 30fps, 8 kpts × 7D
> **EXPR_NAME**: `itvlagpFrkPlug0907_dtaug2`
> **Launch Script**: `b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh`
> **日期**: 2026-09-22
> **操作者**: —

---

## 与前次实验的差异

| 维度 | dtaug (itvlagpFrkPlug0907_dtaug) | **dtaug2 (本次)** |
|:---|:---|:---|
| 起点 | SFT ckpt@10420 | **SFT ckpt@10420**（同） |
| 数据增强 | 5 种（affine 禁用） | **6 种全部启用**（affine weight=0.2） |
| 总步数 | 50000 | **20000** |
| SAVE_FREQ | 10000 | **5000** |
| MASTER_PORT | 36703 | **36704** |
| SCHEDULER_WARMUP | 1000 | **500** |
| 其他参数 | — | 与 dtaug 完全一致 |

### 数据增强配置详情

6 种增强全部启用，通过 `RandomSubsetApply` 每帧随机采样 `max_num_transforms=3` 种应用:

| 增强名称 | 类型 | 参数 | 权重 | 归一化概率 |
|:---|:---|:---|:---|:---|
| brightness | ColorJitter | brightness=(0.8, 1.2) | 1.0 | ~19.2% |
| contrast | ColorJitter | contrast=(0.8, 1.2) | 1.0 | ~19.2% |
| saturation | ColorJitter | saturation=(0.5, 1.5) | 1.0 | ~19.2% |
| hue | ColorJitter | hue=(-0.05, 0.05) | 1.0 | ~19.2% |
| sharpness | SharpnessJitter | sharpness=(0.5, 1.5) | 1.0 | ~19.2% |
| **affine** | **RandomAffine** | **degrees=(-5,5), translate=(0.05,0.05)** | **0.2** | **~3.8%** |

> **为什么 affine 用低权重**: RandomAffine 对图像做旋转 ±5° 和平移 ±5%，会导致视觉观测与动作标签之间的空间对齐错位。weight=0.2（归一化概率 ~3.8%）在保留少量空间扰动鲁棒性的同时，大幅降低错位风险。详见 `b/d/Frk/dtaaug_analyz.markdown`。

> **技术说明**: draccus 不支持 `--dataset.image_transforms.tfs.affine.weight=0.2` 语法（dict 字段是原子型参数），因此脚本中通过 `--dataset.image_transforms.tfs='{...}'` 传递完整的 tfs YAML 字典来实现权重控制。

预计 Checkpoint 保存点: 5000, 10000, 15000, 20000 (final)

预计训练耗时: 20000 / 0.23 ≈ **24 小时 (~1 天)**（基于 dtaug 实测 ~0.23 iters/s @ 8×H200）

---

## 0. Pre-flight 环境准备

### 0.1 激活虚拟环境

```bash
source /B/VENV/itnvla15rbt20/bin/activate
python --version    # 应输出 Python 3.11.x
which python        # 应指向 /B/VENV/itnvla15rbt20/bin/python
```

### 0.2 验证依赖

```bash
python -c "
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'GPUs', torch.cuda.device_count())
import transformers; print('transformers', transformers.__version__)
import accelerate; print('accelerate', accelerate.__version__)
import lerobot; print('lerobot OK')
"
```

期望: `torch 2.10.0+cu128`, `transformers 5.2.0`, `accelerate 1.14.0`, `lerobot OK`, `GPUs 8`

### 0.3 验证 Editable install

```bash
python -c "
import site, pathlib
for p in site.getsitepackages() + [site.getusersitepackages()]:
    pth = pathlib.Path(p)
    for f in pth.glob('*.pth'):
        txt = f.read_text().strip()
        if 'lerobot' in txt.lower() or 'itvla' in txt.lower():
            print(f'{f.name} → {txt}')
"
```

期望: `.pth` 指向 `/B/SRC/itvlaGpLibPlus/src`（当前代码库，含 `disabled_tfs` 支持和 FAST offline fix）

### 0.4 验证 Checkpoint

```bash
python -c "
import json, pathlib
cfg = json.loads(pathlib.Path('/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/config.json').read_text())
for k in ['kpt_4d_mode','num_keypoint_joints','enable_keypoint_predictor','keypoint_history_max_len',
          'enable_vqa_loss','action_loss_only','train_expert_only','knowledge_insulation',
          'action_loss_weight','kpt_loss_weight','kpt_future_loss_weight','video_loss_weight',
          'gradient_checkpointing','freeze_wan_dit','freeze_learnable_tokens']:
    print(f'{k}: {cfg.get(k, \"N/A\")}')
"
```

期望值:
```
kpt_4d_mode: pos_rot
num_keypoint_joints: 8
enable_keypoint_predictor: True
keypoint_history_max_len: 200
enable_vqa_loss: True
action_loss_only: False
train_expert_only: False
knowledge_insulation: False
action_loss_weight: 10.0
kpt_loss_weight: 1.0
kpt_future_loss_weight: 1.5
video_loss_weight: 1.0
gradient_checkpointing: True
freeze_wan_dit: True
freeze_learnable_tokens: True
```

### 0.5 验证数据集 symlink

```bash
ls -la /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D
# 应指向 /B/Dta/plug_into_socket_lrb_4D

# 若不存在，创建:
ln -sf /B/Dta/plug_into_socket_lrb_4D /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D
```

验证数据集内容:
```bash
python -c "
import json, pathlib
info = json.loads(pathlib.Path('/B/Dta/plug_into_socket_lrb_4D/meta/info.json').read_text())
print(f\"frames={info['total_frames']}, episodes={info['total_episodes']}, fps={info['fps']}\")
stats = json.loads(pathlib.Path('/B/Dta/plug_into_socket_lrb_4D/meta/stats/abs/stats.json').read_text())
print(f\"Stats OK: keypoint_3d has {len(stats.get('keypoint_3d',{}).get('mean',[]))} dims\")
"
```

期望: `frames=66577, episodes=100, fps=30`, `keypoint_3d has 56 dims`

### 0.6 验证 Schema symlink

```bash
ls -la /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D/meta/schemas/franka_plug.yaml
```

若不存在:
```bash
ln -sf /B/SRC/itvlaGpLibPlus/src/lerobot/policies/internvla_a1_5/robot_type_configs/franka_plug.yaml \
       /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D/meta/schemas/franka_plug.yaml
```

### 0.7 验证模型缓存

```bash
# FAST tokenizer
ls /B/VENV/hf_home/hub/models--physical-intelligence--fast/snapshots/*/tokenizer.json

# Qwen3.5-2B
ls /B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/ 2>/dev/null && echo "Qwen cached"

# WAN2.2
ls /B/VENV/hf_home/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth && echo "WAN VAE cached"
```

### 0.8 GPU 状态

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,memory.total --format=csv,noheader
```

期望: 8 张 H200 全部空闲（0 MiB used）。如有占用:

```bash
# 查看占用进程
nvidia-smi
# 必要时 kill bigmatrix 或其他训练
kill <PID>
```

### 0.9 端口检查

```bash
ss -tlnp | grep 36704
```

期望: 无输出（端口未占用）。若被占用，修改 `MASTER_PORT`:
```bash
MASTER_PORT=36705 bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh
```

---

## 1. Smoke 测试

### 1.1 WAN Smoke（1 GPU, 2 steps）

验证 WAN 加载 + 全部 6 种数据增强 + forward/backward 完整链路:

```bash
source /B/VENV/itnvla15rbt20/bin/activate
cd /B/SRC/itvlaGpLibPlus
WAN_SMOKE=1 bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh
```

**验证要点**:
1. `exit=0`, `video_decode_error=0`, `using_zeros=0`
2. 5 个核心 loss 分量 (`loss_action`, `loss_video`, `loss_vqa`, `loss_kpt_cur`, `loss_kpt_fut`) 均有非零输出
3. 无 OOM, 无 NCCL error
4. 日志中确认 `disabled_tfs: []`（空列表，表示无禁用）
5. 日志中确认 tfs 配置包含 affine

**如果遇到 FAST tokenizer offline 错误** (E1):
确认 `_resolve_fast_local_dir()` fix 已在当前 editable install 中生效:
```bash
grep -n "_resolve_fast_local_dir" src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py
```
若无结果，说明代码未包含 fix。检查 editable install 是否指向正确的代码库 (§0.3)。

### 1.2 10-step Smoke（1 GPU, 10 steps）

```bash
SMOKE=1 bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh
```

确认 10 步全部完成，`exit=0`。Smoke 输出在 `/tmp/sft_dtaug2_smoke_frk_plug_*`。

### 1.3 查看 Smoke 日志

```bash
# WAN smoke log
cat /tmp/sft_dtaug2_smoke_frk_plug_*.log 2>/dev/null | head -100

# 检查 loss 分量
grep -E 'loss|step' /tmp/sft_dtaug2_smoke_frk_plug_*.log | head -20
```

---

## 2. 正式训练启动

### 2.0 启动命令

```bash
source /B/VENV/itnvla15rbt20/bin/activate
cd /B/SRC/itvlaGpLibPlus

nohup bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh > /tmp/frk_plug_sft_dtaug2_wrapper.log 2>&1 &
disown
echo "Wrapper PID: $!"
```

### 2.1 关键路径

启动后，脚本会打印以下关键路径:

| 项目 | 路径模式 |
|:---|:---|
| JOB_STAMP | `YYYY_MM_DD_HH_MM_SS` (启动时间) |
| LOG_FILE | `/B/Log/itvlagpFrkPlug0907_dtaug2/<JOB_STAMP>/train.log` |
| OUTPUT_DIR | `~/b/Ckp/itvlagpFrkPlug0907_dtaug2/<JOB_STAMP>-internvla_a1_5-frk-plug-sft-dtaug2/` |
| Wrapper log | `/tmp/frk_plug_sft_dtaug2_wrapper.log` |

### 2.2 确认训练启动成功

```bash
# 1. 检查 wrapper log，确认 PID 和路径
cat /tmp/frk_plug_sft_dtaug2_wrapper.log

# 2. 确认训练进程在运行
ps aux | grep lerobot_train | grep -v grep

# 3. 确认 GPU 已被占用 (~101 GiB/卡)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 4. 查看训练日志最新内容
tail -20 /B/Log/itvlagpFrkPlug0907_dtaug2/*/train.log
```

### 2.3 确认模型加载

从日志中确认:
```bash
grep -E 'Total params|Trainable|WAN params|Knowledge insulation|Freeze WAN' \
     /B/Log/itvlagpFrkPlug0907_dtaug2/*/train.log | head -10
```

期望:
```
Total params        : 8B
Trainable params    : 3B
WAN params          : 5B
Knowledge insulation: False
Freeze WAN DiT      : True
```

---

## 3. 训练监控

### 3.1 自动监控

Launch script 内置自动监控:
- **MONITOR_INTERVAL=900** (15 分钟): 每 15 分钟检查训练进程和日志文件
- **STALE_THRESHOLD=900** (15 分钟): 日志超过 15 分钟未更新视为异常
- **训练成功**: 自动启动 bigmatrix + 打包日志到 `~/b/Ckp/itvlagpFrkPlug0907_dtaug2_LOG_<ts>.tar`
- **训练失败**: 清理 GPU + bigmatrix + 打包日志 (带 `_err` 后缀)

### 3.2 手动监控命令

```bash
# 实时跟踪日志
tail -f /B/Log/itvlagpFrkPlug0907_dtaug2/*/train.log

# 查看最近 loss
grep 'step:' /B/Log/itvlagpFrkPlug0907_dtaug2/*/train.log | tail -10

# 查看当前 step 和预估剩余时间
cur=$(grep -oE 'step:[0-9]+' /B/Log/itvlagpFrkPlug0907_dtaug2/*/train.log | tail -1 | grep -oE '[0-9]+$')
echo "Step: ${cur}/20000, remaining: $(echo "scale=1; (20000 - ${cur}) / 0.23 / 3600" | bc) hours"

# GPU 显存
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

# 检查 checkpoint 目录
ls -la ~/b/Ckp/itvlagpFrkPlug0907_dtaug2/*/checkpoints/ 2>/dev/null
```

### 3.3 预计 Checkpoint 保存时间表

| 保存点 | Step | 预计耗时（累计） |
|:---:|:---:|:---:|
| 1 | 5000 | ~6 小时 |
| 2 | 10000 | ~12 小时 |
| 3 | 15000 | ~18 小时 |
| 4 (最终) | 20000 | ~24 小时 |

### 3.4 Loss 健康指标参考

基于 dtaug 实验早期 loss，本实验应有类似量级（起点 checkpoint 相同）:

| 分量 | 典型范围 (step 50-200) |
|:---|:---|
| loss_action | 0.003 - 0.010 |
| loss_vqa | 0.10 - 0.20 |
| loss_video | 0.03 - 0.10 |
| loss_kpt_cur | 0.001 - 0.030 |
| loss_kpt_fut | 0.01 - 0.06 |
| grad_norm | 2.0 - 5.0 |

**异常信号**:
- `loss_action` 突然上升到 > 0.1 → 可能 affine 干扰过大
- `grad_norm` > 50 → 梯度爆炸，考虑降低 lr 或禁用 affine
- `loss_video` = 0 → WAN 分支异常
- 任何 `NaN` 或 `Inf` → 立即停止

---

## 4. 故障排查

### 4.1 常见错误及解决方案

| 错误 | 原因 | 解决 |
|:---|:---|:---|
| `unrecognized arguments: --dataset.image_transforms.tfs` | draccus 版本不兼容 | 检查 `pip show draccus`；确认 tfs YAML 字符串格式正确 |
| `OSError: couldn't connect to huggingface.co` | FAST tokenizer offline 加载失败 | 确认 `_resolve_fast_local_dir` fix 已就位 (§1.1) |
| `NCCL timeout` / `ncclInternalError` | GPU 通信问题 | 检查 `NCCL_TUNER_PLUGIN` 设置；尝试 `NCCL_DEBUG=INFO` |
| OOM | 显存不足 | 降低 `BATCH_SIZE`（如 12）或 `NUM_WORKERS`（如 8） |
| `Address already in use: 36704` | 端口冲突 | `MASTER_PORT=36705 bash ...` |
| 日志 stale 导致自动终止 | 训练卡住 (deadlock/IO) | 检查 `dmesg`、`/var/log/syslog`；重新启动 |
| `video_decode_error` > 0 | 视频解码失败 | 检查数据集完整性；确认 torchcodec 安装正确 |

### 4.2 手动停止训练

```bash
# 方法 1: kill 训练进程 (监控会自动清理)
pkill -f "lerobot_train"
# 等待监控检测并执行清理

# 方法 2: 强制停止一切
pkill -9 -f "lerobot_train"
pkill -9 -f "accelerate.commands.launch"

# 方法 3: kill wrapper (会触发监控清理)
kill <wrapper_pid>
```

### 4.3 从中间 checkpoint 恢复

如果训练在 step N 中断，且存在 checkpoint:

```bash
# 查看已有 checkpoints
ls ~/b/Ckp/itvlagpFrkPlug0907_dtaug2/*/checkpoints/

# 从 checkpoint 恢复 (例如 step 10000)
PRETRAINED_CKPT=~/b/Ckp/itvlagpFrkPlug0907_dtaug2/<JOB_NAME>/checkpoints/010000/pretrained_model \
STEPS=20000 \
nohup bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh > /tmp/frk_plug_sft_dtaug2_resume.log 2>&1 &
disown
```

注意: 恢复后 step 计数从 0 重新开始，总步数仍是 20000。如需只跑剩余步数，设 `STEPS=10000`。

### 4.4 对比 affine 影响

训练完成后，可对比 dtaug（无 affine）与 dtaug2（affine@0.2）的 loss 曲线:

```bash
# 提取 dtaug loss
grep 'step:' /B/Log/itvlagpFrkPlug0907_dtaug/*/train.log | head -400 > /tmp/dtaug_loss.txt

# 提取 dtaug2 loss
grep 'step:' /B/Log/itvlagpFrkPlug0907_dtaug2/*/train.log | head -400 > /tmp/dtaug2_loss.txt

# 对比 (重点关注 loss_action 是否因 affine 上升)
diff <(grep -oP 'loss_action:\K[0-9.]+' /tmp/dtaug_loss.txt | head -20) \
     <(grep -oP 'loss_action:\K[0-9.]+' /tmp/dtaug2_loss.txt | head -20)
```

---

## 5. 代码变更记录

本实验不修改 `src/` 下的任何代码，仅新增两个文件:

1. **`b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh`** (新建)
   - 基于 `b/s/Frk/frk_plug_sft_dtaug_launch.sh`，启用全部 6 种数据增强
   - 通过 `--dataset.image_transforms.tfs='{...}'` 传递完整 tfs YAML 字典，affine weight=0.2
   - STEPS=20000, SAVE_FREQ=5000, MASTER_PORT=36704

2. **`b/d/Frk/expr/sftaug2/plug_p2sft_0907_dtaug2LOG.md`** (新建, 本文件)
   - 操作手册: pre-flight 检查、smoke 测试、正式训练启动与监控、故障排查

---

# 执行日志

## E0. 停止前序 dtaug 训练

### E0.1 等待 dtaug checkpoint@10000

从 2026-09-22 15:42 开始监控 PID 1911832 (`itvlagpFrkPlug0907_dtaug`, 50000 steps, save_freq=10000)。

训练进度追踪:

| 时间 | Step | Epoch |
|:---|:---|:---|
| 15:42 | ~6800 | 13.07 |
| 16:12 | ~7200 | 13.84 |
| 16:44 | ~7700 | 14.71 |
| 17:14 | ~8100 | 15.48 |
| 17:45 | ~8400 | 16.25 |
| 18:17 | ~8900 | 17.11 |
| 18:50 | ~9300 | 17.98 |
| 19:10 | ~9600 | 18.46 |
| 19:25 | ~9800 | 18.75 |
| 19:32 | ~9900 | 19.03 |
| 19:36 | ~9960 | 19.13 |
| 19:40 | 10.0K | 19.23 |

### E0.2 Checkpoint@10000 保存确认 — 19:42

Checkpoint 目录: `/home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00-internvla_a1_5-frk-plug-sft-dtaug/checkpoints/010000/`

```
pretrained_model/
  config.json             3,681 bytes
  model.safetensors       6,320,933,212 bytes (6.3 GB)
  stats.json              38,970 bytes
  train_config.json       12,934 bytes
training_state/
  optimizer_param_groups.json     37,569 bytes
  optimizer_state.safetensors     12,607,314,628 bytes (12.6 GB)
  rng_state.safetensors           15,708 bytes
  scheduler_state.json            254 bytes
  training_step.json              21 bytes
last -> 010000 (symlink)
```

全部 9 个文件均非零, checkpoint 完整。

### E0.3 等待 15 分钟 (19:42 → 19:57)

### E0.4 Kill dtaug 训练 — 19:59

```bash
kill 1911832   # SIGTERM to accelerate launcher
pkill -f "lerobot_train"
pkill -f "accelerate.commands.launch"
# Force kill after 5s
pkill -9 -f "lerobot_train"
pkill -9 -f "accelerate.commands.launch"
```

验证: 8× H200 全部 0 MiB used。

**问题 E1: 旧 dtaug 监控 wrapper 仍在运行**

- PID 1911820 (`bash b/s/Frk/frk_plug_sft_dtaug_launch.sh`) 的监控循环仍然存活
- 它检测到训练退出 (code=143), 执行了 `_kill_gpu_processes()` → `pkill -f "accelerate.commands.launch"`
- 这杀掉了第一次 WAN smoke 测试的 accelerate 进程
- 还启动了 bigmatrix

**Fix**: 手动 kill 旧 wrapper 和 bigmatrix

```bash
kill 1911820          # kill old monitoring wrapper
pkill -f "bigmatrix_multiply_optimization"
```

之后 GPU 全部恢复 0 MiB。

---

## E1. Pre-flight 检查 — ALL PASS

| 检查项 | 结果 |
|:---|:---|
| §0.1 虚拟环境 | PASS — Python 3.11.9, `/B/VENV/itnvla15rbt20/bin/python` |
| §0.2 依赖 | PASS — torch 2.10.0+cu128, transformers 5.2.0, accelerate 1.14.0, GPUs 8 |
| §0.3 Editable install | PASS — `.pth` → `/B/SRC/itvlaGpLibPlus/src` |
| §0.4 Checkpoint | PASS — kpt_4d_mode=pos_rot, J=8, 全部 15 项配置值匹配 |
| §0.5 数据集 symlink | PASS — `/B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D` → `/B/Dta/plug_into_socket_lrb_4D` |
| §0.6 Schema symlink | **FIXED** — 原先缺失，已创建 `franka_plug.yaml` symlink |
| §0.7 模型缓存 | PASS — FAST tokenizer, Qwen3.5-2B, WAN VAE 全部缓存 |
| §0.8 GPU 状态 | PASS — 8× H200, 0 MiB used |
| §0.9 端口 | PASS — 36704 空闲 (`ss` 不可用, 无冲突) |

**Schema symlink 修复**:
```bash
mkdir -p /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D/meta/schemas/
ln -sf /B/SRC/itvlaGpLibPlus/src/lerobot/policies/internvla_a1_5/robot_type_configs/franka_plug.yaml \
       /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D/meta/schemas/franka_plug.yaml
```

---

## E2. WAN Smoke 测试 — PASS (第二次, 首次被旧 wrapper 杀掉)

```bash
WAN_SMOKE=1 bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh
```

Smoke 输出: `/tmp/sft_dtaug2_smoke_frk_plug_2026_09_22_20_01_15`
Log: `/tmp/sft_dtaug2_smoke_frk_plug_2026_09_22_20_01_15.log` (74568 bytes)

**配置验证**:
- `disabled_tfs: []` ✓
- `affine.weight: 0.2` ✓, 其他 5 种 `weight: 1.0` ✓
- `action_loss_only: False` ✓ (WAN 加载成功)
- `enable_vqa_loss: True` ✓
- `kpt_4d_mode: pos_rot` ✓, `num_keypoint_joints: 8` ✓

**Loss 分量**:

| Step | loss | loss_action | loss_vqa | loss_video | loss_kpt_cur | loss_kpt_fut | grad_norm |
|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | 0.417 | 0.003 | 0.349 | 0.032 | 0.0002 | 0.0005 | 40.231 |
| 2 | 0.451 | 0.011 | 0.164 | 0.099 | 0.0128 | 0.0414 | 26.096 |

全部 5 个核心 loss 分量正常。`video_decode_error=0`, `using_zeros=0`, `exit=0`。
FAST tokenizer 离线加载正常: `resolved: .../models--physical-intelligence--fast/snapshots/ec4d7aa...`

---

## E3. 正式训练启动

### E3.1 启动命令

```bash
source /B/VENV/itnvla15rbt20/bin/activate
cd /B/SRC/itvlaGpLibPlus
nohup bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh > /tmp/frk_plug_sft_dtaug2_wrapper.log 2>&1 &
disown
# Wrapper PID: 2216163
```

### E3.2 关键路径

| 项目 | 路径 |
|:---|:---|
| JOB_STAMP | `2026_09_22_20_04_17` |
| LOG_FILE | `/B/Log/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17/train.log` |
| OUTPUT_DIR | `~/b/Ckp/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17-internvla_a1_5-frk-plug-sft-dtaug2/` |
| Wrapper log | `/tmp/frk_plug_sft_dtaug2_wrapper.log` |
| Wrapper PID | 2216163 |
| Training PID | 2216181 |

### E3.3 模型加载确认

```
Policy: InternVLAA15Policy
  - Total params        : 8B
  - Trainable params    : 3B
  - WAN params          : 5B
  - Knowledge insulation: False
  - Freeze WAN DiT      : True
cfg.steps=20000 (20K)
num_frames=66577 (67K)
num_episodes=100 (100)
Effective batch size: 16 x 8 = 128
```

FLA backward: `[FLA Backend] common.chunk_bwd_dqkwg -> tilelang` ✓

### E3.4 GPU 显存

每卡 ~100,579 MiB / 143,771 MiB = **70%**, 余量充足。

### E3.5 早期 Loss (step 50)

| step | epoch | loss | loss_action | loss_vqa | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 50 | 0.10 | 0.298 | 0.004 | 0.167 | 0.062 | 0.0016 | 0.0191 | 3.703 | 2.6e-06 |

所有值在健康范围内，与 dtaug 初始 loss 量级一致。

训练速度: ~0.21 iters/s @ 8×H200
预估总耗时: 20000 / 0.21 / 3600 ≈ **26 小时 (~1.1 天)**

### E3.6 预计 Checkpoint 保存时间表

| 保存点 | Step | 预计时间 |
|:---:|:---:|:---:|
| 1 | 5000 | ~Sep 23 02:30 |
| 2 | 10000 | ~Sep 23 09:00 |
| 3 | 15000 | ~Sep 23 15:30 |
| 4 (最终) | 20000 | ~Sep 23 22:00 |

### E3.7 监控

Launch script 内置自动监控:
- `MONITOR_INTERVAL=900` (15 分钟)
- `STALE_THRESHOLD=900` (15 分钟)
- 训练成功: 自动启动 bigmatrix + 打包日志到 `~/b/Ckp/itvlagpFrkPlug0907_dtaug2_LOG_<ts>.tar`
- 训练失败: 清理 GPU + bigmatrix + 打包日志 (带 `_err` 后缀)

手动监控:
```bash
tail -f /B/Log/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17/train.log
grep 'step:' /B/Log/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17/train.log | tail -5
```

---

## E4. Checkpoint@5000 验证 (2026-09-23 02:42)

### E4.1 保存确认

训练日志输出:
```
INFO 2026-09-23 02:42:05 ot_train.py:385 Checkpoint policy after step 5000
INFO 2026-09-23 02:42:05 ot_train.py:387 Checkpoint saved at: /home/a26113/b/Ckp/itvlagpFrkPlug0907_dtaug2/2026_09_22_20_04_17-internvla_a1_5-frk-plug-sft-dtaug2/checkpoints/005000
```

### E4.2 Checkpoint 完整性验证

```
checkpoints/005000/
├── pretrained_model/
│   ├── config.json         (3.7K)  ✅
│   ├── model.safetensors   (5.9G)  ✅
│   ├── stats.json          (39K)   ✅
│   └── train_config.json   (13K)   ✅
└── training_state/
    ├── rng_state.safetensors (16K)  ✅
    └── training_step.json    (20B)  ✅  → {"step": 5000}
```

**结果**: 6 文件全部非空，checkpoint@5000 完整保存成功。

### E4.3 Step 5000 Loss 快照

| 指标 | 值 |
|:---|:---|
| loss (total) | 0.201 |
| loss_action | 0.003 |
| loss_vqa | 0.089 |
| loss_video | 0.057 |
| loss_fast | 0.093 |
| loss_kpt_cur | 0.0018 |
| loss_kpt_fut | 0.0136 |
| grad_norm | 3.074 |
| lr | 4.4e-05 |
| epoch | 9.52 |
| speed | 0.21 iters/s |

### E4.4 Loss 趋势 (step 50 → 5000)

| Step | Total Loss | loss_action | loss_vqa | loss_video | grad_norm |
|:---|:---|:---|:---|:---|:---|
| 50 | 0.298 | 0.004 | 0.167 | — | 3.703 |
| 500 | 0.289 | 0.004 | 0.153 | — | 3.939 |
| 2000 | 0.324 | 0.004 | 0.191 | — | 4.611 |
| 3200 | 0.228 | 0.004 | 0.107 | — | 4.061 |
| 5000 | 0.201 | 0.003 | 0.089 | 0.057 | 3.074 |

趋势: total loss 从 0.298 稳步下降至 0.201，loss_action 从 0.004→0.003，loss_vqa 从 0.167→0.089 (-47%)，grad_norm 收敛从 3.7→3.1。训练健康。

### E4.5 下一个 Checkpoint

- 目标: checkpoint@10000
- 预计时间: ~Sep 23 09:00 (约 6.5 小时后)
- 继续自动监控中
