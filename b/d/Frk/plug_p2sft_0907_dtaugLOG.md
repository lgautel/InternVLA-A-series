# Phase 2 SFT + Data Augmentation 执行日志 — Franka plug_into_socket 7D

> **目的**: 对已训练 20 epoch 的 SFT checkpoint (step 10420) 继续微调 50000 步，新增图像数据增强（除 affine 外全部启用）
> **起点 Checkpoint**: `/home/a26113/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model/`
> **数据集**: `plug_into_socket_lrb_4D` — 100 episodes, 66577 frames, 30fps, 8 kpts × 7D
> **EXPR_NAME**: `itvlagpFrkPlug0907_dtaug`
> **Launch Script**: `b/s/Frk/frk_plug_sft_dtaug_launch.sh`
> **日期**: 2026-09-22
> **操作者**: Claude Code

---

## 与原 SFT 的差异

| 维度 | 原 SFT (itvlagpFrkPlug0907) | 本次 (itvlagpFrkPlug0907_dtaug) |
|:---|:---|:---|
| 起点 | Warmup ckpt@3126 | **SFT ckpt@10420** (epoch 20) |
| 数据增强 | 无 | **brightness/contrast/saturation/hue/sharpness** (affine 禁用) |
| 总步数 | 52100 (100 epoch) | **50000** |
| SAVE_FREQ | 10420 (每 20 epoch) | **10000** |
| HF 离线模式 | 未设置 | **HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1** |
| MASTER_PORT | 36702 | **36703** |
| 其他参数 | — | 与原 SFT 完全一致 |

活跃数据增强 (5 种，`max_num_transforms=3` 表示每帧随机选 3 种应用):
- `brightness`: ColorJitter(brightness=(0.8, 1.2))
- `contrast`: ColorJitter(contrast=(0.8, 1.2))
- `saturation`: ColorJitter(saturation=(0.5, 1.5))
- `hue`: ColorJitter(hue=(-0.05, 0.05))
- `sharpness`: SharpnessJitter(sharpness=(0.5, 1.5))

预计 Checkpoint 保存点: 10000, 20000, 30000, 40000, 50000 (final)

---

## 0. Pre-flight 环境准备

### 0.1 GPU 状态 — PASS

8× NVIDIA H200 全部空闲 (0 MiB used)。已 kill bigmatrix (PID=1287052)。

### 0.2 Editable install — PASS

`.pth` → `/B/SRC/itvlaGpLibPlus/src` (含 `disabled_tfs` 支持的 `transforms.py`)

### 0.3 Schema — PASS

`franka_plug.yaml` symlink 存在，robot_type=franka_plug

### 0.4 Checkpoint — PASS

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

### 0.5 Dataset symlink — PASS

`/B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D` → `/B/Dta/plug_into_socket_lrb_4D`

### 0.6 Model caches — PASS

- FAST tokenizer: cached (`physical-intelligence--fast`)
- Qwen3.5-2B: cached
- WAN2.2-TI2V-5B: cached (`Wan2.2_VAE.pth` exists)

---

## 1. WAN Smoke Test — PASS (E1 fix applied)

### E1: FAST tokenizer offline loading failure

**现象**: `OSError: We couldn't connect to 'https://huggingface.co' to load the files`
- `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` 阻止了 `AutoProcessor.from_pretrained("physical-intelligence/fast")` 的 HF Hub 访问
- 原 SFT 未设 offline mode (token 当时未过期), 但现在 HF token 已过期需要 offline mode
- `use_fast_action_tokens=true` 使得 FAST tokenizer 在训练时被调用 (`_ensure_tokenizers()`)

**根因**: `AutoProcessor.from_pretrained()` 在 offline mode 下传入 HF repo name 时无法正确解析到缓存的 snapshot 目录

**Fix**: 修改 `src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py`:
1. 新增 `_resolve_fast_local_dir()` 函数: 使用 `huggingface_hub.try_to_load_from_cache()` 将 repo name 解析为本地 snapshot 目录
2. 修改 `_ensure_tokenizers()`: 先解析 local_dir, 再用 `local_files_only=True` 从本地目录加载

```python
def _resolve_fast_local_dir(model_name_or_path: str) -> str:
    local_path = Path(model_name_or_path)
    if local_path.is_dir() and (local_path / "tokenizer.json").is_file():
        return str(local_path)
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(model_name_or_path, "tokenizer.json")
        if cached and Path(cached).is_file():
            return str(Path(cached).parent)
    except Exception:
        pass
    return model_name_or_path
```

解析结果: `physical-intelligence/fast` → `/B/VENV/hf_home/hub/models--physical-intelligence--fast/snapshots/ec4d7aa71691cac0b8bed6942be45684db2110f4`

### WAN Smoke 结果 (1 GPU, 2 steps)

| 分量 | step 1 | step 2 |
|:---|:---|:---|
| loss (total) | 0.339 | 0.452 |
| loss_action | 0.003 | 0.010 |
| loss_vqa | 0.273 | 0.175 |
| loss_video | 0.033 | 0.103 |
| loss_fast | 0.279 | 0.198 |
| loss_kpt_cur | 0.0002 | 0.0133 |
| loss_kpt_fut | 0.0004 | 0.0429 |
| grad_norm | 33.914 | 25.579 |

所有 5 个核心 loss 分量正常, 无 OOM, 无 NCCL error, 无 `video_decode_error`.
`disabled_tfs: ['affine']` 已正确解析.

### 10-step Smoke — PASS

10 steps 完成, exit=0, `video_decode_error=0, using_zeros=0`.

---

## 2. 正式训练启动

### 2.0 启动命令

```bash
nohup bash b/s/Frk/frk_plug_sft_dtaug_launch.sh > /tmp/frk_plug_sft_dtaug_wrapper.log 2>&1 &
disown
# Wrapper PID: 1911820
```

### 2.1 关键路径

- JOB_STAMP: `2026_09_22_07_05_00`
- LOG_FILE: `/B/Log/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00/train.log`
- OUTPUT_DIR: `~/b/Ckp/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00-internvla_a1_5-frk-plug-sft-dtaug/`
- Wrapper log: `/tmp/frk_plug_sft_dtaug_wrapper.log`

### 2.2 模型加载确认

```
Policy: InternVLAA15Policy
  - Total params        : 8B
  - Trainable params    : 3B
  - WAN params          : 5B
  - Knowledge insulation: False
  - Freeze WAN DiT      : True
cfg.steps=50000 (50K)
num_frames=66577 (67K)
num_episodes=100 (100)
Effective batch size: 16 x 8 = 128
```

FLA backward: `[FLA Backend] common.chunk_bwd_dqkwg -> tilelang` ✓

### 2.3 GPU 显存

每卡 ~101,721 MiB / 143,156 MiB = **71%**, 余量充足.

### 2.4 早期 Loss

| step | epoch | loss | loss_action | loss_vqa | loss_video | loss_kpt_cur | loss_kpt_fut | grdn | lr |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 50 | 0.10 | 0.272 | 0.003 | 0.151 | 0.059 | 0.0016 | 0.0191 | 3.3 | 1.3e-06 |
| 100 | 0.19 | 0.276 | 0.004 | 0.147 | 0.053 | 0.0027 | 0.0234 | 3.1 | 3.8e-06 |

训练速度: ~0.23 iters/s @ 8×H200
预估总耗时: 50000 / 0.23 / 3600 ≈ **60 小时 (~2.5 天)**

### 2.5 预计 Checkpoint 保存点

| 保存点 | Step |
|:---:|:---:|
| 1 | 10000 |
| 2 | 20000 |
| 3 | 30000 |
| 4 | 40000 |
| 5 (最终) | 50000 |

### 2.6 监控

Launch script 内置自动监控:
- `MONITOR_INTERVAL=900` (15 分钟)
- `STALE_THRESHOLD=900` (15 分钟)
- 训练成功: 自动启动 bigmatrix + 打包日志到 `~/b/Ckp/itvlagpFrkPlug0907_dtaug_LOG_<ts>.tar`
- 训练失败: 清理 GPU + bigmatrix + 打包日志 (带 `_err` 后缀)

手动监控:
```bash
tail -f /B/Log/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00/train.log
grep 'step:' /B/Log/itvlagpFrkPlug0907_dtaug/2026_09_22_07_05_00/train.log | tail -5
```

### 2.7 无额外 Error (E1 已在 smoke 阶段修复)

---

## 代码变更记录

1. **`b/s/Frk/frk_plug_sft_dtaug_launch.sh`** (新建)
   - 基于 `launch/frk_plug_sft_launch.sh`, 添加 data augmentation, HF offline, 修改步数/保存点/EXPR_NAME

2. **`src/lerobot/policies/internvla_a1_5/transform_internvla_a1_5.py`** (修改)
   - 新增 `_resolve_fast_local_dir()`: 解析 HF repo name 到本地缓存 snapshot 目录
   - 修改 `_ensure_tokenizers()`: 使用 resolved local dir + `local_files_only=True`
   - (之前 session 的修改仍保留: lazy-load `_ensure_tokenizers` 模式)

3. **`src/lerobot/datasets/transforms.py`** (之前 session 已修改)
   - `disabled_tfs: list[str]` 字段 + `ImageTransforms.__init__` 跳过逻辑

---
