# Phase 1 Warmup 执行日志 — Franka plug_into_socket 7D

> **方案文档**: `b/d/Frk/plug_p1warmup.md`
> **开始时间**: 2026-09-07
> **EXPR_NAME**: `itvlagpFrkPlug0907`
> **数据集**: `/B/Dta/plug_into_socket_lrb_4D` (66,577 frames, 100 episodes, 8×7D kpts)
> **机器**: 8× H200

---

## 0. Pre-flight 环境检查

### 0.1 虚拟环境与 PyTorch

```
$ source /B/VENV/itnvla15rbt20/bin/activate
$ python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.device_count())"
torch 2.10.0+cu128 cuda 8
```

### 0.2 Editable install

```
$ cat /B/VENV/itnvla15rbt20/lib/python3.11/site-packages/__editable__.internvla_a1_5-1.0.0.pth
/B/SRC/itvlaGp/src
```
指向正确仓库，无需修复。

### 0.3 7D kpt_4d_mode 验证

```
$ python -c "from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config; ..."
kpt_4d_mode: pos_rot
keypoint_track_input_dim: 7
7D keypoint support OK
```

### 0.4 模型权重

- InternVLA-A1.5-base: `/B/VENV/hf_home/ckpts/InternVLA-A1.5-base/config.json` — OK
- GeoPredict RoboCasa: `/B/VENV/hf_home/ckpts/GeoPredict_robocasa.pth` — OK

### 0.5 Schema symlink

```
$ ls -la src/lerobot/dataset_schemas/configs/franka_plug.yaml
-> ../../../../b/s/Frk/cfg/franka_plug.yaml
```
已在之前修复（从断裂的 `/home/luogang/...` 改为相对路径），cat 正常输出 `robot_type: franka_plug`。

### 0.6 数据 symlink

原始状态: **MISSING** — `/B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D` 不存在。

**修复操作**:
```bash
ln -sfn /B/Dta/plug_into_socket_lrb_4D /B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D
```
修复后验证: OK

### 0.7 Stats 验证

```
STATS OK: keypoint_3d mean has 56 dims, count=[66577]
```

### 0.8 Scripts 验证

- `launch/frk_plug_warmup_launch.sh`: OK (executable)
- `b/s/Frk/run_frk_plug_warmup.sh`: OK (executable)

### 0.9 GPU 状态

PID 214025 — `bigmatrix_multiply_optimization.py` 在全部 8 卡上运行。训练前需要 kill。

### 0.10 Pre-flight 结论

所有项目通过（修复了 data symlink），可以开始 smoke test。

---

## 1. Smoke 测试 (1 GPU × 10 steps)

### 1.1 Error #1: GeoPredict TrackEncoder shape mismatch（已修订）

**命令**: `SMOKE=1 bash launch/frk_plug_warmup_launch.sh`

**原始错误** (2026-09-07):
```
RuntimeError: Shape mismatch for keypoint_encoder.point_patch_embed.conv.weight:
checkpoint (256, 3, 4) vs TrackEncoder (256, 7, 4)
```

**根因**: GeoPredict RoboCasa 预训练权重的 Conv1d 首层输入维度为 3 (3D position only)，但 Franka TrackEncoder 因 `kpt_4d_mode=pos_rot` 输入维度为 7。旧代码在 shape 不匹配时直接抛异常。

**当时临时修复** (已废弃):
- `strict_shape_check=False` + `shape_skipped_sub_keys` — 部分加载中间层、skip 首层 Conv1d

**现行逻辑** (2026-09-08 起，见 `keypoints.py::geopredict_track_encoder_input_compatible`):
- `input_dim == 3` **或** checkpoint `point_patch_embed.conv.weight` shape 与模型一致 → 加载 GeoPredict
- 否则（Franka 7D 典型情况）→ **整网 TrackEncoder 随机 init** + **warning**，不再抛异常、不再部分加载

**Smoke 预期日志** (7D):
```text
WARNING ... GeoPredict TrackEncoder weights were NOT loaded from ... input_dim=7 ...
The entire TrackEncoder will remain randomly initialized.
INFO ... load_geopredict_keypoint_weights: no TrackEncoder keys loaded ...
```

### 1.2 Error #2: Transform reshape 维度不匹配

**错误**:
```
RuntimeError: shape '[1051, 8, 7]' is invalid for input of size 14056
```

**根因**: `keypoint_history_max_len=200` 只传给了 `--policy.*`，但 `--dataset.*` 侧的同名字段仍为默认值 1000。Transform 用 dataset config 的值 (h=1000+1+50=1051) 来 reshape，但数据已按 policy 的 200 裁剪 (实际 14056=251×8×7)。

**修复**: 在 `launch/frk_plug_warmup_launch.sh` 的 dataset 参数段增加:
```bash
--dataset.keypoint_history_max_len=200
```

### 1.3 Smoke 测试通过

第三次运行成功，10 步全部完成:

| step | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.9578 | 1.0743 | 0.318 | 599.2 |
| 2 | 0.4130 | 0.4913 | 0.202 | 290.8 |
| 5 | 0.1942 | 0.3109 | 0.179 | 182.1 |
| 10 | 0.0431 | 0.2026 | 0.393 | 110.6 |

- `video_decode_error=0`, `using_zeros=0`
- Checkpoint 保存于 `~/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_49_25-...-smoke/checkpoints/000010`
- kpt_cur 从 0.958 快速降到 0.043，收敛正常

### 1.4 post_check 语法 bug 修复

`grep -c` 返回 0 时 exit 1，`|| echo 0` 导致变量值为 "0\n0" 引发 `[[` 表达式错误。修复: 改用 `|| decode_err=0` 分离赋值。

---

## 2. 正式训练 (8 GPU × 3126 steps)

### 2.0 步数修正

方案文档中 ceil(66577/128) = 520, 实际脚本计算为 **521** (因为 66577/128 = 520.1328...)。

| 指标 | 方案值 | 实际值 |
|:---|:---:|:---:|
| STEPS_PER_EPOCH | 520 | **521** |
| TOTAL_STEPS | 3120 | **3126** |
| SAVE_FREQ | 1560 | **1563** |
| SCHED_WARMUP_STEPS | 520 | **521** |

差异不影响训练质量。

### 2.1 启动

**命令**:
```bash
nohup bash launch/frk_plug_warmup_launch.sh > /tmp/frk_plug_warmup_prod.log 2>&1 &
```

**关键环境变量**:
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `PROC_PER_NODE=8, BATCH_SIZE=16, EBS=128`
- `STEPS=3126, SAVE_FREQ=1563`

**JOB_STAMP**: `2026_09_07_04_53_50`
**OUTPUT_DIR**: `~/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/`
**LOG_FILE**: `/B/Log/itvlagpFrkPlug0907/2026_09_07_04_53_50/train.log`

**模型加载日志**:
- `post_init_keypoint_weights: initialized keypoint_expert from action_expert weights.` ✓
- GeoPredict: **input 不兼容，整网随机 init**（7D + 3D ckpt）— 见 warning，属预期 ✓
- `Total params: 3B, Trainable params: 927M, WAN params: 0` ✓
- `Effective batch size: 16 x 8 = 128` ✓
- 模型加载耗时 ~90 秒

### 2.2 训练过程 — Loss 轨迹

| step | epoch | loss_kpt_cur | loss_kpt_fut | loss_action | grad_norm | lr |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 0.10 | 0.7070 | 0.8278 | 0.286 | 467.0 | 2.5e-6 |
| 100 | 0.19 | 0.1120 | 0.2150 | 0.235 | 61.0 | 7.3e-6 |
| 250 | 0.48 | 0.0833 | 0.0999 | 0.112 | 16.8 | 2.2e-5 |
| 350 | 0.67 | 0.0072 | 0.0282 | 0.091 | 20.1 | 3.1e-5 |
| 500 | 0.96 | 0.0057 | 0.0262 | 0.074 | 31.8 | 4.6e-5 |
| 700 | 1.35 | 0.0047 | 0.0240 | 0.062 | 20.1 | 4.5e-5 |
| 1000 | 1.92 | 0.0021 | 0.0235 | 0.055 | 13.9 | 4.0e-5 |
| 1563 | 2.98 | **0.0012** | 0.0224 | 0.041 | 5.8 | 2.8e-5 |
| 2000 | 3.94 | 0.0016 | 0.0240 | 0.034 | 2.7 | 1.7e-5 |
| 2500 | 4.81 | 0.0013 | 0.0249 | 0.031 | 1.8 | 9.6e-6 |
| 3000 | 5.77 | 0.0014 | 0.0190 | 0.030 | 1.4 | 5.3e-6 |
| **3126** | **5.96** | **0.0019** | **0.0211** | **0.029** | **1.4** | **5.0e-6** |

### 2.3 Checkpoint 保存

| step | epoch | 时间 | 路径 |
|:---:|:---:|:---|:---|
| 1563 | 3 | 05:09:03 | `~/b/Ckp/itvlagpFrkPlug0907/.../checkpoints/001563/` |
| 3126 | 6 | 05:25:07 | `~/b/Ckp/itvlagpFrkPlug0907/.../checkpoints/003126/` |

每个 checkpoint 包含: `config.json`, `model.safetensors`, `stats.json`, `train_config.json`

### 2.4 训练完成

```
INFO 2026-09-07 05:27:37 ot_train.py:404 End of training
post_check: video_decode_error=0 using_zeros=0 exit=0
```

- 训练耗时: ~30 分钟 (04:55:21 → 05:25:07 训练本体, 05:27:37 完成保存)
- 速度: ~2.0 iter/s
- 无任何错误

---

## 3. 验收

### 3.1 验收清单

| # | 验收项 | 结果 |
|:---:|:---|:---:|
| 1 | 训练正常完成 (exit=0) | ✅ |
| 2 | 2 个 checkpoint (001563 + 003126) | ✅ |
| 3 | loss_kpt_cur < 0.01 (epoch 3 后) | ✅ (0.0012–0.0019) |
| 4 | loss_kpt_fut 收敛 | ✅ (0.019–0.025) |
| 5 | loss_action 稳定 | ✅ (0.029–0.034) |
| 6 | grad_norm 无爆炸 | ✅ (最终 ~1.4) |
| 7 | video_decode_error = 0 | ✅ |
| 8 | using_zeros = 0 | ✅ |
| 9 | ckpt config: kpt=True, J=8, mode=pos_rot | ✅ |
| 10 | 日志归档完成 | ✅ |
| 11 | bigmatrix 运行中 | ✅ |

### 3.2 Checkpoint 配置验证

```
enable_keypoint_predictor: True
num_keypoint_joints: 8
kpt_4d_mode: pos_rot
train_expert_only: True
action_loss_only: True
knowledge_insulation: True
→ Checkpoint config PASS
```

---

## 4. 善后

### 4.1 bigmatrix 启动

```
bigmatrix PID: 320442
bigmatrix running OK
```

### 4.2 日志归档

```
归档路径: ~/b/Ckp/itvlagpFrkPlug0907_LOG_26090705.tar (240K)
```

---

## 5. 文件增删改汇总

### 5.1 新增文件

| 文件 | 说明 |
|:---|:---|
| `launch/frk_plug_warmup_launch.sh` | Franka warmup 专用 launch script |
| `b/s/Frk/run_frk_plug_warmup.sh` | 训练编排与监控 wrapper |
| `b/d/Frk/plug_p1warmup.md` | 实施方案文档 |
| `b/d/Frk/plug_p1warmup_0907LOG.md` | 本日志 |

### 5.2 修改文件

| 文件 | 修改内容 | 原因 |
|:---|:---|:---|
| `src/lerobot/dataset_schemas/configs/franka_plug.yaml` | 修复断裂 symlink: `/home/luogang/...` → `../../../../b/s/Frk/cfg/franka_plug.yaml` | 旧路径指向另一台机器 |
| `src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py` | `load_geopredict_keypoint_weights()` 委托 keypoints 兼容性判定 | 7D 不加载 GeoPredict，整网随机 init + warning |
| `src/lerobot/policies/internvla_a1_5/keypoints.py` | `geopredict_track_encoder_input_compatible()` + 整网加载/跳过 | 废弃 partial load / `shape_skipped_sub_keys` |
| `launch/frk_plug_warmup_launch.sh:193` | 增加 `--dataset.keypoint_history_max_len=200` | dataset config 也需要同步设为 200 |
| `launch/frk_plug_warmup_launch.sh:221-222` | `grep -c ... || echo 0` → `...) || var=0` | 修复 post_check 变量包含换行的语法错误 |

### 5.3 新增 symlink

| 路径 | 目标 |
|:---|:---|
| `/B/VENV/hf_home/lerobot/plug_into_socket_lrb_4D` → `/B/Dta/plug_into_socket_lrb_4D` | LeRobot 数据查找 |

### 5.4 训练产出

| 路径 | 内容 |
|:---|:---|
| `~/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/checkpoints/001563/` | epoch 3 checkpoint |
| `~/b/Ckp/itvlagpFrkPlug0907/2026_09_07_04_53_50-internvla_a1_5-frk-plug-warmup/checkpoints/003126/` | epoch 6 checkpoint (最终) |
| `/B/Log/itvlagpFrkPlug0907/2026_09_07_04_53_50/train.log` | 训练日志 |
| `~/b/Ckp/itvlagpFrkPlug0907_LOG_26090705.tar` | 日志归档 |

---

## 6. Error 汇总

| # | Error | 根因 | 修复 | 影响文件 |
|:---:|:---|:---|:---|:---|
| 1 | `Shape mismatch for ... conv.weight (256,3,4) vs (256,7,4)` | 7D TrackEncoder + 3D GeoPredict ckpt | 改为兼容性判定：不兼容则整网随机 init + warning，不抛异常 | `keypoints.py::geopredict_track_encoder_input_compatible` |
| 2 | `shape '[1051, 8, 7]' is invalid for input of size 14056` | `keypoint_history_max_len=200` 只传给 policy config, dataset config 仍为默认 1000 | 增加 `--dataset.keypoint_history_max_len=200` | `launch/frk_plug_warmup_launch.sh:193` |
| 3 | `[[: 0\n0: syntax error in expression` | `grep -c` 无匹配时输出 "0" 并 exit 1, `|| echo 0` 追加第二个 "0" | 改用 `...) || var=0` | `launch/frk_plug_warmup_launch.sh:221-222` |

---

*日志完成时间: 2026-09-07 05:30*
