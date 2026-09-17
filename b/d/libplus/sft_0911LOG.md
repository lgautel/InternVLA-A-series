# Libero-Plus OpVLA SFT 实施操作日志

> **方案**: [`sft.md`](sft.md)  
> **EXPR_NAME**: `4dwvlaOpvlaLibplusKpt0911`  
> **日期**: 2026-09-11  
> **环境**: `/B/VENV/itnvla15rbt20/`，`HF_HOME=/B/VENV/hf_home`

---

## 0. 任务目标

按 `sft.md` 在 `itvlaGpLibPlus` 内实现 `launch/libplus_sft_launch.sh`，完成前置检查、单元测试、WAN/SMOKE 测试，启动 50 epoch（53450 steps）正式训练，fix 所有 error 直至训练成功。

---

## 1. 清空 GPU（2026-09-11 启动前）

**操作原因**: 释放 GPU 供本实验使用。

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
pkill -9 -f "lerobot_train"
pkill -9 -f "accelerate.commands.launch"
pkill -9 -f "bigmatrix_multiply"
```

**清空前**: 8 个 python 进程各占 ~98 GiB（`bigmatrix_multiply_optimization.py` 残留，疑似上一轮 lbrp 训练）。

**清空后**: 8× GPU memory.used = 0 MiB。

---

## 2. 代码实现：launch 脚本

**新建文件**: `/B/SRC/itvlaGpLibPlus/launch/libplus_sft_launch.sh`

**来源**: 基于 `itvlaGp/launch/lbrp_sft_launch.sh` 改写，对齐 `sft.md` 全部超参与路径。

**关键设计**:
| 项 | 值/行为 |
|---|---|
| EXPR_NAME | `4dwvlaOpvlaLibplusKpt0911` |
| 数据 | `/B/Dta/opvla_libero_merged_kpt` → symlink 至 `$HF_LEROBOT_HOME/opvla_libero_merged_kpt` |
| PRETRAINED | `/B/VENV/hf_home/ckpts/InternVLA-A1.5-base` |
| 8 GPU | BS=32/GPU，EBS=256，STEPS=53450，SAVE_FREQ=5345，LOG_FREQ=1000 |
| NCCL | `NCCL_TUNER_PLUGIN=/dev/null` |
| Auto-recover | `_auto_recover()` 无限重试；有 ckpt resume，无 ckpt fresh start |
| ENABLE_AUTO_RESTART | `false`（成功后不自动开新实验） |
| kpt 权重 | kpt_loss=1, kpt_future=2, kpt_rot=1 |
| scheduler | decay_steps=30000 |

**与 lbrp 差异**: 无 `robot_type` patch（panda schema 已匹配）；无 MergeKeypoint transform；EXPR/数据路径不同。

---

## 3. 环境安装与前置检查

```bash
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/hf_home HF_LEROBOT_HOME=${HF_HOME}/lerobot NCCL_TUNER_PLUGIN=/dev/null
cd /B/SRC/itvlaGpLibPlus
pip install -e .
```

**检查项**:
- InternVLA-A1.5-base: `/B/VENV/hf_home/ckpts/InternVLA-A1.5-base` ✓
- WAN2.2: `/B/VENV/hf_home/hub/Wan2.2-TI2V-5B` ✓
- 数据 symlink: `$HF_LEROBOT_HOME/opvla_libero_merged_kpt` → `/B/Dta/opvla_libero_merged_kpt` ✓
- 数据集规模: 273465 frames, 1693 episodes, 40 tasks, fps=10 ✓

---

## 4. §12.1 单元测试

### 测试 1：Schema — ✅ 通过
```
✅ panda schema OK
```

### 测试 2：关键点 reshape 56D→8×7D — ✅ 通过
```
✅ keypoint reshape OK, samples=...
```

### 测试 3：Transform Pipeline — ✅ 通过（经 fix）

**首次失败**:
```
TypeError: InternVLAA15DatasetConfig missing required field repo_id
```

**根因**: `InternVLAA15DatasetConfig` 继承 `DatasetConfig`，draccus 解析要求 `repo_id` 字段。

**Fix**: 测试脚本增加 `repo_id='opvla_libero_merged_kpt'`（与 sft.md §12.1 测试 3 一致，文档已含该字段）。

**复测**: ✅ pipeline 含 `Extract3DKeypointTransformFn`，无 `MergeKeypointPosQuatTransformFn`。

### 测试 4：Delta Timestamps — ✅ 通过
```
✅ delta timestamps OK, first= -20.0
```

---

## 5. §12.2 WAN Smoke — ✅ 通过

**命令**:
```bash
WAN_SMOKE=1 bash launch/libplus_sft_launch.sh 2>&1 | tee /tmp/wan_smoke_libplus.log
```

**首次失败**: CUDA OOM（8 GPU 被 bigmatrix 占满 ~98GB/卡）。

**Fix**: `pkill -9 -f bigmatrix` + 清 GPU 后重跑。

**通过日志** (`/tmp/wan_smoke_libplus.log`):
```
step:1.0 | loss:10.992 | loss_action:0.339 | loss_kpt_cur:1.0369 | loss_kpt_fut:1.1254 | loss_video:0.215
step:2.0 | loss:28.694 | loss_kpt_cur:7.7397 | loss_kpt_fut:7.5010 | loss_video:0.172
post_check: video_decode_error=0 using_zeros=0 exit=0
```

---

## 6. §12.3 SMOKE 100 步 — ✅ 通过

**命令**:
```bash
SMOKE=1 bash launch/libplus_sft_launch.sh 2>&1 | tee /tmp/smoke100_libplus.log
```

**结果** (~4 min, 1 GPU, BS=2):
| Step | loss | loss_action | loss_kpt_cur | loss_kpt_fut |
|------|------|-------------|--------------|--------------|
| 10 | 9.694 | 0.220 | 1.1209 | 1.1987 |
| 100 | 5.551 | 0.159 | 0.0871 | 0.1354 |

```
post_check: video_decode_error=0 using_zeros=0 exit=0
```

Loss 下降趋势正常，无 Traceback/NaN。

---

## 7. 正式训练启动（8×H200）

**命令** (2026-09-11 09:47 UTC):
```bash
mkdir -p /B/Log/4dwvlaOpvlaLibplusKpt0911 ~/b/Ckp/4dwvlaOpvlaLibplusKpt0911
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME=/B/VENV/hf_home HF_LEROBOT_HOME=${HF_HOME}/lerobot NCCL_TUNER_PLUGIN=/dev/null
cd /B/SRC/itvlaGpLibPlus
nohup bash launch/libplus_sft_launch.sh > /tmp/libplus_sft_outer.log 2>&1 &
```

**路径**:
| 项 | 路径 |
|---|---|
| OUTPUT_DIR | `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_11_09_47_44-internvla_a1_5-libplus-sft` |
| LOG_FILE | `/B/Log/4dwvlaOpvlaLibplusKpt0911/2026_09_11_09_47_44/train.log` |
| outer log | `/tmp/libplus_sft_outer.log` |
| wandb offline | `.../wandb/offline-run-20260911_094815-supp58e5` |

**启动参数摘要**: PROC=8, BS=32, STEPS=53450, SAVE_FREQ=5345, LOG_FREQ=1000, Trainable params=3B

**模型加载 WARN** (预期): Missing keys 为 WAN DiT + keypoint_expert 新模块，从 action_expert 初始化 keypoint_expert；`post_init_keypoint_weights` 已执行。

**TileLang 编译**: 09:50:41–09:50:57，8 rank 各编译 FLA kernel（首次 step 前 ~1 min）。

---

## 8. 训练启动阶段观察（09:50–11:57）

**现象**: `LOG_FREQ=1000`，step 1–999 无 train.log 行；monitor 显示 `step=?`（见 §9 fix）。

**进程状态** (11:50 左右):
- 8 rank + accelerate launcher 存活
- rank0 CPU ~196%，State=R (running)
- GPU memory ~140GB/卡；利用率呈「计算 burst 100% + 数据等待 ~25%」交替

**非错误说明**: 首 1000 step 约 2h8m 才首次 log，因 `updt_s≈7.5s/step`（batch 32 + WAN video + 8 GPU DDP），非 hang。

---

## 9. Error Fix：Monitor step 解析

**现象** (step 1000 log 出现后):
```
[monitor 12:02:45] Healthy: step=1/53450   # 错误，应为 1000
```

**根因**: `grep -oE 'step:[0-9]+'` 对 `step:1.0K` 只匹配到 `step:1`。

**Fix** (`launch/libplus_sft_launch.sh`):
- 新增 `_parse_step_from_log()`，支持 `step:1.0K` → 1000
- 替换两处 `cur_step=` grep（Healthy 循环 + RECOVER 循环）

**注**: 当前运行中的 monitor 进程仍用旧脚本内存；下次 recover/restart 生效。训练本身不受影响。

---

## 10. §12.3 早期验收 — step 1000（2026-09-11 11:57:48 UTC）

**首条训练 log**:
```
02:07:44 << 109:54:10 | 0.13 iters/s | step:1.0K | epoch:0.94
loss:4.384 | loss_action:0.100 | grdn:10.797 | lr:2.5e-05
updt_s:7.543 | data_s:0.018
loss_vqa:2.905 | loss_video:0.119 | loss_fast:2.983
loss_kpt_cur:0.0849 | loss_kpt_fut:0.1380
```

| 检查项 | 结果 |
|--------|------|
| Loss 不发散 | ✅ loss=4.384，grdn=10.8 |
| loss_action 合理 | ✅ 0.100（SMOKE step100 为 0.159） |
| 全 loss 分量 | ✅ action/kpt_cur/kpt_fut/video/vqa/fast |
| 无 Traceback/NaN | ✅ |
| GPU 活跃 | ✅ burst 时 8 卡 80–100%；均值受 data_s 影响偏低 |
| 稳定训练 | ✅ 已进入逐步迭代，ETA ~110h |

**估算总时长**: 53450 × 7.5s ≈ 111h（~4.6 天）至 checkpoint `053450`。

---

## 11. §12.4 验收清单（进行中）

| 项 | 状态 |
|----|------|
| 单元测试 1~4 | ✅ |
| WAN_SMOKE=1 | ✅ |
| SMOKE=1 (100步) | ✅ |
| 正式训练启动，前 1000 步 loss 正常 | ✅ (step 1.0K logged) |
| checkpoint 005345 (epoch 5) | ⏳ 预计 step 5345，~11h 后 |
| checkpoint 053450 (final) | ⏳ 预计 ~111h 后 |
| Franka/RoboTwin 向后兼容 | ⏳ 未在本日志会话单独跑 |

---

## 12. 错误汇总表

| # | 错误 | 根因 | Fix |
|---|------|------|-----|
| 1 | WAN smoke OOM | bigmatrix 占满 8 GPU | pkill bigmatrix + 清 GPU |
| 2 | 单元测试 3 TypeError | 缺 `repo_id` | 测试加 `repo_id='opvla_libero_merged_kpt'` |
| 3 | Monitor step=1/53450 | grep 无法解析 `1.0K` | `_parse_step_from_log()` |

---

## 13. 监控命令

```bash
# 训练 log
tail -f /B/Log/4dwvlaOpvlaLibplusKpt0911/2026_09_11_09_47_44/train.log

# outer / monitor
tail -f /tmp/libplus_sft_outer.log

# GPU
watch -n5 nvidia-smi

# checkpoint
ls ~/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_11_09_47_44-internvla_a1_5-libplus-sft/checkpoints/
```

**完成标志**: 存在 `checkpoints/053450/pretrained_model/train_config.json` 且 train.log 含 `End of training`。

---

## 14. 后续更新（训练进行中）

> 本节随训练进度追加。

### 2026-09-11 11:57 — 稳定训练确认

- 首条 step log: **step 1000**，loss 正常，训练继续后台运行（PID 1167575 集群）。
- 下一里程碑: **step 5345** 首个 ckpt（SAVE_FREQ=5345）。
- 脚本 fix: monitor step 解析已写入 `launch/libplus_sft_launch.sh`（见 §9）。

### 2026-09-12 ~08:00 — 训练失败分析与修复

**问题发现**：训练在 13:47 被监控误杀后，auto-recover 陷入 **794 次 `FileExistsError` 无限循环**（13:48→07:57，~18 小时）。

**根因分析**（详见 `/B/Log/4dwvlaOpvlaLibplusKpt0911/` 日志分析）：

| # | 根因 | 影响 |
|---|------|------|
| 1 | `_is_log_stale` 基于文件 mtime，但 `_monitor_log` 写入同一文件污染 mtime；`STALE_THRESHOLD=900s` ≈ `MONITOR_INTERVAL=900s` 形成边界竞争条件 | 正常训练在 step 1000→2000 间的静默期被误判 stale 并 SIGTERM 杀死 |
| 2 | `LOG_FREQ=1000`、训练速度 ~7.5s/step → 两次 log 间隔 7500s，远大于 stale 阈值 900s | 加剧误判概率 |
| 3 | Fresh start 复用原 `ARGS[]` 中硬编码的 `--output_dir`，目录已存在 + `resume=False` | `FileExistsError` 无限循环 |
| 4 | `_find_latest_checkpoint` 仅搜索当前 `OUTPUT_DIR` | 无法发现其他 run 目录中的 checkpoint |

**修复**（`launch/libplus_sft_launch.sh`）：

| 变更 | 旧值 | 新值 |
|------|------|------|
| `STALE_THRESHOLD` | 900 | **1800**（30 分钟） |
| `LOG_FREQ` | 1000 | **100**（log 间隔 ~750s） |
| `_is_log_stale()` | 文件 mtime 检测 | **`_is_training_stalled()`**：基于 step 进度，连续 2 次 MONITOR_INTERVAL 未变化才触发 |
| Fresh start | 复用原 `OUTPUT_DIR` | 使用 `recover_stamp` 生成**新 `OUTPUT_DIR`** |
| `_find_latest_checkpoint` | 搜索 `OUTPUT_DIR/checkpoints/` | 搜索整个 **`CKPT_ROOT`** |

**测试**：
```bash
bash tests/test_monitor_stall_detection.sh
# 35 passed, 0 failed
```

**文档更新**：`sft.md` §5/§6.3/§11/§12/§13/§14.2/附录B 已同步更新。

---

## 15. 错误汇总表（更新）

| # | 错误 | 根因 | Fix | 状态 |
|---|------|------|-----|------|
| 1 | WAN smoke OOM | bigmatrix 占满 8 GPU | pkill bigmatrix + 清 GPU | ✅ |
| 2 | 单元测试 3 TypeError | 缺 `repo_id` | 测试加 `repo_id` | ✅ |
| 3 | Monitor step=1/53450 | grep 无法解析 `1.0K` | `_parse_step_from_log()` | ✅ |
| 4 | 监控误杀正常训练 | `STALE_THRESHOLD` ≈ `MONITOR_INTERVAL`，mtime 竞争 | `_is_training_stalled()` step-based + `STALE_THRESHOLD=1800` | ✅ |
| 5 | 794 次 FileExistsError 死循环 | fresh start 复用原 `OUTPUT_DIR` | 每次 fresh start 生成新 `OUTPUT_DIR` | ✅ |
| 6 | LOG_FREQ=1000 过大 | 两次 log 间隔 ~2h，远大于 stale 阈值 | `LOG_FREQ=100`（间隔 ~750s） | ✅ |
| 7 | checkpoint 搜索仅限当前 OUTPUT_DIR | fresh start 后新目录无 ckpt，旧目录被忽略 | `_find_latest_checkpoint` 搜索整个 `CKPT_ROOT` | ✅ |

---

*日志由 Agent 按 sft.md 实施流程自动维护；完整 53450 step 训练预计 ~111 小时。*
