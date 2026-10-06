# Pipeline3_2 v2 执行日志

> **开始时间**: 2026-10-04
>
> **Pipeline 版本**: pipeline3_2.markdown
>
> **EXPR_NAME**: `4dwvlaLbPlusGolV2_1002`
>
> **数据**: `/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/` (14D state, 512604 frames)
>
> **与 pipeline3 (1001) 的关键差异**:
> - Warmup: 1 epoch (4005 steps) → 原 2 epoch (8010)
> - SFT: 3 epochs (12015 steps) → 原 4 epoch (16020)
> - SFT p_schedule: [0.3, 0.1] → 原 [0.5, 0.3, 0.1]
> - eval_v2_wrapper.sh: 修复了 GPU server 进程不释放的问题

---

## Step 0: GPU 清理

```
操作: 检查 GPU 状态
结果: 全部 8 GPU 已清空 (0 MiB each, 每卡总显存 143771 MiB ≈ 140 GB)
无需 kill 任何进程
```

---

## Step 0.5: eval_v2_wrapper.sh GPU Server 清理修复

### 根因分析

评估脚本 derived script 中的 server 启动模式为:
```bash
(
    source "${SERVER_VENV}/bin/activate"
    cd "${PROJ}"
    export PYTHONPATH=...
    CUDA_VISIBLE_DEVICES=${GPU_IDX} python evaluation/LIBERO2/policy_server/server_policy.py ...
) &
local SERVER_PID=$!
```

worker 完成后执行 `kill ${SERVER_PID}`，但 `$!` 捕获的是 **subshell** 的 PID，
不是 `python` 进程的 PID。Bash 在 subshell 中有 `source` 修改环境后，不一定会
对最后的 `python` 命令做 exec 优化（将 subshell 替换为 python）。因此:
- `kill ${SERVER_PID}` 杀死的是 subshell wrapper
- python 子进程可能未收到 SIGTERM，成为 orphan 进程
- orphan server 继续占用 GPU 显存（每个 ~13 GB）

在 1001 实验中观察到: 6/8 GPU 完成任务后 server 进程仍在运行（0% GPU 利用率但
占用 13 GB 显存），直到 eval wrapper 退出后才被清理。

### 修复方案（两层防御）

**层 1 — 源头修复（sed patch）**: 在 derived script 的 server 启动命令前插入 `exec`:
```bash
sed -i 's|CUDA_VISIBLE_DEVICES=${GPU_IDX} python evaluation/|CUDA_VISIBLE_DEVICES=${GPU_IDX} exec python evaluation/|' "${DERIVED}"
```
`exec` 使 subshell 被 python 进程替换，确保 `$!` 捕获的 PID 就是 python PID。
`kill ${SERVER_PID}` 直接发送 SIGTERM 给 python。

**层 2 — 安全网（Phase 5 清理）**: 在 eval 完成后，按端口逐个检查并杀死残留 server:
```bash
for i in $(seq 0 7); do
    PORT=$((BASE_PORT + i))
    SERVER_PID=$(lsof -ti tcp:${PORT} 2>/dev/null || true)
    if [ -n "${SERVER_PID}" ]; then
        kill ${SERVER_PID} 2>/dev/null || true
    fi
done
sleep 2
# Force-kill survivors
for i in $(seq 0 7); do
    PORT=$((BASE_PORT + i))
    SERVER_PID=$(lsof -ti tcp:${PORT} 2>/dev/null || true)
    if [ -n "${SERVER_PID}" ]; then
        kill -9 ${SERVER_PID} 2>/dev/null || true
    fi
done
```

### 文件变更

```
文件: b/s/libplus2/gol2/eval_v2_wrapper.sh
变更 1 (line ~108): 新增 sed patch — `exec python` 替换 `python`
变更 2 (line ~115-140): Phase 5 开始处新增端口级 orphan server 清理逻辑
```

---

## Step 0.6: p2v2_sft_launch.sh EXPR_NAME 可配置化

```
文件: b/s/libplus2/gol2/p2v2_sft_launch.sh
变更: line 22 EXPR_NAME="..." → EXPR_NAME="${EXPR_NAME:-4dwvlaLbPlusGolV2_1001}"
      line 23 DATASET_REPO_ID="..." → DATASET_REPO_ID="${DATASET_REPO_ID:-libero_plus_goal_lrb3_4Dv2}"
原因: 支持通过 env var 覆盖 EXPR_NAME，避免为每个实验创建完整副本脚本
```

---

## Step 1: 前提检查 (§0.3)

### 1.1 数据 symlink

```
操作: 检查 ln -sfn /B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2 /B/VENV/hf_home/lerobot/libero_plus_goal_lrb3_4Dv2
结果: SYMLINK OK
```

### 1.2 Schema

```
操作: test -f src/lerobot/dataset_schemas/configs/libero_goal_4dv2.yaml
结果: SCHEMA OK
```

### 1.3 Contract

```
操作: test -f /B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/meta/goal_train_eval_contract.json
结果: CONTRACT OK
```

---

## Step 2: Warmup 训练 (Phase 1, 1 epoch)

### 配置

```
命令: EXPR_NAME=4dwvlaLbPlusGolV2_1002 STEPS=4005 SAVE_FREQ=4005 SCHED_WARMUP_STEPS=1000 \
      nohup bash b/s/libplus2/gol2/p1v2_warmup_launch.sh > /tmp/warmup_1002.log 2>&1 &
PID: 1791914
开始时间: 2026-10-04 01:37:03 UTC
OUTPUT_DIR: ~/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_01_37_03-internvla_a1_5-lbplus-gol-warmup
LOG_FILE: /B/Log/4dwvlaLbPlusGolV2_1002/2026_10_04_01_37_03/train.log

配置覆盖:
  EXPR_NAME: 4dwvlaLbPlusGolV2_1002 (env override)
  DATA_REPO_ID: libero_plus_goal_lrb3_4Dv2 (继承)
  STEPS: 4005 (1 epoch, env override, 原 8010)
  SAVE_FREQ: 4005 (仅保存最终 ckpt, env override, 原 8010)
  SCHED_WARMUP_STEPS: 1000 (1/4 epoch LR ramp, env override, 原 4005)
  SCHED_DECAY_STEPS: 4005 (= STEPS, 自动)
  BATCH_SIZE: 16, PROC_PER_NODE: 8, EBS: 128
  Loss: action=2.0, kpt=10.0, kpt_future=12.0
  action_expert_lr_scale: 0.04
  Augmentation: blackout w=2.0, p_schedule=[0.8]
```

### 启动验证

```
8 GPU 全部启动, 每卡 ~1515 MiB (模型加载中)
WANDB_MODE: offline
kpt_4d_mode: pos_rot (7D), num_keypoint_joints: 8, keypoint_history_max_len: 92
Total params: 3B, Trainable: 927M, Action expert: 460M, WAN: 0, Learnable tokens: 50
Knowledge insulation: True, Inference backend: standard
```

### 训练进度 (实时)

```
step  50  | loss 136.3 | action 0.260 | kpt_cur 0.935 | kpt_fut 1.053 | lr 1.3e-06 | 1.44 it/s
step 200  | loss  19.3 | action 0.185 | kpt_cur 0.092 | kpt_fut 0.150 | lr 8.8e-06 | 1.93 it/s
step 500  | loss   8.1 | action 0.139 | ...                           | lr 2.3e-05 | ~1.95 it/s
step 1000 | loss   6.5 | action 0.124 | kpt_cur 0.033 | kpt_fut 0.048 | lr 4.4e-05 | ~1.98 it/s
step 1500 | loss   5.7 | action 0.121 | kpt_cur 0.025 | kpt_fut 0.044 | lr 3.7e-05 | 1.95 it/s
step 1650 | loss   5.9 | action 0.122 | kpt_cur 0.027 | kpt_fut 0.045 | lr 3.5e-05 | 1.98 it/s
step 2000 | loss   5.2 | action 0.117 | kpt_cur 0.026 | kpt_fut 0.038 | lr 2.4e-05 | 1.93 it/s
step 2500 | loss   5.2 | action 0.116 | kpt_cur 0.030 | kpt_fut 0.039 | lr 1.9e-05 | 1.91 it/s
step 3000 | loss   5.1 | action 0.114 | kpt_cur 0.029 | kpt_fut 0.038 | lr 1.2e-05 | 1.93 it/s
step 3500 | loss   4.6 | action 0.113 | kpt_cur 0.027 | kpt_fut 0.034 | lr 6.9e-06 | 1.87 it/s
step 4005 | loss   4.8 | action 0.115 | kpt_cur 0.025 | kpt_fut 0.036 | lr 5.0e-06 | 2.03 it/s ← 完成
```

### 训练结果

```
结束时间: 2026-10-04 02:13:51 UTC
总耗时: 35 分 26 秒
平均速度: ~1.93 iters/s

最终 loss:
  total: 4.847, action: 0.115, kpt_cur: 0.025, kpt_fut: 0.036
  loss_video: 0.000 (action_loss_only=True, 不加载 WAN)

Checkpoint: ~/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_01_37_03-internvla_a1_5-lbplus-gol-warmup/checkpoints/004005/pretrained_model/
  model.safetensors: 6.3 GB
  stats.json: state_dim=14 ✓, action_dim=7 ✓
  config.json, train_config.json: 正常
```

**Step 2: PASSED** ✅

---

## Step 3: SFT 训练 (Phase 2, 3 epochs)

### 配置

```
命令: EXPR_NAME=4dwvlaLbPlusGolV2_1002 \
      PRETRAINED_CKPT=$HOME/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_01_37_03-internvla_a1_5-lbplus-gol-warmup/checkpoints/004005/pretrained_model \
      STEPS=12015 SAVE_FREQ=4005 SCHEDULER_WARMUP=4005 SCHEDULER_DECAY=12015 \
      P_SCHEDULE="[0.3, 0.1]" \
      nohup bash b/s/libplus2/gol2/p2v2_sft_launch.sh > /tmp/sft_1002.log 2>&1 &
PID: 1798550
开始时间: 2026-10-04 02:17:33 UTC
OUTPUT_DIR: ~/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft
LOG_FILE: /B/Log/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33/train.log

配置覆盖 (env override):
  EXPR_NAME: 4dwvlaLbPlusGolV2_1002
  PRETRAINED_CKPT: warmup@004005 (state_dim=14 已验证)
  STEPS: 12015 (3 epochs, 原 16020)
  SAVE_FREQ: 4005 (3 个 ckpt: 004005, 008010, 012015)
  SCHEDULER_WARMUP: 4005
  SCHEDULER_DECAY: 12015
  P_SCHEDULE: [0.3, 0.1] (epoch 0: 0.3, epoch 1: 0.1, epoch 2: 0.1 clamped, 原 [0.5, 0.3, 0.1])

SFT 特有配置 (vs warmup):
  action_loss_only: false (启用 WAN video branch)
  enable_vqa_loss: true (FAST token loss)
  knowledge_insulation: false (允许 action expert 看到 prefix)
  train_expert_only: false (全模型微调)
  gradient_checkpointing: true
  action_expert_lr_scale: 1.0 (warmup 0.04)
  action_loss_weight: 10.0 (warmup 2.0)
  kpt_loss_weight: 1.0 (warmup 10.0)
  kpt_future_loss_weight: 1.5 (warmup 12.0)
  freeze_learnable_tokens: false (warmup true)
  BATCH_SIZE: 16, PROC_PER_NODE: 8, EBS: 128

GPU 内存: 每卡 ~100 GB (含 WAN 模型)
```

### 启动验证

```
SFT v2 checkpoint guard: state_dim=14 PASSED
8 GPU 全部启动, FAST tokenizer loaded
DDP find_unused_parameters=True (expected warning)
FLA Backend: tilelang
```

### 训练进度

```
step   100 | loss 5.628 | action 0.118 | video 0.530 | vqa 3.838 | fast 3.923 | lr 6.4e-07 | 0.23 it/s
step   500 | loss 4.144 | action 0.095 | video 0.185 | vqa 2.932 | fast 3.022 | lr 5.6e-06 | 0.23 it/s
step  1000 | loss 3.408 | action 0.060 | video 0.127 | vqa 2.486 | fast 2.627 | lr 1.1e-05 | ~0.24 it/s
step  2000 | loss 2.083 | action 0.029 | video 0.107 | vqa 1.397 | fast 1.556 | lr 2.2e-05 | ~0.24 it/s
step  4005 | loss 1.027 | action 0.015 | video 0.098 | ...         | lr 4.3e-05 | ~0.24 it/s ← ckpt 1
step  8010 | loss 0.693 | action 0.011 | video 0.095 | ...         | lr 1.2e-05 | ~0.24 it/s ← ckpt 2
step 12015 | loss 0.660 | action 0.012 | video 0.094 | vqa 0.383 | fast 0.443 | lr 5.0e-06 | 0.24 it/s ← ckpt 3 (主评估)
```

### 训练结果

```
结束时间: 2026-10-04 16:44:20 UTC
总耗时: 14 小时 27 分
平均速度: 0.24 iters/s (~4.1s/step)
video_decode_error: 0
using_zeros: 0
Exit code: 0

最终 loss (step 12015, epoch 3.00):
  total: 0.660, action: 0.012, video: 0.094, vqa: 0.383, fast: 0.443
  kpt_cur: 0.011, kpt_fut: 0.038

Checkpoints:
  004005 (epoch 1) — 2026-10-04 07:12
  008010 (epoch 2) — 2026-10-04 11:57
  012015 (epoch 3) — 2026-10-04 16:39 ← 主评估 ckpt
  last → 012015

主评估 Checkpoint:
  ~/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/
  model.safetensors: 6.3 GB
  stats.json: state_dim=14 ✓, action_dim=7 ✓
  config.json, train_config.json: 正常

Logs archived: ~/b/Ckp/4dwvlaLbPlusGolV2_1002_sft_LOG_20261004_164426.tar
```

**Step 3: PASSED** ✅

---

## Step 4: LIBERO-plus 仿真评估 (Phase 3)

### 配置

```
命令: EXPR_NAME=4dwvlaLbPlusGolV2_1002 \
      CKPT_PATH=$HOME/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model \
      nohup bash b/s/libplus2/gol2/eval_v2_wrapper.sh > /tmp/eval_1002.log 2>&1 &
PID: 1824040
开始时间: 2026-10-05 00:35:53 UTC
EVAL_LOG_DIR: /B/Log/4dwvlaLbPlusGolV2_1002/20261005_003553_eval
EVAL_MODE: full
```

### Preflight v2

```
  ✓ config.json, model.safetensors, stats.json, train_config.json
  ✓ contract schema=2, robot_type=libero_goal_4dv2, state_dim=14
  ✓ ckpt stats state dim=14
  ✓ ckpt config correct
```

### 服务器启动

```
  8 GPU servers started (ports 5784-5791)
  每卡 ~13 GB (inference mode)
  sed patches applied: healthcheck, eval_goal_plus_v2, MagickWand, max_steps, exec-server
```

### GPU 完成时间

```
GPU 6: 03:09 UTC (2h32m) — 最快
GPU 7: 03:30 UTC (2h53m)
GPU 0: 03:54 UTC (3h17m)
GPU 3: 04:29 UTC (3h52m)
GPU 2: 04:40 UTC (4h03m)
GPU 1: 04:53 UTC (4h16m)
GPU 4: 05:38 UTC (5h01m)
GPU 5: 08:16 UTC (7h39m) — 最慢 (Sensor Noise tasks, CPU-bound)
```

### 评估结果

```
完成时间: 2026-10-05 08:16:41 UTC
总耗时: 7 小时 39 分 (受 GPU 5 Sensor Noise 瓶颈限制)
总任务数: 2591, 0 crashes

Overall Success Rate: 1019/2591 = 39.33%

Per-category breakdown:
  Light Conditions     : 228/279 = 81.72%
  Background Textures  : 198/281 = 70.46%
  Objects Layout       : 244/425 = 57.41%
  Language Instructions: 178/410 = 43.41%
  Sensor Noise         :  91/379 = 24.01%
  Camera Viewpoints    :  64/408 = 15.69%
  Robot Initial States :  16/409 =  3.91%

Results JSON: /B/Log/4dwvlaLbPlusGolV2_1002/20261005_003553_eval/overall_results.json
Archive: /B/Log/4dwvlaLbPlusGolV2_1002/eval_goal_20261005_003553.tar.gz
```

### GPU Server 清理验证

```
eval_v2_wrapper.sh Phase 5 端口级清理执行完毕
exec patch 生效: server PID == python PID, kill 直接送达 python 进程
无 orphan server 残留 (bigmatrix 正常重启, PID 2763634)
```

**Step 4: PASSED** ✅

---

## Pipeline 1002 总结

| Step | 阶段 | 状态 | 耗时 | 关键指标 |
|------|------|------|------|----------|
| 0 | GPU 清理 | ✅ | - | 8 GPU 空闲 |
| 0.5 | eval GPU fix | ✅ | - | exec patch + port cleanup |
| 0.6 | p2v2 EXPR_NAME 可配置化 | ✅ | - | env override |
| 1 | 前提检查 | ✅ | - | symlink/schema/contract OK |
| 1.5 | pipeline3_2.markdown | ✅ | - | 1434 行, 30 处更新 |
| 2 | Warmup (1 epoch) | ✅ | 35 min | action=0.115, state_dim=14 |
| 3 | SFT (3 epochs) | ✅ | 14h27m | action=0.012, video=0.094 |
| 4 | Evaluation | ✅ | 7h39m | SR=39.33%, 0 crashes |

### 与 Pipeline 1001 对比

| | 1001 (4 epochs) | 1002 (3 epochs) | 差异 |
|--|---|---|---|
| Warmup epochs | 2 | 1 | -1 epoch |
| SFT epochs | 4 | 3 | -1 epoch |
| SFT p_schedule | [0.5, 0.3, 0.1] | [0.3, 0.1] | 更弱 augmentation |
| Warmup 耗时 | ~70 min | 35 min | -50% |
| SFT 耗时 | ~18.7h | 14.5h | -22% |
| SFT final action loss | 0.008 | 0.012 | +50% |
| SFT final total loss | 0.282 | 0.660 | +134% |
| **Eval Overall SR** | **40.06%** | **39.33%** | **-0.73%** |
| Eval 耗时 | ~7.4h | 7.6h | ~持平 |

**结论**: 减少 1 个 warmup epoch 和 1 个 SFT epoch 节省了约 4.5 小时训练时间，成功率仅下降 0.73%。尽管 SFT loss 未充分收敛（0.660 vs 0.282），但评估性能差异不大，表明额外的 epoch 主要在降低训练 loss（可能过拟合），对泛化的贡献有限。

---

## Step 1.5: pipeline3_2.markdown 创建与验证

```
文件: b/d/libplus2/gol2/pipeline3_2.markdown
行数: 1434 行 (基于 pipeline3.markdown 1428 行)
创建方式: 基于 pipeline3.markdown 系统性修改

验证结果:
  ✓ EXPR_NAME: 30 处 4dwvlaLbPlusGolV2_1001 → 4dwvlaLbPlusGolV2_1002, 无遗漏
  ✓ Warmup: STEPS 8010→4005, SAVE_FREQ 8010→4005, SCHED_WARMUP 4005→1000, ckpt 008010→004005
  ✓ SFT: STEPS 16020→12015, SAVE_FREQ 8010→4005, SCHEDULER_DECAY 16020→12015
  ✓ SFT p_schedule: [0.5, 0.3, 0.1] → [0.3, 0.1], epoch 描述已更新
  ✓ SFT checkpoints: 004005/008010/012015 (3个), 主评估 ckpt=012015
  ✓ 所有路径中的 1001 → 1002
  ✓ Mermaid 图、附录 B、总览表均已同步更新
  ✓ eval GPU cleanup fix (exec patch + port-level cleanup) 已在 §4.3 和 §4.11 中说明
  ✓ p2v2_sft_launch.sh EXPR_NAME 可配置化已在多处说明
```

