# itvlaGp RoboTwin stack_bowls_three 评估 — V2 重跑执行日志 (LOG2)

> 本文档记录 2026-08-07 按 V2 操作手册（`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2.md`）重跑评估的全过程。参考上次经验见 `reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md`。
>
> **输出路径**：`outputs/robotwin/itvlaGp_p2_010k_rerun2/`（保留旧结果 `itvlaGp_p2_010k`）

---

## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
| 2026-08-07 08:21 | 开始执行重跑评估计划 | OK |
| 2026-08-07 08:22 | 15 项预检检查表 | 14/15 PASS；[13] config.json 中 `action_loss_only=False`（训练配置），`inference.py` L270 加载时强制设为 `True` |
| 2026-08-07 08:22 | CuRobo kinematics 运行时验证 | OK |
| 2026-08-07 08:22 | 2-episode 冒烟测试 (demo_clean, GPU0) | **PASS** — 2/2 成功, exit code 0, ~3 min |
| 2026-08-07 08:26 | 双 GPU 并行启动 (nohup) | **FAIL** — 进程随 shell 退出终止 (Problem #1) |
| 2026-08-07 08:29 | 修复后重启：持久化后台 shell (PID 3548135 clean, 3548214 randomized) | OK |
| 2026-08-07 08:31 | 进度快照 #1 | clean 1/100, randomized 0/100 |
| 2026-08-07 08:40 | 进度快照 #2 | clean 7/100 (5S/2F), randomized 4/100 (0S/4F) |
| 2026-08-07 08:50 | 进度快照 #3 | clean 11/100 (5S/6F), randomized 9/100 (1S/8F) |
| 2026-08-07 08:55 | **磁盘空间不足** | 根分区 100% 满，`no space left on device` (Problem #2) |
| 2026-08-07 08:56 | `pip cache purge` 释放空间 | 46G 可用 (96% used)，评估进程仍存活 |
| 2026-08-07 10:40 | 进度快照 #4 | clean 84/100 (54S/30F, 64.2%), randomized 63/100 (15S/48F, 23.8%) |
| 2026-08-07 11:19 | **demo_clean 完成** | **63/100 = 63.0%**, exit code 0 |
| 2026-08-07 11:54 | **demo_randomized 完成** | **22/100 = 22.0%**, exit code 0 |

---

## 问题记录

### Problem #1: nohup 后台进程随 Shell 退出被终止

| 项 | 内容 |
|----|------|
| **发现时机** | 首次双 GPU 并行启动后 90s 检查 |
| **症状** | PID 3546864/3546865 已不存在；日志仅 3 行，无 mp4 |
| **根因** | Cursor Shell 命令结束后子 shell 退出，`nohup ... &` 后台 job 被 SIGHUP 终止 |
| **修复** | 改用 Shell 工具 `block_until_ms=0` 持久化后台运行 |
| **验证** | 重启后 PID 3548135/3548214 存活，日志持续增长 |

### Problem #2: 磁盘空间不足

| 项 | 内容 |
|----|------|
| **发现时机** | 08:55 进度监控时 |
| **症状** | `write failed: no space left on device`；根分区 969G/969G (100%) |
| **根因** | `/home/luogang/.cache/pip` 占用 ~21G；系统盘总体耗尽 |
| **影响** | LOG2 文件写入失败被清空；评估进程仍存活（13 clean + 11 randomized mp4 已写入） |
| **修复** | `pip cache purge` → 释放 46G |
| **验证** | `df -h /` 显示 46G free；pgrep 确认 2 个 inference.py 仍在运行 |

---

## 评估配置

| 项 | 值 |
|----|-----|
| **Checkpoint** | `outputs/internvla_a1_5/p2/checkpoints/010000/pretrained_model` |
| **任务** | `stack_bowls_three` (task_idx=46) |
| **推理后端** | `standard` (3-path MoT) |
| **动作模式** | `abs` |
| **infer-horizon** | 20 |
| **dtype** | bfloat16 |
| **每配置 episode** | 100 |
| **GPU** | demo_clean → GPU0, demo_randomized → GPU1 |

---

## 评估结果

| 配置 | 成功 | 失败 | 总计 | **Success Rate** | 上次 (LOG1) |
|------|------|------|------|-----------------|-------------|
| **demo_clean** | 63 | 37 | 100 | **63.0%** | 64.0% |
| **demo_randomized** | 22 | 78 | 100 | **22.0%** | 16.0% |

**对比分析**：
- demo_clean 与上次基本一致（63% vs 64%），差异在 1 个 episode 以内，属正常随机波动
- demo_randomized 从 16% 提升至 22%（+6pp），可能因 seed 采样差异或 CuRobo 规划成功率波动
- 零 CUDA 错误、零 AttributeError、零模型推理 Traceback

---

## 输出路径

| 路径 | 内容 |
|------|------|
| `outputs/robotwin/itvlaGp_p2_010k_rerun2/robotwin/demo_clean/stack_bowls_three/` | 100 个 .mp4 |
| `outputs/robotwin/itvlaGp_p2_010k_rerun2/robotwin/demo_randomized/stack_bowls_three/` | 100 个 .mp4 |
| `outputs/logs/eval_itvlaGp_p2_010k_rerun2_demo_clean.log` | demo_clean 完整日志 |
| `outputs/logs/eval_itvlaGp_p2_010k_rerun2_demo_randomized.log` | demo_randomized 完整日志 |
| `outputs/logs/smoke_itvlaGp_p2_010k_rerun2.log` | 冒烟测试日志 |

---

## 评估命令

**demo_clean (GPU 0)**:
```bash
cd /home/luogang/SRC/Robot/itvlaGp/third_party/RoboTwin
CUDA_VISIBLE_DEVICES=0 python -u evaluation/RoboTwin/inference.py \
  --ckpt-path "outputs/internvla_a1_5/p2/checkpoints/010000/pretrained_model" \
  --video-dir "outputs/robotwin/itvlaGp_p2_010k_rerun2/robotwin/demo_clean/stack_bowls_three" \
  --task-config demo_clean --task-idx 46 --action-mode abs \
  --infer-horizon 20 --inference-backend standard --num-episodes 100 --dtype bfloat16
```

**demo_randomized (GPU 1)**:
```bash
# 同上, CUDA_VISIBLE_DEVICES=1, --task-config demo_randomized
```

---

## 文件增删改记录

| 文件 | 操作 | 原因 |
|------|------|------|
| `b/d/reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md` | 新建 | 本次重跑执行日志 |
| 代码文件 | 无修改 | 4 项修复已在首次评估时完成 |

---

## 总结

1. **预检 + 冒烟**：14/15 PASS + 2/2 冒烟 PASS
2. **2 个问题已修复**：nohup 进程终止 (Problem #1)、磁盘满 (Problem #2)
3. **正式评估**：demo_clean 63%, demo_randomized 22%，双配置各 100 episode 全部完成
4. **零运行时致命错误**

---

## 评估进度（实时更新）

**已完成。**
