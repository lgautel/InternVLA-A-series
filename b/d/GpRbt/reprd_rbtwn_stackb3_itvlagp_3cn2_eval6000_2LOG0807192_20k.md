# itvlaGp RoboTwin stack_bowls_three 评估 — step-020000 执行日志 (LOG0807192_20k)

> 本文档记录 2026-08-08/09 按 V2 操作手册（`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2.md`）对 **step-020000** checkpoint 执行评估的全过程。参考上次经验见 `reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG.md`、`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG2.md` 与 `reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192.md`。
>
> **与 015k 评估的差异**：本次 checkpoint 为 step **020000**（非 015000）。

---

## 评估配置

| 项 | 值 |
|----|-----|
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |
| **Checkpoint** | `/home/luogang/SRC/Robot/itvlaGp080719_2/outputs/internvla_a1_5/2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k/checkpoints/020000/pretrained_model` |
| **任务** | `stack_bowls_three` (task_idx=46) |
| **推理后端** | `standard` (3-path MoT) |
| **动作模式** | `abs` |
| **infer-horizon** | 20 |
| **dtype** | bfloat16 |
| **每配置 episode** | 100 |
| **GPU** | demo_clean → GPU0, demo_randomized → GPU1 |
| **输出目录** | `outputs/robotwin/itvlaGp_p2_020k_0807192/` |

---

## 时间线 / 操作日志

| 时间 (UTC) | 操作 | 结果 |
|------|------|------|
| 2026-08-08 22:14 | 创建本日志，开始执行评估计划 | OK |
| 2026-08-08 22:15 | 15 项预检检查表 | **15/15 PASS** |
| 2026-08-08 22:15 | 静态测试 v4 | **16/16 PASS** |
| 2026-08-08 22:15–22:19 | 2-episode 冒烟测试 (demo_clean, GPU0) | **PASS** — 2/2 完成, exit 0, 1/2 success, ~4.3 min |
| 2026-08-08 22:20 | 双 GPU 并行启动正式评估 (各 100 ep) | OK — clean PID 2061126 (GPU0), randomized PID 2061200 (GPU1) |
| 2026-08-08 23:49 | 进度快照 #1 (+89 min) | clean 50/100 (54.0%), randomized 38/100 (10.5%), 零致命错误 |
| 2026-08-09 01:15 | **demo_clean 完成** | **57/100 = 57.0%**, exit code 0, 耗时 ~2.93 h |
| 2026-08-09 02:13 | **demo_randomized 完成** | **11/100 = 11.0%**, exit code 0, 耗时 ~3.89 h |

---

## 问题记录

### 预期非致命日志：Expert rollout seed 验证失败

| 项 | 内容 |
|----|------|
| **发现时机** | 正式评估期间 |
| **症状** | `AssertionError: target_pose cannot be None for move action` |
| **根因** | 某些 seed 下 CuRobo 运动规划器无法为碗抓取找到有效预抓取姿态，expert rollout 失败 |
| **次数** | demo_clean 3 次，demo_randomized 1 次 |
| **影响** | 该 seed 被跳过，不计入 100 个有效 episode；**评估流程正常继续** |
| **修复** | 无需修复（预期行为，与 LOG1/LOG2/LOG0807192 一致） |
| **验证** | 双配置各 100/100 episode 完成，exit code 0 |

**本次评估期间无需要修复的运行时错误**（零 CUDA 错误、零 AttributeError、零磁盘满、零进程意外终止）。

---

## 文件增删改记录

| 文件 | 操作 | 原因 |
|------|------|------|
| `b/d/reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192_20k.md` | 新建 | 本次 step-020000 评估执行日志 |
| `outputs/robotwin/itvlaGp_p2_020k_0807192/` | 新建 | 评估视频输出 |
| `outputs/robotwin/itvlaGp_p2_020k_0807192/results_robotwin.csv` | 新建 | 结果统计 CSV |
| `outputs/logs/smoke_itvlaGp_p2_020k_0807192.log` | 新建 | 冒烟测试日志 |
| `outputs/logs/eval_itvlaGp_p2_020k_0807192_demo_clean.log` | 新建 | demo_clean 完整日志 |
| `outputs/logs/eval_itvlaGp_p2_020k_0807192_demo_randomized.log` | 新建 | demo_randomized 完整日志 |
| 代码文件 | 无修改 | 4 项修复已在首次评估时完成 |

---

## 评估结果

| 配置 | 成功 | 失败 | 总计 | **Success Rate** |
|------|------|------|------|-----------------|
| **demo_clean** | 57 | 43 | 100 | **57.0%** |
| **demo_randomized** | 11 | 89 | 100 | **11.0%** |

### 与历史结果对比

| 模型 | Checkpoint | demo_clean | demo_randomized |
|------|-----------|------------|-----------------|
| InternVLA-A1.5 base | step 10000 | 71.0% | 54.0% |
| itvlaGp p2 (LOG1/LOG2) | step 10000 | 63–64% | 16–22% |
| itvlaGp p2 (LOG0807192) | step 15000 | 65.0% | 11.0% |
| **itvlaGp p2 (本次 LOG0807192_20k)** | **step 20000** | **57.0%** | **11.0%** |

**对比分析**：
- demo_clean：57% 略低于 015k 的 65%，与 010k 的 63–64% 接近；step 增加未带来 Easy 配置提升
- demo_randomized：11% 与 015k 完全相同；仍显著低于 base 版 54%
- 域随机化泛化仍是主要瓶颈

### 评估时间

| 配置 | 开始时间 (UTC) | 结束时间 (UTC) | 耗时 | GPU |
|------|---------------|---------------|------|-----|
| demo_clean | 2026-08-08 22:20 | 2026-08-09 01:15 | ~2.93 h | GPU 0 |
| demo_randomized | 2026-08-08 22:20 | 2026-08-09 02:13 | ~3.89 h | GPU 1 |

---

## 评估命令

**冒烟测试 (GPU 0)**:
```bash
cd /home/luogang/SRC/Robot/itvlaGp/third_party/RoboTwin
CUDA_VISIBLE_DEVICES=0 python -u evaluation/RoboTwin/inference.py \
  --ckpt-path "/home/luogang/SRC/Robot/itvlaGp080719_2/outputs/internvla_a1_5/2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k/checkpoints/020000/pretrained_model" \
  --video-dir "outputs/robotwin/itvlaGp_p2_020k_0807192/smoke/demo_clean/stack_bowls_three" \
  --task-config demo_clean --task-idx 46 --action-mode abs \
  --infer-horizon 20 --inference-backend standard --num-episodes 2 --dtype bfloat16
```

**demo_clean (GPU 0, 100 ep)** / **demo_randomized (GPU 1, 100 ep)**：参数同上，换 `--task-config` 和 `--video-dir`。

---

## 输出路径

| 路径 | 内容 |
|------|------|
| `outputs/robotwin/itvlaGp_p2_020k_0807192/robotwin/demo_clean/stack_bowls_three/` | 100 个 .mp4 |
| `outputs/robotwin/itvlaGp_p2_020k_0807192/robotwin/demo_randomized/stack_bowls_three/` | 100 个 .mp4 |
| `outputs/robotwin/itvlaGp_p2_020k_0807192/results_robotwin.csv` | CSV 汇总 |
| `outputs/logs/eval_itvlaGp_p2_020k_0807192_demo_clean.log` | demo_clean 完整日志 |
| `outputs/logs/eval_itvlaGp_p2_020k_0807192_demo_randomized.log` | demo_randomized 完整日志 |
| `outputs/logs/smoke_itvlaGp_p2_020k_0807192.log` | 冒烟测试日志 |

---

## 总结

1. **预检 + 冒烟**：15/15 PASS + 16/16 静态测试 + 2/2 冒烟 PASS
2. **正式评估**：demo_clean **57%**, demo_randomized **11%**，双配置各 100 episode 全部完成
3. **零运行时致命错误**：4 次预期 expert seed 失败，无 CUDA/AttributeError/磁盘满
4. **step 020000 vs 015000**：demo_clean 略降（65%→57%），demo_randomized 持平（11%）

**评估进度：已完成。**
