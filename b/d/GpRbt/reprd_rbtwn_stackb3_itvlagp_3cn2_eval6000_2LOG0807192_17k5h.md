# itvlaGp RoboTwin stack_bowls_three 评估 — step-017500 执行日志 (LOG0807192_17k5h)

> 本文档记录 2026-08-10 按 V2 操作手册（`reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2.md`）对 **step-017500** checkpoint 执行评估的全过程。参考上次经验见历次 LOG 文件。
>
> **说明**：本次使用 `itvlaGp080719_2` 训练 run 的 step **017500** 权重，便于与 010k/012.5k/015k/020k 横向对比。

---

## 评估配置

| 项 | 值 |
|----|-----|
| **代码库** | `/home/luogang/SRC/Robot/itvlaGp` |
| **Checkpoint** | `/home/luogang/SRC/Robot/itvlaGp080719_2/outputs/internvla_a1_5/2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k/checkpoints/017500/pretrained_model` |
| **任务** | `stack_bowls_three` (task_idx=46) |
| **推理后端** | `standard` (3-path MoT) |
| **动作模式** | `abs` |
| **infer-horizon** | 20 |
| **dtype** | bfloat16 |
| **每配置 episode** | 100 |
| **GPU** | demo_clean → GPU0, demo_randomized → GPU1 |
| **输出目录** | `outputs/robotwin/itvlaGp_p2_017k5_0807192/` |

---

## 时间线 / 操作日志

| 时间 (UTC) | 操作 | 结果 |
|------|------|------|
| 2026-08-10 02:04 | 创建本日志，开始执行评估计划 | 进行中 |

---

## 问题记录

（暂无）

---

## 文件增删改记录

| 文件 | 操作 | 原因 |
|------|------|------|
| `b/d/reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2LOG0807192_17k5h.md` | 新建 | 本次 step-017500 评估执行日志 |

---

## 评估进度（实时更新）

**进行中。**
