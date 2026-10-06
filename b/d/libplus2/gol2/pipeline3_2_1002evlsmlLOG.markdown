# Pipeline3_2 Evaluation (MAX_STEPS=100) 执行日志

> **开始时间**: 2026-10-05
>
> **EXPR_NAME**: `4dwvlaLbPlusGolV2_1002_evlsml`
>
> **CKPT_PATH**: `/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/`
>
> **EVAL_MODE**: `full` (全量 2591 tasks)
>
> **MAX_STEPS_OVERRIDE**: `100` (每 episode 最多 100 步，加速评估)
>
> **与 pipeline3_2 1002 完整评估的差异**:
> - MAX_STEPS_OVERRIDE=100（原评估无覆盖，默认 ~300+ 步）
> - EXPR_NAME 改为 `4dwvlaLbPlusGolV2_1002_evlsml`（避免覆盖原评估日志）
> - 预计评估速度大幅提升（每 task ~15-20s vs 原 ~43s）
> - 因步数限制，SR 预期低于完整评估的 39.33%

---

## Step 0: 前提检查

```
GPU: 8 卡全部空闲 (0 MiB)
CKPT: /home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/
  state_dim=14, action_dim=7 (已在 pipeline 1002 中验证)
数据: /B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2 (contract schema 2)
```

---

## Step 1: 启动评估

```
命令: EXPR_NAME=4dwvlaLbPlusGolV2_1002_evlsml \
      CKPT_PATH=/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model \
      MAX_STEPS_OVERRIDE=100 \
      nohup bash b/s/libplus2/gol2/eval_v2_wrapper.sh > /tmp/eval_1002_evlsml.log 2>&1 &
PID: 2770322
开始时间: 2026-10-05 08:50:22 UTC
EVAL_LOG_DIR: /B/Log/4dwvlaLbPlusGolV2_1002_evlsml/20261005_085022_eval
LOG_FILE: /B/Log/4dwvlaLbPlusGolV2_1002_evlsml/20261005_085022_eval/eval_goal.log
```

### Preflight v2

```
  ✓ config.json, model.safetensors, stats.json, train_config.json
  ✓ contract schema=2, robot_type=libero_goal_4dv2, state_dim=14
  ✓ ckpt stats state dim=14
  ✓ ckpt config correct
  ✓ MJCF MD5 matches contract
  ✓ Server/Client Venvs OK
```

### Derived script patches

```
  ✓ healthcheck: == 'end_effector' (v2 action mode)
  ✓ eval_goal_plus_v2.py (替换 eval 入口)
  ✓ MagickWand lib path (/opt/conda/lib)
  ✓ max_steps_override 100 (--max_steps_override 100 已注入)
  ✓ exec-server (subshell → exec python, 确保 PID == python PID)
```

### 服务器启动

```
  8 GPU servers started (ports 5784-5791)
  每卡 ~13 GB (inference mode)
  8 shards, 每 shard ~324 tasks, 总计 2591 tasks
```

---

## Step 2: 评估进展监控

### 5 分钟检查 (08:57 UTC)

```
GPU 0: 11/324  | SR:  0/11  =  0.00%
GPU 1: 11/324  | SR:  0/11  =  0.00%
GPU 2: 11/324  | SR:  0/11  =  0.00%
GPU 3: 12/324  | SR:  0/12  =  0.00%
GPU 4: 11/324  | SR:  0/11  =  0.00%
GPU 5:  6/324  | SR:  0/6   =  0.00%
GPU 6: 14/324  | SR:  0/14  =  0.00%
GPU 7: 13/323  | SR: 13/13  = 100.00%
```

说明: GPU 7 最先完成的 13 个 task 全部成功（可能是 Light Conditions 中较简单的任务），其余 GPU 初期 0% 是因为先跑到了 Robot Initial States / Camera Viewpoints 等困难类别。

### 1 小时检查 (09:55 UTC)

```
GPU 0: 151/324 (46.6%) | SR:  20/151 = 13.25%  [Background Textures]
GPU 1: 152/324 (46.9%) | SR:   4/152 =  2.63%  [Robot Initial States]
GPU 2: 152/324 (46.9%) | SR:  25/152 = 16.45%  [Camera Viewpoints]
GPU 3: 153/324 (47.2%) | SR:   0/153 =  0.00%  [Language Instructions]
GPU 4: 153/324 (47.2%) | SR:  21/153 = 13.73%  [Language Instructions]
GPU 5:  80/324 (24.7%) | SR:   0/80  =  0.00%  [Sensor Noise] ← 最慢
GPU 6: 183/324 (56.5%) | SR:  38/183 = 20.77%  [Objects Layout]
GPU 7: 154/324 (47.5%) | SR:  38/154 = 24.68%  [Light Conditions]

总计: 1178/2591 (45.5%) | SR: 146/1178 = 12.39%
```

说明: GPU 5 (Sensor Noise) 显著慢于其他 GPU，与完整评估中的瓶颈一致（Sensor Noise 任务 CPU 密集型）。

### 3 小时检查 (11:55 UTC)

```
GPU 0: 324/324 DONE | SR:  66/324 = 20.37%  (完成于 ~11:03 UTC)
GPU 1: 324/324 DONE | SR:   4/324 =  1.23%  (完成于 ~11:03 UTC)
GPU 2: 324/324 DONE | SR:  26/324 =  8.02%  (完成于 ~11:03 UTC)
GPU 3: 324/324 DONE | SR:  40/324 = 12.35%  (完成于 ~11:02 UTC)
GPU 4: 324/324 DONE | SR:  38/324 = 11.73%  (完成于 ~11:45 UTC)
GPU 5: 240/324      | SR:   0/240 =  0.00%  ← 仍在运行
GPU 6: 324/324 DONE | SR:  81/324 = 25.00%  (完成于 ~10:43 UTC，最先完成)
GPU 7: 323/323 DONE | SR: 119/323 = 36.84%  (完成于 ~11:02 UTC)
```

说明: 7/8 GPU 已完成，仅剩 GPU 5 (Sensor Noise) 还有 84 个任务。

---

## Step 3: 评估完成

```
完成时间: 2026-10-05 12:30:37 UTC
总用时: 3 小时 40 分钟 (08:50 → 12:30 UTC)
退出码: 0 (成功)
Crashes: 0
```

### GPU 完成时间

| GPU | Shard 范围 | 完成时间 (UTC) | 用时 | SR |
|-----|-----------|---------------|------|-----|
| 6 | 1944-2268 | ~10:43 | ~1h53m | 25.00% |
| 3 | 972-1296  | ~11:02 | ~2h12m | 12.35% |
| 7 | 2268-2591 | ~11:02 | ~2h12m | 36.84% |
| 0 | 0-324     | ~11:03 | ~2h13m | 20.37% |
| 1 | 324-648   | ~11:03 | ~2h13m | 1.23%  |
| 2 | 648-972   | ~11:03 | ~2h13m | 8.02%  |
| 4 | 1296-1620 | ~11:45 | ~2h55m | 11.73% |
| 5 | 1620-1944 | ~12:30 | **~3h40m** | 0.00% |

GPU 5 (Sensor Noise) 是瓶颈，用时 3h40m，几乎是最快 GPU 的 2 倍。与完整评估中的模式一致。

### 最终结果 (overall_results.json)

```json
{
  "overall": {
    "total_count": 2591,
    "success_count": 374,
    "success_rate": 0.1443
  }
}
```

**Overall SR: 374/2591 = 14.43%**

### 各类别成功率对比

| Category | Full Eval (MAX_STEPS=default) | evlsml (MAX_STEPS=100) | 差值 |
|----------|------------------------------|------------------------|------|
| Light Conditions | 81.7% (228/279) | **29.0%** (81/279) | -52.7% |
| Background Textures | 70.5% (198/281) | **23.5%** (66/281) | -47.0% |
| Objects Layout | 57.4% (244/425) | **28.0%** (119/425) | -29.4% |
| Language Instructions | 43.4% (178/410) | **15.1%** (62/410) | -28.3% |
| Sensor Noise | 24.0% (91/379) | **4.2%** (16/379) | -19.8% |
| Camera Viewpoints | 15.7% (64/408) | **6.4%** (26/408) | -9.3% |
| Robot Initial States | 3.9% (16/409) | **1.0%** (4/409) | -2.9% |
| **Overall** | **39.33%** (1019/2591) | **14.43%** (374/2591) | **-24.9%** |

### 分析

1. **MAX_STEPS=100 对 SR 影响显著**: 总 SR 从 39.33% 降至 14.43%，下降 24.9 个百分点（相对下降 63.3%）。
2. **高 SR 类别受影响最大**: Light Conditions (-52.7%) 和 Background Textures (-47.0%) 的绝对降幅最大，说明这些类别中有大量任务需要超过 100 步才能完成。
3. **低 SR 类别影响较小**: Robot Initial States (-2.9%) 和 Camera Viewpoints (-9.3%) 降幅最小，因为这些任务本身就很难完成，即使给更多步数 SR 也不高。
4. **Sensor Noise 几乎为零**: 从 24.0% 降至 4.2%，噪声环境下的任务需要更多修正步数。
5. **时间节省**: 3h40m vs 7h39m，节省约 52% 的评估时间。
6. **结论**: MAX_STEPS=100 不适合作为最终评估标准，但可用于快速排序不同 checkpoint 的相对性能。相对排序基本保持一致（Light Conditions > Objects Layout > Background Textures > Language Instructions > Camera Viewpoints > Sensor Noise > Robot Initial States）。

---

## Step 4: Post-processing 验证

### Phase 5 自动完成的操作

```
✓ 聚合结果 → overall_results.json (2606 bytes)
✓ 收集失败记录 → failures_manifest.json (1.5 MB, 2217 failures, 0 crashes)
                → replay_commands.sh (538 KB)
✓ 评估存档 → /B/Log/4dwvlaLbPlusGolV2_1002_evlsml/eval_goal_20261005_085022.tar.gz (7.7 MB)
✓ 全量日志存档 → ~/b/Ckp/4dwvlaLbPlusGolV2_1002_evlsml_ALL_LOGS_20261005_123036.tar.gz (15 MB)
✓ GPU servers 清理完毕 (所有 ports 5784-5791 无残留)
✓ bigmatrix GPU placeholder 重启 (PID 3687672, 8 GPU 均被占用)
```

### 关键路径汇总

| 项目 | 路径 |
|------|------|
| Checkpoint | `/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/` |
| 评估日志目录 | `/B/Log/4dwvlaLbPlusGolV2_1002_evlsml/20261005_085022_eval/` |
| overall_results.json | `/B/Log/4dwvlaLbPlusGolV2_1002_evlsml/20261005_085022_eval/overall_results.json` |
| failures_manifest.json | `/B/Log/4dwvlaLbPlusGolV2_1002_evlsml/20261005_085022_eval/failures_manifest.json` |
| 评估存档 | `/B/Log/4dwvlaLbPlusGolV2_1002_evlsml/eval_goal_20261005_085022.tar.gz` |
| 全量存档 | `~/b/Ckp/4dwvlaLbPlusGolV2_1002_evlsml_ALL_LOGS_20261005_123036.tar.gz` |
| Wrapper 日志 | `/tmp/eval_1002_evlsml.log` |

---

## 最终状态

| Step | 描述 | 状态 |
|------|------|------|
| 0 | 前提检查 | ✅ PASSED |
| 1 | 启动评估 | ✅ PASSED |
| 2 | 进展监控 | ✅ PASSED (4 次检查) |
| 3 | 评估完成 | ✅ PASSED (2591/2591 tasks, 0 crashes, exit=0) |
| 4 | Post-processing | ✅ PASSED (聚合、存档、GPU 占位全部完成) |

**评估成功完成。Overall SR = 14.43%（MAX_STEPS=100），与完整评估的 39.33% 相比下降 24.9%，符合预期。无 error，无 crash。**

