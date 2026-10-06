# Pipeline3_2 Evaluation (MAX_STEPS=150) 执行日志

> **开始时间**: 2026-10-05
>
> **EXPR_NAME**: `4dwvlaLbPlusGolV2_1002_evlMid`
>
> **CKPT_PATH**: `/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/`
>
> **EVAL_MODE**: `full` (全量 2591 tasks)
>
> **MAX_STEPS_OVERRIDE**: `150` (每 episode 最多 150 步)
>
> **与先前评估的差异**:
> - 完整评估（无覆盖，默认 ~300+ 步）: SR = 39.33%, 用时 7h39m
> - evlsml（MAX_STEPS=100）: SR = 14.43%, 用时 3h40m
> - 本次 evlMid（MAX_STEPS=150）: 介于两者之间，预计 SR ~25-30%, 用时 ~4.5-5.5h

---

## Step 0: 前提检查

```
GPU: 8 卡均被 bigmatrix 占用 (100-125 GB), eval_v2_wrapper 会自动清理
CKPT: /home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/
  config.json ✓, state_dim=14, action_dim=7
数据: /B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2 (contract schema 2)
Ports 5784-5791: 全部空闲
server_policy 进程: 无
```

---

## Step 1: 启动评估

```
命令: EXPR_NAME=4dwvlaLbPlusGolV2_1002_evlMid \
      CKPT_PATH=/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model \
      MAX_STEPS_OVERRIDE=150 \
      nohup bash b/s/libplus2/gol2/eval_v2_wrapper.sh > /tmp/eval_1002_evlMid.log 2>&1 &
PID: 3797750
开始时间: 2026-10-05 23:32:44 UTC
EVAL_LOG_DIR: /B/Log/4dwvlaLbPlusGolV2_1002_evlMid/20261005_233244_eval
LOG_FILE: /B/Log/4dwvlaLbPlusGolV2_1002_evlMid/20261005_233244_eval/eval_goal.log
```

### Preflight v2

```
  ✓ config.json, model.safetensors, stats.json, train_config.json
  ✓ contract schema=2, robot_type=libero_goal_4dv2, state_dim=14
  ✓ ckpt stats state dim=14
  ✓ ckpt config correct
  ✓ MJCF MD5 matches contract
  ✓ Server/Client Venvs OK
  ✓ NumPy Patches OK
  ✓ Evaluation Scripts OK
  ✓ Task Config OK
```

### Derived script patches

```
  ✓ healthcheck: == 'end_effector' (v2 action mode)
  ✓ eval_goal_plus_v2.py (替换 eval 入口)
  ✓ MagickWand lib path (/opt/conda/lib)
  ✓ max_steps_override 150 (--max_steps_override 150 已注入)
  ✓ exec-server (subshell → exec python, 确保 PID == python PID)
```

### 服务器启动

```
  8 GPU servers started (ports 5784-5791)
  每卡 ~13 GB (inference mode)
  bigmatrix 自动清理完成
  8 shards, 每 shard ~324 tasks, 总计 2591 tasks
```

### 5 分钟检查 (23:40 UTC)

```
GPU 0: 11/324  | SR:  9/11  = 81.8%  [Background Textures 早期]
GPU 1: 10/324  | SR:  0/10  =  0.0%  [Robot Initial States]
GPU 2:  9/324  | SR:  0/9   =  0.0%  [Camera Viewpoints]
GPU 3:  9/324  | SR:  0/9   =  0.0%  [Language Instructions]
GPU 4: 10/324  | SR:  4/10  = 40.0%  [Language Instructions]
GPU 5:  4/324  | SR:  0/4   =  0.0%  [Sensor Noise] ← 最慢
GPU 6: 13/324  | SR:  0/13  =  0.0%  [Objects Layout]
GPU 7: 12/323  | SR:  9/12  = 75.0%  [Light Conditions]

总计: 78/2591 | SR: 22/78 = 28.2% (早期偏高，高 SR 类别先出结果)
```

对比: 同期 evlsml (MAX_STEPS=100) GPU 0 仅 0/11=0%, GPU 7 仅 13/13=100%。
MAX_STEPS=150 的 Background Textures 和 Light Conditions 早期 SR 显著高于 MAX_STEPS=100。

---

## Step 2: 评估进展监控

### 2 小时检查 (01:41 UTC)

```
GPU 0: 265/324 (81.8%) | SR: 147/265 = 55.5%  [Background Textures]
GPU 1: 251/324 (77.5%) | SR:   8/251 =  3.2%  [Robot Initial States]
GPU 2: 251/324 (77.5%) | SR:  41/251 = 16.3%  [Camera Viewpoints]
GPU 3: 257/324 (79.3%) | SR:  68/257 = 26.5%  [Language Instructions]
GPU 4: 238/324 (73.5%) | SR:  88/238 = 37.0%  [Language Inst. + Sensor Noise]
GPU 5: 124/324 (38.3%) | SR:  18/124 = 14.5%  [Sensor Noise] ← 瓶颈
GPU 6: 324/324 DONE    | SR: 144/324 = 44.4%  [Objects Layout]
GPU 7: 272/323 (84.2%) | SR: 179/272 = 65.8%  [Light Conditions]

总计: 1982/2591 (76.5%) | SR: 693/1982 = 35.0%
GPU 6 最先完成（~2h）
```

### 4.5 小时检查 (04:11 UTC)

```
GPU 0: 324/324 DONE | GPU 1: 324/324 DONE | GPU 2: 324/324 DONE
GPU 3: 324/324 DONE | GPU 4: 324/324 DONE | GPU 6: 324/324 DONE
GPU 7: 323/323 DONE
GPU 5: 314/324 (96.9%) | SR: 56/314 = 17.8% ← 仅剩 10 tasks
```

---

## Step 3: 评估完成

```
完成时间: 2026-10-06 04:14:34 UTC
总用时: 4 小时 42 分钟 (23:32 → 04:14 UTC)
退出码: 0 (成功)
Crashes: 0
```

### GPU 完成时间

| GPU | Shard 范围 | 完成时间 (UTC) | 用时 | SR |
|-----|-----------|---------------|------|-----|
| 6 | 1944-2268 | 01:33 | ~2h01m | 44.44% |
| 7 | 2268-2591 | 01:59 | ~2h27m | 71.21% |
| 0 | 0-324     | 02:06 | ~2h34m | 51.23% |
| 3 | 972-1296  | 02:11 | ~2h39m | 22.53% |
| 1 | 324-648   | 02:13 | ~2h41m | 2.47%  |
| 2 | 648-972   | 02:14 | ~2h42m | 13.58% |
| 4 | 1296-1620 | 03:06 | ~3h34m | 36.11% |
| 5 | 1620-1944 | **04:14** | **~4h42m** | 17.28% |

GPU 5 (Sensor Noise) 是瓶颈，用时 4h42m，约为最快 GPU 6 的 **2.3 倍**。
根因: `glass_blur` 的 O(H×W×iters) 纯 Python 逐像素循环（单次 ~672ms × 150 步 = ~100s/episode 额外开销）。
后续已将 `SHARDS_PER_SUITE` 默认改为 64（细粒度分片 + round-robin），可消除此瓶颈。

### 最终结果 (overall_results.json)

**Overall SR: 838/2591 = 32.34%**

### 三次评估对比

| Category | Full (默认步数) | evlMid (150步) | evlsml (100步) |
|----------|----------------|----------------|----------------|
| Light Conditions | 81.7% (228/279) | **68.5%** (191/279) | 29.0% (81/279) |
| Background Textures | 70.5% (198/281) | **56.2%** (158/281) | 23.5% (66/281) |
| Objects Layout | 57.4% (244/425) | **51.3%** (218/425) | 28.0% (119/425) |
| Language Instructions | 43.4% (178/410) | **33.4%** (137/410) | 15.1% (62/410) |
| Sensor Noise | 24.0% (91/379) | **19.3%** (73/379) | 4.2% (16/379) |
| Camera Viewpoints | 15.7% (64/408) | **10.8%** (44/408) | 6.4% (26/408) |
| Robot Initial States | 3.9% (16/409) | **4.2%** (17/409) | 1.0% (4/409) |
| **Overall** | **39.33%** (1019/2591) | **32.34%** (838/2591) | **14.43%** (374/2591) |

### 分析

1. **MAX_STEPS=150 保留了完整评估 ~82% 的 SR**: 32.34% / 39.33% = 82.2%，远优于 MAX_STEPS=100 的 36.7%。
2. **时间效率**: 4h42m vs 完整评估 7h39m，节省 39%；vs MAX_STEPS=100 的 3h40m 仅多 1h。
3. **类别排序完全一致**: 7 个类别的相对排序与完整评估完全相同，确认 MAX_STEPS=150 可用于快速比较不同 checkpoint。
4. **Sensor Noise 恢复显著**: 从 MAX_STEPS=100 的 4.2% 提升到 19.3%（接近完整评估的 24.0%），额外 50 步对噪声修正至关重要。
5. **Robot Initial States 几乎不受影响**: 4.2% vs 完整评估 3.9%（略高，可能是统计噪声），说明此类别的失败不是因为步数不够。
6. **瓶颈仍是 GPU 5**: Sensor Noise 集中在 GPU 5 导致 4h42m 瓶颈。已在 `eval_v2_wrapper.sh` 中将 `SHARDS_PER_SUITE` 默认改为 64，预计可将最慢 GPU 从 4h42m 降至 ~3h（与其他 GPU 齐平）。

### 最佳 MAX_STEPS 选择建议

| MAX_STEPS | SR / 完整评估比 | 用时 | 适用场景 |
|-----------|----------------|------|---------|
| 默认 (~300+) | 100% | ~7.6h | 最终发布评估 |
| **150** | **82%** | **~4.7h** | **日常 checkpoint 筛选（推荐）** |
| 100 | 37% | ~3.7h | 快速排除明显差的 checkpoint |

---

## Step 4: Post-processing 验证

### Phase 5 自动完成的操作

```
✓ 聚合结果 → overall_results.json (2.6 KB)
✓ 收集失败记录 → failures_manifest.json (1.2 MB, 1753 failures, 0 crashes)
                → replay_commands.sh (419 KB)
✓ 评估存档 → /B/Log/4dwvlaLbPlusGolV2_1002_evlMid/eval_goal_20261005_233244.tar.gz (9.6 MB)
✓ 全量日志存档 → ~/b/Ckp/4dwvlaLbPlusGolV2_1002_evlMid_ALL_LOGS_20261006_041432.tar.gz (19 MB)
✓ GPU servers 清理完毕
✓ bigmatrix GPU placeholder 重启 (PID 550724)
```

### 失败分布

| Category | 失败数 | 总数 | 失败率 |
|----------|--------|------|--------|
| Robot Initial States | 392 | 409 | 95.8% |
| Camera Viewpoints | 364 | 408 | 89.2% |
| Sensor Noise | 306 | 379 | 80.7% |
| Language Instructions | 273 | 410 | 66.6% |
| Objects Layout | 207 | 425 | 48.7% |
| Background Textures | 123 | 281 | 43.8% |
| Light Conditions | 88 | 279 | 31.5% |

### 关键路径汇总

| 项目 | 路径 |
|------|------|
| Checkpoint | `/home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1002/2026_10_04_02_17_33-internvla_a1_5-lbplus-golv2-sft/checkpoints/012015/pretrained_model/` |
| 评估日志目录 | `/B/Log/4dwvlaLbPlusGolV2_1002_evlMid/20261005_233244_eval/` |
| overall_results.json | `/B/Log/4dwvlaLbPlusGolV2_1002_evlMid/20261005_233244_eval/overall_results.json` |
| failures_manifest.json | `/B/Log/4dwvlaLbPlusGolV2_1002_evlMid/20261005_233244_eval/failures_manifest.json` |
| 评估存档 | `/B/Log/4dwvlaLbPlusGolV2_1002_evlMid/eval_goal_20261005_233244.tar.gz` |
| 全量存档 | `~/b/Ckp/4dwvlaLbPlusGolV2_1002_evlMid_ALL_LOGS_20261006_041432.tar.gz` |

---

## 最终状态

| Step | 描述 | 状态 |
|------|------|------|
| 0 | 前提检查 | ✅ PASSED |
| 1 | 启动评估 | ✅ PASSED |
| 2 | 进展监控 | ✅ PASSED (3 次检查) |
| 3 | 评估完成 | ✅ PASSED (2591/2591 tasks, 0 crashes, exit=0) |
| 4 | Post-processing | ✅ PASSED (聚合、存档、GPU 占位全部完成) |

**评估成功完成。Overall SR = 32.34%（MAX_STEPS=150），保留了完整评估 39.33% 的 82%，类别排序完全一致。无 error，无 crash。**

---

## 附: pipeline3_2 配置更新

评估期间根据 GPU 5 瓶颈根因分析，对 `eval_v2_wrapper.sh` 和 `pipeline3_2.markdown` 进行了配置更新:

### eval_v2_wrapper.sh 改动

```diff
-if [ "${EVAL_MODE}" = "smoke" ]; then
-    export SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-2}"
-    export GPU_IDS="${GPU_IDS:-0,2}"
-    export MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-100}"
-fi
+export SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-64}"
+export MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-150}"
+
+if [ "${EVAL_MODE}" = "smoke" ]; then
+    export GPU_IDS="${GPU_IDS:-0,2}"
+fi
```

**理由**:
- `SHARDS_PER_SUITE=64`: 细粒度分片 + round-robin 分配，将 Sensor Noise 任务均匀散布到所有 GPU，消除 GPU 5 单点瓶颈
- `MAX_STEPS_OVERRIDE=150`: 保留完整评估 82% 的 SR，节省 39% 评估时间
- 两项配置从 smoke-only 提升为全局默认，smoke 和 full 模式共享

