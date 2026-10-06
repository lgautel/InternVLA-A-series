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

