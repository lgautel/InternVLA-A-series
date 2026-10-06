# Pipeline3 v2 执行日志

> **开始时间**: 2026-10-02
>
> **Pipeline 版本**: pipeline3.markdown
>
> **EXPR_NAME**: `4dwvlaLbPlusGolV2_1001`
>
> **数据**: `/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2/` (14D state, 512604 frames)

---

## Step 0: GPU 清理

```
操作: kill 所有占用 GPU 的进程 (8×python, PIDs 2375524/2375525/2375526/2375527/2375529/2375531/2375533/2375534)
结果: 全部 8 GPU 已清空 (0 MiB each, 每卡总显存 143771 MiB ≈ 140 GB)
```

---

## Step 1: 前提检查 (§0.3)

### 1.1 数据 symlink

```
操作: ln -sfn /B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2 /B/VENV/hf_home/lerobot/libero_plus_goal_lrb3_4Dv2
结果: SYMLINK OK
```

### 1.2 Schema 检查

```
文件: src/lerobot/dataset_schemas/configs/libero_goal_4dv2.yaml
结果: SCHEMA OK
数据身份: robot_type=libero_goal_4dv2, state_dim=[14], frames=512604
```

---

## Step 2: Pipeline 验收 (§7.2 步骤 2)

```
命令: bash b/s/libplus2/gol2/accept_pipeline_v2.sh
结果: 26 passed, 0 failed, 0 skipped — ACCEPTANCE PASSED
pytest: 25 passed in 0.16s
所有检查项:
  A1 脚本存在: 6/6 OK
  A2 Bash 语法: 5/5 OK
  A3 v2 身份: 5/5 OK
  A4 Preflight 内容: 2/2 OK
  A5 pytest: 25/25 PASS
  A6 v2 数据: OK
  A7 gol 依赖: 6/6 OK
```

---

## Step 3: Warmup Smoke Test (§7.2 步骤 3)

```
命令: SMOKE=1 bash b/s/libplus2/gol2/p1v2_warmup_launch.sh
时间: 2026-10-02 08:39~08:42
模式: SMOKE (1 GPU, batch_size=2, steps=10)
输出目录: /home/a26113/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_08_39_07-internvla_a1_5-lbplus-gol-warmup-smoke
```

### 验证结果

| 检查项 | 期望 | 实际 | 状态 |
|--------|------|------|------|
| exit code | 0 | 0 | OK |
| loss_action finite | 是 | 0.215→0.338 (10 steps) | OK |
| loss_kpt_cur finite | 是 | 1.009→0.259 | OK |
| loss_kpt_fut finite | 是 | 1.136→0.336 | OK |
| loss_video | 0.000 (action_loss_only=true) | 0.000 | OK |
| loss_fast | 0.000 | 0.000 | OK |
| WAN params | 0 | 0 | OK |
| Knowledge insulation | True | True | OK |
| Inference backend | standard | standard | OK |
| Learnable tokens | 50 | 50 | OK |
| Trainable params | ~927M | 927M | OK |
| video_decode_error | 0 | 0 | OK |
| using_zeros | 0 | 0 | OK |
| Checkpoint saved | 是 | checkpoints/000010 | OK |
| 数据集 | 512604 frames, 4243 episodes | 512604 frames, 4243 episodes | OK |

### 备注

- Missing keys WARNING (keypoint_expert, track_encoder): 预期行为，base model 不含关键点专家权重，warmup 阶段随机初始化
- Unexpected keys WARNING (wan_grid_sizes, learnable_to_wan_proj): 预期行为，action_loss_only=true 不加载 WAN 相关组件
- 10 steps 全部完成，loss 下降趋势正常

**结论**: Warmup Smoke Test **PASSED**

---

## Step 4: Production Warmup Training (§7.2 步骤 4)

```
命令: bash b/s/libplus2/gol2/p1v2_warmup_launch.sh
时间: 2026-10-02 08:45~10:00 (约 1h15m)
模式: PRODUCTION (8 GPU, batch_size=16, EBS=128)
步数: 8010
日志: /B/Log/4dwvlaLbPlusGolV2_1001/2026_10_02_08_45_23/train.log
输出: ~/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_08_45_23-internvla_a1_5-lbplus-gol-warmup
```

### 训练过程

| Step | loss | loss_action | loss_kpt_cur | loss_kpt_fut | lr | iters/s |
|------|------|-------------|-------------|-------------|------|---------|
| 50 | 147.793 | 0.262 | 1.017 | 1.143 | 3.3e-07 | 1.30 |
| 400 | 19.124 | 0.176 | 0.095 | 0.149 | 4.7e-06 | 1.96 |
| 1600 | 6.772 | 0.129 | 0.031 | 0.052 | 2.0e-05 | 2.01 |
| 4000 | 5.922 | 0.115 | 0.036 | 0.045 | 4.9e-05 | 1.99 |
| 6300 | 4.814 | 0.109 | 0.033 | 0.036 | 9.7e-06 | 1.94 |
| 8010 | 4.651 | 0.108 | 0.032 | 0.034 | 5.0e-06 | 2.05 |

### 检验结果

| 检查项 | 期望 | 实际 | 状态 |
|--------|------|------|------|
| exit code | 0 | 0 | OK |
| End of training | 是 | 是 | OK |
| video_decode_error | 0 | 0 | OK |
| using_zeros | 0 | 0 | OK |
| Checkpoint saved | 008010 | 008010 | OK |
| state_dim (stats.json) | 14 | 14 | OK |
| action_dim (stats.json) | 7 | 7 | OK |
| robot_type key | libero_goal_4dv2 | libero_goal_4dv2 | OK |
| Epoch | ~2.0 | 2.00 | OK |
| Knowledge insulation | True | True | OK |
| loss趋势 | 下降 | 147.8→4.7 | OK |

### Checkpoint 路径

```
WARMUP_CKPT=~/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_08_45_23-internvla_a1_5-lbplus-gol-warmup/checkpoints/008010/pretrained_model
```

**结论**: Production Warmup Training **PASSED**

---

## Step 5: SFT Smoke Test (§7.2 步骤 5)

### Bug Fix: p2v2_sft_launch.sh checkpoint guard

```
文件: b/s/libplus2/gol2/p2v2_sft_launch.sh (line 30)
原因: stats.json 结构嵌套在 robot_type key 下 (如 {"libero_goal_4dv2": {"observation.state": ...}})
      guard 代码直接访问 json['observation.state'] 导致 KeyError，fallback 返回 dim=0
修复: 添加 robot_type key 嵌套检测，先检查 'observation.state' 是否在顶层，否则取 next(iter(s.values()))
```

### 验证结果

```
命令: PRETRAINED_CKPT=<warmup_008010> SMOKE=1 bash b/s/libplus2/gol2/p2v2_sft_launch.sh
时间: 2026-10-02 10:03~10:06
模式: SMOKE (1 GPU, batch_size=2, steps=10, action_loss_only=true)
```

| 检查项 | 期望 | 实际 | 状态 |
|--------|------|------|------|
| exit code | 0 | 0 | OK |
| loss_action finite | 是 | 0.110→0.063 | OK |
| loss_kpt_cur finite | 是 | 0.002→0.018 | OK |
| loss_kpt_fut finite | 是 | 0.014→0.018 | OK |
| loss_video | 0.000 (smoke action_loss_only) | 0.000 | OK |
| Trainable params | ~3B (full model) | 3B | OK |
| Knowledge insulation | False | False | OK |
| Gradient checkpointing | True | True (log confirmed) | OK |
| Pretrained weights | loaded from warmup | "Loading weights from local directory" | OK |
| freeze_learnable_tokens | false (per §3.4) | (confirmed in script) | OK |
| video_decode_error | 0 | 0 | OK |
| using_zeros | 0 | 0 | OK |
| Checkpoint saved | 是 | checkpoints/000010 | OK |

**结论**: SFT Smoke Test **PASSED**

---

## Step 6: Production SFT Training (§7.2 步骤 6)

```
命令: PRETRAINED_CKPT=~/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_08_45_23-internvla_a1_5-lbplus-gol-warmup/checkpoints/008010/pretrained_model bash b/s/libplus2/gol2/p2v2_sft_launch.sh
时间: 2026-10-02 10:07 ~ 2026-10-03 04:56 (约 18h49m)
模式: PRODUCTION (8 GPU, batch_size=16, EBS=128)
步数: 16020
日志: /B/Log/4dwvlaLbPlusGolV2_1001/2026_10_02_10_07_07/train.log
输出: ~/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_10_07_07-internvla_a1_5-lbplus-golv2-sft
日志归档: ~/b/Ckp/4dwvlaLbPlusGolV2_1001_sft_LOG_20261003_045612.tar
```

### 训练过程

| Step | loss | loss_action | loss_vqa | loss_video | loss_fast | loss_kpt_cur | loss_kpt_fut | lr | iters/s | epoch |
|------|------|-------------|----------|------------|-----------|-------------|-------------|------|---------|-------|
| 100 | 0.671 | 0.019 | 0.131 | 0.158 | 0.148 | 0.0232 | 0.0386 | 1.2e-06 | 0.23 | 0.02 |
| 1000 | 0.532 | 0.015 | 0.103 | 0.119 | 0.125 | 0.0148 | 0.0414 | 1.2e-05 | 0.24 | 0.25 |
| 4000 | 0.395 | 0.011 | 0.074 | 0.100 | 0.085 | 0.0143 | 0.0347 | 5.0e-05 | 0.24 | 1.00 |
| 8010 | 0.333 | 0.009 | 0.064 | 0.096 | 0.073 | 0.0140 | 0.0343 | 1.4e-05 | 0.24 | 2.00 |
| 12000 | 0.309 | 0.009 | 0.061 | 0.093 | 0.069 | 0.0123 | 0.0333 | 8.0e-06 | 0.24 | 3.00 |
| 14000 | 0.309 | 0.009 | 0.061 | 0.092 | 0.069 | 0.0122 | 0.0337 | 6.8e-06 | 0.24 | 3.50 |
| 16000 | 0.282 | 0.008 | 0.051 | 0.087 | 0.058 | 0.0109 | 0.0347 | 5.0e-06 | 0.25 | 4.00 |

### 检验结果

| 检查项 | 期望 | 实际 | 状态 |
|--------|------|------|------|
| exit code | 0 | 0 | OK |
| End of training | 是 | 是 (04:56:06) | OK |
| video_decode_error | 0 | 0 | OK |
| using_zeros | 0 | 0 | OK |
| Checkpoints saved | 008010, 016020 | 008010, 016020 | OK |
| "last" symlink | → 016020 | → 016020 | OK |
| state_dim (stats.json) | 14 | 14 | OK |
| action_dim (stats.json) | 7 | 7 | OK |
| robot_type key | libero_goal_4dv2 | libero_goal_4dv2 | OK |
| Epoch | ~4.0 | 4.00 | OK |
| Knowledge insulation | False | False | OK |
| Gradient checkpointing | True | True | OK |
| WAN video branch | Loaded (frozen) | Loaded (frozen) | OK |
| loss趋势 | 下降 | 0.671→0.282 | OK |
| 日志归档 | 已保存 | 4dwvlaLbPlusGolV2_1001_sft_LOG_20261003_045612.tar | OK |

### Checkpoint 路径

```
SFT_CKPT=~/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_10_07_07-internvla_a1_5-lbplus-golv2-sft/checkpoints/016020/pretrained_model
```

### Bug Fix 记录

在 Step 5 smoke test 阶段发现并修复了 `p2v2_sft_launch.sh` 的 checkpoint guard bug（详见 Step 5）。该修复在 production 训练中生效，checkpoint guard 正确识别了 state_dim=14。

**结论**: Production SFT Training **PASSED**

---

## Step 7: Eval Smoke Test (§7.2 步骤 7)

```
命令: CKPT_PATH=<sft_016020> EVAL_MODE=smoke bash b/s/libplus2/gol2/eval_v2_wrapper.sh
时间: 2026-10-03 04:57 ~ 05:47 (多次迭代修复)
模式: SMOKE (2 GPU, 2 shards, max_steps=100)
最终日志: /B/Log/4dwvlaLbPlusGolV2_1001/20261003_053441_eval/eval_goal.log
```

### Bug Fixes (评估阶段)

#### Bug 1: preflight_v2.sh stats.json 嵌套结构

```
文件: b/s/libplus2/gol2/preflight_v2.sh (line 47-51)
原因: 同 SFT 阶段 — stats.json 嵌套在 robot_type key 下，直接访问 observation.state 失败
修复: 添加与 p2v2_sft_launch.sh 相同的嵌套检测逻辑
```

#### Bug 2: preflight_v2.sh client mujoco+robosuite EGL

```
文件: b/s/libplus2/gol2/preflight_v2.sh (line 76)
原因: 检查 client venv import robosuite 时缺少 EGL 环境变量导致 ImportError
修复: 添加 env __EGL_VENDOR_LIBRARY_DIRS 和 MUJOCO_GL=egl
```

#### Bug 3: 健康检查 action_mode 不匹配

```
文件: b/s/libplus2/gol2/eval_v2_wrapper.sh (Phase 4)
原因: 原始 run_eval_libero_plus_venv.sh 健康检查断言 action_mode=='joint'
      v2 模型使用 abs/EEF 模式，server metadata 报告 action_mode='end_effector'
修复: eval_v2_wrapper 用 DRY_RUN=1 生成 derived script，然后 sed 替换 'joint' -> 'end_effector'
```

#### Bug 4: load_contract.py 不支持 schema v2

```
文件: 新建 b/s/libplus2/gol2/load_contract_v2.py, eval_goal_plus_v2.py
原因: gol/load_contract.py 只接受 goal_train_eval_contract/1 schema
      v2 数据使用 goal_train_eval_contract/2 schema
修复: 创建 gol2/load_contract_v2.py 支持 v1/v2 schema
      创建 gol2/eval_goal_plus_v2.py monkey-patch load_contract
      eval_v2_wrapper sed 替换 derived script 中的 eval_goal_plus.py -> eval_goal_plus_v2.py
```

#### Bug 5: MagickWand 共享库未找到

```
文件: b/s/libplus2/gol2/eval_v2_wrapper.sh (sed patch)
原因: ImageMagick 安装在 /opt/conda/lib 但 client venv LD_LIBRARY_PATH 中无此路径
      LIBERO-plus Background Textures 类任务依赖 wand/MagickWand
修复: eval_v2_wrapper sed 在 derived script 的 LD_LIBRARY_PATH 中插入 /opt/conda/lib
```

#### Bug 6: MAX_STEPS_OVERRIDE 未传递给 eval 客户端

```
文件: b/s/libplus2/gol2/eval_v2_wrapper.sh (sed patch)
原因: 原始 run_eval_libero_plus_venv.sh 无 --max_steps_override 参数传递
      smoke 模式设置 MAX_STEPS_OVERRIDE=100 但未传到 eval_libero_plus.py
修复: eval_v2_wrapper sed 在 derived script 的 python 命令中插入 --max_steps_override
```

### 验证结果

| 检查项 | 期望 | 实际 | 状态 |
|--------|------|------|------|
| Preflight v2 | 23/23 PASSED | 23/23 PASSED | OK |
| Healthcheck | PASSED | PASSED (gpu0 port 5784, gpu2 port 5785) | OK |
| Server启动 | 模型加载, 端口监听 | 加载完成, ws 连接正常 | OK |
| 任务执行 | 无 crash, 有结果 | 35 tasks completed, 0 crashes | OK |
| max_steps | 100 | 100 (已确认) | OK |
| MagickWand | 无 crash | 0 crashes | OK |
| contract v2 | 可加载 | load_contract_v2 成功 | OK |
| action_mode | end_effector | end_effector (健康检查通过) | OK |

**结论**: Eval Smoke Test **PASSED** (pipeline functional, tasks executing correctly)

---

## Step 8: Full Evaluation (§7.2 步骤 8)

### 配置

```
命令: CKPT_PATH="$HOME/b/Ckp/4dwvlaLbPlusGolV2_1001/2026_10_02_10_07_07-internvla_a1_5-lbplus-golv2-sft/checkpoints/016020/pretrained_model" \
      nohup bash b/s/libplus2/gol2/eval_v2_wrapper.sh > /tmp/eval_full_v2.log 2>&1 &
模式: full (8 GPU, 8 shards, 2591 tasks)
开始: 2026-10-03 05:48:33 UTC
结束: 2026-10-03 13:15:44 UTC
总耗时: 7h27m
PID: 540057
```

### GPU 完成时间

| GPU | Shard Range | Tasks | 完成时间 (UTC) | 耗时 |
|-----|-------------|-------|----------------|------|
| 6 | 1944-2268 | 324 | 08:23:27 | 2h33m |
| 7 | 2268-2591 | 323 | 08:50:49 | 3h00m |
| 0 | 0-324 | 324 | 09:05:04 | 3h15m |
| 3 | 972-1296 | 324 | 09:33:11 | 3h43m |
| 2 | 648-972 | 324 | 09:46:31 | 3h56m |
| 1 | 324-648 | 324 | 10:06:13 | 4h16m |
| 4 | 1296-1620 | 324 | 11:30:27 | 5h40m |
| 5 | 1620-1944 | 324 | 13:15:42 | 7h25m |

> GPU 5 瓶颈原因: shard 1620-1944 包含大量 Sensor Noise 类别任务，CPU 仿真耗时长，
> 导致 GPU 推理服务器大部分时间在等待客户端请求。

### 最终结果

**Overall Success Rate: 40.06% (1038/2591)**

| Category | Total | Success | SR |
|----------|-------|---------|-----|
| Light Conditions | 279 | 210 | **75.27%** |
| Background Textures | 281 | 211 | **75.09%** |
| Objects Layout | 425 | 244 | **57.41%** |
| Language Instructions | 410 | 184 | **44.88%** |
| Camera Viewpoints | 408 | 95 | **23.28%** |
| Sensor Noise | 379 | 72 | **19.00%** |
| Robot Initial States | 409 | 22 | **5.38%** |

### Crashes & Failures

```
Total crashes: 0
Total failures: 1553 (tasks with success=0)
Failures with saved actions: 1553
Failures manifest: /B/Log/4dwvlaLbPlusGolV2_1001/20261003_054824_eval/failures_manifest.json (1.1M)
Replay commands: /B/Log/4dwvlaLbPlusGolV2_1001/20261003_054824_eval/replay_commands.sh (372K)
```

### 产出物

| 产出物 | 路径 | 大小 |
|--------|------|------|
| overall_results.json | `/B/Log/.../20261003_054824_eval/overall_results.json` | 2.6K |
| failures_manifest.json | `/B/Log/.../20261003_054824_eval/failures_manifest.json` | 1.1M |
| replay_commands.sh | `/B/Log/.../20261003_054824_eval/replay_commands.sh` | 372K |
| eval_goal.log | `/B/Log/.../20261003_054824_eval/eval_goal.log` | full log |
| 结果归档 | `/B/Log/4dwvlaLbPlusGolV2_1001/eval_goal_20261003_054824.tar.gz` | 15M |
| per-GPU worker logs | `/B/Log/.../20261003_054824_eval/worker_gpu{0-7}/` | — |

### 结果分析

1. **光照/背景鲁棒性强** (~75%): 模型对视觉外观变化（光照条件、背景纹理）有较好的泛化能力。
2. **物体布局中等** (57.4%): 对物体摆放位置变化有一定适应性。
3. **语言指令中等** (44.9%): 对不同语言描述的泛化能力一般。
4. **视角/噪声弱** (19-23%): 对相机视角和传感器噪声变化泛化不足。
5. **初始状态极弱** (5.4%): 模型几乎完全不能适应机器人初始姿态的变化，是最大的泛化瓶颈。

**结论**: Full Evaluation **COMPLETED** (0 crashes, all 2591 tasks evaluated, results archived)

---

## Pipeline 总结

| Step | 描述 | 状态 | 耗时 |
|------|------|------|------|
| 0 | GPU 清理 | PASSED | <1m |
| 1 | 前提检查 | PASSED | <1m |
| 2 | Preflight v2 | PASSED | <1m |
| 3 | Warmup 训练 | PASSED (8010 steps) | ~2h |
| 4 | Warmup 检查点验证 | PASSED | <1m |
| 5 | SFT 训练 | PASSED (16020 steps, loss 0.671→0.282) | ~18h49m |
| 6 | SFT 检查点验证 | PASSED | <1m |
| 7 | Eval Smoke Test | PASSED (6 bugs fixed) | ~30m |
| 8 | Full Evaluation | PASSED (SR=40.06%, 0 crashes) | ~7h27m |

**Pipeline 状态: ALL STEPS PASSED**

