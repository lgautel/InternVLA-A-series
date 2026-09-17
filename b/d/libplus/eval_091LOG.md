# LIBERO-plus 正式评估执行日志

> Checkpoint: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/026725/pretrained_model/`
> 操作手册: `eval.md`
> 开始时间: 2026-09-14 16:47

---

## §1. 初始状态

### §1.1 GPU 状态

- 8 × NVIDIA H200 (143 GB each)
- 训练进程占用全部 8 GPU (~138 GB/卡), 已 kill -9

### §1.2 代码修复确认 (从 mini eval 继承)

| # | 文件 | 修复内容 | 确认状态 |
|---|------|---------|---------|
| 1 | `transform_internvla_a1_5.py:140-145` | Prompt suffix 根据 `use_fast_action_tokens` 动态选择 | ✅ 已确认 |
| 2 | `policy_backend_internvla_a1_5.py:148` | `getattr(config, "use_fast_action_tokens", True)` | ✅ 已确认 |
| 3 | `evaluation/LIBERO/keypoint_utils.py` | `KeypointExtractor` 存储 `env` 引用而非 `env.sim` | ✅ 已确认 |
| 4 | `LIBERO-plus/libero/libero/envs/env_wrapper.py:52` | `np.fromstring` → `np.frombuffer` (NumPy 2.x 兼容) | ✅ 已确认 |

### §1.3 评估配置

| 参数 | 值 |
|------|-----|
| GPU 数量 | 8 (GPU 0-7) |
| Shards per suite | 8 |
| 总工作单元 | 32 (4 suites × 8 shards) |
| 总任务数 | 10030 |
| Trials per task | 1 |
| `stats_key` / `robot_type` | `panda` / `panda` |
| `gripper_convention` | `libero_native` |
| `use_fast_action_tokens` | `True` (从 config getattr) |
| `replan_steps` | 8 |
| `inference_backend` | standard |
| `action_loss_only` | True |
| 保存视频 | 是 (全部, 含 success/failure) |
| 评估结果目录 | `full_20260914_164719/` |

---

## §2. 评估启动

- 启动时间: 2026-09-14 16:47:19
- 启动命令:
  ```bash
  nohup bash evaluation/LIBERO-plus/run_eval_libero_plus_venv.sh > "${EVAL_DIR}/full_eval.log" 2>&1 &
  ```
- 环境变量:
  ```
  CKPT_PATH=.../checkpoints/026725/pretrained_model
  STATS_KEY_MODE=panda, ROBOT_TYPE_MODE=panda, GRIPPER_CONVENTION=libero_native
  GPU_IDS=0,1,2,3,4,5,6,7, SHARDS_PER_SUITE=8
  NO_VIDEO_FLAG="" (save all videos)
  ```
- 主进程 PID: 2445997
- 8 workers 分别启动 libero_spatial 的 8 个 shards

### §2.1 首批 shard 进度 (libero_spatial, ~17:04)

| GPU | Shard | 完成 | SR | 状态 |
|-----|-------|------|-----|------|
| 0 | [0,301) | 32/301 | 0% | 运行中 |
| 1 | [301,602) | 33/301 | 3% | 运行中 |
| 2 | [902,1202) | 32/300 | 0% | 运行中 |
| 3 | [602,902) | 82/300 | 0% | **FAILED** (EGL) |
| 4 | [1202,1502) | 35/300 | 20% | 运行中 |
| 5 | [1502,1802) | 30/300 | 3.3% | **FAILED** (np.float_) |
| 6 | [1802,2102) | 20/300 | 0% | 运行中 |
| 7 | [2102,2402) | 30/300 | 0% | 运行中 |

---

## §3. 运行时错误与修复

### §3.1 Error #5: `np.float_` removed in NumPy 2.0

- **时间**: ~17:25 (GPU 5, libero_spatial[1502:1802], task 30/300)
- **错误**: `AttributeError: np.float_ was removed in NumPy 2.0`
- **位置**: `LIBERO-plus/libero/libero/envs/env_wrapper.py:105`
- **函数**: `plasma_fractal()` — Sensor Noise 等离子体噪声生成
- **根因**: LIBERO-plus 使用 NumPy 1.x 的 `np.float_`, NumPy 2.x 已移除
- **修复**: `np.float_` -> `np.float64`
- **影响**: 丢失 270 个任务, 需后续重跑

### §3.2 Error #6: EGL Offscreen Framebuffer

- **时间**: ~17:25 (GPU 3, libero_spatial[602:902], task 82/300)
- **错误**: `mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd`
- **触发**: Camera Viewpoints task_id=684
- **根因**: 某些 Camera Viewpoint 角度导致 EGL 创建离屏渲染失败 (驱动级别)
- **修复**: 无法应用层修复, 需后续重跑该 shard
- **影响**: 丢失 218 个任务

### §3.3 EGL 崩溃影响汇总 (libero_spatial)

| GPU | Shard | 崩溃点 | 崩溃原因 | 丢失任务 |
|-----|-------|--------|---------|---------|
| 0 | [0,301) | task 254 | EGL framebuffer | 47 |
| 3 | [602,902) | task 82 | EGL framebuffer | 218 |
| 4 | [1202,1502) | task 192 | EGL framebuffer | 108 |
| 5 | [1502,1802) | task 30 | np.float_ | 270 |
| **总计** | | | | **643/2402 (26.8%)** |

### §3.4 Error #7: 多个 shard 因 EGL 崩溃

- **发现时间**: ~18:40
- **根因分析**: EGL 离屏渲染资源累积问题, 不仅限于 Camera Viewpoints, Sensor Noise 等类别也会触发
- **修复**: 在 `eval_libero_plus.py:215-233` 的任务循环中添加 try-except, 捕获 EGL/mujoco 异常后将该任务标记为失败并继续, 而非让整个 shard 崩溃
- **文件改动**: `evaluation/LIBERO-plus/eval_libero_plus.py` — 包裹 `evaluate_task()` 调用
- **生效范围**: 仅对后续从队列取的新 shard 生效, 已运行的 client 不受影响
- **备注**: libero_spatial 的失败 shard 需要后续重跑

### §3.5 崩溃后恢复

- GPU 0, 3, 4, 5 均自动切换到下一个队列任务
- 队列剩余: 1 object + 8 goal + 8 libero_10 = 17 shards

---

## §4. 中途停止与结果

### §4.1 停止时间
- 运行时间: ~7.5小时 (2026-09-14 16:47 ~ 2026-09-15 00:19)
- 停止原因: 用户主动停止
- 停止操作: kill -9 所有 worker/server/main 进程, GPU 清零

### §4.2 已评估任务统计
- 总观测任务: 6365 / 10030 (63.5%)
- 未评估任务: 3665 (libero_10 大部分, libero_spatial/object/goal 部分 shard)

### §4.3 各子集成功率 (截至停止时)

| Suite | 成功 | 总计 | **SR** |
|-------|------|------|--------|
| libero_spatial | 123 | 1744 | **7.05%** |
| libero_object | 543 | 2164 | **25.09%** |
| libero_goal | 178 | 2166 | **8.22%** |
| libero_10 | 29 | 291 | **9.97%** |
| **总体** | **873** | **6365** | **13.72%** |

### §4.4 各扰动类别初步成功率 (libero_object, 最全面的已完成数据)

libero_object 共完成 2164/2518 任务:
- Background Textures: 97/248 = **39.1%**
- Camera Viewpoints: 29/97 = **29.9%**
- Language Instructions: 102/354 = **28.8%**
- Light Conditions: 111/297 = **37.4%**
- Objects Layout: 3/17 = **17.6%**
- Robot Initial States: 29/382 = **7.6%**
- Sensor Noise: 28/179 = **15.6%**

### §4.5 失败 Shard 汇总 (7个 shard FAILED)

| GPU | Shard | 崩溃点 | 原因 | 是否需重跑 |
|-----|-------|--------|------|----------|
| 0 | spatial[0,301) | 254/301 | EGL 0x8cdd | 是 |
| 3 | spatial[602,902) | 82/300 | EGL 0x8cdd | 是 |
| 4 | spatial[1202,1502) | 192/300 | EGL 0x8cdd | 是 |
| 5 | spatial[1502,1802) | 30/300 | np.float_ | 是 |
| 6 | spatial[1802,2102) | 285/300 | EGL 0x8cdd | 是 |
| 7 | object[1890,2204) | 108/314 | EGL 0x8cdd | 是 |
| 4 | object[630,945) | 260/315 | EGL 0x8cdd | 是 |

**备注**: goal/libero_10 shards 使用了 try-except 修复, 未发生 shard 级别崩溃

### §4.6 输出文件路径

**主目录**: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/eval_libero_plus/step_026725/full_20260914_164719/`

| 类型 | 路径 | 内容 |
|------|------|------|
| 主日志 | `.../full_eval.log` | 评估整体启动日志 |
| Worker日志 | `.../worker_gpu{0-7}/worker.log` | 各 GPU 工作日志 |
| Client日志 | `.../worker_gpu{0-7}/client_libero_*.log` | 各 shard 任务评估详情 |
| Server日志 | `.../worker_gpu{0-7}/server_libero_*.log` | 推理服务器日志 |
| JSON结果 | `.../logs/libero_{suite}/{start}_to_{end}.json` | 12个已完成 shard 结果 |
| 视频 | `.../videos/` | 6342 个 MP4 视频 (success+failure) |

### §4.7 代码文件改动汇总

| 文件 | 改动 | 原因 |
|------|------|------|
| `evaluation/LIBERO-plus/eval_libero_plus.py:215-233` | 为 `evaluate_task()` 添加 try-except | 防止 EGL 崩溃终止整个 shard |
| `LIBERO-plus/libero/libero/envs/env_wrapper.py:105` | `np.float_` → `np.float64` | NumPy 2.x 兼容 |

(以上是本次正式评估新增的改动, 继承自 mini-eval 的4个修复见 §1.2)

