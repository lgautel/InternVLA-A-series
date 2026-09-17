# Standard LIBERO 基线验收执行日志

**日期**: 2026-09-16
**执行人**: Claude (按 eval3_optim3.md §十六 操作手册)
**Checkpoint**: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model`

---

## 0. 前提检查 (§16.3.0)

### 0.1 预检测试 (test_preflight_f1f2.py)

**首次运行**: 123/125 PASS, 2 FAIL

**失败项**: `A24.plus2:push_before_env_step`, `A24.std:push_before_env_step`

**根因分析**: A24 测试用正则扫描 `evaluate_task` 函数源码, 寻找所有含 `num_steps_wait` 的行,
然后在该行后 8 行窗口内检查 `push_keypoint()` 是否在 `LIBERO_DUMMY_ACTION` 之前出现.
我们在上一轮 (eval3_optim3_f1f2LOG 所记录的修改) 为失败复现添加了 `failure_info` dict,
其中包含 `"num_steps_wait": args.num_steps_wait` — 这一行也匹配了 `num_steps_wait`,
但它后面的 8 行窗口是 JSON 字段, 不含 `push_keypoint` 也不含 `LIBERO_DUMMY_ACTION`,
导致 `wait_ok` 被最后这次匹配覆写为 `False`.

**修复方案**: 将 `_repro_base` dict 移到 `evaluate_task` 函数入口处 (主循环之前) 构建,
其中 `args.num_steps_wait` 在循环前就完成引用, 主循环区间内不再出现该字符串.

**修改文件**:
- `evaluation/LIBERO2/eval_libero_std.py`: 新增 `_repro_base` dict (L96-L105), 失败记录改用 `{**_repro_base, ...}`
- `evaluation/LIBERO-plus2/eval_libero_plus.py`: 同上 (L87-L96)

**二次运行**: 125/125 PASS ✅

### 0.2 其他前提检查

| 检查项 | 结果 |
|--------|------|
| Port 5784 | 空闲 ✅ |
| Checkpoint `model.safetensors` | 存在 ✅ |
| LIBERO bddl_files (libero_spatial) | 2031 文件 ✅ |
| EGL vendor `10_nvidia.json` | 存在 ✅ |
| GPU | 4× NVIDIA H200 ✅ |

所有前提检查通过, 进入 §16.3.1 配置环境变量.

---

## 1. 冒烟测试 (Gate 3a)

### 1.1 第一次运行 (Run 1)

**EVAL_LOG_DIR**: `.../checkpoints/032070/libero_std_202609160649`

**配置**:
- `EVAL_MODE=smoke` → 4 suite × 2 tasks × 5 trials = 40 episodes
- `ROTATE_IMAGES=false` (F1 fix 生效)
- `GPU_ID=0`, `PORT=5784`, `SEED=7`, `REPLAN_STEPS=8`

**结果**: **SR=0.00% (0/40) — Gate 3 FAIL**

**全部 8 个 task 均 CRASHED (SIGABRT, signal 6)**:

| Suite | task_id | 进度 | 错误 |
|-------|---------|------|------|
| libero_spatial | 0 | 80% (4/5 episodes 完成后 SIGABRT) | Killed by SIGABRT |
| libero_spatial | 1 | 80% (4/5) | Killed by SIGABRT |
| libero_object | 0 | 80% (4/5) | Killed by SIGABRT |
| libero_object | 1 | 80% (4/5) | Killed by SIGABRT |
| libero_goal | 0 | 80% (4/5) | Killed by SIGABRT |
| libero_goal | 1 | 80% (4/5) | Killed by SIGABRT |
| libero_10 | 0 | 20% (1/5, max_steps=520) | Killed by SIGABRT |
| libero_10 | 1 | (同上) | Killed by SIGABRT |

**根因分析**:

SIGABRT 发生在 forked 子进程中, 是 MuJoCo EGL context 在连续多次 render 后变得不稳定的已知问题.
关键观察:
- `libero_spatial/object/goal` (max_steps=220~300): 在第 5 个 episode 时 crash (已完成 ~880-1200 次 render)
- `libero_10` (max_steps=520): 在第 2 个 episode 时 crash (已完成 ~520 次 render)
- **共同阈值**: 约 500-1000 次 `env.step()` 渲染调用后 EGL context 失效

B10 的 fork-per-task 隔离机制成功防止了 SIGABRT 传播到父进程,
但由于子进程在 SIGABRT 时无法写入结果 JSON, 所有已完成的 episode 结果全部丢失.

**修复方案**:

在 `_run_task_in_subprocess` 中为子进程安装 SIGABRT 信号处理器:
1. 新增 `_partial` 字典, 在 `evaluate_task` 中每完成一个 episode 就追加结果
2. SIGABRT 处理器将已完成的 episode 结果写入 result JSON, 未完成的补为 False
3. 这样即使 SIGABRT 杀死子进程, 已完成 episode 的结果不会丢失

**修改文件**:
- `evaluation/LIBERO2/eval_libero_std.py`:
  - `evaluate_task()`: 新增 `_partial_results` 参数, 每完成 1 episode 就写入 `_partial["successes"]`
  - `_run_task_in_subprocess()`: 新增 `_partial` dict + `_sigabrt_handler` 信号处理器
- `evaluation/LIBERO-plus2/eval_libero_plus.py`: 同上

**修复后预检测试**: 125/125 PASS ✅

### 1.2 第二次运行 (Run 2) — 含 SIGABRT handler
