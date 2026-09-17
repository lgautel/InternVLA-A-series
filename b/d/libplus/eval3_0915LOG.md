# eval3 评估执行日志 (step_032070)

> **日期**: 2026-09-15
> **Checkpoint**: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model/`
> **参考文档**: `eval3.md` v2.1
> **操作者**: Claude Opus 4.6 (自动化)

---

## §1 初始状态检查

| 项目 | 状态 | 说明 |
|------|------|------|
| `evaluation/panda_robosuite_lift.xml` | ✅ 存在 | 38774 bytes, robosuite 1.4.0 Lift MJCF |
| `evaluation/LIBERO/` | ✅ 存在 (READ-ONLY) | 原始代码, 不修改 |
| `evaluation/LIBERO-plus/` | ✅ 存在 (READ-ONLY) | 原始代码, 不修改 |
| `evaluation/LIBERO2/` | ❌ 不存在 | 需创建 |
| `evaluation/LIBERO-plus2/` | ❌ 不存在 | 需创建 |
| Checkpoint config.json | ✅ 可访问 | 3601 bytes |
| SERVER_VENV `/B/VENV/itnvla15rbt20` | ✅ 存在 | |
| CLIENT_VENV `/B/VENV/libero_plus_client` | ✅ 存在 | |
| GPUs | ✅ 8× NVIDIA H200 (143GB each) | GPU 0 has 13GB used |
| B3+B4 patches (env_wrapper.py) | ✅ 已应用 | `np.frombuffer` + `np.float64` |

---

## §2 创建 LIBERO2/ 代码文件

### 2.1 创建目录结构

```bash
mkdir -p evaluation/LIBERO2/policy_server/backends
mkdir -p evaluation/LIBERO-plus2
```

### 2.2 创建的文件列表 (12 files)

| # | 文件 | 大小 | 关键修改 |
|---|------|------|---------|
| 1 | `evaluation/LIBERO2/__init__.py` | 0B | 空 |
| 2 | `evaluation/LIBERO2/keypoint_utils.py` | ~3.5KB | StandaloneFK + KeypointExtractor + KeypointHistory (R_pad=1.8213, body=gripper0_eef) |
| 3 | `evaluation/LIBERO2/model2libero_interface.py` | ~7KB | B1 fix (gripper_convention), keypoint integration (bind_env, push_keypoint, kpt_history in payload) |
| 4 | `evaluation/LIBERO2/policy_server/__init__.py` | 0B | 空 |
| 5 | `evaluation/LIBERO2/policy_server/backends/__init__.py` | 0B | 空 |
| 6 | `evaluation/LIBERO2/policy_server/backends/policy_backend_internvla_a1_5.py` | ~7KB | B2 fix (use_fast_action_tokens=getattr(config, ..., True)), kpt injection to sample |
| 7 | `evaluation/LIBERO2/policy_server/backends/backend_factory.py` | ~3KB | Imports LIBERO2 InternVLAA15Backend |
| 8 | `evaluation/LIBERO2/policy_server/server_policy.py` | ~3KB | Imports LIBERO2 backend_factory |
| 9 | `evaluation/LIBERO-plus2/__init__.py` | 0B | 空 |
| 10 | `evaluation/LIBERO-plus2/eval_libero_plus.py` | ~9KB | B6 (per-task try-except), B7 (delayed imageio import), keypoint support, --save_actions, --categories |
| 11 | `evaluation/LIBERO-plus2/aggregate_results.py` | ~3KB | 从 LIBERO-plus/ 直接复制 |
| 12 | `evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh` | ~6KB | 全自动 8-GPU 启动器 |

---

## §3 语法验证 (§7.5)

```
keypoint_utils.py OK
model2libero_interface.py OK
server_policy.py OK
backend_factory.py OK
policy_backend.py OK
eval_libero_plus.py OK
aggregate_results.py OK
Shell script syntax OK
```

所有 7 个 Python 文件 + 1 个 Shell 脚本语法验证通过.

---

## §4 Preflight 测试 (§8)

### 4.1 Lift MJCF 验证

```
nq=16, nbody=24
gripper0_eef 存在于 body 列表
Lift MJCF verification PASSED
```

### 4.2 StandaloneFK + KeypointHistory 单元测试

```
StandaloneFK loaded: 8 body IDs, R_pad=1.8212722539901733
Zero-pose kpts shape: (8, 7), dtype: float32
Zero-pose link1 pos: [-0.30747736  0.          0.6835881 ]
Zero-pose EEF pos: [-0.2591595  0.         0.9561997]
History: buf.shape=(200, 8, 7), his_len=5
All preflight tests PASSED
```

---

## §5 Policy Server 启动与 Healthcheck

### 5.1 Server 启动 (GPU 0, port 5784)

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/LIBERO2/policy_server/server_policy.py \
    --ckpt_path "...032070/pretrained_model" \
    --host 0.0.0.0 --port 5784 --device cuda \
    --resize_size 224 --stats_key panda --robot_type panda \
    --action_loss_only --inference_backend standard --idle_timeout -1
```

Server metadata 确认:
```
policy_type: internvla_a1_5
stats_key: panda
robot_type: panda
action_mode: joint
chunk_size: 50
action_dim: 7
action_denorm_mode: mean_std
expected_num_input_images: 2
expected_state_dim: 8
```

### 5.2 Healthcheck (9项全通过)

```
[OK] action_mode: joint
[OK] stats_key: panda
[OK] robot_type: panda
[OK] action_dim: 7
[OK] chunk_size: 50
[OK] expected_num_input_images: 2
[OK] expected_state_dim: 8
[OK] preprocessing_owner: server_canonical
[OK] action_denorm_mode: mean_std
HEALTHCHECK: ALL 9 checks PASSED
```

---

## §6 Smoke Test

### 6.1 EGL 错误及修复

**首次运行报错**: `ImportError: Cannot initialize a EGL device display.`

**根因**: `__EGL_VENDOR_LIBRARY_DIRS` 环境变量未设置. EGL 需要知道 vendor JSON 的位置 (`/B/VENV/libero_plus_client/egl_vendor.d/10_nvidia.json`).

**修复**: 在 client 启动命令中添加:
```bash
export __EGL_VENDOR_LIBRARY_DIRS="/B/VENV/libero_plus_client/egl_vendor.d"
```

### 6.2 Smoke Test 结果 (1 task, libero_spatial[5:6])

```
Suite=libero_spatial | shard [5, 6) of 2402 tasks | max_steps=220 | trials/task=1
task_id=5 [Background Textures] 'pick up the black bowl...' -> 0/1
LIBERO-plus2 eval done. Suite=libero_spatial shard=[5,6) SR=0.0000 (0/1)
EXIT CODE: 0
```

- ✅ 无 SIGABRT crash (B7 修复有效 — 无顶层 import imageio)
- ✅ Keypoint extraction 运行无错误
- ✅ Action 序列保存成功 (actions shape=[220, 7])
- ✅ Per-shard JSON 输出正常
- ✅ 干净退出 (exit code 0)

---

## §7 全量评估 (8 GPU, 10030 tasks)

### 7.1 启动参数

```bash
export CKPT_PATH="...032070/pretrained_model"
export LIBERO_HOME="/home/a26113/DATA/LIBERO-plus"
export SERVER_VENV="/B/VENV/itnvla15rbt20"
export CLIENT_VENV="/B/VENV/libero_plus_client"
bash evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh
```

- 8 GPUs (H200 143GB each), 32 shards (4 per GPU, round-robin)
- Ports 5784-5791
- SERVER_STARTUP_WAIT=90s, 3x healthcheck retries
- `--no-save_videos --save_actions`

### 7.2 第一次全量启动结果: SIGABRT (exit 134)

**输出目录**: `.../eval_libero_plus/full_20260915_083752/`

**现象**: 8个GPU上的所有 client 在第 1 个 task 完成后, 尝试运行第 2 个 task 时, 进程被 SIGABRT (signal 6, exit code 134) 杀死. Python 层面的 try-except 无法捕获, 因为是 C 层面的 abort.

**Client log 特征**:
```
task_id=0 [...] 'pick up the black bowl...' -> 0/1   # 第1个task成功完成
# 此后进程被SIGABRT杀死, 无Python traceback
```

**Shell 层面错误**: `set -euo pipefail` 导致 client 非零退出时整个 worker 立即退出, server 也随之被 kill.

**Shell 修复**: 在 client subshell 前后添加 `set +e` / `set -e`, 并修正 `CLIENT_EXIT=$?` (原来的 `local CLIENT_EXIT=$?` 会被 local 的返回值覆盖).

---

## §8 SIGABRT 根因分析与修复 (Bug B10)

### 8.1 根因

**SIGABRT 触发路径**:
1. `eval_libero_plus.py` 启动时, 顶层 `from evaluation.LIBERO2.model2libero_interface import LiberoModelClient` 触发 `import mujoco` → MuJoCo 初始化 EGL context #1
2. 第 1 个 task: `_get_libero_env()` 调用 `OffScreenRenderEnv()` → 创建 `EGLGLContext` → 共享已有 EGL display → 正常
3. 第 1 个 task 完成, `env.close()` → `EGLGLContext.__del__()` → 触发 EGL context 清理
4. 第 2 个 task: `OffScreenRenderEnv()` → `eglCreateContext()` → C 层面 SIGABRT

**本质问题**: MuJoCo 的 EGL 后端不支持在同一进程中 destroy 再 recreate EGL context. 这是 MuJoCo + EGL headless rendering 的已知限制.

### 8.2 修复尝试 (5次)

| # | 方案 | 结果 | 失败原因 |
|---|------|------|---------|
| 1 | 猴子补丁 `env.close = lambda: None; del env` | ❌ SIGABRT | `del env` 触发 `__del__` → C 层面清理 |
| 2 | `_LEAKED_ENVS.append(env)` 阻止 GC | ❌ SIGABRT | 问题不是 env 清理, 而是创建第二个 EGL context |
| 3 | fork-per-task + `_reset_egl()` (eglTerminate/eglReleaseThread) | ❌ 双子进程SIGABRT | `eglTerminate` 破坏了 EGL 状态 |
| 4 | fork + 子进程内 lazy import LiberoModelClient | ❌ EGL_BAD_ALLOC | 父进程的 `from libero.libero import benchmark` 也触发了 EGL |
| 5 | **ALL LIBERO/mujoco imports 仅在子进程内** | ✅ 成功 | 父进程零 EGL 状态, 子进程独立初始化 |

### 8.3 最终方案: fork-per-task 完全隔离

**核心思路**: 父进程 **绝不** import 任何会触发 `import mujoco` 的模块 (包括 `libero.libero.benchmark`, `robosuite`, `LiberoModelClient`). 所有这些 import 只在 `os.fork()` 后的子进程内执行.

**代码架构**:
```
Parent process (evaluate_policy):
  - 从 task_classification.json 直接读取 task metadata (不用 benchmark)
  - n_tasks_in_suite = max(int(item_id) for item_id in id2category)
  - 循环 task_ids, 对每个 task 调用 _run_task_in_subprocess()

_run_task_in_subprocess():
  pid = os.fork()
  if pid == 0:   # CHILD
      from evaluation.LIBERO2.model2libero_interface import LiberoModelClient  # → mujoco EGL
      from libero.libero import benchmark                                       # → mujoco EGL
      child_client = LiberoModelClient(...)
      task = benchmark_dict[suite]().get_task(task_id)
      successes = evaluate_task(task, ...)
      json.dump(result, result_file)
      os._exit(0)
  # PARENT
  os.waitpid(pid, 0)
  return json.load(result_file)
```

**关键设计点**:
- 子进程每次独立创建全新 EGL context, 评估完后通过 `os._exit(0)` 退出 (不触发 `__del__`)
- 父子进程通过临时 JSON 文件传递结果
- 子进程 crash (SIGABRT等) 不影响父进程, 通过 `os.waitpid()` 检测退出状态
- 每个子进程重新 import benchmark (~35s), 是性能代价但保证稳定性

### 8.4 验证结果

**2-task 复现测试** (libero_spatial[0:2]):
```
task_id=0 [Background Textures] 'pick up the black bowl...' -> 0/1 (30s)
task_id=1 [Background Textures] 'pick up the black bowl...' -> 0/1 (30s)   ← NO SIGABRT
Exit code: 0
Total time: ~130s (including 2x ~35s import overhead)
```

### 8.5 性能影响

| 指标 | 原方案 (单进程) | B10 方案 (fork-per-task) |
|------|----------------|------------------------|
| 每 task 时间 | ~30s | ~65s (35s import + 30s eval) |
| 每 shard (~300 tasks) | ~2.5h | ~5.4h |
| 全量 (32 shards, 8 GPUs) | ~10h | ~22h |
| 稳定性 | ❌ 第2个task就SIGABRT | ✅ 完全隔离 |

---

## §9 第二次全量评估 — SIGABRT 再现 (Bug B11)

### 9.1 启动信息

- **启动时间**: 2026-09-15 09:33:39 UTC
- **输出目录**: `.../eval_libero_plus/full_20260915_093308/`
- **PID**: 3543331

### 9.2 结果: 全部 SIGABRT

**现象**: 所有 8 个 GPU 上, 每一个 forked 子进程都被 SIGABRT (signal 6) 杀死. 包括每个 shard 的第一个 task (排除 EGL context 复用问题).

**表现**:
- 子进程成功完成: import benchmark, 连接 server, 创建 env, 开始 episode loop
- 子进程在 evaluation 过程中被 SIGABRT 杀死 (运行约 2-3 分钟后)
- 父进程正确捕获 crash 并继续 (set +e 修复有效)

**关键对比**: 单客户端 debug 测试 (相同 CLIENT_VENV, 相同环境变量) 完全正常, 无 SIGABRT.

### 9.3 根因分析 (B11)

**真正的根因**: 8 个 GPU worker 的 client 子进程同时在 **GPU 0** 上创建 EGL rendering context.

Shell 脚本中, server 通过 `CUDA_VISIBLE_DEVICES=${GPU_IDX}` 指定 GPU, 但 client subshell **没有** 设置 `MUJOCO_EGL_DEVICE_ID`. MuJoCo EGL 默认使用 device 0. 因此:
- 8 个 forked 子进程同时在 GPU 0 上创建 EGL context
- GPU 0 的 EGL 驱动在高并发下触发 SIGABRT (NVIDIA driver 竞争条件或资源限制)

**验证**: 单客户端测试 (仅 1 个 EGL context on GPU 0) 完全正常; 8 并发客户端全部 SIGABRT.

### 9.4 修复: MUJOCO_EGL_DEVICE_ID

在 `run_eval_libero_plus_venv.sh` 的 client subshell 中添加:
```bash
export MUJOCO_EGL_DEVICE_ID=${SLOT_IDX}
```

每个 GPU worker 的 client 使用对应 GPU 进行 EGL rendering:
- GPU 0: server (CUDA inference) + client EGL rendering
- GPU 1: server (CUDA inference) + client EGL rendering
- ...
- GPU 7: server (CUDA inference) + client EGL rendering

每个 GPU 上最多 1 个 EGL context (因 worker 内 task 是串行处理的), 消除了并发竞争.

---

## §10 第三次全量评估 (B11 修复后)

### 10.1 启动信息

- **启动时间**: 2026-09-15 09:54:37 UTC
- **输出目录**: `.../eval_libero_plus/full_20260915_095437/`
- **日志文件**: `/home/a26113/b/Ckp/eval_full_10030_20260915_v2.log`
- **PID**: 3564074
- **修复**: `MUJOCO_EGL_DEVICE_ID=${SLOT_IDX}` 分配 EGL rendering 到不同 GPU

### 10.2 初始验证

- 8 个 server 全部启动, healthcheck 通过
- 8 个 shard 全部开始处理
- **前 15 个 task 全部完成, 0 SIGABRT, 0 CRASHED**
- Task 0 (GPU0): ~129s (含首次 import 开销)
- Task 1 (GPU0): ~75s
- 预估总时长: ~25 小时 (10030 tasks × 75s / 8 GPUs)

### 10.3 Shard 分配

| GPU | Port | EGL Device | Shards |
|-----|------|------------|--------|
| 0 | 5784 | 0 | libero_spatial[0:301], libero_object[0:315], libero_goal[0:324], libero_10[0:315] |
| 1 | 5785 | 1 | libero_spatial[301:602], libero_object[315:630], libero_goal[324:648], libero_10[315:630] |
| 2 | 5786 | 2 | libero_spatial[602:903], libero_object[630:945], libero_goal[648:972], libero_10[630:945] |
| 3 | 5787 | 3 | libero_spatial[903:1204], libero_object[945:1260], libero_goal[972:1296], libero_10[945:1260] |
| 4 | 5788 | 4 | libero_spatial[1204:1505], libero_object[1260:1575], libero_goal[1296:1620], libero_10[1260:1575] |
| 5 | 5789 | 5 | libero_spatial[1505:1806], libero_object[1575:1890], libero_goal[1620:1944], libero_10[1575:1890] |
| 6 | 5790 | 6 | libero_spatial[1806:2107], libero_object[1890:2205], libero_goal[1944:2268], libero_10[1890:2205] |
| 7 | 5791 | 7 | libero_spatial[2107:2402], libero_object[2205:2518], libero_goal[2268:2591], libero_10[2205:2519] |

### 10.4 进度 (持续更新)

> 评估进行中. 预计完成时间: 2026-09-16 ~11:00 UTC.

