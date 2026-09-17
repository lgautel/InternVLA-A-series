# Standard LIBERO 基线验收执行日志 (六 Server 方案)

**日期**: 2026-09-16
**执行人**: Claude (按 eval3_optim3.md §二十 + §十六 操作手册)
**方案**: 六 Server + 六 EGL Client (§二十 标准部署方案)
**Checkpoint**: `/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model`
**启动器**: `evaluation/LIBERO2/run_eval_libero_std_2server_6client_venv.sh`
**参考文档**: `libero_egl_slut.md` (六 server 方案原始探索), `eval3_optim3.md` §二十 (自包含操作手册)

---

## 0. 前提检查 (§16.3.0 + §20.5 S1)

### 0.1 硬件环境确认

| 检查项 | 结果 |
|--------|------|
| GPU | 8× NVIDIA H200, 每卡 143771 MiB ✅ |
| Checkpoint `model.safetensors` | 存在 ✅ |
| `stats.json` 含 `panda` key | True ✅ |
| LIBERO bddl_files (libero_spatial) | 2031 文件 ✅ |
| EGL vendor `10_nvidia.json` | 存在 ✅ |
| 所有目标端口 (5804-5809) | 空闲 ✅ |
| GPU 显存占用 | 全部 0 MiB ✅ |

### 0.2 S1 语法检查 (§20.5)

| 检查项 | 结果 |
|--------|------|
| `bash -n run_eval_libero_std_2server_6client_venv.sh` | PASS ✅ |
| `py_compile aggregate_std_results.py` | PASS ✅ |

### 0.3 预检测试 (test_preflight_f1f2.py)

**结果**: 137/137 PASS ✅

所有前提检查通过, 进入冒烟测试.

---

## 1. 冒烟测试 (Gate 3a) — 六 Server 方案

### 1.1 配置

- **拓扑**: 六 Server (SERVER_INSTANCES_PER_GPU=3, 默认值)
  - GPU 0: server-0/1/2 (ports 5804/5805/5806) ← client GPU 2/3/4
  - GPU 1: server-3/4/5 (ports 5807/5808/5809) ← client GPU 5/6/7
- **EVAL_MODE**: smoke → 4 suite × 2 tasks × 5 trials = 40 episodes
- **BASE_PORT**: 5804
- **RENDER_BACKEND**: egl
- **INFERENCE_BACKEND**: standard (兼容 keypoint)
- **ROTATE_IMAGES**: false (F1 fix)
- **SEED**: 7, **REPLAN_STEPS**: 8

### 1.2 启动命令

```bash
export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
export SERVER_VENV=/B/VENV/itnvla15rbt20
export CLIENT_VENV=/B/VENV/libero_plus_client
export CKPT_PATH=/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model
export VLM_MODEL_PATH=/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc

cd /B/SRC/itvlaGpLibPlus
EVAL_MODE=smoke bash evaluation/LIBERO2/run_eval_libero_std_2server_6client_venv.sh
```

### 1.3 执行过程

**时间线**:
- `15:47:42` — 启动 6 个 server (PID 2071877-2071906)
- `15:48:46-15:48:51` — 全部 6 个 server healthcheck PASS (~64s)
- `15:48:52` — 写入 GPU topology snapshot, 启动 6 个 client
- `15:50:45-15:50:46` — client 0/1 完成 libero_spatial, 进入 libero_10
- `15:53:05` — 全部 client 完成
- `15:53:06` — 聚合结果, cleanup

**Server 启动 → client 完成**: 约 5 分 24 秒
**Client 阶段 (6 并行)**: 约 4 分 13 秒

**GPU topology snapshot 验证**:
- GPU 0: 3 个 server 进程, 每个 12908 MiB (共 38751 MiB)
- GPU 1: 3 个 server 进程, 每个 12908 MiB (共 38750 MiB)
- GPU 2-7: 0 MiB (纯 EGL client, 不加载模型)

### 1.4 冒烟测试结果

**EVAL_LOG_DIR**: `.../checkpoints/032070/libero_std_2server_6client_20260916154739`

| Suite | Episodes | Successes | SR |
|-------|---:|---:|---:|
| libero_spatial | 10 | 10 | **100.00%** |
| libero_object | 10 | 10 | **100.00%** |
| libero_goal | 10 | 9 | **90.00%** |
| libero_10 | 10 | 10 | **100.00%** |
| **TOTAL** | **40** | **39** | **97.50%** |

**Gate 3a 判据**: SR=97.50% ≥ 50% → **PASS** ✅

**稳定性检查**:
- crash_task_count: 0 ✅
- SIGABRT/SIGSEGV/CRASHED: 0 ✅
- 所有 8 个 shard result JSON 均正常写出 ✅
- 全部 8 个 task 的 `error` 字段均为 `null` ✅

**唯一未满分的 task**:
- `libero_goal` task 0 "open the middle drawer of the cabinet": 4/5 (80%)

### 1.5 与上一轮 (eval3_optim3_libLOG Run 1) 对比

| 指标 | 上一轮 (单 server, 无 SIGABRT handler) | 本轮 (六 server) |
|------|---:|---:|
| SR | 0.00% (0/40) — 全部 SIGABRT | **97.50% (39/40)** |
| SIGABRT | 8/8 task crashed | **0** |
| 端到端时间 | ~5 min (全部 crash) | ~5 min 24s (正常完成) |

**结论**: 六 server GPU 分离方案彻底解决了之前的 SIGABRT 问题. 冒烟测试 PASS, 进入全量测试 (Gate 3b).

---

## 2. 全量测试 (Gate 3b) — 六 Server 方案

### 2.1 配置

- **拓扑**: 六 Server (同冒烟)
- **EVAL_MODE**: full → 4 suite × 10 tasks × 50 trials = 2000 episodes
- **验收阈值**: SR ≥ 85% (论文 SR > 95%, 本 checkpoint 训到 60%, 经验衰减 ~10-15 pp)

### 2.2 启动命令

```bash
export LIBERO_HOME=/home/a26113/DATA/LIBERO-plus
export SERVER_VENV=/B/VENV/itnvla15rbt20
export CLIENT_VENV=/B/VENV/libero_plus_client
export CKPT_PATH=/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model
export VLM_MODEL_PATH=/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
unset NUM_TRIALS_PER_TASK

cd /B/SRC/itvlaGpLibPlus
EVAL_MODE=full bash evaluation/LIBERO2/run_eval_libero_std_2server_6client_venv.sh
```

### 2.3 执行过程

**时间线**:
- `15:54:11` — 启动 6 个 server
- `15:55:25-15:55:28` — 全部 6 个 server healthcheck PASS (~74s)
- `15:55:29` — 启动 6 个 client, 开始处理分片

**Suite 完成时间线**:
- `~16:11` — libero_spatial 全部 5 个 shard 完成, 各 client 进入 libero_object
- `~16:39` — libero_object 全部 5 个 shard 完成, 各 client 进入 libero_goal
- `~17:01` — libero_goal 全部 5 个 shard 完成, 各 client 进入 libero_10
- `17:34:30` — 全部 client 完成

**端到端时间**: 约 1 小时 40 分钟 (含 server 启动 ~74s)
**Client 阶段**: 约 1 小时 39 分钟

**分片分配** (round-robin 到 6 个 client):
- 总 shard 数: 20 (4 suite × 5 shard/suite)
- 每个 client: 3-4 个 shard (每 shard 2 tasks × 50 trials = 100 episodes)
- client 0,1: 4 shards (400 episodes); client 2,3,4,5: 3 shards (300 episodes)

**稳定性**:
- 0 个 SIGABRT/SIGSEGV/CRASHED
- 0 个 server 错误
- 0 个缺失结果 JSON
- 全部 40 个 task 的 `error` 字段均为 `null`

### 2.4 全量测试结果

**EVAL_LOG_DIR**: `.../checkpoints/032070/libero_std_2server_6client_20260916155404`

| Suite | Episodes | Successes | SR |
|-------|---:|---:|---:|
| libero_spatial | 500 | 489 | **97.80%** |
| libero_object | 500 | 444 | **88.80%** |
| libero_goal | 500 | 484 | **96.80%** |
| libero_10 | 500 | 488 | **97.60%** |
| **TOTAL** | **2000** | **1905** | **95.25%** |

**Gate 3b 判据**: SR=95.25% ≥ 85% → **PASS** ✅

**crash_task_count**: 0 ✅

### 2.5 逐 Task 详细结果

**libero_spatial** (SR=97.80%):

| task_id | Task Description | Succ/Total | SR |
|---:|:---|---:|---:|
| 0 | pick up the black bowl...place it on the plate table 1 | 47/50 | 94% |
| 1 | ...table 10 | 49/50 | 98% |
| 2 | ...table 12 | 49/50 | 98% |
| 3 | ...table 13 | 50/50 | 100% |
| 4 | ...table 14 | 50/50 | 100% |
| 5 | ...table 15 | 48/50 | 96% |
| 6 | ...table 16 | 49/50 | 98% |
| 7 | ...table 17 | 50/50 | 100% |
| 8 | ...table 18 | 48/50 | 96% |
| 9 | ...table 19 | 49/50 | 98% |

**libero_object** (SR=88.80%):

| task_id | Task Description | Succ/Total | SR |
|---:|:---|---:|---:|
| 0 | pick up alphabet soup...basket table 1 | 43/50 | 86% |
| 1 | ...table 10 | 47/50 | 94% |
| 2 | ...table 12 | 42/50 | 84% |
| 3 | ...table 13 | 42/50 | 84% |
| 4 | ...table 14 | 48/50 | 96% |
| 5 | ...table 15 | 43/50 | 86% |
| 6 | ...table 16 | 43/50 | 86% |
| 7 | ...table 17 | 47/50 | 94% |
| 8 | ...table 18 | 46/50 | 92% |
| 9 | ...table 19 | 43/50 | 86% |

**libero_goal** (SR=96.80%):

| task_id | Task Description | Succ/Total | SR |
|---:|:---|---:|---:|
| 0 | open the middle drawer...cabinet table 1 | 49/50 | 98% |
| 1 | ...table 10 | 50/50 | 100% |
| 2 | ...table 12 | 48/50 | 96% |
| 3 | ...table 13 | 49/50 | 98% |
| 4 | ...table 15 | 49/50 | 98% |
| 5 | ...table 16 | 48/50 | 96% |
| 6 | ...table 17 | 49/50 | 98% |
| 7 | ...table 18 | 48/50 | 96% |
| 8 | ...table 19 | 49/50 | 98% |
| 9 | ...table 2 | 45/50 | 90% |

**libero_10** (SR=97.60%):

| task_id | Task Description | Succ/Total | SR |
|---:|:---|---:|---:|
| 0 | turn on the stove...moka pot table 1 | 50/50 | 100% |
| 1 | ...table 10 | 49/50 | 98% |
| 2 | ...table 12 | 50/50 | 100% |
| 3 | ...table 13 | 49/50 | 98% |
| 4 | ...table 14 | 47/50 | 94% |
| 5 | ...table 15 | 48/50 | 96% |
| 6 | ...table 16 | 48/50 | 96% |
| 7 | ...table 17 | 49/50 | 98% |
| 8 | ...table 18 | 49/50 | 98% |
| 9 | ...table 19 | 49/50 | 98% |

### 2.6 与论文和验收标准的对比

| 指标 | 本次结果 | 验收阈值 (Gate 3b) | 论文参考 |
|------|---:|---:|---:|
| Overall SR | **95.25%** | ≥ 85% | > 95% |
| libero_spatial | 97.80% | — | — |
| libero_object | 88.80% | — | — |
| libero_goal | 96.80% | — | — |
| libero_10 | 97.60% | — | — |
| SIGABRT/Crash | **0** | 0 | — |
| 缺失 JSON | **0** | 0 | — |

**分析**:
- Overall SR 95.25% **超过** 85% 验收阈值, 且与论文的 >95% 一致
- 本 checkpoint 仅训到 60% (step 32070/53450), 能达到 95.25% 说明 F1/F2 修复后模型能力已被正确释放
- `libero_object` (88.80%) 是最弱的 suite, 可能因为抓取 + 放置组合难度更高
- 六 server GPU 分离方案在 2000 episode 中 **零崩溃**, 证明了生产级稳定性

---

## 3. 总结

### 3.1 Gate 通过情况

| Gate | 阈值 | 结果 | 状态 |
|------|------|------|------|
| Gate 3a (冒烟) | SR ≥ 50% | **97.50%** (39/40) | **PASS** ✅ |
| Gate 3b (全量) | SR ≥ 85% | **95.25%** (1905/2000) | **PASS** ✅ |
| 稳定性 | 0 crash | **0 crash / 2040 total episodes** | **PASS** ✅ |

### 3.2 关键路径

- **冒烟 EVAL_LOG_DIR**: `.../032070/libero_std_2server_6client_20260916154739/`
- **全量 EVAL_LOG_DIR**: `.../032070/libero_std_2server_6client_20260916155404/`
- **聚合结果**: `overall_std_results.json` (各 EVAL_LOG_DIR 下)

### 3.3 本次未遇到任何 error

与上一轮 (eval3_optim3_libLOG) 不同, 本次使用六 server GPU 分离方案后:
- **无 SIGABRT** — 之前每次运行都有 8/8 task SIGABRT, 本次 2040 episode 零崩溃
- **无 A24 测试失败** — `_repro_base` 修复仍有效 (预检 137/137 PASS)
- **无 server 错误** — 6 个 server 稳定运行全程
- **无 client 错误** — 6 个 client 全部正常退出

### 3.4 文件清单 (本次新增/修改)

| 文件 | 操作 | 说明 |
|------|------|------|
| `b/d/libplus/eval3_optim3_lib2LOG.md` | **新建** | 本日志文件 |

**无任何代码文件修改** — 所有必要的修复 (F1/F2, A24 _repro_base, SIGABRT handler, 六 server launcher) 均已在前序工作中完成.

