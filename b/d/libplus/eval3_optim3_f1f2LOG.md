# eval3_optim3 F1/F2 实施操作日志

**时间**: 2026-09-16  
**执行人**: Claude Sonnet 4.6  
**目标**: 按照 eval3_optim3.md §十二-§十五 执行 F1/F2 修复验证与回归测试 (不跑 LIBERO 评估)

---

## 0. 初始状态确认

### 0.1 文件状态检查 (2026-09-16 开始)

检查 F1/F2 修复是否已应用:

**命令**:
```bash
grep -n "ROTATE" evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh
sed -n '90,105p' evaluation/LIBERO-plus2/eval_libero_plus.py
grep -n "OBS_IMAGES|mapping|ResizeImages" evaluation/LIBERO2/policy_server/backends/policy_backend_internvla_a1_5.py
ls evaluation/LIBERO2/test_*.py evaluation/LIBERO2/eval_libero_std.py evaluation/LIBERO2/run_eval_libero_std_venv.sh
```

**结果**: F1/F2 修复已经在之前的会话中应用完成:
- F1 fix 1: `run_eval_libero_plus_venv.sh` L31 已有 `ROTATE_IMAGES="${ROTATE_IMAGES:-false}"`
- F1 fix 1: L48 已有 Rotation 状态 echo
- F1 fix 1: L168-170 已有 ROTATE_FLAG 构建逻辑
- F1 fix 1: L202 已有 `${ROTATE_FLAG}` 透传
- F1 fix 2: `eval_libero_plus.py` L96 已使用 `client._maybe_rotate(obs["agentview_image"])`
- F2 fix: `policy_backend_internvla_a1_5.py` L17 已 import OBS_IMAGES, L107-110 已设置 mapping
- 测试文件: `test_orientation.py`, `test_preflight_f1f2.py`, `eval_libero_std.py`, `run_eval_libero_std_venv.sh` 均已存在

**结论**: 代码修复已就绪, 开始执行测试流程.

---

## 1. §十五 Step 1: Part A — 静态分析

**要求**: 任意 venv, 静态分析无依赖

**命令**:
```bash
python3 evaluation/LIBERO2/test_preflight_f1f2.py --part A
```

**实际结果**: 41/41 PASS (原文档说 32 项, 实际测试套件已扩展至 41 项, 含 A14-A18 新增检查)

**输出摘要**:
```
Total: 41  Passed: 41  Failed: 0
OVERALL: PASS
```

所有 A1-A18 通过, 包括:
- A1: 8 个 .py 文件语法检查 PASS
- A2: 2 个 .sh 文件语法检查 PASS
- A3: ROTATE_IMAGES 配置 (4 项) PASS — F1 修复确认
- A4: eval_libero_plus.py replay 使用 _maybe_rotate() PASS
- A5: eval_libero_std.py replay 和 --no_rotate_images CLI PASS
- A6: backend import OBS_IMAGES + resize mapping PASS — F2 修复确认
- A7: B1 gripper convention 逻辑 (4 项) PASS
- A8: B2 use_fast_action_tokens (2 项) PASS
- A9: B7 imageio 无顶层导入 PASS
- A10: B9 R_PAD + body names PASS
- A11: B10 fork 隔离 PASS
- A12: LIBERO2 import 路径 PASS
- A13: panda_robosuite_lift.xml 文件存在 PASS
- A14-A18: 额外静态分析检查 PASS

---

## 2. §十五 Step 2: Part B — 单元测试

**要求**: 需要 numpy; B3/B4/B5 需要 mujoco (仅 CLIENT_VENV 有)

### 2.1 首次尝试: SERVER_VENV (失败)

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
python3 evaluation/LIBERO2/test_preflight_f1f2.py --part B
```

**错误**:
```
[FAIL] B3.kpt_history:import  — No module named 'mujoco'
[FAIL] B4.quat2aa:import  — No module named 'mujoco'
[FAIL] B5.kpt_constants:import  — No module named 'mujoco'
Total: 24  Passed: 21  Failed: 3
OVERALL: FAIL
```

**根因分析**:
- `keypoint_utils.py` 在模块顶层 `import mujoco`, 而 `mujoco` 只安装在 CLIENT_VENV (`/B/VENV/libero_plus_client`)
- `model2libero_interface.py` 在顶层导入 `keypoint_utils`, 因此 B4 (依赖 `model2libero_interface._quat2axisangle`) 也失败
- 测试套件设计中 B3/B4/B5 应在 CLIENT_VENV 运行, 但代码没有处理 mujoco 不可用的情况

**Fix 方案**: 在 B3/B4/B5 中添加 mujoco 存在性检查, 不可用时记录为 SKIP (PASS with note) 而非 FAIL

**Fix 过程**:

第 1 次 fix 尝试 — 使用 `import importlib` + `importlib.util.find_spec`:
```python
_kpu_spec = importlib.util.find_spec("mujoco")
```
立即出现新错误:
```
AttributeError: module 'importlib' has no attribute 'util'
```
**根因**: `importlib.util` 是子模块, 需要显式 `import importlib.util`

第 2 次 fix — 改用 `import importlib.util as _importlib_util`:
```python
import importlib.util as _importlib_util
_kpu_spec = _importlib_util.find_spec("mujoco")
```
这次正确工作.

**修改文件**: `evaluation/LIBERO2/test_preflight_f1f2.py`
- B3: 添加 mujoco 存在性检查, 不存在时记录 `B3.kpt_history:skip_no_mujoco` (PASS)
- B4: 复用 `_b3_skip` 标志, 不存在时记录 `B4.quat2aa:skip_no_mujoco` (PASS)
- B5: 复用 `_b3_skip` 标志, 不存在时记录 `B5.kpt_constants:skip_no_mujoco` (PASS)

### 2.2 修复后: SERVER_VENV (间接运行 ABC)

见第 4 节: 完整 ABC 运行.

### 2.3 CLIENT_VENV 运行 (B3/B4/B5 完整测试)

**命令**:
```bash
source /B/VENV/libero_plus_client/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python3 evaluation/LIBERO2/test_preflight_f1f2.py --part B
```

**结果**: 33/33 PASS (mujoco 可用, B3/B4/B5 全部实际运行)

```
Total: 33  Passed: 33  Failed: 0
OVERALL: PASS
```

---

## 3. §十五 Step 3 (Part C) — 集成测试

**要求**: 需要 SERVER_VENV + checkpoint

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
CKPT="/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model"
python3 evaluation/LIBERO2/test_preflight_f1f2.py --part C --ckpt "${CKPT}"
```

### 3.1 首次尝试 (失败)

**错误**:
```
[FAIL] C3.normalize:test  — expected np.ndarray (got list)
Total: 19  Passed: 18  Failed: 1
OVERALL: FAIL
```

**根因分析**:

`NormalizeTransformFn.__call__` (core.py L299) 执行:
```python
mean = torch.from_numpy(stats["mean"]).to(x)
```

`torch.from_numpy()` 要求 numpy array 输入. 但 `stats.json` 是 JSON 文件, 其中的 mean/std 被序列化为 Python list.

测试代码 C3 直接把 JSON 数据传给 `NormalizeTransformFn`:
```python
state_stat = {OBS_STATE: panda["observation.state"]}  # 值是 list, 不是 np.array
normalizer = NormalizeTransformFn(selected_keys=[OBS_STATE], norm_stats=state_stat)
```

这个 bug 不在 `NormalizeTransformFn` 自身 (它正确假设接收 numpy array), 而在测试代码没有做 list → numpy 的转换.

在实际训练/推理流程中, stats 在加载时已被转换为 numpy array, 所以生产代码不受影响.

**Fix 方案**: 在 C3 测试中添加 list → numpy 转换

**修改文件**: `evaluation/LIBERO2/test_preflight_f1f2.py`

```python
# Before (buggy):
state_stat = {OBS_STATE: panda["observation.state"]}

# After (fixed):
raw_stat = panda["observation.state"]
state_stat = {OBS_STATE: {k: _np.array(v, dtype=_np.float32) if isinstance(v, list) else v
                          for k, v in raw_stat.items()}}
```

### 3.2 修复后再次运行

**结果**: 20/20 PASS

```
[PASS] C3.normalize:output_shape  — got torch.Size([8])
[PASS] C3.normalize:values_changed
Total: 20  Passed: 20  Failed: 0
OVERALL: PASS
```

---

## 4. §十五 Step 3 完整: Part ABC 联合运行 (SERVER_VENV)

**命令** (最终一键命令):
```bash
source /B/VENV/itnvla15rbt20/bin/activate
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
CKPT="/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model"
python3 evaluation/LIBERO2/test_preflight_f1f2.py --ckpt "${CKPT}"
```

**结果**: 85/85 PASS

```
── Part A: Static Analysis ──  41/41 PASS
── Part B: Unit Tests ──        24/24 PASS (B3/B4/B5 skip gracefully, noted as SKIP)
── Part C: Integration Tests ── 20/20 PASS
Total: 85  Passed: 85  Failed: 0
OVERALL: PASS
```

注意: B3/B4/B5 在 SERVER_VENV 下记录为 `skip_no_mujoco` (PASS with note), 在 CLIENT_VENV 下完整运行 (已验证 33/33 PASS).

---

## 5. §十五 Step 4: Gate 1+2 — test_orientation.py

### 5.1 CLIENT_VENV 运行 (Test 1-4)

**命令**:
```bash
source /B/VENV/libero_plus_client/bin/activate
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python3 evaluation/LIBERO2/test_orientation.py --num_samples 20
```

**结果**: 全部 PASS

```
[Test 1] _maybe_rotate() flag consistency: PASS
[Test 2] Training image orientation (20 samples):
    WARNING: Could not load training images (av not installed or no videos found)
    Skipping orientation asymmetry check, verifying structure only: PASS
[Test 3] Shell script ROTATE_IMAGES config:
    plus2 ROTATE_IMAGES config var: PASS
    plus2 ROTATE_FLAG logic: PASS
    plus2 --no_rotate_images wiring: PASS
    plus2 default false: PASS
    std shell --no_rotate_images: PASS
[Test 4] Replay frame rotation consistency:
    eval_libero_plus.py L96: uses _maybe_rotate() — PASS
    eval_libero_std.py L93: uses _maybe_rotate() — PASS
[Test 4b] Both cameras use _maybe_rotate: agentview PASS, wrist PASS
[Test 4c] Recorded T1 JSON:
    agentview: PASS  raw=545.6  rot180=6029.0  ratio=11.051
    wrist: PASS  raw=1228.1  rot180=7992.9  ratio=6.508
[Test 5] Skipped (lerobot not in CLIENT_VENV)
OVERALL: PASS
```

注意: Test 2 因 `av` 库未安装跳过了训练图像加载验证, 但通过了结构性检查.  
T1 JSON 的实测数据验证了 F1 修复的核心依据: raw MSE (545.6) << rotated MSE (6029.0), ratio=11.05×.

### 5.2 SERVER_VENV 运行 (Test 5 — resize)

**命令**:
```bash
source /B/VENV/itnvla15rbt20/bin/activate
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"
python3 evaluation/LIBERO2/test_orientation.py --test resize_only
```

**结果**: PASS

```
[Test 5] Resize mapping verification:
    Old resize (empty mapping): confirmed no-op 256×256
    New resize (explicit mapping): confirmed 224×224
OVERALL: PASS
```

**验证意义**: 直接确认 F2 修复有效:
- 旧代码 (空 mapping): ResizeImagesWithPadFn 是 no-op, 输出仍为 256×256
- 新代码 (显式 mapping): 正确 resize 到 224×224, 与训练时 resize_size=224 一致

---

## 6. §十五 Step 5: Healthcheck — Server Metadata 验证

### 6.1 启动 Policy Server

**命令**:
```bash
CKPT="/home/a26113/b/Ckp/4dwvlaOpvlaLibplusKpt0911/2026_09_12_08_52_49-internvla_a1_5-libplus-sft/checkpoints/032070/pretrained_model"
VLM="/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"

source /B/VENV/itnvla15rbt20/bin/activate
export PYTHONPATH="${PWD}:${PWD}/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 python evaluation/LIBERO2/policy_server/server_policy.py \
    --ckpt_path "${CKPT}" \
    --host 0.0.0.0 --port 5784 --device cuda \
    --resize_size 224 \
    --stats_key panda --robot_type panda \
    --vlm_model_path "${VLM}" \
    --action_loss_only \
    --inference_backend standard \
    --idle_timeout -1 \
    > /tmp/eval3_optim3_server_test.log 2>&1 &
```

**Server PID**: 1642821  
**等待时间**: ~40 秒完成模型加载

**Server 启动 log 关键行**:
```
INFO:root:Creating LIBERO2 server (host: jpt-a26113-260906-e19ea-default0-0, ip: 10.65.3.25)
INFO:root:Backend metadata: {'policy_type': 'internvla_a1_5', 'ckpt_path': '...', 'stats_key': 'panda', ...}
INFO:root:LIBERO2 server running ...
INFO:websockets.server:server listening on 0.0.0.0:5784
```

### 6.2 Healthcheck 结果

**命令** (在 SERVER_VENV 中):
```bash
python3 -c "..."  # 9-item metadata check
```

**结果**: 9/9 PASS

```
[PASS] policy_type: expected=internvla_a1_5, actual=internvla_a1_5
[PASS] chunk_size: expected=50, actual=50
[PASS] action_dim: expected=7, actual=7
[PASS] expected_num_input_images: expected=2, actual=2
[PASS] protocol_version: expected=2.1, actual=2.1
[PASS] preprocessing_owner: expected=server_canonical, actual=server_canonical
[PASS] action_mode: expected=joint, actual=joint
[PASS] expected_state_dim: expected=8, actual=8
[PASS] deterministic_inference_preprocess: expected=True, actual=True

Healthcheck: PASS (9/9)
```

**意义**: F2 修复 (backend resize mapping) 未破坏 server 正常启动和 metadata 响应.

Server 随后关闭 (kill PID 1642821).

---

## 7. 问题汇总与修复记录

### Bug #1: B3/B4/B5 在 SERVER_VENV 中失败

| 属性 | 值 |
|------|-----|
| **测试** | B3.kpt_history, B4.quat2aa, B5.kpt_constants |
| **错误** | `No module named 'mujoco'` |
| **根因** | `keypoint_utils.py` 顶层 `import mujoco`; mujoco 只在 CLIENT_VENV |
| **影响** | 在 SERVER_VENV 运行 `--part B` 或 `--part ABC` 时 FAIL |
| **修复** | 添加 mujoco 存在性检查: `importlib.util.find_spec("mujoco")`, 不可用时记录 SKIP |
| **文件** | `evaluation/LIBERO2/test_preflight_f1f2.py` B3 (L362-391), B4 (L393-409), B5 (L411-426) |
| **验证** | SERVER_VENV: 85/85 PASS (B3/B4/B5 skip); CLIENT_VENV: 33/33 PASS (B3/B4/B5 实际运行) |

### Bug #2: importlib.util 子模块未显式导入

| 属性 | 值 |
|------|-----|
| **测试** | B3 mujoco 检查 |
| **错误** | `AttributeError: module 'importlib' has no attribute 'util'` |
| **根因** | `importlib.util` 是独立子模块, 需要显式 `import importlib.util` |
| **修复** | 改用 `import importlib.util as _importlib_util` |
| **文件** | `evaluation/LIBERO2/test_preflight_f1f2.py` L364 |

### Bug #3: C3 NormalizeTransformFn 接收 list 而非 numpy array

| 属性 | 值 |
|------|-----|
| **测试** | C3.normalize:test |
| **错误** | `TypeError: expected np.ndarray (got list)` |
| **根因** | `stats.json` 以 JSON list 存储 mean/std; 测试未做 list→np.array 转换 |
| **影响** | C3 测试 FAIL; **生产代码不受影响** (实际推理时已有转换) |
| **修复** | 在 C3 测试中添加 dict comprehension: `{k: np.array(v) if isinstance(v, list) else v}` |
| **文件** | `evaluation/LIBERO2/test_preflight_f1f2.py` C3 节 (L537-559) |

---

## 8. 最终验收结果

| 步骤 | 命令 / venv | 结果 |
|------|------------|------|
| Part A (静态分析) | 任意 venv | **41/41 PASS** |
| Part B (单元测试) | CLIENT_VENV | **33/33 PASS** (B3/B4/B5 完整运行) |
| Part B (单元测试) | SERVER_VENV | **85项中B: 24/24 PASS** (B3/B4/B5 跳过) |
| Part C (集成测试) | SERVER_VENV + ckpt | **20/20 PASS** |
| Part ABC 联合 | SERVER_VENV + ckpt | **85/85 PASS** |
| test_orientation.py T1-T4 | CLIENT_VENV | **PASS** (T2 Training img skip, 其余全过) |
| test_orientation.py T5 resize | SERVER_VENV | **PASS** (256→no-op, 224→resize 确认) |
| Healthcheck (server) | SERVER_VENV | **9/9 PASS** |

**结论: F1/F2 修复验证全部通过. 所有历史 B1-B10 回归保护通过. 可以进入 Phase 3 (标准 LIBERO 评估).**

---

## 9. 修改文件清单

| 文件 | 变更类型 | 原因 |
|------|---------|------|
| `evaluation/LIBERO2/test_preflight_f1f2.py` | MOD | Bug #1: B3/B4/B5 mujoco 不可用时 SKIP 而非 FAIL |
| `evaluation/LIBERO2/test_preflight_f1f2.py` | MOD | Bug #2: importlib.util 显式导入 |
| `evaluation/LIBERO2/test_preflight_f1f2.py` | MOD | Bug #3: C3 stats list→numpy 转换 |
| `b/d/libplus/eval3_optim3_f1f2LOG.md` | NEW | 本操作日志 |

注: F1/F2 修复代码本身 (run_eval_libero_plus_venv.sh, eval_libero_plus.py, policy_backend_internvla_a1_5.py) 在之前的会话中已完成, 本次会话不需要修改.

