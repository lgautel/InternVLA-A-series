# grperr_1.2 Implementation Log

> Started: 2026-09-18
> Implements: [grperr_1.2.md](grperr_1.2.md) Phase 0 (zero-risk offline) + Phase 2 code (gripper modes)

---

## 1. Files Modified

| File | Change | Reason |
|:---|:---|:---|
| `b/x/4dwvla_ext/franky_joint_env.py` | Fix `or 0.04` bug; add gripper modes; add delta-w function; add `_last_gripper_closed` | 0c, Phase 2 |
| `b/x/4dwvla_ext/fk_keypoints.py` | Add `append()` / `snapshot()` API | 0b |
| `b/x/4dwvla_ext/vla_inference_server.py` | Protocol v2 branching; full chunk logging | 0a, 0b |
| `b/x/4dwvla_ext/franka_vla_client.py` | Record state_before (not state_after); add `protocol: 2` field | 0b |
| `b/x/4dwvla_ext/franky_controller_direct.py` | Add `gripper_move_width_m()` delegation | Phase 2 |
| `b/x/franky_ext/franka_libfranka_gripper.py` | Add `move_width_m()` method | Phase 2 |
| `b/x/4dwvla_ext/configs/franka_plug_eval.env` | Add gripper mode config variables | Phase 2 config |

## 2. Files Created

| File | Purpose | Reason |
|:---|:---|:---|
| `b/x/4dwvla_ext/distribution_monitor.py` | Online per-channel OOR monitoring vs training [q01,q99] | 0f |
| `b/x/4dwvla_ext/tests/test_gripper_semantics_offline.py` | T1 ramp identity + T2 delta-w + T3 continuous mapping | Tests |
| `b/x/4dwvla_ext/tests/test_obs_state_offline.py` | T6 gripper width observation correctness | Tests |
| `b/x/4dwvla_ext/tests/test_fk_matches_dataset.py` | T5 FK vs dataset keypoint alignment | 0d |
| `b/x/4dwvla_ext/tests/accept_demo_replay.py` | T9 demo replay A/B acceptance test | 0e |

## 3. Files Updated (tests)

| File | Change |
|:---|:---|
| `b/x/4dwvla_ext/tests/test_fk_keypoints_offline.py` | Added `test_append_snapshot_api()` (T_FK.6) and `test_protocol_v2_his_len_semantics()` (T_FK.7) |

---

## 4. Detailed Change Log

### 4.1 Fix 0c: Gripper state false-value bug

**File**: `b/x/4dwvla_ext/franky_joint_env.py`, method `_get_observation()`

**Before**:
```python
g = state["gripper_width"] or 0.04
```

**After**:
```python
g = state["gripper_width"]
if g is None:
    raise RuntimeError(
        "gripper width unavailable; refusing to fabricate observation state"
    )
return {"state": np.concatenate([q, [float(g)]])}
```

**Dummy branch**: Changed `[0.04]` to `[0.078]` (demo episode start open-mode, q90=0.0784).

**Root cause**: Python `or` treats `0.0` as falsy. `0.0 or 0.04 == 0.04`. When gripper fully closes (width=0.0), state was incorrectly reported as 0.04 m — falling in the valley between the bimodal training distribution, corrupting model input.

**Impact**: This bug was dormant because the gripper never actually closed (the absorbing state prevented it). Once the execution semantics fix (Phase 2) enables closing, this bug would have broken the feedback loop at the ramp endpoint.

### 4.2 Fix 0a: Full chunk logging

**File**: `b/x/4dwvla_ext/vla_inference_server.py`

Added logging of the full 50-step chunk's gripper channel immediately after inference, before slicing to `n_exec`:

```python
full_chunk = unnormalize_fn(
    {ACTION: action_pred[:, :actual_action_dim]}
)[ACTION]
full_grip = full_chunk.detach().float().cpu().numpy()[:, -1]
logger.info(
    "[request %d] full_chunk_grip=%s (executing first %d)",
    request_index, format_array(full_grip), n_exec,
)
```

**Purpose**: This is the cheapest judgment point — if slots 20-40 already show `a >= 0.5`, P0-b (n_exec truncation) is the co-dominant cause and Phase 3 should be prioritized. If the entire 50-step chunk stays < 0.3, P0-a (execution semantics) is the sole cause.

### 4.3 Fix 0b: his_len semantics

**Three files changed, one logical fix.**

#### 4.3.1 `fk_keypoints.py` — Added `append()` and `snapshot()` API

```python
def append(self, arm_q7: np.ndarray) -> int:
    """Record a frame into history without treating it as the current frame."""
    self._history.append(self.compute(arm_q7))
    return len(self._history)

def snapshot(self) -> tuple[np.ndarray, int]:
    """Pack history as-is: training's his_kpts excludes the current frame."""
    return self._pack()
```

`step()` is preserved as `append() + snapshot()` for backward compat.

#### 4.3.2 `vla_inference_server.py` — Protocol versioning

```python
proto = msg.get("protocol", 1)
if proto >= 2:
    for executed_q in executed_history:
        fk_computer.append(np.asarray(executed_q, dtype=np.float32))
    kpt_data = fk_computer.snapshot()
else:
    # Old behavior: step() for each history + step() for current
    for executed_q in executed_history:
        fk_computer.step(np.asarray(executed_q, dtype=np.float32))
    kpt_data = fk_computer.step(arm_q)
```

Protocol v1 (old clients): keep existing behavior — `step()` includes current frame in history.
Protocol v2 (new clients): `append()` only past frames, `snapshot()` excludes current frame.

#### 4.3.3 `franka_vla_client.py` — Record state_before, add protocol field

Changed: state recording moved from **after** `env.step()` to **before** it, and from `state_after[:7]` to `state_before[:7]`. This ensures history contains frames strictly earlier than the current inference frame, matching training semantics.

Added `"protocol": 2` to the IPC message.

**Corrected progression**: Step 0 → his_len=0, Step 10 → his_len=10, Step 20 → his_len=20.
**Old (buggy) progression**: Step 0 → his_len=1, Step 10 → his_len=12, Step 20 → his_len=23.

### 4.4 Fix 0f: Distribution monitor

**File**: `b/x/4dwvla_ext/distribution_monitor.py` (new)

Read-only module that loads training stats from `abs_stats.json` and compares per-step 8D state/action against [q01, q99]. Logs warnings only on OOR, provides `episode_summary()` table with per-channel OOR counts, percentages, and max |z|.

### 4.5 Phase 2: Gripper mode changes

#### `franky_joint_env.py` — Three execution modes

Added module-level config variables (all environment-variable driven):
- `VLA_GRIPPER_MODE` — `binary_abs` (default/backward compat) | `binary_delta` | `continuous`
- `W0 = 0.08` — training normalization constant
- `GRIPPER_DEADBAND_M`, `GRIPPER_CLOSE_DELTA_M`, `GRIPPER_OPEN_DELTA_M`, `GRASP_HANDOFF_A`, `GRIPPER_MAX_WIDTH_M`

Added `want_gripper_close_delta()` function implementing the delta-w criterion with hysteresis (grperr_1.2.md §4.2).

Replaced the gripper execution block in `step()` with a three-way branch:
- **`binary_abs`**: Original behavior — `want_gripper_close(a)` checks `a >= 0.5`.
- **`binary_delta`**: Uses `want_gripper_close_delta(a, w_meas, currently_closed)` with hysteresis state.
- **`continuous`**: Computes `w_cmd = W0 * (1 - a)`, clamped to `GRIPPER_MAX_WIDTH_M`. When `a >= GRASP_HANDOFF_A` (0.8), switches to force-control `close_gripper()`. Otherwise, position-controls via `move_width_m()` when outside deadband.

#### `franka_libfranka_gripper.py` — Added `move_width_m()`

```python
def move_width_m(self, width_m: float, speed: float = 0.05) -> None:
    w = max(0.0, min(float(width_m), _MAX_WIDTH_M))
    self._call("move_width", self._gripper.move, w, _close_speed_m_s(speed))
```

Unlike `move()`, takes SI meters directly and does not refuse widening (ramp tracking needs both directions).

#### `franky_controller_direct.py` — Added `gripper_move_width_m()` delegation

```python
def gripper_move_width_m(self, width_m: float, speed: float = 0.05) -> None:
    if self._gripper is None:
        return
    fn = getattr(self._gripper, "move_width_m", None)
    if not callable(fn):
        raise RuntimeError("gripper does not support move_width_m()")
    fn(width_m, speed)
```

#### `configs/franka_plug_eval.env` — Full config update

Added all gripper mode variables with defaults matching backward compat (`VLA_GRIPPER_MODE=binary_abs`). Commented out caliper-dependent values (`FRANKA_CUBE_WIDTH_M`, `FRANKA_HOLD_TOL_M`, `FRANKA_GRIPPER_MAX_WIDTH_M`, `FRANKA_DYNAMICS_FACTOR`) with explicit "fill after measurement" notes.

---

## 5. Test Results

### 5.1 Host-runnable tests (no GPU/robot needed)

| Test | File | Count | Result |
|:---|:---|---:|:---:|
| T1 Ramp identity | `test_gripper_semantics_offline.py` | 34 | ALL PASS |
| T2 Delta-w criterion | `test_gripper_semantics_offline.py` | 4 | ALL PASS |
| T3 Continuous mapping | `test_gripper_semantics_offline.py` | 7 | ALL PASS |
| T6 Observation state | `test_obs_state_offline.py` | 16 | ALL PASS |
| T7 IPC round-trip | `test_ipc_offline.py` | 12 | ALL PASS |
| T8 Safety regression | `test_safety_offline.py` | 36 | ALL PASS |
| **Total** | | **109** | **ALL PASS** |

### 5.2 GPU-container tests (syntax-verified, not executed)

| Test | File | Status | Reason |
|:---|:---|:---|:---|
| T4 his_len semantics | `test_fk_keypoints_offline.py` | Syntax OK | Needs `pytorch_kinematics` + `torch` |
| T5 FK alignment | `test_fk_matches_dataset.py` | Syntax OK | Needs `pytorch_kinematics` + `torch` |
| T9 Demo replay A/B | `accept_demo_replay.py` | Syntax OK | Needs GPU container + model checkpoint |

### 5.3 Distribution monitor (manual verification)

Tested against `abs_stats.json` with:
- HOME state → all channels in range (correct)
- q7=0.4 → flagged as below q01, z=-3.10 (correct)
- Typical action → all channels in range (correct)
- Episode summary table renders correctly

---

## 6. Errors Encountered and Fixes

### E1: `self._controller.gripper.move_width_m()` — AttributeError path

**Error**: The gripper mode agent wrote `self._controller.gripper.move_width_m(w_cmd)`, but `FrankyControllerDirect` does not expose `self._gripper` as a public `gripper` attribute.

**Root cause**: The controller uses delegation pattern — all gripper operations go through wrapper methods (`close_gripper()`, `open_gripper()`, `gripper_width()`, etc.), not direct attribute access.

**Fix**: Added `gripper_move_width_m()` delegation method to `FrankyControllerDirect`, and changed the call to `self._controller.gripper_move_width_m(w_cmd)`.

### E2: FK test requires GPU container

**Error**: `ModuleNotFoundError: No module named 'torch'` when running `test_fk_keypoints_offline.py` on host.

**Root cause**: `fk_keypoints.py` imports `torch` and `pytorch_kinematics` at module level. These are only available in the GPU container's `4dwvla` venv.

**Status**: Expected. Tests T4, T5 are syntax-verified and will run in the GPU container.

### E3: `.pyc` permission error on syntax check

**Error**: `PermissionError` when `py_compile` tried to write `.pyc` cache for `franka_libfranka_gripper.py`.

**Root cause**: The `__pycache__` directory under `franky_ext/` has restricted permissions.

**Status**: Not a real error. AST parse confirms syntax is valid.

---

## 7. What Remains (Not Done Here)

### Phase 1: Physical measurements — COMPLETED
- 1a: Plug thickness = **10 mm** → `FRANKA_CUBE_WIDTH_M=0.010`, `FRANKA_HOLD_TOL_M=0.008` (window [2mm, 18mm])
- 1b: Max-open finger gap = **80 mm** → `FRANKA_GRIPPER_MAX_WIDTH_M=0.080`
  - **Q1 answered**: 满开 80 mm 与示教 79.4 mm 基本一致。之前启动日志报 66 mm 是标定/上报问题，不是物理差异。
- 1c: Try `franka_gripper` homing/recalibration — pending

### Phase 2: Real-robot validation
- Switch `VLA_GRIPPER_MODE` from `binary_abs` to `continuous` or `binary_delta`
- Run Level 0 → Level 2 validation per §15.7

### Phase 3: chunk length (n_exec 10 → 20/50)
- Blocked on 0a results (need to see full chunk grip values first)

### Phase 4: Motion characteristics
- 4a: `FRANKA_DYNAMICS_FACTOR` configurability
- 4b: Batched waypoint motion
- 4c: L2a/L2b safety layer split (action vs observation distribution)

### GPU-container test execution
- T4: `test_fk_keypoints_offline.py` (new append/snapshot/protocol-v2 tests)
- T5: `test_fk_matches_dataset.py` (FK vs dataset alignment)
- T9: `accept_demo_replay.py` (demo replay A/B)

---

## 8. Summary of All Operations

```
Modified:  7 files
Created:   5 files
Updated:   1 test file
Tests run: 109 passed, 0 failed (host-runnable)
Tests created but not runnable on host: 3 (need GPU container)
Errors encountered: 3 (all resolved)
```
