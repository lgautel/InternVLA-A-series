#!/usr/bin/env bash
# test_eval_goal.sh — LIBERO-plus Goal 评估验收测试
# 验证 preflight, contract loader, history buffer, FK, 以及 DRY_RUN 派生脚本
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
GOAL4D_DATASET="${GOAL4D_DATASET:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4D}"
LIBERO_HOME="${LIBERO_HOME:-/B/SRC/LIBERO-plus}"
SERVER_VENV="${SERVER_VENV:-/B/VENV/itnvla15rbt20}"
CLIENT_VENV="${CLIENT_VENV:-/B/VENV/libero_plus_client}"

echo "========================================"
echo " Test Suite: LIBERO-plus Goal Eval"
echo " CKPT_PATH: ${CKPT_PATH}"
echo "========================================"

PASS=0
FAIL=0
tcheck() {
    local desc="$1"
    shift
    if "$@" 2>&1 | tail -5; then
        echo "  ✅ ${desc}"
        PASS=$((PASS + 1))
    else
        echo "  ❌ ${desc}"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "=== T1: Preflight ==="
tcheck "preflight passes" bash "${HERE}/preflight_goal.sh"

echo ""
echo "=== T2: Contract Loader ==="
tcheck "load_contract valid" python3 -c "
import sys; sys.path.insert(0, '${HERE}')
from load_contract import load_goal_contract
c = load_goal_contract('${GOAL4D_DATASET}/meta/goal_train_eval_contract.json')
assert c['schema'] == 'goal_train_eval_contract/1', f'bad schema: {c[\"schema\"]}'
assert c['r_pad'] > 1.8, f'bad r_pad: {c[\"r_pad\"]}'
assert c['num_keypoints'] == 8
assert c['kpt_4d_mode'] == 'pos_rot'
assert c['image_orientation'] == 'raw'
print('Contract loader OK')
"

echo ""
echo "=== T3: KeypointHistory buffer ==="
tcheck "history semantics" python3 -c "
import sys, numpy as np; sys.path.insert(0, '${HERE}')
from history import KeypointHistory

h = KeypointHistory(max_len=92, num_kpts=8, kpt_dim=7)
# T3a: initial state
assert h.his_len == 0, f'initial his_len={h.his_len}'

# T3b: push one frame
kpt = np.random.randn(8, 7).astype(np.float32)
h.push(kpt)
assert h.his_len == 1
buf, ln = h.get_history()
assert buf.shape == (92, 8, 7), f'shape={buf.shape}'
assert ln == 1
assert np.allclose(buf[0], kpt), 'first slot mismatch'
assert np.allclose(buf[1:], 0), 'padding not zero'

# T3c: fill to capacity
h.reset()
for i in range(100):
    h.push(np.full((8, 7), float(i), dtype=np.float32))
assert h.his_len == 92, f'after 100 pushes, his_len={h.his_len}'
buf, ln = h.get_history()
assert buf[0, 0, 0] == 8.0, f'oldest should be 8, got {buf[0,0,0]}'
assert buf[91, 0, 0] == 99.0, f'newest should be 99, got {buf[91,0,0]}'

# T3d: reset clears
h.reset()
assert h.his_len == 0
print('KeypointHistory OK')
"

echo ""
echo "=== T4: pack_like_training ==="
tcheck "pack semantics" python3 -c "
import sys, numpy as np; sys.path.insert(0, '${HERE}')
from history import pack_like_training

T, J, D = 100, 8, 7
kpts = np.random.randn(T, J, D).astype(np.float32)

# T4a: t=0, no history
d = pack_like_training(kpts, t=0, history_len=92, chunk_size=50)
assert d['his_len'] == 0, f't=0 his_len={d[\"his_len\"]}'
assert d['kpt_t'].shape == (J, D)
assert d['kpt_future'].shape == (50, J, D)

# T4b: t=50, 50 frames history
d = pack_like_training(kpts, t=50, history_len=92, chunk_size=50)
assert d['his_len'] == 50
assert np.allclose(d['his_kpts'][:50], kpts[:50])
assert np.allclose(d['his_kpts'][50:], 0), 'padding should be zero'

# T4c: t=99 (last), future clamps to last frame
d = pack_like_training(kpts, t=99, history_len=92, chunk_size=50)
for i in range(50):
    assert np.allclose(d['kpt_future'][i], kpts[99]), f'future[{i}] should clamp to last'
print('pack_like_training OK')
"

echo ""
echo "=== T5: MJCF FK consistency ==="
MUJOCO_PYTHON="${CLIENT_VENV}/bin/python"
[ -f "${MUJOCO_PYTHON}" ] || MUJOCO_PYTHON="${SERVER_VENV}/bin/python"
tcheck "FK produces valid keypoints" "${MUJOCO_PYTHON}" -c "
import sys; sys.path.insert(0, '${HERE}')
import numpy as np, mujoco
from contract import GOAL_MJCF, KEYPOINT_NAMES

model = mujoco.MjModel.from_xml_path(str(GOAL_MJCF))
data = mujoco.MjData(model)
data.qpos[:7] = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
data.qpos[7:9] = [0.04, 0.04]
mujoco.mj_forward(model, data)

# MJCF uses robot0_ prefix for arm links; gripper0_eef has no robot0_ prefix
for name in KEYPOINT_NAMES:
    candidates = [name, f'robot0_{name}']
    bid = -1
    for c in candidates:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, c)
        if bid >= 0:
            break
    assert bid >= 0, f'body {name} not found (tried {candidates})'
    pos = data.xpos[bid]
    assert np.all(np.isfinite(pos)), f'{name} pos not finite: {pos}'
    assert np.linalg.norm(pos) < 5.0, f'{name} pos too large: {pos}'

# Check robot0_link1 is near Goal base (-0.66, 0, 0.912)
link1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0_link1')
pos1 = data.xpos[link1_id]
assert abs(pos1[0] - (-0.66)) < 0.05, f'link1 x={pos1[0]}, expected ~-0.66'
# link1 is above pedestal base (0.912m), typically at ~1.2-1.3m
assert pos1[2] > 0.9 and pos1[2] < 1.5, f'link1 z={pos1[2]}, expected 0.9-1.5'
print('FK consistency OK')
"

echo ""
echo "=== T6: DRY_RUN derived script ==="
tcheck "wrapper derives correctly" bash -c "
export CKPT_PATH='${CKPT_PATH}'
export GOAL4D_DATASET='${GOAL4D_DATASET}'
export LIBERO_HOME='${LIBERO_HOME}'
export SERVER_VENV='${SERVER_VENV}'
export CLIENT_VENV='${CLIENT_VENV}'
DRY_RUN=1 bash '${HERE}/run_eval_goal_plus.sh' 2>&1 | tail -3
"

echo ""
echo "=== T7: Checkpoint keypoint_history_max_len ==="
tcheck "kpt_hist_len from config" python3 -c "
import json
cfg = json.load(open('${CKPT_PATH}/config.json'))
khl = cfg.get('keypoint_history_max_len')
assert khl is not None and khl > 0, f'keypoint_history_max_len={khl}'
assert khl == 92, f'expected 92, got {khl}'
print(f'keypoint_history_max_len={khl} OK')
"

echo ""
echo "=== T8: Image orientation contract ==="
tcheck "orientation=raw" python3 -c "
import json
c = json.load(open('${GOAL4D_DATASET}/meta/goal_train_eval_contract.json'))
assert c['image_orientation'] == 'raw', f'got {c[\"image_orientation\"]}'
print('Image orientation OK')
"

echo ""
echo "=== T9: Stats key structure ==="
tcheck "stats.json has panda key" python3 -c "
import json
s = json.load(open('${CKPT_PATH}/stats.json'))
assert 'panda' in s, f'keys: {list(s.keys())}'
p = s['panda']
assert 'action' in p, f'panda keys: {list(p.keys())}'
a = p['action']
assert 'mean' in a and 'std' in a, f'action keys: {list(a.keys())}'
assert len(a['mean']) == 7, f'action dim={len(a[\"mean\"])}'
print('Stats structure OK')
"

echo ""
echo "========================================"
echo " Result: ${PASS} passed, ${FAIL} failed"
echo "========================================"
if [ ${FAIL} -gt 0 ]; then
    echo "TEST SUITE FAILED"
    exit 1
fi
echo "TEST SUITE PASSED"
