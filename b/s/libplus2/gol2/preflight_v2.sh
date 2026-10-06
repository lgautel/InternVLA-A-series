#!/usr/bin/env bash
set -euo pipefail
# Preflight for v2 evaluation
# Validates contract schema 2, state_dim=14, robot_type=libero_goal_4dv2

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
GOAL4D_DATASET="${GOAL4D_DATASET:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2}"
LIBERO_HOME="${LIBERO_HOME:-/B/SRC/LIBERO-plus}"
SERVER_VENV="${SERVER_VENV:-/B/VENV/itnvla15rbt20}"
CLIENT_VENV="${CLIENT_VENV:-/B/VENV/libero_plus_client}"

PASS=0; FAIL=0
check() {
    local desc="$1"; shift
    if "$@" > /dev/null 2>&1; then
        echo "  OK: ${desc}"; PASS=$((PASS + 1))
    else
        echo "  FAIL: ${desc}"; FAIL=$((FAIL + 1))
    fi
}

echo "=== Preflight v2: Checkpoint ==="
check "config.json" test -f "${CKPT_PATH}/config.json"
check "model.safetensors" test -f "${CKPT_PATH}/model.safetensors"
check "stats.json" test -f "${CKPT_PATH}/stats.json"
check "train_config.json" test -f "${CKPT_PATH}/train_config.json"

echo "=== Preflight v2: Contract (schema 2) ==="
CONTRACT="${GOAL4D_DATASET}/meta/goal_train_eval_contract.json"
check "contract exists" test -f "${CONTRACT}"
check "contract schema=2, robot_type=libero_goal_4dv2, state_dim=14" python3 -c "
import json
c = json.load(open('${CONTRACT}'))
assert c['schema'] == 'goal_train_eval_contract/2', f'schema={c[\"schema\"]}'
assert c['image_orientation'] == 'raw'
assert c['robot_type'] == 'libero_goal_4dv2', f'robot_type={c[\"robot_type\"]}'
assert c['state_dim'] == 14, f'state_dim={c[\"state_dim\"]}'
assert c['stats_key'] == 'libero_goal_4dv2'
assert c['num_keypoints'] == 8
assert c['r_pad'] > 1.8
"

echo "=== Preflight v2: Checkpoint state dim ==="
check "ckpt stats state dim=14" python3 -c "
import json
s = json.load(open('${CKPT_PATH}/stats.json'))
# stats.json may nest under robot_type key (e.g. 'libero_goal_4dv2')
if 'observation.state' not in s:
    s = next(iter(s.values()))
assert len(s['observation.state']['mean']) == 14, f'state dim={len(s[\"observation.state\"][\"mean\"])}'
"
check "ckpt config correct" python3 -c "
import json
cfg = json.load(open('${CKPT_PATH}/config.json'))
assert cfg.get('enable_keypoint_predictor') == True
assert cfg.get('tokenize_state') == True
assert cfg.get('num_keypoint_joints') == 8
assert cfg.get('kpt_4d_mode') == 'pos_rot'
assert cfg.get('chunk_size') == 50
khl = cfg.get('keypoint_history_max_len')
assert khl is not None and khl > 0
"

echo "=== Preflight v2: MJCF MD5 ==="
GOL_DIR="${HERE}/../gol"
MJCF_PATH="${GOL_DIR}/assets/panda_goal_table.xml"
check "MJCF exists" test -f "${MJCF_PATH}"
EXPECTED_MD5=$(python3 -c "import json; print(json.load(open('${CONTRACT}'))['eval_mjcf_md5'])" 2>/dev/null || echo "UNKNOWN")
ACTUAL_MD5=$(md5sum "${MJCF_PATH}" 2>/dev/null | cut -d' ' -f1 || echo "NONE")
check "MJCF MD5 matches contract" test "${ACTUAL_MD5}" = "${EXPECTED_MD5}"

echo "=== Preflight v2: Server/Client Venvs ==="
check "server python" test -f "${SERVER_VENV}/bin/python"
check "server torch cuda" "${SERVER_VENV}/bin/python" -c "import torch; assert torch.cuda.is_available()"
check "client python" test -f "${CLIENT_VENV}/bin/python"
check "client mujoco+robosuite" env __EGL_VENDOR_LIBRARY_DIRS="${CLIENT_VENV}/egl_vendor.d" MUJOCO_GL=egl "${CLIENT_VENV}/bin/python" -c "import mujoco; import robosuite"
check "EGL vendor" test -f "${CLIENT_VENV}/egl_vendor.d/10_nvidia.json"

echo "=== Preflight v2: NumPy Patches ==="
ENV_WRAPPER="${LIBERO_HOME}/libero/libero/envs/env_wrapper.py"
check "np.frombuffer patch" grep -q 'np.frombuffer' "${ENV_WRAPPER}"
check "np.float64 patch" grep -q 'np.float64' "${ENV_WRAPPER}"

echo "=== Preflight v2: Evaluation Scripts ==="
check "eval_libero_plus.py" test -f "${PROJ}/evaluation/LIBERO-plus2/eval_libero_plus.py"
check "run_eval_libero_plus_venv.sh" test -f "${PROJ}/evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"
check "goal_client.py v2 branch" python3 -c "
src = open('${GOL_DIR}/goal_client.py').read()
assert 'goal_train_eval_contract/2' in src, 'goal_client.py missing v2 dispatch'
assert 'pack_state_v2' in src, 'goal_client.py missing pack_state_v2'
"
check "run_eval_goal_plus.sh v2 support" grep -q 'goal_train_eval_contract/2' "${GOL_DIR}/run_eval_goal_plus.sh"

echo "=== Preflight v2: Task Config ==="
TASK_CLS="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
check "task_classification.json" test -f "${TASK_CLS}"
check "libero_goal >2500 tasks" python3 -c "
import json
d = json.load(open('${TASK_CLS}'))
g = d['libero_goal']
n = sum(len(v) if isinstance(v, list) else 1 for v in g.values()) if isinstance(g, dict) else len(g)
assert n > 2500
"

echo ""
echo "Result: ${PASS} passed, ${FAIL} failed"
if [ ${FAIL} -gt 0 ]; then echo "PREFLIGHT FAILED"; exit 1; fi
echo "PREFLIGHT v2 PASSED"
