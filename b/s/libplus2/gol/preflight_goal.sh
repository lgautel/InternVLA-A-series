#!/usr/bin/env bash
# preflight_goal.sh — LIBERO-plus Goal evaluation 预检清单
# 非交互, 有任何关键失败立即 exit 1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
GOAL4D_DATASET="${GOAL4D_DATASET:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4D}"
LIBERO_HOME="${LIBERO_HOME:-/B/SRC/LIBERO-plus}"
SERVER_VENV="${SERVER_VENV:-/B/VENV/itnvla15rbt20}"
CLIENT_VENV="${CLIENT_VENV:-/B/VENV/libero_plus_client}"

PASS=0
FAIL=0
check() {
    local desc="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  ✅ ${desc}"
        PASS=$((PASS + 1))
    else
        echo "  ❌ ${desc}"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Preflight: Checkpoint Files ==="
check "config.json exists" test -f "${CKPT_PATH}/config.json"
check "model.safetensors exists" test -f "${CKPT_PATH}/model.safetensors"
check "stats.json exists" test -f "${CKPT_PATH}/stats.json"
check "train_config.json exists" test -f "${CKPT_PATH}/train_config.json"

echo "=== Preflight: Contract ==="
CONTRACT="${GOAL4D_DATASET}/meta/goal_train_eval_contract.json"
check "contract file exists" test -f "${CONTRACT}"
check "contract schema valid" python3 -c "
import json
c = json.load(open('${CONTRACT}'))
assert c['schema'] == 'goal_train_eval_contract/1', f'schema={c[\"schema\"]}'
assert c['image_orientation'] == 'raw'
assert c['robot_type'] == 'panda'
assert c['num_keypoints'] == 8
assert c['r_pad'] > 1.8
"

echo "=== Preflight: MJCF MD5 ==="
MJCF_PATH="${HERE}/assets/panda_goal_table.xml"
check "MJCF exists" test -f "${MJCF_PATH}"
EXPECTED_MD5=$(python3 -c "import json; print(json.load(open('${CONTRACT}'))['eval_mjcf_md5'])" 2>/dev/null || echo "UNKNOWN")
ACTUAL_MD5=$(md5sum "${MJCF_PATH}" 2>/dev/null | cut -d' ' -f1 || echo "NONE")
check "MJCF MD5 matches contract (${ACTUAL_MD5})" test "${ACTUAL_MD5}" = "${EXPECTED_MD5}"

echo "=== Preflight: Checkpoint Config Consistency ==="
check "checkpoint config correct" python3 -c "
import json
cfg = json.load(open('${CKPT_PATH}/config.json'))
assert cfg.get('enable_keypoint_predictor') == True, 'enable_keypoint_predictor must be True'
assert cfg.get('tokenize_state') == True, 'tokenize_state must be True'
assert cfg.get('num_keypoint_joints') == 8, f'num_keypoint_joints={cfg.get(\"num_keypoint_joints\")}'
assert cfg.get('kpt_4d_mode') == 'pos_rot', f'kpt_4d_mode={cfg.get(\"kpt_4d_mode\")}'
assert cfg.get('chunk_size') == 50, f'chunk_size={cfg.get(\"chunk_size\")}'
khl = cfg.get('keypoint_history_max_len')
assert khl is not None and khl > 0, f'keypoint_history_max_len={khl}'
"

echo "=== Preflight: Server Venv ==="
check "server python exists" test -f "${SERVER_VENV}/bin/python"
check "server torch cuda available" "${SERVER_VENV}/bin/python" -c "import torch; assert torch.cuda.is_available()"
check "server transformers available" "${SERVER_VENV}/bin/python" -c "import transformers"

echo "=== Preflight: Client Venv ==="
check "client python exists" test -f "${CLIENT_VENV}/bin/python"
check "client mujoco+robosuite available" "${CLIENT_VENV}/bin/python" -c "import mujoco; import robosuite"

echo "=== Preflight: EGL Vendor ==="
check "EGL vendor JSON exists" test -f "${CLIENT_VENV}/egl_vendor.d/10_nvidia.json"

echo "=== Preflight: NumPy Patches (B3+B4) ==="
ENV_WRAPPER="${LIBERO_HOME}/libero/libero/envs/env_wrapper.py"
check "np.frombuffer patch (B3)" grep -q 'np.frombuffer' "${ENV_WRAPPER}"
check "np.float64 patch (B4)" grep -q 'np.float64' "${ENV_WRAPPER}"

echo "=== Preflight: LIBERO-plus Task Config ==="
TASK_CLS="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
check "task_classification.json exists" test -f "${TASK_CLS}"
check "libero_goal has >2500 tasks" python3 -c "
import json
d = json.load(open('${TASK_CLS}'))
g = d['libero_goal']
n = sum(len(v) if isinstance(v, list) else 1 for v in g.values()) if isinstance(g, dict) else len(g)
assert n > 2500, f'expected >2500, got {n}'
"

echo "=== Preflight: Evaluation Scripts ==="
check "eval_libero_plus.py" test -f "${PROJ}/evaluation/LIBERO-plus2/eval_libero_plus.py"
check "run_eval_libero_plus_venv.sh" test -f "${PROJ}/evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"
check "server_policy.py" test -f "${PROJ}/evaluation/LIBERO2/policy_server/server_policy.py"
check "aggregate_results.py" test -f "${PROJ}/evaluation/LIBERO-plus2/aggregate_results.py"

echo "=== Preflight: Goal Extension Scripts ==="
check "eval_goal_plus.py" test -f "${HERE}/eval_goal_plus.py"
check "goal_client.py" test -f "${HERE}/goal_client.py"
check "run_eval_goal_plus.sh" test -f "${HERE}/run_eval_goal_plus.sh"
check "contract.py" test -f "${HERE}/contract.py"
check "load_contract.py" test -f "${HERE}/load_contract.py"
check "history.py" test -f "${HERE}/history.py"

echo "=== Preflight: VLM Weights ==="
check "Qwen3.5-2B cached" test -d "/B/VENV/hf_home/hub/models--Qwen--Qwen3.5-2B"

echo ""
echo "Result: ${PASS} passed, ${FAIL} failed"
if [ ${FAIL} -gt 0 ]; then
    echo "PREFLIGHT FAILED"
    exit 1
fi
echo "PREFLIGHT PASSED"
