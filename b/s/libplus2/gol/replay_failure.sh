#!/usr/bin/env bash
# replay_failure.sh — Replay a single failed LIBERO-plus Goal evaluation task.
#
# Modes:
#   1. Replay against a running server:
#        bash replay_failure.sh --failure-json <path> --host 127.0.0.1 --port 5784
#
#   2. Replay with auto-started server (starts, runs one task, stops):
#        CKPT_PATH=<path> bash replay_failure.sh --failure-json <path> --start-server --server-gpu 0
#
#   3. Start server only (for manual replays):
#        CKPT_PATH=<path> bash replay_failure.sh --start-server-only --server-gpu 0 --port 5790
#
#   4. Replay by task ID (reads seed/config from eval_log_dir failure JSONs):
#        bash replay_failure.sh --task-id 42 --eval-log-dir <dir> --host 127.0.0.1 --port 5784
#
# Output:
#   --output-dir  (default: /B/Log/<EXPR_NAME>/replay_<TS>/)
#     Contains: actions NPZ, failure video (if still failing), per-step log, failure JSON copy.
#
# Required env (for server start):
#   CKPT_PATH, SERVER_VENV, GOAL4D_DATASET (or defaults from eval_goal_wrapper.sh)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"
EXPR_NAME="${EXPR_NAME:-4dwvlaLbPlusGol0929}"

# --- Argument parsing ---
FAILURE_JSON=""
TASK_ID=""
EVAL_LOG_DIR_ARG=""
HOST="127.0.0.1"
PORT="5784"
START_SERVER=false
START_SERVER_ONLY=false
SERVER_GPU="${SERVER_GPU:-0}"
OUTPUT_DIR=""
SEED=""
SUITE="libero_goal"

while [[ $# -gt 0 ]]; do
    case $1 in
        --failure-json)   FAILURE_JSON="$2"; shift 2 ;;
        --task-id)        TASK_ID="$2"; shift 2 ;;
        --eval-log-dir)   EVAL_LOG_DIR_ARG="$2"; shift 2 ;;
        --host)           HOST="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --start-server)   START_SERVER=true; shift ;;
        --start-server-only) START_SERVER_ONLY=true; START_SERVER=true; shift ;;
        --server-gpu)     SERVER_GPU="$2"; shift 2 ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --suite)          SUITE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# --- Resolve task info from failure JSON or --task-id ---
if [ -n "${FAILURE_JSON}" ]; then
    [ -f "${FAILURE_JSON}" ] || { echo "FATAL: ${FAILURE_JSON} not found" >&2; exit 1; }
    TASK_ID=$(python3 -c "import json; print(json.load(open('${FAILURE_JSON}'))['task_id'])")
    SEED=${SEED:-$(python3 -c "import json; print(json.load(open('${FAILURE_JSON}'))['seed'])")}
    TASK_NAME=$(python3 -c "import json; print(json.load(open('${FAILURE_JSON}')).get('task_name', 'unknown'))")
    echo "Failure JSON: ${FAILURE_JSON}"
    echo "  task_id=${TASK_ID}, seed=${SEED}, task=${TASK_NAME}"
elif [ -n "${TASK_ID}" ]; then
    SEED="${SEED:-7}"
    TASK_NAME="task${TASK_ID}"
    # If eval_log_dir provided, try to find the failure JSON
    if [ -n "${EVAL_LOG_DIR_ARG}" ]; then
        FJ_GLOB="${EVAL_LOG_DIR_ARG}/failures/${SUITE}/task${TASK_ID}_ep*.json"
        FJ_FIRST=$(ls ${FJ_GLOB} 2>/dev/null | head -1 || true)
        if [ -n "${FJ_FIRST}" ]; then
            FAILURE_JSON="${FJ_FIRST}"
            SEED=$(python3 -c "import json; print(json.load(open('${FAILURE_JSON}'))['seed'])")
            TASK_NAME=$(python3 -c "import json; print(json.load(open('${FAILURE_JSON}')).get('task_name', 'unknown'))")
            echo "Found failure JSON: ${FAILURE_JSON}"
        fi
    fi
elif ! ${START_SERVER_ONLY}; then
    echo "FATAL: Provide --failure-json or --task-id" >&2
    exit 1
fi

# --- Environment ---
export GOAL4D_DATASET="${GOAL4D_DATASET:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4D}"
export LIBERO_HOME="${LIBERO_HOME:-/B/SRC/LIBERO-plus}"
export SERVER_VENV="${SERVER_VENV:-/B/VENV/itnvla15rbt20}"
export CLIENT_VENV="${CLIENT_VENV:-/B/VENV/libero_plus_client}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-/B/Log/${EXPR_NAME}/replay_${TS}}"
mkdir -p "${OUTPUT_DIR}"

# --- Server management ---
SERVER_PID=""
cleanup_server() {
    if [ -n "${SERVER_PID}" ]; then
        echo "Stopping replay server (PID=${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup_server EXIT

if ${START_SERVER}; then
    : "${CKPT_PATH:?ERROR: CKPT_PATH required for --start-server}"

    STATS_KEY_MODE="${STATS_KEY_MODE:-panda}"
    ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-panda}"
    INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
    RESIZE_SIZE="${RESIZE_SIZE:-224}"

    echo "Starting inference server on GPU ${SERVER_GPU}, port ${PORT}..."
    (
        source "${SERVER_VENV}/bin/activate"
        cd "${PROJ}"
        export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
        CUDA_VISIBLE_DEVICES=${SERVER_GPU} python evaluation/LIBERO2/policy_server/server_policy.py \
            --ckpt_path "${CKPT_PATH}" \
            --host 0.0.0.0 --port ${PORT} --device cuda \
            --resize_size ${RESIZE_SIZE} \
            --stats_key "${STATS_KEY_MODE}" --robot_type "${ROBOT_TYPE_MODE}" \
            --action_loss_only \
            --inference_backend "${INFERENCE_BACKEND}" \
            --idle_timeout -1 \
            > "${OUTPUT_DIR}/server.log" 2>&1
    ) &
    SERVER_PID=$!
    echo "Server PID=${SERVER_PID}, waiting 90s for startup..."
    sleep 90

    # Healthcheck
    HC_OK=false
    for attempt in 1 2 3; do
        if (
            source "${SERVER_VENV}/bin/activate"
            cd "${PROJ}"
            export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
            python3 -c "
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
c = WebsocketClientPolicy(host='${HOST}', port=${PORT})
m = c.get_server_metadata()
print('Server OK:', m.get('model_name', 'unknown'))
" 2>/dev/null
        ); then
            HC_OK=true
            break
        fi
        echo "Healthcheck attempt ${attempt} failed, retrying in 15s..."
        sleep 15
    done

    if ! ${HC_OK}; then
        echo "FATAL: Server healthcheck failed after 3 attempts" >&2
        echo "Server log:"
        tail -20 "${OUTPUT_DIR}/server.log"
        exit 1
    fi

    if ${START_SERVER_ONLY}; then
        echo "Server running on ${HOST}:${PORT} (GPU ${SERVER_GPU}, PID ${SERVER_PID})"
        echo "Press Ctrl+C to stop."
        trap - EXIT
        wait "${SERVER_PID}"
        exit 0
    fi
fi

# --- Run single-task replay ---
echo ""
echo "============================================================"
echo " Replaying task_id=${TASK_ID} (${TASK_NAME})"
echo "   seed=${SEED}, suite=${SUITE}"
echo "   server=${HOST}:${PORT}"
echo "   output=${OUTPUT_DIR}"
echo "============================================================"

TASK_CLS="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
LIBERO_CONFIG_DIR="${OUTPUT_DIR}/libero_config"
LP_BENCH="${LIBERO_HOME}/libero/libero"
mkdir -p "${LIBERO_CONFIG_DIR}"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LP_BENCH}
bddl_files: ${LP_BENCH}/bddl_files
init_states: ${LP_BENCH}/init_files
datasets: ${LP_BENCH}/../datasets
assets: ${LP_BENCH}/assets
YAML

export GOAL4D_CONTRACT="${GOAL4D_DATASET}/meta/goal_train_eval_contract.json"
export ROTATE_IMAGES=false
export STATS_KEY_MODE=panda
export ROBOT_TYPE_MODE=panda
export INFERENCE_BACKEND=standard
export RENDER_BACKEND="${RENDER_BACKEND:-egl}"
export PROJ EVAL_LOG_DIR="${OUTPUT_DIR}"
unset DISABLE_KEYPOINTS

START_IDX="${TASK_ID}"
END_IDX=$((TASK_ID + 1))

MUJOCO_EGL_DEVICE_ID="${REPLAY_EGL_GPU:-${SERVER_GPU}}"
if [ "${RENDER_BACKEND}" = "osmesa" ]; then
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
    export GALLIUM_DRIVER=llvmpipe
    unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
else
    export MUJOCO_EGL_DEVICE_ID
    if [ -d "${CLIENT_VENV}/egl_vendor.d" ]; then
        export __EGL_VENDOR_LIBRARY_DIRS="${CLIENT_VENV}/egl_vendor.d"
    fi
fi

(
    source "${CLIENT_VENV}/bin/activate"
    cd "${PROJ}"
    export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
    export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

    python "${HERE}/eval_goal_plus.py" -- \
        --host "${HOST}" --port "${PORT}" \
        --task_suite_name "${SUITE}" \
        --num_trials_per_task 1 \
        --num_steps_wait 10 \
        --seed "${SEED}" \
        --replan_steps 8 \
        --start_idx "${START_IDX}" --end_idx "${END_IDX}" \
        --task_classification_path "${TASK_CLS}" \
        --eval_log_dir "${OUTPUT_DIR}" \
        --gripper_convention libero_native \
        --enable_keypoints \
        --save_actions \
        --save_failure_videos \
        > "${OUTPUT_DIR}/replay.log" 2>&1
)
REPLAY_EXIT=$?

echo ""
if [ ${REPLAY_EXIT} -eq 0 ]; then
    echo "Replay completed (exit=0)"
else
    echo "Replay finished with errors (exit=${REPLAY_EXIT})"
fi

# --- Copy failure JSON for reference ---
if [ -n "${FAILURE_JSON}" ] && [ -f "${FAILURE_JSON}" ]; then
    cp "${FAILURE_JSON}" "${OUTPUT_DIR}/original_failure.json"
fi

# --- Show replay results ---
echo ""
echo "=== Replay Results ==="
echo "Output dir: ${OUTPUT_DIR}"

# Check if task succeeded this time
RESULT_JSON=$(ls "${OUTPUT_DIR}/logs/${SUITE}/"*.json 2>/dev/null | head -1 || true)
if [ -n "${RESULT_JSON}" ]; then
    python3 -c "
import json
r = json.load(open('${RESULT_JSON}'))
for cat, data in r.items():
    s = data.get('success_count', 0)
    t = data.get('total_count', 0)
    print(f'  {cat}: {s}/{t}')
" 2>/dev/null || true
fi

# Check for action NPZ
ACTION_NPZ=$(ls "${OUTPUT_DIR}/actions/${SUITE}/"*.npz 2>/dev/null | head -1 || true)
if [ -n "${ACTION_NPZ}" ]; then
    echo "Action NPZ: ${ACTION_NPZ}"
    python3 -c "
import numpy as np
d = np.load('${ACTION_NPZ}', allow_pickle=True)
print(f'  steps={d[\"actions\"].shape[0]}, success={d[\"success\"]}')
" 2>/dev/null || true
fi

# Check for failure video
FAIL_VID=$(ls "${OUTPUT_DIR}/videos/${SUITE}/"*failure*.mp4 2>/dev/null | head -1 || true)
if [ -n "${FAIL_VID}" ]; then
    echo "Failure video: ${FAIL_VID}"
fi

# Check for new failure JSON
NEW_FAIL=$(ls "${OUTPUT_DIR}/failures/${SUITE}/"*.json 2>/dev/null | head -1 || true)
if [ -n "${NEW_FAIL}" ]; then
    echo "Still failing: ${NEW_FAIL}"
else
    echo "Task SUCCEEDED on replay!"
fi

echo ""
echo "Full log: ${OUTPUT_DIR}/replay.log"
