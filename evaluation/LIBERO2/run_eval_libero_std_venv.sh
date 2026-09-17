#!/usr/bin/env bash
set -euo pipefail

# ─── Standard LIBERO baseline evaluation ───
# Reuses LIBERO2 server + LIBERO2/eval_libero_std.py client
# Validates in-distribution performance after F1/F2 fixes

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
: "${LIBERO_HOME:?ERROR: LIBERO_HOME not set}"
: "${SERVER_VENV:?ERROR: SERVER_VENV not set}"
: "${CLIENT_VENV:?ERROR: CLIENT_VENV not set}"

PROJ="${PROJ:-$(cd "$(dirname "$0")/../.." && pwd)}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5784}"
STATS_KEY_MODE="${STATS_KEY_MODE:-panda}"
ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-panda}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
SEED="${SEED:-7}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-90}"
ROTATE_IMAGES="${ROTATE_IMAGES:-false}"
# Render backend: auto | egl | osmesa
#   egl    = NVIDIA GPU, fast, but subject to MuJoCo EGL SIGABRT in forked children
#   osmesa = Mesa llvmpipe CPU, immune to EGL SIGABRT, ~slower (needs libosmesa6)
RENDER_BACKEND="${RENDER_BACKEND:-auto}"

# ─── Eval mode: smoke (fast) or full ───
EVAL_MODE="${EVAL_MODE:-smoke}"  # smoke | full
if [ "${EVAL_MODE}" = "smoke" ]; then
    NUM_TRIALS=5
    END_IDX=2           # first 2 tasks per suite
else
    NUM_TRIALS=50
    END_IDX=-1          # all tasks
fi

# SAVE_FAILURE_VIDEOS: set to "true" to save MP4 for failed episodes (default off)
SAVE_FAILURE_VIDEOS="${SAVE_FAILURE_VIDEOS:-false}"

SUITES="libero_spatial libero_object libero_goal libero_10"
# Output lives under the checkpoint step dir: .../checkpoints/<step>/libero_std_YYYYMMDDHHMM/
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_PATH}/../libero_std_$(date +%Y%m%d%H%M)}"

echo "============================================"
echo "Standard LIBERO Baseline Evaluation"
echo "  Mode: ${EVAL_MODE}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  GPU: ${GPU_ID}"
echo "  Trials/task: ${NUM_TRIALS}"
echo "  End idx: ${END_IDX}"
echo "  Rotation: $([ "${ROTATE_IMAGES}" = "true" ] && echo "ENABLED (180°)" || echo "DISABLED (raw)")"
echo "  Render backend: ${RENDER_BACKEND}"
echo "  Output: ${EVAL_LOG_DIR}"
echo "============================================"

mkdir -p "${EVAL_LOG_DIR}"

# ─── LIBERO config.yaml ───
LIBERO_CONFIG_DIR="${EVAL_LOG_DIR}/libero_config"
mkdir -p "${LIBERO_CONFIG_DIR}"
LP_BENCH="${LIBERO_HOME}/libero/libero"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LP_BENCH}
bddl_files: ${LP_BENCH}/bddl_files
init_states: ${LP_BENCH}/init_files
datasets: ${LP_BENCH}/../datasets
assets: ${LP_BENCH}/assets
YAML

# ─── Start Server ───
(
    source "${SERVER_VENV}/bin/activate"
    cd "${PROJ}"
    export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/LIBERO2/policy_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --host 0.0.0.0 --port ${PORT} --device cuda \
        --resize_size ${RESIZE_SIZE} \
        --stats_key "${STATS_KEY_MODE}" --robot_type "${ROBOT_TYPE_MODE}" \
        ${VLM_MODEL_PATH:+--vlm_model_path "${VLM_MODEL_PATH}"} \
        ${ACTION_LOSS_ONLY_FLAG} \
        --inference_backend "${INFERENCE_BACKEND}" \
        --idle_timeout -1 \
        > "${EVAL_LOG_DIR}/server.log" 2>&1
) &
SERVER_PID=$!
echo "Server PID=${SERVER_PID}, waiting ${SERVER_STARTUP_WAIT}s..."
sleep ${SERVER_STARTUP_WAIT}

cleanup() {
    echo "Shutting down server (PID=${SERVER_PID})..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ─── Healthcheck ───
HC_OK=0
for attempt in 1 2 3; do
    if (
        source "${SERVER_VENV}/bin/activate"
        cd "${PROJ}"
        export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
        python3 -c "
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy
c = WebsocketClientPolicy(host='127.0.0.1', port=${PORT})
m = c.get_server_metadata()
assert m.get('action_mode') == 'joint', f'action_mode={m.get(\"action_mode\")}'
assert m.get('preprocessing_owner') == 'server_canonical'
assert m.get('stats_key') == '${STATS_KEY_MODE}', f'stats_key={m.get(\"stats_key\")}'
assert int(m.get('resize_size') or 0) == int(${RESIZE_SIZE}), f'resize_size={m.get(\"resize_size\")}'
print('HEALTHCHECK PASSED')
"
    ) >> "${EVAL_LOG_DIR}/healthcheck.log" 2>&1; then
        HC_OK=1
        break
    fi
    echo "Healthcheck attempt ${attempt} failed, retrying in 30s..."
    sleep 30
done

if [ ${HC_OK} -eq 0 ]; then
    echo "FATAL: healthcheck failed after 3 attempts"
    exit 1
fi
echo "Server ready."

ROTATE_FLAG=""
if [ "${ROTATE_IMAGES}" = "true" ]; then
    ROTATE_FLAG="--rotate_images"
fi

FAILURE_VIDEO_FLAG=""
if [ "${SAVE_FAILURE_VIDEOS}" = "true" ]; then
    FAILURE_VIDEO_FLAG="--save_failure_videos"
fi

# ─── Run eval for each suite ───
TOTAL_SUCC=0
TOTAL_EP=0

for suite in ${SUITES}; do
    echo "[$(date)] Starting ${suite}..."
    set +e
    (
        source "${CLIENT_VENV}/bin/activate"
        cd "${PROJ}"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
        # eval_libero_std.py resolves the backend via render_backend.py; these
        # exports only provide the per-backend extras it cannot infer.
        export RENDER_BACKEND="${RENDER_BACKEND}"
        if [ "${RENDER_BACKEND}" = "osmesa" ]; then
            export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
            export GALLIUM_DRIVER=llvmpipe
            unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
        else
            export MUJOCO_EGL_DEVICE_ID=0
            if [ -d "${CLIENT_VENV}/egl_vendor.d" ]; then
                export __EGL_VENDOR_LIBRARY_DIRS="${CLIENT_VENV}/egl_vendor.d"
            fi
        fi
        export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
        export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
        python evaluation/LIBERO2/eval_libero_std.py \
            --host 127.0.0.1 --port ${PORT} \
            --task_suite_name "${suite}" \
            --num_trials_per_task ${NUM_TRIALS} \
            --seed ${SEED} \
            --replan_steps ${REPLAN_STEPS} \
            --end_idx ${END_IDX} \
            --eval_log_dir "${EVAL_LOG_DIR}" \
            ${ROTATE_FLAG} \
            --gripper_convention libero_native \
            --enable_keypoints \
            ${FAILURE_VIDEO_FLAG} \
            > "${EVAL_LOG_DIR}/${suite}.log" 2>&1
    )
    EXIT=$?
    set -e

    if [ ${EXIT} -ne 0 ]; then
        echo "  ${suite}: FAILED (exit=${EXIT})"
    else
        JSON_FILE=$(ls -t "${EVAL_LOG_DIR}/logs/${suite}"/std_*.json 2>/dev/null | head -1)
        if [ -n "${JSON_FILE}" ]; then
            SR=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(f\"{100*d['overall_sr']:.2f}\")")
            SUCC=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(d['total_successes'])")
            EP=$(python3 -c "import json; d=json.load(open('${JSON_FILE}')); print(d['total_episodes'])")
            echo "  ${suite}: SR=${SR}% (${SUCC}/${EP})"
            TOTAL_SUCC=$((TOTAL_SUCC + SUCC))
            TOTAL_EP=$((TOTAL_EP + EP))
        fi
    fi
done

# ─── Summary ───
echo ""
echo "============================================"
if [ ${TOTAL_EP} -gt 0 ]; then
    OVERALL_SR=$(python3 -c "print(f'{100.0*${TOTAL_SUCC}/${TOTAL_EP}:.2f}')")
    echo "Standard LIBERO ${EVAL_MODE}: SR=${OVERALL_SR}% (${TOTAL_SUCC}/${TOTAL_EP})"
    if [ "${EVAL_MODE}" = "smoke" ]; then
        THRESHOLD=50
    else
        THRESHOLD=85
    fi
    PASS=$(python3 -c "print('PASS' if ${TOTAL_SUCC}/${TOTAL_EP} >= ${THRESHOLD}/100 else 'FAIL')")
    echo "Gate 3: ${PASS} (threshold=${THRESHOLD}%)"
else
    echo "No results collected."
    PASS="FAIL"
fi
echo "Results: ${EVAL_LOG_DIR}"
echo "============================================"

[ "${PASS}" = "PASS" ] && exit 0 || exit 1
