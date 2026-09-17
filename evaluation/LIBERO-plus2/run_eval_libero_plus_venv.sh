#!/usr/bin/env bash
set -euo pipefail

# ─── Required parameters ───
: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
: "${LIBERO_HOME:?ERROR: LIBERO_HOME not set}"
: "${SERVER_VENV:?ERROR: SERVER_VENV not set}"
: "${CLIENT_VENV:?ERROR: CLIENT_VENV not set}"

# ─── Optional parameters with defaults ───
PROJ="${PROJ:-$(cd "$(dirname "$0")/../.." && pwd)}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-5784}"
SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-8}"
STATS_KEY_MODE="${STATS_KEY_MODE:-panda}"
ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-panda}"
GRIPPER_CONVENTION="${GRIPPER_CONVENTION:-libero_native}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
SAVE_ACTIONS_FLAG="${SAVE_ACTIONS_FLAG:---save_actions}"
# SAVE_FAILURE_VIDEOS: set to "true" to save MP4 for failed episodes (default off; JSON always saved)
SAVE_FAILURE_VIDEOS="${SAVE_FAILURE_VIDEOS:-false}"
# Output lives under the checkpoint step dir: .../checkpoints/<step>/libero_plus_YYYYMMDDHHMM/
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_PATH}/../libero_plus_$(date +%Y%m%d%H%M)}"
CATEGORIES="${CATEGORIES:-}"
DISABLE_KEYPOINTS="${DISABLE_KEYPOINTS:-}"
ROTATE_IMAGES="${ROTATE_IMAGES:-false}"
SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-90}"
# Render backend: auto | egl | osmesa
#   egl    = NVIDIA GPU, fast, but subject to MuJoCo EGL SIGABRT in forked children
#   osmesa = Mesa llvmpipe CPU, immune to EGL SIGABRT, ~slower (needs libosmesa6)
RENDER_BACKEND="${RENDER_BACKEND:-auto}"

TASK_CLS="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
SUITES="libero_spatial libero_object libero_goal libero_10"
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================"
echo "LIBERO-plus2 Evaluation"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  GPUs: ${GPU_IDS} (${NUM_GPUS} total)"
echo "  Output: ${EVAL_LOG_DIR}"
echo "  Stats key: ${STATS_KEY_MODE}"
echo "  Robot type: ${ROBOT_TYPE_MODE}"
echo "  Gripper: ${GRIPPER_CONVENTION}"
echo "  Keypoints: $([ -z "${DISABLE_KEYPOINTS}" ] && echo "ENABLED" || echo "DISABLED")"
echo "  Rotation: $([ "${ROTATE_IMAGES}" = "true" ] && echo "ENABLED (180°)" || echo "DISABLED (raw)")"
echo "  Render backend: ${RENDER_BACKEND} (${NUM_GPUS} concurrent client workers)"
echo "============================================"

# ─── LIBERO config.yaml (flat format, B8) ───
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

# ─── Build task queue ───
QUEUE_FILE="${EVAL_LOG_DIR}/task_queue.txt"
mkdir -p "${EVAL_LOG_DIR}"
> "${QUEUE_FILE}"
for suite in ${SUITES}; do
    TOTAL=$(python3 -c "import json; d=json.load(open('${TASK_CLS}')); s=d['${suite}']; print(len(s) if isinstance(s,list) else sum(len(v) for v in s.values()))")
    SHARD_SIZE=$(( (TOTAL + SHARDS_PER_SUITE - 1) / SHARDS_PER_SUITE ))
    for s in $(seq 0 $((SHARDS_PER_SUITE - 1))); do
        START=$((s * SHARD_SIZE))
        END=$(( (s+1) * SHARD_SIZE ))
        [ ${END} -gt ${TOTAL} ] && END=${TOTAL}
        [ ${START} -ge ${END} ] && continue
        echo "${suite} ${START} ${END}" >> "${QUEUE_FILE}"
    done
done
echo "Task queue: $(wc -l < "${QUEUE_FILE}") shards"

# ─── Pre-assign shards to GPUs (round-robin) ───
for i in $(seq 0 $((NUM_GPUS - 1))); do
    > "${EVAL_LOG_DIR}/gpu${i}_queue.txt"
done
SHARD_IDX=0
while IFS=' ' read -r SUITE START END; do
    GPU_IDX=$((SHARD_IDX % NUM_GPUS))
    echo "${SUITE} ${START} ${END}" >> "${EVAL_LOG_DIR}/gpu${GPU_IDX}_queue.txt"
    SHARD_IDX=$((SHARD_IDX + 1))
done < "${QUEUE_FILE}"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "GPU ${GPU_ARRAY[$i]}: $(wc -l < "${EVAL_LOG_DIR}/gpu${i}_queue.txt") shards"
done

# ─── GPU worker: start server once, process all assigned shards sequentially ───
gpu_worker() {
    local GPU_IDX=$1
    local SLOT_IDX=$2
    local PORT=$((BASE_PORT + SLOT_IDX))
    local WORKER_DIR="${EVAL_LOG_DIR}/worker_gpu${GPU_IDX}"
    local GPU_QUEUE="${EVAL_LOG_DIR}/gpu${SLOT_IDX}_queue.txt"
    mkdir -p "${WORKER_DIR}"

    if [ ! -s "${GPU_QUEUE}" ]; then
        echo "[$(date)] GPU ${GPU_IDX}: No shards assigned, skipping" | tee -a "${WORKER_DIR}/worker.log"
        return 0
    fi

    # ─── Start Server (once for all shards) ───
    (
        source "${SERVER_VENV}/bin/activate"
        cd "${PROJ}"
        export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
        CUDA_VISIBLE_DEVICES=${GPU_IDX} python evaluation/LIBERO2/policy_server/server_policy.py \
            --ckpt_path "${CKPT_PATH}" \
            --host 0.0.0.0 --port ${PORT} --device cuda \
            --resize_size ${RESIZE_SIZE} \
            --stats_key "${STATS_KEY_MODE}" --robot_type "${ROBOT_TYPE_MODE}" \
            ${VLM_MODEL_PATH:+--vlm_model_path "${VLM_MODEL_PATH}"} \
            ${ACTION_LOSS_ONLY_FLAG} \
            --inference_backend "${INFERENCE_BACKEND}" \
            --idle_timeout -1 \
            > "${WORKER_DIR}/server.log" 2>&1
    ) &
    local SERVER_PID=$!
    echo "[$(date)] GPU ${GPU_IDX}: Server PID=${SERVER_PID}, port=${PORT}, waiting ${SERVER_STARTUP_WAIT}s..." | tee -a "${WORKER_DIR}/worker.log"
    sleep ${SERVER_STARTUP_WAIT}

    # ─── Healthcheck ───
    local HC_OK=0
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
print('HEALTHCHECK PASSED (gpu${GPU_IDX}, port ${PORT})')
"
        ) >> "${WORKER_DIR}/worker.log" 2>&1; then
            HC_OK=1
            break
        fi
        echo "[$(date)] GPU ${GPU_IDX}: Healthcheck attempt ${attempt} failed, retrying in 30s..." | tee -a "${WORKER_DIR}/worker.log"
        sleep 30
    done

    if [ ${HC_OK} -eq 0 ]; then
        echo "FATAL: GPU ${GPU_IDX} healthcheck failed after 3 attempts" | tee -a "${WORKER_DIR}/worker.log"
        kill ${SERVER_PID} 2>/dev/null || true
        return 1
    fi

    # ─── Process all assigned shards sequentially ───
    local KPT_FLAGS=""
    if [ -z "${DISABLE_KEYPOINTS}" ]; then
        KPT_FLAGS="--enable_keypoints"
    else
        KPT_FLAGS="--no-enable_keypoints"
    fi
    local CAT_FLAG=""
    if [ -n "${CATEGORIES}" ]; then
        CAT_FLAG="--categories ${CATEGORIES}"
    fi
    local ROTATE_FLAG=""
    if [ "${ROTATE_IMAGES}" = "true" ]; then
        ROTATE_FLAG="--rotate_images"
    fi
    local FAILURE_VIDEO_FLAG=""
    if [ "${SAVE_FAILURE_VIDEOS}" = "true" ]; then
        FAILURE_VIDEO_FLAG="--save_failure_videos"
    fi

    local SHARD_NUM=0
    while IFS=' ' read -r SUITE START END; do
        SHARD_NUM=$((SHARD_NUM + 1))
        echo "[$(date)] GPU ${GPU_IDX}: Starting shard ${SHARD_NUM} — ${SUITE}[${START}:${END}]" | tee -a "${WORKER_DIR}/worker.log"

        set +e
        (
            source "${CLIENT_VENV}/bin/activate"
            cd "${PROJ}"
            export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
            # eval_libero_plus.py resolves the backend via render_backend.py; these
            # exports only provide the per-backend extras it cannot infer.
            export RENDER_BACKEND="${RENDER_BACKEND}"
            # llvmpipe thread pool is sized per worker so NUM_GPUS concurrent
            # clients stay inside the container CPU quota.
            export RENDER_N_WORKERS="${NUM_GPUS}"
            if [ "${RENDER_BACKEND}" = "osmesa" ]; then
                export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
                export GALLIUM_DRIVER=llvmpipe
                unset MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
            else
                export MUJOCO_EGL_DEVICE_ID=${SLOT_IDX}
                if [ -d "${CLIENT_VENV}/egl_vendor.d" ]; then
                    export __EGL_VENDOR_LIBRARY_DIRS="${CLIENT_VENV}/egl_vendor.d"
                fi
            fi
            export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
            export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
            python evaluation/LIBERO-plus2/eval_libero_plus.py \
                --host 127.0.0.1 --port ${PORT} \
                --task_suite_name "${SUITE}" \
                --num_trials_per_task ${NUM_TRIALS_PER_TASK} \
                --num_steps_wait ${NUM_STEPS_WAIT} \
                --seed ${SEED} \
                --replan_steps ${REPLAN_STEPS} \
                --start_idx ${START} --end_idx ${END} \
                --task_classification_path "${TASK_CLS}" \
                --eval_log_dir "${EVAL_LOG_DIR}" \
                --gripper_convention "${GRIPPER_CONVENTION}" \
                ${KPT_FLAGS} ${CAT_FLAG} \
                ${ROTATE_FLAG} \
                ${FAILURE_VIDEO_FLAG} ${SAVE_ACTIONS_FLAG} \
                > "${WORKER_DIR}/client_${SUITE}_${START}_${END}.log" 2>&1
        )
        CLIENT_EXIT=$?
        set -e

        if [ ${CLIENT_EXIT} -ne 0 ]; then
            echo "[$(date)] FAILED: gpu${GPU_IDX} ${SUITE}[${START}:${END}] exit=${CLIENT_EXIT}" | tee -a "${WORKER_DIR}/worker.log"
        else
            echo "[$(date)] DONE: gpu${GPU_IDX} ${SUITE}[${START}:${END}]" | tee -a "${WORKER_DIR}/worker.log"
        fi
    done < "${GPU_QUEUE}"

    # ─── Shutdown Server ───
    echo "[$(date)] GPU ${GPU_IDX}: All shards complete, shutting down server" | tee -a "${WORKER_DIR}/worker.log"
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    return 0
}

# ─── Launch all GPU workers in parallel ───
echo "Starting eval at $(date) — ${NUM_GPUS} GPUs, $(wc -l < "${QUEUE_FILE}") shards"
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_worker ${GPU_ARRAY[$i]} $i &
done
wait
echo "All shards complete at $(date)"

# ─── Aggregate results ───
echo "Aggregating results..."
(
    source "${CLIENT_VENV}/bin/activate"
    cd "${PROJ}"
    export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
    python evaluation/LIBERO-plus2/aggregate_results.py --root "${EVAL_LOG_DIR}"
)
echo "Results: ${EVAL_LOG_DIR}/overall_results.json"
echo "Done at $(date)"
