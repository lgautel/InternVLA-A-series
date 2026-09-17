#!/usr/bin/env bash
set -euo pipefail

# Model servers on SERVER_GPU_IDS and six independent LIBERO/EGL clients on
# CLIENT_GPU_IDS.  The client GPUs never load the policy model.
#
# Default topology (SERVER_INSTANCES_PER_GPU=3, i.e. six servers -- see
# b/d/libplus/eval3_optim3.md §二十 for the full standard-deployment writeup):
#   GPU 0: server-0/1/2 (ports BASE_PORT+0/1/2) <- clients GPU 2/3/4
#   GPU 1: server-3/4/5 (ports BASE_PORT+3/4/5) <- clients GPU 5/6/7
#
# Set SERVER_INSTANCES_PER_GPU=1 to fall back to the two-server topology:
#   server-0: GPU 0, port BASE_PORT     <- clients GPU 2,3,4
#   server-1: GPU 1, port BASE_PORT + 1 <- clients GPU 5,6,7
#
# Important: MUJOCO_EGL_DEVICE_ID is the physical EGL device index.  It is
# deliberately set to the global client GPU id, even though CUDA_VISIBLE_DEVICES
# is also restricted to that one client GPU.  MuJoCo's EGL backend enumerates
# EGL devices directly (not CUDA's remapped local ordinals).
#
# This is a separate launcher so the original one-server-per-GPU launcher
# remains available for comparison.

: "${LIBERO_HOME:?ERROR: LIBERO_HOME not set}"
: "${SERVER_VENV:?ERROR: SERVER_VENV not set}"
: "${CLIENT_VENV:?ERROR: CLIENT_VENV not set}"

PROJ="${PROJ:-$(cd "$(dirname "$0")/../.." && pwd)}"
CKPT_PATH="${CKPT_PATH:-}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
SERVER_GPU_IDS="${SERVER_GPU_IDS:-0,1}"
CLIENT_GPU_IDS="${CLIENT_GPU_IDS:-2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-5784}"
SERVER_INSTANCES_PER_GPU="${SERVER_INSTANCES_PER_GPU:-3}"
SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-6}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
EVAL_MODE="${EVAL_MODE:-full}"  # full | smoke
SMOKE_SUITE="${SMOKE_SUITE:-libero_spatial}"
SMOKE_TASKS="${SMOKE_TASKS:-6}"
MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
SEED="${SEED:-7}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
STATS_KEY_MODE="${STATS_KEY_MODE:-panda}"
ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-panda}"
GRIPPER_CONVENTION="${GRIPPER_CONVENTION:-libero_native}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
# `standard` is the compatible default for the LIBERO2 keypoint client.
# The current optimized backend does not accept `his_kpts`; use it only with
# DISABLE_KEYPOINTS=true (or after the optimized backend is fixed).
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
SAVE_ACTIONS_FLAG="${SAVE_ACTIONS_FLAG:---save_actions}"
SAVE_FAILURE_VIDEOS="${SAVE_FAILURE_VIDEOS:-false}"
DISABLE_KEYPOINTS="${DISABLE_KEYPOINTS:-}"
ROTATE_IMAGES="${ROTATE_IMAGES:-false}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-300}"
SERVER_POLL_INTERVAL="${SERVER_POLL_INTERVAL:-2}"
RENDER_BACKEND="${RENDER_BACKEND:-egl}"
EGL_VENDOR_DIR="${EGL_VENDOR_DIR:-${CLIENT_VENV}/egl_vendor.d}"

if [[ "${RENDER_BACKEND}" != "egl" ]]; then
    echo "ERROR: this launcher is the EGL split-topology experiment; set RENDER_BACKEND=egl" >&2
    exit 2
fi
if [[ -z "${CKPT_PATH}" ]]; then
    echo "ERROR: CKPT_PATH not set (the two servers load the same model checkpoint)" >&2
    exit 2
fi

IFS=',' read -ra SERVER_GPU_ARRAY <<< "${SERVER_GPU_IDS}"
IFS=',' read -ra CLIENT_GPU_ARRAY <<< "${CLIENT_GPU_IDS}"
NUM_SERVERS=${#SERVER_GPU_ARRAY[@]}
NUM_CLIENTS=${#CLIENT_GPU_ARRAY[@]}

if [[ ${NUM_SERVERS} -ne 2 ]]; then
    echo "ERROR: exactly two server GPUs are required; got ${SERVER_GPU_IDS}" >&2
    exit 2
fi
if [[ ${NUM_CLIENTS} -ne 6 ]]; then
    echo "ERROR: exactly six client GPUs are required; got ${CLIENT_GPU_IDS}" >&2
    exit 2
fi
if [[ ! "${SERVER_INSTANCES_PER_GPU}" =~ ^[13]$ ]]; then
    echo "ERROR: SERVER_INSTANCES_PER_GPU must be 1 or 3; got ${SERVER_INSTANCES_PER_GPU}" >&2
    exit 2
fi

NUM_SERVER_INSTANCES=$((NUM_SERVERS * SERVER_INSTANCES_PER_GPU))
if (( NUM_CLIENTS % NUM_SERVER_INSTANCES != 0 )); then
    echo "ERROR: six clients cannot be evenly assigned to ${NUM_SERVER_INSTANCES} server instances" >&2
    exit 2
fi
CLIENTS_PER_SERVER=$((NUM_CLIENTS / NUM_SERVER_INSTANCES))
declare -a SERVER_INSTANCE_GPU_ARRAY=()
for server_gpu in "${SERVER_GPU_ARRAY[@]}"; do
    for _ in $(seq 1 "${SERVER_INSTANCES_PER_GPU}"); do
        SERVER_INSTANCE_GPU_ARRAY+=("${server_gpu}")
    done
done

declare -A SEEN_GPU
for gpu in "${SERVER_GPU_ARRAY[@]}" "${CLIENT_GPU_ARRAY[@]}"; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: GPU id must be a non-negative integer, got '${gpu}'" >&2
        exit 2
    fi
    if [[ -n "${SEEN_GPU[${gpu}]:-}" ]]; then
        echo "ERROR: server/client GPU sets overlap or contain duplicates: ${gpu}" >&2
        exit 2
    fi
    SEEN_GPU["${gpu}"]=1
done

if [[ "${EVAL_MODE}" == "smoke" ]]; then
    SUITES="${SMOKE_SUITE}"
    SHARDS_PER_SUITE="${NUM_CLIENTS}"
    MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-20}"
fi

if [[ -n "${MAX_STEPS_OVERRIDE}" ]]; then
    MAX_STEPS_FLAG=(--max_steps_override "${MAX_STEPS_OVERRIDE}")
else
    MAX_STEPS_FLAG=()
fi

if [[ -z "${DISABLE_KEYPOINTS}" ]]; then
    KPT_FLAGS=(--enable_keypoints)
else
    KPT_FLAGS=(--no-enable_keypoints)
fi
if [[ "${ROTATE_IMAGES}" == "true" ]]; then
    ROTATE_FLAG=(--rotate_images)
else
    ROTATE_FLAG=()
fi
if [[ "${SAVE_FAILURE_VIDEOS}" == "true" ]]; then
    FAILURE_VIDEO_FLAG=(--save_failure_videos)
else
    FAILURE_VIDEO_FLAG=()
fi

TASK_CLS="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
if [[ ! -f "${TASK_CLS}" ]]; then
    echo "ERROR: task classification file not found: ${TASK_CLS}" >&2
    exit 2
fi

EVAL_LOG_DIR="${EVAL_LOG_DIR:-${CKPT_PATH}/../libero_plus_$(date +%Y%m%d%H%M)}"
LIBERO_CONFIG_DIR="${EVAL_LOG_DIR}/libero_config"
QUEUE_FILE="${EVAL_LOG_DIR}/task_queue.txt"
mkdir -p "${EVAL_LOG_DIR}" "${LIBERO_CONFIG_DIR}"

LP_BENCH="${LIBERO_HOME}/libero/libero"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LP_BENCH}
bddl_files: ${LP_BENCH}/bddl_files
init_states: ${LP_BENCH}/init_files
datasets: ${LP_BENCH}/../datasets
assets: ${LP_BENCH}/assets
YAML

echo "============================================"
echo "LIBERO-plus2: ${NUM_SERVER_INSTANCES} policy servers + 6 EGL clients"
echo "  Mode: ${EVAL_MODE}"
echo "  Checkpoint: ${CKPT_PATH}"
echo "  Server GPUs: ${SERVER_GPU_IDS} (${SERVER_INSTANCES_PER_GPU} instances/GPU)"
echo "  Server ports: ${BASE_PORT}..$((BASE_PORT + NUM_SERVER_INSTANCES - 1))"
echo "  Clients: GPU ${CLIENT_GPU_IDS}"
for client_idx in $(seq 0 $((NUM_CLIENTS - 1))); do
    group=$((client_idx / CLIENTS_PER_SERVER))
    echo "  Mapping: client GPU ${CLIENT_GPU_ARRAY[${client_idx}]} -> server${group} GPU ${SERVER_INSTANCE_GPU_ARRAY[${group}]} port $((BASE_PORT + group))"
done
echo "  Render: EGL (client GPU is separate from server GPU)"
echo "  Inference backend: ${INFERENCE_BACKEND}"
echo "  Output: ${EVAL_LOG_DIR}"
echo "============================================"

# Build task shards.  In smoke mode the first six tasks become one shard per
# client, which exercises all six client GPUs and both server endpoints.
: > "${QUEUE_FILE}"
for suite in ${SUITES}; do
    TOTAL="$(python3 - "${TASK_CLS}" "${suite}" "${EVAL_MODE}" "${SMOKE_TASKS}" <<'PY'
import json
import sys

path, suite, mode, smoke_tasks = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
items = data[suite]
total = len(items) if isinstance(items, list) else sum(len(v) for v in items.values())
if mode == "smoke":
    total = min(total, int(smoke_tasks))
print(total)
PY
)"
    SHARD_SIZE=$(( (TOTAL + SHARDS_PER_SUITE - 1) / SHARDS_PER_SUITE ))
    for shard in $(seq 0 $((SHARDS_PER_SUITE - 1))); do
        START=$((shard * SHARD_SIZE))
        END=$(( (shard + 1) * SHARD_SIZE ))
        [[ ${END} -gt ${TOTAL} ]] && END=${TOTAL}
        [[ ${START} -ge ${END} ]] && continue
        echo "${suite} ${START} ${END}" >> "${QUEUE_FILE}"
    done
done

for idx in $(seq 0 $((NUM_CLIENTS - 1))); do
    : > "${EVAL_LOG_DIR}/client${idx}_queue.txt"
done
SHARD_IDX=0
while IFS=' ' read -r suite start end; do
    client_idx=$((SHARD_IDX % NUM_CLIENTS))
    echo "${suite} ${start} ${end}" >> "${EVAL_LOG_DIR}/client${client_idx}_queue.txt"
    SHARD_IDX=$((SHARD_IDX + 1))
done < "${QUEUE_FILE}"

declare -a SERVER_PIDS=()
declare -a CLIENT_PIDS=()
STOP_REQUESTED=0

cleanup() {
    if [[ ${STOP_REQUESTED} -eq 1 ]]; then
        return
    fi
    STOP_REQUESTED=1
    echo "[$(date)] Cleaning up split topology..." >&2
    for pid in "${CLIENT_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    for pid in "${SERVER_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
    done
    for pid in "${CLIENT_PIDS[@]:-}" "${SERVER_PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && wait "${pid}" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

start_server() {
    local group=$1
    local server_gpu="${SERVER_INSTANCE_GPU_ARRAY[${group}]}"
    local port=$((BASE_PORT + group))
    local server_dir="${EVAL_LOG_DIR}/server${group}"
    mkdir -p "${server_dir}"

    (
        source "${SERVER_VENV}/bin/activate"
        cd "${PROJ}"
        export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
        export CUDA_VISIBLE_DEVICES="${server_gpu}"
        # The model server must not initialise a MuJoCo/EGL context.
        unset MUJOCO_GL PYOPENGL_PLATFORM MUJOCO_EGL_DEVICE_ID __EGL_VENDOR_LIBRARY_DIRS
        exec python evaluation/LIBERO2/policy_server/server_policy.py \
            --ckpt_path "${CKPT_PATH}" \
            --host 0.0.0.0 --port "${port}" --device cuda \
            --resize_size "${RESIZE_SIZE}" \
            --stats_key "${STATS_KEY_MODE}" --robot_type "${ROBOT_TYPE_MODE}" \
            ${VLM_MODEL_PATH:+--vlm_model_path "${VLM_MODEL_PATH}"} \
            ${ACTION_LOSS_ONLY_FLAG} \
            --inference_backend "${INFERENCE_BACKEND}" \
            --idle_timeout -1
    ) > "${server_dir}/server.log" 2>&1 &
    SERVER_PIDS[${group}]=$!
    echo "[$(date)] server${group}: pid=${SERVER_PIDS[${group}]} GPU=${server_gpu} port=${port}"
}

wait_for_server() {
    local group=$1
    local port=$((BASE_PORT + group))
    local log="${EVAL_LOG_DIR}/server${group}/healthcheck.log"
    local started
    started="$(date +%s)"
    while true; do
        if (
            source "${SERVER_VENV}/bin/activate"
            cd "${PROJ}"
            export PYTHONPATH="${PROJ}:${PROJ}/src:${PYTHONPATH:-}"
            python - "${port}" "${STATS_KEY_MODE}" "${RESIZE_SIZE}" <<'PY'
import sys
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy

port, stats_key, resize_size = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
client = WebsocketClientPolicy(host="127.0.0.1", port=port)
metadata = client.get_server_metadata()
assert metadata.get("action_mode") == "joint", metadata
assert metadata.get("preprocessing_owner") == "server_canonical", metadata
assert metadata.get("stats_key") == stats_key, metadata
assert int(metadata.get("resize_size") or 0) == resize_size, metadata
print("HEALTHCHECK PASSED", metadata)
PY
        ) >> "${log}" 2>&1; then
            echo "[$(date)] server${group}: healthcheck passed on port ${port}"
            return 0
        fi
        if ! kill -0 "${SERVER_PIDS[${group}]}" 2>/dev/null; then
            echo "ERROR: server${group} exited before becoming ready; see ${EVAL_LOG_DIR}/server${group}/server.log" >&2
            return 1
        fi
        if (( "$(date +%s)" - started >= SERVER_STARTUP_TIMEOUT )); then
            echo "ERROR: server${group} did not become ready; see ${EVAL_LOG_DIR}/server${group}/server.log" >&2
            return 1
        fi
        sleep "${SERVER_POLL_INTERVAL}"
    done
}

record_gpu_snapshot() {
    local snapshot="${EVAL_LOG_DIR}/gpu_topology_snapshot.txt"
    {
        echo "# split topology: server GPUs=${SERVER_GPU_IDS}; client GPUs=${CLIENT_GPU_IDS}"
        echo "# server PIDs: ${SERVER_PIDS[*]}"
        echo "# CUDA_VISIBLE_DEVICES is process-local for each server/client"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
            echo "# compute processes"
            nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader || true
        else
            echo "nvidia-smi unavailable"
        fi
    } > "${snapshot}"
    echo "[$(date)] GPU topology snapshot: ${snapshot}"
}

start_client() {
    local client_idx=$1
    local client_gpu="${CLIENT_GPU_ARRAY[${client_idx}]}"
    local group=$((client_idx / CLIENTS_PER_SERVER))
    local port=$((BASE_PORT + group))
    local queue="${EVAL_LOG_DIR}/client${client_idx}_queue.txt"
    local client_dir="${EVAL_LOG_DIR}/client${client_idx}"
    mkdir -p "${client_dir}"
    if [[ ! -s "${queue}" ]]; then
        echo "[$(date)] client${client_idx}: no shard, skipping"
        return 0
    fi

    (
        source "${CLIENT_VENV}/bin/activate"
        cd "${PROJ}"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
        export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"

        # Separate the renderer from both model servers.  EGL uses physical
        # device enumeration, hence MUJOCO_EGL_DEVICE_ID=client_gpu.
        export RENDER_BACKEND=egl
        export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
        export MUJOCO_EGL_DEVICE_ID="${client_gpu}"
        export CUDA_VISIBLE_DEVICES="${client_gpu}"
        if [[ -d "${EGL_VENDOR_DIR}" ]]; then
            export __EGL_VENDOR_LIBRARY_DIRS="${EGL_VENDOR_DIR}"
        fi
        export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

        local status=0
        while IFS=' ' read -r suite start end; do
            echo "[$(date)] client${client_idx} GPU=${client_gpu} -> server${group}:${port}: ${suite}[${start}:${end}]"
            set +e
            python evaluation/LIBERO-plus2/eval_libero_plus.py \
                --host 127.0.0.1 --port "${port}" \
                --task_suite_name "${suite}" \
                --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
                --num_steps_wait "${NUM_STEPS_WAIT}" \
                --seed "${SEED}" \
                --replan_steps "${REPLAN_STEPS}" \
                --start_idx "${start}" --end_idx "${end}" \
                --task_classification_path "${TASK_CLS}" \
                --eval_log_dir "${EVAL_LOG_DIR}" \
                --gripper_convention "${GRIPPER_CONVENTION}" \
                "${KPT_FLAGS[@]}" "${ROTATE_FLAG[@]}" "${MAX_STEPS_FLAG[@]}" \
                "${FAILURE_VIDEO_FLAG[@]}" "${SAVE_ACTIONS_FLAG}" \
                > "${client_dir}/${suite}_${start}_${end}.log" 2>&1
            rc=$?
            set -e
            if [[ ${rc} -ne 0 ]]; then
                echo "[$(date)] client${client_idx}: shard failed rc=${rc}"
                status=1
            fi
        done < "${queue}"
        exit "${status}"
    ) &
    CLIENT_PIDS[${client_idx}]=$!
    echo "[$(date)] client${client_idx}: pid=${CLIENT_PIDS[${client_idx}]} GPU=${client_gpu} -> server${group} port=${port}"
}

for group in $(seq 0 $((NUM_SERVER_INSTANCES - 1))); do
    start_server "${group}"
done
for group in $(seq 0 $((NUM_SERVER_INSTANCES - 1))); do
    wait_for_server "${group}"
done
record_gpu_snapshot

echo "[$(date)] All ${NUM_SERVER_INSTANCES} model servers ready; launching six EGL clients..."
for client_idx in $(seq 0 $((NUM_CLIENTS - 1))); do
    start_client "${client_idx}"
done

CLIENT_STATUS=0
for pid in "${CLIENT_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && ! wait "${pid}"; then
        CLIENT_STATUS=1
    fi
done

if [[ ${CLIENT_STATUS} -ne 0 ]]; then
    echo "ERROR: at least one split-topology client failed; inspect ${EVAL_LOG_DIR}/client*/" >&2
else
    echo "[$(date)] All split-topology clients completed."
fi

(
    source "${CLIENT_VENV}/bin/activate"
    cd "${PROJ}"
    export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
    python evaluation/LIBERO-plus2/aggregate_results.py --root "${EVAL_LOG_DIR}"
)

echo "Results: ${EVAL_LOG_DIR}/overall_results.json"
exit "${CLIENT_STATUS}"
