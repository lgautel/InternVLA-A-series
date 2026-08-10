#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# venv-based variant of run_eval_libero_plus.sh, written for the InternVLA-A1.5
# LIBERO-Plus (Camera+Robot) reproduction documented in
# b/d/p/reprd_liberop_cam_rb.md.
#
# Differences from run_eval_libero_plus.sh:
#   - server/client envs are plain `uv venv` virtualenvs (not conda envs):
#       SERVER_VENV (default /mnt/r/VENV/ivla15)
#       CLIENT_VENV (default /mnt/r/VENV/ivla15_libero_plus_client)
#     activated via `source <venv>/bin/activate` instead of `conda activate`.
#   - passes through a `--categories` filter to eval_libero_plus.py so only the
#     requested LIBERO-plus perturbation categories are simulated (e.g.
#     "Camera Viewpoints,Robot Initial States"), skipping the rest of the
#     ~10k-task benchmark to save compute.
#   - defaults to GPU_IDS=0,1,2,3 (this reproduction reserves GPU4-7 for other
#     jobs already running on the shared machine).
#
# ONE-TIME SETUP (NOT done by this script): see b/d/p/reprd_liberop_cam_rb.md
# Part A section 1-2 (two venvs + LIBERO-plus repo/assets download).
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------- User-configurable defaults ----------------
SERVER_VENV="${SERVER_VENV:-/mnt/r/VENV/ivla15}"
CLIENT_VENV="${CLIENT_VENV:-/mnt/r/VENV/ivla15_libero_plus_client}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-5774}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
SEED="${SEED:-7}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-4}"
NO_ROTATE_FLAG=""
NO_BINARIZE_FLAG=""
NO_VIDEO_FLAG="${NO_VIDEO_FLAG:-}"
DEBUG_FLAG=""

# Perturbation-category filter passed straight through to eval_libero_plus.py.
# Empty (default) = evaluate all 7 categories (original LIBERO-plus behavior).
CATEGORIES="${CATEGORIES:-}"
CATEGORIES_FLAG=()
if [[ -n "${CATEGORIES}" ]]; then
  CATEGORIES_FLAG=(--categories "${CATEGORIES}")
fi

GPU_MEM_FREE_THRESHOLD_MB="${GPU_MEM_FREE_THRESHOLD_MB:-30000}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
SERVER_READY_TIMEOUT_SEC="${SERVER_READY_TIMEOUT_SEC:-600}"

TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

# Required paths — override via env vars.
CKPT_PATH="${CKPT_PATH:-}"
LIBERO_HOME="${LIBERO_HOME:-}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"
WAN_MODEL_PATH="${WAN_MODEL_PATH:-}"
WAN_VAE_PATH="${WAN_VAE_PATH:-}"
ACTION_LOSS_ONLY_FLAG="${ACTION_LOSS_ONLY_FLAG:---action_loss_only}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
STATS_KEY_MODE="${STATS_KEY_MODE:-suite}"
ROBOT_TYPE_MODE="${ROBOT_TYPE_MODE:-suite}"

EVAL_LOG_DIR="${EVAL_LOG_DIR:-}"

if [[ -z "${CKPT_PATH}" ]]; then
  echo "[ERROR] CKPT_PATH is required." >&2
  exit 2
fi
if [[ -z "${LIBERO_HOME}" ]]; then
  echo "[ERROR] LIBERO_HOME is required (point at the LIBERO-plus repo root)." >&2
  exit 2
fi
if [[ ! -x "${SERVER_VENV}/bin/activate" && ! -f "${SERVER_VENV}/bin/activate" ]]; then
  echo "[ERROR] SERVER_VENV not found at ${SERVER_VENV}" >&2
  exit 2
fi
if [[ ! -f "${CLIENT_VENV}/bin/activate" ]]; then
  echo "[ERROR] CLIENT_VENV not found at ${CLIENT_VENV}" >&2
  exit 2
fi
TASK_CLASSIFICATION="${LIBERO_HOME}/libero/libero/benchmark/task_classification.json"
if [[ ! -f "${TASK_CLASSIFICATION}" ]]; then
  echo "[ERROR] task_classification.json not found at ${TASK_CLASSIFICATION}" >&2
  exit 2
fi

if [[ -z "${EVAL_LOG_DIR}" ]]; then
  EVAL_LOG_DIR="outputs/sim_eval/libero_plus/$(date +%Y%m%d_%H%M%S)_camrb"
fi
mkdir -p "${EVAL_LOG_DIR}"

LIBERO_CONFIG_DIR="${LIBERO_CONFIG_DIR:-${EVAL_LOG_DIR}/libero_config}"
mkdir -p "${LIBERO_CONFIG_DIR}"
LP_BENCH_ROOT="${LIBERO_HOME}/libero/libero"
cat > "${LIBERO_CONFIG_DIR}/config.yaml" <<YAML
benchmark_root: ${LP_BENCH_ROOT}
bddl_files: ${LP_BENCH_ROOT}/bddl_files
init_states: ${LP_BENCH_ROOT}/init_files
datasets: ${LP_BENCH_ROOT}/../datasets
assets: ${LP_BENCH_ROOT}/assets
YAML

QUEUE_FILE="${EVAL_LOG_DIR}/task_queue.txt"
QUEUE_LOCK="${EVAL_LOG_DIR}/queue.lock"
PIDS_FILE="${EVAL_LOG_DIR}/server_pids.txt"
WORKER_PIDS_FILE="${EVAL_LOG_DIR}/worker_pids.txt"
: > "${PIDS_FILE}"
: > "${WORKER_PIDS_FILE}"

cleanup() {
  for f in "${WORKER_PIDS_FILE}" "${PIDS_FILE}"; do
    [[ -f "${f}" ]] || continue
    while read -r pid; do
      [[ -n "${pid}" ]] || continue
      kill "${pid}" >/dev/null 2>&1 || true
    done < "${f}"
  done
}
trap cleanup EXIT INT TERM

# ---------------- Build the (suite, shard) work queue ----------------
build_queue() {
  python3 - "$TASK_CLASSIFICATION" "$SHARDS_PER_SUITE" "${TASK_SUITES[@]}" <<'PY' > "${QUEUE_FILE}"
import json, sys
path = sys.argv[1]
shards = int(sys.argv[2])
suites = sys.argv[3:]
mapping = json.load(open(path))
for suite in suites:
    n = len(mapping[suite])
    shards_eff = min(shards, n)
    base = n // shards_eff
    rem = n % shards_eff
    start = 0
    for i in range(shards_eff):
        size = base + (1 if i < rem else 0)
        end = start + size
        print(f"{suite} {start} {end}")
        start = end
PY
}
build_queue
NUM_UNITS=$(wc -l < "${QUEUE_FILE}")

# ---------------- GPU discovery ----------------
detect_free_gpus() {
  if [[ -n "${GPU_IDS}" ]]; then
    echo "${GPU_IDS}" | tr ',' '\n' | sed '/^$/d'
    return
  fi
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
    | awk -v t="${GPU_MEM_FREE_THRESHOLD_MB}" -F', *' '$2+0 >= t {print $1}'
}

mapfile -t FREE_GPUS < <(detect_free_gpus)
NUM_FREE=${#FREE_GPUS[@]}

if (( NUM_FREE == 0 )); then
  echo "[ERROR] No GPUs selected. Set GPU_IDS=... explicitly." >&2
  exit 1
fi

NUM_WORKERS=${NUM_FREE}
if (( NUM_WORKERS > NUM_UNITS )); then
  NUM_WORKERS=${NUM_UNITS}
fi
WORKER_GPUS=("${FREE_GPUS[@]:0:${NUM_WORKERS}}")

echo "GPUs (fixed)       : ${FREE_GPUS[*]}"
echo "Workers            : ${NUM_WORKERS} on GPU(s) ${WORKER_GPUS[*]}"
echo "Work units         : ${NUM_UNITS} (suites=${#TASK_SUITES[@]} x shards=${SHARDS_PER_SUITE})"
echo "Ckpt               : ${CKPT_PATH}"
echo "LIBERO-plus home   : ${LIBERO_HOME}"
echo "Categories filter  : ${CATEGORIES:-ALL}"
echo "Log dir            : ${EVAL_LOG_DIR}"

resolve_stats_key() {
  local suite="$1"
  if [[ "${STATS_KEY_MODE}" == "suite" ]]; then
    echo "${suite}"
  else
    echo "${STATS_KEY_MODE}"
  fi
}

resolve_robot_type() {
  local suite="$1"
  if [[ "${ROBOT_TYPE_MODE}" == "suite" ]]; then
    echo "${suite}"
  else
    echo "${ROBOT_TYPE_MODE}"
  fi
}

# ---------------- Worker definition ----------------
run_worker() {
  local gpu_id="$1"
  local worker_idx="$2"
  local port=$((BASE_PORT + worker_idx))

  local worker_dir="${EVAL_LOG_DIR}/worker_gpu${gpu_id}"
  mkdir -p "${worker_dir}"
  local worker_log="${worker_dir}/worker.log"

  while true; do
    local unit=""
    {
      flock 9
      local line
      line=$(head -n 1 "${QUEUE_FILE}" 2>/dev/null || true)
      if [[ -n "${line}" ]]; then
        sed -i '1d' "${QUEUE_FILE}"
      fi
      unit="${line}"
    } 9>"${QUEUE_LOCK}"
    if [[ -z "${unit}" ]]; then
      break
    fi

    local suite start end
    read -r suite start end <<< "${unit}"

    local task_dir="${EVAL_LOG_DIR}"
    local server_log="${worker_dir}/server_${suite}_${start}_${end}.log"
    local client_log="${worker_dir}/client_${suite}_${start}_${end}.log"

    local stats_key
    stats_key=$(resolve_stats_key "${suite}")
    local robot_type
    robot_type=$(resolve_robot_type "${suite}")

    echo "[gpu=${gpu_id} port=${port}] server up for ${suite} [${start},${end}) (stats_key=${stats_key})" | tee -a "${worker_log}"

    (
      source "${SERVER_VENV}/bin/activate"
      cd "${PROJ_ROOT}"
      PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src:${PYTHONPATH:-}" \
      CUDA_VISIBLE_DEVICES="${gpu_id}" exec python evaluation/LIBERO/policy_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --host "0.0.0.0" \
        --port "${port}" \
        --device cuda \
        --resize_size "${RESIZE_SIZE}" \
        --stats_key "${stats_key}" \
        --robot_type "${robot_type}" \
        --vlm_model_path "${VLM_MODEL_PATH}" \
        --wan_model_path "${WAN_MODEL_PATH}" \
        --wan_vae_path "${WAN_VAE_PATH}" \
        ${ACTION_LOSS_ONLY_FLAG} \
        --inference_backend "${INFERENCE_BACKEND}" \
        --idle_timeout -1
    ) >"${server_log}" 2>&1 &
    local server_pid=$!
    echo "${server_pid}" >> "${PIDS_FILE}"

    if ! (
      source "${SERVER_VENV}/bin/activate"
      cd "${PROJ_ROOT}"
      PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src:${PYTHONPATH:-}" python - <<PY
import time, sys
from evaluation.LIBERO.policy_server.tools.websocket_policy_client import WebsocketClientPolicy

host = "${HOST}"
port = int("${port}")
timeout = int("${SERVER_READY_TIMEOUT_SEC}")
deadline = time.time() + timeout
while time.time() < deadline:
    try:
        client = WebsocketClientPolicy(host=host, port=port)
        resp = client.predict_action({"type": "ping", "request_id": "healthcheck"})
        if resp.get("ok", False):
            print("server ready")
            sys.exit(0)
    except Exception:
        time.sleep(1)
sys.exit(f"server healthcheck failed after {timeout}s")
PY
    ) >>"${worker_log}" 2>&1; then
      echo "[gpu=${gpu_id}] [${suite} ${start}-${end}] server failed to come up; skipping" | tee -a "${worker_log}" >&2
      kill "${server_pid}" >/dev/null 2>&1 || true
      wait "${server_pid}" 2>/dev/null || true
      continue
    fi

    echo "==== [gpu=${gpu_id}] [${suite} ${start}-${end}] starting ====" | tee -a "${worker_log}"
    (
      source "${CLIENT_VENV}/bin/activate"
      cd "${PROJ_ROOT}"
      export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_DIR}"
      export MAGICK_HOME="${CLIENT_VENV}"
      export LD_LIBRARY_PATH="${CLIENT_VENV}/lib:${LD_LIBRARY_PATH:-}"
      PYTHONPATH="${LIBERO_HOME}:${PROJ_ROOT}:${PYTHONPATH:-}" \
      python evaluation/LIBERO-plus/eval_libero_plus.py \
        --host "${HOST}" \
        --port "${port}" \
        --task_suite_name "${suite}" \
        --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
        --num_steps_wait "${NUM_STEPS_WAIT}" \
        --seed "${SEED}" \
        --replan_steps "${REPLAN_STEPS}" \
        --start_idx "${start}" \
        --end_idx "${end}" \
        --task_classification_path "${TASK_CLASSIFICATION}" \
        --eval_log_dir "${task_dir}" \
        "${CATEGORIES_FLAG[@]}" \
        ${NO_ROTATE_FLAG} ${NO_BINARIZE_FLAG} ${NO_VIDEO_FLAG} ${DEBUG_FLAG}
    ) >"${client_log}" 2>&1 || echo "[gpu=${gpu_id}] [${suite} ${start}-${end}] FAILED (see ${client_log})" | tee -a "${worker_log}" >&2
    echo "==== [gpu=${gpu_id}] [${suite} ${start}-${end}] done ====" | tee -a "${worker_log}"

    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" 2>/dev/null || true
  done

  echo "[gpu=${gpu_id}] queue drained" | tee -a "${worker_log}"
}

# ---------------- Spawn workers ----------------
WORKER_PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
  run_worker "${WORKER_GPUS[$i]}" "$i" &
  WORKER_PIDS+=($!)
  echo "$!" >> "${WORKER_PIDS_FILE}"
done

EXIT_CODE=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    EXIT_CODE=1
  fi
done

# ---------------- Aggregate ----------------
echo "All shards done. Aggregating..."
(
  source "${CLIENT_VENV}/bin/activate"
  cd "${PROJ_ROOT}"
  python evaluation/LIBERO-plus/aggregate_results.py --root "${EVAL_LOG_DIR}"
) || echo "[WARN] aggregation failed; per-shard json still under ${EVAL_LOG_DIR}/logs" >&2

echo "Done. Logs: ${EVAL_LOG_DIR}"
exit "${EXIT_CODE}"
