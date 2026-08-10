#!/usr/bin/env bash
# =============================================================================
# Watchdog wrapper around internvla_a15_finetune_libero_venv.sh.
#
# Written for the InternVLA-A1.5 LIBERO-Plus (Camera+Robot) reproduction
# documented in b/d/p/reprd_liberop_cam_rb.md, problem record #8.
#
# Why this exists: on this shared 8xH200 machine, torch's distributed
# rendezvous (accelerate launch --multi_gpu, c10d TCPStore) has been observed
# to hang indefinitely (no error, no GPU memory allocated, no CPU/IO activity
# on the 4 worker processes) when the machine is under heavy contention from
# OTHER users' jobs (observed concurrently: a large Ray/data_juicer cluster,
# several EgoDex data-processing processes, and >150GB of GPU memory in use on
# GPU0-3 by unrelated jobs). A minimal Accelerator()-only sanity script
# reliably completes rendezvous in ~2s when run standalone, which rules out a
# bug in our training script/config; this is an environmental flakiness issue
# on this specific shared node, not a code bug.
#
# Mitigation: launch the real training job, watch its log for either (a) the
# first "step" log line (ot_train.py:367) appearing within TIMEOUT_SEC, or
# (b) a fatal Python traceback / CUDA OOM. If neither happens in time, kill
# the whole process group and retry with a freshly-checked free
# MASTER_PORT (avoids the port-conflict fallback bug from problem record #8a
# where accelerate silently fell back to the machine's public hostname
# instead of 127.0.0.1). Once training is confirmed running (first step
# logged), the watchdog stops watching and lets training run to completion
# on its own (it does NOT kill a healthy run).
# =============================================================================
set -uo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJ_ROOT}"

VENV_ROOT="${VENV_ROOT:-/mnt/r/VENV/ivla15}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PROC_PER_NODE="${PROC_PER_NODE:-4}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-a15_libero4suite_100k}"
LOG_DIR="${LOG_DIR:-/mnt/r/tmp}"
TIMEOUT_SEC="${TIMEOUT_SEC:-480}"       # max time to wait for first step log per attempt
MAX_ATTEMPTS="${MAX_ATTEMPTS:-15}"
POLL_SEC="${POLL_SEC:-10}"

free_port() {
  source "${VENV_ROOT}/bin/activate"
  python3 -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('127.0.0.1', 0))
print(s.getsockname()[1])
s.close()
"
}

attempt=0
while (( attempt < MAX_ATTEMPTS )); do
  attempt=$((attempt + 1))
  PORT="$(free_port)"
  JOB_NAME="${JOB_NAME_PREFIX}_attempt${attempt}_$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/train_watchdog_attempt${attempt}.log"

  echo "[watchdog] attempt ${attempt}/${MAX_ATTEMPTS}: JOB_NAME=${JOB_NAME} MASTER_PORT=${PORT} log=${LOG_FILE}"

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" PROC_PER_NODE="${PROC_PER_NODE}" \
  MASTER_PORT="${PORT}" JOB_NAME="${JOB_NAME}" \
    nohup bash launch/internvla_a15_finetune_libero_venv.sh > "${LOG_FILE}" 2>&1 &
  RUN_PID=$!
  echo "[watchdog] launcher pid=${RUN_PID}"

  waited=0
  success=0
  while (( waited < TIMEOUT_SEC )); do
    sleep "${POLL_SEC}"
    waited=$((waited + POLL_SEC))

    if grep -q "ot_train.py:367" "${LOG_FILE}" 2>/dev/null; then
      echo "[watchdog] SUCCESS: training is stepping (attempt ${attempt}, waited ${waited}s). JOB_NAME=${JOB_NAME}"
      echo "${JOB_NAME}" > "${LOG_DIR}/current_train_job_name.txt"
      success=1
      break
    fi
    if grep -qE "Traceback \(most recent call last\)|CUDA out of memory|ChildFailedError" "${LOG_FILE}" 2>/dev/null; then
      echo "[watchdog] FAILURE detected in attempt ${attempt} after ${waited}s, will retry:"
      tail -n 30 "${LOG_FILE}"
      break
    fi
    # Process died without a clear error string (e.g. killed externally).
    if ! kill -0 "${RUN_PID}" 2>/dev/null && ! pgrep -f "lerobot_train.py.*${JOB_NAME}" >/dev/null 2>&1; then
      echo "[watchdog] launcher process exited early (attempt ${attempt}, waited ${waited}s), will retry."
      break
    fi
  done

  if (( success == 1 )); then
    echo "[watchdog] training is healthy, watchdog exiting (job keeps running independently: JOB_NAME=${JOB_NAME})."
    exit 0
  fi

  echo "[watchdog] attempt ${attempt} did not reach a first training step within ${TIMEOUT_SEC}s; killing and retrying."
  pkill -9 -f "lerobot_train.py.*${JOB_NAME}" >/dev/null 2>&1 || true
  sleep 5
done

echo "[watchdog] giving up after ${MAX_ATTEMPTS} attempts. See ${LOG_DIR}/train_watchdog_attempt*.log" >&2
exit 1
