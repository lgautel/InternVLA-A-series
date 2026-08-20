#!/usr/bin/env bash
# =============================================================================
# GeoP Phase2 (080719) training monitor + post-run GCS upload wrapper.
#
# Runs the formal 080719 fine-tune script, periodically logs step/GPU metrics,
# and on successful completion uploads the project tree to GCS.
#
# Usage (recommended for formal 8-GPU run):
#   cd /tmp/SRC/InternVLA-A-series
#   nohup bash launch/internvla_a15_geop_phase2_finetune_stackb3_080719_monitor.sh \
#     >> outputs/internvla_a1_5/monitor_080719_geop_phase2.log 2>&1 &
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

TRAIN_SCRIPT="${SCRIPT_DIR}/internvla_a15_geop_phase2_finetune_stackb3_080719.sh"
TRAIN_LOG="${TRAIN_LOG:-${PROJ_ROOT}/outputs/internvla_a1_5/train_080719_geop_phase2.log}"
MONITOR_LOG="${MONITOR_LOG:-${PROJ_ROOT}/outputs/internvla_a1_5/monitor_080719_geop_phase2.log}"
GCS_UPLOAD_LOG="${GCS_UPLOAD_LOG:-${PROJ_ROOT}/outputs/gcloud_upload_itnvla080719.log}"
GCS_DEST="${GCS_DEST:-gs://physical-ai-data-eu/VENV/tmp/itnvla080719/}"
LOCAL_SRC="${LOCAL_SRC:-/tmp/SRC/InternVLA-A-series}"
POLL_SEC="${POLL_SEC:-300}"
STEPS="${STEPS:-10000}"

mkdir -p "$(dirname "${TRAIN_LOG}")" "$(dirname "${MONITOR_LOG}")"

_log() {
    echo "[$(date -Iseconds)] $*" | tee -a "${MONITOR_LOG}"
}

_monitor_loop() {
    local train_pid="$1"
    while kill -0 "${train_pid}" 2>/dev/null; do
        local ts step_line gpu proc_count
        ts="$(date -Iseconds)"
        step_line="$(grep -E 'step:' "${TRAIN_LOG}" 2>/dev/null | tail -1 || true)"
        gpu="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' '; ' || echo 'nvidia-smi unavailable')"
        proc_count="$(pgrep -cf 'lerobot_train.py' 2>/dev/null || echo 0)"
        echo "[${ts}] latest=${step_line:-pending} | lerobot_procs=${proc_count} | GPU: ${gpu}" >> "${MONITOR_LOG}"
        sleep "${POLL_SEC}"
    done
}

_log "Monitor started (POLL_SEC=${POLL_SEC}, target_steps=${STEPS})"
_log "Train script: ${TRAIN_SCRIPT}"
_log "Train log: ${TRAIN_LOG}"
_log "GCS dest: ${GCS_DEST}"
_log "Local src: ${LOCAL_SRC}/"

export GCS_UPLOAD_ON_SUCCESS=true
export GCS_DEST="${GCS_DEST}"
export LOCAL_SRC="${LOCAL_SRC}"
export GCS_UPLOAD_LOG="${GCS_UPLOAD_LOG}"

bash "${TRAIN_SCRIPT}" 2>&1 | tee -a "${TRAIN_LOG}" &
TRAIN_PID=$!

_monitor_loop "${TRAIN_PID}" &
MONITOR_PID=$!

wait "${TRAIN_PID}"
TRAIN_EXIT=$?

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

if [[ "${TRAIN_EXIT}" -eq 0 ]]; then
    _log "Training completed successfully (exit 0)."
    if [[ -f "${GCS_UPLOAD_LOG}" ]]; then
        _log "GCS upload log: ${GCS_UPLOAD_LOG}"
    else
        _log "WARNING: GCS upload log not found; upload may have been skipped or failed."
    fi
else
    _log "Training failed (exit ${TRAIN_EXIT}); GCS upload skipped."
    exit "${TRAIN_EXIT}"
fi
