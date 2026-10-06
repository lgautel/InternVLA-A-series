#!/usr/bin/env bash
set -euo pipefail
# Phase 3 Evaluation — v2 (14D state) variant
# Overrides EXPR_NAME, GOAL4D_DATASET, and uses preflight_v2.sh instead of preflight_goal.sh.
#
# eval_goal_wrapper.sh hardcodes "${HERE}/preflight_goal.sh", so we inline the preflight
# call then delegate the remaining phases.
#
# Usage:
#   CKPT_PATH=<v2_sft_ckpt> bash b/s/libplus2/gol2/eval_v2_wrapper.sh
#   CKPT_PATH=<v2_sft_ckpt> EVAL_MODE=smoke bash b/s/libplus2/gol2/eval_v2_wrapper.sh

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GOL="${HERE}/../gol"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"

export EXPR_NAME="${EXPR_NAME:-4dwvlaLbPlusGolV2_1001}"
export GOAL4D_DATASET="${GOAL4D_DATASET:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2}"
export CKPT_PATH

EVAL_MODE="${EVAL_MODE:-full}"
TS="$(date +%Y%m%d_%H%M%S)"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/B/Log/${EXPR_NAME}/${TS}_eval}"
mkdir -p "${EVAL_LOG_DIR}"
LOG_FILE="${EVAL_LOG_DIR}/eval_goal.log"

echo "============================================================"             | tee -a "${LOG_FILE}"
echo " LIBERO-plus libero_goal Evaluation — v2 (14D state)"                     | tee -a "${LOG_FILE}"
echo " EXPR_NAME:      ${EXPR_NAME}"                                            | tee -a "${LOG_FILE}"
echo " CKPT_PATH:      ${CKPT_PATH}"                                            | tee -a "${LOG_FILE}"
echo " GOAL4D_DATASET: ${GOAL4D_DATASET}"                                       | tee -a "${LOG_FILE}"
echo " EVAL_MODE:      ${EVAL_MODE}"                                            | tee -a "${LOG_FILE}"
echo " LOG_FILE:       ${LOG_FILE}"                                              | tee -a "${LOG_FILE}"
echo "============================================================"             | tee -a "${LOG_FILE}"

# === Phase 1: Preflight (v2) ===
echo "[$(date)] Phase 1: Preflight v2 checks..." | tee -a "${LOG_FILE}"
if ! bash "${HERE}/preflight_v2.sh" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "FATAL: Preflight v2 failed. See ${LOG_FILE}" >&2
    exit 1
fi
echo "[$(date)] Preflight v2 PASSED" | tee -a "${LOG_FILE}"

# === Phase 2: GPU Cleanup ===
echo "[$(date)] Phase 2: GPU cleanup..." | tee -a "${LOG_FILE}"
pkill -f bigmatrix_multiply_optimization || true
sleep 3

# === Phase 3: Read keypoint_history_max_len from checkpoint ===
KPT_HIST_LEN=$(python3 -c "
import json
cfg = json.load(open('${CKPT_PATH}/config.json'))
print(cfg.get('keypoint_history_max_len', 92))
")
echo "[$(date)] keypoint_history_max_len from checkpoint: ${KPT_HIST_LEN}" | tee -a "${LOG_FILE}"
export KPT_HISTORY_MAX_LEN="${KPT_HIST_LEN}"

# === Phase 4: Run evaluation (delegates to gol/run_eval_goal_plus.sh) ===
echo "[$(date)] Phase 4: Starting evaluation (${EVAL_MODE})..." | tee -a "${LOG_FILE}"
export EVAL_LOG_DIR
export CATEGORIES="${CATEGORIES:-}"
export RENDER_BACKEND="${RENDER_BACKEND:-egl}"
export SAVE_FAILURE_VIDEOS="${SAVE_FAILURE_VIDEOS:-false}"

LIBERO_HOME="${LIBERO_HOME:-/B/SRC/LIBERO-plus}"
SERVER_VENV="${SERVER_VENV:-/B/VENV/itnvla15rbt20}"
CLIENT_VENV="${CLIENT_VENV:-/B/VENV/libero_plus_client}"
export LIBERO_HOME SERVER_VENV CLIENT_VENV

export SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-64}"
export MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-150}"

if [ "${EVAL_MODE}" = "smoke" ]; then
    export GPU_IDS="${GPU_IDS:-0,2}"
fi

CONTRACT="${GOAL4D_DATASET}/meta/goal_train_eval_contract.json"
CONTRACT_SCHEMA=$(python3 -c "import json; print(json.load(open('${CONTRACT}'))['schema'])")
if [ "${CONTRACT_SCHEMA}" = "goal_train_eval_contract/2" ]; then
    GOAL_ROBOT_TYPE="libero_goal_4dv2"
else
    GOAL_ROBOT_TYPE="panda"
fi
export GOAL4D_CONTRACT="${CONTRACT}"
export ROTATE_IMAGES=false
export STATS_KEY_MODE="${GOAL_ROBOT_TYPE}"
export ROBOT_TYPE_MODE="${GOAL_ROBOT_TYPE}"
export INFERENCE_BACKEND=standard
export PROJ

set +e
# Use DRY_RUN to generate the derived script, then patch the healthcheck
# for v2: action_mode is 'end_effector' not 'joint' (v2 uses abs/EEF actions)
DRY_RUN=1 bash "${GOL}/run_eval_goal_plus.sh" 2>&1 | tee -a "${LOG_FILE}"
DERIVED="${EVAL_LOG_DIR}/run_eval_goal_plus.derived.sh"
if [ -f "${DERIVED}" ]; then
    sed -i "s/== 'joint'/== 'end_effector'/" "${DERIVED}"
    GOL_RESOLVED="$(cd "${GOL}" && pwd)"
    HERE_RESOLVED="$(cd "${HERE}" && pwd)"
    sed -i "s|${GOL_RESOLVED}/eval_goal_plus.py|${HERE_RESOLVED}/eval_goal_plus_v2.py|g" "${DERIVED}"
    # Add MagickWand lib path for LIBERO-plus background texture tasks
    sed -i 's|CLIENT_VENV}/lib:/usr/local|CLIENT_VENV}/lib:/opt/conda/lib:/usr/local|' "${DERIVED}"
    # Pass MAX_STEPS_OVERRIDE to eval client if set
    if [ -n "${MAX_STEPS_OVERRIDE:-}" ] && [ "${MAX_STEPS_OVERRIDE}" -gt 0 ] 2>/dev/null; then
        sed -i "s|--seed \${SEED}|--seed \${SEED} --max_steps_override ${MAX_STEPS_OVERRIDE}|" "${DERIVED}"
    fi
    # Fix orphan GPU servers: subshell `( source ..; python server )` captures the
    # subshell PID, not python's.  `kill $PID` kills the subshell wrapper but
    # python may survive if bash didn't exec-optimize it.  Adding `exec` ensures
    # the subshell is replaced by python, so $PID == python PID.
    sed -i 's|CUDA_VISIBLE_DEVICES=${GPU_IDX} python evaluation/|CUDA_VISIBLE_DEVICES=${GPU_IDX} exec python evaluation/|' "${DERIVED}"
    echo "[$(date)] Patched derived: healthcheck, eval_goal_plus_v2, MagickWand, max_steps, exec-server" | tee -a "${LOG_FILE}"
fi
bash "${DERIVED}" 2>&1 | tee -a "${LOG_FILE}"
EVAL_EXIT=$?
set -e

# === Phase 5: Post-eval ===
echo "[$(date)] Phase 5: Post-evaluation..." | tee -a "${LOG_FILE}"

# Kill any orphan server_policy.py processes left behind by gpu_worker.
# Safety net in case the exec patch above was insufficient or the derived
# script predates the patch.
BASE_PORT="${BASE_PORT:-5784}"
for i in $(seq 0 7); do
    PORT=$((BASE_PORT + i))
    SERVER_PID=$(lsof -ti tcp:${PORT} 2>/dev/null || true)
    if [ -n "${SERVER_PID}" ]; then
        echo "[$(date)] Killing orphan server on port ${PORT} (PID ${SERVER_PID})" | tee -a "${LOG_FILE}"
        kill ${SERVER_PID} 2>/dev/null || true
    fi
done
sleep 2
# Force-kill any survivors
for i in $(seq 0 7); do
    PORT=$((BASE_PORT + i))
    SERVER_PID=$(lsof -ti tcp:${PORT} 2>/dev/null || true)
    if [ -n "${SERVER_PID}" ]; then
        echo "[$(date)] Force-killing stubborn server on port ${PORT} (PID ${SERVER_PID})" | tee -a "${LOG_FILE}"
        kill -9 ${SERVER_PID} 2>/dev/null || true
    fi
done

if [ ${EVAL_EXIT} -eq 0 ]; then
    echo "Evaluation COMPLETED SUCCESSFULLY" | tee -a "${LOG_FILE}"
else
    echo "Evaluation FINISHED WITH ERRORS (exit=${EVAL_EXIT})" | tee -a "${LOG_FILE}"
fi

# Aggregate results (produces overall_results.json)
if [ -d "${EVAL_LOG_DIR}" ] && [ -f "${PROJ}/evaluation/LIBERO-plus2/aggregate_results.py" ]; then
    echo "[$(date)] Aggregating results..." | tee -a "${LOG_FILE}"
    (
        source "${CLIENT_VENV}/bin/activate"
        cd "${PROJ}"
        export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
        python evaluation/LIBERO-plus2/aggregate_results.py --root "${EVAL_LOG_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    ) || echo "WARNING: Aggregation failed" | tee -a "${LOG_FILE}"
fi

# Collect failure records into failures_manifest.json + replay_commands.sh
echo "[$(date)] Collecting failure records..." | tee -a "${LOG_FILE}"
(
    python3 "${GOL}/collect_failures.py" \
        --eval-log-dir "${EVAL_LOG_DIR}" \
        --suite libero_goal \
        --task-classification "${LIBERO_HOME}/libero/libero/benchmark/task_classification.json" \
        2>&1 | tee -a "${LOG_FILE}"
) || echo "WARNING: Failure collection failed (non-fatal)" | tee -a "${LOG_FILE}"

# Show summary from overall_results.json
if [ -f "${EVAL_LOG_DIR}/overall_results.json" ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "=== Results Summary ===" | tee -a "${LOG_FILE}"
    python3 -c "
import json
r = json.load(open('${EVAL_LOG_DIR}/overall_results.json'))
sr = r.get('success_rate', r.get('overall_success_rate', 'N/A'))
print(f'Overall SR: {sr}')
cats = r.get('by_category', r.get('category_results', {}))
for cat, data in sorted(cats.items()):
    csr = data.get('success_rate', data.get('sr', 0))
    n = data.get('total', data.get('n_tasks', 0))
    print(f'  {cat}: {csr:.1%} ({n} tasks)' if isinstance(csr, float) else f'  {cat}: {csr} ({n} tasks)')
" 2>/dev/null | tee -a "${LOG_FILE}" || true
fi

# Archive results (after post-processing so archive includes manifests)
ARCHIVE="/B/Log/${EXPR_NAME}/eval_goal_${TS}.tar.gz"
if [ -d "${EVAL_LOG_DIR}" ]; then
    tar -czf "${ARCHIVE}" -C "$(dirname "${EVAL_LOG_DIR}")" "$(basename "${EVAL_LOG_DIR}")" 2>/dev/null || true
    echo "Results archived to: ${ARCHIVE}" | tee -a "${LOG_FILE}"
fi

# Full experiment log archive: all training + eval logs → ~/b/Ckp/
ALL_LOGS_ARCHIVE="${HOME}/b/Ckp/${EXPR_NAME}_ALL_LOGS_$(date +%Y%m%d_%H%M%S).tar.gz"
if [ -d "/B/Log/${EXPR_NAME}" ]; then
    echo "[$(date)] Archiving all experiment logs to ${ALL_LOGS_ARCHIVE}..." | tee -a "${LOG_FILE}"
    tar -czf "${ALL_LOGS_ARCHIVE}" -C /B/Log "${EXPR_NAME}" 2>/dev/null || true
    echo "Full logs archived to: ${ALL_LOGS_ARCHIVE}" | tee -a "${LOG_FILE}"
fi

echo "[$(date)] Restarting bigmatrix GPU placeholder..." | tee -a "${LOG_FILE}"
nohup python "${PROJ}/b/d/GpRbt/bigmatrix_multiply_optimization.py" > /dev/null 2>&1 &
echo "bigmatrix PID: $!" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[$(date)] Done." | tee -a "${LOG_FILE}"
echo "  Log:     ${LOG_FILE}"
echo "  Results: ${EVAL_LOG_DIR}"
echo "  Archive: ${ARCHIVE}"
echo "  Full logs: ${ALL_LOGS_ARCHIVE}"
exit ${EVAL_EXIT}
