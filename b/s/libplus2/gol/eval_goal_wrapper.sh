#!/usr/bin/env bash
# eval_goal_wrapper.sh — LIBERO-plus libero_goal 评估总入口
#
# 用法:
#   CKPT_PATH=<path> bash b/s/libplus2/gol/eval_goal_wrapper.sh
#
# 必需环境变量:
#   CKPT_PATH       — checkpoint pretrained_model 目录
#
# 可选:
#   EVAL_MODE           — smoke (快速验证) 或 full (完整评估, 默认)
#   CATEGORIES          — 限定 perturbation category, 如 "Language Instructions"
#   GPU_IDS             — GPU 列表, 默认 0,1,2,3,4,5,6,7
#   RENDER_BACKEND      — egl (默认) | osmesa | auto
#   SAVE_FAILURE_VIDEOS — true 保存失败视频
#   EVAL_LOG_DIR        — 输出目录 (默认: checkpoint 同级)
#   EXPR_NAME           — 实验名 (默认: 4dwvlaLbPlusGol0929)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"
EXPR_NAME="${EXPR_NAME:-4dwvlaLbPlusGol0929}"

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
export GOAL4D_DATASET="${GOAL4D_DATASET:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4D}"
export LIBERO_HOME="${LIBERO_HOME:-/B/SRC/LIBERO-plus}"
export SERVER_VENV="${SERVER_VENV:-/B/VENV/itnvla15rbt20}"
export CLIENT_VENV="${CLIENT_VENV:-/B/VENV/libero_plus_client}"

EVAL_MODE="${EVAL_MODE:-full}"
TS="$(date +%Y%m%d_%H%M%S)"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/B/Log/${EXPR_NAME}/${TS}_eval}"
mkdir -p "${EVAL_LOG_DIR}"
LOG_FILE="${EVAL_LOG_DIR}/eval_goal.log"

echo "============================================================" | tee "${LOG_FILE}"
echo " LIBERO-plus libero_goal Evaluation"                          | tee -a "${LOG_FILE}"
echo "  EXPR_NAME:    ${EXPR_NAME}"                                  | tee -a "${LOG_FILE}"
echo "  CKPT_PATH:    ${CKPT_PATH}"                                  | tee -a "${LOG_FILE}"
echo "  EVAL_MODE:    ${EVAL_MODE}"                                  | tee -a "${LOG_FILE}"
echo "  LOG_FILE:     ${LOG_FILE}"                                   | tee -a "${LOG_FILE}"
echo "  EVAL_LOG_DIR: ${EVAL_LOG_DIR}"                               | tee -a "${LOG_FILE}"
echo "  GOAL4D_DATASET: ${GOAL4D_DATASET}"                          | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

# === Phase 1: Preflight ===
echo "[$(date)] Phase 1: Preflight checks..." | tee -a "${LOG_FILE}"
if ! bash "${HERE}/preflight_goal.sh" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "FATAL: Preflight failed. See ${LOG_FILE}" >&2
    exit 1
fi
echo "[$(date)] Preflight PASSED" | tee -a "${LOG_FILE}"

# === Phase 2: GPU Cleanup ===
echo "[$(date)] Phase 2: GPU cleanup..." | tee -a "${LOG_FILE}"
pkill -f bigmatrix_multiply_optimization 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | tee -a "${LOG_FILE}"
echo "[$(date)] GPU cleanup done" | tee -a "${LOG_FILE}"

# === Phase 3: Read keypoint_history_max_len from checkpoint ===
KPT_HIST_LEN=$(python3 -c "
import json
cfg = json.load(open('${CKPT_PATH}/config.json'))
print(cfg.get('keypoint_history_max_len', 92))
")
echo "[$(date)] keypoint_history_max_len from checkpoint: ${KPT_HIST_LEN}" | tee -a "${LOG_FILE}"
export KPT_HISTORY_MAX_LEN="${KPT_HIST_LEN}"

# === Phase 4: Run evaluation ===
echo "[$(date)] Phase 4: Starting evaluation (${EVAL_MODE})..." | tee -a "${LOG_FILE}"
export CKPT_PATH EVAL_LOG_DIR
export CATEGORIES="${CATEGORIES:-}"
export RENDER_BACKEND="${RENDER_BACKEND:-egl}"
export SAVE_FAILURE_VIDEOS="${SAVE_FAILURE_VIDEOS:-false}"

if [ "${EVAL_MODE}" = "smoke" ]; then
    export SHARDS_PER_SUITE="${SHARDS_PER_SUITE:-2}"
    export GPU_IDS="${GPU_IDS:-0,2}"
    [ -z "${CATEGORIES}" ] && export CATEGORIES="Language Instructions"
    export MAX_STEPS_OVERRIDE="${MAX_STEPS_OVERRIDE:-100}"
    echo "[$(date)] Smoke mode: GPU_IDS=${GPU_IDS}, CATEGORIES=${CATEGORIES}, MAX_STEPS_OVERRIDE=${MAX_STEPS_OVERRIDE}" | tee -a "${LOG_FILE}"
fi

START_TIME=$(date +%s)
set +e
bash "${HERE}/run_eval_goal_plus.sh" 2>&1 | tee -a "${LOG_FILE}"
EVAL_EXIT=$?
set -e
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo "[$(date)] Evaluation finished in ${ELAPSED_MIN} min (exit=${EVAL_EXIT})" | tee -a "${LOG_FILE}"

# === Phase 5: Post-eval ===
echo "[$(date)] Phase 5: Post-evaluation..." | tee -a "${LOG_FILE}"

if [ ${EVAL_EXIT} -eq 0 ]; then
    echo "Evaluation COMPLETED SUCCESSFULLY" | tee -a "${LOG_FILE}"
else
    echo "Evaluation FINISHED WITH ERRORS (exit=${EVAL_EXIT})" | tee -a "${LOG_FILE}"
fi

# Aggregate results (if the aggregation script exists and eval produced output)
if [ -d "${EVAL_LOG_DIR}" ] && [ -f "${PROJ}/evaluation/LIBERO-plus2/aggregate_results.py" ]; then
    echo "[$(date)] Aggregating results..." | tee -a "${LOG_FILE}"
    (
        source "${CLIENT_VENV}/bin/activate"
        cd "${PROJ}"
        export PYTHONPATH="${LIBERO_HOME}:${PROJ}:${PYTHONPATH:-}"
        python evaluation/LIBERO-plus2/aggregate_results.py --root "${EVAL_LOG_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    ) || echo "WARNING: Aggregation failed" | tee -a "${LOG_FILE}"
fi

# Collect failure records into manifest + replay commands
echo "[$(date)] Collecting failure records..." | tee -a "${LOG_FILE}"
(
    python3 "${HERE}/collect_failures.py" \
        --eval-log-dir "${EVAL_LOG_DIR}" \
        --suite libero_goal \
        --task-classification "${LIBERO_HOME}/libero/libero/benchmark/task_classification.json" \
        2>&1 | tee -a "${LOG_FILE}"
) || echo "WARNING: Failure collection failed (non-fatal)" | tee -a "${LOG_FILE}"

# Archive results
ARCHIVE="/B/Log/${EXPR_NAME}/eval_goal_${TS}.tar.gz"
if [ -d "${EVAL_LOG_DIR}" ]; then
    tar -czf "${ARCHIVE}" -C "$(dirname "${EVAL_LOG_DIR}")" "$(basename "${EVAL_LOG_DIR}")" 2>/dev/null || true
    echo "Results archived to: ${ARCHIVE}" | tee -a "${LOG_FILE}"
fi

# Show summary
if [ -f "${EVAL_LOG_DIR}/overall_results.json" ]; then
    echo "" | tee -a "${LOG_FILE}"
    echo "=== Results Summary ===" | tee -a "${LOG_FILE}"
    python3 -c "
import json, sys
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

# Restart bigmatrix
echo "[$(date)] Restarting bigmatrix GPU placeholder..." | tee -a "${LOG_FILE}"
nohup python "${PROJ}/b/d/GpRbt/bigmatrix_multiply_optimization.py" > /dev/null 2>&1 &
echo "bigmatrix PID: $!" | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "[$(date)] Done." | tee -a "${LOG_FILE}"
echo "  Log:     ${LOG_FILE}"
echo "  Results: ${EVAL_LOG_DIR}"
echo "  Archive: ${ARCHIVE}"
