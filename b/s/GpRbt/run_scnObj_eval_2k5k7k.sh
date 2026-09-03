#!/usr/bin/env bash
# 串行评测 scan_object @2500 / @5000 / @7500，日志写入 eval2k5k7kLOG.md
set -euo pipefail

REPO_ROOT="/home/luogang/SRC/Robot/itvlaGp"
EVAL_SH="${REPO_ROOT}/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh"
EVAL_LOG="${REPO_ROOT}/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval2k5k7kLOG.md"
GCS_JOB="gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30"
KPT_META="/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/scan_object_kptsim_lrbv30/meta/keypoints_meta.json"
BATCH_LOG="${REPO_ROOT}/outputs/logs/batch_scnObj_2k5k7k.log"

now() { date '+%Y-%m-%d %H:%M:%S'; }

log_batch() {
  echo "[$(now)] $*" | tee -a "${BATCH_LOG}"
}

append_timeline() {
  echo "| $(now) | $1 | $2 |" >> "${EVAL_LOG}"
}

mkdir -p "$(dirname "${BATCH_LOG}")"

log_batch "========== scan_object @2500/@5000/@7500 批次开始 =========="
append_timeline "批次编排脚本启动" "见 ${BATCH_LOG}"

STEPS=(002500 005000 007500)
for step in "${STEPS[@]}"; do
  log_batch "---------- 开始 ckpt-step=${step} ----------"
  append_timeline "开始 ckpt-step=${step} 全流程" "进行中"

  set +e
  bash "${EVAL_SH}" \
    --task-name scan_object \
    --task-idx 41 \
    --gcs-job "${GCS_JOB}" \
    --ckpt-step "${step}" \
    --expect-repo-id scan_object_kptsim_lrbv30 \
    --expect-offset -0.6748,-1.0345,0.6219 \
    --kpt-meta "${KPT_META}" \
    --eval-log "${EVAL_LOG}" \
    2>&1 | tee -a "${BATCH_LOG}"
  ec=${PIPESTATUS[0]}
  set -e

  if [[ "${ec}" -ne 0 ]]; then
    log_batch "FAIL ckpt-step=${step} exit=${ec}"
    append_timeline "ckpt-step=${step} 全流程" "FAIL exit=${ec}"
    exit "${ec}"
  fi
  log_batch "OK ckpt-step=${step}"
  append_timeline "ckpt-step=${step} 全流程" "OK"
done

log_batch "========== 三步全部完成 =========="
append_timeline "批次全部完成" "OK"
