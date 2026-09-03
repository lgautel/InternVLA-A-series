#!/usr/bin/env bash
# Phase 1: 400-step warmup. Reuses launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

TASK="${1:?usage: phase1_warmup.sh <task_name>}"
resolve_task_paths "${TASK}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

if [[ "${DRY_RUN}" != "1" ]]; then
  v30_ready || rbt_die "Phase1 需要完整 v3.0 数据: ${TASK_V30}"
  ensure_lerobot_home_link
else
  rbt_log "DRY-RUN: 跳过 v3.0 完整性检查和 LeRobot symlink"
fi
WARMUP_STEPS="${WARMUP_STEPS:-400}"

EXISTING_CKPT="$(warmup_ckpt_path)"
if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" && -n "${EXISTING_CKPT}" ]]; then
  rbt_log "跳过 Phase1: 已有 ckpt@400 ${EXISTING_CKPT}"
  write_state "warmup" "skipped" "{\"ckpt\":\"${EXISTING_CKPT}\"}"
  exit 0
fi

LAUNCH="${ITVLAGP_ROOT}/launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh"
rbt_require_file "${LAUNCH}"
rbt_require_file "${PRETRAINED_PATH}/config.json" "InternVLA-A1.5-base"
rbt_require_file "${GEOPREDICT_CKPT}" "GeoPredict checkpoint"

init_train_run warmup

write_state "warmup" "running" "{\"output_dir\":\"${OUTPUT_DIR}\",\"job_stamp\":\"${JOB_STAMP}\"}"

rbt_log "==== Phase1 warmup ${TASK} ===="
rbt_log "JOB_STAMP=${JOB_STAMP}"
rbt_log "JOB_NAME=${JOB_NAME}"
rbt_log "OUTPUT_DIR=${OUTPUT_DIR}  (wandb offline -> OUTPUT_DIR/wandb; 本仓库无 TensorBoard)"
rbt_log "LOG_FILE=${LOG_FILE}"
rbt_log "DATA_REPO_ID=${TASK_REPO_ID}"
rbt_log "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
rbt_log "PRETRAINED_PATH=${PRETRAINED_PATH}"
WARMUP_SAVE_FREQ="${WARMUP_SAVE_FREQ:-$((WARMUP_STEPS / 4))}"
if [[ "${WARMUP_SAVE_FREQ}" -lt 1 ]]; then
  WARMUP_SAVE_FREQ=1
fi
rbt_log "warmup STEPS=${WARMUP_STEPS} SAVE_FREQ=${WARMUP_SAVE_FREQ} (每 1/4 总 step + 最后一步必存)"

export VENV_ROOT
export PROJ_ROOT="${ITVLAGP_ROOT}"
export PYTHON="${TRAIN_PYTHON}"
export HF_HOME HF_LEROBOT_HOME
export CUDA_VISIBLE_DEVICES PROC_PER_NODE BATCH_SIZE
export DATA_REPO_ID="${TASK_REPO_ID}"
export NORM_STATS="${TASK_NORM_TRAIN}"
export PRETRAINED_PATH GEOPREDICT_CKPT
export JOB_NAME OUTPUT_DIR LOG_FILE
export WANDB_NAME="${JOB_NAME}"
export MASTER_PORT="${WARMUP_MASTER_PORT}"
export NODE_COUNT="${NODE_COUNT:-1}"
export NODE_RANK="${NODE_RANK:-0}"

if [[ "${DRY_RUN}" == "1" ]]; then
  rbt_log "DRY-RUN: SMOKE=0 STEPS=${WARMUP_STEPS} ${LAUNCH}"
  write_state "warmup" "dry_run" "{\"output_dir\":\"${OUTPUT_DIR}\"}"
  exit 0
fi

run_one() {
  local smoke="$1"
  local steps="$2"
  local extra_name="$3"
  if [[ "${smoke}" == "1" ]]; then
    SMOKE=1 STEPS="${steps}" PROC_PER_NODE=1 BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
      OUTPUT_DIR="${OUTPUT_DIR}${extra_name}" \
      LOG_FILE="${TASK_LOG_DIR}/warmup_smoke_${JOB_STAMP}.log" \
      JOB_NAME="${JOB_NAME}-smoke" \
      WANDB_NAME="${JOB_NAME}-smoke" \
      bash "${LAUNCH}"
  else
    SMOKE=0 STEPS="${steps}" SAVE_FREQ="${WARMUP_SAVE_FREQ}" \
      bash "${LAUNCH}"
  fi
}

if [[ "${SKIP_SMOKE}" != "1" ]]; then
  rbt_log "warmup smoke (1 GPU, 1 step)"
  run_one 1 1 "_smoke"
fi

rbt_log "warmup 正式 ${WARMUP_STEPS} step"
run_one 0 "${WARMUP_STEPS}" ""

ln -sfn "${OUTPUT_DIR}" "${TASK_WARMUP_LATEST}"
CKPT="$(warmup_ckpt_path)"
[[ -n "${CKPT}" ]] || rbt_die "warmup 结束但找不到 checkpoints/000400/pretrained_model"
write_state "warmup" "ok" "{\"ckpt\":\"${CKPT}\",\"output_dir\":\"${OUTPUT_DIR}\",\"job_stamp\":\"${JOB_STAMP}\"}"
rbt_log "Phase1 完成 ckpt=${CKPT}"
