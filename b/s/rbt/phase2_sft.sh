#!/usr/bin/env bash
# Phase 2: SFT from this task's warmup ckpt@400.
# Steps and checkpoint schedule are computed per task from total_frames and SFT_EPOCHS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

TASK="${1:?usage: phase2_sft.sh <task_name>}"
resolve_task_paths "${TASK}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

if [[ "${DRY_RUN}" != "1" ]]; then
  v30_ready || rbt_die "Phase2 需要完整 v3.0 数据: ${TASK_V30}"
  ensure_lerobot_home_link
else
  rbt_log "DRY-RUN: 跳过 v3.0 完整性检查和 LeRobot symlink"
fi

WARMUP_CKPT="${WARMUP_CKPT:-$(warmup_ckpt_path)}"
if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -z "${WARMUP_CKPT}" ]]; then
    WARMUP_CKPT="${TASK_WARMUP_DIR}/<pending-ckpt-400>"
    rbt_log "DRY-RUN: 尚无 warmup ckpt@400, 使用占位路径 ${WARMUP_CKPT}"
  fi
else
  [[ -n "${WARMUP_CKPT}" ]] || rbt_die "找不到 warmup ckpt@400, 请先跑 Phase1"
  [[ -d "${WARMUP_CKPT}" ]] || rbt_die "warmup ckpt 目录不存在: ${WARMUP_CKPT}"
fi

if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" && -L "${TASK_SFT_LATEST}" ]]; then
  rbt_log "跳过 Phase2: 已有 sft/latest -> $(readlink -f "${TASK_SFT_LATEST}" || true)"
  write_state "sft" "skipped" "{\"output_dir\":\"$(readlink -f "${TASK_SFT_LATEST}" || true)\"}"
  exit 0
fi

SFT_EPOCHS="${SFT_EPOCHS:-76}"
SFT_INFO="${TASK_V30}/meta/info.json"
if [[ "${DRY_RUN}" == "1" && ! -f "${SFT_INFO}" ]]; then
  SFT_INFO="${TASK_SRC}/meta/info.json"
  rbt_log "DRY-RUN: v3.0 info.json 尚不存在，使用源任务 info.json 估算 SFT schedule"
fi

eval "$(
  "${TRAIN_PYTHON}" "${SCRIPT_DIR}/compute_sft_steps.py" \
    --info "${SFT_INFO}" \
    --epochs "${SFT_EPOCHS}" \
    --n-gpus "${PROC_PER_NODE}" \
    --batch-size "${BATCH_SIZE}" \
    --n-nodes "${NODE_COUNT:-1}" \
    --as-exports
)"

EFFECTIVE_BS=$(( PROC_PER_NODE * BATCH_SIZE * ${NODE_COUNT:-1} ))
if [[ "${EFFECTIVE_BS}" -ne "${SFT_EFFECTIVE_BATCH_TARGET:-128}" ]]; then
  rbt_log "警告: 有效 batch=${EFFECTIVE_BS} 不等于目标 ${SFT_EFFECTIVE_BATCH_TARGET:-128}; 步数按实际有效 batch 计算"
fi

LAUNCH="${ITVLAGP_ROOT}/launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh"
rbt_require_file "${LAUNCH}"
rbt_require_dir "${WAN_DIR}" "WAN2.2 权重目录"

init_train_run sft
write_state "sft" "running" "{\"output_dir\":\"${OUTPUT_DIR}\",\"steps\":${SFT_STEPS},\"epochs\":${SFT_EPOCHS},\"job_stamp\":\"${JOB_STAMP}\"}"

rbt_log "==== Phase2 SFT ${TASK} ===="
rbt_log "JOB_STAMP=${JOB_STAMP}"
rbt_log "JOB_NAME=${JOB_NAME}  (wandb run name)"
rbt_log "OUTPUT_DIR=${OUTPUT_DIR}"
rbt_log "  checkpoints: OUTPUT_DIR/checkpoints/<step>/"
rbt_log "  wandb:       OUTPUT_DIR/wandb/   (offline; 本仓库无 TensorBoard)"
rbt_log "LOG_FILE=${LOG_FILE}"
rbt_log "WARMUP_CKPT=${WARMUP_CKPT}"
rbt_log "frames=${SFT_TOTAL_FRAMES} effective_bs=${SFT_EFFECTIVE_BATCH} spe=${SFT_STEPS_PER_EPOCH}"
rbt_log "EPOCHS=${SFT_EPOCHS} STEPS=${SFT_STEPS} SAVE_EVERY_EPOCHS=${SFT_SAVE_EVERY_EPOCHS} SAVE_FREQ=${SFT_SAVE_FREQ}"
rbt_log "checkpoint at epochs: ${SFT_SAVE_AT_EPOCHS}"
rbt_log "checkpoint at steps:  ${SFT_SAVE_STEPS}  (每 1/4 总 epoch + 最后一步必存)"

export VENV_ROOT
export PROJ_ROOT="${ITVLAGP_ROOT}"
export PYTHON="${TRAIN_PYTHON}"
export HF_HOME HF_LEROBOT_HOME
export CUDA_VISIBLE_DEVICES PROC_PER_NODE BATCH_SIZE
export DATA_REPO_ID="${TASK_REPO_ID}"
export NORM_STATS="${TASK_NORM_TRAIN}"
export WARMUP_CKPT
export WAN_DIR
export JOB_NAME OUTPUT_DIR LOG_FILE
export WANDB_NAME="${JOB_NAME}"
export MASTER_PORT="${SFT_MASTER_PORT}"
export NODE_COUNT="${NODE_COUNT:-1}"
export NODE_RANK="${NODE_RANK:-0}"

if [[ "${DRY_RUN}" == "1" ]]; then
  rbt_log "DRY-RUN: SMOKE=0 STEPS=${SFT_STEPS} SAVE_FREQ=${SFT_SAVE_FREQ} ${LAUNCH}"
  write_state "sft" "dry_run" "{\"output_dir\":\"${OUTPUT_DIR}\",\"steps\":${SFT_STEPS},\"epochs\":${SFT_EPOCHS},\"save_at_epochs\":\"${SFT_SAVE_AT_EPOCHS}\",\"job_stamp\":\"${JOB_STAMP}\"}"
  exit 0
fi

run_one() {
  local smoke="$1"
  if [[ "${smoke}" == "1" ]]; then
    SMOKE=1 STEPS=1 SAVE_FREQ=1 PROC_PER_NODE=1 BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}" \
      CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
      OUTPUT_DIR="${OUTPUT_DIR}_smoke" \
      LOG_FILE="${TASK_LOG_DIR}/sft_smoke_${JOB_STAMP}.log" \
      JOB_NAME="${JOB_NAME}-smoke" \
      WANDB_NAME="${JOB_NAME}-smoke" \
      bash "${LAUNCH}"
  else
    SMOKE=0 STEPS="${SFT_STEPS}" SAVE_FREQ="${SFT_SAVE_FREQ}" \
      SCHEDULER_WARMUP="${SFT_SCHEDULER_WARMUP}" \
      bash "${LAUNCH}"
  fi
}

if [[ "${SKIP_SMOKE}" != "1" ]]; then
  rbt_log "sft smoke (1 GPU, 1 step)"
  run_one 1
fi

rbt_log "sft 正式 ${SFT_STEPS} step"
run_one 0

ln -sfn "${OUTPUT_DIR}" "${TASK_SFT_LATEST}"
write_state "sft" "ok" "{\"output_dir\":\"${OUTPUT_DIR}\",\"steps\":${SFT_STEPS},\"epochs\":${SFT_EPOCHS},\"save_at_epochs\":\"${SFT_SAVE_AT_EPOCHS}\",\"save_steps\":\"${SFT_SAVE_STEPS}\",\"warmup_ckpt\":\"${WARMUP_CKPT}\",\"job_stamp\":\"${JOB_STAMP}\"}"
rbt_log "Phase2 完成 OUTPUT_DIR=${OUTPUT_DIR}"
