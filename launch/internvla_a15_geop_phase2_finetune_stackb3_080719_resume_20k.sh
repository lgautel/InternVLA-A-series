#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 2 resume (080719_2): continue from step 10000 -> 20000
# Native resume via --resume=true --config_path=.../010000/.../train_config.json
# Same OUTPUT_DIR as 080719 run; Action + Kpt only (config loaded from checkpoint)
# 8x H200, venv: /tmp/itnvla15rbt20/
###############################################################################

export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export WANDB_MODE=offline
export USE_LIBUV=${USE_LIBUV:-0}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

VENV_ROOT="${VENV_ROOT:-/tmp/itnvla15rbt20}"
PYTHON="${PYTHON:-${VENV_ROOT}/bin/python}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:${VENV_ROOT}/lib/pulseaudio:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-36505}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

JOB_NAME="${JOB_NAME:-2026_08_07_12_25_57-internvla_a1_5-geop-phase2-action-kpt-only-stackb3-abs-10k}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/internvla_a1_5/${JOB_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-${OUTPUT_DIR}/checkpoints/010000/pretrained_model/train_config.json}"
STEPS="${STEPS:-20000}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: resume config not found at ${CONFIG_PATH}" >&2
    exit 1
fi

echo "PYTHON=${PYTHON}"
echo "Resuming OUTPUT_DIR=${OUTPUT_DIR}"
echo "CONFIG_PATH=${CONFIG_PATH}"
echo "STEPS=${STEPS} (continue from step 10000)"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --config_path="${CONFIG_PATH}"
    --resume=true
    --steps="${STEPS}"
)

set +e
"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}"
TRAIN_EXIT=$?
set -e

if [[ "${TRAIN_EXIT}" -eq 0 ]] && [[ "${GCS_UPLOAD_ON_SUCCESS:-false}" == "true" ]]; then
    GCS_DEST="${GCS_DEST:-gs://physical-ai-data-eu/VENV/tmp/itnvla080719_2/}"
    LOCAL_SRC="${LOCAL_SRC:-${PROJ_ROOT}}"
    GCS_UPLOAD_LOG="${GCS_UPLOAD_LOG:-${PROJ_ROOT}/outputs/gcloud_upload_itnvla080719_2.log}"
    echo "[$(date -Iseconds)] Training finished; uploading ${LOCAL_SRC}/ -> ${GCS_DEST}"
    gcloud storage cp -r "${LOCAL_SRC}/" "${GCS_DEST}" 2>&1 | tee "${GCS_UPLOAD_LOG}"
    echo "[$(date -Iseconds)] GCS upload complete. Log: ${GCS_UPLOAD_LOG}"
fi

exit "${TRAIN_EXIT}"
