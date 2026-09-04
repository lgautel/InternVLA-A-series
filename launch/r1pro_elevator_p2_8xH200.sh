#!/usr/bin/env bash
set -euo pipefail
# R1 Pro Elevator Phase 2 SFT — 8×H200 wrapper
# Usage: WARMUP_CKPT=<path> bash launch/r1pro_elevator_p2_8xH200.sh
#        WAN_SMOKE=1 WARMUP_CKPT=<path> bash launch/r1pro_elevator_p2_8xH200.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="/B/VENV/itnvla15rbt20"

export TRAIN_VENV="${VENV}"
export PYTHON="${VENV}/bin/python"
export HF_HOME="/B/VENV/hf_home"
# HF_LEROBOT_HOME defaults to ${HF_HOME}/lerobot in the inner script

# LD_LIBRARY_PATH for torchcodec/NPP
export LD_LIBRARY_PATH="${VENV}/lib/python3.11/site-packages/nvidia/npp/lib:\
${VENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
${VENV}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${VENV}/lib/python3.11/site-packages/torch/lib:\
${VENV}/lib:${LD_LIBRARY_PATH:-}"

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/itvla-triton-cache}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Disable NCCL tuner plugin (cluster has libnccl-tuner.so but no config)
export NCCL_TUNER_PLUGIN="libnccl-tuner-disabled.so"

# Only set 8-GPU defaults when not in smoke mode
if [[ "${WAN_SMOKE:-0}" == "0" && "${SMOKE:-0}" == "0" ]]; then
    export PROC_PER_NODE="${PROC_PER_NODE:-8}"
    export BATCH_SIZE="${BATCH_SIZE:-16}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export NUM_WORKERS="${NUM_WORKERS:-12}"
    export STEPS="${STEPS:-2130}"
    export SAVE_FREQ="${SAVE_FREQ:-213}"
    export SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-213}"
fi
export MASTER_PORT="${MASTER_PORT:-36603}"

# Monitoring & archive
export EXPR_NAME="${EXPR_NAME:-ItvlaGpR1proElvtH200}"
export ARCHIVE_SOURCE="${ARCHIVE_SOURCE:-/B}"
export ARCHIVE_DEST="${ARCHIVE_DEST:-${HOME}/b/Ckp}"
export BIGMATRIX_SCRIPT="${PROJ_ROOT}/b/d/GpRbt/bigmatrix_multiply_optimization.py"

exec bash "${PROJ_ROOT}/launch/internvla_a15_r1pro_geop_phase2_elevator.sh"
