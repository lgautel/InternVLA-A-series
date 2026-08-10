#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Resume variant of internvla_a15_finetune_libero_venv.sh.
#
# Why this script exists: the original 100k-step run (JOB_NAME=
# a15_libero4suite_100k_20260728_151045) was killed at step ~35200 around
# 2026-07-29 01:04 UTC — not by an error inside training (no traceback, no
# OOM in dmesg, host did not reboot: `uptime` shows 18+ days), but because
# the enclosing terminal/IDE session that hosted the `accelerate launch`
# process was itself torn down (new terminal sessions with fresh PIDs
# appeared at 01:01 UTC). This is the same class of problem as Problem #8
# in b/d/p/reprd_liberop_cam_rb.md (killing the process that owns the
# elastic-agent/TCPStore also kills all workers), just triggered by a
# session reset instead of manual nohup/disown. See problem log entry
# "Problem #9" for full details.
#
# Fix strategy: resume from the last saved checkpoint (035000, which has a
# full training_state: optimizer/scheduler/rng) using lerobot's native
# --resume=true --config_path=... mechanism (src/lerobot/configs/train.py
# validate(): resume loads the ENTIRE TrainPipelineConfig from
# checkpoints/last/pretrained_model/train_config.json, so most CLI flags
# below are only needed for accelerate's own process-launch bookkeeping,
# not for the policy/dataset config itself).
###############################################################################

export HF_HOME="${HF_HOME:-/mnt/r/CKPT/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

VENV_ROOT="${VENV_ROOT:-/mnt/r/VENV/ivla15}"
source "${VENV_ROOT}/bin/activate"

export WANDB_MODE=offline

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6380}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export USE_LIBUV=${USE_LIBUV:-0}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
PROC_PER_NODE="${PROC_PER_NODE:-4}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${VENV_ROOT}/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

JOB_NAME="${JOB_NAME:-a15_libero4suite_100k_20260728_151045}"
OUTPUT_DIR="outputs/internvla_a1_5/${JOB_NAME}"
CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: resume config not found at ${CONFIG_PATH}" >&2
    exit 1
fi

echo "Resuming JOB_NAME=${JOB_NAME} from ${CONFIG_PATH}"

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
)

accelerate launch "${ARGS[@]}"
