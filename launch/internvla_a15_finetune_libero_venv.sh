#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# venv-based variant of internvla_a15_finetune_libero.sh, written for the
# InternVLA-A1.5 LIBERO-Plus (Camera+Robot) reproduction documented in
# b/d/p/reprd_liberop_cam_rb.md.
#
# Differences from the original launch/internvla_a15_finetune_libero.sh:
#   - Activates a plain `uv venv` virtualenv (/mnt/r/VENV/ivla15) instead of a
#     conda env.
#   - PRETRAINED_PATH points at the locally downloaded InternVLA-A1.5-base
#     checkpoint (/mnt/r/CKPT/InternVLA-A1.5-base) instead of the HF repo id,
#     to avoid re-downloading ~5.4GB of weights every launch.
#   - WAN checkpoint/config/VAE paths are overridden to the locally downloaded
#     Wan2.2-TI2V-5B copy (/mnt/r/CKPT/Wan2.2-TI2V-5B) instead of the default
#     HF repo id "Wan-AI/Wan2.2-TI2V-5B" (which would otherwise trigger a
#     second, redundant download into the HF cache).
#   - DATASET_REPO_ID is written explicitly as the four suite names
#     (libero_spatial/object/goal/10) instead of relying on the original
#     script's `find data/libero -name 'libero_*_no_noops*_lerobot'` glob,
#     because nvidia/LIBERO_LeRobot_v3 ships each suite directly as
#     data/<suite_name>/ (see repro manual Part A section 3 for the exact
#     `data/` symlink chain: data -> /mnt/r/CKPT/hf_home/lerobot ->
#     (symlinks) -> /mnt/r/DATA/libero_lerobot_v3/<suite>).
#   - Restricted to GPU0-3 (GPU4-7 are occupied by unrelated jobs on this
#     shared machine).
###############################################################################

export HF_HOME="${HF_HOME:-/mnt/r/CKPT/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

VENV_ROOT="${VENV_ROOT:-/mnt/r/VENV/ivla15}"
source "${VENV_ROOT}/bin/activate"

export WANDB_MODE=offline

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# USE_LIBUV=0: PyTorch>=2.4 defaults TCPStore's server backend to the newer
# libuv implementation. On this shared machine, rank0's TCPStore(is_master=True,
# use_libuv=True) constructor call was observed to hang indefinitely (all 4
# ranks stuck in native code at torch/distributed/rendezvous.py:191, port never
# actually opened for listening, confirmed via py-spy stack dumps), matching a
# known upstream libuv-backend hang (see
# https://github.com/pytorch/pytorch/pull/127957 and
# https://docs.pytorch.org/tutorials/intermediate/TCPStore_libuv_backend.html
# "Exit Route 3"). Falling back to the legacy (non-libuv) TCPStore backend
# avoids the hang. See b/d/p/reprd_liberop_cam_rb.md problem log #8.
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
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd "${PROJ_ROOT}"

# 1. policy config
POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/r/CKPT/InternVLA-A1.5-base}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3.5-2B}"
WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-/mnt/r/CKPT/Wan2.2-TI2V-5B}"
WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-/mnt/r/CKPT/Wan2.2-TI2V-5B}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/mnt/r/CKPT/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

# 2. dataset config: 4 LIBERO suites, joint fine-tune (see manual section 0/3).
DATASET_REPO_ID="${DATASET_REPO_ID:-libero_spatial libero_object libero_goal libero_10}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"

ACTION_TYPE=abs
USE_EXTERNAL_STATS=false

# Idempotent robot_type patch: give each suite its own robot_type so per-suite
# stats don't collide (see original script's comment; here repo name IS the
# suite name already, no _no_noops suffix to strip).
if [ "${NODE_RANK}" = "0" ]; then
    for repo in ${DATASET_REPO_ID}; do
        info_json="data/${repo}/meta/info.json"
        subset_robot_type="$(basename "${repo}")"
        if [ -f "${info_json}" ]; then
            python -c "
import json
p = '${info_json}'
target = '${subset_robot_type}'
with open(p) as f:
    d = json.load(f)
if d.get('robot_type') != target:
    d['robot_type'] = target
    with open(p, 'w') as f:
        json.dump(d, f, indent=4)
    print(f'[info-patch] robot_type -> {target} in {p}')
else:
    print(f'[info-patch] already {target} in {p}')
"
        else
            echo "[info-patch] WARNING: ${info_json} not found"
        fi
    done
fi

# 3. output configs
BASE_OUTPUT_DIR="outputs/${POLICY}"
PRETRAINED_DETAIL="a15_base"
JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-libero-${ACTION_TYPE}-${PRETRAINED_DETAIL}-finetune}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"

ARGS=(
    # ---- Accelerate / distributed ----
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    # ---- Output ----
    --output_dir="${OUTPUT_DIR}"
    --num_workers=8
    --job_name="${JOB_NAME}"

    # ---- Policy ----
    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps=${STEPS}
    --policy.scheduler_decay_lr=5e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.vlm_model_name_or_path=${VLM_MODEL_PATH}
    --policy.enable_vqa_loss=true
    --policy.tokenize_state=true
    --policy.knowledge_insulation=false
    --policy.video_loss_only=false
    --policy.video_loss_weight=1
    --policy.action_loss_only=false
    --policy.freeze_learnable_tokens=false
    --policy.num_learnable_tokens=50
    --policy.wan_checkpoint_path=${WAN_CHECKPOINT_PATH}
    --policy.wan_config_path=${WAN_CONFIG_PATH}
    --policy.vae_path=${WAN_VAE_PATH}

    # ---- Dataset ----
    --dataset.type="$POLICY"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.action_mode="$ACTION_TYPE"
    --dataset.use_external_stats="$USE_EXTERNAL_STATS"
    --dataset.dist_loading=false
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=true

    # ---- Training ----
    --seed=42
    --batch_size=${BATCH_SIZE}
    --steps=${STEPS}
    --save_freq=${SAVE_FREQ}
    --log_freq=${LOG_FREQ}

    # ---- Logging ----
    --wandb.enable=true
    --wandb.project=${POLICY}
    --wandb.mode=offline
)

accelerate launch "${ARGS[@]}"
