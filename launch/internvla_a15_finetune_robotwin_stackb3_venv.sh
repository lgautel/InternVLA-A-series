#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# venv-based fine-tune script for InternVLA-A1.5 on RoboTwin stack_bowls_three.
#
# Based on launch/internvla_a15_finetune_libero_venv.sh (verified working) with
# training hyperparameters aligned to launch/internvla_a15_finetune_robotwin.sh,
# then overridden for this run per user request in b/d/p/reprd_rbtwn_stackb3.md:
#   - 8 GPUs, per-GPU batch_size tried 32 first (OOM with WAN+3cams), now 16
#   - steps=10000
#
# See b/d/p/reprd_rbtwn_stackb3.md for full context.
###############################################################################

################################# ENV config ##################################

export HF_HOME="${HF_HOME:-/mnt/r/CKPT/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

VENV_ROOT="${VENV_ROOT:-/mnt/r/VENV/ivla15}"
source "${VENV_ROOT}/bin/activate"

export WANDB_MODE=offline

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-35999}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# USE_LIBUV=0: fall back to the legacy (non-libuv) TCPStore backend to avoid
# potential hangs in PyTorch 2.10's libuv implementation.
# See b/d/p/reprd_liberop_cam_rb.md problem log #8.
export USE_LIBUV=${USE_LIBUV:-0}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
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

# 2. dataset config: single RoboTwin task (LeRobot v3.0 converted copy)
DATASET_REPO_ID="${DATASET_REPO_ID:-robotwin/stack_bowls_three}"
ACTION_TYPE=abs
USE_EXTERNAL_STATS=true
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/agg_1repos_1c27ca3df3/stats.json}"

echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "EXTERNAL_STATS_PATH=${EXTERNAL_STATS_PATH}"

# 3. output configs
BASE_OUTPUT_DIR="outputs/${POLICY}"
PRETRAINED_DETAIL="a15_base"
JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-robotwin-stack_bowls_three-${ACTION_TYPE}-${PRETRAINED_DETAIL}-finetune}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

# User-requested: 8 GPUs, 10k steps. First try batch=32 OOMed (~127GB used +
# 19GB lm_head alloc failed on H200 with WAN video loss + 3 cameras). Default
# now 16 (effective BS=128). Override via BATCH_SIZE=... if needed.
# SAVE_FREQ=2500 -> checkpoints at 2.5k/5k/7.5k/10k.
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-50}"

# dist_loading=false is safer for this 50-episode dataset on 8 ranks
# (avoids per-rank shards becoming too sparse for large per-GPU batch).
DIST_LOADING="${DIST_LOADING:-false}"

echo "STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE} PROC_PER_NODE=${PROC_PER_NODE} DIST_LOADING=${DIST_LOADING}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

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
    --policy.scheduler_warmup_steps=1000
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
    --policy.freeze_learnable_tokens=true
    --policy.num_learnable_tokens=50
    --policy.wan_checkpoint_path=${WAN_CHECKPOINT_PATH}
    --policy.wan_config_path=${WAN_CONFIG_PATH}
    --policy.vae_path=${WAN_VAE_PATH}

    # ---- Dataset ----
    --dataset.type="$POLICY"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.action_mode="$ACTION_TYPE"
    --dataset.use_external_stats="$USE_EXTERNAL_STATS"
    --dataset.external_stats_path=${EXTERNAL_STATS_PATH}
    --dataset.dist_loading=${DIST_LOADING}
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
