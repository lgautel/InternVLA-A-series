#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Experiment B Phase 1: Keypoint Expert Warmup on R1 Pro
#
# Based on: launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh
# Changes:  num_keypoint_joints 14->16, dataset -> R1 Pro, history_max_len 300
#
# See: b/d/r1pro_migration_design.md §7.3
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
export MASTER_PORT=${MASTER_PORT:-36602}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
GEOPREDICT_CKPT="${GEOPREDICT_CKPT:-${HF_HOME}/ckpts/GeoPredict_robocasa.pth}"
# repo_id is resolved as HF_LEROBOT_HOME/<repo_id>, so the keypoint dataset produced by
# util_scripts/generate_r1pro_keypoints.py must be written to ${HF_LEROBOT_HOME}/${DATA_REPO_ID}.
DATA_REPO_ID="${DATA_REPO_ID:-open0630_mj_clean_kpt16}"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${HF_LEROBOT_HOME}/stats/abs/${DATA_REPO_ID}/stats.json}"

BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-100}"
LOG_FREQ="${LOG_FREQ:-10}"

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-r1pro-geop-phase1-kpt-warmup}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}}"

echo "=== Experiment B Phase 1: Keypoint Warmup ==="
echo "PRETRAINED_PATH=${PRETRAINED_PATH}"
echo "GEOPREDICT_CKPT=${GEOPREDICT_CKPT}"
echo "DATA=${DATA_REPO_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "STEPS=${STEPS} BS=${BATCH_SIZE}"
echo "loss: kpt=10 action=2 kpt_future=2 | expert-only, action_loss_only=true"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers=8
    --job_name="${JOB_NAME}"

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.push_to_hub=false
    --policy.pretrained_path="${PRETRAINED_PATH}"
    --policy.geopredict_checkpoint_path="${GEOPREDICT_CKPT}"
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps=50
    --policy.scheduler_decay_steps=${STEPS}
    --policy.scheduler_decay_lr=5e-6
    --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B
    --policy.train_expert_only=true
    --policy.knowledge_insulation=true
    --policy.knowledge_insulation_kpt=true
    --policy.enable_vqa_loss=false
    --policy.tokenize_state=true
    --policy.video_loss_weight=1
    --policy.freeze_learnable_tokens=true
    --policy.num_learnable_tokens=50
    --policy.action_loss_only=true
    --policy.keypoint_history_max_len=300

    --policy.enable_keypoint_predictor=true
    --policy.num_keypoint_joints=16
    --policy.kpt_loss_weight=10.0
    --policy.action_loss_weight=2.0
    --policy.kpt_future_loss_weight=2.0
    --policy.kpt_to_action_detach=false
    --policy.freeze_keypoint_modules=false
    --policy.action_expert_lr_scale=0.04
    --policy.kpt_expert_lr_scale=1.0
    --policy.track_encoder_lr_scale=1.0
    --policy.init_kpt_expert_from_action=true

    --dataset.type=${POLICY}
    --dataset.repo_id="${DATA_REPO_ID}"
    --dataset.enable_keypoint_predictor=true
    --dataset.num_keypoint_joints=16
    --dataset.action_mode=abs
    --dataset.use_external_stats=true
    --dataset.external_stats_path="${EXTERNAL_STATS_PATH}"
    --dataset.dist_loading=false
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=false

    --seed=42
    --batch_size=${BATCH_SIZE}
    --steps=${STEPS}
    --save_freq=${SAVE_FREQ}
    --log_freq=${LOG_FREQ}

    --wandb.enable=true
    --wandb.project=${POLICY}
    --wandb.mode=offline
)

"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}"
