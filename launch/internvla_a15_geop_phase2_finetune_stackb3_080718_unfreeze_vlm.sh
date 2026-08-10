#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 2 fine-tune — VLM-unfrozen variant (reference, NOT the default 080718 run)
#
# Diff vs internvla_a15_geop_phase2_finetune_stackb3_080718.sh:
#   - train_expert_only=false          → Qwen3.5 VLM weights become trainable
#   - vlm_lr_scale=0.1                 → VLM LR = 5e-6 (10× lower than experts)
#   - freeze_vision_encoder=true       → only tune language_model + lm_head (conservative)
#
# Gradient routing (verified in tests/test_geop_phase2_vlm_freeze_verify.py):
#   - loss_vlm (FAST/VQA CE)           → updates VLM (visual if unfrozen, LM, lm_head)
#   - loss_action (flow matching)      → blocked from VLM by knowledge_insulation=true
#   - loss_kpt                         → blocked from VLM by knowledge_insulation_kpt=true
#   - loss_video                       → does NOT reach VLM (action_expert path only)
#
# KI / soft-KI notes:
#   - ki_gradient_scale / ki_kpt_gradient_scale exist in config but are NOT wired in
#     modeling_internvla_a1_5.py yet; only hard detach is active.
#   - Keep knowledge_insulation=true unless you accept action→VLM gradient leakage.
#
# LR strategy rationale:
#   - Base LR 5e-5 for action/kpt experts (same as frozen-VLM run)
#   - vlm_lr_scale=0.1 → effective VLM LR 5e-6 to protect pretrained world knowledge
#   - Alternative: freeze_vision_encoder=false + vlm_lr_scale=0.05 for full VLM at 2.5e-6
#
# Risk: FAST-token CE (lambda_vqa=1.0) on robot samples can shift VLM language space;
# monitor loss_fast / loss_subtask and eval success rate vs frozen-VLM baseline.
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
export MASTER_PORT=${MASTER_PORT:-36502}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-${PROJ_ROOT}/outputs/internvla_a1_5/2026_08_04_05_41_51-internvla_a1_5-geop-phase1-kpt-warmup-stackb3-abs/checkpoints/000300/pretrained_model}"
WAN_DIR="${WAN_DIR:-/tmp/hf_home/hub/Wan2.2-TI2V-5B}"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-/tmp/hf_home/lerobot/stats/aloha/abs/agg_1repos_1c27ca3df3/stats.json}"

BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-10000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-50}"

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-geop-phase2-unfreeze-vlm-stackb3-abs-10k}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}}"

echo "PYTHON=${PYTHON}"
echo "PRETRAINED_PATH=${PRETRAINED_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "VLM: train_expert_only=false, vlm_lr_scale=0.1, freeze_vision_encoder=true"

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
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps=1000
    --policy.scheduler_decay_steps=${STEPS}
    --policy.scheduler_decay_lr=5e-6
    --policy.freeze_vision_encoder=true
    --policy.train_expert_only=false
    --policy.vlm_lr_scale=0.1
    --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B
    --policy.enable_vqa_loss=true
    --policy.tokenize_state=true
    --policy.video_loss_only=false
    --policy.video_loss_weight=0.1
    --policy.freeze_learnable_tokens=true
    --policy.num_learnable_tokens=50
    --policy.wan_checkpoint_path="${WAN_DIR}"
    --policy.wan_config_path="${WAN_DIR}"
    --policy.vae_path="${WAN_DIR}/Wan2.2_VAE.pth"

    --policy.enable_keypoint_predictor=true
    --policy.num_keypoint_joints=14
    --policy.action_loss_weight=10.0
    --policy.kpt_loss_weight=0.1
    --policy.kpt_future_loss_weight=0.1
    --policy.kpt_to_action_detach=false
    --policy.knowledge_insulation=true
    --policy.knowledge_insulation_kpt=true
    --policy.ki_gradient_scale=0.0
    --policy.ki_kpt_gradient_scale=0.0
    --policy.freeze_keypoint_modules=false
    --policy.action_expert_lr_scale=1.0
    --policy.kpt_expert_lr_scale=1.0
    --policy.track_encoder_lr_scale=1.0
    --policy.init_kpt_expert_from_action=false
    --policy.action_loss_only=false

    --dataset.type=${POLICY}
    --dataset.repo_id=robotwin/stack_bowls_three_kpt
    --dataset.enable_keypoint_predictor=true
    --dataset.num_keypoint_joints=14
    --dataset.action_mode=abs
    --dataset.use_external_stats=true
    --dataset.external_stats_path="${EXTERNAL_STATS_PATH}"
    --dataset.dist_loading=false
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=true

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
