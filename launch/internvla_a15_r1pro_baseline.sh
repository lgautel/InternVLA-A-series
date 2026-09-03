#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Experiment A: InternVLA-A1.5 baseline on R1 Pro (no GeoPredict)
#
# All hyperparams mirror Phase 2 of Experiment B exactly, except:
#   enable_keypoint_predictor=false
#
# The three flags Phase 2 sets explicitly but this script omits (ki_gradient_scale,
# action_expert_lr_scale, freeze_vision_encoder) have defaults equal to Phase 2's
# explicit values, so the two runs really do differ in one variable only.
#
# KNOWN CONFOUND: A starts from the base checkpoint, B starts from Phase 1's output,
# so B sees 400 extra steps of R1 Pro data. Phase 1 runs with
# action_expert_lr_scale=0.04, so the action expert barely moves, but the asymmetry
# is real and belongs in any writeup of the comparison.
#
# See: b/d/r1pro_migration_design.md §7.2
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
export MASTER_PORT=${MASTER_PORT:-36601}
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
WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
# repo_id is resolved as HF_LEROBOT_HOME/<repo_id>, so the keypoint dataset produced by
# util_scripts/generate_r1pro_keypoints.py must be written to ${HF_LEROBOT_HOME}/${DATA_REPO_ID}.
DATA_REPO_ID="${DATA_REPO_ID:-open0630_mj_clean_kpt16}"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${HF_LEROBOT_HOME}/stats/abs/${DATA_REPO_ID}/stats.json}"

BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-50}"

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-r1pro-baseline-abs-${STEPS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}}"

echo "=== Experiment A: R1 Pro Baseline (no GeoPredict) ==="
echo "PRETRAINED_PATH=${PRETRAINED_PATH}"
echo "DATA=${DATA_REPO_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "STEPS=${STEPS} BS=${BATCH_SIZE}"

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
    --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B
    --policy.train_expert_only=true
    --policy.knowledge_insulation=true
    --policy.enable_vqa_loss=false
    --policy.tokenize_state=true
    --policy.video_loss_only=false
    --policy.video_loss_weight=0.0
    --policy.action_loss_only=false
    --policy.action_loss_weight=10.0
    --policy.freeze_learnable_tokens=false
    --policy.num_learnable_tokens=50
    --policy.wan_checkpoint_path="${WAN_DIR}"
    --policy.wan_config_path="${WAN_DIR}"
    --policy.vae_path="${WAN_DIR}/Wan2.2_VAE.pth"
    --policy.keypoint_history_max_len=300

    --policy.enable_keypoint_predictor=false

    --dataset.type=${POLICY}
    --dataset.repo_id="${DATA_REPO_ID}"
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
