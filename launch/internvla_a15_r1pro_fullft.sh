#!/bin/bash
# InternVLA-A1.5 R1 Pro 全模型微调（修正版）
# 关键改动：train_expert_only=false, knowledge_insulation=false,
#          开启 VQA/video/FAST 辅助损失，训练整个 VLM+action expert
#
# 用法：
#   cd /home/a26215/InternVLA-A
#   nohup bash launch/internvla_a15_r1pro_fullft.sh > outputs/fullft.log 2>&1 &
set -euo pipefail

# ── Crater 环境 ──
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export PATH="/usr/local/nvidia/bin:$PATH"
export NCCL_TIMEOUT=1800
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
echo "" > /tmp/nccl_tuner.pb
export NCCL_TUNER_CONFIG_PATH=/tmp/nccl_tuner.pb
export CC=gcc
export WANDB_MODE=disabled
export WANDB__REQUIRE_CORE=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# ── 路径 ──
PROJ_ROOT="/home/a26215/InternVLA-A"
VENV_PYTHON="/tmp/venv_r1pro/bin/python"
PRETRAINED_PATH="/home/a26215/InternVLA-A/pretrained_weights/InternVLA-A1.5-base"
DATASET_REPO_ID="open0630_mj_clean_kpt16"
STATS_PATH="/tmp/venv_r1pro/var/hf_home/lerobot/stats/abs/${DATASET_REPO_ID}/stats.json"
HF_LEROBOT_HOME="/tmp/venv_r1pro/var/hf_home/lerobot"

# ── 训练参数 ──
STEPS="${STEPS:-25000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LR="${LR:-5e-5}"
WARMUP="${WARMUP:-1000}"

# ── GPU 计数 ──
GPU_COUNT=$(${VENV_PYTHON} -c "import torch; print(torch.cuda.device_count())")
echo "GPUs: $GPU_COUNT"

TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S)
JOB_NAME="r1pro-fullft"
OUTPUT_DIR="${PROJ_ROOT}/outputs/internvla_a1_5/${TIMESTAMP}-${JOB_NAME}"

cd "$PROJ_ROOT"
export HF_LEROBOT_HOME

${VENV_PYTHON} -m accelerate.commands.launch \
    --num_processes=${GPU_COUNT} \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    src/lerobot/scripts/lerobot_train.py \
    --output_dir="${OUTPUT_DIR}" \
    --num_workers=8 \
    --job_name="${JOB_NAME}" \
    \
    --policy.type=internvla_a1_5 \
    --policy.repo_id=lerobot_lab/internvla_a1_5 \
    --policy.push_to_hub=false \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.optimizer_lr=${LR} \
    --policy.scheduler_warmup_steps=${WARMUP} \
    --policy.scheduler_decay_steps=${STEPS} \
    --policy.scheduler_decay_lr=5e-6 \
    --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B \
    \
    --policy.train_expert_only=false \
    --policy.knowledge_insulation=false \
    --policy.freeze_vision_encoder=false \
    --policy.enable_vqa_loss=true \
    --policy.tokenize_state=true \
    --policy.video_loss_weight=1.0 \
    --policy.action_loss_only=false \
    --policy.video_loss_only=false \
    --policy.freeze_learnable_tokens=true \
    --policy.num_learnable_tokens=50 \
    --policy.wan_checkpoint_path=/tmp/Wan2.2-TI2V-5B \
    --policy.wan_config_path=/tmp/Wan2.2-TI2V-5B \
    --policy.vae_path=/tmp/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
    \
    --policy.enable_keypoint_predictor=false \
    \
    --dataset.type=internvla_a1_5 \
    --dataset.repo_id=${DATASET_REPO_ID} \
    --dataset.action_mode=abs \
    --dataset.use_external_stats=true \
    --dataset.external_stats_path=${STATS_PATH} \
    --dataset.dist_loading=false \
    --dataset.tokenize_state=true \
    --dataset.use_fast_action_tokens=true \
    --dataset.video_backend=pyav \
    \
    --seed=42 \
    --batch_size=${BATCH_SIZE} \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --log_freq=50 \
    --wandb.enable=true \
    --wandb.project=internvla_a1_5 \
    --wandb.mode=offline
