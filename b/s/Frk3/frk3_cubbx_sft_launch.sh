#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Franka CubInBx Phase 2 SFT Launch Script
# ============================================================
# EXPR_NAME: 4dwvlaFrkCubBx0924
# Dataset:   put_cube_into_box_lrb3_4D (56 ep, 36953 frames)
# Starting:  Warmup ckpt@1156
# Training:  20 epochs = 5780 steps, save every 2 epochs
# Features:  WAN video + data augmentation + p_schedule
# ============================================================

EXPR_NAME="4dwvlaFrkCubBx0924"
DATASET_REPO_ID="put_cube_into_box_lrb3_4D"

# ── Warmup checkpoint ──
PRETRAINED_CKPT="${PRETRAINED_CKPT:-/home/a26113/b/Ckp/4dwvlaFrkCubBx0924/2026_09_25_01_12_18-internvla_a1_5-frk3-cubbx-warmup/checkpoints/001156/pretrained_model}"

# ── Training parameters ──
STEPS="${STEPS:-5780}"
SAVE_FREQ="${SAVE_FREQ:-578}"
LOG_FREQ="${LOG_FREQ:-50}"
LR="${LR:-5e-5}"
DECAY_LR="${DECAY_LR:-5e-6}"
SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-289}"
SCHEDULER_DECAY="${SCHEDULER_DECAY:-$STEPS}"
MASTER_PORT="${MASTER_PORT:-36705}"

# ── Data augmentation ──
P_SCHEDULE="${P_SCHEDULE:-[0.2, 0.4, 0.6, 0.5, 0.3]}"
P_EPOCH_INTERVAL="${P_EPOCH_INTERVAL:-2}"

TFS_CONFIG='{brightness: {type: ColorJitter, weight: 1.0, kwargs: {brightness: [0.8, 1.2]}}, contrast: {type: ColorJitter, weight: 1.0, kwargs: {contrast: [0.8, 1.2]}}, saturation: {type: ColorJitter, weight: 1.0, kwargs: {saturation: [0.5, 1.5]}}, hue: {type: ColorJitter, weight: 1.0, kwargs: {hue: [-0.05, 0.05]}}, sharpness: {type: SharpnessJitter, weight: 1.0, kwargs: {sharpness: [0.5, 1.5]}}, affine: {type: RandomAffine, weight: 0.2, kwargs: {degrees: [-5, 5], translate: [0.05, 0.05], scale: [0.95, 1.05]}}, blackout: {type: RandomBlackout, weight: 1.0, kwargs: {noise_scale: 0.01}}}'

# ── WAN paths ──
WAN_PATH="${WAN_PATH:-/B/VENV/hf_home/hub/Wan2.2-TI2V-5B}"

# ── Environment ──
source /B/VENV/itnvla15rbt20/bin/activate
export HF_HOME="/B/VENV/hf_home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_TUNER_PLUGIN=/dev/null
VENV_ROOT="/B/VENV/itnvla15rbt20"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/torch/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

# ── Mode selection ──
if [[ "${WAN_SMOKE:-}" == "1" ]]; then
    MODE="WAN_SMOKE"
    PROC_PER_NODE=8
    BATCH_SIZE=2
    STEPS=2
    SAVE_FREQ=2
    NUM_WORKERS=2
    ACTION_LOSS_ONLY=false
elif [[ "${SMOKE:-}" == "1" ]]; then
    MODE="SMOKE"
    PROC_PER_NODE=1
    BATCH_SIZE=2
    STEPS=10
    SAVE_FREQ=10
    NUM_WORKERS=2
    ACTION_LOSS_ONLY=true
else
    MODE="PRODUCTION"
    PROC_PER_NODE="${PROC_PER_NODE:-8}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    NUM_WORKERS="${NUM_WORKERS:-12}"
    ACTION_LOSS_ONLY=false
fi

# ── Paths ──
JOB_STAMP=$(date +%Y_%m_%d_%H_%M_%S)

if [[ "$MODE" == "SMOKE" || "$MODE" == "WAN_SMOKE" ]]; then
    OUTPUT_DIR="/tmp/sft_smoke_frk3_cubbx_${JOB_STAMP}"
    JOB_NAME="${JOB_STAMP}-internvla_a1_5-frk3-cubbx-sft-smoke"
    LOG_DIR="/tmp"
else
    OUTPUT_DIR="$HOME/b/Ckp/${EXPR_NAME}/${JOB_STAMP}-internvla_a1_5-frk3-cubbx-sft"
    JOB_NAME="${JOB_STAMP}-internvla_a1_5-frk3-cubbx-sft"
    LOG_DIR="/B/Log/${EXPR_NAME}/${JOB_STAMP}"
    mkdir -p "$LOG_DIR"
fi

LOG_FILE="${LOG_DIR}/train.log"

# ── Print config ──
echo "============================================================"
echo "  Franka CubInBx Phase 2 SFT"
echo "============================================================"
echo "Mode:           $MODE"
echo "EXPR_NAME:      $EXPR_NAME"
echo "PRETRAINED_CKPT: $PRETRAINED_CKPT"
echo "DATASET:        $DATASET_REPO_ID"
echo "GPUs:           $PROC_PER_NODE"
echo "Batch size:     $BATCH_SIZE (EBS=$(($BATCH_SIZE * $PROC_PER_NODE)))"
echo "Steps:          $STEPS"
echo "Save freq:      $SAVE_FREQ"
echo "LR:             $LR → $DECAY_LR (warmup=$SCHEDULER_WARMUP)"
echo "Action loss only: $ACTION_LOSS_ONLY"
echo "Data augmentation: 7 transforms, p_schedule=$P_SCHEDULE, interval=$P_EPOCH_INTERVAL"
echo "Output:         $OUTPUT_DIR"
echo "Log:            $LOG_FILE"
echo "============================================================"

# ── Build ARGS ──
ARGS=(
    --policy.type=internvla_a1_5
    --policy.pretrained_path="$PRETRAINED_CKPT"
    --policy.repo_id=lerobot_lab/internvla_a1_5
    --policy.push_to_hub=false

    # ── SFT: Unfreeze VLM + load WAN ──
    --policy.train_expert_only=false
    --policy.action_loss_only=$ACTION_LOSS_ONLY
    --policy.enable_vqa_loss=false
    --policy.gradient_checkpointing=true

    # ── Disable knowledge insulation ──
    --policy.knowledge_insulation=false
    --policy.knowledge_insulation_kpt=false

    # ── Loss weights ──
    --policy.action_loss_weight=10.0
    --policy.kpt_loss_weight=1.0
    --policy.kpt_future_loss_weight=1.5
    --policy.video_loss_weight=1.0
    --policy.lambda_vqa=1.0

    # ── LR scales (all 1.0 for SFT) ──
    --policy.vlm_lr_scale=1.0
    --policy.action_expert_lr_scale=1.0
    --policy.kpt_expert_lr_scale=1.0
    --policy.track_encoder_lr_scale=1.0

    # ── Keypoint config ──
    --policy.enable_keypoint_predictor=true
    --policy.num_keypoint_joints=8
    --policy.kpt_4d_mode=pos_rot
    --policy.keypoint_history_max_len=92
    --policy.init_kpt_expert_from_action=false
    --policy.freeze_keypoint_modules=false

    # ── Frozen modules ──
    --policy.freeze_wan_dit=true
    --policy.freeze_learnable_tokens=true

    # ── Video branch ──
    --policy.num_video_frames=4
    --policy.video_height=224
    --policy.video_width=224
    --policy.video_micro_batch_size=1

    # ── WAN paths ──
    --policy.wan_checkpoint_path="$WAN_PATH"
    --policy.wan_config_path="$WAN_PATH"
    --policy.vae_path="$WAN_PATH/Wan2.2_VAE.pth"

    # ── Optimizer / Scheduler ──
    --policy.optimizer_lr=$LR
    --policy.optimizer_weight_decay=0.01
    --policy.optimizer_grad_clip_norm=1.0
    --policy.scheduler_warmup_steps=$SCHEDULER_WARMUP
    --policy.scheduler_decay_steps=$SCHEDULER_DECAY
    --policy.scheduler_decay_lr=$DECAY_LR

    # ── Dataset ──
    --dataset.type=internvla_a1_5
    --dataset.repo_id=$DATASET_REPO_ID
    --dataset.action_mode=abs
    --dataset.use_external_stats=false
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=true
    --dataset.video_backend=torchcodec

    # ── Keypoint data config ──
    --dataset.enable_keypoint_predictor=true
    --dataset.num_keypoint_joints=8
    --dataset.kpt_4d_mode=pos_rot
    --dataset.keypoint_history_max_len=92

    # ── Image augmentation ──
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=3
    --dataset.image_transforms.random_order=false
    "--dataset.image_transforms.p_schedule=${P_SCHEDULE}"
    "--dataset.image_transforms.p_epoch_interval=${P_EPOCH_INTERVAL}"
    '--dataset.image_transforms.disabled_tfs=[]'
    "--dataset.image_transforms.tfs=${TFS_CONFIG}"

    # ── Training ──
    --steps=$STEPS
    --batch_size=$BATCH_SIZE
    --save_freq=$SAVE_FREQ
    --log_freq=$LOG_FREQ
    --num_workers=$NUM_WORKERS
    --seed=42

    # ── WandB ──
    --wandb.enable=true
    --wandb.project=$EXPR_NAME
    --wandb.mode=offline

    # ── Output ──
    --output_dir=$OUTPUT_DIR
    --job_name=$JOB_NAME
)

# SMOKE mode overrides
if [[ "$ACTION_LOSS_ONLY" == "true" ]]; then
    ARGS+=(--policy.action_loss_only=true)
    ARGS+=(--policy.enable_vqa_loss=false)
fi

# ── Launch training ──
echo "[$(date)] Starting training..."

python -m accelerate.commands.launch \
    --num_processes=$PROC_PER_NODE \
    --num_machines=1 \
    --main_process_port=$MASTER_PORT \
    src/lerobot/scripts/lerobot_train.py \
    "${ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"

TRAIN_EXIT=$?

echo "[$(date)] Training exited with code: $TRAIN_EXIT"

# ── Post-check ──
if [[ "$MODE" == "SMOKE" || "$MODE" == "WAN_SMOKE" ]]; then
    echo "[Post-check] Smoke mode — checking results..."
    if grep -q "video_decode_error" "$LOG_FILE"; then
        VDE=$(grep "video_decode_error" "$LOG_FILE" | tail -1 | grep -oP 'video_decode_error:\K[0-9.]+' || echo "?")
        echo "  video_decode_error: $VDE"
    fi
    if grep -q "using_zeros" "$LOG_FILE"; then
        UZ=$(grep "using_zeros" "$LOG_FILE" | tail -1 | grep -oP 'using_zeros:\K[0-9.]+' || echo "?")
        echo "  using_zeros: $UZ"
    fi
    echo "[$(date)] Smoke test done. Exit=$TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Production post-processing ──
if [[ $TRAIN_EXIT -eq 0 ]]; then
    echo "============================================================"
    echo "  TRAINING COMPLETED SUCCESSFULLY"
    echo "============================================================"

    # Start GPU placeholder
    if [[ -f "b/d/GpRbt/bigmatrix_multiply_optimization.py" ]]; then
        echo "Starting bigmatrix GPU placeholder..."
        nohup python b/d/GpRbt/bigmatrix_multiply_optimization.py > /dev/null 2>&1 &
        disown
        echo "bigmatrix PID: $!"
    fi

    # Archive logs
    TAR_NAME="${HOME}/b/Ckp/${EXPR_NAME}_sft_LOG_$(date +%Y%m%d_%H%M%S).tar"
    tar -cf "$TAR_NAME" \
        -C "$(dirname "$LOG_FILE")" "$(basename "$LOG_FILE")" \
        2>/dev/null || true
    echo "Logs archived to: $TAR_NAME"

else
    echo "============================================================"
    echo "  TRAINING FAILED (exit=$TRAIN_EXIT)"
    echo "============================================================"

    # Clean up GPU processes
    pkill -f lerobot_train || true
    sleep 5

    # Start GPU placeholder
    if [[ -f "b/d/GpRbt/bigmatrix_multiply_optimization.py" ]]; then
        nohup python b/d/GpRbt/bigmatrix_multiply_optimization.py > /dev/null 2>&1 &
        disown
    fi

    # Archive logs with _err suffix
    TAR_NAME="${HOME}/b/Ckp/${EXPR_NAME}_sft_LOG_$(date +%Y%m%d_%H%M%S)_err.tar"
    tar -cf "$TAR_NAME" \
        -C "$(dirname "$LOG_FILE")" "$(basename "$LOG_FILE")" \
        2>/dev/null || true
    echo "Error logs archived to: $TAR_NAME"
fi

exit $TRAIN_EXIT
