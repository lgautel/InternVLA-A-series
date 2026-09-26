#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 2 SFT + Full Data Augmentation (incl. affine@0.2)
# — Franka plug_into_socket 7D keypoints
#
# Based on: b/s/Frk/frk_plug_sft_dtaug_launch.sh (Phase 2 SFT + DataAug)
# Changes:  - All 6 augmentations enabled (affine weight=0.2, rest weight=1.0)
#           - STEPS=20000 (vs 50000)
#           - SAVE_FREQ=5000
#           - EXPR_NAME=itvlagpFrkPlug0907_dtaug2
#           - MASTER_PORT=36704
#
# Why affine weight=0.2: RandomAffine (rotation ±5°, translate ±5%) risks
# misaligning visual observations with action labels. By setting weight=0.2
# vs 1.0 for color/sharpness transforms, affine is sampled with ~3.8%
# normalized probability (vs ~19.2% each for the others), reducing but not
# eliminating spatial augmentation. See b/d/Frk/dtaaug_analyz.markdown.
#
# Usage:
#   bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh            # production 8 GPU
#   SMOKE=1 bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh    # 1 GPU smoke test
#   WAN_SMOKE=1 bash b/d/Frk/expr/sftaug2/frk_plug_sft_dtaug2_launch.sh # WAN smoke
###############################################################################

# ── Virtual environment ───────────────────────────────────────────────────
VENV_ROOT="${VENV_ROOT:-/B/VENV/itnvla15rbt20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
PYTHON="${PYTHON:-${VENV_ROOT}/bin/python}"

# ── Experiment name ───────────────────────────────────────────────────────
EXPR_NAME="${EXPR_NAME:-itvlagpFrkPlug0907_dtaug2ps}"

# ── Environment variables ─────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/B/VENV/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_LIBUV="${USE_LIBUV:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# NCCL: disable GCP tuner plugin (no config → ncclInternalError)
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-libnccl-tuner-disabled.so}"

# Triton cache on local XFS (avoid Ceph multi-rank file lock)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/itvla-triton-cache}"

# LD_LIBRARY_PATH: include NVIDIA NPP (torchcodec dependency)
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/torch/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36704}"

# ── Model & data paths ───────────────────────────────────────────────────
POLICY="internvla_a1_5"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${HOME}/b/Ckp/itvlagpFrkPlug0907/2026_09_07_07_17_08-internvla_a1_5-frk-plug-sft/checkpoints/010420/pretrained_model}"
DATA_SRC="${DATA_SRC:-/B/Dta/plug_into_socket_lrb_4D}"
DATA_REPO_ID="${DATA_REPO_ID:-plug_into_socket_lrb_4D}"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${DATA_SRC}/meta/stats/abs/stats.json}"
WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
VIDEO_MICRO_BATCH_SIZE="${VIDEO_MICRO_BATCH_SIZE:-1}"

# ── Output paths ──────────────────────────────────────────────────────────
CKPT_ROOT="${CKPT_ROOT:-${HOME}/b/Ckp/${EXPR_NAME}}"
LOG_ROOT="${LOG_ROOT:-/B/Log/${EXPR_NAME}}"

# ── Post-training monitoring ─────────────────────────────────────────────
MONITOR_INTERVAL="${MONITOR_INTERVAL:-900}"
STALE_THRESHOLD="${STALE_THRESHOLD:-900}"
BIGMATRIX_SCRIPT="${BIGMATRIX_SCRIPT:-${PROJ_ROOT}/b/d/GpRbt/bigmatrix_multiply_optimization.py}"
BIGMATRIX_MAX_RETRIES="${BIGMATRIX_MAX_RETRIES:-5}"

# ── Image augmentation: full tfs dict with affine weight=0.2 ─────────────
# draccus treats dict fields as atomic — `--dataset.image_transforms.tfs.affine.weight=0.2`
# does NOT work. We must pass the complete tfs dict as a YAML string.
TFS_CONFIG='{brightness: {weight: 1.0, type: ColorJitter, kwargs: {brightness: [0.8, 1.2]}}, contrast: {weight: 1.0, type: ColorJitter, kwargs: {contrast: [0.8, 1.2]}}, saturation: {weight: 1.0, type: ColorJitter, kwargs: {saturation: [0.5, 1.5]}}, hue: {weight: 1.0, type: ColorJitter, kwargs: {hue: [-0.05, 0.05]}}, sharpness: {weight: 1.0, type: SharpnessJitter, kwargs: {sharpness: [0.5, 1.5]}}, affine: {weight: 0.2, type: RandomAffine, kwargs: {degrees: [-5.0, 5.0], translate: [0.05, 0.05]}}, blackout: {weight: 1.0, type: RandomBlackout, kwargs: {noise_scale: 0.01}}}'

# ── p-schedule configuration ─────────────────────────────────────────────
P_SCHEDULE="${P_SCHEDULE:-[0.3, 0.4, 0.5, 0.6, 0.5, 0.4]}"
P_EPOCH_INTERVAL="${P_EPOCH_INTERVAL:-5}"

# ── Smoke vs production ──────────────────────────────────────────────────
WAN_SMOKE="${WAN_SMOKE:-0}"
SMOKE="${SMOKE:-0}"

if [[ "${WAN_SMOKE}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    PROC_PER_NODE="${PROC_PER_NODE:-1}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    STEPS="${STEPS:-2}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    SAVE_FREQ="${SAVE_FREQ:-2}"
    LOG_FREQ="${LOG_FREQ:-1}"
    SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1}"
    WANDB_ENABLE="${WANDB_ENABLE:-false}"
    JOB_SUFFIX="frk-plug-sft-dtaug2ps-wan-smoke"
elif [[ "${SMOKE}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    PROC_PER_NODE="${PROC_PER_NODE:-1}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    STEPS="${STEPS:-10}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    SAVE_FREQ="${SAVE_FREQ:-10}"
    LOG_FREQ="${LOG_FREQ:-1}"
    SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-2}"
    WANDB_ENABLE="${WANDB_ENABLE:-false}"
    JOB_SUFFIX="frk-plug-sft-dtaug2ps-smoke"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    PROC_PER_NODE="${PROC_PER_NODE:-8}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    STEPS="${STEPS:-20000}"
    NUM_WORKERS="${NUM_WORKERS:-12}"
    SAVE_FREQ="${SAVE_FREQ:-5000}"
    LOG_FREQ="${LOG_FREQ:-50}"
    SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-500}"
    WANDB_ENABLE="${WANDB_ENABLE:-true}"
    JOB_SUFFIX="frk-plug-sft-dtaug2ps"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

cd "${PROJ_ROOT}"

JOB_STAMP="${JOB_STAMP:-$(date +'%Y_%m_%d_%H_%M_%S')}"
JOB_NAME="${JOB_NAME:-${JOB_STAMP}-${POLICY}-${JOB_SUFFIX}}"

if [[ "${WAN_SMOKE}" == "1" || "${SMOKE}" == "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sft_dtaug2_smoke_frk_plug_${JOB_STAMP}}"
    LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/${JOB_NAME}}"
    LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${JOB_STAMP}/train.log}"
fi

echo "=== Phase 2 SFT + Full Data Augmentation (incl. affine@0.2): Franka plug_into_socket 7D ==="
echo "EXPR_NAME=${EXPR_NAME}"
echo "PRETRAINED_CKPT=${PRETRAINED_CKPT}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "EXTERNAL_STATS_PATH=${EXTERNAL_STATS_PATH}"
echo "WAN_DIR=${WAN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "SMOKE=${SMOKE} WAN_SMOKE=${WAN_SMOKE} PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS} SAVE_FREQ=${SAVE_FREQ}"
echo "kpt_4d_mode=pos_rot (7D), num_keypoint_joints=8"
echo "Loss weights: action=10.0, kpt=1.0, kpt_future=1.5, video=1.0, vqa=on"
echo "Image augmentation: ALL 6 (brightness/contrast/saturation/hue/sharpness/affine)"
echo "  affine weight=0.2 (normalized p≈3.8%), others weight=1.0 (normalized p≈19.2% each)"
echo "  p_schedule=${P_SCHEDULE}, change every ${P_EPOCH_INTERVAL} epochs"

mkdir -p "$(dirname "${LOG_FILE}")"

# ── accelerate args ───────────────────────────────────────────────────────
LAUNCH_ARGS=()
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    LAUNCH_ARGS+=(--multi_gpu)
fi
LAUNCH_ARGS+=(
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
)

ARGS=(
    "${LAUNCH_ARGS[@]}"
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS}"
    --job_name="${JOB_NAME}"

    # ── Model & starting point (SFT ckpt@10420) ──
    --policy.type="${POLICY}"
    --policy.repo_id=lerobot_lab/"${POLICY}"
    --policy.push_to_hub=false
    --policy.pretrained_path="${PRETRAINED_CKPT}"
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16
    --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B

    # ── Optimizer ──
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP}"
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5e-6

    # ── Phase 2: full fine-tuning (VLM unfrozen) ──
    --policy.train_expert_only=false
    --policy.knowledge_insulation=false
    --policy.knowledge_insulation_kpt=false
    --policy.freeze_vision_encoder=false
    --policy.enable_vqa_loss=true
    --policy.tokenize_state=true

    # ── WAN video foresight ──
    --policy.action_loss_only=false
    --policy.video_loss_weight=1
    --policy.video_loss_only=false
    --policy.freeze_wan_dit=true
    --policy.freeze_learnable_tokens=true
    --policy.num_learnable_tokens=50
    --policy.wan_checkpoint_path="${WAN_DIR}"
    --policy.wan_config_path="${WAN_DIR}"
    --policy.vae_path="${WAN_DIR}/Wan2.2_VAE.pth"
    --policy.video_micro_batch_size="${VIDEO_MICRO_BATCH_SIZE}"

    # ── 7D keypoints (Franka 8 joints) ──
    --policy.enable_keypoint_predictor=true
    --policy.num_keypoint_joints=8
    --policy.kpt_4d_mode=pos_rot
    --policy.kpt_rot_loss_weight=1.0
    --policy.keypoint_history_max_len=200

    # ── Phase 2 loss weights ──
    --policy.action_loss_weight=10.0
    --policy.kpt_loss_weight=1.0
    --policy.kpt_future_loss_weight=1.5
    --policy.kpt_to_action_detach=false

    # ── No re-init from action expert / no GeoPredict reload ──
    --policy.init_kpt_expert_from_action=false

    # ── Learning rate groups (all at full speed) ──
    --policy.freeze_keypoint_modules=false
    --policy.action_expert_lr_scale=1.0
    --policy.kpt_expert_lr_scale=1.0
    --policy.track_encoder_lr_scale=1.0

    # ── Dataset ──
    --dataset.type="${POLICY}"
    --dataset.repo_id="${DATA_REPO_ID}"
    --dataset.enable_keypoint_predictor=true
    --dataset.num_keypoint_joints=8
    --dataset.kpt_4d_mode=pos_rot
    --dataset.keypoint_history_max_len=200
    --dataset.action_mode=abs
    --dataset.use_external_stats=true
    --dataset.external_stats_path="${EXTERNAL_STATS_PATH}"
    --dataset.dist_loading=false
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=true
    --dataset.video_backend=torchcodec

    # ── Image augmentation: ALL 6 transforms, affine at weight=0.2, p-schedule ──
    --dataset.image_transforms.enable=true
    --dataset.image_transforms.max_num_transforms=3
    --dataset.image_transforms.random_order=false
    "--dataset.image_transforms.p_schedule=${P_SCHEDULE}"
    "--dataset.image_transforms.p_epoch_interval=${P_EPOCH_INTERVAL}"
    '--dataset.image_transforms.disabled_tfs=[]'
    "--dataset.image_transforms.tfs=${TFS_CONFIG}"

    # ── Training ──
    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    # ── WandB ──
    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${POLICY}"
    --wandb.mode=offline
)

###############################################################################
# Monitoring helpers (adapted from b/s/Frk/frk_plug_sft_dtaug_launch.sh)
###############################################################################

_monitor_ts() { date +'%y%m%d%H'; }
_monitor_log() {
    local msg="[monitor $(date +'%H:%M:%S')] $*"
    echo "${msg}"
    echo "${msg}" >> "${LOG_FILE}" 2>/dev/null || true
}

_is_log_stale() {
    [[ ! -f "${LOG_FILE}" ]] && return 0
    local age=$(( $(date +%s) - $(stat -c %Y "${LOG_FILE}") ))
    [[ ${age} -gt ${STALE_THRESHOLD} ]]
}

_are_outputs_complete() {
    local final_step
    final_step=$(printf "%06d" "${STEPS}")
    [[ -d "${OUTPUT_DIR}/checkpoints/${final_step}/pretrained_model" ]]
}

_kill_gpu_processes() {
    _monitor_log "Killing GPU processes..."
    [[ -n "${TRAIN_PID:-}" ]] && kill "${TRAIN_PID}" 2>/dev/null || true
    pkill -f "lerobot_train" 2>/dev/null || true
    pkill -f "accelerate.commands.launch" 2>/dev/null || true
    sleep 5
    pkill -9 -f "lerobot_train" 2>/dev/null || true
    pkill -9 -f "accelerate.commands.launch" 2>/dev/null || true
    sleep 2
}

_start_bigmatrix() {
    _monitor_log "Starting bigmatrix_multiply_optimization.py..."
    local retry=0
    while [[ ${retry} -lt ${BIGMATRIX_MAX_RETRIES} ]]; do
        nohup "${PYTHON}" -u "${BIGMATRIX_SCRIPT}" \
            > /tmp/bigmatrix_multiply_optimization.log 2>&1 &
        local bg_pid=$!
        disown "${bg_pid}" 2>/dev/null || true
        sleep 15
        if kill -0 "${bg_pid}" 2>/dev/null; then
            _monitor_log "bigmatrix started (PID=${bg_pid})"
            return 0
        fi
        _monitor_log "bigmatrix attempt $((retry + 1))/${BIGMATRIX_MAX_RETRIES} failed, retrying..."
        retry=$((retry + 1))
        sleep 5
    done
    _monitor_log "ERROR: bigmatrix failed after ${BIGMATRIX_MAX_RETRIES} attempts"
    return 1
}

_archive_and_cleanup() {
    local suffix="$1"

    _kill_gpu_processes
    _start_bigmatrix || true

    local ts
    ts=$(_monitor_ts)
    local archive_name="${EXPR_NAME}_LOG_${ts}${suffix}"
    local archive_dest="${HOME}/b/Ckp"
    mkdir -p "${archive_dest}"
    _monitor_log "Archiving ${LOG_ROOT} → ${archive_dest}/${archive_name}.tar"

    tar -cf "${archive_dest}/${archive_name}.tar" \
        -C "$(dirname "${LOG_ROOT}")" "$(basename "${LOG_ROOT}")" 2>/dev/null || true
    _monitor_log "Archive done: ${archive_dest}/${archive_name}.tar ($(du -sh "${archive_dest}/${archive_name}.tar" 2>/dev/null | cut -f1))"
}

###############################################################################
# Training execution
###############################################################################

if [[ "${WAN_SMOKE}" == "1" || "${SMOKE}" == "1" ]]; then
    # ── Smoke mode: blocking execution, no monitoring ──
    set -o pipefail
    "${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    train_exit=${PIPESTATUS[0]}
    decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" 2>/dev/null) || decode_err=0
    zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" 2>/dev/null) || zero_frames=0
    echo "post_check: video_decode_error=${decode_err} using_zeros=${zero_frames} exit=${train_exit}"
    exit "${train_exit}"
fi

# ── Formal training: background execution + automated monitoring ──
set +e

_monitor_log "=== Franka SFT+FullDataAug (affine@0.2) automated monitoring enabled ==="
_monitor_log "EXPR_NAME=${EXPR_NAME}  INTERVAL=${MONITOR_INTERVAL}s  STALE=${STALE_THRESHOLD}s"
_monitor_log "OUTPUT_DIR=${OUTPUT_DIR}"
_monitor_log "CKPT_ROOT=${CKPT_ROOT}  LOG_ROOT=${LOG_ROOT}"

"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
_monitor_log "Training started (PID=${TRAIN_PID})"
_monitor_log "Log: tail -f ${LOG_FILE}"

_poll_sec=60
_elapsed=0

while true; do
    sleep ${_poll_sec}
    _elapsed=$((_elapsed + _poll_sec))

    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        wait "${TRAIN_PID}" 2>/dev/null
        train_exit=$?
        _monitor_log "Training exited (code=${train_exit})"
        sleep 10

        if [[ "${train_exit}" -eq 0 ]] && _are_outputs_complete; then
            _monitor_log "SUCCESS: Training completed normally"
            _archive_and_cleanup ""
        else
            _monitor_log "ERROR: exit=${train_exit}, outputs_complete=$(_are_outputs_complete && echo yes || echo no)"
            _archive_and_cleanup "_err"
        fi
        break
    fi

    if [[ ${_elapsed} -ge ${MONITOR_INTERVAL} ]]; then
        _elapsed=0
        if _is_log_stale; then
            _monitor_log "ERROR: Log stale >${STALE_THRESHOLD}s while PID=${TRAIN_PID} alive"
            _archive_and_cleanup "_err"
            break
        fi
        cur_step=$(grep -oE 'step:[0-9]+' "${LOG_FILE}" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        _monitor_log "Healthy: step=${cur_step:-?}/${STEPS}"
    fi
done

_monitor_log "Monitoring finished"
