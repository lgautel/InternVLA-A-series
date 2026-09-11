#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 2 GeoP SFT — LIBERO 4-suite, 8×7D keypoints (kpt_4d_mode=pos_rot)
#
# Full fine-tune from Phase 1 warmup checkpoint with WAN video foresight.
# Requires WARMUP_CKPT pointing to Phase 1 output, e.g.:
#   export WARMUP_CKPT=outputs/internvla_a1_5/<warmup_job>/checkpoints/000400/pretrained_model
#
# Usage:
#   export WARMUP_CKPT=/path/to/warmup/checkpoints/000400/pretrained_model
#   bash launch/internvla_a15_finetune_libero_geop.sh
#   SMOKE=1 WARMUP_CKPT=... bash launch/internvla_a15_finetune_libero_geop.sh
#   WAN_SMOKE=1 WARMUP_CKPT=... bash launch/internvla_a15_finetune_libero_geop.sh
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export HF_HOME="${HF_HOME:-/tmp/zwy}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_LIBUV="${USE_LIBUV:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-libnccl-tuner-disabled.so}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/itvla-triton-cache}"

CONDA_ENV="${CONDA_ENV:-internvla_a1_5}"
CONDA_ROOT="${CONDA_ROOT:-/opt/conda}"

if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX}")" != "${CONDA_ENV}" ]]; then
  if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  else
    echo "ERROR: ${CONDA_ENV} not active and conda not found at ${CONDA_ROOT}. Run: conda activate ${CONDA_ENV}" >&2
    exit 1
  fi
fi

PYTHON="${PYTHON:-python}"
PY_SITE="${CONDA_PREFIX}/lib/python3.11/site-packages"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${CONDA_PREFIX}/lib:\
${PY_SITE}/torch/lib:\
${PY_SITE}/nvidia/cuda_runtime/lib:\
${PY_SITE}/nvidia/cuda_nvrtc/lib:\
${PY_SITE}/nvidia/npp/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

WARMUP_CKPT="${WARMUP_CKPT:?请先 export WARMUP_CKPT=<Phase1 Warmup checkpoint 路径>}"

POLICY="internvla_a1_5"

# Phase 2 requires Wan2.2-TI2V-5B (config.json + weights + VAE).
_wan_preflight() {
  local missing=0
  if [[ ! -f "${WAN_CHECKPOINT_PATH}/config.json" ]]; then
    echo "ERROR: WAN config.json missing: ${WAN_CHECKPOINT_PATH}/config.json" >&2
    missing=1
  fi
  if [[ ! -f "${WAN_VAE_PATH}" ]]; then
    echo "ERROR: WAN VAE missing: ${WAN_VAE_PATH}" >&2
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    cat >&2 <<EOF
Download Wan2.2-TI2V-5B into ${WAN_CHECKPOINT_PATH} (~32 GB):
  hf download Wan-AI/Wan2.2-TI2V-5B --local-dir ${WAN_CHECKPOINT_PATH}
EOF
    exit 1
  fi
}
PRETRAINED_DETAIL="geop_warmup"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3.5-2B}"
WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${WAN_CHECKPOINT_PATH}}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_CHECKPOINT_PATH}/Wan2.2_VAE.pth}"
VIDEO_MICRO_BATCH_SIZE="${VIDEO_MICRO_BATCH_SIZE:-4}"

DATASET_REPO_ID="${DATASET_REPO_ID:-libero_spatial libero_object libero_goal libero_10}"
ACTION_TYPE=abs
USE_EXTERNAL_STATS=false

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36802}"

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
  JOB_SUFFIX="libero-geop-wan-smoke"
elif [[ "${SMOKE}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PROC_PER_NODE="${PROC_PER_NODE:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  STEPS="${STEPS:-100}"
  NUM_WORKERS="${NUM_WORKERS:-2}"
  SAVE_FREQ="${SAVE_FREQ:-100}"
  LOG_FREQ="${LOG_FREQ:-10}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-50}"
  WANDB_ENABLE="${WANDB_ENABLE:-false}"
  JOB_SUFFIX="libero-geop-sft-smoke"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  PROC_PER_NODE="${PROC_PER_NODE:-8}"
  BATCH_SIZE="${BATCH_SIZE:-32}"
  STEPS="${STEPS:-100000}"
  NUM_WORKERS="${NUM_WORKERS:-32}"
  SAVE_FREQ="${SAVE_FREQ:-5000}"
  LOG_FREQ="${LOG_FREQ:-200}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-2000}"
  WANDB_ENABLE="${WANDB_ENABLE:-true}"
  JOB_SUFFIX="libero-geop-sft"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

cd "${PROJ_ROOT}"

_wan_preflight

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-libero-${ACTION_TYPE}-${PRETRAINED_DETAIL}-${JOB_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

echo "=== Phase 2 SFT: LIBERO GeoP 7D (kpt_4d_mode=pos_rot) ==="
echo "PROJ_ROOT=${PROJ_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "WARMUP_CKPT=${WARMUP_CKPT}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "WAN_CHECKPOINT_PATH=${WAN_CHECKPOINT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SMOKE=${SMOKE} WAN_SMOKE=${WAN_SMOKE} PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS}"
echo "num_keypoint_joints=8, kpt_4d_mode=pos_rot (56D)"

if [[ "${NODE_RANK}" == "0" ]]; then
  for repo in ${DATASET_REPO_ID}; do
    info_json="${HF_LEROBOT_HOME}/${repo}/meta/info.json"
    if [[ -f "${info_json}" ]]; then
      python -c "
import json
p = '${info_json}'
target = '${repo}'
with open(p) as f:
    d = json.load(f)
if d.get('robot_type') != target:
    d['robot_type'] = target
    with open(p, 'w') as f:
        json.dump(d, f, indent=4)
    print(f'[info-patch] robot_type -> {target} in {p}')
"
    else
      echo "[info-patch] WARNING: ${info_json} not found"
    fi
  done
fi

mkdir -p "$(dirname "${LOG_FILE}")"

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

  --policy.type="${POLICY}"
  --policy.repo_id=lerobot_lab/"${POLICY}"
  --policy.push_to_hub=false
  --policy.pretrained_path="${WARMUP_CKPT}"
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.vlm_model_name_or_path="${VLM_MODEL_PATH}"

  --policy.optimizer_lr=5e-5
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP}"
  --policy.scheduler_decay_steps="${STEPS}"
  --policy.scheduler_decay_lr=5e-6

  --policy.train_expert_only=false
  --policy.knowledge_insulation=false
  --policy.knowledge_insulation_kpt=false
  --policy.freeze_vision_encoder=false
  --policy.enable_vqa_loss=true
  --policy.tokenize_state=true

  --policy.action_loss_only=false
  --policy.video_loss_weight=1
  --policy.video_loss_only=false
  --policy.freeze_wan_dit=true
  --policy.freeze_learnable_tokens=true
  --policy.num_learnable_tokens=50
  --policy.wan_checkpoint_path="${WAN_CHECKPOINT_PATH}"
  --policy.wan_config_path="${WAN_CONFIG_PATH}"
  --policy.vae_path="${WAN_VAE_PATH}"
  --policy.video_micro_batch_size="${VIDEO_MICRO_BATCH_SIZE}"

  --policy.enable_keypoint_predictor=true
  --policy.num_keypoint_joints=8
  --policy.kpt_4d_mode=pos_rot
  --policy.keypoint_history_max_len=200
  --policy.action_loss_weight=10.0
  --policy.kpt_loss_weight=1.0
  --policy.kpt_future_loss_weight=1.5
  --policy.kpt_to_action_detach=false
  --policy.init_kpt_expert_from_action=false
  --policy.freeze_keypoint_modules=false
  --policy.action_expert_lr_scale=1.0
  --policy.kpt_expert_lr_scale=1.0
  --policy.track_encoder_lr_scale=1.0

  --dataset.type="${POLICY}"
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.enable_keypoint_predictor=true
  --dataset.num_keypoint_joints=8
  --dataset.kpt_4d_mode=pos_rot
  --dataset.keypoint_history_max_len=200
  --dataset.action_mode="${ACTION_TYPE}"
  --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
  --dataset.dist_loading=false
  --dataset.tokenize_state=true
  --dataset.use_fast_action_tokens=true
  --dataset.video_backend=torchcodec

  --seed=42
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --log_freq="${LOG_FREQ}"

  --wandb.enable="${WANDB_ENABLE}"
  --wandb.project="${POLICY}"
  --wandb.mode=offline
)

set -o pipefail
"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
train_exit=${PIPESTATUS[0]}

decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" 2>/dev/null) || decode_err=0
zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" 2>/dev/null) || zero_frames=0
echo "post_check: video_decode_error=${decode_err} using_zeros=${zero_frames} exit=${train_exit}"
exit "${train_exit}"
