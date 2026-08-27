#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Phase 1 Kpt Expert Warmup — kptsim voxel GT, 8×H200
# venv (self-contained): /tmp/itnvla15rbt20/
# code (editable):       /tmp/SRC/InternVLA-A-series/
# See: b/d/itrnVLA15_GeoP_3dtrj_3cn4_wrmup8G.md
#
# Usage:
#   bash launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh
#   SMOKE=1 bash launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh
###############################################################################

VENV_ROOT="${VENV_ROOT:-/tmp/itnvla15rbt20}"
PROJ_ROOT="${PROJ_ROOT:-/tmp/SRC/InternVLA-A-series}"
PYTHON="${PYTHON:-${VENV_ROOT}/bin/python}"

export HF_HOME="${HF_HOME:-${VENV_ROOT}/var/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${VENV_ROOT}/var/datasets}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_LIBUV="${USE_LIBUV:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/torch/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36201}"

SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PROC_PER_NODE="${PROC_PER_NODE:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  STEPS="${STEPS:-100}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
  SAVE_FREQ="${SAVE_FREQ:-100}"
  LOG_FREQ="${LOG_FREQ:-10}"
  WANDB_ENABLE="${WANDB_ENABLE:-false}"
  JOB_SUFFIX="smoke100-kptsim-voxel"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  PROC_PER_NODE="${PROC_PER_NODE:-8}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  STEPS="${STEPS:-400}"
  NUM_WORKERS="${NUM_WORKERS:-12}"
  SAVE_FREQ="${SAVE_FREQ:-100}"
  LOG_FREQ="${LOG_FREQ:-10}"
  WANDB_ENABLE="${WANDB_ENABLE:-true}"
  JOB_SUFFIX="geop-phase1-kpt-warmup-kptsim-voxel-8g"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

POLICY="internvla_a1_5"
DATA_REPO_ID="${DATA_REPO_ID:-stack_bowls_three_kptsim_lrbv30}"
NORM_STATS="${NORM_STATS:-${HF_LEROBOT_HOME}/${DATA_REPO_ID}/norm_stat.json}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
GEOPREDICT_CKPT="${GEOPREDICT_CKPT:-${HF_HOME}/ckpts/GeoPredict_robocasa.pth}"

cd "${PROJ_ROOT}"

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${JOB_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"

echo "VENV_ROOT=${VENV_ROOT}"
echo "PROJ_ROOT=${PROJ_ROOT}"
echo "HF_HOME=${HF_HOME}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SMOKE=${SMOKE} PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS}"

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
  --job_name="${JOB_NAME}"
  --num_workers="${NUM_WORKERS}"
  --policy.type="${POLICY}"
  --policy.repo_id=lerobot_lab/"${POLICY}"
  --policy.push_to_hub=false
  --policy.pretrained_path="${PRETRAINED_PATH}"
  --policy.dtype=bfloat16
  --policy.optimizer_lr=5e-5
  --policy.scheduler_warmup_steps=50
  --policy.scheduler_decay_steps="${STEPS}"
  --policy.scheduler_decay_lr=5e-6
  --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B
  --policy.enable_vqa_loss=false
  --policy.tokenize_state=true
  --policy.video_loss_weight=1
  --policy.freeze_learnable_tokens=true
  --policy.num_learnable_tokens=50
  --policy.train_expert_only=true
  --policy.enable_keypoint_predictor=true
  --policy.num_keypoint_joints=14
  --policy.action_loss_weight=2.0
  --policy.kpt_loss_weight=10.0
  --policy.kpt_future_loss_weight=2.0
  --policy.knowledge_insulation=true
  --policy.knowledge_insulation_kpt=true
  --policy.kpt_to_action_detach=false
  --policy.freeze_keypoint_modules=false
  --policy.action_expert_lr_scale=0.04
  --policy.kpt_expert_lr_scale=1.0
  --policy.track_encoder_lr_scale=1.0
  --policy.init_kpt_expert_from_action=true
  --policy.action_loss_only=true
  --policy.geopredict_checkpoint_path="${GEOPREDICT_CKPT}"
  --dataset.type="${POLICY}"
  --dataset.repo_id="${DATA_REPO_ID}"
  --dataset.enable_keypoint_predictor=true
  --dataset.num_keypoint_joints=14
  --dataset.action_mode=abs
  --dataset.tokenize_state=true
  --dataset.use_fast_action_tokens=true
  --dataset.use_external_stats=true
  --dataset.external_stats_path="${NORM_STATS}"
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

mkdir -p "$(dirname "${LOG_FILE}")"
set -o pipefail
"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
train_exit=${PIPESTATUS[0]}

decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" || true)
zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" || true)
echo "post_check: video_decode_error=${decode_err} using_zeros=${zero_frames} exit=${train_exit}"
if [[ "${decode_err}" -ne 0 || "${zero_frames}" -ne 0 ]]; then
  echo "WARNING: video decode failures — see wrmup8G.md Appendix A" >&2
fi
exit "${train_exit}"
