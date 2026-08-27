#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# GeoP Phase 2 fine-tune — kptsim voxel GT, 8×H200
# venv (self-contained): /tmp/itnvla15rbt20/
# code (editable):       /tmp/SRC/InternVLA-A-series/
# See: b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.md
#
# Starting checkpoint: Warmup ckpt@400 (NOT InternVLA-A1.5-base)
# Training: full finetune (VLM + experts + kpt), WAN DiT frozen only
# Hyperparams: aligned with internvla_a15_finetune_robotwin.sh + stackb3_venv
#
# Usage:
#   bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
#   WAN_SMOKE=1 bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
#   SMOKE=1 bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
# hanging_mug (GCS RunPkg + GitHub, see b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg.md):
#   DATA_REPO_ID=hanging_mug_kptsim_lrbv30 PROJ_ROOT=/tmp/SRC/itvlaGp \
#     WARMUP_CKPT=/tmp/RunPkg/Ckp/warmup_hanging_mug_kptsim_400step/checkpoints/000400/pretrained_model \
#     bash launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
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
export MASTER_PORT="${MASTER_PORT:-36202}"

WARMUP_JOB="${WARMUP_JOB:-2026_08_11_03_04_19-internvla_a1_5-geop-phase1-kpt-warmup-kptsim-voxel-8g}"
WARMUP_CKPT="${WARMUP_CKPT:-${PROJ_ROOT}/outputs/internvla_a1_5/${WARMUP_JOB}/checkpoints/000400/pretrained_model}"

WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3.5-2B}"

POLICY="internvla_a1_5"
DATA_REPO_ID="${DATA_REPO_ID:-stack_bowls_three_kptsim_lrbv30}"
NORM_STATS="${NORM_STATS:-${HF_LEROBOT_HOME}/${DATA_REPO_ID}/norm_stat.json}"
DIST_LOADING="${DIST_LOADING:-false}"

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
  JOB_SUFFIX="geop-phase2-wan-smoke-kptsim-voxel"
elif [[ "${SMOKE}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PROC_PER_NODE="${PROC_PER_NODE:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  STEPS="${STEPS:-100}"
  NUM_WORKERS="${NUM_WORKERS:-4}"
  SAVE_FREQ="${SAVE_FREQ:-100}"
  LOG_FREQ="${LOG_FREQ:-10}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-50}"
  WANDB_ENABLE="${WANDB_ENABLE:-false}"
  JOB_SUFFIX="geop-phase2-smoke100-kptsim-voxel"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  PROC_PER_NODE="${PROC_PER_NODE:-8}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  STEPS="${STEPS:-10000}"
  NUM_WORKERS="${NUM_WORKERS:-12}"
  SAVE_FREQ="${SAVE_FREQ:-2500}"
  LOG_FREQ="${LOG_FREQ:-50}"
  SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"
  WANDB_ENABLE="${WANDB_ENABLE:-true}"
  JOB_SUFFIX="geop-phase2-finetune-kptsim-voxel-8g-10k"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

cd "${PROJ_ROOT}"

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${JOB_SUFFIX}}"
# 正式 10k：若编排脚本已设 LOG_FILE，checkpoint 落到 LOG_DIR（dirname(LOG_FILE)）下。
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "${WAN_SMOKE}" != "1" && "${SMOKE}" != "1" && -n "${LOG_FILE:-}" ]]; then
    OUTPUT_DIR="$(dirname "${LOG_FILE}")/${JOB_NAME}"
  else
    OUTPUT_DIR="${PROJ_ROOT}/outputs/${POLICY}/${JOB_NAME}"
  fi
fi
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

# 头信息必须进 LOG_FILE：后面的 accelerate|tee 会覆盖该文件，因此先写入再 tee -a。
# 编排脚本 check_launch_log() 会 grep DATA_REPO_ID= 与 post_check:。
{
  echo "VENV_ROOT=${VENV_ROOT}"
  echo "PROJ_ROOT=${PROJ_ROOT}"
  echo "HF_HOME=${HF_HOME}"
  echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
  echo "DATA_REPO_ID=${DATA_REPO_ID}"
  echo "WARMUP_CKPT=${WARMUP_CKPT}"
  echo "WAN_DIR=${WAN_DIR}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "WAN_SMOKE=${WAN_SMOKE} SMOKE=${SMOKE} PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS}"
} | tee "${LOG_FILE}"

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
  --policy.pretrained_path="${WARMUP_CKPT}"
  # A800 80G 上 Phase 2 同时训练 VLM 与 experts；启用 activation
  # checkpointing 保持每卡 batch=16 和有效 batch=128，避免首步 OOM。
  --policy.gradient_checkpointing=true
  --policy.dtype=bfloat16
  --policy.optimizer_lr=5e-5
  --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP}"
  --policy.scheduler_decay_steps="${STEPS}"
  --policy.scheduler_decay_lr=5e-6
  --policy.freeze_vision_encoder=false
  --policy.train_expert_only=false
  --policy.vlm_model_name_or_path="${VLM_MODEL_PATH}"
  --policy.enable_vqa_loss=true
  --policy.tokenize_state=true
  --policy.knowledge_insulation=false
  --policy.video_loss_only=false
  --policy.video_loss_weight=1
  --policy.action_loss_only=false
  --policy.freeze_wan_dit=true
  --policy.freeze_learnable_tokens=true
  --policy.num_learnable_tokens=50
  --policy.wan_checkpoint_path="${WAN_DIR}"
  --policy.wan_config_path="${WAN_DIR}"
  --policy.vae_path="${WAN_DIR}/Wan2.2_VAE.pth"
  --policy.video_micro_batch_size="${VIDEO_MICRO_BATCH_SIZE:-1}"
  --policy.enable_keypoint_predictor=true
  --policy.num_keypoint_joints=14
  --policy.action_loss_weight=10.0
  --policy.kpt_loss_weight=1.0
  --policy.kpt_future_loss_weight=1.5
  --policy.kpt_to_action_detach=false
  --policy.freeze_keypoint_modules=false
  --policy.action_expert_lr_scale=1.0
  --policy.kpt_expert_lr_scale=1.0
  --policy.track_encoder_lr_scale=1.0
  --policy.init_kpt_expert_from_action=false
  --dataset.type="${POLICY}"
  --dataset.repo_id="${DATA_REPO_ID}"
  --dataset.enable_keypoint_predictor=true
  --dataset.num_keypoint_joints=14
  --dataset.action_mode=abs
  --dataset.tokenize_state=true
  --dataset.use_fast_action_tokens=true
  --dataset.use_external_stats=true
  --dataset.external_stats_path="${NORM_STATS}"
  --dataset.dist_loading="${DIST_LOADING}"
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
set +e
"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
train_exit=${PIPESTATUS[0]}
set -e

decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" || true)
zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" || true)
# set -e + pipefail 时 accelerate 非 0 会在此处之前就退出；上面 set +e 保证 post_check 一定落盘。
echo "post_check: video_decode_error=${decode_err} using_zeros=${zero_frames} exit=${train_exit}" | tee -a "${LOG_FILE}"
if [[ "${decode_err}" -ne 0 || "${zero_frames}" -ne 0 ]]; then
  echo "WARNING: video decode failures — see wrmup8G.md Appendix A" >&2
fi
exit "${train_exit}"
