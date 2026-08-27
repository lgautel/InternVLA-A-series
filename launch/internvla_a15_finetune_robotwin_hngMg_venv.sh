#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# venv-based fine-tune script for InternVLA-A1.5 on RoboTwin hanging_mug.
#
# Based on launch/internvla_a15_finetune_robotwin_stackb3_venv.sh (verified
# on stack_bowls_three) with paths rewritten for this host:
#   - venv: /tmp/itnvla15rbt20
#   - InternVLA-A1.5-base: ${HF_HOME}/ckpts/InternVLA-A1.5-base
#   - dataset repo_id: robotwin/hanging_mug  (must be LeRobot v3.0)
#
# This host has 6x H200. Defaults (all overridable via env):
#   STEPS=12500  BATCH_SIZE=16  SAVE_FREQ=2500  LOG_FREQ=50
#   CUDA_VISIBLE_DEVICES / PROC_PER_NODE  auto-detect nvidia-smi if unset
#   MASTER_PORT=36111  DIST_LOADING=false
#   freeze_learnable_tokens=true, action_loss_only=false
#   per-GPU batch_size=16 (32 OOM with WAN + 3 cameras on H200)
#
# Override example:
#   STEPS=12500 BATCH_SIZE=16 PROC_PER_NODE=6 \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
#     bash launch/internvla_a15_finetune_robotwin_hngMg_venv.sh
#
# See b/d/p/reprd_rbtwn_hngMg.md for full context.
###############################################################################

################################# ENV config ##################################

export HF_HOME="${HF_HOME:-/tmp/itnvla15rbt20/var/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"

VENV_ROOT="${VENV_ROOT:-/tmp/itnvla15rbt20}"
# shellcheck disable=SC1091
source "${VENV_ROOT}/bin/activate"

# This venv's editable internvla-a1-5 may point at another checkout.
# Force this repository's `src/` onto PYTHONPATH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

export WANDB_MODE=offline
# Conversion used HF_HUB_OFFLINE=1; that env persists across shells in this session
# and blocks FAST tokenizer + Qwen processor loads. Training needs the local HF
# cache (physical-intelligence/fast, Qwen/Qwen3.5-2B) with hub lookups enabled.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_DISABLE_TELEMETRY

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-36111}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

# USE_LIBUV=0: fall back to the legacy (non-libuv) TCPStore backend to avoid
# potential hangs in PyTorch 2.10's libuv implementation.
export USE_LIBUV=${USE_LIBUV:-0}

# Detect visible GPUs. This host is 6x H200; do not default to 8.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ -z "${_ngpu}" || "${_ngpu}" -lt 1 ]]; then
        echo "ERROR: nvidia-smi reported no GPUs and CUDA_VISIBLE_DEVICES is unset." >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_ngpu - 1)))"
    PROC_PER_NODE="${PROC_PER_NODE:-${_ngpu}}"
else
    _ngpu="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
    PROC_PER_NODE="${PROC_PER_NODE:-${_ngpu}}"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PROC_PER_NODE=${PROC_PER_NODE}"

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

# Host toolkit is CUDA 13; the torch wheel is cu128. Do not prepend CUDA 13's
# lib64 (libnppicc.so.13). torchcodec 0.10 needs:
#   1) ${VENV_ROOT}/lib first (libstdc++ CXXABI_1.3.15 for the venv ffmpeg)
#   2) pip nvidia/* /lib (libnppicc.so.12 and other CUDA 12 libs)
NV_LIBS="$(find "${VENV_ROOT}/lib/python3.11/site-packages/nvidia" -type d -name lib 2>/dev/null | paste -sd:)"
export LD_LIBRARY_PATH="${VENV_ROOT}/lib${NV_LIBS:+:${NV_LIBS}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

############################## TRAINING config ################################

echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
echo "PYTHONPATH = ${PYTHONPATH}"

cd "${PROJ_ROOT}"

# 1. policy config
POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3.5-2B}"
WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

# 2. dataset config: single RoboTwin task (LeRobot v3.0 converted copy)
DATASET_REPO_ID="${DATASET_REPO_ID:-robotwin/hanging_mug}"
ACTION_TYPE=abs
USE_EXTERNAL_STATS=true
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/agg_1repos_4eb657cb6a/stats.json}"

echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "EXTERNAL_STATS_PATH=${EXTERNAL_STATS_PATH}"

# 3. output configs
BASE_OUTPUT_DIR="outputs/${POLICY}"
PRETRAINED_DETAIL="a15_base"
JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-robotwin-hanging_mug-${ACTION_TYPE}-${PRETRAINED_DETAIL}-finetune}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

# 6 GPU, bs=16 (effective 96). STEPS is overridable; default 12500.
# SAVE_FREQ=2500 -> checkpoints at 2.5k/5k/7.5k/10k/12.5k.
STEPS="${STEPS:-12500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-50}"

# dist_loading=false is safer for this 50-episode dataset on few ranks.
DIST_LOADING="${DIST_LOADING:-false}"

echo "STEPS=${STEPS} BATCH_SIZE=${BATCH_SIZE} PROC_PER_NODE=${PROC_PER_NODE} DIST_LOADING=${DIST_LOADING}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

############################## Path preflight #################################

_fail=0
_need_file() {
    local p="$1" label="$2"
    if [[ ! -e "${p}" ]]; then
        echo "ERROR: missing ${label}: ${p}" >&2
        _fail=1
    fi
}

_need_file "${PRETRAINED_PATH}/model.safetensors" "InternVLA-A1.5-base weights"
_need_file "${WAN_VAE_PATH}" "WAN VAE"
_need_file "${WAN_CHECKPOINT_PATH}/config.json" "WAN config"
_need_file "${EXTERNAL_STATS_PATH}" "external abs stats"

DATASET_INFO="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json"
if [[ ! -f "${DATASET_INFO}" ]]; then
    echo "ERROR: missing dataset info.json: ${DATASET_INFO}" >&2
    echo "  Convert hanging_mug to LeRobot v3.0 and symlink it to ${HF_LEROBOT_HOME}/${DATASET_REPO_ID}" >&2
    _fail=1
else
    _ver="$("${VENV_ROOT}/bin/python" -c "import json; print(json.load(open('${DATASET_INFO}')).get('codebase_version',''))")"
    if [[ "${_ver}" != "v3.0" ]]; then
        echo "ERROR: ${DATASET_INFO} codebase_version=${_ver} (need v3.0)" >&2
        _fail=1
    else
        echo "dataset codebase_version=${_ver} at ${DATASET_INFO}"
    fi
fi

if [[ "${_fail}" -ne 0 ]]; then
    echo "Preflight failed. Fix the paths above before launching training." >&2
    exit 1
fi

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

# The venv has the accelerate package (1.14.0) but no console-script shims in
# ${VENV_ROOT}/bin (no `accelerate` executable). Invoke the module entry point.
"${VENV_ROOT}/bin/python" -m accelerate.commands.launch "${ARGS[@]}"
