#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# venv fine-tune: InternVLA-A1.5 on one RoboTwin 2.0 task.
#
# Must `source` the venv (activate sets HF_HOME + LD_LIBRARY_PATH). Do not only
# invoke ${VENV_ROOT}/bin/python.
#
# The default task remains scan_object for backward-compatible examples, but
# TASK_NAME can be set to any prepared RoboTwin task. DATASET_REPO_ID can also
# be supplied directly when the repo id does not follow robotwin/<task>.
#
# Steps / save_freq are computed from:
#   STEPS = ceil(num_frames * NUM_EPOCHS / TOTAL_BATCH_SIZE)
#   SAVE_FREQ = STEPS // 4          # 25% / 50% / 75%; last step is always saved
#
# Defaults (all overridable):
#   VENV_ROOT=/B/VENV/itnvla15rbt20
#   ROBOTWIN_CLEAN_ROOT=/B/Dta/RoboTwin-Clean
#   CKPT_BASE=/B/Ckp
#   TASK_NAME=scan_object           # any prepared RoboTwin task
#   DATASET_REPO_ID=robotwin/<task> # optional; defaults from TASK_NAME
#   NUM_EPOCHS=76
#   TOTAL_BATCH_SIZE=128            # global; per-GPU = TOTAL / PROC_PER_NODE
#   ITNVLA_STAMP=$(date +%y%m%d%H%M)
#   RUN_STAMP=$(date +%y%m%d%H%M)
#
# Output layout:
#   ${CKPT_BASE}/itnVla_${ITNVLA_STAMP}/rbt2/${TASK_NAME}/
#     train_${RUN_STAMP}.log
#     ckpt_${RUN_STAMP}/            # --output_dir (wandb + checkpoints live here)
#       checkpoints/<step>/
#       wandb/
#
# Smoke (4 steps, still loads WAN):
#   SMOKE=1 bash launch/internvla_a15_finetune_robotwin_venv.sh
#
# See b/d/p/reprd_rbtwn_scnObj.md for a scan_object example and the generic
# TASK_NAME/DATASET_REPO_ID usage.
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REQUESTED_TASK_NAME="${TASK_NAME:-}"
_REQUESTED_DATASET_REPO_ID="${DATASET_REPO_ID:-}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/internvla_a15_robotwin_common.sh"

activate_itnvla_venv
cd "${PROJ_ROOT}"

# Allow callers to select a task with either TASK_NAME or DATASET_REPO_ID.
TASK_NAME="${_REQUESTED_TASK_NAME:-${_REQUESTED_DATASET_REPO_ID##*/}}"
TASK_NAME="${TASK_NAME:-scan_object}"
DATASET_REPO_ID="${_REQUESTED_DATASET_REPO_ID:-robotwin/${TASK_NAME}}"
if [[ -z "${TASK_NAME}" || "${TASK_NAME}" == */* || "${TASK_NAME}" == "." || "${TASK_NAME}" == ".." ]]; then
    echo "ERROR: invalid TASK_NAME=${TASK_NAME}" >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36222}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

detect_gpus
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

############################## Training knobs #################################

POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3.5-2B}"
WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B/Wan2.2_VAE.pth}"

USE_EXTERNAL_STATS=true

NUM_EPOCHS="${NUM_EPOCHS:-76}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-128}"
DIST_LOADING="${DIST_LOADING:-false}"
LOG_FREQ="${LOG_FREQ:-50}"

ITNVLA_STAMP="${ITNVLA_STAMP:-$(compact_stamp)}"
RUN_STAMP="${RUN_STAMP:-$(compact_stamp)}"

OUTPUT_ROOT="${CKPT_BASE}/itnVla_${ITNVLA_STAMP}/rbt2/${TASK_NAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/ckpt_${RUN_STAMP}"
LOG_FILE="${OUTPUT_ROOT}/train_${RUN_STAMP}.log"
JOB_NAME="${JOB_NAME:-${ITNVLA_STAMP}-${POLICY}-robotwin-${TASK_NAME}-${ACTION_TYPE}-finetune-${RUN_STAMP}}"

DATASET_INFO="${HF_LEROBOT_HOME}/${DATASET_REPO_ID}/meta/info.json"

############################## Compute STEPS / SAVE_FREQ ######################

if [[ ! -f "${DATASET_INFO}" ]]; then
    echo "ERROR: missing dataset info.json: ${DATASET_INFO}" >&2
    echo "  Run: TASK_NAME=${TASK_NAME} bash launch/internvla_a15_prepare_robotwin.sh" >&2
    exit 1
fi

ROBOT_TYPE="${ROBOT_TYPE:-}"
if [[ -z "${ROBOT_TYPE}" ]]; then
    ROBOT_TYPE="$("${VENV_ROOT}/bin/python" -c \
        "import json; print(json.load(open('${DATASET_INFO}')).get('robot_type', ''))")"
fi
if [[ -z "${ROBOT_TYPE}" ]]; then
    echo "ERROR: dataset has no robot_type: ${DATASET_INFO}" >&2
    exit 1
fi
GROUP_NAME="$(stats_group_name "${DATASET_REPO_ID}")"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${HF_HOME}/lerobot/stats/${ROBOT_TYPE}/${ACTION_TYPE}/${GROUP_NAME}/stats.json}"

read -r NUM_FRAMES COMPUTED_STEPS < <(
    NUM_EPOCHS="${NUM_EPOCHS}" TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE}" DATASET_INFO="${DATASET_INFO}" \
    "${VENV_ROOT}/bin/python" - <<'PY'
import json, math, os
info = json.load(open(os.environ["DATASET_INFO"]))
n = int(info["total_frames"])
epochs = int(os.environ["NUM_EPOCHS"])
bs = int(os.environ["TOTAL_BATCH_SIZE"])
if bs < 1:
    raise SystemExit("TOTAL_BATCH_SIZE must be >= 1")
steps = int(math.ceil(n * epochs / bs))
print(n, steps)
PY
)

if [[ $((TOTAL_BATCH_SIZE % PROC_PER_NODE)) -ne 0 ]]; then
    echo "ERROR: TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} is not divisible by PROC_PER_NODE=${PROC_PER_NODE}." >&2
    echo "  Pick a TOTAL_BATCH_SIZE multiple of the GPU count, e.g. $((16 * PROC_PER_NODE)) (16/GPU)." >&2
    echo "  WAN + 3 cameras can OOM at 32/GPU on H200; 16/GPU is the known-good cap." >&2
    exit 1
fi
BATCH_SIZE=$((TOTAL_BATCH_SIZE / PROC_PER_NODE))
if [[ "${BATCH_SIZE}" -gt 16 ]]; then
    echo "WARNING: per-GPU BATCH_SIZE=${BATCH_SIZE} > 16. WAN + 3 cameras can OOM at 32/GPU on H200." >&2
    echo "  Known-good: 16/GPU (TOTAL_BATCH_SIZE=$((16 * PROC_PER_NODE)) on ${PROC_PER_NODE} GPUs)." >&2
    echo "  Continue anyway, or re-run with TOTAL_BATCH_SIZE=$((16 * PROC_PER_NODE))." >&2
fi

if [[ "${SMOKE:-0}" == "1" ]]; then
    STEPS="${SMOKE_STEPS:-4}"
    SAVE_FREQ="${SMOKE_SAVE_FREQ:-2}"
    LOG_FREQ=1
    JOB_NAME="smoke-${JOB_NAME}"
    echo "SMOKE=1: overriding STEPS=${STEPS} SAVE_FREQ=${SAVE_FREQ}"
else
    STEPS="${STEPS:-${COMPUTED_STEPS}}"
    SAVE_FREQ="${SAVE_FREQ:-$((STEPS / 4))}"
    if [[ "${SAVE_FREQ}" -lt 1 ]]; then
        SAVE_FREQ=1
    fi
fi

WARMUP_STEPS="${WARMUP_STEPS:-$((STEPS / 10))}"
if [[ "${WARMUP_STEPS}" -lt 1 ]]; then
    WARMUP_STEPS=1
fi
if [[ "${WARMUP_STEPS}" -gt "${STEPS}" ]]; then
    WARMUP_STEPS="${STEPS}"
fi

echo "NUM_FRAMES=${NUM_FRAMES} NUM_EPOCHS=${NUM_EPOCHS} TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
echo "BATCH_SIZE(per GPU)=${BATCH_SIZE} PROC_PER_NODE=${PROC_PER_NODE} DIST_LOADING=${DIST_LOADING}"
echo "STEPS=${STEPS} SAVE_FREQ=${SAVE_FREQ} WARMUP_STEPS=${WARMUP_STEPS} LOG_FREQ=${LOG_FREQ}"
echo "ckpt steps ~= $((SAVE_FREQ)) / $((SAVE_FREQ * 2)) / $((SAVE_FREQ * 3)) / ${STEPS} (last always saved)"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "OUTPUT_DIR =${OUTPUT_DIR}"
echo "LOG_FILE   =${LOG_FILE}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "ROBOT_TYPE=${ROBOT_TYPE}"
echo "EXTERNAL_STATS_PATH=${EXTERNAL_STATS_PATH}"

############################## Preflight ######################################

_fail=0
need_file "${PRETRAINED_PATH}/model.safetensors" "InternVLA-A1.5-base weights" || _fail=1
need_file "${WAN_VAE_PATH}" "WAN VAE" || _fail=1
need_file "${WAN_CHECKPOINT_PATH}/config.json" "WAN config" || _fail=1
need_file "${EXTERNAL_STATS_PATH}" "external ${ACTION_TYPE} stats" || _fail=1
need_file "${DATASET_INFO}" "dataset info.json" || _fail=1

if [[ -f "${DATASET_INFO}" ]]; then
    _ver="$("${VENV_ROOT}/bin/python" -c "import json; print(json.load(open('${DATASET_INFO}')).get('codebase_version',''))")"
    if [[ "${_ver}" != "v3.0" ]]; then
        echo "ERROR: ${DATASET_INFO} codebase_version=${_ver} (need v3.0)" >&2
        _fail=1
    else
        echo "dataset codebase_version=${_ver} at ${DATASET_INFO}"
    fi
fi

if [[ "${_fail}" -ne 0 ]]; then
    echo "Preflight failed. Run data prep first:" >&2
    echo "  TASK_NAME=${TASK_NAME} bash launch/internvla_a15_prepare_robotwin.sh" >&2
    exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "ERROR: output_dir already exists (resume=false): ${OUTPUT_DIR}" >&2
    echo "  Reruns must use a new RUN_STAMP. Example: RUN_STAMP=\$(date +%y%m%d%H%M) $0" >&2
    exit 1
fi

# Do NOT mkdir OUTPUT_DIR: lerobot_train.py raises FileExistsError if it already exists.
mkdir -p "${OUTPUT_ROOT}"
echo "${JOB_NAME}" > "${OUTPUT_ROOT}/job_${RUN_STAMP}.txt"
{
    echo "ITNVLA_STAMP=${ITNVLA_STAMP}"
    echo "RUN_STAMP=${RUN_STAMP}"
    echo "JOB_NAME=${JOB_NAME}"
    echo "STEPS=${STEPS}"
    echo "SAVE_FREQ=${SAVE_FREQ}"
    echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "NUM_EPOCHS=${NUM_EPOCHS}"
    echo "NUM_FRAMES=${NUM_FRAMES}"
    echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
    echo "ROBOT_TYPE=${ROBOT_TYPE}"
    echo "EXTERNAL_STATS_PATH=${EXTERNAL_STATS_PATH}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
} > "${OUTPUT_ROOT}/run_${RUN_STAMP}.env"

############################## Launch #########################################

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS:-8}"
    --job_name="${JOB_NAME}"

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.pretrained_path=${PRETRAINED_PATH}
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps=${WARMUP_STEPS}
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

    --dataset.type="$POLICY"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.action_mode="$ACTION_TYPE"
    --dataset.use_external_stats="$USE_EXTERNAL_STATS"
    --dataset.external_stats_path=${EXTERNAL_STATS_PATH}
    --dataset.dist_loading=${DIST_LOADING}
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

echo "===== launching training; log -> ${LOG_FILE} ====="
# venv may ship accelerate as a package without a console-script shim.
# Do not use nohup & disown (HUP kills DDP children). Use tmux/screen, or
# this process in the foreground / Cursor background.
set -o pipefail
"${VENV_ROOT}/bin/python" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
