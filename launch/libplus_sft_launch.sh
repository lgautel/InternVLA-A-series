#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# OpVLA Libero merged kpt SFT — 8-joint 7D keypoints (kpt_4d_mode=pos_rot)
#
# Direct SFT from InternVLA-A1.5-base on /B/Dta/opvla_libero_merged_kpt/
# See: b/d/libplus/sft.md
#
# Usage (production, 8-GPU):
#   bash launch/libplus_sft_launch.sh
#
# Usage (WAN smoke, 1GPU 2steps):
#   WAN_SMOKE=1 bash launch/libplus_sft_launch.sh
#
# Usage (smoke 100 steps):
#   SMOKE=1 bash launch/libplus_sft_launch.sh
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EXPR_NAME="${EXPR_NAME:-4dwvlaOpvlaLibplusKpt0911}"
VENV_ROOT="${VENV_ROOT:-/B/VENV/itnvla15rbt20}"
PYTHON="${PYTHON:-${VENV_ROOT}/bin/python}"

export HF_HOME="${HF_HOME:-/B/VENV/hf_home}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_LIBUV="${USE_LIBUV:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-/dev/null}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/itvla-triton-cache}"

export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/torch/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

POLICY="internvla_a1_5"
DATA_SRC="${DATA_SRC:-/B/Dta/opvla_libero_merged_kpt}"
DATA_REPO_ID="${DATA_REPO_ID:-opvla_libero_merged_kpt}"
EXTERNAL_STATS_PATH="${EXTERNAL_STATS_PATH:-${DATA_SRC}/meta/stats.json}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
VLM_PATH="${VLM_PATH:-Qwen/Qwen3.5-2B}"
WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
VIDEO_MICRO_BATCH_SIZE="${VIDEO_MICRO_BATCH_SIZE:-2}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36704}"

CKPT_ROOT="${CKPT_ROOT:-${HOME}/b/Ckp/${EXPR_NAME}}"
LOG_ROOT="${LOG_ROOT:-/B/Log/${EXPR_NAME}}"

MONITOR_INTERVAL="${MONITOR_INTERVAL:-900}"
STALE_THRESHOLD="${STALE_THRESHOLD:-900}"
BIGMATRIX_SCRIPT="${BIGMATRIX_SCRIPT:-${PROJ_ROOT}/b/d/GpRbt/bigmatrix_multiply_optimization.py}"
BIGMATRIX_MAX_RETRIES="${BIGMATRIX_MAX_RETRIES:-5}"
MAX_RESUME_ATTEMPTS="${MAX_RESUME_ATTEMPTS:-0}"
ENABLE_AUTO_RESTART="${ENABLE_AUTO_RESTART:-false}"

SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-30000}"

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
    JOB_SUFFIX="libplus-wan-smoke"
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
    JOB_SUFFIX="libplus-sft-smoke"
else
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    PROC_PER_NODE="${PROC_PER_NODE:-8}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    STEPS="${STEPS:-53450}"
    NUM_WORKERS="${NUM_WORKERS:-12}"
    SAVE_FREQ="${SAVE_FREQ:-5345}"
    LOG_FREQ="${LOG_FREQ:-1000}"
    SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"
    WANDB_ENABLE="${WANDB_ENABLE:-true}"
    JOB_SUFFIX="libplus-sft"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

cd "${PROJ_ROOT}"

_wan_preflight() {
    local missing=0
    if [[ ! -f "${WAN_DIR}/config.json" ]]; then
        echo "ERROR: WAN config.json missing: ${WAN_DIR}/config.json" >&2
        missing=1
    fi
    if [[ ! -f "${WAN_DIR}/Wan2.2_VAE.pth" ]]; then
        echo "ERROR: WAN VAE missing: ${WAN_DIR}/Wan2.2_VAE.pth" >&2
        missing=1
    fi
    [[ "${missing}" -eq 0 ]] || exit 1
}

_wan_preflight

if [[ ! -L "${HF_LEROBOT_HOME}/${DATA_REPO_ID}" ]]; then
    ln -sfn "${DATA_SRC}" "${HF_LEROBOT_HOME}/${DATA_REPO_ID}"
    echo "[data] Created symlink: ${HF_LEROBOT_HOME}/${DATA_REPO_ID} -> ${DATA_SRC}"
fi

JOB_STAMP="${JOB_STAMP:-$(date +'%Y_%m_%d_%H_%M_%S')}"
JOB_NAME="${JOB_NAME:-${JOB_STAMP}-${POLICY}-${JOB_SUFFIX}}"

if [[ "${WAN_SMOKE}" == "1" || "${SMOKE}" == "1" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR:-/tmp/sft_smoke_libplus_${JOB_STAMP}}"
    LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
else
    OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/${JOB_NAME}}"
    LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${JOB_STAMP}/train.log}"
fi

echo "=== OpVLA Libplus SFT: 8-joint 7D keypoints (kpt_4d_mode=pos_rot) ==="
echo "EXPR_NAME=${EXPR_NAME}"
echo "PRETRAINED_PATH=${PRETRAINED_PATH}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "SMOKE=${SMOKE} WAN_SMOKE=${WAN_SMOKE} PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS}"

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
    --policy.pretrained_path="${PRETRAINED_PATH}"
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16
    --policy.vlm_model_name_or_path="${VLM_PATH}"

    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP}"
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"
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
    --policy.freeze_learnable_tokens=false
    --policy.num_learnable_tokens=50
    --policy.wan_checkpoint_path="${WAN_DIR}"
    --policy.wan_config_path="${WAN_DIR}"
    --policy.vae_path="${WAN_DIR}/Wan2.2_VAE.pth"
    --policy.video_micro_batch_size="${VIDEO_MICRO_BATCH_SIZE}"

    --policy.enable_keypoint_predictor=true
    --policy.num_keypoint_joints=8
    --policy.kpt_4d_mode=pos_rot
    --policy.kpt_rot_loss_weight=1.0
    --policy.keypoint_history_max_len=200

    --policy.action_loss_weight=10.0
    --policy.kpt_loss_weight=1.0
    --policy.kpt_future_loss_weight=2.0
    --policy.kpt_to_action_detach=false
    --policy.init_kpt_expert_from_action=true

    --policy.freeze_keypoint_modules=false
    --policy.action_expert_lr_scale=1.0
    --policy.kpt_expert_lr_scale=1.0
    --policy.track_encoder_lr_scale=1.0

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

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${POLICY}"
    --wandb.mode=offline
)

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

_find_latest_checkpoint() {
    local ckpt_dir="${OUTPUT_DIR}/checkpoints"
    if [[ ! -d "${ckpt_dir}" ]]; then
        echo ""
        return
    fi
    local latest
    latest=$(ls -1d "${ckpt_dir}"/*/pretrained_model 2>/dev/null | sort -V | tail -1)
    echo "${latest:-}"
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
        retry=$((retry + 1))
        sleep 5
    done
    return 1
}

_archive_and_cleanup() {
    local suffix="$1"
    _kill_gpu_processes
    _start_bigmatrix || true
    local ts archive_name archive_dest archive_path counter
    ts=$(_monitor_ts)
    archive_name="${EXPR_NAME}_LOG_${ts}${suffix}"
    archive_dest="${HOME}/b/Ckp"
    mkdir -p "${archive_dest}"
    archive_path="${archive_dest}/${archive_name}.tar"
    counter=1
    while [[ -f "${archive_path}" ]]; do
        archive_path="${archive_dest}/${archive_name}_${counter}.tar"
        counter=$((counter + 1))
    done
    _monitor_log "Archiving ${LOG_ROOT} → ${archive_path}"
    tar -cf "${archive_path}" \
        -C "$(dirname "${LOG_ROOT}")" "$(basename "${LOG_ROOT}")" 2>/dev/null || true
}

_auto_recover() {
    local attempt=0
    while true; do
        attempt=$((attempt + 1))
        local latest_ckpt
        latest_ckpt=$(_find_latest_checkpoint)

        _kill_gpu_processes
        pkill -f "bigmatrix_multiply" 2>/dev/null || true
        sleep 5

        local recover_stamp
        recover_stamp=$(date +'%Y_%m_%d_%H_%M_%S')
        LOG_FILE="${LOG_ROOT}/${recover_stamp}/train.log"
        mkdir -p "$(dirname "${LOG_FILE}")"

        if [[ -n "${latest_ckpt}" ]]; then
            _monitor_log "RECOVER: Resume from ${latest_ckpt} (attempt ${attempt})"
            "${PYTHON}" -m accelerate.commands.launch "${LAUNCH_ARGS[@]}" \
                src/lerobot/scripts/lerobot_train.py \
                --config_path="${latest_ckpt}/train_config.json" \
                --resume=true \
                --output_dir="${OUTPUT_DIR}" \
                --num_workers="${NUM_WORKERS}" \
                --job_name="${JOB_NAME}" \
                >> "${LOG_FILE}" 2>&1 &
        else
            _monitor_log "RECOVER: No checkpoint — fresh start (attempt ${attempt})"
            "${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" \
                >> "${LOG_FILE}" 2>&1 &
        fi
        TRAIN_PID=$!

        local poll_sec=60 elapsed=0
        while true; do
            sleep ${poll_sec}
            elapsed=$((elapsed + poll_sec))

            if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
                wait "${TRAIN_PID}" 2>/dev/null
                local recover_exit=$?
                _monitor_log "RECOVER: exited code=${recover_exit}"
                sleep 10
                if [[ "${recover_exit}" -eq 0 ]] && _are_outputs_complete; then
                    _monitor_log "RECOVER: SUCCESS"
                    return 0
                fi
                break
            fi

            if [[ ${elapsed} -ge ${MONITOR_INTERVAL} ]]; then
                elapsed=0
                if _is_log_stale; then
                    _monitor_log "RECOVER: log stale — retry"
                    break
                fi
                local cur_step
                cur_step=$(grep -oE 'step:[0-9]+' "${LOG_FILE}" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
                _monitor_log "RECOVER: step=${cur_step:-?}/${STEPS}"
            fi
        done
        _monitor_log "RECOVER: attempt ${attempt} failed — retrying"
    done
}

_next_suffix() {
    local current="$1"
    if [[ "${current}" =~ ([A-Z])$ ]]; then
        local letter="${BASH_REMATCH[1]}"
        local base="${current%${letter}}"
        printf "%s\\$(printf '%%03o' $(($(printf '%d' "'${letter}") + 1)))" "${base}"
    else
        echo "${current}A"
    fi
}

_auto_restart_next() {
    if [[ "${ENABLE_AUTO_RESTART}" != "true" ]]; then
        _monitor_log "Auto-restart disabled"
        return 0
    fi
    local next_name
    next_name=$(_next_suffix "${EXPR_NAME}")
    pkill -f "bigmatrix_multiply" 2>/dev/null || true
    sleep 10
    export EXPR_NAME="${next_name}"
    export SMOKE=0 WAN_SMOKE=0 JOB_STAMP="" OUTPUT_DIR="" LOG_FILE=""
    nohup bash "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")" \
        > "/tmp/auto_restart_${next_name}.log" 2>&1 &
}

if [[ "${WAN_SMOKE}" == "1" || "${SMOKE}" == "1" ]]; then
    set -o pipefail
    "${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    train_exit=${PIPESTATUS[0]}
    decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" 2>/dev/null) || decode_err=0
    zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" 2>/dev/null) || zero_frames=0
    echo "post_check: video_decode_error=${decode_err} using_zeros=${zero_frames} exit=${train_exit}"
    exit "${train_exit}"
fi

set +e
_monitor_log "=== OpVLA Libplus SFT monitoring enabled ==="
_monitor_log "OUTPUT_DIR=${OUTPUT_DIR} LOG=${LOG_FILE}"

"${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
_monitor_log "Training PID=${TRAIN_PID}"

_poll_sec=60
_elapsed=0

while true; do
    sleep ${_poll_sec}
    _elapsed=$((_elapsed + _poll_sec))

    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
        wait "${TRAIN_PID}" 2>/dev/null
        train_exit=$?
        _monitor_log "Training exited code=${train_exit}"
        sleep 10
        if [[ "${train_exit}" -eq 0 ]] && _are_outputs_complete; then
            _monitor_log "SUCCESS"
            _archive_and_cleanup ""
            _auto_restart_next
        else
            _archive_and_cleanup "_err"
            if _auto_recover; then
                _archive_and_cleanup "_resumed"
                _auto_restart_next
            fi
        fi
        break
    fi

    if [[ ${_elapsed} -ge ${MONITOR_INTERVAL} ]]; then
        _elapsed=0
        if _is_log_stale; then
            _archive_and_cleanup "_err"
            if _auto_recover; then
                _archive_and_cleanup "_resumed"
                _auto_restart_next
            fi
            break
        fi
        cur_step=$(grep -oE 'step:[0-9]+' "${LOG_FILE}" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        _monitor_log "Healthy: step=${cur_step:-?}/${STEPS}"
    fi
done

_monitor_log "Monitoring finished"
