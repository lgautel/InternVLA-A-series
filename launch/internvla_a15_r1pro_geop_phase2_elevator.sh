#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# R1 Pro Elevator Button Task — Phase 2 Full SFT (GeoPredict E1 7D Keypoints)
#
# 数据集: elevator0714_lerobot_4D (100 ep / 27145 frames, E1 7D kpt, 3 cameras)
# 起点:   Phase 1 Warmup ckpt@400 (WARMUP_CKPT 必须在外部 export 或传入)
# 策略:   全量微调 (VLM + Action + Kpt Expert) + WAN video loss + VQA/FAST
# E1 改动: kpt_4d_mode=pos_rot → 自动派生 keypoint_track_input_dim=7, keypoint_dim=7
#
# Usage (正式, 2-GPU 本机):
#   export WARMUP_CKPT=/path/to/phase1/checkpoints/000400/pretrained_model
#   bash launch/internvla_a15_r1pro_geop_phase2_elevator.sh
#
# Usage (WAN smoke, 1GPU 2steps):
#   WAN_SMOKE=1 WARMUP_CKPT=... bash launch/internvla_a15_r1pro_geop_phase2_elevator.sh
#
# Usage (smoke 100 steps):
#   SMOKE=1 WARMUP_CKPT=... bash launch/internvla_a15_r1pro_geop_phase2_elevator.sh
#
# 换机器时修改的关键变量 (或通过 export 覆盖):
#   TRAIN_VENV, PROJ_ROOT, WAN_DIR, HF_HOME, KPT_4D_MODE
#   PROC_PER_NODE, BATCH_SIZE, CUDA_VISIBLE_DEVICES, EXPR_NAME
#
# 正式训练自带 post-training 监控 (错误/成功检测 + 自动打包 + bigmatrix 启动)
# 监控配置: MONITOR_INTERVAL, STALE_THRESHOLD, ARCHIVE_SOURCE, ARCHIVE_DEST
#
# 参考文档: b/d/R1Pro/p2sft_plan.md
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
EXPR_NAME="${EXPR_NAME:-ItvlaGpR1proElvt0904}"
TRAIN_VENV="${TRAIN_VENV:-/home/luogang/miniforge3/envs/itvlaGp}"
PYTHON="${PYTHON:-${TRAIN_VENV}/bin/python}"

export HF_HOME="${HF_HOME:-/home/luogang/hf_home}"
# HF_LEROBOT_HOME 默认值已与 constants.py 一致: ${HF_HOME}/lerobot
# 数据集通过 symlink 注册到该目录，不依赖也不侵入实际数据位置
KPT_4D_MODE="${KPT_4D_MODE:-pos_rot}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USE_LIBUV="${USE_LIBUV:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-libnccl-tuner-disabled.so}"

# Triton cache 放本机 XFS，避免 Ceph/NFS 多 rank 文件锁竞争 (sft0827LOG.md §22:48)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/itvla-triton-cache}"

# WARMUP_CKPT 必须由外部 export，或在命令行传入
WARMUP_CKPT="${WARMUP_CKPT:?请先 export WARMUP_CKPT=<Phase1_ckpt@400/pretrained_model 路径>}"

WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
DATA_REPO_ID="${DATA_REPO_ID:-elevator0714_lerobot_4D}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
NORM_STATS="${NORM_STATS:-${HF_LEROBOT_HOME}/${DATA_REPO_ID}/meta/norm_stat_abs.json}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-36603}"

WAN_SMOKE="${WAN_SMOKE:-0}"
SMOKE="${SMOKE:-0}"

# ── Post-training monitoring (formal mode only) ──
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1800}"
STALE_THRESHOLD="${STALE_THRESHOLD:-900}"
ARCHIVE_SOURCE="${ARCHIVE_SOURCE:-/B}"
ARCHIVE_DEST="${ARCHIVE_DEST:-${HOME}/b/Ckp}"
BIGMATRIX_SCRIPT="${BIGMATRIX_SCRIPT:-${PROJ_ROOT}/b/d/GpRbt/bigmatrix_multiply_optimization.py}"
BIGMATRIX_MAX_RETRIES="${BIGMATRIX_MAX_RETRIES:-5}"

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
    JOB_SUFFIX="r1pro-elev-geop-p2-wan-smoke"
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
    JOB_SUFFIX="r1pro-elev-geop-p2-smoke100"
else
    # 正式训练：本机 2×RTX PRO 6000 Blackwell (~97 GB/卡)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    PROC_PER_NODE="${PROC_PER_NODE:-2}"
    BATCH_SIZE="${BATCH_SIZE:-8}"    # 保守起点；显存足够可升至 12
    STEPS="${STEPS:-10000}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    SAVE_FREQ="${SAVE_FREQ:-2500}"
    LOG_FREQ="${LOG_FREQ:-50}"
    SCHEDULER_WARMUP="${SCHEDULER_WARMUP:-1000}"
    WANDB_ENABLE="${WANDB_ENABLE:-true}"
    JOB_SUFFIX="r1pro-elev-geop-p2-e1-sft"
fi

NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

cd "${PROJ_ROOT}"

JOB_NAME="${JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-internvla_a1_5-${JOB_SUFFIX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/internvla_a1_5/${JOB_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

echo "=== R1 Pro Elevator Phase 2 SFT (E1 7D GeoPredict) ==="
echo "EXPR_NAME=${EXPR_NAME}"
echo "KPT_4D_MODE=${KPT_4D_MODE}"
echo "WARMUP_CKPT=${WARMUP_CKPT}"
echo "DATA_REPO_ID=${DATA_REPO_ID}"
echo "PROC=${NUM_PROCESSES} BS=${BATCH_SIZE} STEPS=${STEPS}"
echo "WAN_DIR=${WAN_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# accelerate 分布式参数
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

    # ── 模型与起点 ──
    --policy.type=internvla_a1_5
    --policy.repo_id=lerobot_lab/internvla_a1_5
    --policy.push_to_hub=false
    --policy.pretrained_path="${WARMUP_CKPT}"
    # gradient_checkpointing: WAN+3摄像头+全训 显存压力大，必须开启
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16
    --policy.vlm_model_name_or_path=Qwen/Qwen3.5-2B

    # ── 优化器 ──
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP}"
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5e-6

    # ── 全量微调开关（Phase 2 全开，与 Phase 1 相反）──
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

    # ── E1 7D 关键点（核心差异）──
    --policy.enable_keypoint_predictor=true
    --policy.num_keypoint_joints=16
    --policy.kpt_4d_mode="${KPT_4D_MODE}"  # pos_only → 3D | pos_rot → 7D (pos+quat)
    --policy.kpt_rot_loss_weight=1.0       # 旋转 loss 权重（相对位置 loss）
    --policy.keypoint_history_max_len=200  # must match Phase 1 (200 → pos_embed [50, 256])

    # ── Phase 2 loss 权重 ──
    --policy.action_loss_weight=10.0
    --policy.kpt_loss_weight=1.0
    --policy.kpt_future_loss_weight=1.5
    --policy.kpt_to_action_detach=false

    # ── Phase 2 安全检查（必须 false）──
    --policy.init_kpt_expert_from_action=false
    # 注意：不设 --policy.geopredict_checkpoint_path（Phase 1 已写入 ckpt）

    # ── 学习率分组 ──
    --policy.freeze_keypoint_modules=false
    --policy.action_expert_lr_scale=1.0
    --policy.kpt_expert_lr_scale=1.0
    --policy.track_encoder_lr_scale=1.0

    # ── 数据集 ──
    --dataset.type=internvla_a1_5
    --dataset.repo_id="${DATA_REPO_ID}"
    --dataset.enable_keypoint_predictor=true
    --dataset.num_keypoint_joints=16
    --dataset.kpt_4d_mode="${KPT_4D_MODE}" # 与 policy 保持一致
    --dataset.keypoint_history_max_len=200
    --dataset.action_mode=abs
    --dataset.use_external_stats=true
    --dataset.external_stats_path="${NORM_STATS}"
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
    --wandb.project=internvla_a1_5
    --wandb.mode=offline
)

###############################################################################
# Monitoring helpers
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

    # 1. 立即释放 GPU（最高优先级）
    _kill_gpu_processes

    # 2. 立即拉起 bigmatrix 占用 GPU（高优先级，后台运行）
    _start_bigmatrix || true

    # 3. 慢慢做打包备份（低优先级，不占 GPU，前台同步执行）
    local ts
    ts=$(_monitor_ts)
    local archive_name="${EXPR_NAME}_${ts}${suffix}"
    mkdir -p "${ARCHIVE_DEST}"
    _monitor_log "Archiving ${ARCHIVE_SOURCE} → ${ARCHIVE_DEST}/${archive_name}.tar (bigmatrix already running)"

    tar -cf "${ARCHIVE_DEST}/${archive_name}.tar" \
        -C "$(dirname "${ARCHIVE_SOURCE}")" "$(basename "${ARCHIVE_SOURCE}")"
    _monitor_log "Archive done: ${ARCHIVE_DEST}/${archive_name}.tar ($(du -sh "${ARCHIVE_DEST}/${archive_name}.tar" 2>/dev/null | cut -f1))"
}

###############################################################################
# Training execution
###############################################################################

if [[ "${WAN_SMOKE}" == "1" || "${SMOKE}" == "1" ]]; then
    # ── Smoke mode: blocking execution, no monitoring ──
    set -o pipefail
    "${PYTHON}" -m accelerate.commands.launch "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
    train_exit=${PIPESTATUS[0]}
    echo "post_check: exit=${train_exit}"
    exit "${train_exit}"
fi

# ── Formal training: background execution + automated monitoring ──
set +e

_monitor_log "=== Automated monitoring enabled ==="
_monitor_log "EXPR_NAME=${EXPR_NAME}  INTERVAL=${MONITOR_INTERVAL}s  STALE=${STALE_THRESHOLD}s"
_monitor_log "ARCHIVE: ${ARCHIVE_SOURCE} → ${ARCHIVE_DEST}"

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
        cur_step=$(grep -oE 'step[=: ]+[0-9]+' "${LOG_FILE}" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')
        _monitor_log "Healthy: step=${cur_step:-?}/${STEPS}"
    fi
done

_monitor_log "Monitoring finished"
