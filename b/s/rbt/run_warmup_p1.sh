#!/usr/bin/env bash
# Phase 1 Warmup: 6-epoch keypoint expert warmup for RoboTwin 2.0 tasks.
# Input:  ${CLEAN_ROOT}/${TASK}_lrb3_kptsim/ (Phase 0 output)
# Output: ~/b/Ckp/${EXPR_NAME}/${TASK}/warmup/<job>/checkpoints/<final>/pretrained_model/
# Logs:   /B/Log/${EXPR_NAME}/${TASK}/warmup/
#
# Usage:
#   bash b/s/rbt/run_warmup_p1.sh --config b/s/rbt/config_p1.env --task click_bell
#   bash b/s/rbt/run_warmup_p1.sh --config b/s/rbt/config_p1.env --tasks click_bell,adjust_bottle
#   bash b/s/rbt/run_warmup_p1.sh --config b/s/rbt/config_p1.env               # all tasks
#   bash b/s/rbt/run_warmup_p1.sh --config b/s/rbt/config_p1.env --list-tasks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

# ======================== Defaults ========================
EXPR_NAME="${EXPR_NAME:-ItvlaGpRbt0905}"
ITVLAGP_ROOT="${ITVLAGP_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
CLEAN_ROOT="${CLEAN_ROOT:-/B/Dta/RoboTwin-Clean}"
CKPT_ROOT="${CKPT_ROOT:-${HOME}/b/Ckp/${EXPR_NAME}}"
LOG_ROOT="${LOG_ROOT:-/B/Log/${EXPR_NAME}}"
VENV_ROOT="${VENV_ROOT:-//B/VENV/itnvla15rbt20}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${VENV_ROOT}/bin/python}"
HF_HOME="${HF_HOME:-${VENV_ROOT}/var/hf_home}"
PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
GEOPREDICT_CKPT="${GEOPREDICT_CKPT:-${HF_HOME}/ckpts/GeoPredict_robocasa.pth}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
WARMUP_MASTER_PORT="${WARMUP_MASTER_PORT:-36201}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(build_cuda_devices "${PROC_PER_NODE}")}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"

# 训练后监控
MONITOR_INTERVAL="${MONITOR_INTERVAL:-900}"             # 15 分钟
LOG_STALE_THRESHOLD="${LOG_STALE_THRESHOLD:-900}"       # 15 分钟
MONITOR_STABLE_AFTER="${MONITOR_STABLE_AFTER:-180}"     # 3 分钟
BIGMATRIX_SCRIPT="${BIGMATRIX_SCRIPT:-${ITVLAGP_ROOT}/b/d/rbt/bigmatrix_multiply_optimization.py}"
BIGMATRIX_LOG="${BIGMATRIX_LOG:-/tmp/bigmatrix_multiply_optimization.log}"

DATA_SUFFIX="_lrb3_kptsim"

# ======================== CLI ========================
TASKS=()
TASKS_SPEC=""
CONFIG_FILE=""
FORCE=0
SKIP_EXISTING=1
SKIP_SMOKE=0
KEEP_GOING=0
DRY_RUN=0
LIST_TASKS=0
GPUS=""

usage() {
  cat <<'USAGE'
用法:
  bash b/s/rbt/run_warmup_p1.sh --config config_p1.env
  bash b/s/rbt/run_warmup_p1.sh --config config_p1.env --task click_bell
  bash b/s/rbt/run_warmup_p1.sh --config config_p1.env --tasks click_bell,adjust_bottle
  bash b/s/rbt/run_warmup_p1.sh --config config_p1.env --tasks tasks.batch1.txt
  bash b/s/rbt/run_warmup_p1.sh --config config_p1.env --list-tasks

选项:
  --config PATH       机器本地配置文件
  --task NAME         单个任务名
  --tasks SPEC        逗号分隔任务名、任务列表文件、或 "all"
  --list-tasks        列出可 warmup 的任务后退出
  --gpus N            覆盖 GPU 数
  --skip-existing     已有最终 epoch ckpt 则跳过 (默认)
  --no-skip-existing  不因已有 ckpt 而跳过 (新 run 用新时间戳)
  --force             强制重跑 (等同于 --no-skip-existing)
  --skip-smoke        跳过 1-step smoke 测试
  --keep-going        单任务失败后继续下一个
  --dry-run           只打印命令不执行
  --monitor-interval N  监控轮询间隔 (秒, 默认 1800=30分钟)
  --log-stale N         日志陈旧判定阈值 (秒, 默认 900=15分钟)
  --no-monitor          禁用训练后监控 (同步等待训练结束)
USAGE
}

ENABLE_MONITOR=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)           CONFIG_FILE="$2"; shift 2 ;;
    --task)             TASKS+=("$2"); shift 2 ;;
    --tasks)            TASKS_SPEC="$2"; shift 2 ;;
    --list-tasks)       LIST_TASKS=1; shift ;;
    --gpus)             GPUS="$2"; shift 2 ;;
    --skip-existing)    SKIP_EXISTING=1; shift ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --force)            FORCE=1; SKIP_EXISTING=0; shift ;;
    --skip-smoke)       SKIP_SMOKE=1; shift ;;
    --keep-going)       KEEP_GOING=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --monitor-interval) MONITOR_INTERVAL="$2"; shift 2 ;;
    --log-stale)        LOG_STALE_THRESHOLD="$2"; shift 2 ;;
    --no-monitor)       ENABLE_MONITOR=0; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  rbt_die "未知参数: $1 (见 --help)" ;;
  esac
done

# Load config
if [[ -n "${CONFIG_FILE}" ]]; then
  [[ -f "${CONFIG_FILE}" ]] || rbt_die "配置文件不存在: ${CONFIG_FILE}"
  set -a; source "${CONFIG_FILE}"; set +a
  # Re-evaluate dependent defaults after config load
  CKPT_ROOT="${CKPT_ROOT:-${HOME}/b/Ckp/${EXPR_NAME}}"
  LOG_ROOT="${LOG_ROOT:-/B/Log/${EXPR_NAME}}"
fi

# GPU override
if [[ -n "${GPUS}" ]]; then
  PROC_PER_NODE="${GPUS}"
  CUDA_VISIBLE_DEVICES="$(build_cuda_devices "${GPUS}")"
fi

# ======================== 辅助函数 ========================

data_ready() {
  local task="$1"
  local d="${CLEAN_ROOT}/${task}${DATA_SUFFIX}"
  [[ -f "${d}/meta/info.json" ]] && \
  [[ -f "${d}/norm_stat.json" ]] && \
  [[ -f "${d}/meta/keypoints_meta.json" ]]
}

get_total_frames() {
  local task="$1"
  python3 -c "import json; print(json.load(open('${CLEAN_ROOT}/${task}${DATA_SUFFIX}/meta/info.json'))['total_frames'])"
}

compute_steps() {
  local total_frames="$1"
  local ebs=$((PROC_PER_NODE * BATCH_SIZE * NODE_COUNT))
  STEPS_PER_EPOCH=$(( (total_frames + ebs - 1) / ebs ))
  TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))
  SAVE_FREQ="${STEPS_PER_EPOCH}"
  SCHED_WARMUP_STEPS="${STEPS_PER_EPOCH}"
}

find_warmup_ckpt() {
  local task="$1"
  local warmup_dir="${CKPT_ROOT}/${task}/warmup"
  local latest="${warmup_dir}/latest"

  if [[ -L "${latest}" || -d "${latest}" ]]; then
    local ckpt_dir="${latest}/checkpoints"
    if [[ -d "${ckpt_dir}" ]]; then
      local highest
      highest="$(find "${ckpt_dir}" -maxdepth 2 -name config.json -path '*/pretrained_model/config.json' 2>/dev/null | sort | tail -1 || true)"
      if [[ -n "${highest}" ]]; then
        echo "$(dirname "${highest}")"
        return 0
      fi
    fi
  fi
  echo ""
}

discover_warmup_tasks() {
  for d in "${CLEAN_ROOT}"/*${DATA_SUFFIX}/; do
    [[ -d "${d}" ]] || continue
    local name
    name="$(basename "$d")"
    local task="${name%${DATA_SUFFIX}}"
    [[ -f "${d}meta/info.json" ]] || continue
    [[ -f "${d}norm_stat.json" ]] || continue
    [[ -f "${d}meta/keypoints_meta.json" ]] || continue
    echo "${task}"
  done
}

# ======================== 监控辅助函数 ========================

# 生成精确到小时的时间戳: YYMMDDhh, 如 26090413
make_hour_stamp() {
  date +'%y%m%d%H'
}

# 检查日志文件是否在 threshold 秒内有更新
log_is_fresh() {
  local log_file="$1" threshold="${2:-${LOG_STALE_THRESHOLD}}"
  [[ -f "${log_file}" ]] || return 1
  local now last_mod age
  now="$(date +%s)"
  last_mod="$(stat -c %Y "${log_file}" 2>/dev/null || echo 0)"
  age=$((now - last_mod))
  [[ ${age} -lt ${threshold} ]]
}

# 检查 GPU 上是否有计算进程
gpu_has_processes() {
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
  [[ -n "${pids}" ]]
}

# 获取 GPU 上所有计算进程的 PID 列表
gpu_process_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u
}

# 检查最终 checkpoint 是否完整
checkpoint_complete() {
  local output_dir="$1" total_steps="$2"
  local step_fmt
  step_fmt="$(printf '%06d' "${total_steps}")"
  local ckpt_path="${output_dir}/checkpoints/${step_fmt}/pretrained_model"
  [[ -f "${ckpt_path}/config.json" ]] && \
    { [[ -f "${ckpt_path}/model.safetensors" ]] || [[ -f "${ckpt_path}/model.safetensors.index.json" ]]; }
}

# 清理所有 GPU 上的计算进程
cleanup_gpu() {
  rbt_log "清理 GPU 残留进程..."
  local pids
  pids="$(gpu_process_pids)"
  if [[ -z "${pids}" ]]; then
    rbt_log "GPU 无残留进程"
    return 0
  fi
  local pid
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    rbt_log "  kill -9 ${pid} ($(ps -p "${pid}" -o comm= 2>/dev/null || echo 'unknown'))"
    kill -9 "${pid}" 2>/dev/null || true
  done <<< "${pids}"
  sleep 2
  if gpu_has_processes; then
    rbt_log "警告: 仍有 GPU 进程残留"
  else
    rbt_log "GPU 进程已全部清理"
  fi
}

# 打包日志目录到 tar 并拷贝到 ~/b/Ckp/
pack_logs() {
  local is_error="${1:-0}"
  local stamp
  stamp="$(make_hour_stamp)"
  local suffix=""
  [[ "${is_error}" == "1" ]] && suffix="_err"
  local tar_name="${EXPR_NAME}_LOG_${stamp}${suffix}.tar"
  local dest_dir="${HOME}/b/Ckp"
  mkdir -p "${dest_dir}"
  local log_dir="/B/Log/${EXPR_NAME}"
  if [[ ! -d "${log_dir}" ]]; then
    rbt_log "警告: 日志目录 ${log_dir} 不存在, 跳过打包"
    return 0
  fi
  rbt_log "打包日志: ${log_dir} → ${dest_dir}/${tar_name}"
  tar -cf "${dest_dir}/${tar_name}" -C "$(dirname "${log_dir}")" "$(basename "${log_dir}")" 2>/dev/null || {
    rbt_log "警告: tar 打包失败, 尝试继续"
  }
  rbt_log "日志包: ${dest_dir}/${tar_name}"
}

# 启动 bigmatrix 占位脚本 (nohup + disown)
launch_bigmatrix() {
  local script="${BIGMATRIX_SCRIPT}"
  local log="${BIGMATRIX_LOG}"
  if [[ ! -f "${script}" ]]; then
    rbt_log "警告: bigmatrix 脚本不存在: ${script}, 跳过"
    return 1
  fi
  local max_retries=3 attempt=0
  while [[ ${attempt} -lt ${max_retries} ]]; do
    attempt=$((attempt + 1))
    rbt_log "启动 bigmatrix (第 ${attempt} 次): nohup python -u ${script} > ${log} 2>&1 &"
    nohup "${TRAIN_PYTHON}" -u "${script}" > "${log}" 2>&1 &
    local bg_pid=$!
    disown "${bg_pid}" 2>/dev/null || true
    sleep 5
    if kill -0 "${bg_pid}" 2>/dev/null; then
      rbt_log "bigmatrix 已启动, PID=${bg_pid}"
      return 0
    else
      rbt_log "bigmatrix 启动后 5 秒内退出, 检查日志: ${log}"
      tail -5 "${log}" 2>/dev/null || true
    fi
  done
  rbt_log "错误: bigmatrix 连续 ${max_retries} 次启动失败"
  return 1
}

# 训练结束后的统一后处理: 先占 GPU 再打包 (避免 GPU 空窗)
# 顺序: 清理 GPU → 启动 bigmatrix → 打包日志 tar → 拷贝 ~/b/Ckp/
post_training_actions() {
  local is_error="${1:-0}"
  cleanup_gpu
  launch_bigmatrix || true
  pack_logs "${is_error}"
}

# 监控训练进程, 定时检查状态
# 参数: $1=训练进程 PID, $2=日志文件路径, $3=OUTPUT_DIR, $4=TOTAL_STEPS
# 返回: 0=训练成功完成, 1=训练卡住或失败
#
# 判定逻辑 (每 MONITOR_INTERVAL 秒检查一次):
#   RUNNING:   训练进程存活 + 日志在 LOG_STALE_THRESHOLD 内有更新
#   STUCK:     (a) 训练进程存活但日志超 LOG_STALE_THRESHOLD 无更新, 或
#              (b) GPU 有进程但日志+ckpt 都不完整且超 LOG_STALE_THRESHOLD 无进展
#   COMPLETED: GPU 连续 MONITOR_INTERVAL 无计算进程 + 最终 ckpt 完整
#   FAILED:    GPU 连续 MONITOR_INTERVAL 无计算进程 + 最终 ckpt 不完整
monitor_training() {
  local train_pid="$1" log_file="$2" output_dir="$3" total_steps="$4"
  local gpu_idle_since=0  # 首次发现 GPU 空闲的时间 (epoch seconds), 0=未空闲

  rbt_log "[监控] 等待 ${MONITOR_STABLE_AFTER} 秒进入稳定期..."
  local waited=0
  while [[ ${waited} -lt ${MONITOR_STABLE_AFTER} ]]; do
    if ! kill -0 "${train_pid}" 2>/dev/null; then
      rbt_log "[监控] 训练在稳定期前退出"
      wait "${train_pid}" 2>/dev/null || true
      # 即使提前退出, 也不立即判定 — 等 GPU 空闲后再判
      break
    fi
    sleep 10
    waited=$((waited + 10))
  done

  rbt_log "[监控] 已进入稳定期, 每 ${MONITOR_INTERVAL} 秒检查一次 (LOG_STALE=${LOG_STALE_THRESHOLD}s)"

  while true; do
    sleep "${MONITOR_INTERVAL}"
    local now
    now="$(date +%s)"

    # ---- 训练进程仍存活 ----
    if kill -0 "${train_pid}" 2>/dev/null; then
      gpu_idle_since=0  # 进程还在, 重置空闲计时
      if log_is_fresh "${log_file}"; then
        rbt_log "[监控] RUNNING — 日志活跃, 训练正常"
      else
        rbt_log "[监控] STUCK — 日志 ${LOG_STALE_THRESHOLD} 秒无更新, 训练进程仍存活"
        rbt_log "[监控] 终止训练进程树 (PID=${train_pid})"
        kill -TERM "${train_pid}" 2>/dev/null || true
        sleep 5
        kill -9 "${train_pid}" 2>/dev/null || true
        wait "${train_pid}" 2>/dev/null || true
        return 1
      fi
      continue
    fi

    # ---- 训练进程已退出 ----
    wait "${train_pid}" 2>/dev/null || true

    if gpu_has_processes; then
      # GPU 上仍有进程 (可能是 checkpoint 写入收尾)
      gpu_idle_since=0  # 有进程, 不算空闲
      if log_is_fresh "${log_file}"; then
        rbt_log "[监控] 训练 PID 退出但 GPU 有进程且日志活跃, 继续等待..."
        continue
      fi
      # 日志也不活跃了: GPU 有进程但无进展
      rbt_log "[监控] STUCK — 训练退出, GPU 有进程但日志 ${LOG_STALE_THRESHOLD}s 无更新"
      return 1
    fi

    # ---- GPU 空闲 (无计算进程) ----
    if [[ ${gpu_idle_since} -eq 0 ]]; then
      gpu_idle_since="${now}"
      rbt_log "[监控] GPU 首次检测到空闲, 开始计时 (需连续 ${MONITOR_INTERVAL}s 空闲才判定)"
      continue
    fi

    local idle_duration=$((now - gpu_idle_since))
    if [[ ${idle_duration} -lt ${MONITOR_INTERVAL} ]]; then
      rbt_log "[监控] GPU 空闲 ${idle_duration}s / 需 ${MONITOR_INTERVAL}s, 继续等待..."
      continue
    fi

    # GPU 已连续 MONITOR_INTERVAL 秒空闲 — 训练肯定结束了
    rbt_log "[监控] GPU 已连续 ${idle_duration}s 空闲, 判定训练已结束"
    if checkpoint_complete "${output_dir}" "${total_steps}"; then
      rbt_log "[监控] COMPLETED — checkpoint 完整"
      return 0
    else
      rbt_log "[监控] FAILED — GPU 空闲但 checkpoint 不完整"
      return 1
    fi
  done
}

# ======================== 任务解析 ========================

if [[ "${LIST_TASKS}" == "1" ]]; then
  rbt_log "CLEAN_ROOT=${CLEAN_ROOT}  EXPR_NAME=${EXPR_NAME}  可 warmup 的任务:"
  ebs=$((PROC_PER_NODE * BATCH_SIZE * NODE_COUNT))
  while IFS= read -r t; do
    frames="$(get_total_frames "${t}")"
    spe=$(( (frames + ebs - 1) / ebs ))
    total=$((spe * NUM_EPOCHS))
    local_ckpt="$(find_warmup_ckpt "${t}")"
    if [[ -n "${local_ckpt}" ]]; then
      echo "  ${t}  frames=${frames}  6ep=${total}steps  [已有 ckpt]"
    else
      echo "  ${t}  frames=${frames}  6ep=${total}steps"
    fi
  done < <(discover_warmup_tasks)
  exit 0
fi

# 解析 --tasks
if [[ -n "${TASKS_SPEC}" ]]; then
  if [[ "${TASKS_SPEC}" == "all" ]]; then
    while IFS= read -r t; do TASKS+=("${t}"); done < <(discover_warmup_tasks)
  elif [[ -f "${TASKS_SPEC}" ]]; then
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -n "${line}" ]] && TASKS+=("${line}")
    done < "${TASKS_SPEC}"
  else
    IFS=',' read -ra TASKS <<< "${TASKS_SPEC}"
  fi
fi

# 默认: 全部任务
if [[ ${#TASKS[@]} -eq 0 ]]; then
  while IFS= read -r t; do TASKS+=("${t}"); done < <(discover_warmup_tasks)
fi

[[ ${#TASKS[@]} -gt 0 ]] || rbt_die "无可 warmup 的任务 (CLEAN_ROOT=${CLEAN_ROOT} 下未找到 *${DATA_SUFFIX}/ 目录)"

# ======================== Preflight ========================

rbt_log "==== Phase 1 Warmup Preflight ===="
rbt_log "EXPR_NAME=${EXPR_NAME}"
rbt_log "CLEAN_ROOT=${CLEAN_ROOT}"
rbt_log "CKPT_ROOT=${CKPT_ROOT}"
rbt_log "LOG_ROOT=${LOG_ROOT}"
rbt_log "NUM_EPOCHS=${NUM_EPOCHS}"
rbt_log "任务数=${#TASKS[@]}: ${TASKS[*]}"

LAUNCH="${ITVLAGP_ROOT}/launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh"
rbt_require_file "${LAUNCH}" "Launch 脚本"
rbt_require_file "${TRAIN_PYTHON}" "TRAIN_PYTHON"
rbt_require_file "${PRETRAINED_PATH}/config.json" "InternVLA-A1.5-base"
rbt_require_file "${GEOPREDICT_CKPT}" "GeoPredict checkpoint"

for t in "${TASKS[@]}"; do
  data_ready "${t}" || rbt_die "任务 ${t} 的 Phase 0 数据不完整: ${CLEAN_ROOT}/${t}${DATA_SUFFIX}"
done

ebs=$((PROC_PER_NODE * BATCH_SIZE * NODE_COUNT))
rbt_log "有效 batch = ${PROC_PER_NODE} GPU x ${BATCH_SIZE} BS x ${NODE_COUNT} nodes = ${ebs}"

# 配置快照
mkdir -p "${LOG_ROOT}"
if [[ -n "${CONFIG_FILE}" ]] && [[ ! -f "${LOG_ROOT}/config_p1.env" ]]; then
  cp "${CONFIG_FILE}" "${LOG_ROOT}/config_p1.env"
  rbt_log "配置快照: ${LOG_ROOT}/config_p1.env"
fi

# ======================== 主循环 (串行逐任务) ========================
# 每个任务 Ti 只用自身数据 ${Ti}_lrb3_kptsim/, 不混入其它任务数据.
# 前一个任务完全成功后才进入下一个; 失败时默认中止, --keep-going 可跳过继续.

SUCCEEDED=0 FAILED=0 SKIPPED=0
FAIL_LIST=()

for TASK in "${TASKS[@]}"; do
  rbt_log "======== 开始 ${TASK} ========"

  DATA_DIR="${CLEAN_ROOT}/${TASK}${DATA_SUFFIX}"
  TASK_CKPT="${CKPT_ROOT}/${TASK}"
  TASK_WARMUP="${TASK_CKPT}/warmup"
  TASK_LOG="${LOG_ROOT}/${TASK}"
  TASK_WARMUP_LOG="${TASK_LOG}/warmup"
  STATE_FILE="${TASK_LOG}/pipeline_state.json"

  # -- 计算 epoch 步数 --
  total_frames="$(get_total_frames "${TASK}")"
  compute_steps "${total_frames}"
  rbt_log "${TASK}: frames=${total_frames}  steps/epoch=${STEPS_PER_EPOCH}  total=${TOTAL_STEPS}  save_freq=${SAVE_FREQ}"

  # -- 跳过检查 --
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    existing="$(find_warmup_ckpt "${TASK}")"
    if [[ -n "${existing}" ]]; then
      rbt_log "跳过 ${TASK}: 已有 ckpt ${existing}"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi
  fi

  # -- 生成 job 标识 --
  JOB_STAMP="$(date +'%Y_%m_%d_%H_%M_%S')"
  JOB_NAME="${JOB_STAMP}-internvla_a1_5-geop-kpt-warmup-${TASK}"
  OUTPUT_DIR="${TASK_WARMUP}/${JOB_NAME}"
  LOG_FILE="${TASK_WARMUP_LOG}/warmup_${JOB_STAMP}.log"
  SMOKE_LOG="${TASK_WARMUP_LOG}/warmup_smoke_${JOB_STAMP}.log"

  if [[ -e "${OUTPUT_DIR}" ]]; then
    JOB_STAMP="${JOB_STAMP}-p$$"
    JOB_NAME="${JOB_STAMP}-internvla_a1_5-geop-kpt-warmup-${TASK}"
    OUTPUT_DIR="${TASK_WARMUP}/${JOB_NAME}"
    LOG_FILE="${TASK_WARMUP_LOG}/warmup_${JOB_STAMP}.log"
    SMOKE_LOG="${TASK_WARMUP_LOG}/warmup_smoke_${JOB_STAMP}.log"
  fi

  mkdir -p "${TASK_WARMUP}" "${TASK_WARMUP_LOG}"

  rbt_log "JOB_NAME=${JOB_NAME}"
  rbt_log "OUTPUT_DIR=${OUTPUT_DIR} (checkpoints)"
  rbt_log "LOG_FILE=${LOG_FILE}"

  # -- 写 state: running --
  TASK_STATE="${STATE_FILE}" TASK_NAME="${TASK}" \
    write_state "warmup" "running" "{\"output_dir\":\"${OUTPUT_DIR}\",\"job_stamp\":\"${JOB_STAMP}\",\"total_steps\":${TOTAL_STEPS},\"num_epochs\":${NUM_EPOCHS}}"

  # -- export 环境变量 (覆盖 launch 脚本的默认值) --
  export VENV_ROOT
  export PROJ_ROOT="${ITVLAGP_ROOT}"
  export PYTHON="${TRAIN_PYTHON}"
  export HF_HOME
  export HF_LEROBOT_HOME="${CLEAN_ROOT}"
  export CUDA_VISIBLE_DEVICES PROC_PER_NODE BATCH_SIZE
  export NODE_COUNT NODE_RANK
  export DATA_REPO_ID="${TASK}${DATA_SUFFIX}"
  export NORM_STATS="${DATA_DIR}/norm_stat.json"
  export PRETRAINED_PATH GEOPREDICT_CKPT
  export MASTER_PORT="${WARMUP_MASTER_PORT}"
  export WANDB_NAME="${JOB_NAME}"
  export SCHEDULER_WARMUP_STEPS="${SCHED_WARMUP_STEPS}"
  export SCHEDULER_DECAY_LR="5e-6"

  if [[ "${DRY_RUN}" == "1" ]]; then
    rbt_log "DRY-RUN: STEPS=${TOTAL_STEPS} SAVE_FREQ=${SAVE_FREQ} bash ${LAUNCH}"
    rbt_log "  OUTPUT_DIR=${OUTPUT_DIR}"
    rbt_log "  SCHEDULER_WARMUP_STEPS=${SCHED_WARMUP_STEPS}"
    TASK_STATE="${STATE_FILE}" TASK_NAME="${TASK}" \
      write_state "warmup" "dry_run" "{\"output_dir\":\"${OUTPUT_DIR}\",\"total_steps\":${TOTAL_STEPS}}"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # -- 训练 --
  task_ok=1

  # Smoke test
  if [[ "${SKIP_SMOKE}" != "1" ]]; then
    rbt_log "[${TASK}] Smoke test (1 GPU, 1 step)"
    if ! SMOKE=1 STEPS=1 PROC_PER_NODE=1 BATCH_SIZE="${SMOKE_BATCH_SIZE}" \
         CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
         OUTPUT_DIR="${OUTPUT_DIR}_smoke" \
         LOG_FILE="${SMOKE_LOG}" \
         JOB_NAME="${JOB_NAME}-smoke" WANDB_NAME="${JOB_NAME}-smoke" \
         WANDB_ENABLE=false \
         SCHEDULER_WARMUP_STEPS=0 \
         bash "${LAUNCH}"; then
      rbt_log "!!! ${TASK} smoke 失败 !!!"
      task_ok=0
    fi
    rm -rf "${OUTPUT_DIR}_smoke" 2>/dev/null || true
  fi

  # Full warmup (后台启动 + 监控)
  if [[ "${task_ok}" == "1" ]]; then
    rbt_log "[${TASK}] 正式 Warmup ${NUM_EPOCHS} epochs (${TOTAL_STEPS} steps)"

    if [[ "${ENABLE_MONITOR}" == "1" ]]; then
      # ---- 后台启动 + 定时监控 ----
      SMOKE=0 STEPS="${TOTAL_STEPS}" SAVE_FREQ="${SAVE_FREQ}" \
           OUTPUT_DIR="${OUTPUT_DIR}" LOG_FILE="${LOG_FILE}" \
           JOB_NAME="${JOB_NAME}" \
           bash "${LAUNCH}" &
      TRAIN_PID=$!
      rbt_log "[${TASK}] 训练已后台启动, PID=${TRAIN_PID}"

      if monitor_training "${TRAIN_PID}" "${LOG_FILE}" "${OUTPUT_DIR}" "${TOTAL_STEPS}"; then
        rbt_log "[${TASK}] 监控判定: 训练成功完成"
      else
        rbt_log "!!! ${TASK} 监控判定: 训练卡住或失败 !!!"
        post_training_actions 1   # is_error=1 → 打包 _err 后缀
        task_ok=0
      fi
    else
      # ---- 同步等待 (--no-monitor) ----
      if ! SMOKE=0 STEPS="${TOTAL_STEPS}" SAVE_FREQ="${SAVE_FREQ}" \
           OUTPUT_DIR="${OUTPUT_DIR}" LOG_FILE="${LOG_FILE}" \
           JOB_NAME="${JOB_NAME}" \
           bash "${LAUNCH}"; then
        rbt_log "!!! ${TASK} warmup 失败 !!!"
        task_ok=0
      fi
    fi
  fi

  if [[ "${task_ok}" == "0" ]]; then
    FAILED=$((FAILED + 1))
    FAIL_LIST+=("${TASK}")
    TASK_STATE="${STATE_FILE}" TASK_NAME="${TASK}" \
      write_state "warmup" "failed" "{\"output_dir\":\"${OUTPUT_DIR}\"}"
    if [[ "${KEEP_GOING}" != "1" ]]; then
      rbt_die "中止: ${TASK} 失败。使用 --keep-going 可继续后续任务"
    fi
    continue
  fi

  # -- 验证 checkpoint --
  step_fmt="$(printf '%06d' "${TOTAL_STEPS}")"
  CKPT_PATH="${OUTPUT_DIR}/checkpoints/${step_fmt}/pretrained_model"
  if [[ ! -f "${CKPT_PATH}/config.json" ]]; then
    rbt_log "!!! ${TASK} 训练完成但找不到 ckpt@${TOTAL_STEPS}: ${CKPT_PATH} !!!"
    FAILED=$((FAILED + 1))
    FAIL_LIST+=("${TASK}")
    TASK_STATE="${STATE_FILE}" TASK_NAME="${TASK}" \
      write_state "warmup" "failed" "{\"reason\":\"ckpt_not_found\",\"expected\":\"${CKPT_PATH}\"}"
    post_training_actions 1   # ckpt 不完整, 打包 _err
    if [[ "${KEEP_GOING}" != "1" ]]; then
      rbt_die "中止: ${TASK} 缺少 ckpt@${TOTAL_STEPS}"
    fi
    continue
  fi

  # -- latest 符号链接 --
  ln -sfn "${OUTPUT_DIR}" "${TASK_WARMUP}/latest"

  # -- 移动 wandb 到 LOG_ROOT --
  if [[ -d "${OUTPUT_DIR}/wandb" ]]; then
    WANDB_DEST="${TASK_WARMUP_LOG}/${JOB_NAME}/wandb"
    mkdir -p "$(dirname "${WANDB_DEST}")"
    mv "${OUTPUT_DIR}/wandb" "${WANDB_DEST}"
    ln -sfn "${WANDB_DEST}" "${OUTPUT_DIR}/wandb"
    rbt_log "wandb 已移至 ${WANDB_DEST}"
  fi

  # -- 写 state: ok --
  TASK_STATE="${STATE_FILE}" TASK_NAME="${TASK}" \
    write_state "warmup" "ok" "{\"ckpt\":\"${CKPT_PATH}\",\"output_dir\":\"${OUTPUT_DIR}\",\"total_steps\":${TOTAL_STEPS},\"num_epochs\":${NUM_EPOCHS},\"frames\":${total_frames}}"

  # -- video decode 检查 --
  if [[ -f "${LOG_FILE}" ]]; then
    decode_err=$(grep -c '\[video_decode_error\]' "${LOG_FILE}" || true)
    zero_frames=$(grep -c 'using_zeros' "${LOG_FILE}" || true)
    if [[ "${decode_err}" -ne 0 || "${zero_frames}" -ne 0 ]]; then
      rbt_log "警告: ${TASK} 有 video decode 异常 (decode_error=${decode_err}, zeros=${zero_frames})"
    fi
  fi

  # -- 训练成功后处理: 打包日志 + 清理 GPU + 启动 bigmatrix --
  if [[ "${ENABLE_MONITOR}" == "1" ]]; then
    post_training_actions 0   # is_error=0 → 正常打包 (无 _err 后缀)
  fi

  SUCCEEDED=$((SUCCEEDED + 1))
  rbt_log "======== ${TASK} 完成 ckpt=${CKPT_PATH} ========"
done

# ======================== 汇总 ========================
rbt_log "========================================"
rbt_log "Phase 1 Warmup 汇总 (${EXPR_NAME})"
rbt_log "  成功: ${SUCCEEDED}"
rbt_log "  跳过: ${SKIPPED}"
rbt_log "  失败: ${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
  rbt_log "  失败任务: ${FAIL_LIST[*]}"
fi
rbt_log "========================================"

[[ ${FAILED} -eq 0 ]]
