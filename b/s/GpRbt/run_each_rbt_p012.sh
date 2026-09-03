#!/usr/bin/env bash
# Loop Phase0 (data) -> Phase1 (warmup 400) -> Phase2 (SFT, configurable epochs @ effective bs 128)
# over a list of RoboTwin 2.0 tasks. Does not reimplement extract / train; it calls
# existing GeoPredict + itvlaGp scripts with per-task isolated paths.
#
# Design: b/d/rbt/run_ech_rbt_p012.md
# Code lives in b/s/rbt/ (this directory).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

FROM_STAGE="phase0"
UNTIL_STAGE="sft"
LIST_TASKS=0
TASKS_SPEC=""
CONFIG_FILE=""
DRY_RUN=0
FORCE=0
SKIP_EXISTING=1
KEEP_GOING=0
SKIP_SMOKE="${SKIP_SMOKE:-0}"
GPUS=""
SFT_EPOCHS_CLI=""

STAGES=(phase0 warmup sft)

usage() {
  cat <<'EOF'
用法:
  bash b/s/rbt/run_each_rbt_p012.sh --config b/s/rbt/config.env
  bash b/s/rbt/run_each_rbt_p012.sh --config config.env --tasks b/s/rbt/tasks.batch1.txt
  bash b/s/rbt/run_each_rbt_p012.sh --config config.env --tasks place_bread_skillet,pick_dual_bottles
  bash b/s/rbt/run_each_rbt_p012.sh --config config.env --list-tasks
  bash b/s/rbt/run_each_rbt_p012.sh --config config.env --from warmup --until sft
  bash b/s/rbt/run_each_rbt_p012.sh --config config.env --dry-run

选项:
  --config PATH           机器本地路径配置 (source 一个 env 文件)
  --tasks SPEC            任务列表文件、逗号分隔任务名、或 all
                          省略时默认 b/s/rbt/tasks.batch1.txt
                          (place_bread_skillet, pick_dual_bottles)
  --list-tasks            列出 CLEAN_ROOT 下的源任务后退出
  --from STAGE            phase0 | warmup | sft  (默认 phase0)
  --until STAGE           phase0 | warmup | sft  (默认 sft)
  --gpus N                覆盖 PROC_PER_NODE 与 CUDA_VISIBLE_DEVICES=0..N-1
  --sft-epochs N          覆盖 SFT 总 epoch 数 (默认 76, 或 config 中 SFT_EPOCHS)
  --skip-existing         阶段产物已存在则跳过 (默认)
  --no-skip-existing      不因已有产物而跳过 (仍不覆盖 ckpt 目录; 新 run 用时间戳)
  --force                 强制重做当前范围内的阶段 (Phase0 会重建数据)
  --skip-smoke            跳过 warmup/sft 的 1 步 smoke
  --keep-going            某个任务失败后继续下一个
  --dry-run               只打印将要执行的命令
  -h, --help              帮助

CLEAN_ROOT 默认 /home/a26113/Dta/RoboTwin-Clean (必须可配置, 换机器请改 --config).
设计文档: b/d/rbt/run_ech_rbt_p012.md
EOF
}

stage_index() {
  local s="$1" i
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "${s}" ]]; then
      echo "${i}"
      return 0
    fi
  done
  return 1
}

should_run_stage() {
  local s="$1"
  local idx from_i until_i
  idx="$(stage_index "${s}")" || rbt_die "未知阶段: ${s}"
  from_i="$(stage_index "${FROM_STAGE}")" || rbt_die "未知 --from: ${FROM_STAGE}"
  until_i="$(stage_index "${UNTIL_STAGE}")" || rbt_die "未知 --until: ${UNTIL_STAGE}"
  [[ "${idx}" -ge "${from_i}" && "${idx}" -le "${until_i}" ]]
}

load_tasks() {
  local spec="$1"
  TASKS=()
  if [[ "${spec}" == "all" ]]; then
    local name
    while IFS= read -r name || [[ -n "${name}" ]]; do
      [[ -z "${name}" ]] && continue
      TASKS+=("${name}")
    done < <("${TRAIN_PYTHON:-python3}" "${SCRIPT_DIR}/discover_source_tasks.py" --clean-root "${CLEAN_ROOT}" --names-only)
    [[ ${#TASKS[@]} -gt 0 ]] || rbt_die "CLEAN_ROOT 下没有源任务: ${CLEAN_ROOT}"
    return 0
  fi
  if [[ -f "${spec}" ]]; then
    local line
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%%#*}"
      line="${line#"${line%%[![:space:]]*}"}"
      line="${line%"${line##*[![:space:]]}"}"
      [[ -z "${line}" ]] && continue
      TASKS+=("${line}")
    done < "${spec}"
  else
    local item IFS=,
    read -r -a TASKS <<< "${spec}"
    local trimmed=()
    for item in "${TASKS[@]}"; do
      item="${item#"${item%%[![:space:]]*}"}"
      item="${item%"${item##*[![:space:]]}"}"
      [[ -n "${item}" ]] && trimmed+=("${item}")
    done
    TASKS=("${trimmed[@]}")
  fi
  [[ ${#TASKS[@]} -gt 0 ]] || rbt_die "任务列表为空: ${spec}"
}

apply_defaults() {
  ITVLAGP_ROOT="${ITVLAGP_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
  if [[ -z "${GEOPREDICT_ROOT:-}" && -d "${ITVLAGP_ROOT}/../GeoPredict" ]]; then
    GEOPREDICT_ROOT="$(cd "${ITVLAGP_ROOT}/../GeoPredict" && pwd)"
  fi
  GEOPREDICT_ROOT="${GEOPREDICT_ROOT:-}"
  ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-}"
  CLEAN_ROOT="${CLEAN_ROOT:-/home/a26113/Dta/RoboTwin-Clean}"
  CKPT_ROOT="$(rbt_expand "${CKPT_ROOT:-${HOME}/Ckp/itvlaGp}")"

  KPTSIM_ROOT="${KPTSIM_ROOT:-${CLEAN_ROOT}}"
  LRB_ROOT="${LRB_ROOT:-${CLEAN_ROOT}}"
  V30_ROOT="${V30_ROOT:-${CLEAN_ROOT}}"
  CONVERT_WORK_ROOT="${CONVERT_WORK_ROOT:-${CKPT_ROOT}/.convert_ws}"
  NORM_STATS_DIR="${NORM_STATS_DIR:-${CKPT_ROOT}/norm_stats}"

  VENV_ROOT="${VENV_ROOT:-/tmp/itnvla15rbt20}"
  TRAIN_PYTHON="${TRAIN_PYTHON:-${VENV_ROOT}/bin/python}"
  EXTRACT_PYTHON="${EXTRACT_PYTHON:-${TRAIN_PYTHON}}"

  HF_HOME="${HF_HOME:-${VENV_ROOT}/var/hf_home}"
  HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${V30_ROOT}}"
  PRETRAINED_PATH="${PRETRAINED_PATH:-${HF_HOME}/ckpts/InternVLA-A1.5-base}"
  GEOPREDICT_CKPT="${GEOPREDICT_CKPT:-${HF_HOME}/ckpts/GeoPredict_robocasa.pth}"
  WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
  URDF_PATH="${URDF_PATH:-${ROBOTWIN_ROOT}/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}"

  PROC_PER_NODE="${PROC_PER_NODE:-8}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  NODE_COUNT="${NODE_COUNT:-1}"
  NODE_RANK="${NODE_RANK:-0}"
  WARMUP_STEPS="${WARMUP_STEPS:-400}"
  SFT_EPOCHS="${SFT_EPOCHS:-76}"
  SFT_EFFECTIVE_BATCH_TARGET="${SFT_EFFECTIVE_BATCH_TARGET:-128}"
  WARMUP_MASTER_PORT="${WARMUP_MASTER_PORT:-36201}"
  SFT_MASTER_PORT="${SFT_MASTER_PORT:-36202}"
  WARMUP_SAVE_FREQ="${WARMUP_SAVE_FREQ:-100}"
  NUM_KEYPOINT_JOINTS="${NUM_KEYPOINT_JOINTS:-14}"

  if [[ -n "${GPUS}" ]]; then
    PROC_PER_NODE="${GPUS}"
    CUDA_VISIBLE_DEVICES="$(build_cuda_devices "${GPUS}")"
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(build_cuda_devices "${PROC_PER_NODE}")}"
}

preflight() {
  rbt_log "==== Preflight ===="
  [[ -n "${GEOPREDICT_ROOT}" ]] || rbt_die "请设置 GEOPREDICT_ROOT"
  [[ -n "${CLEAN_ROOT}" ]] || rbt_die "请设置 CLEAN_ROOT"
  [[ -n "${ROBOTWIN_ROOT}" ]] || rbt_die "请设置 ROBOTWIN_ROOT (用于 URDF)"
  rbt_require_dir "${ITVLAGP_ROOT}" "ITVLAGP_ROOT"
  rbt_require_dir "${GEOPREDICT_ROOT}" "GEOPREDICT_ROOT"
  rbt_require_dir "${CLEAN_ROOT}" "CLEAN_ROOT"
  rbt_require_file "${URDF_PATH}" "URDF"
  rbt_require_file "${TRAIN_PYTHON}" "TRAIN_PYTHON"
  rbt_require_file "${EXTRACT_PYTHON}" "EXTRACT_PYTHON"
  rbt_require_file "${ITVLAGP_ROOT}/launch/internvla_a15_geop_phase1_kpt_warmup_kptsim_8g.sh"
  rbt_require_file "${ITVLAGP_ROOT}/launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh"
  rbt_mkdir "${CKPT_ROOT}" "${NORM_STATS_DIR}" "${CONVERT_WORK_ROOT}"

  local n_dev
  n_dev="$(count_cuda_devices "${CUDA_VISIBLE_DEVICES}")"
  if [[ "${n_dev}" -ne "${PROC_PER_NODE}" ]]; then
    rbt_log "警告: CUDA_VISIBLE_DEVICES 设备数=${n_dev} 与 PROC_PER_NODE=${PROC_PER_NODE} 不一致"
  fi
  local ebs=$((PROC_PER_NODE * BATCH_SIZE * NODE_COUNT))
  rbt_log "有效 batch = ${PROC_PER_NODE} * ${BATCH_SIZE} * ${NODE_COUNT} = ${ebs} (目标 ${SFT_EFFECTIVE_BATCH_TARGET})"
  if [[ "${ebs}" -ne "${SFT_EFFECTIVE_BATCH_TARGET}" ]]; then
    rbt_log "警告: 有效 batch 不是 ${SFT_EFFECTIVE_BATCH_TARGET}; SFT 步数按实际有效 batch 与 SFT_EPOCHS=${SFT_EPOCHS} 换算"
  fi
  rbt_log "SFT_EPOCHS=${SFT_EPOCHS} (各任务 steps/save 点按该任务的 total_frames 单独计算)"

  local t
  for t in "${TASKS[@]}"; do
    case "${t}" in
      *_kptsim|*_kptsim_lrb|*_kptsim_lrbv30|*_old)
        rbt_die "「${t}」像是流水线产物目录名, 不是源任务. 源任务是 CLEAN_ROOT 下带 meta/info.json 的子文件夹"
        ;;
    esac
    [[ -d "${CLEAN_ROOT}/${t}" ]] || rbt_die "源任务目录不存在: ${CLEAN_ROOT}/${t}"
    [[ -f "${CLEAN_ROOT}/${t}/meta/info.json" ]] || rbt_die "缺少 info.json: ${CLEAN_ROOT}/${t}"
  done
  rbt_log "任务数=${#TASKS[@]}: ${TASKS[*]}"
  rbt_log "ITVLAGP_ROOT=${ITVLAGP_ROOT}"
  rbt_log "GEOPREDICT_ROOT=${GEOPREDICT_ROOT}"
  rbt_log "CLEAN_ROOT=${CLEAN_ROOT}"
  rbt_log "CKPT_ROOT=${CKPT_ROOT}"
  rbt_log "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
}

run_task() {
  local task="$1"
  resolve_task_paths "${task}"
  rbt_mkdir "${TASK_CKPT_DIR}" "${TASK_LOG_DIR}" "${TASK_WARMUP_DIR}" "${TASK_SFT_DIR}"

  local lock="${TASK_CKPT_DIR}/.lock"
  if [[ -d "${lock}" ]]; then
    rbt_die "任务 ${task} 已有锁 ${lock} (若确认无其它进程, 删除该目录后重试)"
  fi
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir "${lock}"
  fi
  local rc=0
  cleanup_lock() { [[ "${DRY_RUN}" == "1" ]] || rmdir "${lock}" 2>/dev/null || true; }

  export ITVLAGP_ROOT GEOPREDICT_ROOT ROBOTWIN_ROOT CLEAN_ROOT
  export CKPT_ROOT KPTSIM_ROOT LRB_ROOT V30_ROOT CONVERT_WORK_ROOT NORM_STATS_DIR
  export VENV_ROOT TRAIN_PYTHON EXTRACT_PYTHON
  export HF_HOME HF_LEROBOT_HOME PRETRAINED_PATH GEOPREDICT_CKPT WAN_DIR URDF_PATH
  export PROC_PER_NODE BATCH_SIZE NODE_COUNT NODE_RANK
  export CUDA_VISIBLE_DEVICES WARMUP_STEPS SFT_EPOCHS SFT_EFFECTIVE_BATCH_TARGET
  export WARMUP_MASTER_PORT SFT_MASTER_PORT WARMUP_SAVE_FREQ
  export FORCE SKIP_EXISTING DRY_RUN SKIP_SMOKE NUM_KEYPOINT_JOINTS

  if should_run_stage phase0; then
    bash "${SCRIPT_DIR}/phase0_prep_data.sh" "${task}" || rc=$?
  fi
  if [[ "${rc}" -eq 0 ]] && should_run_stage warmup; then
    bash "${SCRIPT_DIR}/phase1_warmup.sh" "${task}" || rc=$?
  fi
  if [[ "${rc}" -eq 0 ]] && should_run_stage sft; then
    bash "${SCRIPT_DIR}/phase2_sft.sh" "${task}" || rc=$?
  fi
  cleanup_lock
  return "${rc}"
}

# --- CLI ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --tasks) TASKS_SPEC="$2"; shift 2 ;;
    --list-tasks) LIST_TASKS=1; shift ;;
    --from) FROM_STAGE="$2"; shift 2 ;;
    --until) UNTIL_STAGE="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --sft-epochs) SFT_EPOCHS_CLI="$2"; shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --no-skip-existing) SKIP_EXISTING=0; shift ;;
    --force) FORCE=1; shift ;;
    --skip-smoke) SKIP_SMOKE=1; shift ;;
    --keep-going) KEEP_GOING=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) rbt_die "未知参数: $1 (见 --help)" ;;
  esac
done

[[ -n "${TASKS_SPEC}" || "${LIST_TASKS}" == "1" ]] || TASKS_SPEC="${SCRIPT_DIR}/tasks.batch1.txt"
if [[ -n "${CONFIG_FILE}" ]]; then
  [[ -f "${CONFIG_FILE}" ]] || rbt_die "找不到配置文件: ${CONFIG_FILE}"
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_FILE}"
  set +a
fi

apply_defaults
if [[ -n "${SFT_EPOCHS_CLI}" ]]; then
  SFT_EPOCHS="${SFT_EPOCHS_CLI}"
fi
[[ "${SFT_EPOCHS}" =~ ^[0-9]+$ ]] && [[ "${SFT_EPOCHS}" -gt 0 ]] || rbt_die "SFT_EPOCHS 须为正整数, 当前=${SFT_EPOCHS}"

if [[ "${LIST_TASKS}" == "1" ]]; then
  [[ -n "${CLEAN_ROOT}" ]] || rbt_die "请设置 CLEAN_ROOT"
  rbt_log "CLEAN_ROOT=${CLEAN_ROOT} 源任务:"
  python3 "${SCRIPT_DIR}/discover_source_tasks.py" --clean-root "${CLEAN_ROOT}"
  exit 0
fi

load_tasks "${TASKS_SPEC}"
preflight

failed=()
for task in "${TASKS[@]}"; do
  rbt_log "######## 开始任务 ${task} ########"
  if run_task "${task}"; then
    rbt_log "######## 完成任务 ${task} ########"
  else
    rbt_log "######## 失败任务 ${task} ########"
    failed+=("${task}")
    if [[ "${KEEP_GOING}" != "1" ]]; then
      rbt_die "任务 ${task} 失败; 使用 --keep-going 可继续后续任务"
    fi
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  rbt_die "失败任务: ${failed[*]}"
fi
rbt_log "全部 ${#TASKS[@]} 个任务完成"
