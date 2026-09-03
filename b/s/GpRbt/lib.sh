#!/usr/bin/env bash
# Shared helpers for the RoboTwin per-task P0/P1/P2 loop.
# Sourced by run_each_rbt_p012.sh and phase*.sh. Do not execute directly.
set -euo pipefail

rbt_log() {
  local ts
  ts="$(date +'%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] $*"
}

rbt_die() {
  echo "错误: $*" >&2
  exit 1
}

rbt_abs() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "${p}"
  else
    python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${p}"
  fi
}

rbt_expand() {
  # Expand ~ and $VARS in a path-like string.
  local raw="$1"
  eval echo "${raw}"
}

rbt_require_file() {
  local f="$1" msg="${2:-}"
  [[ -f "${f}" ]] || rbt_die "${msg:-找不到文件}: ${f}"
}

rbt_require_dir() {
  local d="$1" msg="${2:-}"
  [[ -d "${d}" ]] || rbt_die "${msg:-找不到目录}: ${d}"
}

rbt_mkdir() {
  mkdir -p "$@"
}

rbt_json_get() {
  local file="$1" key="$2"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], ''))" "${file}" "${key}"
}

count_cuda_devices() {
  local list="${1:-}"
  if [[ -z "${list}" ]]; then
    echo 0
    return
  fi
  local commas="${list//[^,]/}"
  echo $(( ${#commas} + 1 ))
}

build_cuda_devices() {
  local n="$1" i
  local -a devices=()
  for ((i = 0; i < n; i++)); do
    devices+=("$i")
  done
  local IFS=,
  echo "${devices[*]}"
}

# Resolve all per-task paths into globals. Safe to call repeatedly.
resolve_task_paths() {
  local task="$1"
  TASK_NAME="${task}"
  TASK_SRC="${CLEAN_ROOT}/${task}"
  TASK_KPTSIM="${KPTSIM_ROOT}/${task}_kptsim"
  TASK_LRB="${LRB_ROOT}/${task}_kptsim_lrb"
  TASK_V30="${V30_ROOT}/${task}_kptsim_lrbv30"
  TASK_REPO_ID="${task}_kptsim_lrbv30"
  TASK_NORM_RAW="${NORM_STATS_DIR}/robotwin_norm_stats_${task}.json"
  TASK_NORM_TRAIN="${TASK_V30}/norm_stat.json"
  TASK_KPT_META="${TASK_V30}/meta/keypoints_meta.json"
  TASK_CKPT_DIR="${CKPT_ROOT}/${task}"
  TASK_WARMUP_DIR="${TASK_CKPT_DIR}/warmup"
  TASK_SFT_DIR="${TASK_CKPT_DIR}/sft"
  TASK_LOG_DIR="${TASK_CKPT_DIR}/logs"
  TASK_STATE="${TASK_CKPT_DIR}/pipeline_state.json"
  TASK_CONVERT_WS="${CONVERT_WORK_ROOT}/${task}"
  TASK_WARMUP_LATEST="${TASK_WARMUP_DIR}/latest"
  TASK_SFT_LATEST="${TASK_SFT_DIR}/latest"
}

print_task_paths() {
  cat <<EOF
TASK_NAME        : ${TASK_NAME}
TASK_SRC         : ${TASK_SRC}
TASK_KPTSIM      : ${TASK_KPTSIM}
TASK_LRB         : ${TASK_LRB}
TASK_V30         : ${TASK_V30}
TASK_REPO_ID     : ${TASK_REPO_ID}
TASK_NORM_RAW    : ${TASK_NORM_RAW}
TASK_CKPT_DIR    : ${TASK_CKPT_DIR}
TASK_CONVERT_WS  : ${TASK_CONVERT_WS}
EOF
}

write_state() {
  local phase="$1" status="$2" extra_json="${3:-{}}"
  python3 - "${TASK_STATE}" "${TASK_NAME}" "${phase}" "${status}" "${extra_json}" <<'PY'
import json, sys, datetime
from pathlib import Path

path, task, phase, status, extra = sys.argv[1:6]
p = Path(path)
data = {}
if p.exists():
    data = json.loads(p.read_text())
data.setdefault("task", task)
data.setdefault("phases", {})
try:
    extra_obj = json.loads(extra) if extra else {}
except json.JSONDecodeError:
    extra_obj = {"raw": extra}
data["phases"][phase] = {
    "status": status,
    "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    **extra_obj,
}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
}

phase_status() {
  local phase="$1"
  if [[ ! -f "${TASK_STATE}" ]]; then
    echo ""
    return
  fi
  python3 -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get('phases',{}).get(sys.argv[2],{}).get('status',''))
" "${TASK_STATE}" "${phase}"
}

v30_ready() {
  [[ -f "${TASK_V30}/meta/info.json" ]] && [[ -f "${TASK_V30}/norm_stat.json" ]] && [[ -f "${TASK_V30}/meta/keypoints_meta.json" ]]
}

_is_pretrained_dir() {
  local p="$1"
  [[ -f "${p}/config.json" ]] || return 1
  [[ -f "${p}/model.safetensors" || -f "${p}/model.safetensors.index.json" ]]
}

warmup_ckpt_path() {
  if [[ -n "${LOCAL_WARMUP_CKPT_ROOT:-}" ]]; then
    local local_p="${LOCAL_WARMUP_CKPT_ROOT}/${TASK_NAME}-000400"
    if _is_pretrained_dir "${local_p}"; then
      echo "${local_p}"
      return 0
    fi
  fi
  local p="${TASK_WARMUP_LATEST}/checkpoints/000400/pretrained_model"
  if _is_pretrained_dir "${p}"; then
    echo "${p}"
    return 0
  fi
  local found=""
  if [[ -d "${TASK_WARMUP_DIR}" ]]; then
    found="$(find "${TASK_WARMUP_DIR}" -path '*/checkpoints/000400/pretrained_model/config.json' 2>/dev/null | sort | tail -1 || true)"
  fi
  if [[ -n "${found}" ]]; then
    echo "$(dirname "${found}")"
    return 0
  fi
  echo ""
}

ensure_lerobot_home_link() {
  # Training factory resolves HF_LEROBOT_HOME / repo_id. If the v3.0 dataset
  # does not already live there, drop a symlink (never copy).
  local dest="${HF_LEROBOT_HOME}/${TASK_REPO_ID}"
  local src_abs dest_abs
  src_abs="$(rbt_abs "${TASK_V30}")"
  rbt_mkdir "${HF_LEROBOT_HOME}"
  dest_abs="$(rbt_abs "${dest}")"
  if [[ "${src_abs}" == "${dest_abs}" ]]; then
    return 0
  fi
  if [[ -e "${dest}" && ! -L "${dest}" ]]; then
    rbt_die "HF_LEROBOT_HOME 下已有非符号链接 ${dest}, 拒绝覆盖"
  fi
  ln -sfn "${src_abs}" "${dest}"
  rbt_log "symlink ${dest} -> ${src_abs}"
}

# Timestamped identity for one training run. Same task can be launched many times:
# each run gets a unique JOB_STAMP used as OUTPUT_DIR name, wandb run name, and log suffix.
# Do NOT mkdir OUTPUT_DIR: TrainPipelineConfig raises FileExistsError if it already exists.
init_train_run() {
  local phase="$1"
  local parent=""
  case "${phase}" in
    warmup) parent="${TASK_WARMUP_DIR}" ;;
    sft) parent="${TASK_SFT_DIR}" ;;
    *) rbt_die "init_train_run: 未知 phase ${phase}" ;;
  esac
  JOB_STAMP="$(date +'%Y_%m_%d_%H_%M_%S')"
  JOB_NAME="${JOB_STAMP}-internvla_a1_5-geop-kpt-${phase}-${TASK_NAME}"
  OUTPUT_DIR="${parent}/${JOB_NAME}"
  if [[ -e "${OUTPUT_DIR}" ]]; then
    JOB_STAMP="${JOB_STAMP}-p$$"
    JOB_NAME="${JOB_STAMP}-internvla_a1_5-geop-kpt-${phase}-${TASK_NAME}"
    OUTPUT_DIR="${parent}/${JOB_NAME}"
  fi
  LOG_FILE="${TASK_LOG_DIR}/${phase}_${JOB_STAMP}.log"
  rbt_mkdir "${parent}" "${TASK_LOG_DIR}"
  if [[ -e "${OUTPUT_DIR}" ]]; then
    rbt_die "OUTPUT_DIR 已存在, 拒绝覆盖: ${OUTPUT_DIR}"
  fi
}

run_logged() {
  local log_file="$1"
  shift
  rbt_mkdir "$(dirname "${log_file}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    rbt_log "DRY-RUN: $*"
    return 0
  fi
  rbt_log "RUN: $*"
  {
    echo "===== $(date +'%Y-%m-%d %H:%M:%S') ====="
    echo "CMD: $*"
  } >> "${log_file}"
  "$@" >> "${log_file}" 2>&1
}
