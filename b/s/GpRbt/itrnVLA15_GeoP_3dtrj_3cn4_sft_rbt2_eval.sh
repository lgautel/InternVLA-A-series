#!/usr/bin/env bash
# =============================================================================
# itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh
#
# 本机 RoboTwin 2.0 评测编排：按手册直接调用 evaluation/RoboTwin/inference.py
# （不用 eval.sh）。默认执行 hanging_mug @ GCS step-010000 全流程，并写入
# b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_evalLOG.md。
#
# 手册: b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md
#
# 用法（建议 tmux / 持久化会话，不要在会退出的 IDE shell 里裸跑正式 100 ep）:
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh --until smoke
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh --task-name stack_bowls_three --ckpt PATH --skip-gcs
#
# 环境变量与 CLI 同名（大写下划线），CLI 优先。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 与 evaluation/RoboTwin/inference.py 的 TASK_NAMES 保持一致。
TASK_NAMES=(
  adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
  click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block
  handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad
  move_playingcard_away move_stapler_pad open_laptop open_microwave
  pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
  place_bread_basket place_bread_skillet place_burger_fries place_can_basket
  place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup
  place_fan place_mouse_pad place_object_basket place_object_scale
  place_object_stand place_phone_stand place_shoe press_stapler
  put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object
  shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two
  stack_bowls_three stack_bowls_two stamp_seal turn_switch
)

STAGES=(gcs preflight smoke eval summarize)

# -----------------------------------------------------------------------------
# 默认（环境变量可先设；CLI 优先）。空值表示稍后按任务预设填充。
# -----------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-${_DEFAULT_REPO}}"
CONDA_ROOT="${CONDA_ROOT:-/home/luogang/miniforge3}"
CONDA_ENV="${CONDA_ENV:-itvlaGp}"
PYTHON="${PYTHON:-}"

TASK_NAME="${TASK_NAME:-}"
TASK_IDX="${TASK_IDX:-}"
CKPT="${CKPT:-}"
CKPT_STEP="${CKPT_STEP:-010000}"
GCS_JOB="${GCS_JOB:-}"
GCS_CKPT="${GCS_CKPT:-}"
KPT_META="${KPT_META:-}"
KPT_DATA_ROOT="${KPT_DATA_ROOT:-/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean}"
KPT_COORD_MODE="${KPT_COORD_MODE:-voxel}"
KPT_VARIANT="${KPT_VARIANT:-kptsim_lrbv30}"
OUT="${OUT:-}"
RUN_ID="${RUN_ID:-}"
EVAL_LOG="${EVAL_LOG:-}"
LOG_DIR="${LOG_DIR:-}"

ACTION_MODE="${ACTION_MODE:-abs}"
DTYPE="${DTYPE:-bfloat16}"
INFER_HORIZON="${INFER_HORIZON:-20}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
NUM_EPISODES="${NUM_EPISODES:-100}"
SMOKE_EPISODES="${SMOKE_EPISODES:-2}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
SEED="${SEED:-42}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
STATS_KEY="${STATS_KEY:-aloha}"
FPS="${FPS:-30}"
TRANSFORMERS_EXPECT="${TRANSFORMERS_EXPECT:-5.2.0}"
EXPECT_REPO_ID="${EXPECT_REPO_ID:-}"
EXPECT_OFFSET="${EXPECT_OFFSET:-}"
MIN_DISK_GB="${MIN_DISK_GB:-20}"
MIN_CKPT_BYTES="${MIN_CKPT_BYTES:-1000000000}"

CONFIGS="${CONFIGS:-demo_clean,demo_randomized}"
GPUS="${GPUS:-0,1}"
SMOKE_GPU="${SMOKE_GPU:-}"

FROM_STAGE="${FROM_STAGE:-gcs}"
UNTIL_STAGE="${UNTIL_STAGE:-summarize}"
DRY_RUN=0
FORCE_DOWNLOAD=0
SKIP_GCS=0
SKIP_SMOKE=0
SKIP_EVAL=0
SEQUENTIAL=0
KEEP_GOING=0
RESET_LOG=0
LIST_TASKS=0
PRINT_CONFIG=0
STATUS_ONLY=0
CKPT_EXPLICIT=0

usage() {
  cat <<'EOF'
用法: itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh [选项]

按 b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md 编排本机 RoboTwin 评测。
默认任务 hanging_mug、GCS step-010000、conda itvlaGp。其它任务改 --task-name / --ckpt 即可。

路径 / 环境:
  --repo-root PATH          仓库根（默认: 本脚本上两级）
  --conda-root PATH         conda 根（默认 /home/luogang/miniforge3）
  --conda-env NAME          conda 环境（默认 itvlaGp）
  --python PATH             python 解释器（默认: conda 环境内 python）
  --ckpt PATH               本地 pretrained_model 目录
  --ckpt-step N             GCS 步数目录名，如 010000（默认 010000）
  --gcs-job URI             GCS job 前缀（含 checkpoints/ 的上一级）
  --gcs-ckpt URI            GCS pretrained_model 完整 URI（覆盖 job+step）
  --out PATH                评测输出根目录
  --run-id ID               日志/输出命名（默认 itvlaGp_<slug>_p2_<step>）
  --eval-log PATH           执行记录 markdown
  --log-dir PATH            inference 文本日志目录（默认 <repo>/outputs/logs）
  --kpt-meta PATH           keypoints_meta.json
  --kpt-data-root PATH      按任务名拼 meta 的根目录
  --kpt-variant NAME        meta 目录后缀（默认 kptsim_lrbv30）
  --kpt-coord-mode MODE     voxel|footprint（默认 voxel）

任务 / 推理:
  --task-name NAME          RoboTwin 任务名（默认 hanging_mug）
  --task-idx N              任务索引（与 --task-name 必须一致）
  --configs LIST            逗号分隔，默认 demo_clean,demo_randomized
  --gpus LIST               逗号分隔 GPU，按 configs 顺序分配（默认 0,1）
  --smoke-gpu N             冒烟 GPU（默认 configs 的第一张卡）
  --num-episodes N          正式评测 episode（默认 100）
  --smoke-episodes N        冒烟 episode（默认 2）
  --action-mode abs|delta   默认 abs
  --dtype bfloat16|float32  默认 bfloat16
  --infer-horizon N         默认 20
  --inference-backend NAME  默认 standard（GeoP 必须 standard）
  --instruction-type TYPE   默认 unseen
  --seed N                  默认 42
  --resize-size N           默认 224
  --expect-repo-id ID       校验 train_config.dataset.repo_id；空则跳过
  --expect-offset X,Y,Z     校验 meta coord_offset（容差 1e-3）；空则只检查文件
  --min-disk-gb N           预检磁盘下限（默认 20）

阶段:
  --from STAGE              gcs | preflight | smoke | eval | summarize
  --until STAGE
  --skip-gcs                不下载，要求 --ckpt 已存在
  --skip-smoke
  --skip-eval
  --force-download          本地 ckpt 已存在也重新拉 GCS
  --sequential              多配置串行（默认双卡并行）
  --keep-going              冒烟失败仍继续正式评测
  --reset-log               覆盖已有 eval LOG
  --dry-run                 只打印命令
  --print-config            打印解析后的变量并退出
  --list-tasks              打印 task_idx 与任务名
  --status                  只统计已有 mp4 / 写 LOG，不跑推理
  -h, --help

示例:
  bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh
  bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh --until smoke
  bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh \
    --task-name stack_bowls_three --skip-gcs \
    --ckpt /path/to/pretrained_model
EOF
}

now() { date '+%Y-%m-%d %H:%M:%S'; }

task_idx_of() {
  local name="$1" i
  for i in "${!TASK_NAMES[@]}"; do
    if [[ "${TASK_NAMES[$i]}" == "${name}" ]]; then
      echo "${i}"
      return 0
    fi
  done
  return 1
}

task_name_of() {
  local idx="$1"
  if [[ "${idx}" =~ ^[0-9]+$ ]] && (( idx >= 0 && idx < ${#TASK_NAMES[@]} )); then
    echo "${TASK_NAMES[$idx]}"
    return 0
  fi
  return 1
}

task_slug() {
  case "$1" in
    hanging_mug) echo hngMg ;;
    stack_bowls_three) echo stkb3 ;;
    stack_bowls_two) echo stkb2 ;;
    scan_object) echo scnObj ;;
    *) echo "$1" | tr -c 'A-Za-z0-9\n' '_' | sed 's/_$//' ;;
  esac
}

format_step_tag() {
  local n=$((10#${CKPT_STEP}))
  if (( n % 1000 == 0 )); then
    printf '%03dk' $((n / 1000))
  else
    printf '%06d' "${n}"
  fi
}

stage_index() {
  local name="$1" i
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "${name}" ]]; then
      echo "${i}"
      return 0
    fi
  done
  echo "错误: 未知阶段 '${name}'。合法: ${STAGES[*]}" >&2
  exit 1
}

split_csv() {
  local IFS=,
  # shellcheck disable=SC2206
  local arr=($1)
  printf '%s\n' "${arr[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)            REPO_ROOT="$2"; shift 2 ;;
    --conda-root)           CONDA_ROOT="$2"; shift 2 ;;
    --conda-env)            CONDA_ENV="$2"; shift 2 ;;
    --python)               PYTHON="$2"; shift 2 ;;
    --ckpt|--ckpt-path)     CKPT="$2"; CKPT_EXPLICIT=1; shift 2 ;;
    --ckpt-step)            CKPT_STEP="$2"; shift 2 ;;
    --gcs-job)              GCS_JOB="$2"; shift 2 ;;
    --gcs-ckpt)             GCS_CKPT="$2"; shift 2 ;;
    --out|--output-dir)     OUT="$2"; shift 2 ;;
    --run-id)               RUN_ID="$2"; shift 2 ;;
    --eval-log)             EVAL_LOG="$2"; shift 2 ;;
    --log-dir)              LOG_DIR="$2"; shift 2 ;;
    --kpt-meta|--kpt-meta-path) KPT_META="$2"; shift 2 ;;
    --kpt-data-root)        KPT_DATA_ROOT="$2"; shift 2 ;;
    --kpt-variant)          KPT_VARIANT="$2"; shift 2 ;;
    --kpt-coord-mode)       KPT_COORD_MODE="$2"; shift 2 ;;
    --task-name)            TASK_NAME="$2"; shift 2 ;;
    --task-idx)             TASK_IDX="$2"; shift 2 ;;
    --configs)              CONFIGS="$2"; shift 2 ;;
    --gpus)                 GPUS="$2"; shift 2 ;;
    --smoke-gpu)            SMOKE_GPU="$2"; shift 2 ;;
    --num-episodes)         NUM_EPISODES="$2"; shift 2 ;;
    --smoke-episodes)       SMOKE_EPISODES="$2"; shift 2 ;;
    --action-mode)          ACTION_MODE="$2"; shift 2 ;;
    --dtype)                DTYPE="$2"; shift 2 ;;
    --infer-horizon)        INFER_HORIZON="$2"; shift 2 ;;
    --inference-backend)    INFERENCE_BACKEND="$2"; shift 2 ;;
    --instruction-type)     INSTRUCTION_TYPE="$2"; shift 2 ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --resize-size)          RESIZE_SIZE="$2"; shift 2 ;;
    --expect-repo-id)       EXPECT_REPO_ID="$2"; shift 2 ;;
    --expect-offset)        EXPECT_OFFSET="$2"; shift 2 ;;
    --min-disk-gb)          MIN_DISK_GB="$2"; shift 2 ;;
    --from)                 FROM_STAGE="$2"; shift 2 ;;
    --until)                UNTIL_STAGE="$2"; shift 2 ;;
    --skip-gcs)             SKIP_GCS=1; shift ;;
    --skip-smoke)           SKIP_SMOKE=1; shift ;;
    --skip-eval)            SKIP_EVAL=1; shift ;;
    --force-download)       FORCE_DOWNLOAD=1; shift ;;
    --sequential)           SEQUENTIAL=1; shift ;;
    --keep-going)           KEEP_GOING=1; shift ;;
    --reset-log)            RESET_LOG=1; shift ;;
    --dry-run)              DRY_RUN=1; shift ;;
    --print-config)         PRINT_CONFIG=1; shift ;;
    --list-tasks)           LIST_TASKS=1; shift ;;
    --status)               STATUS_ONLY=1; FROM_STAGE=summarize; UNTIL_STAGE=summarize; shift ;;
    -h|--help)              usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${LIST_TASKS}" -eq 1 ]]; then
  local_i=0
  for name in "${TASK_NAMES[@]}"; do
    printf '%2d  %s\n' "${local_i}" "${name}"
    local_i=$((local_i + 1))
  done
  exit 0
fi

# --- 解析任务名 / 索引 ---
if [[ -z "${TASK_NAME}" && -z "${TASK_IDX}" ]]; then
  TASK_NAME="hanging_mug"
fi
if [[ -n "${TASK_NAME}" && -z "${TASK_IDX}" ]]; then
  TASK_IDX="$(task_idx_of "${TASK_NAME}")" || {
    echo "错误: 未知任务名 '${TASK_NAME}'。用 --list-tasks 查看。" >&2
    exit 1
  }
elif [[ -z "${TASK_NAME}" && -n "${TASK_IDX}" ]]; then
  TASK_NAME="$(task_name_of "${TASK_IDX}")" || {
    echo "错误: task_idx ${TASK_IDX} 超出范围 0..$(( ${#TASK_NAMES[@]} - 1 ))" >&2
    exit 1
  }
else
  resolved="$(task_idx_of "${TASK_NAME}")" || {
    echo "错误: 未知任务名 '${TASK_NAME}'" >&2
    exit 1
  }
  if [[ "${resolved}" != "${TASK_IDX}" ]]; then
    echo "错误: --task-name ${TASK_NAME} 的索引是 ${resolved}，与 --task-idx ${TASK_IDX} 不一致" >&2
    exit 1
  fi
fi

SLUG="$(task_slug "${TASK_NAME}")"
STEP_TAG="$(format_step_tag)"
REPO_ROOT="${REPO_ROOT%/}"
CONDA_ROOT="${CONDA_ROOT%/}"
KPT_DATA_ROOT="${KPT_DATA_ROOT%/}"

# hanging_mug 手册默认 GCS；其它任务必须显式给 --gcs-job 或 --ckpt。
if [[ -z "${GCS_JOB}" && "${TASK_NAME}" == "hanging_mug" && "${CKPT_EXPLICIT}" -eq 0 ]]; then
  GCS_JOB="gs://physical-ai-data-eu/VENV/tmp/2026_08_26_01_24_16-internvla_a1_5-geop-phase2-finetune-kptsim-voxel-8g-10k"
fi
if [[ -z "${GCS_JOB}" && "${TASK_NAME}" == "scan_object" && "${CKPT_EXPLICIT}" -eq 0 ]]; then
  GCS_JOB="gs://physical-ai-data-eu/VENV/tmp/2026_08_26_07_15_45-itvlaGp_p2_8g10k_scan_object_kptsim_lrbv30"
fi
if [[ -z "${GCS_CKPT}" && -n "${GCS_JOB}" ]]; then
  GCS_CKPT="${GCS_JOB%/}/checkpoints/${CKPT_STEP}/pretrained_model"
fi

CKPT="${CKPT:-${REPO_ROOT}/outputs-gcs/${TASK_NAME}_p2_${STEP_TAG}/checkpoints/${CKPT_STEP}/pretrained_model}"
CKPT="${CKPT%/}"
RUN_ID="${RUN_ID:-itvlaGp_${SLUG}_p2_${STEP_TAG}}"
OUT="${OUT:-${REPO_ROOT}/outputs/robotwin/${RUN_ID}}"
OUT="${OUT%/}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/logs}"
LOG_DIR="${LOG_DIR%/}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/run_${RUN_ID}.log}"

if [[ -z "${EVAL_LOG}" ]]; then
  if [[ "${TASK_NAME}" == "hanging_mug" ]]; then
    EVAL_LOG="${REPO_ROOT}/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_evalLOG.md"
  else
    EVAL_LOG="${REPO_ROOT}/b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_${SLUG}_evalLOG.md"
  fi
fi

KPT_META="${KPT_META:-${KPT_DATA_ROOT}/${TASK_NAME}_${KPT_VARIANT}/meta/keypoints_meta.json}"

if [[ -z "${EXPECT_REPO_ID}" ]]; then
  EXPECT_REPO_ID="${TASK_NAME}_${KPT_VARIANT}"
fi
if [[ -z "${EXPECT_OFFSET}" && "${TASK_NAME}" == "hanging_mug" ]]; then
  EXPECT_OFFSET="-0.7718,-1.0504,0.4779"
fi
if [[ -z "${EXPECT_OFFSET}" && "${TASK_NAME}" == "stack_bowls_three" ]]; then
  EXPECT_OFFSET="-0.8117,-1.0236,0.5046"
fi
if [[ -z "${EXPECT_OFFSET}" && "${TASK_NAME}" == "scan_object" ]]; then
  EXPECT_OFFSET="-0.6748,-1.0345,0.6219"
fi

mapfile -t CONFIG_ARR < <(split_csv "${CONFIGS}")
mapfile -t GPU_ARR < <(split_csv "${GPUS}")
if [[ ${#CONFIG_ARR[@]} -lt 1 ]]; then
  echo "错误: --configs 为空" >&2
  exit 1
fi
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "错误: --gpus 为空" >&2
  exit 1
fi
SMOKE_GPU="${SMOKE_GPU:-${GPU_ARR[0]}}"

INFERENCE_PY="${REPO_ROOT}/evaluation/RoboTwin/inference.py"
ROBOTWIN_ROOT="${REPO_ROOT}/third_party/RoboTwin"
STATS_PY="${REPO_ROOT}/util_scripts/robotwin_result_stats.py"

FROM_IDX="$(stage_index "${FROM_STAGE}")"
UNTIL_IDX="$(stage_index "${UNTIL_STAGE}")"
if [[ "${FROM_IDX}" -gt "${UNTIL_IDX}" ]]; then
  echo "错误: --from ${FROM_STAGE} 在 --until ${UNTIL_STAGE} 之后" >&2
  exit 1
fi

should_run() {
  local name="$1" idx
  idx="$(stage_index "${name}")"
  if [[ "${idx}" -lt "${FROM_IDX}" || "${idx}" -gt "${UNTIL_IDX}" ]]; then
    return 1
  fi
  case "${name}" in
    gcs)   [[ "${SKIP_GCS}" -eq 0 ]] || return 1 ;;
    smoke) [[ "${SKIP_SMOKE}" -eq 0 ]] || return 1 ;;
    eval)  [[ "${SKIP_EVAL}" -eq 0 ]] || return 1 ;;
  esac
  return 0
}

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

echo_banner() {
  echo "========== $* =========="
}

print_config() {
  cat <<EOF
REPO_ROOT=${REPO_ROOT}
CONDA_ROOT=${CONDA_ROOT}
CONDA_ENV=${CONDA_ENV}
PYTHON=${PYTHON:-"(activate 后填充)"}
TASK_NAME=${TASK_NAME}
TASK_IDX=${TASK_IDX}
CKPT_STEP=${CKPT_STEP}
GCS_JOB=${GCS_JOB:-"(空)"}
GCS_CKPT=${GCS_CKPT:-"(空)"}
CKPT=${CKPT}
KPT_META=${KPT_META}
KPT_COORD_MODE=${KPT_COORD_MODE}
OUT=${OUT}
RUN_ID=${RUN_ID}
EVAL_LOG=${EVAL_LOG}
LOG_DIR=${LOG_DIR}
CONFIGS=${CONFIGS}
GPUS=${GPUS}
SMOKE_GPU=${SMOKE_GPU}
ACTION_MODE=${ACTION_MODE}
DTYPE=${DTYPE}
INFER_HORIZON=${INFER_HORIZON}
INFERENCE_BACKEND=${INFERENCE_BACKEND}
NUM_EPISODES=${NUM_EPISODES}
SMOKE_EPISODES=${SMOKE_EPISODES}
EXPECT_REPO_ID=${EXPECT_REPO_ID:-"(不校验)"}
EXPECT_OFFSET=${EXPECT_OFFSET:-"(不校验数值)"}
FROM=${FROM_STAGE} UNTIL=${UNTIL_STAGE}
EOF
}

if [[ "${PRINT_CONFIG}" -eq 1 ]]; then
  print_config
  exit 0
fi

# -----------------------------------------------------------------------------
# eval LOG
# -----------------------------------------------------------------------------
log_append() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${EVAL_LOG}")"
  printf '%s\n' "$*" >> "${EVAL_LOG}"
}

log_event() {
  local action="$1" result="${2:-OK}"
  log_append "| $(now) | ${action} | ${result} |"
  echo "[$(now)] ${action}  →  ${result}"
}

log_problem() {
  local symptom="$1" cause="${2:-}" fix="${3:-}" verify="${4:-}"
  log_append ""
  log_append "### Problem: ${symptom}"
  log_append ""
  log_append "| 项 | 内容 |"
  log_append "|----|------|"
  log_append "| **发现时机** | $(now) |"
  log_append "| **症状** | ${symptom} |"
  log_append "| **根因** | ${cause} |"
  log_append "| **修复** | ${fix} |"
  log_append "| **验证** | ${verify} |"
  log_append ""
}

log_command() {
  local cmd="$1" reason="${2:-}"
  log_append ""
  log_append "### 命令 $(now)"
  log_append ""
  log_append "**理由**：${reason}"
  log_append ""
  log_append '```bash'
  log_append "${cmd}"
  log_append '```'
  log_append ""
}

log_file_change() {
  local path="$1" op="$2" reason="${3:-}"
  log_append "| $(now) | \`${path}\` | ${op} | ${reason} |"
}

eval_handbook_md() {
  case "$1" in
    hanging_mug) echo "itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg_eval.md" ;;
    scan_object) echo "itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_scnObj_eval.md" ;;
    stack_bowls_three) echo "reprd_rbtwn_stackb3_itvlagp_3cn2_eval6000_2.md" ;;
    *) echo "itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh" ;;
  esac
}

append_log_init_sections() {
  local handbook
  handbook="$(eval_handbook_md "${TASK_NAME}")"
  log_append ""
  log_append "## 手册与脚本"
  log_append ""
  log_append "| 项 | 路径 |"
  log_append "|----|------|"
  log_append "| 操作手册 | \`${REPO_ROOT}/b/d/${handbook}\` |"
  log_append "| 评测脚本 | \`${REPO_ROOT}/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh\` |"
  log_append "| inference 入口 | \`${INFERENCE_PY}\` |"
  log_append "| RoboTwin 任务源码 | \`${ROBOTWIN_ROOT}/envs/${TASK_NAME}.py\` |"
  log_append ""
  log_append "## 关键路径速查"
  log_append ""
  log_append "| 用途 | 路径 |"
  log_append "|------|------|"
  log_append "| GCS job | \`${GCS_JOB:-（未设）}\` |"
  log_append "| GCS ckpt | \`${GCS_CKPT:-（未设）}\` |"
  log_append "| 本机 ckpt | \`${CKPT}\` |"
  log_append "| kpt meta | \`${KPT_META}\` |"
  log_append "| expect offset | \`${EXPECT_OFFSET:-（不校验数值）}\` |"
  log_append "| 评测输出 | \`${OUT}\` |"
  log_append "| 冒烟视频 | \`$(video_dir_for demo_clean smoke)\` |"
  log_append "| clean 视频 | \`$(video_dir_for demo_clean robotwin)\` |"
  log_append "| randomized 视频 | \`$(video_dir_for demo_randomized robotwin)\` |"
  log_append "| 冒烟 inference 日志 | \`${LOG_DIR}/smoke_${RUN_ID}.log\` |"
  log_append "| clean inference 日志 | \`${LOG_DIR}/eval_${RUN_ID}_demo_clean.log\` |"
  log_append "| randomized inference 日志 | \`${LOG_DIR}/eval_${RUN_ID}_demo_randomized.log\` |"
  log_append ""
  log_append "## 问题记录（报错 → 根因 → 修复 → 验证）"
  log_append ""
  log_append "（运行中遇错自动追加；无则留空）"
  log_append ""
  log_append "## 文件增删改记录"
  log_append ""
  log_append "| 时间 | 文件 | 操作 | 缘由 |"
  log_append "|------|------|------|------|"
  log_append ""
  log_append "## 操作命令记录"
  log_append ""
  log_append "（各阶段关键命令见下文「命令」小节与时间线）"
  log_append ""
}

init_eval_log() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] 将写入 EVAL_LOG=${EVAL_LOG}"
    return 0
  fi
  mkdir -p "$(dirname "${EVAL_LOG}")" "${LOG_DIR}" "${OUT}"
  if [[ -f "${EVAL_LOG}" && "${RESET_LOG}" -eq 0 ]]; then
    log_append ""
    log_append "---"
    log_append ""
    log_append "## 再次运行 $(now)"
    log_append ""
    log_append "| 时间 | 操作 | 结果 |"
    log_append "|------|------|------|"
    return 0
  fi
  local handbook
  handbook="$(eval_handbook_md "${TASK_NAME}")"
  cat > "${EVAL_LOG}" <<EOF
# itvlaGp RoboTwin \`${TASK_NAME}\` 评估执行日志

> 由 \`b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh\` 自动记录。
> 手册：[\`${handbook}\`](${handbook})

---

## 评估配置

| 项 | 值 |
|----|-----|
| **开始时间** | $(now) |
| **代码库** | \`${REPO_ROOT}\` |
| **Conda 环境** | \`${CONDA_ENV}\`（\`${CONDA_ROOT}/envs/${CONDA_ENV}\`） |
| **任务** | \`${TASK_NAME}\` (task_idx=${TASK_IDX}) |
| **Checkpoint** | \`${CKPT}\` |
| **GCS** | \`${GCS_CKPT:-（本地 ckpt）}\` |
| **kpt meta** | \`${KPT_META}\` |
| **kpt 坐标模式** | \`${KPT_COORD_MODE}\` |
| **推理后端** | \`${INFERENCE_BACKEND}\` |
| **动作模式** | \`${ACTION_MODE}\` |
| **dtype** | \`${DTYPE}\` |
| **infer-horizon** | ${INFER_HORIZON} |
| **每配置 episode** | ${NUM_EPISODES} |
| **配置** | \`${CONFIGS}\` |
| **GPU** | \`${GPUS}\` |
| **输出目录** | \`${OUT}\` |
| **RUN_ID** | \`${RUN_ID}\` |
| **控制台日志** | \`${RUN_LOG}\` |

---

EOF
  append_log_init_sections
  cat >> "${EVAL_LOG}" <<EOF
## 时间线 / 操作日志

| 时间 | 操作 | 结果 |
|------|------|------|
EOF
}

count_videos() {
  local dir="$1" kind="$2"
  local n=0
  if [[ -d "${dir}" ]]; then
    n="$(find "${dir}" -maxdepth 1 -type f -name "${kind}_*.mp4" 2>/dev/null | wc -l | tr -d ' ')"
  fi
  echo "${n}"
}

video_dir_for() {
  local cfg="$1" kind="${2:-robotwin}"
  echo "${OUT}/${kind}/${cfg}/${TASK_NAME}"
}

# -----------------------------------------------------------------------------
# conda
# -----------------------------------------------------------------------------
activate_conda() {
  if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    echo "错误: 找不到 conda.sh: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  if [[ -z "${PYTHON}" ]]; then
    PYTHON="$(command -v python)"
  fi
  case "${PYTHON}" in
    *"/envs/${CONDA_ENV}/"*) ;;
    *)
      echo "警告: python 不在 conda 环境 ${CONDA_ENV} 内: ${PYTHON}" >&2
      ;;
  esac
  export PYTHONPATH="${REPO_ROOT}/src:${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}

run_inference() {
  local gpu="$1" video_dir="$2" task_config="$3" n_ep="$4" log_file="$5"
  echo_banner "${task_config}  gpu=${gpu}  episodes=${n_ep}"
  echo "video-dir=${video_dir}"
  echo "log=${log_file}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] cd %q && CUDA_VISIBLE_DEVICES=%s %s -u %s --ckpt-path %s --video-dir %s --task-config %s --task-idx %s --num-episodes %s --kpt-meta-path %s ... > %s\n' \
      "${ROBOTWIN_ROOT}" "${gpu}" "${PYTHON:-python}" "${INFERENCE_PY}" \
      "${CKPT}" "${video_dir}" "${task_config}" "${TASK_IDX}" "${n_ep}" "${KPT_META}" "${log_file}"
    return 0
  fi
  mkdir -p "$(dirname "${log_file}")"
  (
    cd "${ROBOTWIN_ROOT}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -u "${INFERENCE_PY}" \
      --ckpt-path "${CKPT}" \
      --video-dir "${video_dir}" \
      --task-config "${task_config}" \
      --task-idx "${TASK_IDX}" \
      --instruction-type "${INSTRUCTION_TYPE}" \
      --seed "${SEED}" \
      --stats-key "${STATS_KEY}" \
      --resize-size "${RESIZE_SIZE}" \
      --action-mode "${ACTION_MODE}" \
      --infer-horizon "${INFER_HORIZON}" \
      --inference-backend "${INFERENCE_BACKEND}" \
      --num-episodes "${n_ep}" \
      --fps "${FPS}" \
      --dtype "${DTYPE}" \
      --kpt-coord-mode "${KPT_COORD_MODE}" \
      --kpt-meta-path "${KPT_META}"
  ) > "${log_file}" 2>&1
}

ckpt_complete() {
  local f="${CKPT}/model.safetensors"
  [[ -f "${CKPT}/config.json" && -f "${CKPT}/stats.json" && -f "${f}" ]] || return 1
  [[ "$(find "${CKPT}" -maxdepth 1 -name '*.gstmp' | wc -l | tr -d ' ')" == "0" ]] || return 1
  local sz
  sz="$(stat -c '%s' "${f}" 2>/dev/null || echo 0)"
  [[ "${sz}" -ge "${MIN_CKPT_BYTES}" ]]
}

# -----------------------------------------------------------------------------
# stages
# -----------------------------------------------------------------------------
stage_gcs() {
  echo_banner "GCS 下载 pretrained_model"
  if [[ "${FORCE_DOWNLOAD}" -eq 0 ]] && ckpt_complete; then
    log_event "GCS 下载（本地 ckpt 已完整，跳过）" "SKIP ${CKPT}"
    return 0
  fi
  if [[ -z "${GCS_CKPT}" ]]; then
    echo "错误: 本地 ckpt 不完整且未设置 --gcs-job / --gcs-ckpt。CKPT=${CKPT}" >&2
    log_event "GCS 下载" "FAIL 无 GCS URI 且本地 ckpt 不完整"
    exit 1
  fi
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "错误: 找不到 gcloud" >&2
    log_event "GCS 下载" "FAIL 无 gcloud"
    exit 1
  fi
  run mkdir -p "${CKPT}"
  local gcs_cmd="gcloud storage cp ${GCS_CKPT}/config.json ${GCS_CKPT}/stats.json ${GCS_CKPT}/train_config.json ${GCS_CKPT}/model.safetensors ${CKPT}/"
  log_command "${gcs_cmd}" "从 GCS 拉取 step-${CKPT_STEP} 的 pretrained_model 四文件到本机，供预检与 inference 加载"
  log_event "GCS 开始下载 ${GCS_CKPT}" "..."
  run gcloud storage cp \
    "${GCS_CKPT}/config.json" \
    "${GCS_CKPT}/stats.json" \
    "${GCS_CKPT}/train_config.json" \
    "${GCS_CKPT}/model.safetensors" \
    "${CKPT}/"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if ! ckpt_complete; then
    log_event "GCS 下载" "FAIL ckpt 不完整 ${CKPT}"
    log_problem "ckpt 下载后不完整" "传输中断或漏文件" "检查 *.gstmp 后重跑 --force-download" "失败"
    exit 1
  fi
  local bytes
  bytes="$(stat -c '%s' "${CKPT}/model.safetensors")"
  log_event "GCS 下载完成" "OK model.safetensors=${bytes} bytes"
}

preflight_item() {
  local label="$1"
  shift
  echo -n "${label} "
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "(dry-run)"
    return 0
  fi
  if "$@" ; then
    echo "OK"
    return 0
  fi
  echo "FAIL"
  return 1
}

stage_preflight() {
  echo_banner "预检"
  local fail=0
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    log_event "预检" "SKIP dry-run"
    return 0
  fi

  preflight_item "[0] python interpreter:" \
    "${PYTHON}" -c "import sys; p=sys.executable; print(p); assert '/envs/${CONDA_ENV}/' in p.replace('\\\\','/'), p" \
    || fail=1

  preflight_item "[1] conda ${CONDA_ENV}:" \
    test -x "${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python" \
    || fail=1

  preflight_item "[2] torch+cuda:" \
    "${PYTHON}" -c "import torch; assert torch.cuda.is_available(); print(f'{torch.__version__} nGPU={torch.cuda.device_count()}')" \
    || fail=1

  preflight_item "[3] transformers ${TRANSFORMERS_EXPECT}:" \
    "${PYTHON}" -c "import transformers; assert transformers.__version__=='${TRANSFORMERS_EXPECT}'; print(transformers.__version__)" \
    || fail=1

  preflight_item "[4] Qwen3.5 patch:" \
    "${PYTHON}" -c "from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration; print('OK')" \
    || fail=1

  preflight_item "[5] flash-attn:" \
    "${PYTHON}" -c "import flash_attn; print(flash_attn.__version__)" \
    || fail=1

  preflight_item "[6] fla:" \
    "${PYTHON}" -c "import fla; print('OK')" \
    || fail=1

  preflight_item "[7] sapien:" \
    "${PYTHON}" -c "import sapien; print(sapien.__version__)" \
    || fail=1

  preflight_item "[8] curobo kinematics:" \
    "${PYTHON}" -c "from curobo.curobolib import kinematics; print('OK')" \
    || fail=1

  preflight_item "[9] scipy Rotation:" \
    "${PYTHON}" -c "from scipy.spatial.transform import Rotation; print('OK')" \
    || fail=1

  preflight_item "[10] RoboTwin task file:" \
    test -f "${ROBOTWIN_ROOT}/envs/${TASK_NAME}.py" \
    || fail=1

  preflight_item "[11] ckpt files:" \
    bash -c "test -f '${CKPT}/config.json' && test -f '${CKPT}/model.safetensors' && test -f '${CKPT}/stats.json'" \
    || fail=1

  preflight_item "[12] ckpt kpt + repo_id:" \
    "${PYTHON}" -c "
import json, os
from pathlib import Path
ckpt = Path(os.environ['CKPT'])
c = json.load(open(ckpt / 'config.json'))
assert c.get('enable_keypoint_predictor') is True, 'enable_keypoint_predictor'
tc = ckpt / 'train_config.json'
expect = os.environ.get('EXPECT_REPO_ID', '')
if tc.exists() and expect:
    t = json.load(open(tc))
    rid = t.get('dataset', {}).get('repo_id', '')
    assert rid == expect, f'repo_id={rid} expect={expect}'
print('OK')
" || fail=1

  preflight_item "[13] kpt meta:" \
    "${PYTHON}" -c "
import json, os
from pathlib import Path
p = Path(os.environ['KPT_META'])
assert p.is_file(), p
m = json.load(open(p))
names = m.get('keypoint_names') or []
assert len(names) >= 7 and names[6] == 'fl_eef_tcp', names
o = m['coord_offset']
expect = os.environ.get('EXPECT_OFFSET', '').strip()
if expect:
    exp = [float(x) for x in expect.split(',')]
    for a, b in zip(o, exp):
        assert abs(a - b) < 1e-3, (o, exp)
print('OK', [round(x, 4) for x in o])
" || fail=1

  preflight_item "[14] inference voxel+expert_success:" \
    "${PYTHON}" -c "
from pathlib import Path
src = Path('${INFERENCE_PY}').read_text()
assert 'def get_keypoints_kptsim_voxel' in src
assert 'expert_success' in src
assert '--kpt-meta-path' in src
idx = src.find('episode_info = task_env.play_once()')
chunk = src[idx:idx+800]
cs = ce = None
for i, line in enumerate(chunk.splitlines()):
    if line.strip().startswith('#'):
        continue
    if 'check_success' in line and cs is None:
        cs = i
    if 'close_env' in line and ce is None:
        ce = i
assert cs is not None and ce is not None and cs < ce
print('OK')
" || fail=1

  echo -n "[15] disk: "
  df -h "${REPO_ROOT}" | awk 'NR==2{print $4" free"}'
  "${PYTHON}" -c "
import os, shutil
free = shutil.disk_usage(os.environ['REPO_ROOT']).free
need = int(os.environ['MIN_DISK_GB']) * (1024**3)
assert free >= need, f'free={free} need={need}'
print('OK')
" || fail=1

  if [[ "${fail}" -ne 0 ]]; then
    log_event "预检" "FAIL 见上方 FAIL 项"
    log_problem "预检未全部通过" "依赖/ckpt/meta/inference.py 之一失败" "按手册 §12 排查后重跑" "失败"
    exit 1
  fi
  log_event "预检 15 项" "OK"
  log_append ""
  log_append "**预检说明**：项 [4][6] 打印的 \`Python 3.10 is below the recommended 3.11\` 来自 transformers/fla 导入时的提示，**非错误**；本机 RoboTwin 评测沿用 conda \`itvlaGp\`（Python 3.10.20），与 hanging_mug / stack_bowls 评测一致。"
}

check_smoke_log() {
  local log_file="$1"
  if grep -q 'AttributeError' "${log_file}"; then
    echo "冒烟日志含 AttributeError"
    return 1
  fi
  if grep -q "is_left_gripper_open" "${log_file}"; then
    echo "冒烟日志含 is_left_gripper_open（expert_success 顺序问题）"
    return 1
  fi
  if grep -q "got unexpected keyword argument 'his_kpts'" "${log_file}"; then
    echo "用了 optimized backend"
    return 1
  fi
  local voxel_line
  voxel_line="$(grep -E 'Using kptsim voxel keypoints from' "${log_file}" | tail -1 || true)"
  if [[ -z "${voxel_line}" && "${KPT_COORD_MODE}" == "voxel" ]]; then
    echo "冒烟日志没有 voxel keypoints 行"
    return 1
  fi
  if [[ -n "${voxel_line}" && "${TASK_NAME}" != "stack_bowls_three" ]]; then
    if grep -q 'stack_bowls_three' <<<"${voxel_line}"; then
      echo "voxel meta 误指向 stack_bowls_three: ${voxel_line}"
      return 1
    fi
  fi
  if [[ -n "${voxel_line}" ]] && ! grep -q "${TASK_NAME}" <<<"${voxel_line}"; then
    echo "voxel 行不含任务名 ${TASK_NAME}: ${voxel_line}"
    return 1
  fi
  return 0
}

stage_smoke() {
  echo_banner "冒烟 ${SMOKE_EPISODES} ep"
  local cfg="${CONFIG_ARR[0]}"
  local vdir log_file
  vdir="$(video_dir_for "${cfg}" smoke)"
  log_file="${LOG_DIR}/smoke_${RUN_ID}.log"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    run_inference "${SMOKE_GPU}" "${vdir}" "${cfg}" "${SMOKE_EPISODES}" "${log_file}"
    log_event "冒烟 ${SMOKE_EPISODES} ep ${cfg}" "SKIP dry-run"
    return 0
  fi
  local ec=0
  run_inference "${SMOKE_GPU}" "${vdir}" "${cfg}" "${SMOKE_EPISODES}" "${log_file}" || ec=$?
  local s f t msg
  s="$(count_videos "${vdir}" success)"
  f="$(count_videos "${vdir}" failure)"
  t=$((s + f))
  msg="exit=${ec} ${s}S/${f}F/${t}mp4 log=${log_file}"
  if [[ "${ec}" -ne 0 ]]; then
    log_event "冒烟 ${SMOKE_EPISODES} ep ${cfg}" "FAIL ${msg}"
    log_problem "冒烟非零退出" "见 ${log_file}" "按手册 §12" "失败"
    [[ "${KEEP_GOING}" -eq 1 ]] || exit "${ec}"
    return 0
  fi
  local reason=""
  reason="$(check_smoke_log "${log_file}" || true)"
  if [[ -n "${reason}" ]]; then
    log_event "冒烟 ${SMOKE_EPISODES} ep ${cfg}" "FAIL ${reason}"
    log_problem "冒烟日志校验失败" "${reason}" "检查 --kpt-meta-path / backend / expert_success" "失败"
    [[ "${KEEP_GOING}" -eq 1 ]] || exit 1
    return 0
  fi
  if [[ "${t}" -lt "${SMOKE_EPISODES}" ]]; then
    log_event "冒烟 ${SMOKE_EPISODES} ep ${cfg}" "FAIL 仅 ${t} 个 mp4（期望 >= ${SMOKE_EPISODES}）"
    [[ "${KEEP_GOING}" -eq 1 ]] || exit 1
    return 0
  fi
  log_event "冒烟 ${SMOKE_EPISODES} ep ${cfg}" "OK ${msg}"
}

stage_eval() {
  echo_banner "正式评估 ${NUM_EPISODES} ep × ${#CONFIG_ARR[@]} 配置"
  local -a pids=() cfgs_running=() logs_running=()
  local i cfg gpu vdir log_file

  launch_one() {
    local cfg="$1" gpu="$2"
    local vdir log_file
    vdir="$(video_dir_for "${cfg}" robotwin)"
    log_file="${LOG_DIR}/eval_${RUN_ID}_${cfg}.log"
    log_event "启动 ${cfg} GPU${gpu} ${NUM_EPISODES} ep" "..."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      run_inference "${gpu}" "${vdir}" "${cfg}" "${NUM_EPISODES}" "${log_file}"
      return 0
    fi
    run_inference "${gpu}" "${vdir}" "${cfg}" "${NUM_EPISODES}" "${log_file}" &
    pids+=("$!")
    CHILD_PIDS+=("$!")
    cfgs_running+=("${cfg}")
    logs_running+=("${log_file}")
  }

  local use_parallel=0
  if [[ "${SEQUENTIAL}" -eq 0 && ${#CONFIG_ARR[@]} -gt 1 && ${#GPU_ARR[@]} -ge ${#CONFIG_ARR[@]} ]]; then
    use_parallel=1
  fi

  if [[ "${use_parallel}" -eq 1 ]]; then
    echo "并行: ${#CONFIG_ARR[@]} 配置 / ${#GPU_ARR[@]} GPU"
    for i in "${!CONFIG_ARR[@]}"; do
      launch_one "${CONFIG_ARR[$i]}" "${GPU_ARR[$i]}"
    done
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      return 0
    fi
    local fail=0
    for i in "${!pids[@]}"; do
      local ec=0
      wait "${pids[$i]}" || ec=$?
      cfg="${cfgs_running[$i]}"
      vdir="$(video_dir_for "${cfg}" robotwin)"
      s="$(count_videos "${vdir}" success)"
      f="$(count_videos "${vdir}" failure)"
      t=$((s + f))
      if [[ "${ec}" -ne 0 ]]; then
        log_event "${cfg} ${NUM_EPISODES} ep" "FAIL exit=${ec} ${s}S/${f}F log=${logs_running[$i]}"
        fail=1
      else
        local rate="n/a"
        if [[ "${t}" -gt 0 ]]; then
          rate="$(${PYTHON} -c "print(f'{$s*100/$t:.1f}%')")"
        fi
        log_event "${cfg} ${NUM_EPISODES} ep" "OK ${s}/${t} = ${rate}"
      fi
    done
    [[ "${fail}" -eq 0 ]] || { [[ "${KEEP_GOING}" -eq 1 ]] || exit 1; }
  else
    echo "串行评测"
    for i in "${!CONFIG_ARR[@]}"; do
      cfg="${CONFIG_ARR[$i]}"
      gpu="${GPU_ARR[$(( i < ${#GPU_ARR[@]} ? i : ${#GPU_ARR[@]}-1 ))]}"
      vdir="$(video_dir_for "${cfg}" robotwin)"
      log_file="${LOG_DIR}/eval_${RUN_ID}_${cfg}.log"
      log_event "启动 ${cfg} GPU${gpu} ${NUM_EPISODES} ep" "..."
      local ec=0
      run_inference "${gpu}" "${vdir}" "${cfg}" "${NUM_EPISODES}" "${log_file}" || ec=$?
      s="$(count_videos "${vdir}" success)"
      f="$(count_videos "${vdir}" failure)"
      t=$((s + f))
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        continue
      fi
      if [[ "${ec}" -ne 0 ]]; then
        log_event "${cfg} ${NUM_EPISODES} ep" "FAIL exit=${ec} ${s}S/${f}F log=${log_file}"
        [[ "${KEEP_GOING}" -eq 1 ]] || exit "${ec}"
      else
        local rate="n/a"
        if [[ "${t}" -gt 0 ]]; then
          rate="$(${PYTHON} -c "print(f'{$s*100/$t:.1f}%')")"
        fi
        log_event "${cfg} ${NUM_EPISODES} ep" "OK ${s}/${t} = ${rate}"
      fi
    done
  fi
}

stage_summarize() {
  echo_banner "汇总"
  local cfg s f t rate line
  log_append ""
  log_append "## 最终结果 ($(now))"
  log_append ""
  log_append "| 配置 | 成功 | 失败 | 总计 | Success Rate |"
  log_append "|------|------|------|------|--------------|"
  echo "配置  成功/失败/总计  成功率"
  for cfg in "${CONFIG_ARR[@]}"; do
    local vdir
    vdir="$(video_dir_for "${cfg}" robotwin)"
    s="$(count_videos "${vdir}" success)"
    f="$(count_videos "${vdir}" failure)"
    t=$((s + f))
    if [[ "${t}" -gt 0 ]]; then
      if [[ -n "${PYTHON:-}" && "${DRY_RUN}" -eq 0 ]]; then
        rate="$(${PYTHON} -c "print(f'{$s*100/$t:.1f}%')")"
      else
        rate="$(python3 -c "print(f'{$s*100/$t:.1f}%')" 2>/dev/null || echo "?")"
      fi
    else
      rate="n/a"
    fi
    line="| **${cfg}** | ${s} | ${f} | ${t} | **${rate}** |"
    log_append "${line}"
    echo "${cfg}: ${t}/${NUM_EPISODES}  ${s}S/${f}F  ${rate}"
  done
  log_append ""
  log_append "**输出路径**:"
  log_append ""
  for cfg in "${CONFIG_ARR[@]}"; do
    log_append "- \`$(video_dir_for "${cfg}" robotwin)/\`"
    log_append "- \`${LOG_DIR}/eval_${RUN_ID}_${cfg}.log\`"
  done
  log_append ""
  if [[ "${DRY_RUN}" -eq 0 && -f "${STATS_PY}" && -n "${PYTHON:-}" ]]; then
    (cd "${REPO_ROOT}" && "${PYTHON}" "${STATS_PY}" "${OUT}") || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader || true
  fi
  df -h "${REPO_ROOT}" | tail -1 || true
  log_event "汇总写入 ${EVAL_LOG}" "OK"
}

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
CHILD_PIDS=()
on_signal() {
  echo "收到信号，结束子进程..." >&2
  local pid
  for pid in "${CHILD_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  log_event "脚本被信号中断" "FAIL" || true
  exit 130
}
trap on_signal INT TERM

export CKPT KPT_META EXPECT_REPO_ID EXPECT_OFFSET REPO_ROOT MIN_DISK_GB TASK_NAME RUN_LOG

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "$(dirname "${RUN_LOG}")"
  exec > >(tee -a "${RUN_LOG}") 2>&1
  echo "[runner] 控制台完整日志: ${RUN_LOG}"
fi

echo_banner "itvlaGp RoboTwin eval  ${TASK_NAME}  idx=${TASK_IDX}"
print_config
echo "建议在 tmux 中运行正式 100 ep；inference.py 启动会 rmtree video-dir。"
echo

init_eval_log
log_event "解析配置 ${TASK_NAME} idx=${TASK_IDX} run=${RUN_ID}" "OK"
log_event "控制台完整日志" "${RUN_LOG}"
if [[ -n "${EVAL_INVOCATION:-}" ]]; then
  log_command "${EVAL_INVOCATION}" "用户/Agent 启动本次评测的完整 shell 命令"
fi
log_file_change "${REPO_ROOT}/b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_eval.sh" "修改（评测前）" "scan_object 默认 GCS/offset；LOG 增手册/路径/问题/文件/命令节"

need_conda=0
if should_run preflight || should_run smoke || should_run eval; then
  need_conda=1
fi
if should_run summarize && [[ -f "${STATS_PY}" ]]; then
  need_conda=1
fi

if [[ "${need_conda}" -eq 1 && "${DRY_RUN}" -eq 0 ]]; then
  activate_conda
  log_event "conda activate ${CONDA_ENV}" "OK ${PYTHON}"
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  PYTHON="${PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
  echo "[dry-run] 将 conda activate ${CONDA_ENV} → ${PYTHON}"
fi

if should_run gcs; then
  stage_gcs
fi
if should_run preflight; then
  stage_preflight
fi
if should_run smoke; then
  stage_smoke
fi
if should_run eval; then
  stage_eval
fi
if should_run summarize; then
  stage_summarize
fi

echo_banner "完成"
echo "EVAL_LOG=${EVAL_LOG}"
echo "OUT=${OUT}"
