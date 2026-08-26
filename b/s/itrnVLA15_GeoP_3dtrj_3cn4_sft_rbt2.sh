#!/usr/bin/env bash
# =============================================================================
# itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.sh
#
# 远端 8×H200 VM 编排：源码已 clone 后，自动执行 Phase 2 SFT 落地手册中
# 除评测以外的全部步骤。
#
# 手册: b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg.md
# 训练: launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh
#
# 用法（在已 clone 的仓库根或任意 cwd）:
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.sh
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.sh --until preflight
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.sh --skip-train
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.sh --dry-run
#
# 不含 §13 RoboTwin 评测。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_DEFAULT_PROJ="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# -----------------------------------------------------------------------------
# 默认（环境变量可先设；CLI 优先）
# -----------------------------------------------------------------------------
PROJ_ROOT="${PROJ_ROOT:-${_DEFAULT_PROJ}}"
VENV_ROOT="${VENV_ROOT:-/tmp/itnvla15rbt20}"
RUNPKG_ROOT="${RUNPKG_ROOT:-/tmp/RunPkg}"

GCS_PKG="${GCS_PKG:-gs://physical-ai-data-eu/VENV/tmp/RP/RunPkg_hngMg0825.tar.zst}"
GCS_VENV="${GCS_VENV:-gs://physical-ai-data-eu/VENV/tmp/itnvla15rbt20_0811.tar}"
GCS_PROBE="${GCS_PROBE:-gs://physical-ai-data-eu/VENV/tmp/}"
GCP_PROJECT="${GCP_PROJECT:-}"
GCLOUD_SDK_DIR="${GCLOUD_SDK_DIR:-${HOME}/google-cloud-sdk}"

DATA_REPO_ID="${DATA_REPO_ID:-hanging_mug_kptsim_lrbv30}"
DATA_DST_REL="${DATA_DST_REL:-Dta/hanging_mug_kptsim_lrbv30}"
CKPT_DST_REL="${CKPT_DST_REL:-Ckp/warmup_hanging_mug_kptsim_400step/checkpoints/000400}"
WARMUP_CKPT="${WARMUP_CKPT:-}"
WAN_DIR="${WAN_DIR:-}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-}"
HF_HOME="${HF_HOME:-}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-}"

EXPECT_GPUS="${EXPECT_GPUS:-8}"
GPUS="${GPUS:-}"
CUDA_VISIBLE_DEVICES_TRAIN="${CUDA_VISIBLE_DEVICES_TRAIN:-${CUDA_VISIBLE_DEVICES:-}}"
EXPECT_GPUS_EXPLICIT=0
GPUS_EXPLICIT=0
CUDA_DEVICES_EXPLICIT=0
BATCH_SIZE="${BATCH_SIZE:-}"
STEPS="${STEPS:-}"

FROM_STAGE="${FROM_STAGE:-gcloud}"
UNTIL_STAGE="${UNTIL_STAGE:-train}"

DRY_RUN=0
FORCE=0
SKIP_WAN_SMOKE=0
SKIP_SMOKE=0
SKIP_TRAIN=0

STAGES=(gcloud runpkg venv install symlink wan data-check preflight wan-smoke smoke train)

usage() {
  cat <<'EOF'
用法: itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2.sh [选项]

在源码已 clone 的 VM 上编排 Phase 2 SFT（不含评测）。
默认跑完全部阶段，含 8 卡 10k。

路径:
  --proj-root PATH          仓库根（默认: 本脚本上两级）
  --venv-root PATH          venv（默认 /tmp/itnvla15rbt20）
  --runpkg-root PATH        RunPkg 解压根（默认 /tmp/RunPkg）
  --gcs-pkg URI             RunPkg tar.zst GCS 路径
  --gcs-venv URI            venv tar GCS 路径
  --gcs-probe URI           gcloud 登录探测前缀
  --gcp-project ID          可选 gcloud config set project
  --data-repo-id ID         LeRobot repo_id
  --data-dst-rel REL        包内数据相对路径
  --ckpt-dst-rel REL        包内 Warmup ckpt 相对路径
  --warmup-ckpt PATH        pretrained_model 目录
  --wan-dir PATH            Wan2.2-TI2V-5B 目录
  --launch-script PATH      Phase 2 launch 脚本
  --hf-home PATH            HF_HOME
  --hf-lerobot-home PATH    HF_LEROBOT_HOME
  --log-dir PATH            日志目录（默认 /tmp/<DATA_REPO_ID>；正式 10k checkpoint 也写在这里）

训练:
  --gpus N                  正式 10k 使用 N 卡（默认 8）；自动设 EXPECT_GPUS、PROC_PER_NODE
  --cuda-visible-devices L  正式 10k 物理 GPU 列表，如 0,1,2,3,4,5（未设时 --gpus N → 0..N-1）
  --expect-gpus N           Preflight 期望可见 GPU 数（默认 8；--gpus 会覆盖为同值）
  --batch-size N            仅正式 10k 传给 launch
  --steps N                 仅正式 10k 传给 launch

阶段:
  --from STAGE              从该阶段开始
  --until STAGE             做到该阶段为止
  --skip-wan-smoke          跳过 WAN Smoke
  --skip-smoke              跳过 Smoke 100
  --skip-train              跳过 8 卡 10k
  --force                   已存在的 RunPkg/venv 也重新下载解压
  --dry-run                 只打印命令
  -h, --help

阶段 id（顺序）:
  gcloud runpkg venv install symlink wan data-check preflight wan-smoke smoke train

环境变量与 CLI 同名（大写下划线），CLI 优先。
EOF
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

count_cuda_devices() {
  local list="$1"
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

finalize_gpu_config() {
  local device_count=0

  if [[ "${CUDA_DEVICES_EXPLICIT}" -eq 1 ]]; then
    device_count="$(count_cuda_devices "${CUDA_VISIBLE_DEVICES_TRAIN}")"
    if [[ "${device_count}" -lt 1 ]]; then
      echo "错误: --cuda-visible-devices 无效: ${CUDA_VISIBLE_DEVICES_TRAIN}" >&2
      exit 1
    fi
    if [[ "${GPUS_EXPLICIT}" -eq 1 && "${GPUS}" != "${device_count}" ]]; then
      echo "错误: --gpus ${GPUS} 与 --cuda-visible-devices 数量 ${device_count} 不一致" >&2
      exit 1
    fi
    GPUS="${device_count}"
  elif [[ "${GPUS_EXPLICIT}" -eq 1 ]]; then
    if [[ "${GPUS}" -lt 1 ]]; then
      echo "错误: --gpus 必须 >= 1，当前: ${GPUS}" >&2
      exit 1
    fi
    CUDA_VISIBLE_DEVICES_TRAIN="$(build_cuda_devices "${GPUS}")"
    device_count="${GPUS}"
  else
    device_count="${EXPECT_GPUS}"
  fi

  if [[ "${GPUS_EXPLICIT}" -eq 1 || "${CUDA_DEVICES_EXPLICIT}" -eq 1 ]]; then
    if [[ "${EXPECT_GPUS_EXPLICIT}" -eq 1 && "${EXPECT_GPUS}" != "${GPUS}" ]]; then
      echo "警告: --expect-gpus ${EXPECT_GPUS} 与训练卡数 ${GPUS} 不一致，以训练卡数为准" >&2
    fi
    EXPECT_GPUS="${GPUS}"
    TRAIN_PROC_PER_NODE="${GPUS}"
    TRAIN_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_TRAIN}"
  else
    TRAIN_PROC_PER_NODE=""
    TRAIN_CUDA_VISIBLE_DEVICES=""
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proj-root)          PROJ_ROOT="$2"; shift 2 ;;
    --venv-root)          VENV_ROOT="$2"; shift 2 ;;
    --runpkg-root)        RUNPKG_ROOT="$2"; shift 2 ;;
    --gcs-pkg)            GCS_PKG="$2"; shift 2 ;;
    --gcs-venv)           GCS_VENV="$2"; shift 2 ;;
    --gcs-probe)          GCS_PROBE="$2"; shift 2 ;;
    --gcp-project)        GCP_PROJECT="$2"; shift 2 ;;
    --gcloud-sdk-dir)     GCLOUD_SDK_DIR="$2"; shift 2 ;;
    --data-repo-id)       DATA_REPO_ID="$2"; shift 2 ;;
    --data-dst-rel)       DATA_DST_REL="$2"; shift 2 ;;
    --ckpt-dst-rel)       CKPT_DST_REL="$2"; shift 2 ;;
    --warmup-ckpt)        WARMUP_CKPT="$2"; shift 2 ;;
    --wan-dir)            WAN_DIR="$2"; shift 2 ;;
    --launch-script)      LAUNCH_SCRIPT="$2"; shift 2 ;;
    --hf-home)            HF_HOME="$2"; shift 2 ;;
    --hf-lerobot-home)    HF_LEROBOT_HOME="$2"; shift 2 ;;
    --log-dir)            LOG_DIR="$2"; shift 2 ;;
    --gpus)               GPUS="$2"; GPUS_EXPLICIT=1; shift 2 ;;
    --cuda-visible-devices)
      CUDA_VISIBLE_DEVICES_TRAIN="$2"
      CUDA_DEVICES_EXPLICIT=1
      shift 2
      ;;
    --expect-gpus)        EXPECT_GPUS="$2"; EXPECT_GPUS_EXPLICIT=1; shift 2 ;;
    --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
    --steps)              STEPS="$2"; shift 2 ;;
    --from)               FROM_STAGE="$2"; shift 2 ;;
    --until)              UNTIL_STAGE="$2"; shift 2 ;;
    --skip-wan-smoke)     SKIP_WAN_SMOKE=1; shift ;;
    --skip-smoke)         SKIP_SMOKE=1; shift ;;
    --skip-train)         SKIP_TRAIN=1; shift ;;
    --force)              FORCE=1; shift ;;
    --dry-run)            DRY_RUN=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# 须在 CLI 解析 DATA_REPO_ID / --log-dir 之后，默认日志目录才随任务名变化。
LOG_DIR="${LOG_DIR:-/tmp/${DATA_REPO_ID}}"

finalize_gpu_config

DATA_DST_REL="${DATA_DST_REL#/}"
CKPT_DST_REL="${CKPT_DST_REL#/}"
PROJ_ROOT="${PROJ_ROOT%/}"
VENV_ROOT="${VENV_ROOT%/}"
RUNPKG_ROOT="${RUNPKG_ROOT%/}"

HF_HOME="${HF_HOME:-${VENV_ROOT}/var/hf_home}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${VENV_ROOT}/var/datasets}"
WAN_DIR="${WAN_DIR:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
WARMUP_CKPT="${WARMUP_CKPT:-${RUNPKG_ROOT}/${CKPT_DST_REL}/pretrained_model}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${PROJ_ROOT}/launch/internvla_a15_geop_phase2_finetune_kptsim_8g.sh}"
DATA_ENTITY="${RUNPKG_ROOT}/${DATA_DST_REL}"
DATA_LINK="${HF_LEROBOT_HOME}/${DATA_REPO_ID}"
PYTHON="${VENV_ROOT}/bin/python"

# Preflight / pip / launch 都读这些路径；提前 export，避免 python -c KeyError。
export PROJ_ROOT VENV_ROOT RUNPKG_ROOT
export HF_HOME HF_LEROBOT_HOME WAN_DIR WARMUP_CKPT DATA_REPO_ID

WAN_SMOKE_LOG="${LOG_DIR%/}/wan_smoke.log"
SMOKE_LOG="${LOG_DIR%/}/smoke100.log"
TRAIN_LOG="${LOG_DIR%/}/8g_10k.log"
TRAIN_JOB_NAME="${JOB_NAME:-}"
TRAIN_OUTPUT_DIR="${OUTPUT_DIR:-}"

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
    wan-smoke) [[ "${SKIP_WAN_SMOKE}" -eq 0 ]] || return 1 ;;
    smoke)     [[ "${SKIP_SMOKE}" -eq 0 ]] || return 1 ;;
    train)     [[ "${SKIP_TRAIN}" -eq 0 ]] || return 1 ;;
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

require_file() {
  local path="$1" label="${2:-file}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] 将检查 ${label}: ${path}"
    return 0
  fi
  if [[ ! -f "${path}" ]]; then
    echo "错误: 缺少 ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1" label="${2:-dir}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] 将检查 ${label}: ${path}"
    return 0
  fi
  if [[ ! -d "${path}" ]]; then
    echo "错误: 缺少 ${label}: ${path}" >&2
    exit 1
  fi
}

have_gcloud() {
  command -v gcloud >/dev/null 2>&1
}

refresh_gcloud_path() {
  if [[ -f "${GCLOUD_SDK_DIR}/path.bash.inc" ]]; then
    # shellcheck disable=SC1090
    source "${GCLOUD_SDK_DIR}/path.bash.inc"
  fi
  hash -r 2>/dev/null || true
}

install_gcloud_apt() {
  echo "[install] Debian/Ubuntu apt → google-cloud-cli"
  run sudo apt-get update -y
  run sudo apt-get install -y apt-transport-https ca-certificates gnupg curl
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  sudo mkdir -p /usr/share/keyrings
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
  echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
    | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
}

install_gcloud_tarball() {
  echo "[install] official tarball → ${GCLOUD_SDK_DIR}"
  local arch url tmp
  case "$(uname -m)" in
    x86_64|amd64) arch="linux-x86_64" ;;
    aarch64|arm64) arch="linux-arm" ;;
    *)
      echo "错误: 不支持的架构 $(uname -m)" >&2
      exit 1
      ;;
  esac
  url="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-${arch}.tar.gz"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] curl ${url} && install.sh → ${GCLOUD_SDK_DIR}"
    return
  fi
  tmp="$(mktemp -d)"
  curl -fsSL "${url}" -o "${tmp}/gcloud.tgz"
  mkdir -p "$(dirname "${GCLOUD_SDK_DIR}")"
  rm -rf "${GCLOUD_SDK_DIR}"
  tar -xzf "${tmp}/gcloud.tgz" -C "$(dirname "${GCLOUD_SDK_DIR}")"
  rm -rf "${tmp}"
  "${GCLOUD_SDK_DIR}/install.sh" --quiet --usage-reporting false --path-update true \
    --command-completion true --rc-path "${HOME}/.bashrc"
  refresh_gcloud_path
}

ensure_gcloud_installed() {
  if have_gcloud; then
    echo "[skip] gcloud 已在 PATH: $(command -v gcloud)"
    return
  fi
  if [[ -x "${GCLOUD_SDK_DIR}/bin/gcloud" ]]; then
    refresh_gcloud_path
    if have_gcloud; then
      echo "[skip] 已从 ${GCLOUD_SDK_DIR} 加载 gcloud"
      return
    fi
  fi
  if command -v apt-get >/dev/null 2>&1; then
    install_gcloud_apt
  else
    install_gcloud_tarball
  fi
  refresh_gcloud_path
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  if ! have_gcloud; then
    echo "错误: 安装后仍找不到 gcloud。请 source ~/.bashrc 或把 ${GCLOUD_SDK_DIR}/bin 加入 PATH" >&2
    exit 1
  fi
  echo "[ok] gcloud=$(command -v gcloud)"
}

can_read_gcs() {
  gcloud storage ls "${GCS_PROBE}" >/dev/null 2>&1
}

ensure_gcloud_login() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] 将探测 GCS ${GCS_PROBE}，失败则 gcloud auth login --no-launch-browser"
    return
  fi
  if can_read_gcs; then
    echo "[skip] 已能读取 ${GCS_PROBE}"
    gcloud auth list 2>/dev/null || true
    return
  fi
  echo "[login] 无法读取 ${GCS_PROBE}，开始交互登录（SSH 下用浏览器授权码）"
  gcloud auth login --no-launch-browser
  if [[ -n "${GCP_PROJECT}" ]]; then
    gcloud config set project "${GCP_PROJECT}"
  fi
  if ! can_read_gcs; then
    echo "错误: 登录后仍无法 ls ${GCS_PROBE}" >&2
    exit 1
  fi
  echo "[ok] GCS 可读: ${GCS_PROBE}"
}

ensure_zstd() {
  if command -v zstd >/dev/null 2>&1; then
    return
  fi
  echo "[install] zstd"
  if command -v apt-get >/dev/null 2>&1; then
    run sudo apt-get update -y
    run sudo apt-get install -y zstd
  elif command -v yum >/dev/null 2>&1; then
    run sudo yum install -y zstd
  else
    echo "错误: 需要 zstd 才能解压 .tar.zst" >&2
    exit 1
  fi
}

gcs_cp() {
  local src="$1" dst="$2"
  run gcloud storage cp "${src}" "${dst}"
}

extract_tar_zstd() {
  local archive="$1" dest="$2"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] tar --zstd -xf ${archive} -C ${dest}"
    return
  fi
  mkdir -p "${dest}"
  if tar --help 2>&1 | grep -q -- '--zstd'; then
    tar --zstd -xf "${archive}" -C "${dest}"
  else
    zstd -dc "${archive}" | tar -xf - -C "${dest}"
  fi
}

runpkg_ok() {
  [[ -f "${DATA_ENTITY}/meta/info.json" ]] \
    && [[ -f "${WARMUP_CKPT}/model.safetensors" ]]
}

venv_ok() {
  [[ -x "${VENV_ROOT}/bin/python" ]] && [[ -f "${VENV_ROOT}/pyvenv.cfg" ]]
}

export_train_env() {
  export VENV_ROOT PROJ_ROOT DATA_REPO_ID WARMUP_CKPT
  export HF_HOME HF_LEROBOT_HOME
  export WANDB_MODE="${WANDB_MODE:-offline}"
}

run_launch() {
  local log_file="$1"
  shift
  export_train_env
  export LOG_FILE="${log_file}"
  unset WAN_SMOKE SMOKE || true
  local extra_env=()
  while [[ $# -gt 0 ]]; do
    extra_env+=("$1")
    shift
  done
  mkdir -p "$(dirname "${log_file}")" "${LOG_DIR}"
  echo "[launch] ${extra_env[*]} LOG_FILE=${log_file} bash ${LAUNCH_SCRIPT}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] env'
    printf ' %q' VENV_ROOT="${VENV_ROOT}" PROJ_ROOT="${PROJ_ROOT}" \
      DATA_REPO_ID="${DATA_REPO_ID}" WARMUP_CKPT="${WARMUP_CKPT}" \
      HF_HOME="${HF_HOME}" HF_LEROBOT_HOME="${HF_LEROBOT_HOME}" \
      LOG_FILE="${log_file}" "${extra_env[@]}" \
      bash "${LAUNCH_SCRIPT}"
    printf '\n'
    return 0
  fi
  (
    cd "${PROJ_ROOT}"
    env "${extra_env[@]}" bash "${LAUNCH_SCRIPT}"
  )
}

check_launch_log() {
  local log_file="$1"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  if [[ ! -f "${log_file}" ]]; then
    echo "错误: 日志不存在 ${log_file}" >&2
    exit 1
  fi
  # launch 头信息是 DATA_REPO_ID=...；accelerate 配置 dump 是 'repo_id': '...'。
  # 旧版 launch 只 tee 了 accelerate 输出，头信息/post_check 不会进 LOG_FILE。
  if ! grep -q "DATA_REPO_ID=${DATA_REPO_ID}" "${log_file}" \
    && ! grep -qE "repo_id[=:'\" ]+${DATA_REPO_ID}" "${log_file}"; then
    echo "错误: 日志未包含 DATA_REPO_ID=${DATA_REPO_ID}（也无 repo_id=${DATA_REPO_ID}）" >&2
    echo "      日志: ${log_file}" >&2
    exit 1
  fi
  local line
  line="$(grep 'post_check:' "${log_file}" | tail -1 || true)"
  if [[ -z "${line}" ]]; then
    echo "错误: 日志无 post_check 行: ${log_file}" >&2
    echo "      旧版 launch 把 post_check 只打到终端；请用已修复的 launch 重跑该阶段。" >&2
    exit 1
  fi
  echo "  ${line}"
  if [[ "${line}" != *"video_decode_error=0"* || "${line}" != *"using_zeros=0"* || "${line}" != *"exit=0"* ]]; then
    echo "错误: post_check 未通过: ${line}" >&2
    exit 1
  fi
}

stage_gcloud() {
  echo_banner "gcloud (§1.0)"
  ensure_gcloud_installed
  ensure_gcloud_login
}

stage_runpkg() {
  echo_banner "RunPkg (§1.2)"
  if [[ "${FORCE}" -eq 0 ]] && runpkg_ok; then
    echo "[skip] RunPkg 已就绪: ${DATA_ENTITY} / ${WARMUP_CKPT}"
    return
  fi
  ensure_zstd
  local local_tar="/tmp/$(basename "${GCS_PKG}")"
  gcs_cp "${GCS_PKG}" "${local_tar}"
  local extract_parent
  extract_parent="$(dirname "${RUNPKG_ROOT}")"
  extract_tar_zstd "${local_tar}" "${extract_parent}"
  run rm -f "${local_tar}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  if ! runpkg_ok; then
    echo "错误: 解压后 RunPkg 校验失败" >&2
    echo "  data: ${DATA_ENTITY}/meta/info.json" >&2
    echo "  ckpt: ${WARMUP_CKPT}/model.safetensors" >&2
    exit 1
  fi
  echo "[ok] DATA + WARMUP_CKPT"
}

stage_venv() {
  echo_banner "venv (§1.3)"
  if [[ "${FORCE}" -eq 0 ]] && venv_ok; then
    echo "[skip] venv 已就绪: ${VENV_ROOT}"
    return
  fi
  local local_tar="/tmp/$(basename "${GCS_VENV}")"
  gcs_cp "${GCS_VENV}" "${local_tar}"
  local extract_parent
  extract_parent="$(dirname "${VENV_ROOT}")"
  run mkdir -p "${extract_parent}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] tar -xf ${local_tar} -C ${extract_parent}"
    echo "[dry-run] chmod +x ${VENV_ROOT}/bin/*"
  else
    mkdir -p "${extract_parent}"
    tar -xf "${local_tar}" -C "${extract_parent}"
    chmod +x "${VENV_ROOT}/bin/"* 2>/dev/null || true
  fi
  run rm -f "${local_tar}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return
  fi
  if ! venv_ok; then
    echo "错误: venv 解压后缺少 ${VENV_ROOT}/bin/python 或 pyvenv.cfg" >&2
    exit 1
  fi
  echo "[ok] ${VENV_ROOT}"
}

stage_install() {
  echo_banner "pip install -e + transformers patch (§1.4)"
  require_dir "${PROJ_ROOT}" "PROJ_ROOT"
  if [[ "${DRY_RUN}" -eq 0 && ! -f "${PROJ_ROOT}/pyproject.toml" && ! -f "${PROJ_ROOT}/setup.py" ]]; then
    echo "错误: ${PROJ_ROOT} 不是可安装的 Python 包（无 pyproject.toml / setup.py）" >&2
    exit 1
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${PYTHON} -m pip install -e ${PROJ_ROOT}"
    echo "[dry-run] cp transformers_replace/models → site-packages/transformers/"
    return
  fi
  if [[ ! -x "${PYTHON}" ]]; then
    echo "错误: venv python 不可执行: ${PYTHON}" >&2
    exit 1
  fi
  "${PYTHON}" -m pip install -e "${PROJ_ROOT}"
  chmod +x "${LAUNCH_SCRIPT}" 2>/dev/null || true

  local tf_dir
  tf_dir="$("${PYTHON}" -c "import transformers, os; print(os.path.dirname(transformers.__file__))")"
  mkdir -p "${tf_dir}/models"
  local pol src
  for pol in pi0 pi05 internvla_a1_5; do
    src="${PROJ_ROOT}/src/lerobot/policies/${pol}/transformers_replace/models"
    if [[ -d "${src}" ]]; then
      echo "[patch] ${src} → ${tf_dir}/"
      cp -r "${src}" "${tf_dir}/"
    fi
  done
  "${PYTHON}" -c "import lerobot, inspect; p=inspect.getfile(lerobot); print(p)"
}

stage_symlink() {
  echo_banner "dataset symlink (§1.5)"
  run mkdir -p "${HF_LEROBOT_HOME}"
  run ln -sfn "${DATA_ENTITY}" "${DATA_LINK}"
  require_file "${DATA_LINK}/meta/info.json" "info.json"
  require_file "${DATA_LINK}/norm_stat.json" "norm_stat.json"
  echo "[ok] ${DATA_LINK} -> ${DATA_ENTITY}"
}

stage_wan() {
  echo_banner "WAN weights (§6)"
  export HF_HOME WAN_DIR
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] test ${WAN_DIR}/Wan2.2_VAE.pth 或 snapshot_download"
    return
  fi
  if [[ -f "${WAN_DIR}/Wan2.2_VAE.pth" ]]; then
    echo "[skip] WAN 已存在: ${WAN_DIR}"
    du -sh "${WAN_DIR}" || true
    return
  fi
  echo "[download] Wan-AI/Wan2.2-TI2V-5B → ${WAN_DIR}"
  mkdir -p "${WAN_DIR}"
  "${PYTHON}" <<PY
import os
from huggingface_hub import snapshot_download

wan_dir = os.environ["WAN_DIR"]
snapshot_download("Wan-AI/Wan2.2-TI2V-5B", local_dir=wan_dir)
print("WAN downloaded to:", wan_dir)
PY
  require_file "${WAN_DIR}/Wan2.2_VAE.pth" "Wan2.2_VAE.pth"
}

stage_data_check() {
  echo_banner "data Layer 1 (§7.2)"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] parquet + keypoints_meta 检查 ${DATA_LINK}"
    return
  fi
  DATA="${DATA_LINK}" "${PYTHON}" <<'PY'
import json, os
from pathlib import Path
data = Path(os.environ["DATA"])
info = json.loads((data / "meta" / "info.json").read_text())
print("episodes:", info.get("total_episodes"), "frames:", info.get("total_frames"))
import pyarrow.parquet as pq
parquet_dir = data / "data" / "chunk-000"
cands = sorted(parquet_dir.glob("*.parquet"))
if not cands:
    raise SystemExit(f"no parquet under {parquet_dir}")
pf = pq.read_table(str(cands[0]), columns=["observation.keypoint_3d"])
kpt = pf["observation.keypoint_3d"][0].as_py()
print("keypoint_3d len:", len(kpt), "sample:", kpt[:3])
meta = json.loads((data / "meta" / "keypoints_meta.json").read_text())
print("coord_offset:", meta.get("coord_offset"))
if len(kpt) != 42:
    raise SystemExit(f"expected keypoint_3d len 42, got {len(kpt)}")
PY
}

stage_preflight() {
  echo_banner "Preflight (§8)"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] Preflight: torch/lerobot/WAN/data/ckpt/launch/GPU=${EXPECT_GPUS}"
    return
  fi
  export PROJ_ROOT HF_HOME
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/torch/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:\
${VENV_ROOT}/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"

  "${PYTHON}" -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.device_count())"
  "${PYTHON}" -c "
import os, inspect, lerobot
p=inspect.getfile(lerobot)
print(p)
root=os.environ['PROJ_ROOT']
assert root.rstrip('/') in p, (root, p)
"
  require_file "${WAN_DIR}/Wan2.2_VAE.pth" "WAN"
  require_file "${DATA_LINK}/meta/info.json" "DATA"
  require_file "${DATA_LINK}/norm_stat.json" "NORM"
  require_file "${WARMUP_CKPT}/model.safetensors" "WARMUP_CKPT"
  CKPT="${WARMUP_CKPT}" "${PYTHON}" -c "
import json, os
c=json.load(open(os.environ['CKPT']+'/config.json'))
assert c.get('enable_keypoint_predictor')==True, c
print('enable_keypoint_predictor OK')
"
  if pgrep -af lerobot_train | grep -v "pgrep" >/dev/null 2>&1; then
    echo "警告: 已有 lerobot_train 进程" >&2
    pgrep -af lerobot_train || true
  else
    echo "no train procs (OK)"
  fi
  test -x "${LAUNCH_SCRIPT}" || chmod +x "${LAUNCH_SCRIPT}"
  test -x "${LAUNCH_SCRIPT}" && echo "LAUNCH OK"
  EXPECT_GPUS="${EXPECT_GPUS}" "${PYTHON}" -c "
import os, torch
n=int(os.environ['EXPECT_GPUS'])
got=torch.cuda.device_count()
assert got>=n, (got, n)
print('GPU', got, 'OK (expect >=', n, ')')
"
  echo "[ok] Preflight done"
}

stage_wan_smoke() {
  echo_banner "WAN Smoke (§9.1)"
  require_file "${LAUNCH_SCRIPT}" "launch"
  run_launch "${WAN_SMOKE_LOG}" WAN_SMOKE=1
  check_launch_log "${WAN_SMOKE_LOG}"
}

stage_smoke() {
  echo_banner "Smoke 100 (§9.2)"
  require_file "${LAUNCH_SCRIPT}" "launch"
  run_launch "${SMOKE_LOG}" SMOKE=1
  check_launch_log "${SMOKE_LOG}"
}

stage_train() {
  local gpu_count="${TRAIN_PROC_PER_NODE:-8}"
  echo_banner "${gpu_count}-GPU 10k (§10)"
  require_file "${LAUNCH_SCRIPT}" "launch"
  TRAIN_JOB_NAME="${TRAIN_JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')-itvlaGp_p2_8g10k_${DATA_REPO_ID}}"
  TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${LOG_DIR%/}/${TRAIN_JOB_NAME}}"
  local extras=("JOB_NAME=${TRAIN_JOB_NAME}" "OUTPUT_DIR=${TRAIN_OUTPUT_DIR}")
  if [[ -n "${TRAIN_PROC_PER_NODE}" ]]; then
    extras+=("PROC_PER_NODE=${TRAIN_PROC_PER_NODE}")
    extras+=("CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}")
  fi
  if [[ -n "${BATCH_SIZE}" ]]; then
    extras+=("BATCH_SIZE=${BATCH_SIZE}")
  fi
  if [[ -n "${STEPS}" ]]; then
    extras+=("STEPS=${STEPS}")
  fi
  run_launch "${TRAIN_LOG}" "${extras[@]}"
  check_launch_log "${TRAIN_LOG}"
  echo "[ok] 训练日志: ${TRAIN_LOG}"
  echo "     checkpoint 目录: ${TRAIN_OUTPUT_DIR}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    ls -1dt "${TRAIN_OUTPUT_DIR}/checkpoints/"* 2>/dev/null | head -5 || true
  fi
}

echo "========== Phase 2 SFT 编排 =========="
echo "DRY_RUN      : ${DRY_RUN}  FORCE=${FORCE}"
echo "FROM/UNTIL   : ${FROM_STAGE} → ${UNTIL_STAGE}"
echo "PROJ_ROOT    : ${PROJ_ROOT}"
echo "VENV_ROOT    : ${VENV_ROOT}"
echo "RUNPKG_ROOT  : ${RUNPKG_ROOT}"
echo "GCS_PKG      : ${GCS_PKG}"
echo "GCS_VENV     : ${GCS_VENV}"
echo "DATA_REPO_ID : ${DATA_REPO_ID}"
echo "WARMUP_CKPT  : ${WARMUP_CKPT}"
echo "WAN_DIR      : ${WAN_DIR}"
echo "LAUNCH       : ${LAUNCH_SCRIPT}"
echo "HF_HOME      : ${HF_HOME}"
echo "HF_LEROBOT   : ${HF_LEROBOT_HOME}"
echo "LOG_DIR      : ${LOG_DIR}"
echo "EXPECT_GPUS  : ${EXPECT_GPUS}"
if [[ -n "${TRAIN_PROC_PER_NODE}" ]]; then
  echo "TRAIN_GPUS   : ${TRAIN_PROC_PER_NODE} (CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES})"
else
  echo "TRAIN_GPUS   : 8 (launch 默认 CUDA_VISIBLE_DEVICES=0-7)"
fi
echo "skip         : wan-smoke=${SKIP_WAN_SMOKE} smoke=${SKIP_SMOKE} train=${SKIP_TRAIN}"
echo "=============================================="
if [[ "${HF_HOME}" != "${VENV_ROOT}/"* ]]; then
  echo "警告: HF_HOME 不在 VENV_ROOT 下。手册默认是 ${VENV_ROOT}/var/hf_home；可用 --hf-home 显式指定。" >&2
fi
if [[ "${HF_LEROBOT_HOME}" != "${VENV_ROOT}/"* ]]; then
  echo "警告: HF_LEROBOT_HOME 不在 VENV_ROOT 下。手册默认是 ${VENV_ROOT}/var/datasets；可用 --hf-lerobot-home 显式指定。" >&2
fi

if should_run gcloud; then stage_gcloud; fi
if should_run runpkg; then stage_runpkg; fi
if should_run venv; then stage_venv; fi
if should_run install; then stage_install; fi
if should_run symlink; then stage_symlink; fi
if should_run wan; then stage_wan; fi
if should_run data-check; then stage_data_check; fi
if should_run preflight; then stage_preflight; fi
if should_run wan-smoke; then
  stage_wan_smoke
elif [[ "${SKIP_WAN_SMOKE}" -eq 1 ]]; then
  echo "[skip] wan-smoke (--skip-wan-smoke)"
fi
if should_run smoke; then
  stage_smoke
elif [[ "${SKIP_SMOKE}" -eq 1 ]]; then
  echo "[skip] smoke (--skip-smoke)"
fi
if should_run train; then
  stage_train
elif [[ "${SKIP_TRAIN}" -eq 1 ]]; then
  echo "[skip] train (--skip-train)"
fi

echo "完成（不含评测 §13）。"
echo "  下一步评测见 b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg.md §13"
