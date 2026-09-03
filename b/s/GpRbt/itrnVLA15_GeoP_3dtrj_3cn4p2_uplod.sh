#!/usr/bin/env bash
# =============================================================================
# itrnVLA15_GeoP_3dtrj_3cn4p2_uplod.sh
#
# 将 hanging_mug 1G Warmup 产物打成 RunPkg（数据 + Warmup ckpt），压缩后上传 GCS。
# 源码不打进包：远端复用 GCS venv itnvla0801116 内的 InternVLA-A-series。
#
# 包内结构:
#   RunPkg/Dta/hanging_mug_kptsim_lrbv30/
#   RunPkg/Ckp/warmup_hanging_mug_kptsim_400step/checkpoints/000400/
#
# 参考:
#   b/d/itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_hngMg.md
#   b/d/itrnVLA15_GeoP_3dtrj_3cn4_wrmup1G_hngMg_LOG.md
#   b/d/itrnVLA15_GeoP_3dtrj_3cn4_sft_rbt2_hngMg.md
#
# 用法:
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4p2_uplod.sh
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4p2_uplod.sh --dry-run
#   bash b/s/itrnVLA15_GeoP_3dtrj_3cn4p2_uplod.sh --skip-upload
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# 默认路径（可用环境变量或 CLI 覆盖）
# -----------------------------------------------------------------------------
STAGING_ROOT="${STAGING_ROOT:-/tmp/RunPkg}"

DATA_SRC="${DATA_SRC:-/home/luogang/share/zwy/Projects/DATA/RoboTwin-Clean/hanging_mug_kptsim_lrbv30}"
DATA_DST_REL="${DATA_DST_REL:-Dta/hanging_mug_kptsim_lrbv30}"

CKPT_SRC="${CKPT_SRC:-/home/luogang/SRC/Robot/itvlaGp/outputs/internvla_a1_5/warmup_hanging_mug_kptsim_400step/checkpoints/000400}"
CKPT_DST_REL="${CKPT_DST_REL:-Ckp/warmup_hanging_mug_kptsim_400step/checkpoints/000400}"

TAR_DIR="${TAR_DIR:-/tmp}"
TAR_NAME="${TAR_NAME:-}"   # 空则按压缩格式自动命名
GCS_DST="${GCS_DST:-gs://physical-ai-data-eu/VENV/tmp/RP/RunPkg_hngMg0825.tar.zst}"
GCLOUD_CMD="${GCLOUD_CMD:-gcloud storage cp}"

PACK_MODE="${PACK_MODE:-staging}"
COMPRESS="${COMPRESS:-zstd}"
COMPRESS_LEVEL="${COMPRESS_LEVEL:-1}"

DRY_RUN=0
SKIP_UPLOAD=0
KEEP_STAGING=0
CLEAN_STAGING=1

usage() {
  cat <<'EOF'
用法: itrnVLA15_GeoP_3dtrj_3cn4p2_uplod.sh [选项]

路径:
  --staging-root PATH     暂存根（默认 /tmp/RunPkg）
  --data-src PATH         数据源目录
  --data-dst-rel REL      包内数据相对路径
  --ckpt-src PATH         Warmup checkpoint 源目录
  --ckpt-dst-rel REL      包内 checkpoint 相对路径（默认 Ckp/.../000400）
  --tar-dir PATH          归档输出目录
  --tar-name NAME         归档文件名（默认随 compress 自动选 .tar.zst/.tar.gz/.tar）
  --tar-path PATH         归档完整路径（覆盖 tar-dir + tar-name）
  --gcs-dst URI           GCS 目标
  --gcloud-cmd STR        上传命令

打包:
  --pack-mode MODE        staging（默认；direct 已废弃，自动转 staging）
  --compress FMT          zstd（默认）| gzip | none
  --compress-level N      压缩级别（zstd/gzip，默认 1=偏速度）

行为:
  --dry-run               只打印命令
  --skip-upload           只打包不上传
  --keep-staging          staging 模式上传后保留暂存目录
  --clean-staging         staging 开始前清空暂存根（默认）
  --no-clean-staging      staging 不清空暂存根
  -h, --help

环境变量与 CLI 同名（大写下划线），CLI 优先。
EOF
}

TAR_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --staging-root)     STAGING_ROOT="$2"; shift 2 ;;
    --data-src)         DATA_SRC="$2"; shift 2 ;;
    --data-dst-rel)     DATA_DST_REL="$2"; shift 2 ;;
    --ckpt-src)         CKPT_SRC="$2"; shift 2 ;;
    --ckpt-dst-rel)     CKPT_DST_REL="$2"; shift 2 ;;
    --tar-dir)          TAR_DIR="$2"; shift 2 ;;
    --tar-name)         TAR_NAME="$2"; shift 2 ;;
    --tar-path)         TAR_PATH="$2"; shift 2 ;;
    --gcs-dst)          GCS_DST="$2"; shift 2 ;;
    --gcloud-cmd)       GCLOUD_CMD="$2"; shift 2 ;;
    --pack-mode)        PACK_MODE="$2"; shift 2 ;;
    --compress)         COMPRESS="$2"; shift 2 ;;
    --compress-level)   COMPRESS_LEVEL="$2"; shift 2 ;;
    --use-staging)      PACK_MODE=staging; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --skip-upload)      SKIP_UPLOAD=1; shift ;;
    --keep-staging)     KEEP_STAGING=1; shift ;;
    --clean-staging)    CLEAN_STAGING=1; shift ;;
    --no-clean-staging) CLEAN_STAGING=0; shift ;;
    --src-src|--src-dst-rel|--src-exclude)
      echo "警告: $1 已废弃（不再打包源码），忽略" >&2
      shift 2
      ;;
    -h|--help)          usage; exit 0 ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${PACK_MODE}" in
  direct)
    echo "警告: pack-mode=direct 已废弃，已自动改用 staging" >&2
    PACK_MODE=staging
    ;;
  staging) ;;
  *)
    echo "错误: --pack-mode 必须是 staging" >&2
    exit 1
    ;;
esac

case "${COMPRESS}" in
  zstd|gzip|none) ;;
  *)
    echo "错误: --compress 必须是 zstd、gzip 或 none" >&2
    exit 1
    ;;
esac

if [[ -z "${TAR_NAME}" ]]; then
  case "${COMPRESS}" in
    zstd) TAR_NAME="RunPkg_hngMg0825.tar.zst" ;;
    gzip) TAR_NAME="RunPkg_hngMg0825.tar.gz" ;;
    none) TAR_NAME="RunPkg_hngMg0825.tar" ;;
  esac
fi

if [[ -z "${TAR_PATH}" ]]; then
  TAR_PATH="${TAR_DIR%/}/${TAR_NAME}"
fi

DATA_DST_REL="${DATA_DST_REL#/}"
CKPT_DST_REL="${CKPT_DST_REL#/}"
DATA_DST="${STAGING_ROOT%/}/${DATA_DST_REL}"
CKPT_DST="${STAGING_ROOT%/}/${CKPT_DST_REL}"

run() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

elapsed() {
  local start="$1" label="$2"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "${label}: (dry-run)"
    return
  fi
  local now
  now=$(date +%s)
  echo "${label}: $((now - start))s"
}

require_dir() {
  local label="$1" path="$2"
  if [[ ! -d "${path}" ]]; then
    echo "错误: ${label} 不存在或不是目录: ${path}" >&2
    exit 1
  fi
}

tar_compress_args() {
  TAR_COMPRESS_ARGS=()
  case "${COMPRESS}" in
    zstd)
      if tar --help 2>&1 | grep -q -- '--zstd'; then
        TAR_COMPRESS_ARGS=(--zstd)
      else
        echo "警告: tar 不支持 --zstd，回退 gzip" >&2
        TAR_COMPRESS_ARGS=(-z)
      fi
      ;;
    gzip) TAR_COMPRESS_ARGS=(-z) ;;
    none) TAR_COMPRESS_ARGS=() ;;
  esac
}

apply_compress_env() {
  case "${COMPRESS}" in
    zstd) export ZSTD_CLEVEL="${COMPRESS_LEVEL}" ;;
    gzip) export GZIP="-${COMPRESS_LEVEL}" ;;
  esac
}

pack_staging() {
  if [[ "${CLEAN_STAGING}" -eq 1 ]]; then
    run rm -rf "${STAGING_ROOT}"
  fi
  run mkdir -p "${DATA_DST}" "$(dirname "${CKPT_DST}")"

  echo "[staging] 并行复制 data / ckpt ..."
  local t0 pid_data pid_ckpt
  t0=$(date +%s)

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '[dry-run] rsync data -> %q\n' "${DATA_DST}"
    printf '[dry-run] rsync ckpt -> %q\n' "${CKPT_DST}"
  else
    rsync -a --delete "${DATA_SRC%/}/" "${DATA_DST%/}/" &
    pid_data=$!
    rsync -a --delete "${CKPT_SRC%/}/" "${CKPT_DST%/}/" &
    pid_ckpt=$!
    wait "${pid_data}" "${pid_ckpt}"
  fi
  elapsed "${t0}" "  并行复制耗时"

  tar_compress_args
  apply_compress_env
  local staging_parent staging_base
  staging_parent="$(dirname "${STAGING_ROOT}")"
  staging_base="$(basename "${STAGING_ROOT}")"

  run mkdir -p "$(dirname "${TAR_PATH}")"
  [[ -f "${TAR_PATH}" ]] && run rm -f "${TAR_PATH}"
  local tar_args=(tar)
  if [[ ${#TAR_COMPRESS_ARGS[@]} -gt 0 ]]; then
    tar_args+=("${TAR_COMPRESS_ARGS[@]}")
  fi
  tar_args+=(-cf "${TAR_PATH}" -C "${staging_parent}" "${staging_base}")
  run "${tar_args[@]}"
}

verify_archive() {
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "归档校验: (dry-run)"
    return
  fi

  set +o pipefail
  echo "归档校验 ..."
  local list_cmd=()
  case "${TAR_PATH}" in
    *.zst) list_cmd=(zstd -dc "${TAR_PATH}") ;;
    *.gz) list_cmd=(gzip -dc "${TAR_PATH}") ;;
    *) list_cmd=(cat "${TAR_PATH}") ;;
  esac

  local listing sample data_ok ckpt_ok staging_base
  staging_base="$(basename "${STAGING_ROOT}")"
  listing=$("${list_cmd[@]}" | tar -tf -)
  sample=$(printf '%s\n' "${listing}" | head -5)
  if [[ -z "${sample}" ]]; then
    echo "错误: 归档为空或无法读取" >&2
    exit 1
  fi

  data_ok=$(printf '%s\n' "${listing}" | grep -c "^${staging_base}/${DATA_DST_REL}/" || true)
  ckpt_ok=$(printf '%s\n' "${listing}" | grep -c "^${staging_base}/${CKPT_DST_REL}/" || true)

  if [[ "${data_ok}" -eq 0 || "${ckpt_ok}" -eq 0 ]]; then
    echo "错误: 归档目录结构异常 (data=${data_ok}, ckpt=${ckpt_ok})" >&2
    printf '%s\n' "${listing}" | head -10 >&2
    exit 1
  fi

  if printf '%s\n' "${listing}" | grep -q "^${staging_base}/Src/"; then
    echo "错误: 归档不应包含 Src/ 目录" >&2
    exit 1
  fi

  echo "  data entries : ${data_ok}"
  echo "  ckpt entries : ${ckpt_ok}"
  echo "  sample paths :"
  printf '%s\n' "${listing}" | grep -E "^${staging_base}/(${DATA_DST_REL}|${CKPT_DST_REL})/" | head -3 | sed 's/^/    /'
  set -o pipefail
}

verify_gcs_and_remove_local_tar() {
  if [[ "${DRY_RUN}" -eq 1 || "${SKIP_UPLOAD}" -eq 1 ]]; then
    return
  fi

  echo "校验 GCS 归档 ${GCS_DST} ..."
  local remote_size
  if ! remote_size=$(gcloud storage objects describe "${GCS_DST}" --format="value(size)" 2>/dev/null); then
    echo "错误: GCS 上未找到 ${GCS_DST}" >&2
    exit 1
  fi

  local local_size
  local_size=$(stat -c%s "${TAR_PATH}")
  if [[ "${remote_size}" != "${local_size}" ]]; then
    echo "错误: GCS 对象大小 (${remote_size}) 与本地 (${local_size}) 不一致，保留本地归档" >&2
    exit 1
  fi

  echo "GCS 校验通过 (${remote_size} bytes)，删除本地归档 ${TAR_PATH}"
  run rm -f "${TAR_PATH}"
}

echo "========== RunPkg 打包上传 =========="
echo "PACK_MODE    : ${PACK_MODE}"
echo "COMPRESS     : ${COMPRESS} (level ${COMPRESS_LEVEL})"
echo "DATA_SRC     : ${DATA_SRC}"
echo "DATA_DST_REL : ${DATA_DST_REL}"
echo "CKPT_SRC     : ${CKPT_SRC}"
echo "CKPT_DST_REL : ${CKPT_DST_REL}"
echo "TAR_PATH     : ${TAR_PATH}"
echo "GCS_DST      : ${GCS_DST}"
echo "===================================="

require_dir "DATA_SRC" "${DATA_SRC}"
require_dir "CKPT_SRC" "${CKPT_SRC}"

t_pack=$(date +%s)
echo "[1/2] staging 打包 → ${TAR_PATH} ..."
pack_staging
elapsed "${t_pack}" "打包总耗时"
verify_archive

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "归档大小: $(du -h "${TAR_PATH}" | awk '{print $1}')"
fi

if [[ "${SKIP_UPLOAD}" -eq 1 ]]; then
  echo "[2/2] 跳过上传 (--skip-upload)"
else
  echo "[2/2] 上传到 ${GCS_DST} ..."
  t_up=$(date +%s)
  # shellcheck disable=SC2086
  run ${GCLOUD_CMD} "${TAR_PATH}" "${GCS_DST}"
  elapsed "${t_up}" "上传耗时"
fi

if [[ "${PACK_MODE}" == "staging" && "${KEEP_STAGING}" -eq 0 ]]; then
  echo "清理暂存目录 ${STAGING_ROOT}"
  run rm -rf "${STAGING_ROOT}"
fi

verify_gcs_and_remove_local_tar

echo "完成."
if [[ -f "${TAR_PATH}" ]]; then
  echo "  本地归档 : ${TAR_PATH}"
else
  echo "  本地归档 : (已删除)"
fi
echo "  GCS      : ${GCS_DST}"
