#!/usr/bin/env bash
# Batch process all RoboTwin 2.0 v2.1 source tasks:
# generate kptsim 3D keypoint LeRobot v3.0 datasets -> {task}_lrb3_kptsim/
set -euo pipefail

CLEAN_ROOT="${CLEAN_ROOT:-/B/Dta/RoboTwin-Clean}"
GEOPREDICT_ROOT="${GEOPREDICT_ROOT:-/B/SRC/GeoPredict}"
ITVLAGP_ROOT="${ITVLAGP_ROOT:-/B/SRC/itvlaGp}"
URDF_PATH="${URDF_PATH:-/B/SRC/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf}"
PYTHON="${PYTHON:-python3}"
FORCE="${FORCE:-0}"
KEEP_GOING="${KEEP_GOING:-0}"
NORM_STATS_DIR="${CLEAN_ROOT}/.norm_stats"
CONVERT_WS_ROOT="${CLEAN_ROOT}/.convert_ws"

TASKS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)       FORCE=1; shift ;;
    --keep-going)  KEEP_GOING=1; shift ;;
    --tasks)       IFS=',' read -ra TASKS <<< "$2"; shift 2 ;;
    --task)        TASKS+=("$2"); shift 2 ;;
    *)             echo "Unknown arg: $1"; exit 1 ;;
  esac
done

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }
die() { echo "[$(ts)] ERROR: $*" >&2; exit 1; }

lrb3_kptsim_ready() {
  local d="${CLEAN_ROOT}/${1}_lrb3_kptsim"
  [[ -f "${d}/meta/info.json" ]] && [[ -f "${d}/norm_stat.json" ]] && [[ -f "${d}/meta/keypoints_meta.json" ]]
}

if [[ ${#TASKS[@]} -eq 0 ]]; then
  while IFS= read -r name; do
    TASKS+=("${name}")
  done < <(
    for d in "${CLEAN_ROOT}"/*/; do
      name="$(basename "$d")"
      [[ "${name}" == *_lrb3 ]]        && continue
      [[ "${name}" == *_lrb3_kptsim ]] && continue
      [[ "${name}" == *_kptsim* ]]     && continue
      [[ "${name}" == *_old ]]         && continue
      [[ "${name}" == .* ]]            && continue
      [[ -f "${d}meta/info.json" ]]    || continue
      ver="$("${PYTHON}" -c "import json; print(json.load(open('${d}meta/info.json')).get('codebase_version',''))")"
      [[ "${ver}" == "v2.1" ]]         || { log "Skip ${name}: version ${ver} (not v2.1)" >&2; continue; }
      echo "${name}"
    done
  )
fi

log "Tasks to process: ${#TASKS[@]}"
[[ ${#TASKS[@]} -gt 0 ]] || { log "No tasks found"; exit 0; }

[[ -d "${CLEAN_ROOT}" ]]                                         || die "CLEAN_ROOT missing: ${CLEAN_ROOT}"
[[ -f "${URDF_PATH}" ]]                                         || die "URDF missing: ${URDF_PATH}"
[[ -f "${GEOPREDICT_ROOT}/b/script/kpt/run_extract.py" ]]       || die "Extract script missing"
[[ -f "${ITVLAGP_ROOT}/util_scripts/inject_kptsim_keypoints.py" ]] || die "Inject script missing"
[[ -f "${ITVLAGP_ROOT}/b/s/rbt/layer1_check.py" ]]              || die "Layer-1 check missing"
"${PYTHON}" -c "import sapien" 2>/dev/null                       || die "sapien not installed"
"${PYTHON}" -c "import transforms3d" 2>/dev/null                 || die "transforms3d not installed"

mkdir -p "${NORM_STATS_DIR}"

SUCCEEDED=0
FAILED=0
SKIPPED=0
FAIL_LIST=()

for TASK in "${TASKS[@]}"; do
  log "======== ${TASK} ========"

  TASK_SRC="${CLEAN_ROOT}/${TASK}"
  TASK_KPTSIM="${CLEAN_ROOT}/${TASK}_kptsim"
  TASK_LRB="${CLEAN_ROOT}/${TASK}_kptsim_lrb"
  TASK_FINAL="${CLEAN_ROOT}/${TASK}_lrb3_kptsim"
  TASK_NORM="${NORM_STATS_DIR}/robotwin_norm_stats_${TASK}.json"
  TASK_CONVERT_WS="${CONVERT_WS_ROOT}/${TASK}"

  if [[ "${FORCE}" != "1" ]] && lrb3_kptsim_ready "${TASK}"; then
    log "Skip ${TASK}: already complete"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if (
    set -e

    # Stage 1: SAPIEN FK
    if [[ "${FORCE}" == "1" ]] || [[ ! -f "${TASK_KPTSIM}/keypoints_meta.json" ]]; then
      log "[${TASK}] Stage 1/5: SAPIEN FK"
      rm -rf "${TASK_KPTSIM}"
      ( cd "${GEOPREDICT_ROOT}" && "${PYTHON}" b/script/kpt/run_extract.py \
          --dataset_dir "${TASK_SRC}" --urdf_path "${URDF_PATH}" --output_dir "${TASK_KPTSIM}" )
    else
      log "[${TASK}] Stage 1/5: reuse ${TASK_KPTSIM}"
    fi

    # Stage 2: norm stats
    if [[ "${FORCE}" == "1" ]] || [[ ! -f "${TASK_NORM}" ]]; then
      log "[${TASK}] Stage 2/5: norm stats"
      "${PYTHON}" "${GEOPREDICT_ROOT}/tools/compute_robotwin_norm_stats.py" \
        --dataset_dir "${TASK_SRC}" --output "${TASK_NORM}"
    else
      log "[${TASK}] Stage 2/5: reuse norm stats"
    fi

    # Stage 3: inject
    if [[ "${FORCE}" == "1" ]] || [[ ! -f "${TASK_LRB}/meta/keypoints_meta.json" ]]; then
      log "[${TASK}] Stage 3/5: inject keypoints"
      "${PYTHON}" "${ITVLAGP_ROOT}/util_scripts/inject_kptsim_keypoints.py" \
        --source "${TASK_SRC}" --kptsim_dir "${TASK_KPTSIM}" --dest "${TASK_LRB}" \
        --norm_stats_path "${TASK_NORM}" --coord_mode voxel --force
    else
      log "[${TASK}] Stage 3/5: reuse injected dataset"
    fi

    # Stage 4: Layer-1 check
    log "[${TASK}] Stage 4/5: Layer-1 check"
    "${PYTHON}" "${ITVLAGP_ROOT}/b/s/rbt/layer1_check.py" \
      --dest-root "${TASK_LRB}" --kptsim-root "${TASK_KPTSIM}" --task "${TASK}"

    # Stage 5: v2.1→v3.0
    log "[${TASK}] Stage 5/5: v2.1→v3.0"
    rm -rf "${TASK_CONVERT_WS}"
    mkdir -p "${TASK_CONVERT_WS}/robotwin"
    ln -sfn "${TASK_LRB}" "${TASK_CONVERT_WS}/robotwin/${TASK}_kptsim"

    "${PYTHON}" "${ITVLAGP_ROOT}/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py" \
      --repo-id="robotwin/${TASK}_kptsim" --root="${TASK_CONVERT_WS}" \
      --push-to-hub=false --force-conversion

    CONVERT_OUT="${TASK_CONVERT_WS}/robotwin/${TASK}_kptsim_v30"
    [[ -d "${CONVERT_OUT}" ]] || { log "ERROR: conversion output missing"; exit 1; }

    [[ -d "${TASK_FINAL}" ]] && rm -rf "${TASK_FINAL}"
    mkdir -p "${TASK_FINAL}"
    rsync -a --delete "${CONVERT_OUT}/" "${TASK_FINAL}/"
    mkdir -p "${TASK_FINAL}/meta"
    cp -f "${TASK_LRB}/meta/keypoints_meta.json" "${TASK_FINAL}/meta/keypoints_meta.json"
    cp -f "${TASK_LRB}/norm_stat.json" "${TASK_FINAL}/norm_stat.json"
    rm -rf "${TASK_CONVERT_WS}"

    # Layer-2 verify
    "${PYTHON}" -c "
import json
from pathlib import Path
root = Path('${TASK_FINAL}')
info = json.loads((root / 'meta' / 'info.json').read_text())
ver = str(info.get('codebase_version', ''))
assert '3.0' in ver or ver.startswith('v3'), f'codebase_version={ver}'
assert 'observation.keypoint_3d' in info.get('features', {}), 'v30 missing keypoint_3d'
assert (root / 'norm_stat.json').is_file(), 'missing norm_stat.json'
assert (root / 'meta' / 'keypoints_meta.json').is_file(), 'missing keypoints_meta.json'
print(f'Layer-2 OK: {ver}, episodes={info.get(\"total_episodes\")}, frames={info.get(\"total_frames\")}')
"

    # Cleanup intermediates
    log "[${TASK}] Cleanup intermediates"
    rm -rf "${TASK_KPTSIM}"
    rm -rf "${TASK_LRB}"
    rm -f  "${TASK_NORM}"
    log "[${TASK}] DONE"

  ); then
    SUCCEEDED=$((SUCCEEDED + 1))
  else
    FAILED=$((FAILED + 1))
    FAIL_LIST+=("${TASK}")
    log "!!! ${TASK} FAILED !!!"
    if [[ "${KEEP_GOING}" != "1" ]]; then
      die "Aborted at ${TASK}. Use --keep-going to continue."
    fi
  fi
done

log "======== SUMMARY ========"
log "  Succeeded: ${SUCCEEDED}"
log "  Skipped:   ${SKIPPED}"
log "  Failed:    ${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
  log "  Failed tasks: ${FAIL_LIST[*]}"
fi

rmdir "${NORM_STATS_DIR}" 2>/dev/null || true
rmdir "${CONVERT_WS_ROOT}" 2>/dev/null || true

[[ ${FAILED} -eq 0 ]]
