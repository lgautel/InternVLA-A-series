#!/usr/bin/env bash
# Phase 0: RoboTwin 2.0 task -> LeRobot v3.0 training dataset with 3D keypoints.
# Reuses GeoPredict SAPIEN extractor + itvlaGp inject + v21-to-v30 converter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "${SCRIPT_DIR}/lib.sh"

TASK="${1:?usage: phase0_prep_data.sh <task_name>}"
resolve_task_paths "${TASK}"
FORCE="${FORCE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
JOB_STAMP="$(date +'%Y_%m_%d_%H_%M_%S')"
LOG="${TASK_LOG_DIR}/phase0_${JOB_STAMP}.log"

rbt_mkdir "${TASK_LOG_DIR}" "${TASK_CKPT_DIR}"
rbt_log "==== Phase0 ${TASK} ===="
print_task_paths

if [[ "${SKIP_EXISTING}" == "1" && "${FORCE}" != "1" ]] && v30_ready; then
  rbt_log "跳过 Phase0: 已存在完整 v3.0 数据 ${TASK_V30}"
  write_state "phase0" "skipped" "{\"reason\":\"v30_ready\",\"v30\":\"${TASK_V30}\"}"
  exit 0
fi

write_state "phase0" "running" "{\"log\":\"${LOG}\",\"job_stamp\":\"${JOB_STAMP}\"}"

[[ -d "${TASK_SRC}" ]] || rbt_die "源任务目录不存在: ${TASK_SRC}"
rbt_require_file "${TASK_SRC}/meta/info.json" "源数据集 info.json"
rbt_require_file "${URDF_PATH}" "aloha-agilex URDF"
rbt_require_file "${GEOPREDICT_ROOT}/b/script/kpt/run_extract.py"
rbt_require_file "${ITVLAGP_ROOT}/util_scripts/inject_kptsim_keypoints.py"
rbt_require_file "${ITVLAGP_ROOT}/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py"
rbt_require_file "${GEOPREDICT_ROOT}/tools/compute_robotwin_norm_stats.py"

# --- 1. SAPIEN extract (must run from GeoPredict root; uses RoboTwin Python) ---
if [[ "${FORCE}" == "1" ]] || [[ ! -f "${TASK_KPTSIM}/keypoints_meta.json" ]]; then
  rbt_log "SAPIEN 提取 -> ${TASK_KPTSIM}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    rbt_log "DRY-RUN: extract ${TASK_SRC}"
  else
    rbt_mkdir "${TASK_KPTSIM}"
    (
      cd "${GEOPREDICT_ROOT}"
      "${EXTRACT_PYTHON}" b/script/kpt/run_extract.py \
        --dataset_dir "${TASK_SRC}" \
        --urdf_path "${URDF_PATH}" \
        --output_dir "${TASK_KPTSIM}"
    ) >> "${LOG}" 2>&1
  fi
else
  rbt_log "复用已有 kptsim: ${TASK_KPTSIM}"
fi

# --- 2. Per-task norm stats (state / actions keys) ---
if [[ "${FORCE}" == "1" ]] || [[ ! -f "${TASK_NORM_RAW}" ]]; then
  rbt_log "计算归一化统计 -> ${TASK_NORM_RAW}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    rbt_mkdir "$(dirname "${TASK_NORM_RAW}")"
    "${TRAIN_PYTHON}" "${GEOPREDICT_ROOT}/tools/compute_robotwin_norm_stats.py" \
      --dataset_dir "${TASK_SRC}" \
      --output "${TASK_NORM_RAW}" >> "${LOG}" 2>&1
  fi
else
  rbt_log "复用已有 norm stats: ${TASK_NORM_RAW}"
fi

# --- 3. Inject keypoints into a task-local v2.1 copy ---
if [[ "${FORCE}" == "1" ]] || [[ ! -f "${TASK_LRB}/meta/keypoints_meta.json" ]]; then
  rbt_log "注入 keypoint_3d -> ${TASK_LRB}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${TRAIN_PYTHON}" "${ITVLAGP_ROOT}/util_scripts/inject_kptsim_keypoints.py" \
      --source "${TASK_SRC}" \
      --kptsim_dir "${TASK_KPTSIM}" \
      --dest "${TASK_LRB}" \
      --norm_stats_path "${TASK_NORM_RAW}" \
      --coord_mode voxel \
      --force >> "${LOG}" 2>&1
  fi
else
  rbt_log "复用已注入数据集: ${TASK_LRB}"
fi

# --- 4. Layer-1 acceptance (generic, parameterized) ---
if [[ "${DRY_RUN}" != "1" ]]; then
  rbt_log "Layer-1 验收"
  "${TRAIN_PYTHON}" "${SCRIPT_DIR}/layer1_check.py" \
    --dest-root "${TASK_LRB}" \
    --kptsim-root "${TASK_KPTSIM}" \
    --task "${TASK}" >> "${LOG}" 2>&1
fi

# --- 5. v2.1 -> v3.0 in an isolated convert workspace, then rsync to TASK_V30 ---
# convert script does Path(root)/repo_id and writes sibling {name}_v30, then rmtree's it.
# Isolation per task avoids clobbering other conversions.
CONVERT_REPO_REL="robotwin/${TASK}_kptsim"
CONVERT_LINK_PARENT="${TASK_CONVERT_WS}/robotwin"
CONVERT_LINK="${CONVERT_LINK_PARENT}/${TASK}_kptsim"
CONVERT_OUT="${CONVERT_LINK_PARENT}/${TASK}_kptsim_v30"

if [[ "${FORCE}" == "1" ]] || ! v30_ready; then
  rbt_log "v2.1 -> v3.0 (workspace=${TASK_CONVERT_WS})"
  if [[ "${DRY_RUN}" != "1" ]]; then
    rm -rf "${TASK_CONVERT_WS}"
    rbt_mkdir "${CONVERT_LINK_PARENT}"
    ln -sfn "${TASK_LRB}" "${CONVERT_LINK}"
    "${TRAIN_PYTHON}" "${ITVLAGP_ROOT}/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py" \
      --repo-id="${CONVERT_REPO_REL}" \
      --root="${TASK_CONVERT_WS}" \
      --push-to-hub=false \
      --force-conversion >> "${LOG}" 2>&1
    [[ -d "${CONVERT_OUT}" ]] || rbt_die "转换未产出 ${CONVERT_OUT}"
    rbt_mkdir "${TASK_V30}"
    rsync -a --delete "${CONVERT_OUT}/" "${TASK_V30}/"
    rbt_mkdir "${TASK_V30}/meta"
    cp -f "${TASK_LRB}/meta/keypoints_meta.json" "${TASK_V30}/meta/keypoints_meta.json"
    cp -f "${TASK_LRB}/norm_stat.json" "${TASK_V30}/norm_stat.json"
    rm -rf "${TASK_CONVERT_WS}"
  fi
else
  rbt_log "复用已有 v3.0: ${TASK_V30}"
fi

v30_ready || rbt_die "Phase0 结束后 v3.0 仍不完整: ${TASK_V30}"
if [[ "${DRY_RUN}" != "1" ]]; then
  "${TRAIN_PYTHON}" - "${TASK_V30}" <<'PY' >> "${LOG}" 2>&1
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())
ver = str(info.get("codebase_version", ""))
assert "3.0" in ver or ver.startswith("v3"), f"codebase_version={ver}"
assert "observation.keypoint_3d" in info.get("features", {}), "v30 missing keypoint_3d"
print("Layer-2 v3.0 OK", ver, "episodes", info.get("total_episodes"), "frames", info.get("total_frames"))
PY
fi
write_state "phase0" "ok" "{\"v30\":\"${TASK_V30}\",\"repo_id\":\"${TASK_REPO_ID}\"}"
rbt_log "Phase0 完成: ${TASK_V30}"
