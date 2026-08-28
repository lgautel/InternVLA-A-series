#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Prepare one or more RoboTwin 2.0 tasks for InternVLA-A1.5 fine-tuning.
#
# For each task this script:
#   1) sources the configurable venv and optionally editable-reinstalls this repo
#   2) converts LeRobot v2.1 -> v3.0 without touching the Clean source
#   3) persists the result next to the source as <task>_lrb3
#   4) points repo_id=robotwin/<task> at the v3.0 result
#   5) smoke-loads the dataset and computes external action/state stats
#
# Usage (from anywhere; PROJ_ROOT is inferred from this script):
#   bash launch/internvla_a15_prepare_robotwin.sh
#
# Single-task overrides:
#   VENV_ROOT=/B/VENV/itnvla15rbt20
#   ROBOTWIN_CLEAN_ROOT=/B/Dta/RoboTwin-Clean
#   TASK_NAME=scan_object
#   ACTION_TYPE=abs
#   SKIP_PIP_INSTALL=1       # skip pip install -e .
#   SKIP_CONVERT=1           # reuse existing <task>_lrb3
#
# Multiple tasks:
#   TASK_NAMES="scan_object hanging_mug"
#   TASK_NAMES="scan_object,hanging_mug"
#   ALL_TASKS=1               # all direct task dirs without the _lrb3 suffix
#
# ALL_TASKS=1 intentionally includes a source that is already v3.0 but does
# not yet have a corresponding _lrb3 directory. That source is copied, not
# converted. Existing _lrb3 directories are never selected automatically.
#
# See b/d/p/reprd_rbtwn_scnObj.md and b/d/p/reprd_rbtwn_scnObjLOG.md
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/internvla_a15_robotwin_common.sh"

activate_itnvla_venv
cd "${PROJ_ROOT}"
print_path_banner

############################## Select tasks ###################################

declare -a TASKS=()
if [[ "${ALL_TASKS:-0}" == "1" ]]; then
    shopt -s nullglob
    for _candidate in "${ROBOTWIN_CLEAN_ROOT}"/*; do
        [[ -d "${_candidate}" ]] || continue
        _task="${_candidate##*/}"
        [[ "${_task}" == .* || "${_task}" == *_lrb3 ]] && continue
        [[ -f "${_candidate}/meta/info.json" ]] || continue
        TASKS+=("${_task}")
    done
    shopt -u nullglob
elif [[ -n "${TASK_NAMES:-}" ]]; then
    _task_names="${TASK_NAMES//,/ }"
    read -r -a TASKS <<< "${_task_names}"
else
    TASKS=("${TASK_NAME}")
fi

if [[ "${#TASKS[@]}" -eq 0 ]]; then
    echo "ERROR: no RoboTwin tasks selected." >&2
    echo "  Set TASK_NAME, TASK_NAMES, or ALL_TASKS=1." >&2
    exit 1
fi

for _task in "${TASKS[@]}"; do
    if [[ -z "${_task}" || "${_task}" == */* || "${_task}" == "." || "${_task}" == ".." ]]; then
        echo "ERROR: invalid task name: ${_task}" >&2
        exit 1
    fi
done

echo "Selected tasks (${#TASKS[@]}): ${TASKS[*]}"

############################## 1. editable reinstall ##########################

if [[ "${SKIP_PIP_INSTALL:-0}" != "1" ]]; then
    echo "===== pip install -e . into ${VENV_ROOT} ====="
    "${VENV_ROOT}/bin/python" -m pip install -e "${PROJ_ROOT}"
else
    echo "SKIP_PIP_INSTALL=1: not reinstalling the repo"
fi

############################## 2. transformers patch ##########################

PY_SITE="$("${VENV_ROOT}/bin/python" -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")"
if [[ ! -f "${PY_SITE}/models/qwen3_5/modeling_qwen3_5.py" ]]; then
    echo "Patching transformers Qwen3.5 into ${PY_SITE}"
    cp -r "${PROJ_ROOT}/src/lerobot/policies/pi0/transformers_replace/models" "${PY_SITE}"
    cp -r "${PROJ_ROOT}/src/lerobot/policies/pi05/transformers_replace/models" "${PY_SITE}"
    cp -r "${PROJ_ROOT}/src/lerobot/policies/internvla_a1_5/transformers_replace/models" "${PY_SITE}"
else
    echo "Transformers Qwen3.5 patch already present: ${PY_SITE}/models/qwen3_5/modeling_qwen3_5.py"
fi

############################## 3. data root symlink ###########################

mkdir -p "${HF_LEROBOT_HOME}/robotwin"
ln -sfn "${HF_LEROBOT_HOME}" "${PROJ_ROOT}/data"
echo "data -> ${HF_LEROBOT_HOME}"

############################## Per-task preparation ###########################

for TASK_NAME in "${TASKS[@]}"; do
    SRC_DIR="${ROBOTWIN_CLEAN_ROOT}/${TASK_NAME}"
    DST_DIR="${ROBOTWIN_CLEAN_ROOT}/${TASK_NAME}_lrb3"
    REPO_ID="robotwin/${TASK_NAME}"
    REPO_ID_LRB3="robotwin/${TASK_NAME}_lrb3"
    SRC_INFO="${SRC_DIR}/meta/info.json"

    echo
    echo "###############################################################################"
    echo "===== prepare RoboTwin task: ${TASK_NAME} ====="

    if [[ ! -f "${SRC_INFO}" ]]; then
        echo "ERROR: RoboTwin source dataset not found: ${SRC_INFO}" >&2
        echo "  Set ROBOTWIN_CLEAN_ROOT (currently ${ROBOTWIN_CLEAN_ROOT})" >&2
        exit 1
    fi

    _src_ver="$(
        INFO_PATH="${SRC_INFO}" "${VENV_ROOT}/bin/python" - <<'PY'
import json
import os

with open(os.environ["INFO_PATH"]) as f:
    print(json.load(f).get("codebase_version", ""))
PY
    )"
    _robot_type="$(
        INFO_PATH="${SRC_INFO}" "${VENV_ROOT}/bin/python" - <<'PY'
import json
import os

with open(os.environ["INFO_PATH"]) as f:
    print(json.load(f).get("robot_type", ""))
PY
    )"
    echo "source ${SRC_DIR} codebase_version=${_src_ver} robot_type=${_robot_type}"

    if [[ "${_src_ver}" != "v2.1" && "${_src_ver}" != "v3.0" ]]; then
        echo "ERROR: unexpected source codebase_version=${_src_ver} at ${SRC_DIR}" >&2
        exit 1
    fi
    if [[ -z "${_robot_type}" ]]; then
        echo "ERROR: source dataset has no robot_type: ${SRC_INFO}" >&2
        exit 1
    fi

    ########################## Convert / copy v3.0 #############################

    if [[ "${SKIP_CONVERT:-0}" == "1" ]]; then
        if [[ ! -f "${DST_DIR}/meta/info.json" ]]; then
            echo "ERROR: SKIP_CONVERT=1 but converted dataset is missing: ${DST_DIR}" >&2
            exit 1
        fi
        echo "SKIP_CONVERT=1 and ${DST_DIR} exists: skipping conversion"
    elif [[ "${_src_ver}" == "v3.0" ]]; then
        echo "Source is already v3.0; copying to ${DST_DIR} without conversion"
        rm -rf "${DST_DIR}"
        mkdir -p "${DST_DIR}"
        rsync -a "${SRC_DIR}/" "${DST_DIR}/"
    else
        # Conversion reads ${HF_LEROBOT_HOME}/<old-repo-id>. Point that at the
        # original Clean tree (read-only). The converter writes
        # ${HF_LEROBOT_HOME}/<new-repo-id>.
        rm -rf "${HF_LEROBOT_HOME:?}/${REPO_ID}"
        ln -s "${SRC_DIR}" "${HF_LEROBOT_HOME}/${REPO_ID}"
        rm -rf "${HF_LEROBOT_HOME:?}/${REPO_ID_LRB3}"

        echo "===== convert ${REPO_ID} (v2.1) -> ${REPO_ID_LRB3} (v3.0) ====="
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
            "${VENV_ROOT}/bin/python" \
            "${PROJ_ROOT}/src/lerobot/datasets/v30/convert_my_dataset_v21_to_v30.py" \
            --old-repo-id "${REPO_ID}" \
            --new-repo-id "${REPO_ID_LRB3}" \
            --push-to-hub false \
            --force-conversion

        if [[ ! -f "${HF_LEROBOT_HOME}/${REPO_ID_LRB3}/meta/info.json" ]]; then
            echo "ERROR: conversion did not write ${HF_LEROBOT_HOME}/${REPO_ID_LRB3}" >&2
            exit 1
        fi

        rm -rf "${DST_DIR}"
        mkdir -p "${DST_DIR}"
        rsync -a "${HF_LEROBOT_HOME}/${REPO_ID_LRB3}/" "${DST_DIR}/"
        # The durable copy lives under ROBOTWIN_CLEAN_ROOT. Do not retain a
        # second full dataset in the HF cache after the copy succeeds.
        rm -rf "${HF_LEROBOT_HOME:?}/${REPO_ID_LRB3}"
    fi

    # Training always loads repo_id=robotwin/<task>, which must resolve to
    # v3.0. Replace real directories as well as stale symlinks; ln -sfn alone
    # would otherwise nest a link inside an existing directory.
    for _link_name in "${HF_LEROBOT_HOME}/${REPO_ID}" "${HF_LEROBOT_HOME}/${REPO_ID_LRB3}"; do
        if [[ -e "${_link_name}" || -L "${_link_name}" ]]; then
            rm -rf "${_link_name}"
        fi
        ln -s "${DST_DIR}" "${_link_name}"
    done

    _dst_ver="$(
        INFO_PATH="${DST_DIR}/meta/info.json" "${VENV_ROOT}/bin/python" - <<'PY'
import json
import os

with open(os.environ["INFO_PATH"]) as f:
    print(json.load(f).get("codebase_version", ""))
PY
    )"
    if [[ "${_dst_ver}" != "v3.0" ]]; then
        echo "ERROR: converted dataset codebase_version=${_dst_ver} (need v3.0): ${DST_DIR}" >&2
        exit 1
    fi
    echo "converted dataset at ${DST_DIR} codebase_version=${_dst_ver}"
    echo "training symlink: ${HF_LEROBOT_HOME}/${REPO_ID} -> ${DST_DIR}"

    ############################## Smoke load #################################

    echo "===== smoke: LeRobotDataset('${REPO_ID}') ====="
    TASK_REPO_ID="${REPO_ID}" "${VENV_ROOT}/bin/python" - <<'PY'
import os

from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = os.environ["TASK_REPO_ID"]
ds = LeRobotDataset(repo_id, root=None, download_videos=False)
print("version", ds.meta._version)
print("episodes", ds.meta.total_episodes, "frames", ds.meta.total_frames)
print("robot", ds.meta.robot_type, "fps", ds.meta.fps)
print("cameras", ds.meta.camera_keys)
print("len", len(ds))
sample = ds[0]
for key in ds.meta.camera_keys:
    tensor = sample[key]
    print(key, "shape", tuple(tensor.shape), "min", float(tensor.min()), "max", float(tensor.max()))
    if float(tensor.max()) <= 0:
        raise SystemExit(f"FAIL: {key} looks like a zero fallback frame")
print("task", sample.get("task") or sample.get("observation.language") or "")
print("SMOKE_DATASET_OK")
PY

    ############################## External stats ##############################

    GROUP_NAME="$(stats_group_name "${REPO_ID}")"
    STATS_PATH="${HF_LEROBOT_HOME}/stats/${_robot_type}/${ACTION_TYPE}/${GROUP_NAME}/stats.json"
    echo "===== compute_norm_stats_multi ${ACTION_TYPE} chunk_size=${CHUNK_SIZE} ====="
    "${VENV_ROOT}/bin/python" "${PROJ_ROOT}/util_scripts/compute_norm_stats_multi.py" \
        --action_mode "${ACTION_TYPE}" \
        --chunk_size "${CHUNK_SIZE}" \
        --num_workers "${STATS_NUM_WORKERS:-8}" \
        --repo_ids "${REPO_ID}"

    if [[ ! -f "${STATS_PATH}" ]]; then
        echo "ERROR: stats not written: ${STATS_PATH}" >&2
        exit 1
    fi
    echo "EXTERNAL_STATS_PATH=${STATS_PATH}"
    STATS_FILE="${STATS_PATH}" "${VENV_ROOT}/bin/python" - <<'PY'
import json
import os

with open(os.environ["STATS_FILE"]) as f:
    stats = json.load(f)
for key, value in stats.items():
    if isinstance(value, dict) and "mean" in value:
        mean = value["mean"]
        dim = len(mean) if isinstance(mean, list) else "?"
        print(f"{key}: dim={dim} count={value.get('count')}")
print("SMOKE_STATS_OK")
PY
done

echo
echo "===== prepare done: ${#TASKS[@]} task(s) ====="
for TASK_NAME in "${TASKS[@]}"; do
    echo "  ${TASK_NAME}: ${ROBOTWIN_CLEAN_ROOT}/${TASK_NAME}_lrb3"
done
