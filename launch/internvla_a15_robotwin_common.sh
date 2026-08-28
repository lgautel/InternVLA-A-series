#!/usr/bin/env bash
# Shared env helpers for RoboTwin single-task InternVLA-A1.5 fine-tunes.
# Source this file from other launch scripts. Do not execute it directly.
#
# Overridable (export before source, or before calling the parent script):
#   VENV_ROOT              default /B/VENV/itnvla15rbt20
#   HF_HOME_OVERRIDE       if set, used AFTER venv activate (activate itself sets
#                          HF_HOME=$VENV_ROOT/var/hf_home)
#   ROBOTWIN_CLEAN_ROOT    default /B/Dta/RoboTwin-Clean
#   CKPT_BASE              default /B/Ckp
#   TASK_NAME              default scan_object
#   PROJ_ROOT              inferred from this file's location if unset

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this file, do not execute it: source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="${PROJ_ROOT:-$(cd "${_COMMON_DIR}/.." && pwd)}"

VENV_ROOT="${VENV_ROOT:-/B/VENV/itnvla15rbt20}"
ROBOTWIN_CLEAN_ROOT="${ROBOTWIN_CLEAN_ROOT:-/B/Dta/RoboTwin-Clean}"
CKPT_BASE="${CKPT_BASE:-/B/Ckp}"
TASK_NAME="${TASK_NAME:-scan_object}"
ACTION_TYPE="${ACTION_TYPE:-abs}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"

compact_stamp() {
    date +'%y%m%d%H%M'
}

activate_itnvla_venv() {
    if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
        echo "ERROR: venv activate not found: ${VENV_ROOT}/bin/activate" >&2
        echo "  Set VENV_ROOT to the internvla venv (must be sourced, not just its python)." >&2
        return 1
    fi
    # Must source: this venv's activate exports HF_HOME, HF_LEROBOT_HOME, LD_LIBRARY_PATH.
    # Using ${VENV_ROOT}/bin/python without source skips those exports.
    # shellcheck disable=SC1091
    source "${VENV_ROOT}/bin/activate"
    if [[ -n "${HF_HOME_OVERRIDE:-}" ]]; then
        export HF_HOME="${HF_HOME_OVERRIDE}"
        export HF_LEROBOT_HOME="${HF_LEROBOT_HOME_OVERRIDE:-${HF_HOME}/lerobot}"
    fi
    export HF_HOME="${HF_HOME:-${VENV_ROOT}/var/hf_home}"
    export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
    export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_TELEMETRY
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export TOKENIZERS_PARALLELISM=false
    export USE_LIBUV="${USE_LIBUV:-0}"
    # This container exposes an NCCL tuner plugin, but no tuner config path.
    # Disable the optional plugin unless the caller supplies a real config.
    export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-UNUSED}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
}

detect_gpus() {
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local _ngpu
        _ngpu="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -z "${_ngpu}" || "${_ngpu}" -lt 1 ]]; then
            echo "ERROR: nvidia-smi reported no GPUs and CUDA_VISIBLE_DEVICES is unset." >&2
            return 1
        fi
        export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((_ngpu - 1)))"
        PROC_PER_NODE="${PROC_PER_NODE:-${_ngpu}}"
    else
        local _ngpu
        _ngpu="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
        PROC_PER_NODE="${PROC_PER_NODE:-${_ngpu}}"
    fi
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PROC_PER_NODE=${PROC_PER_NODE}"
}

need_file() {
    local p="$1" label="$2"
    if [[ ! -e "${p}" ]]; then
        echo "ERROR: missing ${label}: ${p}" >&2
        return 1
    fi
}

stats_group_name() {
    local repo_id="${1:-robotwin/${TASK_NAME}}"
    "${VENV_ROOT}/bin/python" -c "import hashlib; print('agg_1repos_' + hashlib.sha1('${repo_id}'.encode()).hexdigest()[:10])"
}

print_path_banner() {
    echo "PROJ_ROOT            = ${PROJ_ROOT}"
    echo "VENV_ROOT            = ${VENV_ROOT}"
    echo "HF_HOME              = ${HF_HOME}"
    echo "HF_LEROBOT_HOME      = ${HF_LEROBOT_HOME}"
    echo "ROBOTWIN_CLEAN_ROOT  = ${ROBOTWIN_CLEAN_ROOT}"
    echo "CKPT_BASE            = ${CKPT_BASE}"
    echo "TASK_NAME            = ${TASK_NAME}"
    echo "which python         = $(command -v python)"
}
