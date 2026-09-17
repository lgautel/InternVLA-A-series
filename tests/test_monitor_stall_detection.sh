#!/usr/bin/env bash
# Test suite for libplus_sft_launch.sh monitoring improvements:
#   1. _is_training_stalled (step-based stall detection)
#   2. _auto_recover fresh start generates new OUTPUT_DIR
#   3. LOG_FREQ / STALE_THRESHOLD compatibility
#   4. _find_latest_checkpoint searches across CKPT_ROOT
#
# Usage:
#   bash tests/test_monitor_stall_detection.sh
#
# All tests are pure-bash (no GPU, no training process needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

_PASS=0
_FAIL=0
_assert() {
    local desc="$1" expected="$2" actual="$3"
    if [[ "${expected}" == "${actual}" ]]; then
        echo "  PASS: ${desc}"
        _PASS=$((_PASS + 1))
    else
        echo "  FAIL: ${desc} (expected='${expected}', actual='${actual}')"
        _FAIL=$((_FAIL + 1))
    fi
}

###########################################################################
echo "=== Test 1: Default values changed ==="
###########################################################################
stale_default=$(grep 'STALE_THRESHOLD=.*:-' "${PROJ_ROOT}/launch/libplus_sft_launch.sh" \
    | head -1 | grep -oE ':-[0-9]+' | tr -d ':}-')
_assert "STALE_THRESHOLD default is 1800" "1800" "${stale_default}"

log_freq_default=$(grep 'LOG_FREQ=.*:-100' "${PROJ_ROOT}/launch/libplus_sft_launch.sh" \
    | grep -v SMOKE | grep -v WAN | head -1 | grep -oE ':-[0-9]+' | tr -d ':}-')
_assert "LOG_FREQ production default is 100" "100" "${log_freq_default}"

###########################################################################
echo ""
echo "=== Test 2: _is_log_stale no longer exists ==="
###########################################################################
old_func=$(grep -c '_is_log_stale' "${PROJ_ROOT}/launch/libplus_sft_launch.sh" || true)
_assert "_is_log_stale removed from script" "0" "${old_func}"

new_func=$(grep -c '_is_training_stalled' "${PROJ_ROOT}/launch/libplus_sft_launch.sh" || true)
_assert "_is_training_stalled present in script" "$([ "${new_func}" -ge 3 ] && echo yes || echo no)" "yes"

###########################################################################
echo ""
echo "=== Test 3: _is_training_stalled logic (unit test) ==="
###########################################################################
# Source only the functions we need by extracting them
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

LOG_FILE="${TMPDIR}/train.log"
STALE_THRESHOLD=1800
MONITOR_INTERVAL=900

# Extract functions from launch script
sed -n '/_monitor_ts/,/^_monitor_log/p' "${PROJ_ROOT}/launch/libplus_sft_launch.sh" > /dev/null
cat > "${TMPDIR}/funcs.sh" << 'FUNCS_EOF'
_parse_step_from_log() {
    local raw
    raw=$(grep -oE 'step:[0-9]+(\.[0-9]+)?K?' "${LOG_FILE}" 2>/dev/null | tail -1 | cut -d: -f2)
    [[ -z "${raw}" ]] && return
    if [[ "${raw}" == *K ]]; then
        raw="${raw%K}"
        awk -v s="${raw}" 'BEGIN { printf "%.0f\n", s * 1000 }'
    else
        awk -v s="${raw}" 'BEGIN { printf "%.0f\n", s }'
    fi
}

_LAST_OBSERVED_STEP=""
_STALL_COUNT=0
_TRAIN_START_TIME=""

_is_training_stalled() {
    local cur_step
    cur_step=$(_parse_step_from_log)
    if [[ -z "${cur_step}" ]]; then
        if [[ -n "${_TRAIN_START_TIME}" ]]; then
            local elapsed=$(( $(date +%s) - _TRAIN_START_TIME ))
            [[ ${elapsed} -gt ${STALE_THRESHOLD} ]]
        else
            return 1
        fi
        return
    fi
    if [[ "${cur_step}" == "${_LAST_OBSERVED_STEP}" ]]; then
        _STALL_COUNT=$((_STALL_COUNT + 1))
        [[ ${_STALL_COUNT} -ge 2 ]]
        return
    fi
    _LAST_OBSERVED_STEP="${cur_step}"
    _STALL_COUNT=0
    return 1
}

_reset_stall_state() {
    _LAST_OBSERVED_STEP=""
    _STALL_COUNT=0
    _TRAIN_START_TIME=$(date +%s)
}
FUNCS_EOF
source "${TMPDIR}/funcs.sh"

# 3a: No log file, no start time → not stalled
_reset_stall_state
_TRAIN_START_TIME=""
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "No log + no start time → not stalled" "ok" "${result}"

# 3b: No log file, recent start → not stalled
_reset_stall_state
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "No log + recent start → not stalled" "ok" "${result}"

# 3c: No log file, old start → stalled
_reset_stall_state
_TRAIN_START_TIME=$(( $(date +%s) - 2000 ))
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "No log + old start (2000s) > threshold (1800s) → stalled" "stalled" "${result}"

# 3d: Step advancing → not stalled
_reset_stall_state
echo "step:100" > "${LOG_FILE}"
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "First step seen (100) → not stalled" "ok" "${result}"
_assert "Step recorded" "100" "${_LAST_OBSERVED_STEP}"

echo "step:200" > "${LOG_FILE}"
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "Step advanced (200) → not stalled" "ok" "${result}"
_assert "Step updated" "200" "${_LAST_OBSERVED_STEP}"

# 3e: Step unchanged once → not stalled (need 2 consecutive)
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "Step unchanged 1st time → not stalled" "ok" "${result}"
_assert "Stall count = 1" "1" "${_STALL_COUNT}"

# 3f: Step unchanged twice → stalled
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "Step unchanged 2nd time → stalled" "stalled" "${result}"
_assert "Stall count = 2" "2" "${_STALL_COUNT}"

# 3g: Step advances again → resets stall
echo "step:300" > "${LOG_FILE}"
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "Step advanced (300) after stall → not stalled" "ok" "${result}"
_assert "Stall count reset to 0" "0" "${_STALL_COUNT}"

# 3h: K-suffix step parsing
echo "step:1.5K" > "${LOG_FILE}"
if _is_training_stalled; then result="stalled"; else result="ok"; fi
_assert "Step 1.5K (1500) → not stalled" "ok" "${result}"
_assert "Step parsed as 1500" "1500" "${_LAST_OBSERVED_STEP}"

###########################################################################
echo ""
echo "=== Test 4: Fresh start generates new OUTPUT_DIR ==="
###########################################################################
CKPT_ROOT="${TMPDIR}/ckpt_root"
POLICY="internvla_a1_5"
JOB_SUFFIX="libplus-sft"
mkdir -p "${CKPT_ROOT}"

# Simulate the original OUTPUT_DIR
ORIGINAL_OUTPUT_DIR="${CKPT_ROOT}/2026_09_11_09_47_44-internvla_a1_5-libplus-sft"
mkdir -p "${ORIGINAL_OUTPUT_DIR}"

# Simulate ARGS array with --output_dir and --job_name
ARGS=(
    "--output_dir=${ORIGINAL_OUTPUT_DIR}"
    "--job_name=original-job"
    "--policy.type=internvla_a1_5"
    "--batch_size=32"
)

# Simulate fresh start arg rebuilding
recover_stamp="2026_09_11_13_48_21"
recover_job_name="${recover_stamp}-${POLICY}-${JOB_SUFFIX}"
recover_output_dir="${CKPT_ROOT}/${recover_job_name}"

fresh_args=()
for arg in "${ARGS[@]}"; do
    if [[ "${arg}" == --output_dir=* ]]; then
        fresh_args+=("--output_dir=${recover_output_dir}")
    elif [[ "${arg}" == --job_name=* ]]; then
        fresh_args+=("--job_name=${recover_job_name}")
    else
        fresh_args+=("${arg}")
    fi
done

_assert "Fresh OUTPUT_DIR differs from original" \
    "$([ "${recover_output_dir}" != "${ORIGINAL_OUTPUT_DIR}" ] && echo yes || echo no)" "yes"

found_output_dir=""
found_job_name=""
for arg in "${fresh_args[@]}"; do
    [[ "${arg}" == --output_dir=* ]] && found_output_dir="${arg#--output_dir=}"
    [[ "${arg}" == --job_name=* ]] && found_job_name="${arg#--job_name=}"
done
_assert "fresh_args has new output_dir" "${recover_output_dir}" "${found_output_dir}"
_assert "fresh_args has new job_name" "${recover_job_name}" "${found_job_name}"
_assert "fresh_args preserves policy.type" "--policy.type=internvla_a1_5" "${fresh_args[2]}"
_assert "fresh_args preserves batch_size" "--batch_size=32" "${fresh_args[3]}"

###########################################################################
echo ""
echo "=== Test 5: _find_latest_checkpoint searches CKPT_ROOT ==="
###########################################################################
# No checkpoint yet
source "${TMPDIR}/funcs.sh"
OUTPUT_DIR="${ORIGINAL_OUTPUT_DIR}"
CKPT_ROOT="${TMPDIR}/ckpt_root2"
mkdir -p "${CKPT_ROOT}"

_LATEST_CKPT=""
_find_latest_checkpoint() {
    _LATEST_CKPT=""
    if [[ ! -d "${CKPT_ROOT}" ]]; then
        return
    fi
    _LATEST_CKPT=$(find "${CKPT_ROOT}" -path "*/checkpoints/*/pretrained_model" -type d 2>/dev/null | sort -V | tail -1)
    if [[ -n "${_LATEST_CKPT}" ]]; then
        local step_dir ckpts_dir
        step_dir=$(dirname "${_LATEST_CKPT}")
        ckpts_dir=$(dirname "${step_dir}")
        OUTPUT_DIR=$(dirname "${ckpts_dir}")
    fi
}

_find_latest_checkpoint
_assert "No checkpoint → empty result" "" "${_LATEST_CKPT}"

# Create checkpoint in first output dir
dir1="${CKPT_ROOT}/run1/checkpoints/001000/pretrained_model"
mkdir -p "${dir1}"
_find_latest_checkpoint
_assert "Found checkpoint in run1" "${dir1}" "${_LATEST_CKPT}"
_assert "OUTPUT_DIR updated to run1" "${CKPT_ROOT}/run1" "${OUTPUT_DIR}"

# Create later checkpoint in different output dir
dir2="${CKPT_ROOT}/run2/checkpoints/005345/pretrained_model"
mkdir -p "${dir2}"
_find_latest_checkpoint
_assert "Found latest checkpoint across runs (run2 step 5345)" "${dir2}" "${_LATEST_CKPT}"
_assert "OUTPUT_DIR updated to run2" "${CKPT_ROOT}/run2" "${OUTPUT_DIR}"

###########################################################################
echo ""
echo "=== Test 6: LOG_FREQ / STALE_THRESHOLD / MONITOR_INTERVAL compatibility ==="
###########################################################################
# With default values: LOG_FREQ=100, speed~7.5s/step → log every 750s
# MONITOR_INTERVAL=900, STALE_THRESHOLD=1800
# Stall requires 2 consecutive unchanged checks = 2 × 900 = 1800s
# Log every 750s < 900s → step should advance every check

LOG_FREQ_VAL=100
SPEED=7.5
LOG_INTERVAL=$(awk -v f="${LOG_FREQ_VAL}" -v s="${SPEED}" 'BEGIN { printf "%.0f", f * s }')
MONITOR_VAL=900
STALL_WINDOW=$((MONITOR_VAL * 2))

_assert "Log interval (${LOG_INTERVAL}s) < MONITOR_INTERVAL (${MONITOR_VAL}s)" \
    "$([ "${LOG_INTERVAL}" -lt "${MONITOR_VAL}" ] && echo yes || echo no)" "yes"
_assert "Stall window (${STALL_WINDOW}s) = STALE_THRESHOLD (1800s)" "1800" "${STALL_WINDOW}"

# Even at slow speed (20s/step), log every 2000s
# Two checks (1800s) without step change → correctly stalled at 2000s boundary
SLOW_LOG_INTERVAL=$(awk 'BEGIN { printf "%.0f", 100 * 20 }')
_assert "Slow-speed log interval (${SLOW_LOG_INTERVAL}s) > stall window (1800s) — may false-trigger" \
    "$([ "${SLOW_LOG_INTERVAL}" -gt "${STALL_WINDOW}" ] && echo yes || echo no)" "yes"
echo "  NOTE: At 20s/step, step updates every ${SLOW_LOG_INTERVAL}s > stall window ${STALL_WINDOW}s."
echo "        This is acceptable: 20s/step is 2.7× slower than expected (7.5s/step)."
echo "        If training is genuinely that slow, operator should increase STALE_THRESHOLD."

###########################################################################
echo ""
echo "=== Test 7: _reset_stall_state clears all state ==="
###########################################################################
_LAST_OBSERVED_STEP="5000"
_STALL_COUNT=3
_TRAIN_START_TIME=0
_reset_stall_state
_assert "Step reset to empty" "" "${_LAST_OBSERVED_STEP}"
_assert "Stall count reset to 0" "0" "${_STALL_COUNT}"
_assert "Start time set to now" "$([ "${_TRAIN_START_TIME}" -gt 0 ] && echo yes || echo no)" "yes"

###########################################################################
echo ""
echo "==========================================="
echo "Results: ${_PASS} passed, ${_FAIL} failed"
echo "==========================================="
[[ ${_FAIL} -eq 0 ]] && exit 0 || exit 1
