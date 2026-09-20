#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# End-to-end LIBERO 3D/4D keypoint regeneration pipeline (v2).
#
# Usage:
#   bash b/s/libplus/run_pipeline.sh
#
# Environment overrides:
#   SOURCE_DATASET   - Source LeRobot v3 dataset (must have joint_position)
#   HF_LEROBOT_HOME  - Where to store per-suite splits
#   DEST_MERGED      - Output merged dataset with keypoints
#   MJCF_PATH        - Path to (or where to export) the Lift MJCF
#   SKIP_MJCF=1      - Reuse existing MJCF
#   SKIP_KEYPOINTS=1 - Reuse existing keypoints
#   SKIP_SPLIT=1     - Reuse existing splits
#   SKIP_VERIFY=1    - Skip verification
#   SMOKE=1          - FK on first parquet only (fast sanity check)
#   FORCE=1          - Overwrite existing outputs
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Configuration ────────────────────────────────────────────────────────────
SOURCE_DATASET="${SOURCE_DATASET:-/B/Dta/libero_merged_lerobot_v3}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/B/VENV/hf_home/lerobot}"
DEST_MERGED="${DEST_MERGED:-/B/Dta/libero_merged_kpt_v2}"
MJCF_PATH="${MJCF_PATH:-${PROJ_ROOT}/evaluation/panda_robosuite_lift.xml}"

VLA_PYTHON="${VLA_PYTHON:-/B/VENV/itnvla15rbt20/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/B/VENV/libero_plus_client/bin/python}"

SKIP_MJCF="${SKIP_MJCF:-0}"
SKIP_KEYPOINTS="${SKIP_KEYPOINTS:-0}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"
SKIP_VERIFY="${SKIP_VERIFY:-0}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"

echo "=== LIBERO 3D/4D Keypoint Generation Pipeline (v2) ==="
echo "PROJ_ROOT=${PROJ_ROOT}"
echo "SOURCE_DATASET=${SOURCE_DATASET}"
echo "DEST_MERGED=${DEST_MERGED}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "MJCF_PATH=${MJCF_PATH}"
echo ""

if [[ ! -d "${SOURCE_DATASET}" ]]; then
    echo "ERROR: SOURCE_DATASET not found: ${SOURCE_DATASET}" >&2
    echo "This should be the LeRobot v3 converted LIBERO dataset" >&2
    echo "(with observation.state.joint_position column)." >&2
    exit 1
fi

cd "${PROJ_ROOT}"

# ── Step 0: Export Lift MJCF ─────────────────────────────────────────────
if [[ "${SKIP_MJCF}" != "1" ]]; then
    echo ">>> Step 0/4: Export Lift MJCF"
    if [[ -f "${MJCF_PATH}" ]]; then
        echo "  MJCF already exists: ${MJCF_PATH}"
        if [[ "${FORCE}" != "1" ]]; then
            echo "  Using existing MJCF (pass FORCE=1 to re-export)"
        else
            echo "  Re-exporting (FORCE=1)"
            MUJOCO_GL=egl "${LIBERO_PYTHON}" b/s/libplus/export_lift_mjcf.py \
                --output "${MJCF_PATH}"
        fi
    else
        MUJOCO_GL=egl "${LIBERO_PYTHON}" b/s/libplus/export_lift_mjcf.py \
            --output "${MJCF_PATH}"
    fi
else
    echo ">>> Step 0/4: SKIP (MJCF)"
    if [[ ! -f "${MJCF_PATH}" ]]; then
        echo "ERROR: MJCF not found: ${MJCF_PATH}" >&2
        exit 1
    fi
fi

# ── Step 1-2: Generate keypoints ─────────────────────────────────────────
if [[ "${SKIP_KEYPOINTS}" != "1" ]]; then
    echo ""
    echo ">>> Step 1-2/4: Generate FK keypoints"

    if [[ "${SMOKE}" == "1" ]]; then
        echo "  SMOKE mode: testing FK on first parquet only"
        "${VLA_PYTHON}" b/s/libplus/smoke_test.py \
            --source "${SOURCE_DATASET}" \
            --xml "${MJCF_PATH}"
        echo "  SMOKE passed. Run without SMOKE=1 for full generation."
        exit 0
    fi

    KPT_ARGS=(
        --source "${SOURCE_DATASET}"
        --dest "${DEST_MERGED}"
        --xml "${MJCF_PATH}"
    )
    [[ "${FORCE}" == "1" ]] && KPT_ARGS+=(--force)

    "${VLA_PYTHON}" b/s/libplus/generate_libero_kpt_v2.py "${KPT_ARGS[@]}"
else
    echo ">>> Step 1-2/4: SKIP (keypoints)"
    if [[ ! -d "${DEST_MERGED}" ]]; then
        echo "ERROR: Destination not found: ${DEST_MERGED}" >&2
        exit 1
    fi
fi

# ── Step 3: Split into 4 suites ─────────────────────────────────────────
if [[ "${SKIP_SPLIT}" != "1" ]]; then
    echo ""
    echo ">>> Step 3/4: Split into 4 LIBERO suites"
    SPLIT_ARGS=(
        --source "${DEST_MERGED}"
        --repo-id "local/libero_merged_kpt_v2"
        --output-dir "${HF_LEROBOT_HOME}"
    )
    [[ "${FORCE}" == "1" ]] && SPLIT_ARGS+=(--force)

    "${VLA_PYTHON}" util_scripts/split_libero_merged.py "${SPLIT_ARGS[@]}"
else
    echo ">>> Step 3/4: SKIP (split)"
fi

# ── Step 4: Verification ────────────────────────────────────────────────
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    echo ""
    echo ">>> Step 4/4: Verification"

    echo "  4a: Keypoint generation verification"
    "${VLA_PYTHON}" b/s/libplus/verify_kpt_generation.py \
        --dataset "${DEST_MERGED}" \
        --max-files 20

    echo ""
    echo "  4b: Train-eval consistency verification"
    "${VLA_PYTHON}" b/s/libplus/verify_train_eval_consistency.py \
        --dataset "${DEST_MERGED}" \
        --eval-xml "${MJCF_PATH}"
else
    echo ">>> Step 4/4: SKIP (verification)"
fi

echo ""
echo "=== Pipeline Complete ==="
echo "Merged dataset: ${DEST_MERGED}"
echo "Suite datasets: ${HF_LEROBOT_HOME}/libero_{spatial,object,goal,10}"
echo ""
echo "Next steps:"
echo "  1. Update training script: kpt_4d_mode=pos_rot, keypoint_history_max_len=200"
echo "  2. Update loss function: antipodal_mse for rotation components"
echo "  3. Run training"
echo "  4. Evaluate with LIBERO2 pipeline"
