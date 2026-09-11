#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# End-to-end LIBERO GeoP data pipeline:
#   1. Export Panda MJCF from robosuite
#   2. Generate observation.keypoint_3d (7D pos+quat) via MuJoCo FK
#   3. Split merged dataset into 4 LIBERO suites
#   4. Optionally symlink suites into data/ for training scripts
#
# Usage:
#   bash util_scripts/setup_libero_geop.sh
#
# Environment overrides:
#   SOURCE_DATASET=/tmp/zwy/libero_lerobot_v3
#   HF_LEROBOT_HOME=/tmp/zwy
#   SKIP_MJCF=1          # reuse existing /tmp/zwy/panda_robosuite_full.xml
#   SKIP_KEYPOINTS=1     # reuse libero_merged_kpt
#   SKIP_SPLIT=1         # reuse 4-suite split
#   SKIP_SYMLINK=1       # do not create data/ symlinks
#   SMOKE=1              # FK on first parquet only (fast sanity check)
#   FORCE=1              # overwrite merged_kpt + re-split suite dirs
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_DATASET="${SOURCE_DATASET:-/tmp/zwy/libero_lerobot_v3}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/tmp/zwy}"
MERGED_KPT="${MERGED_KPT:-${HF_LEROBOT_HOME}/libero_merged_kpt}"
MJCF_PATH="${MJCF_PATH:-/tmp/zwy/panda_robosuite_full.xml}"
PYTHON="${PYTHON:-python}"

SKIP_MJCF="${SKIP_MJCF:-0}"
SKIP_KEYPOINTS="${SKIP_KEYPOINTS:-0}"
SKIP_SPLIT="${SKIP_SPLIT:-0}"
SKIP_SYMLINK="${SKIP_SYMLINK:-0}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"

echo "=== LIBERO GeoP Data Setup ==="
echo "PROJ_ROOT=${PROJ_ROOT}"
echo "SOURCE_DATASET=${SOURCE_DATASET}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "MERGED_KPT=${MERGED_KPT}"
echo "MJCF_PATH=${MJCF_PATH}"
echo "SMOKE=${SMOKE}"

if [[ ! -d "${SOURCE_DATASET}" ]]; then
  echo "ERROR: SOURCE_DATASET not found: ${SOURCE_DATASET}" >&2
  exit 1
fi

cd "${PROJ_ROOT}"

# ── Step 1: Export Panda MJCF ─────────────────────────────────────────────
if [[ "${SKIP_MJCF}" != "1" ]]; then
  echo ""
  echo ">>> Step 1/4: Export Panda MJCF"
  MUJOCO_GL=egl "${PYTHON}" util_scripts/export_panda_mjcf.py --output "${MJCF_PATH}"
else
  echo ">>> Step 1/4: SKIP (MJCF)"
  if [[ ! -f "${MJCF_PATH}" ]]; then
    echo "ERROR: MJCF not found at ${MJCF_PATH}" >&2
    exit 1
  fi
fi

# ── Step 2: Generate keypoints ──────────────────────────────────────────────
if [[ "${SKIP_KEYPOINTS}" != "1" ]]; then
  echo ""
  echo ">>> Step 2/4: Generate observation.keypoint_3d"
  KPT_ARGS=(
    --source "${SOURCE_DATASET}"
    --dest "${MERGED_KPT}"
    --xml "${MJCF_PATH}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    KPT_ARGS+=(--force)
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    echo "SMOKE mode: testing FK on one parquet file only"
    "${PYTHON}" util_scripts/smoke_test_libero_fk.py \
      --source "${SOURCE_DATASET}" \
      --xml "${MJCF_PATH}"
    echo "SMOKE FK passed. Run without SMOKE=1 to generate full keypoints."
    exit 0
  fi
  "${PYTHON}" util_scripts/generate_libero_keypoints.py "${KPT_ARGS[@]}"
else
  echo ">>> Step 2/4: SKIP (keypoints)"
  if [[ ! -d "${MERGED_KPT}" ]]; then
    echo "ERROR: MERGED_KPT not found: ${MERGED_KPT}" >&2
    exit 1
  fi
fi

# ── Step 3: Split into 4 suites ───────────────────────────────────────────────
if [[ "${SKIP_SPLIT}" != "1" ]]; then
  echo ""
  echo ">>> Step 3/4: Split merged dataset into 4 LIBERO suites"
  SPLIT_ARGS=(
    --source "${MERGED_KPT}"
    --repo-id "local/libero_merged_kpt"
    --output-dir "${HF_LEROBOT_HOME}"
  )
  if [[ "${FORCE}" == "1" ]]; then
    SPLIT_ARGS+=(--force)
  fi
  "${PYTHON}" util_scripts/split_libero_merged.py "${SPLIT_ARGS[@]}"
else
  echo ">>> Step 3/4: SKIP (split)"
fi

# ── Step 4: Symlink into data/ for convenience ────────────────────────────────
if [[ "${SKIP_SYMLINK}" != "1" ]]; then
  echo ""
  echo ">>> Step 4/4: Symlink suites into ${PROJ_ROOT}/data/"
  mkdir -p "${PROJ_ROOT}/data"
  for suite in libero_spatial libero_object libero_goal libero_10; do
    src="${HF_LEROBOT_HOME}/${suite}"
    dst="${PROJ_ROOT}/data/${suite}"
    if [[ ! -d "${src}" ]]; then
      echo "WARNING: ${src} not found, skipping symlink"
      continue
    fi
    if [[ -L "${dst}" || -e "${dst}" ]]; then
      echo "  ${dst} already exists, skipping"
    else
      ln -s "${src}" "${dst}"
      echo "  linked ${dst} -> ${src}"
    fi
  done
else
  echo ">>> Step 4/4: SKIP (symlinks)"
fi

echo ""
echo "=== Setup complete ==="
echo "Datasets ready under: ${HF_LEROBOT_HOME}/libero_{spatial,object,goal,10}"
echo ""
echo "Next steps:"
echo "  # Phase 1 warmup"
echo "  bash launch/internvla_a15_geop_phase1_libero_warmup.sh"
echo ""
echo "  # Phase 2 SFT (after warmup)"
echo "  export WARMUP_CKPT=outputs/internvla_a1_5/<warmup_job>/checkpoints/000400/pretrained_model"
echo "  bash launch/internvla_a15_finetune_libero_geop.sh"
