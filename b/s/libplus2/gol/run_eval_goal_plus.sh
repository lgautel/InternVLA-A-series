#!/usr/bin/env bash
# Run the existing LIBERO-plus2 evaluation on libero_goal with the Goal 4D client.
#
# evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh is NOT edited. This wrapper derives a
# private copy of it (into $EVAL_LOG_DIR) with exactly two textual changes, and refuses to
# run if any of them does not apply exactly once (so an upstream change cannot silently
# make it evaluate with the wrong client):
#   1. `python evaluation/LIBERO-plus2/eval_libero_plus.py`  ->  `python <this dir>/eval_goal_plus.py --`
#   2. SUITES="libero_spatial libero_object libero_goal libero_10"  ->  SUITES="libero_goal"
#   (ROTATE_IMAGES stays false, STATS_KEY_MODE/ROBOT_TYPE_MODE stay panda, both enforced below)
#
# Required (same as the original script): CKPT_PATH LIBERO_HOME SERVER_VENV CLIENT_VENV
# Required (new):  GOAL4D_DATASET=<converted goal dataset>   (its meta/goal_train_eval_contract.json is used)
# Useful:          GPU_IDS  RENDER_BACKEND=egl|osmesa|auto  CATEGORIES="Language ..."  EVAL_LOG_DIR
#                  DRY_RUN=1  -> write and check the derived script, run nothing
#
# Forced (not overridable, they are part of the train/eval contract):
#   ROTATE_IMAGES=false  STATS_KEY_MODE=panda  ROBOT_TYPE_MODE=panda  INFERENCE_BACKEND=standard
#   DISABLE_KEYPOINTS unset (keypoints on)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="${PROJ:-$(cd "${HERE}/../../../.." && pwd)}"
ORIG="${PROJ}/evaluation/LIBERO-plus2/run_eval_libero_plus_venv.sh"

: "${GOAL4D_DATASET:?ERROR: GOAL4D_DATASET not set (converted Goal dataset directory)}"
CONTRACT="${GOAL4D_DATASET}/meta/goal_train_eval_contract.json"
[ -f "${CONTRACT}" ] || { echo "FATAL: ${CONTRACT} missing"; exit 2; }
[ -f "${ORIG}" ] || { echo "FATAL: ${ORIG} missing"; exit 2; }

CONTRACT_SCHEMA=$(python3 -c "import json; print(json.load(open('${CONTRACT}'))['schema'])")
if [ "${CONTRACT_SCHEMA}" = "goal_train_eval_contract/2" ]; then
  GOAL_ROBOT_TYPE="libero_goal_4dv2"
else
  GOAL_ROBOT_TYPE="panda"
fi

if [ "${ROTATE_IMAGES:-false}" != "false" ] || [ -n "${DISABLE_KEYPOINTS:-}" ] \
   || [ "${STATS_KEY_MODE:-${GOAL_ROBOT_TYPE}}" != "${GOAL_ROBOT_TYPE}" ] \
   || [ "${ROBOT_TYPE_MODE:-${GOAL_ROBOT_TYPE}}" != "${GOAL_ROBOT_TYPE}" ] \
   || [ "${INFERENCE_BACKEND:-standard}" != "standard" ]; then
  echo "FATAL: ROTATE_IMAGES/DISABLE_KEYPOINTS/STATS_KEY_MODE/ROBOT_TYPE_MODE/INFERENCE_BACKEND are fixed by the Goal contract"
  exit 2
fi

: "${CKPT_PATH:?ERROR: CKPT_PATH not set}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/B/Log/${EXPR_NAME:-4dwvlaLbPlusGol0929}/$(date +%Y%m%d_%H%M%S)_eval}"
mkdir -p "${EVAL_LOG_DIR}"
DERIVED="${EVAL_LOG_DIR}/run_eval_goal_plus.derived.sh"

count() { grep -cF -- "$1" "${ORIG}" || true; }
OLD_CALL='python evaluation/LIBERO-plus2/eval_libero_plus.py'
OLD_SUITES='SUITES="libero_spatial libero_object libero_goal libero_10"'
[ "$(count "${OLD_CALL}")" = "1" ]   || { echo "FATAL: expected exactly one '${OLD_CALL}' in ${ORIG}"; exit 3; }
[ "$(count "${OLD_SUITES}")" = "1" ] || { echo "FATAL: expected exactly one SUITES line in ${ORIG}"; exit 3; }

NEW_CALL="python ${HERE}/eval_goal_plus.py --"
python3 - "$ORIG" "$DERIVED" "$OLD_CALL" "$NEW_CALL" "$OLD_SUITES" 'SUITES="libero_goal"' <<'PY'
import sys
src, dst, a, b, c, d = sys.argv[1:]
s = open(src).read()
assert s.count(a) == 1 and s.count(c) == 1
s = s.replace(a, b).replace(c, d)
open(dst, "w").write(s)
PY
chmod +x "${DERIVED}"
# the derived script computes PROJ from its own location unless PROJ is exported
export PROJ EVAL_LOG_DIR
export GOAL4D_CONTRACT="${CONTRACT}"
export ROTATE_IMAGES=false STATS_KEY_MODE="${GOAL_ROBOT_TYPE}" ROBOT_TYPE_MODE="${GOAL_ROBOT_TYPE}" INFERENCE_BACKEND=standard
unset DISABLE_KEYPOINTS

echo "derived: ${DERIVED}"
echo "contract: ${GOAL4D_CONTRACT}"
diff <(cat "${ORIG}") "${DERIVED}" || true   # diff exits 1 when files differ; that is expected

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1: not executing"
  exit 0
fi
exec bash "${DERIVED}"
