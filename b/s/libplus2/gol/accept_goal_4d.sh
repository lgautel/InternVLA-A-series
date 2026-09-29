#!/usr/bin/env bash
# Acceptance entry for the Goal 3D/4D pipeline (gen2).
#
#   bash accept_goal_4d.sh            # offline stages S1-S3 (no dataset needed, ~5-8 min)
#   DATASET=/path/to/goal_4D bash accept_goal_4d.sh   # + S4 verifier, S5 training-path check, S6 judge self-test
#   SKIP_SLOW=1 bash accept_goal_4d.sh                # skip end-to-end conversions in v2 tests
#
# The full 4,243-episode conversion is NOT run here. That run is:
#   python generate_goal_4d.py --dest /B/Dta/LIBERO/libero_plus_goal_lrb3_4D --confirm-full-run --force
# Then re-run this script with DATASET=/B/Dta/LIBERO/libero_plus_goal_lrb3_4D (S5 uses --full).
set -euo pipefail
ROOT="/B/SRC/itvlaGpLibPlus"
PY="${PY:-/B/VENV/itnvla15rbt20/bin/python}"
GOL="$ROOT/b/s/libplus2/gol"
VENV_ROOT="$(dirname "$(dirname "$PY")")"
export HF_HOME="${HF_HOME:-/B/VENV/hf_home}"
# same library path as launch/libplus_sft_launch.sh so torchcodec can load
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${VENV_ROOT}/lib:${VENV_ROOT}/lib/python3.11/site-packages/torch/lib:${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${VENV_ROOT}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:${VENV_ROOT}/lib/python3.11/site-packages/nvidia/npp/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

stage() { echo; echo "=================== $* ==================="; }

stage "S1  derived MJCF is fresh (assets/panda_goal_table.xml)"
"$PY" "$GOL/make_goal_mjcf.py" --check

stage "S1b eval wrapper derives exactly 2 changes and refuses contract-breaking env (no run)"
DRYDIR="$(mktemp -d)"
mkdir -p "$DRYDIR/ds/meta"
"$PY" - "$DRYDIR/ds/meta/goal_train_eval_contract.json" <<'PY'
import sys
import json
json.dump({"schema": "goal_train_eval_contract/1"}, open(sys.argv[1], "w"))
PY
OUT="$(DRY_RUN=1 CKPT_PATH=/tmp/x LIBERO_HOME=/x SERVER_VENV=/x CLIENT_VENV=/x GOAL4D_DATASET="$DRYDIR/ds" \
  EVAL_LOG_DIR="$DRYDIR/log" bash "$GOL/run_eval_goal_plus.sh")"
echo "$OUT"
echo "$OUT" | grep -q 'SUITES="libero_goal"'
echo "$OUT" | grep -q 'eval_goal_plus.py --'
[ "$(echo "$OUT" | grep -c '^[<>] ')" = "4" ]      # exactly two changed lines (old+new each)
if ROTATE_IMAGES=true DRY_RUN=1 CKPT_PATH=/tmp/x LIBERO_HOME=/x SERVER_VENV=/x CLIENT_VENV=/x \
   GOAL4D_DATASET="$DRYDIR/ds" EVAL_LOG_DIR="$DRYDIR/log2" bash "$GOL/run_eval_goal_plus.sh" >/dev/null 2>&1; then
  echo "FAIL: wrapper accepted ROTATE_IMAGES=true"; exit 1
fi
echo "wrapper OK"

stage "S2  gen1 tests (FK, history packing, generator, eval clock)"
"$PY" "$GOL/test_goal_4d.py"

stage "S3  gen2 tests (A-G: FK equivalence, constants, orientation, guards, eval client, import hook, e2e)"
"$PY" "$GOL/test_goal_4d_v2.py"

if [[ -n "${DATASET:-}" ]]; then
  FULL=""
  if [[ "${FULL_CHECK:-1}" == "1" ]] && [[ "$(basename "$DATASET")" != *smoke* ]]; then FULL="--full"; fi
  stage "S4  dataset verifier V01-V18 on $DATASET $FULL"
  CUDA_VISIBLE_DEVICES="" "$PY" "$GOL/verify_goal_dataset.py" --dataset "$DATASET" $FULL --video-backend torchcodec

  stage "S5  real training data path (make_dataset + Extract3DKeypointTransformFn)"
  CUDA_VISIBLE_DEVICES="" "$PY" "$GOL/check_training_path.py" --dataset "$DATASET"

  stage "S6  live-orientation judge self-test (no simulator; the LIVE half needs a client venv, see doc)"
  CUDA_VISIBLE_DEVICES="" "$PY" "$GOL/live_orientation_goal.py" selftest --dataset "$DATASET" \
    --task put_the_bowl_on_the_plate
else
  echo; echo "[skip] S4/S5/S6 need DATASET=<converted goal dataset>"
fi
echo; echo "ACCEPT OK"
