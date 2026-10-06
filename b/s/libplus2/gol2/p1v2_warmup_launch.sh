#!/usr/bin/env bash
set -euo pipefail
# Phase 1 Warmup — v2 (14D state) variant
# Overrides EXPR_NAME, DATA_SRC, DATA_REPO_ID then delegates to gol/p1_warmup_launch.sh.
# All other parameters (LR, steps, loss weights, augmentation, keypoint config) inherit.
#
# p1_warmup_launch.sh uses ${VAR:-default} for these three variables, so export works.
#
# Usage:
#   bash b/s/libplus2/gol2/p1v2_warmup_launch.sh           # production 8 GPU
#   SMOKE=1 bash b/s/libplus2/gol2/p1v2_warmup_launch.sh   # 1 GPU smoke test

export EXPR_NAME="${EXPR_NAME:-4dwvlaLbPlusGolV2_1001}"
export DATA_SRC="${DATA_SRC:-/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2}"
export DATA_REPO_ID="${DATA_REPO_ID:-libero_plus_goal_lrb3_4Dv2}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/../gol/p1_warmup_launch.sh" "$@"
