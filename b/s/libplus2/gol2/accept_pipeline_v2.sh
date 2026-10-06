#!/usr/bin/env bash
set -euo pipefail
# Acceptance for v2 pipeline scripts.
# Validates that all scripts exist, have correct syntax, and point to v2 data.
#
# Usage:
#   bash b/s/libplus2/gol2/accept_pipeline_v2.sh
#   CKPT_PATH=<v2_ckpt> bash b/s/libplus2/gol2/accept_pipeline_v2.sh

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SCRIPT_DIR="b/s/libplus2/gol2"
GOL_DIR="b/s/libplus2/gol"
V2_DATA="/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2"
PYTHON="${PYTHON:-python3}"

PASS=0; FAIL=0; SKIP=0
pass_() { PASS=$((PASS+1)); echo "  OK  $1"; }
fail_() { FAIL=$((FAIL+1)); echo "  FAIL $1"; }
skip_() { SKIP=$((SKIP+1)); echo "  SKIP $1 ($2)"; }

echo "============================================"
echo " Pipeline v2 Acceptance"
echo "============================================"

# --- A1: Script existence ---
echo ""
echo "=== A1: Script existence ==="
for f in p1v2_warmup_launch.sh p2v2_sft_launch.sh eval_v2_wrapper.sh preflight_v2.sh test_pipeline_v2.py accept_pipeline_v2.sh; do
    if [ -f "${SCRIPT_DIR}/${f}" ]; then pass_ "A1: ${f}"; else fail_ "A1: ${f} missing"; fi
done

# --- A2: Bash syntax ---
echo ""
echo "=== A2: Bash syntax ==="
for f in p1v2_warmup_launch.sh p2v2_sft_launch.sh eval_v2_wrapper.sh preflight_v2.sh accept_pipeline_v2.sh; do
    if bash -n "${SCRIPT_DIR}/${f}" 2>/dev/null; then pass_ "A2: ${f} syntax"; else fail_ "A2: ${f} syntax error"; fi
done

# --- A3: v2 identity in scripts ---
echo ""
echo "=== A3: v2 identity ==="
for f in p1v2_warmup_launch.sh p2v2_sft_launch.sh eval_v2_wrapper.sh; do
    if grep -q "4dwvlaLbPlusGolV2_1001" "${SCRIPT_DIR}/${f}" 2>/dev/null; then
        pass_ "A3: ${f} has EXPR_NAME"
    else
        fail_ "A3: ${f} missing EXPR_NAME"
    fi
done
for f in p1v2_warmup_launch.sh p2v2_sft_launch.sh; do
    if grep -q "libero_plus_goal_lrb3_4Dv2" "${SCRIPT_DIR}/${f}" 2>/dev/null; then
        pass_ "A3: ${f} has v2 data"
    else
        fail_ "A3: ${f} missing v2 data"
    fi
done

# --- A4: Preflight v2 content ---
echo ""
echo "=== A4: Preflight v2 ==="
if grep -q "goal_train_eval_contract/2" "${SCRIPT_DIR}/preflight_v2.sh" 2>/dev/null; then
    pass_ "A4: preflight checks schema 2"
else
    fail_ "A4: preflight missing schema 2 check"
fi
if grep -q "state_dim" "${SCRIPT_DIR}/preflight_v2.sh" 2>/dev/null; then
    pass_ "A4: preflight checks state_dim"
else
    fail_ "A4: preflight missing state_dim check"
fi

# --- A5: Unit tests ---
echo ""
echo "=== A5: Unit tests ==="
if $PYTHON -m pytest "${SCRIPT_DIR}/test_pipeline_v2.py" -v --tb=short 2>&1; then
    pass_ "A5: test_pipeline_v2.py"
else
    fail_ "A5: test_pipeline_v2.py"
fi

# --- A6: v2 data checks ---
echo ""
echo "=== A6: v2 data ==="
if [ -d "${V2_DATA}" ]; then
    $PYTHON -c "
import json
info = json.load(open('${V2_DATA}/meta/info.json'))
assert info['robot_type'] == 'libero_goal_4dv2'
assert info['features']['observation.state']['shape'] == [14]
assert info['total_frames'] == 512604

stats = json.load(open('${V2_DATA}/meta/stats.json'))
assert len(stats['observation.state']['mean']) == 14
assert len(stats['action']['mean']) == 7

contract = json.load(open('${V2_DATA}/meta/goal_train_eval_contract.json'))
assert contract['schema'] == 'goal_train_eval_contract/2'
assert contract['state_dim'] == 14
print('v2 data validated')
" 2>&1 && pass_ "A6: v2 data identity" || fail_ "A6: v2 data identity"
else
    skip_ "A6: v2 data" "directory not found"
fi

# --- A7: gol/ dependencies ---
echo ""
echo "=== A7: gol/ dependencies ==="
for f in p1_warmup_launch.sh p2_sft_launch.sh eval_goal_wrapper.sh goal_client.py run_eval_goal_plus.sh; do
    if [ -f "${GOL_DIR}/${f}" ]; then pass_ "A7: ${f}"; else fail_ "A7: ${f} missing"; fi
done
if grep -q "goal_train_eval_contract/2" "${GOL_DIR}/goal_client.py" 2>/dev/null; then
    pass_ "A7: goal_client.py v2 branch"
else
    fail_ "A7: goal_client.py missing v2 branch"
fi

# --- Summary ---
echo ""
echo "============================================"
echo " Result: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "============================================"
if [ "$FAIL" -gt 0 ]; then echo "ACCEPTANCE FAILED"; exit 1; fi
echo "ACCEPTANCE PASSED"
