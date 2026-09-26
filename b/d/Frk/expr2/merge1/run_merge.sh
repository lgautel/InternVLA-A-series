#!/bin/bash
# Weighted merge of InternVLA-A1.5 checkpoints with smoke test
#
# Usage: bash run_merge.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Checkpoint paths (from user requirement)
CKPT1="/home/nvidia/bt/ckp/4wvlaFrk/plug/itvlagpFrkPlug0907Dtaug010000/"
CKPT2="/home/nvidia/bt/ckp/4wvlaFrk/plug/4wvlaFrkPlugCkp010420/"
OUTPUT="/home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w03pure1w07/"

# Weights (0.3 for augmented, 0.7 for non-augmented)
WEIGHT1=0.3
WEIGHT2=0.7

echo "=========================================="
echo "InternVLA-A1.5 Checkpoint Weighted Merge"
echo "=========================================="
echo ""
echo "Checkpoint 1 (augmented, 10k steps): $CKPT1"
echo "  Weight: $WEIGHT1"
echo ""
echo "Checkpoint 2 (non-augmented): $CKPT2"
echo "  Weight: $WEIGHT2"
echo ""
echo "Output: $OUTPUT"
echo ""

# Verify input checkpoints exist
if [[ ! -d "$CKPT1" ]]; then
    echo "ERROR: Checkpoint 1 not found at $CKPT1"
    exit 1
fi

if [[ ! -d "$CKPT2" ]]; then
    echo "ERROR: Checkpoint 2 not found at $CKPT2"
    exit 1
fi

# Run merge
echo "=========================================="
echo "Step 1: Merging checkpoints"
echo "=========================================="
python3 "${SCRIPT_DIR}/merge_checkpoints.py" \
    --ckpt1 "$CKPT1" --weight1 $WEIGHT1 \
    --ckpt2 "$CKPT2" --weight2 $WEIGHT2 \
    --output "$OUTPUT"

if [[ $? -ne 0 ]]; then
    echo ""
    echo "ERROR: Merge failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "SUCCESS: Merge completed"
echo "=========================================="
echo "Merged checkpoint saved to: $OUTPUT"
echo ""
echo "To run smoke test, use the internvla_a1_5 conda environment:"
echo "  bash run_smoke_test_in_env.sh"
echo "  OR"
echo "  conda activate internvla_a1_5"
echo "  python3 smoke_test.py --checkpoint $OUTPUT"
