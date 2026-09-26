#!/bin/bash
# Weighted merge of 3 InternVLA-A1.5 checkpoints with smoke test
#
# Usage: bash run_merge_3way.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Checkpoint paths (from user requirement)
CKPT1="/home/nvidia/bt/ckp/4wvlaFrk/plug/itvlagpFrkPlug0907Dtaug010000/"
CKPT2="/home/nvidia/bt/ckp/4wvlaFrk/plug/itvlagpFrkPlug0907_dtaug2_5k/"
CKPT3="/home/nvidia/bt/ckp/4wvlaFrk/plug/4wvlaFrkPlugCkp010420/"
OUTPUT="/home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w01augtw5k02pure1w07/"

# Weights
WEIGHT1=0.1  # Data augmentation, 10k steps
WEIGHT2=0.2  # Affine data augmentation, 5k steps
WEIGHT3=0.7  # No augmentation

echo "=========================================="
echo "InternVLA-A1.5 3-Way Checkpoint Merge"
echo "=========================================="
echo ""
echo "Checkpoint 1 (data aug, 10k steps): $CKPT1"
echo "  Weight: $WEIGHT1"
echo ""
echo "Checkpoint 2 (affine aug, 5k steps): $CKPT2"
echo "  Weight: $WEIGHT2"
echo ""
echo "Checkpoint 3 (no aug): $CKPT3"
echo "  Weight: $WEIGHT3"
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

if [[ ! -d "$CKPT3" ]]; then
    echo "ERROR: Checkpoint 3 not found at $CKPT3"
    exit 1
fi

# Run merge
echo "=========================================="
echo "Step 1: Merging 3 checkpoints"
echo "=========================================="
python3 "${SCRIPT_DIR}/merge_checkpoints_multi.py" \
    --ckpt "$CKPT1" --weight $WEIGHT1 \
    --ckpt "$CKPT2" --weight $WEIGHT2 \
    --ckpt "$CKPT3" --weight $WEIGHT3 \
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
echo "  bash run_smoke_test_in_env.sh $OUTPUT"
echo "  OR"
echo "  conda activate internvla_a1_5"
echo "  python3 smoke_test.py --checkpoint $OUTPUT"
