#!/bin/bash
# Run smoke test in the internvla_a1_5 conda environment
#
# Usage: bash run_smoke_test_in_env.sh [checkpoint_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${1:-/home/nvidia/bt/ckp/4wvlaFrk/plug/aug1w03pure1w07/}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH"
    exit 1
fi

# Check if internvla_a1_5 environment exists
if ! conda env list | grep -q "internvla_a1_5"; then
    echo "ERROR: internvla_a1_5 conda environment not found"
    echo "Available environments:"
    conda env list
    echo ""
    echo "Please create the environment first:"
    echo "  conda create -y -n internvla_a1_5 python=3.11"
    echo "  conda activate internvla_a1_5"
    echo "  # Install dependencies as per CLAUDE.md"
    exit 1
fi

echo "=========================================="
echo "Running smoke test in internvla_a1_5 env"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""

# Activate environment and run test
eval "$(conda shell.bash hook)"
conda activate internvla_a1_5

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python3)"
echo ""

python3 "${SCRIPT_DIR}/smoke_test.py" --checkpoint "$CHECKPOINT"
