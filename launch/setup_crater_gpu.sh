#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# GPU node quick setup for R1 Pro training on Crater
#
# Run this ONCE after GPU node starts. Takes ~5-10 min (mostly pip install).
# After this, run the training launch scripts directly.
#
# Prerequisites (done on CPU node beforehand):
#   - Code at /home/a26215/InternVLA-A/
#   - Dataset at /home/a26215/openpi-datasets/open0630_mj_clean_kpt16
#   - Weights at /home/a26215/InternVLA-A/pretrained_weights/
#   - norm_stats at /home/a26215/InternVLA-A/assets/norm_stats/abs/stats.json
###############################################################################

PROJ_ROOT="/home/a26215/InternVLA-A"
VENV_ROOT="/tmp/itnvla15_r1pro"
HF_HOME_DIR="${VENV_ROOT}/var/hf_home"
HF_LEROBOT_HOME="${HF_HOME_DIR}/lerobot"

echo "=== Step 1: Create venv ==="
if [ ! -f "${VENV_ROOT}/bin/python" ]; then
    python3 -m venv "${VENV_ROOT}" --system-site-packages
    echo "Created venv at ${VENV_ROOT}"
else
    echo "Venv already exists at ${VENV_ROOT}"
fi

echo "=== Step 2: Install package ==="
${VENV_ROOT}/bin/pip install -e "${PROJ_ROOT}" 2>&1 | tail -5

echo "=== Step 3: Install CUDA-specific packages ==="
${VENV_ROOT}/bin/pip install --force-reinstall "torchcodec" \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
${VENV_ROOT}/bin/pip install nvidia-npp-cu12 2>&1 | tail -2

echo "=== Step 4: Patch Transformers ==="
TRANSFORMERS_DIR=$(${VENV_ROOT}/bin/python -c "import transformers; print(transformers.__file__.rsplit('/',1)[0])")
for policy_dir in pi0 pi05 internvla_a1_5; do
    src="${PROJ_ROOT}/src/lerobot/policies/${policy_dir}/transformers_replace/models"
    if [ -d "${src}" ]; then
        cp -r "${src}" "${TRANSFORMERS_DIR}/"
        echo "  Patched ${policy_dir}"
    fi
done

echo "=== Step 5: Setup directories ==="
mkdir -p "${HF_HOME_DIR}/ckpts"
mkdir -p "${HF_HOME_DIR}/hub"
mkdir -p "${HF_LEROBOT_HOME}"

# Symlink dataset
ln -sfn /home/a26215/openpi-datasets/open0630_mj_clean_kpt16 \
    "${HF_LEROBOT_HOME}/open0630_mj_clean_kpt16"

# Symlink weights
ln -sfn "${PROJ_ROOT}/pretrained_weights/InternVLA-A1.5-base" \
    "${HF_HOME_DIR}/ckpts/InternVLA-A1.5-base"

# GeoPredict checkpoint (if exists)
if [ -f "${PROJ_ROOT}/pretrained_weights/GeoPredict_robocasa.pth" ]; then
    ln -sfn "${PROJ_ROOT}/pretrained_weights/GeoPredict_robocasa.pth" \
        "${HF_HOME_DIR}/ckpts/GeoPredict_robocasa.pth"
fi

# Qwen3.5-2B (if pre-downloaded)
if [ -d "${PROJ_ROOT}/pretrained_weights/Qwen3.5-2B" ]; then
    mkdir -p "${HF_HOME_DIR}/hub/models--Qwen--Qwen3.5-2B"
    ln -sfn "${PROJ_ROOT}/pretrained_weights/Qwen3.5-2B" \
        "${HF_HOME_DIR}/hub/models--Qwen--Qwen3.5-2B/snapshots/latest"
fi

# WAN2.2 (if pre-downloaded)
if [ -d "${PROJ_ROOT}/pretrained_weights/Wan2.2-TI2V-5B" ]; then
    ln -sfn "${PROJ_ROOT}/pretrained_weights/Wan2.2-TI2V-5B" \
        "${HF_HOME_DIR}/hub/Wan2.2-TI2V-5B"
fi

# norm_stats symlink
mkdir -p "${HF_LEROBOT_HOME}/stats/abs/open0630_mj_clean_kpt16"
ln -sfn "${PROJ_ROOT}/assets/norm_stats/abs/stats.json" \
    "${HF_LEROBOT_HOME}/stats/abs/open0630_mj_clean_kpt16/stats.json"

echo "=== Step 6: Verify ==="
echo "  Python: $(${VENV_ROOT}/bin/python --version)"
echo "  Torch:  $(${VENV_ROOT}/bin/python -c 'import torch; print(torch.__version__)')"
echo "  CUDA:   $(${VENV_ROOT}/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())')"
${VENV_ROOT}/bin/python -c "
import pinocchio; print('  Pinocchio:', pinocchio.__version__)
from lerobot.dataset_schemas import get_schema
s = get_schema('r1_pro')
print('  Schema r1_pro: state_keys=%s action_keys=%s' % (s.get_state_keys(), s.get_action_keys()))
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Environment variables for training:"
echo "  export VENV_ROOT=${VENV_ROOT}"
echo "  export HF_HOME=${HF_HOME_DIR}"
echo "  export HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "  export PYTHON=${VENV_ROOT}/bin/python"
echo ""
echo "Next steps:"
echo "  1. Smoke test:  SMOKE=1 bash launch/internvla_a15_r1pro_geop_phase1.sh"
echo "  2. Baseline:    bash launch/internvla_a15_r1pro_baseline.sh"
echo "  3. Phase 1:     bash launch/internvla_a15_r1pro_geop_phase1.sh"
echo "  4. Phase 2:     PRETRAINED_PATH=<phase1_ckpt> bash launch/internvla_a15_r1pro_geop_phase2.sh"
