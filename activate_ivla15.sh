#!/usr/bin/env bash
source /home/luogang/miniforge3/etc/profile.d/conda.sh
conda activate ivla15
export REPO_ROOT=/home/luogang/SRC/Robot/InternVLA-A-series
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/third_party/RoboTwin:${PYTHONPATH:-}"
export HF_HOME="${HOME}/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
echo "ivla15 environment activated. REPO_ROOT=${REPO_ROOT}"
