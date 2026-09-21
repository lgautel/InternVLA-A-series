#!/usr/bin/env bash
# End-to-end data remediation for the franka3 plug-into-socket dataset.
#
#   raw capture -> semantic gate (expect BLOCK) -> fix -> FK keypoints
#               -> v3.0 convert -> semantic gate (expect clean) -> symlink for training
#
# Nothing in src/ or in the existing launch scripts is modified: the output is a new
# dataset directory, and training picks it up purely through DATA_SRC / DATA_REPO_ID.
#
# Usage:
#   bash b/s/Frk2/ds/run_franka3_data_fix.sh                    # default: gripper mask
#   GRIPPER_MODE=lag GRIPPER_LAG=3 bash b/s/Frk2/ds/run_franka3_data_fix.sh
#   SKIP_KEYPOINTS=1 bash b/s/Frk2/ds/run_franka3_data_fix.sh   # gate + fix only
set -euo pipefail

# this script lives at <root>/b/s/Frk2/ds/ -> four levels up is the repo root
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV_ROOT="${VENV_ROOT:-/B/VENV/itnvla15rbt20}"
PY="${VENV_ROOT}/bin/python"

SRC="${SRC:-/B/Dta/plug_into_socket_franka3_15hz_lerobot}"
FIXED="${FIXED:-${HOME}/b/Dta/plug_into_socket_franka3_15hz_fixed}"
OUT_4D="${OUT_4D:-${HOME}/b/Dta/plug_into_socket_franka3_15hz_fixed_4d}"
REPO_ID="${REPO_ID:-plug_into_socket_franka3_15hz_fixed_4d}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/B/VENV/hf_home/lerobot}"

GRIPPER_MODE="${GRIPPER_MODE:-mask}"
GRIPPER_LAG="${GRIPPER_LAG:-3}"
EE_MODE="${EE_MODE:-keep}"
URDF="${URDF:-${PROJ_ROOT}/b/d/Frk2/fr3v2_1_franka_hand.urdf}"
GATE_CFG="${PROJ_ROOT}/b/s/Frk2/cfg/data_gate.yaml"
SKIP_KEYPOINTS="${SKIP_KEYPOINTS:-0}"

cd "${PROJ_ROOT}"
echo "================ Step 1/5: 原始数据集语义门禁 (预期阻断) ================"
set +e
"${PY}" b/s/Frk2/ds/verify_franka3_semantics.py \
    --dataset "${SRC}" --config "${GATE_CFG}" \
    --json-out /tmp/franka3_gate_raw.json
RAW_RC=$?
set -e
if [[ ${RAW_RC} -eq 0 ]]; then
  echo "注意: 原始数据集已通过阻断项, 可能已被修过. 继续."
else
  echo "如预期: 原始数据集存在阻断级缺陷 (退出码 ${RAW_RC})."
fi

echo
echo "================ Step 2/5: 构建修正数据集 ================"
"${PY}" b/s/Frk2/ds/build_franka3_fixed_dataset.py \
    --src "${SRC}" --dst "${FIXED}" \
    --gripper-state-mode "${GRIPPER_MODE}" --gripper-lag "${GRIPPER_LAG}" \
    --ee-state-mode "${EE_MODE}" \
    --link-videos --force

echo
echo "================ Step 3/5: 修正数据集复检 (预期 0 阻断) ================"
"${PY}" b/s/Frk2/ds/verify_franka3_semantics.py \
    --dataset "${FIXED}" --config "${GATE_CFG}" \
    --json-out /tmp/franka3_gate_fixed.json

if [[ "${SKIP_KEYPOINTS}" == "1" ]]; then
  echo
  echo "SKIP_KEYPOINTS=1 -> 跳过 Step 4/5, 修正数据集在 ${FIXED}"
  exit 0
fi

echo
echo "================ Step 4/5: 重新生成 FK 关键点 ================"
# The gripper/EE edits never touch state[0:7], so the keypoints are numerically
# identical to the source ones; they are regenerated anyway so that
# keypoints_meta.json / train_inference_contract.json point at the new dataset.
"${PY}" b/s/Frk2/generate_franka2_keypoints.py \
    --source "${FIXED}" --dest "${OUT_4D}" --urdf "${URDF}" --force

echo
echo "================ Step 5/5: 关键点校验 + 训练软链 ================"
"${PY}" b/s/Frk2/verify_franka2_keypoints.py --dataset "${OUT_4D}" --urdf "${URDF}"
"${PY}" b/s/Frk2/ds/verify_franka3_semantics.py \
    --dataset "${OUT_4D}" --config "${GATE_CFG}"

mkdir -p "${HF_LEROBOT_HOME}"
ln -sfn "${OUT_4D}" "${HF_LEROBOT_HOME}/${REPO_ID}"
echo "软链: ${HF_LEROBOT_HOME}/${REPO_ID} -> ${OUT_4D}"

cat <<EOF

================ 完成 ================
训练时这样指向新数据集 (无需改任何框架代码):

  export DATA_SRC=${OUT_4D}
  export DATA_REPO_ID=${REPO_ID}
  bash b/s/Frk2/run_frk2_plug_warmup.sh

夹爪极性契约: ${OUT_4D}/meta/gripper_contract.json
部署时必须读取该文件, 不要相信 info.json 里的 "gripper_width" 这个名字.
EOF
