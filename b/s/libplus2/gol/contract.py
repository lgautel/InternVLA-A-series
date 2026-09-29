"""Train/eval contract for LIBERO-Plus Goal robot keypoints.

Numbers that are measured on ``/B/Dta/LIBERO/rlds/libero_goal`` live in the
analysis note. This module only freezes the quantities the generator and the
eval runtime must share. ``R_pad`` is NOT a constant: Pass 1 of the generator
writes it into ``keypoints_meta.json``, and eval refuses to start without that
file. The old Lift-scene default 1.8212722539901733 belongs to a different base
pose and must not be reused.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# robosuite MountedPanda on a LIBERO table arena.
# table_full_size[0] = 1.0, base_xpos_offset["table"](1.0) = (-0.66, 0, 0),
# pedestal height 0.912 m. Verified: FK eef vs observation.state[0:3]
# max error 6.1e-4 m on sampled Goal episodes. Lift base (-0.56, 0, 0.912)
# is a systematic +0.10 m in x.
GOAL_BASE_XPOS = np.array([-0.66, 0.0, 0.912], dtype=np.float64)

# Static attachment from robot0_right_hand to gripper0_eef.
# robosuite panda_gripper.xml: quat wxyz (0.707107, 0, 0, -0.707107), then +0.097 m along gripper z.
GRIPPER_ATTACH_QUAT_WXYZ = np.array([0.707107, 0.0, 0.0, -0.707107], dtype=np.float64)
EEF_OFFSET_IN_GRIPPER = np.array([0.0, 0.0, 0.097], dtype=np.float64)

ARM_BODY_NAMES = [f"link{i}" for i in range(1, 8)]
HAND_BODY_NAME = "right_hand"
NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7  # px, py, pz, qx, qy, qz, qw
BBOX_MARGIN = 0.15
FPS = 20  # control rate of this RLDS release; the paired LeRobot copy declares 20 Hz
# H = 200 is the value every existing launch/eval default already uses:
# launch/libplus_sft_launch.sh (--policy/--dataset.keypoint_history_max_len=200),
# LiberoModelClient(kpt_history_max_len=200) and KeypointHistory(max_len=200).
# Goal episodes are 75..299 steps (mean 121), so H=200 holds the whole past for
# almost every sample. gen1 used 128, which silently disagreed with the client.
HISTORY_MAX_LEN = 200
CHUNK_SIZE = 50
KPT_4D_MODE = "pos_rot"

# --- dataset / server identity (a wrong value here crashes or silently mis-prompts) ---
# panda.yaml maps observation.images.image -> image0 and image2 -> image1, and
# eval3.md fixes --stats_key panda --robot_type panda. Any other robot_type made
# the server fall back to the 1-image mapping (INVALID_IMAGE_COUNT).
ROBOT_TYPE = "panda"
STATS_KEY = "panda"

# --- image orientation ---
# The RLDS JPEGs are rot180 of the robosuite raw buffer (measured, see gen2 sec 3).
# evaluation/LIBERO2/train_eval_contract.json says image_orientation = raw and
# LiberoModelClient(rotate_images=False) enforces it. The generator therefore
# writes raw, i.e. img[::-1, ::-1] of the RLDS frame.
RLDS_IMAGE_ORIENTATION = "rot180"
IMAGE_ORIENTATION = "raw"

# --- tolerances ---
# FK eef vs observation.state[0:3]. Measured max on sampled Goal episodes 6.1e-4 m.
EPISODE_EEF_TOL_M = 2.0e-3
# Live env: FK eef vs obs["robot0_eef_pos"]. Looser; catches a wrong arena base (0.1 m).
LIVE_EEF_TOL_M = 5.0e-3

CONTRACT_FILENAME = "goal_train_eval_contract.json"

# Stored quaternion is xyzw with qw >= 0. MuJoCo native order is wxyz.
QUAT_LAYOUT = "xyzw"
QUAT_HEMISPHERE = "qw>=0"

KIN_XML = Path(__file__).resolve().parent / "assets" / "panda_kin.xml"
# Mesh-free copy of the robosuite Lift export with the base moved to GOAL_BASE_XPOS.
GOAL_MJCF = Path(__file__).resolve().parent / "assets" / "panda_goal_table.xml"
LIFT_MJCF_SOURCE = Path(__file__).resolve().parents[4] / "evaluation" / "panda_robosuite_lift.xml"

KEYPOINT_NAMES = [
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link7",
    "gripper0_eef",
]


def feature_names() -> list[str]:
    return [f"{name}_{comp}" for name in KEYPOINT_NAMES for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")]
