"""v2 Goal dataset state layout: 14D = joint7 + eef6 + gripper_opening.

This is the single source of truth for v2 field names, slices, and the
gripper_opening formula.  repack, verify, test, and the eval client all
import from here.
"""

from __future__ import annotations

import numpy as np

ROBOT_TYPE = "libero_goal_4dv2"
STATS_KEY = "libero_goal_4dv2"
CONTRACT_SCHEMA = "goal_train_eval_contract/2"
STATE_DIM = 14
JOINT_SLICE = slice(0, 7)
EEF_SLICE = slice(7, 13)
GRIPPER_INDEX = 13
STATE_NAMES = [
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az",
    "gripper",
]
FINGER_NAMES = ["finger_l", "finger_r"]
ACTION_NAMES = ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]
FINGER_ANTISYM_TOL_M = 2.0e-3
COPY_ATOL = 0.0


def gripper_opening(finger_l: np.ndarray, finger_r: np.ndarray) -> np.ndarray:
    """g = q_L - q_R.  Input shapes must match; output has the same shape as finger_l."""
    fl = np.asarray(finger_l, dtype=np.float32)
    fr = np.asarray(finger_r, dtype=np.float32)
    return fl - fr


def pack_state_v2(
    joint7: np.ndarray,
    eef6: np.ndarray,
    finger_l: np.ndarray,
    finger_r: np.ndarray,
) -> np.ndarray:
    """Return [..., 14] float32 state vector.

    joint7: last dim 7 (joint angles, radians).
    eef6:   last dim 6 (eef_pos 3 + eef_axisangle 3).
    finger_l, finger_r: last dim 1 or broadcastable scalars.
    """
    j = np.asarray(joint7, dtype=np.float32)
    e = np.asarray(eef6, dtype=np.float32)
    fl = np.asarray(finger_l, dtype=np.float32)
    fr = np.asarray(finger_r, dtype=np.float32)

    if j.shape[-1] != 7:
        raise ValueError(f"joint7 last dim must be 7, got {j.shape}")
    if e.shape[-1] != 6:
        raise ValueError(f"eef6 last dim must be 6, got {e.shape}")

    g = gripper_opening(fl, fr)
    if g.ndim == 0:
        g = g.reshape(1)
    if g.ndim == 1 and j.ndim == 1:
        pass
    elif g.ndim < j.ndim:
        g = np.expand_dims(g, axis=tuple(range(j.ndim - g.ndim)))

    out = np.concatenate([j, e, g.reshape(*j.shape[:-1], 1)], axis=-1)
    if not np.all(np.isfinite(out)):
        raise ValueError("pack_state_v2: output contains non-finite values (NaN or Inf)")
    return out


def qpos9_from_columns(
    joint7: np.ndarray, fingers2: np.ndarray
) -> np.ndarray:
    """[joint7, finger_l, finger_r], shape [9] or [T, 9].

    Do NOT read from the 14D state vector — use the retained columns directly.
    """
    j = np.asarray(joint7, dtype=np.float32)
    f = np.asarray(fingers2, dtype=np.float32)
    if j.shape[-1] != 7:
        raise ValueError(f"joint7 last dim must be 7, got {j.shape}")
    if f.shape[-1] != 2:
        raise ValueError(f"fingers2 last dim must be 2, got {f.shape}")
    return np.concatenate([j, f], axis=-1)
