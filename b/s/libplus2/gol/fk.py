"""Goal-tabletop MuJoCo FK.

The kinematic XML is the robosuite 1.4.1 Panda chain with meshes removed.
``gripper0_eef`` is not a body in that file; it is the composition

    right_hand --quat--> gripper --(+0.097 z)--> eef

which is what ``observation.state[0:3]`` records (the grip site). The
axis-angle in ``state[3:6]`` is the orientation of ``right_hand``, not of the
eef. Those two frames differ by a constant 90 degrees about z.

Finger joints are stored in qpos[7:9] for a stable 9-D interface with the
RLDS reader, but they do not move any of the eight keypoint bodies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contract import (
    ARM_BODY_NAMES,
    EEF_OFFSET_IN_GRIPPER,
    GOAL_BASE_XPOS,
    GRIPPER_ATTACH_QUAT_WXYZ,
    HAND_BODY_NAME,
    KEYPOINT_DIM,
    KIN_XML,
    NUM_KEYPOINTS,
)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_mat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def wxyz_to_xyzw_hemisphere(q: np.ndarray) -> np.ndarray:
    out = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    if out[3] < 0:
        out = -out
    return out


def axisangle_to_quat_xyzw(axisangle: np.ndarray) -> np.ndarray:
    """Inverse of robosuite ``quat2axisangle``. Returns xyzw, qw >= 0."""
    angle = float(np.linalg.norm(axisangle))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = np.asarray(axisangle, dtype=np.float64) / angle
    q = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
    return -q if q[3] < 0 else q


def rotation_error_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    dot = min(1.0, abs(float(np.dot(q_a, q_b))))
    return float(np.degrees(2.0 * np.arccos(dot)))


def qpos9_from_state(joint_state: np.ndarray, state: np.ndarray) -> np.ndarray:
    """joint_state [7] + gripper fingers state[6:8] -> [9]."""
    joint = np.asarray(joint_state, dtype=np.float64).reshape(-1)
    st = np.asarray(state, dtype=np.float64).reshape(-1)
    if joint.shape[0] != 7 or st.shape[0] < 8:
        raise ValueError(f"expected joint[7] and state[>=8], got {joint.shape} {st.shape}")
    q = np.empty(9, dtype=np.float64)
    q[:7] = joint
    q[7:9] = st[6:8]
    return q


class GoalTableFK:
    """Eight keypoints in the LIBERO Goal table world frame."""

    def __init__(self, xml_path: str | Path = KIN_XML, base_xpos: np.ndarray | None = None):
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 7:
            raise RuntimeError(f"kinematic Panda XML must have nq=7, got {self.model.nq}")
        self.base_xpos = np.asarray(GOAL_BASE_XPOS if base_xpos is None else base_xpos, dtype=np.float64)
        self._arm_ids = [self.model.body(name).id for name in ARM_BODY_NAMES]
        self._hand_id = self.model.body(HAND_BODY_NAME).id

    def world_poses(self, qpos9: np.ndarray, base_xpos: np.ndarray | None = None) -> np.ndarray:
        """Return [8, 7] with positions in metres and quaternions in wxyz.

        The last row is gripper0_eef. ``qpos9[7:9]`` is ignored.
        """
        base = self.base_xpos if base_xpos is None else np.asarray(base_xpos, dtype=np.float64)
        self.data.qpos[:7] = np.asarray(qpos9, dtype=np.float64)[:7]
        self._mujoco.mj_forward(self.model, self.data)

        out = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float64)
        for i, bid in enumerate(self._arm_ids):
            out[i, :3] = self.data.xpos[bid] + base
            out[i, 3:] = self.data.xquat[bid]
        hand_pos = self.data.xpos[self._hand_id] + base
        hand_quat = self.data.xquat[self._hand_id].copy()
        grip_quat = quat_mul_wxyz(hand_quat, GRIPPER_ATTACH_QUAT_WXYZ)
        out[7, :3] = hand_pos + quat_to_mat_wxyz(grip_quat) @ EEF_OFFSET_IN_GRIPPER
        out[7, 3:] = grip_quat
        return out

    def hand_pose(self, qpos9: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (position in the table world frame, quaternion wxyz) of right_hand."""
        self.world_poses(qpos9)
        pos = self.data.xpos[self._hand_id] + self.base_xpos
        return pos.copy(), self.data.xquat[self._hand_id].copy()

    def hand_quat_wxyz(self) -> np.ndarray:
        """right_hand quaternion from the last ``world_poses`` call."""
        return self.data.xquat[self._hand_id].copy()

    def keypoints(self, qpos9: np.ndarray, r_pad: float) -> np.ndarray:
        """[8, 7] float32. Positions divided by ``r_pad``. Quaternions xyzw, qw >= 0."""
        if r_pad <= 0:
            raise ValueError(f"r_pad must be positive, got {r_pad}")
        poses = self.world_poses(qpos9)
        out = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        out[:, :3] = (poses[:, :3] / r_pad).astype(np.float32)
        for i in range(NUM_KEYPOINTS):
            out[i, 3:] = wxyz_to_xyzw_hemisphere(poses[i, 3:]).astype(np.float32)
        return out

    def batch_world(self, qpos9: np.ndarray) -> np.ndarray:
        out = np.empty((len(qpos9), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float64)
        for t, q in enumerate(qpos9):
            out[t] = self.world_poses(q)
        return out

    def batch_keypoints(self, qpos9: np.ndarray, r_pad: float) -> np.ndarray:
        out = np.empty((len(qpos9), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for t, q in enumerate(qpos9):
            out[t] = self.keypoints(q, r_pad)
        return out


def compute_r_pad(global_min: np.ndarray, global_max: np.ndarray, margin: float) -> float:
    """R_pad = max(|min|, |max|) * (1 + margin). The margin is inside this value."""
    extremes = np.maximum(np.abs(global_min), np.abs(global_max))
    return float(extremes.max() * (1.0 + margin))
