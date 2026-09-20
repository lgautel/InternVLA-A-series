#!/usr/bin/env python3
"""GL-free MuJoCo forward kinematics for the LIBERO Panda keypoints.

The repo's keypoint generator exports a full robosuite `Lift` scene first
(`util_scripts/export_panda_mjcf.py`), but `import robosuite` builds a GL
context at import time and this host has neither an EGL device nor libOSMesa.
That whole route is therefore unavailable, and the exported
`evaluation/panda_robosuite_lift.xml` is not in the repo either.

Only the kinematic chain matters for keypoints, and it can be rebuilt without
robosuite: load the bare Panda MJCF, run `mj_forward`, then compose the two
fixed transforms that robosuite would have applied when attaching the gripper:

    robot0_link7  --(robot.xml)-->  robot0_right_hand
                  --(pos 0 0 0,      quat 0.707107 0 0 -0.707107)--> gripper0_right_gripper
                  --(pos 0 0 0.097,  quat identity)               --> gripper0_eef

`gripper0_grip_site` sits at the origin of the `eef` body with identity
orientation, so the 8th keypoint equals `observation.state[0:3]` up to the
constant arena base offset -- which is what `self_check` exploits to prove the
chain is correct against the recorded data.

Position and orientation in `observation.state` do NOT share a frame:
robosuite's `eef_pos` reads `site_xpos[gripper0_grip_site]` while `eef_quat`
reads `get_body_xquat(robot0_right_hand)` (robosuite/robots/single_arm.py:305
and :309, with `_eef_name = "right_hand"`). The 8th keypoint's quaternion is
therefore the state's orientation rotated by a constant -90 deg about z.

Output convention is byte-for-byte the one used by both the training
generator (util_scripts/generate_libero_keypoints.py) and the eval-time
extractor (evaluation/LIBERO2/keypoint_utils.py):
positions divided by R_pad, quaternions as xyzw with qw >= 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROBOSUITE_ASSETS = Path(
    "/B/VENV/libero_plus_client/lib/python3.10/site-packages/robosuite/models/assets"
)
PANDA_XML = ROBOSUITE_ASSETS / "robots/panda/robot.xml"

# R_pad recorded in the training set's meta/keypoints_meta.json and hardcoded
# as DEFAULT_R_PAD in evaluation/LIBERO2/keypoint_utils.py.
DEFAULT_R_PAD = 1.8212722539901733

# robosuite Lift places robot0_base here; the training keypoints live in this
# frame regardless of which arena the episode was actually recorded in.
LIFT_BASE_XPOS = np.array([-0.56, 0.0, 0.912])

KEYPOINT_ARM_BODIES = [f"link{i}" for i in range(1, 8)]
NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7

# Static gripper attachment, read from robosuite's panda_gripper.xml.
_Q_RIGHT_GRIPPER = np.array([0.707107, 0.0, 0.0, -0.707107])  # wxyz
_P_EEF = np.array([0.0, 0.0, 0.097])

# Arena base positions, from LIBERO's base_xpos_offset tables
# (libero/libero/envs/robots/{mounted_panda,on_the_ground_panda}.py) combined
# with the per-problem arena choice. Verified against the data by analyze_4d.py.
ARENA_BASE_XPOS: dict[str, np.ndarray] = {
    "table": np.array([-0.66, 0.0, 0.912]),
    "kitchen_table": np.array([-0.66, 0.0, 0.912]),
    "study_table": np.array([-0.75, 0.0, 0.912]),
    "living_room_table": np.array([-0.51, 0.0, 0.42]),
    "floor": np.array([-0.60, 0.0, 0.0]),
}

# Suite -> arena. libero_10 mixes three scenes, resolved per task by `arena_of`.
_SUITE_ARENA = {
    "libero_spatial": "table",
    "libero_goal": "table",
    "libero_object": "floor",
}
_SCENE_ARENA = {
    "KITCHEN": "kitchen_table",
    "LIVING_ROOM": "living_room_table",
    "STUDY": "study_table",
}


def arena_of(subset: str, task_name: str) -> str:
    """Arena key for an episode, from its suite and original demo filename."""
    if subset in _SUITE_ARENA:
        return _SUITE_ARENA[subset]
    for prefix, arena in _SCENE_ARENA.items():
        if task_name.startswith(prefix):
            return arena
    raise KeyError(f"cannot resolve arena for {subset}/{task_name}")


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def wxyz_to_xyzw_hemisphere(q: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz -> stored xyzw, flipped so qw >= 0."""
    out = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    if out[3] < 0:
        out = -out
    return out


def axisangle_to_quat_xyzw(axisangle: np.ndarray) -> np.ndarray:
    """Inverse of robosuite's `quat2axisangle`, returning xyzw with qw >= 0."""
    angle = float(np.linalg.norm(axisangle))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axisangle / angle
    q = np.concatenate([axis * np.sin(angle / 2.0), [np.cos(angle / 2.0)]])
    return -q if q[3] < 0 else q


class PandaFK:
    """Forward kinematics for the 8 keypoint bodies of the LIBERO Panda."""

    def __init__(self, xml_path: Path | str = PANDA_XML, base_xpos: np.ndarray | None = None):
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.base_xpos = np.asarray(LIFT_BASE_XPOS if base_xpos is None else base_xpos, dtype=np.float64)
        self._arm_ids = [self.model.body(name).id for name in KEYPOINT_ARM_BODIES]
        self._hand_id = self.model.body("right_hand").id

    def world_poses(self, qpos9: np.ndarray, base_xpos: np.ndarray | None = None) -> np.ndarray:
        """[9] qpos -> [8, 7] of (x, y, z, qw, qx, qy, qz) in world frame.

        Quaternions stay in MuJoCo wxyz order here; `keypoints` converts them.
        """
        base = self.base_xpos if base_xpos is None else np.asarray(base_xpos, dtype=np.float64)
        self.data.qpos[:7] = np.asarray(qpos9, dtype=np.float64)[:7]
        self._mujoco.mj_forward(self.model, self.data)

        out = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float64)
        for i, bid in enumerate(self._arm_ids):
            out[i, :3] = self.data.xpos[bid] + base
            out[i, 3:] = self.data.xquat[bid]

        # gripper0_eef, composed from the two static attachment transforms
        hand_pos, hand_quat = self.hand_pose(base)
        grip_quat = quat_mul(hand_quat, _Q_RIGHT_GRIPPER)
        out[7, :3] = hand_pos + quat_to_mat(grip_quat) @ _P_EEF
        out[7, 3:] = grip_quat
        return out

    def hand_pose(self, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`robot0_right_hand` pose -- the frame `observation.state[3:6]` uses.

        Assumes `mj_forward` already ran for the desired qpos.
        """
        return self.data.xpos[self._hand_id] + base, self.data.xquat[self._hand_id].copy()

    def keypoints(
        self,
        qpos9: np.ndarray,
        base_xpos: np.ndarray | None = None,
        r_pad: float = DEFAULT_R_PAD,
    ) -> np.ndarray:
        """[9] qpos -> [8, 7] keypoints in the stored convention."""
        poses = self.world_poses(qpos9, base_xpos)
        out = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        out[:, :3] = poses[:, :3] / r_pad
        for i in range(NUM_KEYPOINTS):
            out[i, 3:] = wxyz_to_xyzw_hemisphere(poses[i, 3:])
        return out

    def batch_world_poses(self, qpos9: np.ndarray, base_xpos: np.ndarray | None = None) -> np.ndarray:
        """[T, 9] -> [T, 8, 7]."""
        out = np.empty((len(qpos9), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float64)
        for t, q in enumerate(qpos9):
            out[t] = self.world_poses(q, base_xpos)
        return out

    def batch_keypoints(
        self,
        qpos9: np.ndarray,
        base_xpos: np.ndarray | None = None,
        r_pad: float = DEFAULT_R_PAD,
    ) -> np.ndarray:
        out = np.empty((len(qpos9), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for t, q in enumerate(qpos9):
            out[t] = self.keypoints(q, base_xpos, r_pad)
        return out


def self_check(max_episodes_per_subset: int = 3) -> int:
    """Prove the chain against recorded data.

    Two independent checks per episode, both in the live arena frame:
      * position: FK(gripper0_eef) vs `observation.state[0:3]`
      * orientation: FK(robot0_right_hand) vs `observation.state[3:6]`
    """
    from rlds_reader import SUBSETS, iter_episodes

    fk = PandaFK()
    worst_pos = 0.0
    worst_rot = 0.0
    print(f"{'subset':15s} {'task':46s} {'arena':18s} {'pos_err_m':>10s} {'rot_err_deg':>11s}")
    for subset in SUBSETS:
        for ep in iter_episodes(subset, max_episodes=max_episodes_per_subset):
            arena = arena_of(subset, ep.task_name)
            base = ARENA_BASE_XPOS[arena]
            qpos = ep.qpos9()
            step = max(1, len(qpos) // 20)
            pos_err, rot_err = [], []
            for t in range(0, len(qpos), step):
                pose = fk.world_poses(qpos[t], base_xpos=base)[7]
                pos_err.append(np.linalg.norm(pose[:3] - ep.state[t, :3]))
                q_fk = wxyz_to_xyzw_hemisphere(fk.hand_pose(base)[1])
                q_gt = axisangle_to_quat_xyzw(ep.state[t, 3:6].astype(np.float64))
                dot = min(1.0, abs(float(np.dot(q_fk, q_gt))))
                rot_err.append(np.degrees(2 * np.arccos(dot)))
            pos_max, rot_max = max(pos_err), max(rot_err)
            worst_pos, worst_rot = max(worst_pos, pos_max), max(worst_rot, rot_max)
            print(f"{subset:15s} {ep.task_name[:46]:46s} {arena:18s} {pos_max:10.2e} {rot_max:11.4f}")

    print(f"\nworst position error : {worst_pos:.3e} m")
    print(f"worst rotation error : {worst_rot:.4f} deg")
    ok = worst_pos < 2e-3 and worst_rot < 0.5
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(self_check())
