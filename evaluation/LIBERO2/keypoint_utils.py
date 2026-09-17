"""Runtime 4D keypoint extraction using standalone Lift MJCF FK.

Instead of reading body positions from live env.sim.data (which has
arena-dependent robot base offsets), we:
  1. Extract 9D qpos from the live env (7 arm joints + 2 gripper fingers)
  2. Inject into a standalone robosuite Lift MJCF (fixed base at [-0.56, 0, 0.912])
  3. Run mj_forward on the standalone MJCF
  4. Read body pos/quat from the standalone MJCF

This matches the training pipeline's FK (same Lift-family MJCF and qpos injection).
EEF body names differ across XMLs (`gripper0_eef` vs `gripper0_right_eef`);
StandaloneFK resolves whichever name exists on the loaded model.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

DEFAULT_R_PAD: float = 1.8212722539901733

# Same candidate order as util_scripts/generate_libero_keypoints.py.
# Lift MJCF uses gripper0_eef; the training XML wrote gripper0_right_eef
# into keypoints_meta.json. Resolve against the loaded model — do not
# require the metadata string to match the eval MJCF name.
EEF_BODY_CANDIDATES: tuple[str, ...] = ("gripper0_eef", "gripper0_right_eef")

KEYPOINT_ARM_BODY_NAMES: list[str] = [
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
]


def resolve_eef_body_name(model) -> str:
    """Return the EEF body that exists on this MuJoCo model."""
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except (KeyError, ValueError):
            continue
    raise KeyError(f"No EEF body found (tried {', '.join(EEF_BODY_CANDIDATES)})")


def default_keypoint_body_names(model) -> list[str]:
    return [*KEYPOINT_ARM_BODY_NAMES, resolve_eef_body_name(model)]


# Lift-MJCF default (gripper0_eef). StandaloneFK overwrites via resolve.
DEFAULT_KEYPOINT_BODIES: list[str] = [
    *KEYPOINT_ARM_BODY_NAMES,
    "gripper0_eef",
]

KEYPOINT_DIM: int = 7
_ROBOT_NQPOS: int = 9

_DEFAULT_MJCF_PATH = Path(__file__).resolve().parents[1] / "panda_robosuite_lift.xml"


class StandaloneFK:
    """Standalone robosuite Lift MJCF FK for keypoint extraction."""

    def __init__(
        self,
        mjcf_path: str | Path = _DEFAULT_MJCF_PATH,
        body_names: list[str] | None = None,
        r_pad: float = DEFAULT_R_PAD,
    ) -> None:
        mjcf_path = Path(mjcf_path)
        if not mjcf_path.exists():
            raise FileNotFoundError(
                f"Lift MJCF not found: {mjcf_path}\n"
                "Re-export with: MUJOCO_GL=glx python util_scripts/export_panda_mjcf.py"
            )
        self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self._data = mujoco.MjData(self._model)
        self.body_names = body_names or default_keypoint_body_names(self._model)
        self.r_pad = r_pad
        self._body_ids: list[int] = []
        for name in self.body_names:
            self._body_ids.append(self._model.body(name).id)

    def extract(self, qpos9: np.ndarray) -> np.ndarray:
        """Extract 8x7D keypoints from a 9D qpos vector.

        Args:
            qpos9: [9] float arm_joints[7] + gripper_fingers[2]

        Returns:
            kpts: [K, 7] float32, normalized (pos/R_pad, quat xyzw hemisphere)
        """
        self._data.qpos[:_ROBOT_NQPOS] = qpos9[:_ROBOT_NQPOS]
        mujoco.mj_forward(self._model, self._data)

        K = len(self._body_ids)
        kpts = np.empty((K, KEYPOINT_DIM), dtype=np.float32)
        for i, bid in enumerate(self._body_ids):
            kpts[i, :3] = self._data.xpos[bid] / self.r_pad
            w, x, y, z = self._data.xquat[bid]          # MuJoCo: wxyz
            xyzw = np.array([x, y, z, w], dtype=np.float32)
            if xyzw[3] < 0:                              # hemisphere: qw >= 0
                xyzw = -xyzw
            kpts[i, 3:] = xyzw
        return kpts


def _get_robot_qpos(env) -> np.ndarray:
    """Extract 9D qpos from a live LIBERO robosuite env."""
    robot = env.robots[0]
    joint_pos = env.sim.data.qpos[robot._ref_joint_pos_indexes]           # [7]
    gripper_pos = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] # [2]
    return np.concatenate([joint_pos, gripper_pos]).astype(np.float64)


class KeypointExtractor:
    """Extract 8x7D keypoints from the current env qpos via standalone Lift FK."""

    def __init__(self, env, fk: StandaloneFK | None = None) -> None:
        self._env = env
        self._fk = fk if fk is not None else StandaloneFK()

    def extract(self) -> np.ndarray:
        qpos9 = _get_robot_qpos(self._env)
        return self._fk.extract(qpos9)


class KeypointHistory:
    """Rolling buffer (oldest-first, zero-padded at back)."""

    def __init__(self, max_len: int = 200, num_kpts: int = 8, kpt_dim: int = 7):
        self.max_len = max_len
        self.num_kpts = num_kpts
        self.kpt_dim = kpt_dim
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros((self.max_len, self.num_kpts, self.kpt_dim), dtype=np.float32)
        self._len = 0

    def push(self, kpt: np.ndarray) -> None:
        if self._len < self.max_len:
            self._buffer[self._len] = kpt
            self._len += 1
        else:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = kpt

    @property
    def his_len(self) -> int:
        return self._len

    def get_history(self) -> tuple[np.ndarray, int]:
        return self._buffer.copy(), self._len
