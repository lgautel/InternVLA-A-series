"""LiberoModelClient specialised for the Goal 4D contract (an extension, not a patch).

``evaluation/LIBERO2/model2libero_interface.py::LiberoModelClient`` is reused as is.
This subclass only changes four things, each tied to an incident:

1. FK and R_pad come from the training dataset's contract file, not from the Lift
   defaults (base -0.56 m, R_pad 1.8213).                              (arena base / R_pad)
2. ``push_keypoint`` is ignored until the first policy request of the episode. The
   base eval loop calls it during the 10 dummy wait steps; training never has
   those frames in ``his_kpts``, so without this the first request would see
   his_len=10 instead of 0.                                            (his_len)
3. The first policy request of every episode compares FK eef with
   ``obs["robot0_eef_pos"]``. A wrong base frame is 0.10 m off and aborts the
   episode instead of being silently evaluated.                        (coord drift)
4. Orientation: the Goal contract must agree with evaluation/LIBERO2's
   train_eval_contract.json (raw). Otherwise refuse to start.          (F1)

Importing this module imports mujoco through the base class. Do it only inside the
forked eval child (see eval_goal_plus.py), never in the parent that forks.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from contract import GOAL_MJCF, LIVE_EEF_TOL_M  # noqa: E402
from contract_v2 import pack_state_v2  # noqa: E402
from load_contract import load_goal_contract  # noqa: E402

from evaluation.LIBERO2.keypoint_utils import _get_robot_qpos  # noqa: E402
from evaluation.LIBERO2.model2libero_interface import LiberoModelClient  # noqa: E402
from evaluation.LIBERO2.orientation_contract import load_train_eval_contract  # noqa: E402

CONTRACT_ENV = "GOAL4D_CONTRACT"


class GoalFKMismatch(RuntimeError):
    """FK end-effector disagrees with the live simulator: wrong base frame or wrong qpos."""


class GoalLiberoModelClient(LiberoModelClient):
    def __init__(self, *args: Any, contract_path: str | os.PathLike | None = None, **kwargs: Any) -> None:
        path = contract_path or os.environ.get(CONTRACT_ENV)
        if not path:
            raise RuntimeError(f"{CONTRACT_ENV} is not set; eval needs the training dataset's goal contract")
        contract = load_goal_contract(path)

        global_orient = str(load_train_eval_contract().get("image_orientation", "raw")).lower()
        if str(contract["image_orientation"]).lower() != global_orient:
            raise RuntimeError(
                f"goal contract image_orientation={contract['image_orientation']!r} but "
                f"evaluation/LIBERO2/train_eval_contract.json says {global_orient!r}. "
                "Regenerate the dataset with --image-orientation raw, or change that contract on purpose."
            )

        import hashlib

        md5 = hashlib.md5(Path(GOAL_MJCF).read_bytes()).hexdigest()
        if md5 != contract["eval_mjcf_md5"]:
            raise RuntimeError(
                f"panda_goal_table.xml md5 {md5} != contract {contract['eval_mjcf_md5']}: "
                "the eval FK model is not the one the training data was checked against"
            )

        kpt_hist_len = int(os.environ.get("KPT_HISTORY_MAX_LEN", contract["keypoint_history_max_len"]))
        wanted = {
            "mjcf_path": str(GOAL_MJCF),
            "kpt_r_pad": float(contract["r_pad"]),
            "kpt_history_max_len": kpt_hist_len,
        }
        for key, value in wanted.items():
            if key in kwargs and kwargs[key] not in (None, value):
                raise ValueError(f"{key}={kwargs[key]!r} conflicts with the goal contract value {value!r}")
            kwargs[key] = value

        super().__init__(*args, **kwargs)
        self.contract = contract
        self._armed = False
        self.live_fk_max_err_m = 0.0
        self.live_fk_violations = 0
        self._live_tol = float(contract.get("live_eef_tol_m", LIVE_EEF_TOL_M))

    # --- (v2) 14-dim state override ----------------------------------------
    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        if self.contract.get("schema") != "goal_train_eval_contract/2":
            return super()._extract_state(obs)
        joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
        fingers = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if joint.shape != (7,) or eef_pos.shape != (3,) or eef_quat.shape != (4,) or fingers.shape != (2,):
            raise ValueError(
                f"v2 obs shapes joint={joint.shape} eef={eef_pos.shape} "
                f"quat={eef_quat.shape} fingers={fingers.shape}"
            )
        from evaluation.LIBERO2.model2libero_interface import _quat2axisangle
        axisangle = _quat2axisangle(eef_quat)
        return pack_state_v2(joint, np.concatenate([eef_pos, axisangle]), fingers[0], fingers[1])

    # --- (2) history clock -------------------------------------------------
    def reset(self, task_description: str | None = None) -> None:
        super().reset(task_description)
        self._armed = False

    def push_keypoint(self) -> None:
        """No-op until the first policy request of this episode has been made."""
        if not self._armed:
            return
        super().push_keypoint()

    @property
    def his_len(self) -> int:
        return 0 if self._kpt_history is None else self._kpt_history.his_len

    # --- (3) live FK guard ---------------------------------------------------
    def _check_live_fk(self, obs: dict[str, Any], first: bool) -> None:
        if not self.enable_keypoints or self._kpt_extractor is None:
            return
        env = self._kpt_extractor._env
        kpt = self._fk.extract(_get_robot_qpos(env))
        fk_eef = kpt[7, :3].astype(np.float64) * self._fk.r_pad
        err = float(np.linalg.norm(fk_eef - np.asarray(obs["robot0_eef_pos"], dtype=np.float64)))
        self.live_fk_max_err_m = max(self.live_fk_max_err_m, err)
        if err > self._live_tol:
            self.live_fk_violations += 1
            if first:
                raise GoalFKMismatch(
                    f"FK eef {fk_eef.round(4).tolist()} vs live obs {list(np.round(obs['robot0_eef_pos'], 4))}: "
                    f"{err * 1e3:.1f} mm > {self._live_tol * 1e3:.1f} mm. The base frame of this arena is not the "
                    "Goal table base (-0.66, 0, 0.912)."
                )
            warnings.warn(f"live FK drift {err * 1e3:.1f} mm", RuntimeWarning, stacklevel=2)

    def step(self, obs: dict[str, Any], lang: str) -> np.ndarray:
        first = not self._armed
        if first:
            self._check_live_fk(obs, first=True)
        action = super().step(obs, lang)  # may reset() internally on a language change
        if not first:
            self._check_live_fk(obs, first=False)
        self._armed = True
        return action
