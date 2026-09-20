"""Train-inference aligned FK keypoint computer for Franka (v2).

This module computes 7D keypoints from joint angles at inference time,
matching the exact semantics of the offline generation pipeline.

Key invariants (see train_inference_contract.json):
  - his_len = number of frames strictly before current frame
  - kpt_t = current frame (not in history)
  - history fill: valid frames at front, zero-pad at back
  - FK uses the same URDF and Pinocchio as training data generation

Usage (inference loop):
    fk = FKKeypointComputerV2(urdf_path, meta_path)
    for control_step in range(max_steps):
        arm_q7 = robot.get_joint_positions()[:7]
        his_kpts, his_len, kpt_t = fk.step(arm_q7)
        if control_step % n_exec == 0:
            action = model.infer(..., his_kpts=his_kpts, his_len=his_len)
        fk.commit()  # AFTER consuming results
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FKKeypointComputerV2:
    def __init__(self, urdf_path: str, meta_path: str, link_prefix: str = "fr3v2_1"):
        with open(meta_path) as f:
            meta = json.load(f)
        self.r_pad = meta["bbox_radius"]
        self.H = 200
        self.J = meta["num_keypoints"]
        self.D = meta["keypoint_dim"]

        contract_path = Path(meta_path).parent / "train_inference_contract.json"
        if contract_path.exists():
            with open(contract_path) as f:
                contract = json.load(f)
            self.H = contract.get("inference_constraints", {}).get(
                "keypoint_history_max_len", self.H
            )
            actual_hash = hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest()
            expected_hash = contract.get("urdf_sha256")
            if expected_hash and actual_hash != expected_hash:
                logger.warning(
                    "URDF hash mismatch! Train=%s Infer=%s. "
                    "This may cause train-inference inconsistency.",
                    expected_hash[:16], actual_hash[:16],
                )

        from generate_franka2_keypoints import FrankaFKExtractor7D
        self._extractor = FrankaFKExtractor7D(urdf_path, link_prefix)
        self._history: list[np.ndarray] = []
        self._kpt_t: np.ndarray | None = None

    def compute_and_normalize(self, arm_q7: np.ndarray) -> np.ndarray:
        kpt = self._extractor.compute(arm_q7.astype(np.float64))
        kpt[:, :3] /= self.r_pad
        return kpt

    def step(self, arm_q7: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute FK for current frame. Does NOT add to history.

        Returns:
            his_kpts: [H, J, D] float32, valid frames at front, zero-padded
            his_len: int, number of frames strictly before current frame
            kpt_t: [J, D] float32, current frame keypoints
        """
        self._kpt_t = self.compute_and_normalize(arm_q7)
        his_kpts, his_len = self._pack_history()
        return his_kpts, his_len, self._kpt_t

    def commit(self):
        """Add current kpt_t to history. Call after step() and after
        the inference server has consumed the results."""
        if self._kpt_t is not None:
            self._history.append(self._kpt_t.copy())
            if len(self._history) > self.H:
                self._history = self._history[-self.H:]

    def _pack_history(self) -> tuple[np.ndarray, int]:
        his_len = len(self._history)
        his_kpts = np.zeros((self.H, self.J, self.D), dtype=np.float32)
        actual = min(his_len, self.H)
        if actual > 0:
            his_kpts[:actual] = np.array(self._history[-actual:])
        return his_kpts, actual

    def reset(self):
        """Call at the start of each episode."""
        self._history = []
        self._kpt_t = None

    @property
    def current_his_len(self) -> int:
        return min(len(self._history), self.H)
