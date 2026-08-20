"""Offline FK-based 3D keypoint generation for RoboTwin `stack_bowls_three` (aloha dual-arm).

Implements Phase 2 data preparation for the InternVLA-A1.5 + GeoPredict 3D keypoint fusion
(see design docs `b/d/itrnVLA15_GeoP_3dtrj_3cn2.md` §15 and, more specifically,
`b/d/itrnVLA15_GeoP_3dtrj_3cn2_rbt2stak3.md` §4 "关键点数据管道").

`stack_bowls_three` has no `observation.keypoint_3d` ground truth, but its 14-dim
`observation.state` **is** the aloha dual-arm's joint angles (unlike RoboCasa, where state is an
EEF pose derived *from* FK, aloha's state is the direct FK *input*). This script:

1. Loads the aloha-agilex URDF with `pinocchio` and resolves the 14 keypoint-link frame IDs
   (design doc §2.1.2: 7 links per arm -- `{side}_link1..6` + `{side}_camera` acting as the
   wrist-mounted EEF proxy -- see class docstring of :func:`compute_fk_keypoints_batch` for the
   full mapping).
2. Copies the (read-only) source dataset to a new working directory (never mutates the
   original), then computes FK-derived keypoints for every frame's `observation.state` and
   writes them as a new `observation.keypoint_3d` column (`[42]` = 14 x 3, footprint-relative,
   design doc §2.2.1) into every `data/chunk-*/file-*.parquet` file of the copy.
3. Declares the new feature in the copy's `meta/info.json` (design doc §4.2.2) so
   `LeRobotDataset`/`InternVLAA15Config.keypoint_3d_delta_indices` can pick it up transparently.

Usage:
    python util_scripts/generate_aloha_keypoints.py \\
        --source /mnt/r/DATA/RoboTwin-Clean/stack_bowls_three \\
        --dest /mnt/r/DATA/RoboTwin-Clean-FK/stack_bowls_three
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ALOHA_URDF = (
    "/mnt/r/share/zwy/Projects/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
)

# 14 keypoints, left arm (7) then right arm (7) -- design doc §2.1.2.
KEYPOINT_LINKS: list[str] = [
    "fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6", "left_camera",
    "fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6", "right_camera",
]
NUM_KEYPOINTS = len(KEYPOINT_LINKS)  # 14
FOOTPRINT_FRAME = "footprint"

# `observation.state[0:6]`/`[7:13]` are the left/right arm's 6 revolute-joint drive targets
# (design doc §4.1.3, cross-checked against this dataset's own `meta/info.json` "names" for
# `observation.state`: left_waist, left_shoulder, left_elbow, left_forearm_roll,
# left_wrist_angle, left_wrist_rotate, left_gripper, then the mirrored right_* names).
# `state[6]`/`state[13]` are the grippers, which do NOT affect any of the 14 keypoint links
# (the wrist cameras and *_link6 are rigidly/fixed-jointed independent of the gripper fingers
# *_link7/8), so they are intentionally not fed into FK at all.
LEFT_ARM_JOINTS = ["fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6"]
RIGHT_ARM_JOINTS = ["fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6"]

KEYPOINT_FEATURE_NAMES: list[str] = [
    f"{link}_{axis}" for link in KEYPOINT_LINKS for axis in ("x", "y", "z")
]


class AlohaFKKeypointExtractor:
    """Wraps a `pinocchio` model of the aloha-agilex platform for batched FK keypoint extraction.

    Only the 12 front-left/front-right arm revolute joints are driven from `observation.state`;
    the rear-left/rear-right arms present in the full-platform URDF (design doc §1.1.2: "aloha
    URDF 定义了整个平台（含 4 条臂 + 底座）") and the two grippers are left at their neutral
    configuration since none of the 14 keypoint links depend on them.
    """

    def __init__(self, urdf_path: str = DEFAULT_ALOHA_URDF):
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.frame_ids = [self.model.getFrameId(name) for name in KEYPOINT_LINKS]
        self.fp_frame_id = self.model.getFrameId(FOOTPRINT_FRAME)

        self._left_idx_q = [self.model.joints[self.model.getJointId(j)].idx_q for j in LEFT_ARM_JOINTS]
        self._right_idx_q = [self.model.joints[self.model.getJointId(j)].idx_q for j in RIGHT_ARM_JOINTS]

        # `footprint`'s parent joint is the fixed "universe" root (no DOF in the chain between
        # world and footprint -- verified empirically: `model.frames[fp_frame_id].parentJoint == 0`
        # and joint 0 ("universe") has no `idx_q`), so its placement is q-independent and only
        # needs to be computed once.
        q_neutral = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q_neutral)
        pin.updateFramePlacements(self.model, self.data)
        self._fp_placement_inv = self.data.oMf[self.fp_frame_id].inverse()

    def compute(self, state14: np.ndarray) -> np.ndarray:
        """Map one `observation.state[14]` frame to `[14, 3]` footprint-relative keypoints."""
        pin = self._pin
        q = pin.neutral(self.model)
        for idx_q, angle in zip(self._left_idx_q, state14[0:6]):
            q[idx_q] = angle
        for idx_q, angle in zip(self._right_idx_q, state14[7:13]):
            q[idx_q] = angle

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        keypoints = np.empty((NUM_KEYPOINTS, 3), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            rel_placement = self._fp_placement_inv * self.data.oMf[fid]
            keypoints[i] = rel_placement.translation
        return keypoints

    def compute_batch(self, states: np.ndarray) -> np.ndarray:
        """`[N, 14]` -> `[N, 14, 3]`. Not vectorized inside pinocchio (it has no batched FK API
        in the Python bindings we use), but each single-frame FK call is O(tens of microseconds),
        so a plain Python loop over even tens of thousands of frames finishes in a few seconds."""
        out = np.empty((states.shape[0], NUM_KEYPOINTS, 3), dtype=np.float32)
        for i in range(states.shape[0]):
            out[i] = self.compute(states[i])
        return out


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing existing destination %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(
                f"Destination {dest} already exists. Pass --force to overwrite, or pick a new --dest."
            )
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying dataset %s -> %s (rsync -a) ...", source, dest)
    subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)


def _add_keypoint_column_to_parquet_files(dest: Path, extractor: AlohaFKKeypointExtractor) -> dict:
    """Compute + write `observation.keypoint_3d` into every parquet chunk under `dest/data`.

    Returns summary stats (min/max/mean per-axis, over all frames) for a physical-sanity check.
    """
    parquet_files = sorted((dest / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dest / 'data'}")

    all_kpts = []
    total_rows = 0
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)  # [N, 14]
        kpts = extractor.compute_batch(states)  # [N, 14, 3]
        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts]  # each row: [42]
        df.to_parquet(pq_path)
        logger.info("Wrote observation.keypoint_3d for %d rows -> %s", len(df), pq_path)
        all_kpts.append(kpts)
        total_rows += len(df)

    all_kpts = np.concatenate(all_kpts, axis=0)  # [total_rows, 14, 3]
    stats = {
        "total_rows": total_rows,
        "min": all_kpts.reshape(-1, 3).min(axis=0).tolist(),
        "max": all_kpts.reshape(-1, 3).max(axis=0).tolist(),
        "mean": all_kpts.reshape(-1, 3).mean(axis=0).tolist(),
        "per_link_dist_from_footprint_mean": {
            link: float(np.linalg.norm(all_kpts[:, i], axis=-1).mean())
            for i, link in enumerate(KEYPOINT_LINKS)
        },
    }
    return stats


def _update_info_json(dest: Path) -> None:
    from lerobot.datasets.utils import load_json, write_json

    info_path = dest / "meta" / "info.json"
    info = load_json(info_path)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * 3],
        "names": KEYPOINT_FEATURE_NAMES,
        "fps": info.get("fps", 15),
    }
    write_json(info, info_path)
    logger.info("Updated %s with observation.keypoint_3d feature declaration.", info_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=str, default="/mnt/r/DATA/RoboTwin-Clean/stack_bowls_three",
        help="Read-only source LeRobot dataset directory (never modified).",
    )
    parser.add_argument(
        "--dest", type=str, default="/mnt/r/DATA/RoboTwin-Clean-FK/stack_bowls_three",
        help="Working-copy destination directory that will receive the observation.keypoint_3d column.",
    )
    parser.add_argument("--urdf", type=str, default=DEFAULT_ALOHA_URDF, help="Path to the aloha-agilex URDF.")
    parser.add_argument("--force", action="store_true", help="Overwrite --dest if it already exists.")
    parser.add_argument(
        "--skip-copy", action="store_true",
        help="Assume --dest already contains a copy of --source (e.g. from a previous partial run) "
        "and only (re-)run the FK column generation + info.json update.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not args.skip_copy:
        _copy_dataset(source, dest, force=args.force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy given but {dest} does not exist.")
        logger.info("--skip-copy: reusing existing copy at %s", dest)

    logger.info("Loading aloha-agilex URDF from %s ...", args.urdf)
    extractor = AlohaFKKeypointExtractor(args.urdf)

    stats = _add_keypoint_column_to_parquet_files(dest, extractor)
    _update_info_json(dest)

    logger.info("Done. Summary over %d frames:", stats["total_rows"])
    logger.info("  per-axis min  (x,y,z) = %s", stats["min"])
    logger.info("  per-axis max  (x,y,z) = %s", stats["max"])
    logger.info("  per-axis mean (x,y,z) = %s", stats["mean"])
    logger.info("  mean distance from footprint per link:")
    for link, dist in stats["per_link_dist_from_footprint_mean"].items():
        logger.info("    %-14s %.4f m", link, dist)


if __name__ == "__main__":
    main()
