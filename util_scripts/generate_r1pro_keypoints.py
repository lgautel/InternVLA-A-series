"""Offline FK-based 3D keypoint generation for R1 Pro (dual-arm mobile robot).

Two-pass pipeline:
  Pass 1: FK all frames → collect global bounding box → compute voxel-space offset
  Pass 2: apply offset → validate range → write `observation.keypoint_3d` into parquet

The output dataset is a copy of the source with the new column added.

`--dest` MUST live under `HF_LEROBOT_HOME`, because both the norm-stats script and
training resolve a dataset purely from its `repo_id`:
`LeRobotDataset.root = HF_LEROBOT_HOME / repo_id` (`lerobot/datasets/lerobot_dataset.py`).
A dataset written anywhere else is invisible to `--dataset.repo_id`.

Usage:
    export HF_LEROBOT_HOME=/tmp/hf_home/lerobot
    python util_scripts/generate_r1pro_keypoints.py \
        --source ~/openpi-datasets/open0630_mj_clean \
        --dest "${HF_LEROBOT_HOME}/open0630_mj_clean_kpt16"

TORSO CONVENTION -- read this before touching `--torso-q`
---------------------------------------------------------
Both arms hang off `torso_link4`, so the torso joints rigidly transform all 16
keypoints. Whatever `torso_q` is used here becomes part of the keypoint coordinate
system and MUST be reproduced bit-for-bit at inference time; it is recorded in
`meta/keypoints_meta.json` for exactly that reason.

The default is all-zeros, and for the open0630 datasets that costs nothing. The physical
torso is held at [0.8, -1.4, -0.60, 0.0] rad during the task (the recorded torso column
is an all-zero placeholder, not a measurement), and that pose keeps the arm mounting
plate level: torso_joint1/2 rotate about +Y and torso_joint3 about -Y, so the net pitch
is 0.8 - 1.4 - (-0.60) = 0, and the yaw joint is at 0. Measured on the URDF, it differs
from the zero pose by a pure 21.0 cm translation with 9.5e-15 degrees of rotation.

Because `compute_auto_offset` re-centres the measured bounding box, that translation
cancels exactly: across 3000 random arm poses, zero-torso and real-torso FK produce
voxel coordinates agreeing to 8.9e-16 m. The two choices yield bit-identical training
data, so there is nothing to gain by switching -- and note this equivalence relies on
the plate staying level. A pose with non-zero net pitch or yaw would leave a genuine
rotation that no offset can absorb.

The failure mode this guards against is at DEPLOYMENT, where the cancellation does NOT
happen: inference reads the stored `coord_offset` instead of recomputing it, so a
torso_q mismatch shifts every keypoint by the full 21.0 cm, silently. The robot's torso
encoder reports the real pose and feeding it to FK is the natural thing to do, which is
exactly why it must not be done. Read `torso_q` back from `keypoints_meta.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 16
KEYPOINT_LINKS: list[str] = [
    "left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4",
    "left_arm_link5", "left_arm_link6", "left_arm_link7", "left_gripper_link",
    "right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4",
    "right_arm_link5", "right_arm_link6", "right_arm_link7", "right_gripper_link",
]

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_joint{i}" for i in range(1, 5)]

VOXEL_CENTER = np.array([0.8, 0.8, 0.5], dtype=np.float32)
VOXEL_MIN = np.array([0.0, 0.0, 0.0], dtype=np.float32)
VOXEL_MAX = np.array([1.6, 1.6, 1.0], dtype=np.float32)

# See the module docstring. Zeros and TORSO_Q_PHYSICAL are provably interchangeable here
# (the difference is a pure translation that compute_auto_offset cancels), so the default
# stays at zeros; what matters is that inference uses whatever ends up in
# meta/keypoints_meta.json.
TORSO_Q_DEFAULT = (0.0, 0.0, 0.0, 0.0)

# The pose the operator sets before each run. Recorded for provenance -- it is not the
# default, and switching to it would change nothing.
TORSO_Q_PHYSICAL = (0.8, -1.4, -0.60, 0.0)

KEYPOINT_FEATURE_NAMES: list[str] = [
    f"{link}_{axis}" for link in KEYPOINT_LINKS for axis in ("x", "y", "z")
]


class R1ProFKExtractor:
    def __init__(self, urdf_path: str, torso_q=TORSO_Q_DEFAULT):
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.frame_ids = [self.model.getFrameId(name) for name in KEYPOINT_LINKS]
        for i, fid in enumerate(self.frame_ids):
            if fid >= self.model.nframes:
                raise ValueError(f"Frame '{KEYPOINT_LINKS[i]}' not found in URDF")

        self._left_idx_q = []
        self._right_idx_q = []
        self._torso_idx_q = []
        for joints_list, idx_list, label in [
            (LEFT_ARM_JOINTS, self._left_idx_q, "left_arm"),
            (RIGHT_ARM_JOINTS, self._right_idx_q, "right_arm"),
            (TORSO_JOINTS, self._torso_idx_q, "torso"),
        ]:
            for jname in joints_list:
                jid = self.model.getJointId(jname)
                nq = self.model.joints[jid].nq
                if nq != 1:
                    raise ValueError(
                        f"Joint '{jname}' ({label}) has nq={nq}, expected 1. "
                        "RUBY/continuous joints require SE2 parameterization."
                    )
                idx_list.append(self.model.joints[jid].idx_q)

        # The torso holds one pose for the whole dataset, so bake it into the base
        # configuration once instead of rewriting those 4 entries on every frame.
        self.torso_q = np.asarray(torso_q, dtype=np.float64)
        if self.torso_q.shape != (4,):
            raise ValueError(f"torso_q must have 4 entries, got shape {self.torso_q.shape}")
        self._q_base = pin.neutral(self.model)
        for idx_q, angle in zip(self._torso_idx_q, self.torso_q, strict=True):
            self._q_base[idx_q] = float(angle)

    def compute(self, left_arm: np.ndarray, right_arm: np.ndarray) -> np.ndarray:
        """Compute 16 keypoints in base_link-relative coordinates.

        Args:
            left_arm: [7] joint angles
            right_arm: [7] joint angles

        Returns:
            [16, 3] float32, base_link-relative, at the torso pose fixed in __init__
        """
        pin = self._pin
        q = self._q_base.copy()

        for idx_q, angle in zip(self._left_idx_q, left_arm, strict=True):
            q[idx_q] = float(angle)
        for idx_q, angle in zip(self._right_idx_q, right_arm, strict=True):
            q[idx_q] = float(angle)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        keypoints = np.empty((NUM_KEYPOINTS, 3), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            keypoints[i] = self.data.oMf[fid].translation
        return keypoints

    def compute_batch(self, left_arms: np.ndarray, right_arms: np.ndarray) -> np.ndarray:
        """[N, 7], [N, 7] -> [N, 16, 3]. Measured at ~34k frames/s, so a plain Python
        loop covers the whole 383k-frame dataset in about 11 seconds per pass."""
        n = left_arms.shape[0]
        out = np.empty((n, NUM_KEYPOINTS, 3), dtype=np.float32)
        for i in range(n):
            out[i] = self.compute(left_arms[i], right_arms[i])
        return out


def compute_auto_offset(global_min: np.ndarray, global_max: np.ndarray) -> np.ndarray:
    workspace_center = (global_min + global_max) / 2.0
    return (workspace_center - VOXEL_CENTER).astype(np.float32)


def _get_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return files


def _read_joint_angles(df: pd.DataFrame, pq_path: Path | None = None):
    if len(df) == 0:
        logger.warning("Skipping empty parquet (0 rows): %s", pq_path)
        return None, None
    for col in ("observation.state.left_arm", "observation.state.right_arm"):
        if col not in df.columns:
            raise KeyError(
                f"Required column '{col}' not found in {pq_path}. "
                f"Available: {list(df.columns)}"
            )
    left = np.stack(df["observation.state.left_arm"].values).astype(np.float64)
    right = np.stack(df["observation.state.right_arm"].values).astype(np.float64)
    for name, arr in [("left_arm", left), ("right_arm", right)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN in observation.state.{name} in {pq_path}")
    return left, right


def _read_recorded_torso(df: pd.DataFrame, pq_path: Path) -> np.ndarray | None:
    """Return the dataset's torso pose, erroring out if it is not constant.

    A varying torso column would break the "one fixed rigid transform" argument in the
    module docstring: the keypoint frame would drift within an episode, producing
    plausible-looking but meaningless trajectories. Better to stop than to ship those.
    """
    col = "observation.state.torso"
    if col not in df.columns:
        return None
    torso = np.stack(df[col].to_numpy()).astype(np.float64)
    spread = float(np.abs(torso - torso[0]).max())
    if spread > 1e-6:
        raise ValueError(
            f"{col} varies within {pq_path} (max deviation {spread:.6f} rad). This script "
            "assumes a constant torso pose, so the keypoints it would produce are not in a "
            "single coordinate frame. Re-derive the keypoint pipeline before continuing."
        )
    return torso[0]


def pass1_compute_bbox(parquet_files: list[Path], extractor: R1ProFKExtractor):
    """Pass 1: FK all frames, collect global min/max and the dataset's recorded torso."""
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    recorded_torso = None

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        left, right = _read_joint_angles(df, pq_path)
        if left is None:
            continue
        torso = _read_recorded_torso(df, pq_path)
        if torso is not None:
            if recorded_torso is not None and not np.allclose(recorded_torso, torso, atol=1e-6):
                raise ValueError(
                    f"Torso pose differs between parquet files: {recorded_torso} vs {torso} "
                    f"({pq_path}). The keypoints would span two coordinate frames."
                )
            recorded_torso = torso
        kpts = extractor.compute_batch(left, right)
        frame_min = kpts.reshape(-1, 3).min(axis=0)
        frame_max = kpts.reshape(-1, 3).max(axis=0)
        global_min = np.minimum(global_min, frame_min)
        global_max = np.maximum(global_max, frame_max)
        total_frames += len(df)
        logger.info("  Pass 1: %s — %d frames (running total: %d)", pq_path.name, len(df), total_frames)

    if recorded_torso is not None and not np.allclose(recorded_torso, extractor.torso_q, atol=1e-6):
        logger.warning(
            "Dataset records torso=%s but FK used torso_q=%s. For the open0630 datasets the "
            "recorded column is an all-zero placeholder rather than the true pose, so this is "
            "expected -- but it means the keypoint frame is the one you passed, not the one in "
            "the data. Inference must use torso_q from meta/keypoints_meta.json.",
            np.array2string(recorded_torso, precision=4),
            np.array2string(extractor.torso_q, precision=4),
        )

    return global_min.astype(np.float32), global_max.astype(np.float32), total_frames


def pass2_write_keypoints(parquet_files: list[Path], extractor: R1ProFKExtractor,
                          offset: np.ndarray, source_data_dir: Path, dest: Path):
    """Pass 2: FK + offset → write observation.keypoint_3d into parquet."""
    total_frames = 0
    warn_count = 0

    for pq_path in parquet_files:
        rel = pq_path.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(dest_pq)
        left, right = _read_joint_angles(df, dest_pq)
        if left is None:
            continue
        kpts = extractor.compute_batch(left, right)
        kpts_voxel = kpts - offset[np.newaxis, np.newaxis, :]

        oob = (kpts_voxel.reshape(-1, 3) < VOXEL_MIN).any() or (kpts_voxel.reshape(-1, 3) > VOXEL_MAX).any()
        if oob:
            warn_count += 1
            frame_min = kpts_voxel.reshape(-1, 3).min(axis=0)
            frame_max = kpts_voxel.reshape(-1, 3).max(axis=0)
            logger.warning("  OOB in %s: min=%s max=%s", dest_pq.name, frame_min, frame_max)

        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts_voxel]
        df.to_parquet(dest_pq)
        total_frames += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if warn_count:
        logger.warning("WARNING: %d parquet files had out-of-bound keypoints. "
                       "Consider widening voxel space bounds.", warn_count)
    return total_frames


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing existing destination %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} already exists. Use --force to overwrite.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying dataset %s -> %s ...", source, dest)
    subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)


def _update_info_json(dest: Path) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float64",
        "shape": [NUM_KEYPOINTS * 3],
        "names": KEYPOINT_FEATURE_NAMES,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d feature.", info_path)


def _write_meta(dest: Path, offset: np.ndarray, global_min: np.ndarray,
                global_max: np.ndarray, total_frames: int, torso_q: np.ndarray,
                urdf_path: str) -> None:
    """Record everything inference needs to reproduce this exact keypoint frame.

    `torso_q` and `urdf` are here because they silently define the coordinate system:
    change either one and the same joint angles map to different keypoints.
    """
    meta = {
        "coord_offset": offset.tolist(),
        "global_min_base_relative": global_min.tolist(),
        "global_max_base_relative": global_max.tolist(),
        "voxel_center": VOXEL_CENTER.tolist(),
        "voxel_bounds": [VOXEL_MIN.tolist(), VOXEL_MAX.tolist()],
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_links": KEYPOINT_LINKS,
        "total_frames": total_frames,
        "coordinate_system": "base_link-relative + offset (voxel space)",
        "torso_q": torso_q.tolist(),
        "torso_q_physical": list(TORSO_Q_PHYSICAL),
        "torso_q_note": (
            "torso_q is what FK used and what inference MUST reuse. Do NOT substitute the live "
            "torso encoder, and do NOT substitute torso_q_physical: offline extraction cancels "
            "the difference by recomputing coord_offset, but inference reads coord_offset from "
            "this file instead, so a mismatch shifts every keypoint by ~21 cm without erroring."
        ),
        "urdf": str(urdf_path),
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, required=True,
                        help="Source LeRobot dataset directory (read-only).")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination for the copy with keypoints. Must be "
                             "$HF_LEROBOT_HOME/<repo_id>, since training and the norm-stats "
                             "script locate a dataset only through that path.")
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "assets" / "r1_pro_with_gripper.urdf"),
                        help="Path to R1 Pro URDF.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --dest if it exists.")
    parser.add_argument("--skip-copy", action="store_true",
                        help="Reuse existing --dest (e.g. from a partial run).")
    parser.add_argument("--torso-q", type=float, nargs=4, default=list(TORSO_Q_DEFAULT),
                        metavar=("J1", "J2", "J3", "J4"),
                        help="Torso joint angles (rad) baked into FK. Defines the keypoint "
                             "coordinate frame; see the module docstring before changing it. "
                             "Recorded in meta/keypoints_meta.json for inference to mirror.")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not args.skip_copy:
        _copy_dataset(source, dest, args.force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy but {dest} does not exist.")
        logger.info("--skip-copy: reusing %s", dest)

    logger.info("Loading R1 Pro URDF from %s ...", args.urdf)
    logger.info("FK torso_q = %s (keypoint frame; inference must mirror this)", args.torso_q)
    extractor = R1ProFKExtractor(args.urdf, torso_q=args.torso_q)

    source_parquets = _get_parquet_files(source / "data")

    logger.info("=== Pass 1: computing global bounding box ===")
    global_min, global_max, total_frames = pass1_compute_bbox(source_parquets, extractor)
    offset = compute_auto_offset(global_min, global_max)
    voxel_min = global_min - offset
    voxel_max = global_max - offset

    logger.info("Global min (base-rel): %s", global_min)
    logger.info("Global max (base-rel): %s", global_max)
    logger.info("Auto offset: %s", offset)
    logger.info("Voxel-space min: %s", voxel_min)
    logger.info("Voxel-space max: %s", voxel_max)

    # Check all three axes, not just z. The reachable workspace is widest in y (measured
    # span ~1.96 m against the voxel's 1.6 m), so y is the axis most likely to overflow.
    for axis, name in enumerate("xyz"):
        if voxel_min[axis] < VOXEL_MIN[axis] or voxel_max[axis] > VOXEL_MAX[axis]:
            logger.warning(
                "%s-axis leaves the voxel space: [%.3f, %.3f] vs allowed [%.1f, %.1f]. "
                "GeoPredict's pretrained prior assumes keypoints live inside these bounds; "
                "either widen the voxel space or accept a prior mismatch on this axis.",
                name, voxel_min[axis], voxel_max[axis], VOXEL_MIN[axis], VOXEL_MAX[axis],
            )

    logger.info("=== Pass 2: writing keypoints with offset ===")
    frames_written = pass2_write_keypoints(source_parquets, extractor, offset, source / "data", dest)

    _update_info_json(dest)
    _write_meta(dest, offset, global_min, global_max, frames_written,
                extractor.torso_q, args.urdf)

    logger.info("Done. %d frames written across %d parquet files.",
                frames_written, len(source_parquets))


if __name__ == "__main__":
    main()
