"""Offline FK-based 7D keypoint generation for R1 Pro — E1 scheme (position + quaternion).

Two-pass pipeline:
  Pass 1: FK all frames → collect global position bounding box → compute R_pad
  Pass 2: position / R_pad, quaternion hemisphere-normalize → write parquet

Output: observation.keypoint_3d [112] = 16 keypoints × 7D (px,py,pz,qx,qy,qz,qw).

Position normalization: isotropic division by R_pad = max(|all axis extremes|) × (1 + margin).
Quaternion normalization: hemisphere constraint (qw >= 0).

See dta_3dtrj_E2.md for design rationale.

Usage:
    python util_scripts/generate_r1pro_keypoints_e1.py \
        --source /home/luogang/DATA/elevator0714_lerobot \
        --dest /home/luogang/DATA/elevator0714_lerobot_4D

TORSO CONVENTION: same as generate_r1pro_keypoints.py — see that file's docstring.
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
KEYPOINT_DIM = 7  # 3 (pos) + 4 (quat xyzw)
BBOX_MARGIN = 0.15  # 15% safety margin for R_pad

KEYPOINT_LINKS: list[str] = [
    "left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4",
    "left_arm_link5", "left_arm_link6", "left_arm_link7", "left_gripper_link",
    "right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4",
    "right_arm_link5", "right_arm_link6", "right_arm_link7", "right_gripper_link",
]

LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
TORSO_JOINTS = [f"torso_joint{i}" for i in range(1, 5)]

TORSO_Q_DEFAULT = (0.0, 0.0, 0.0, 0.0)
TORSO_Q_PHYSICAL = (0.8, -1.4, -0.60, 0.0)

KEYPOINT_FEATURE_NAMES: list[str] = [
    f"{link}_{comp}"
    for link in KEYPOINT_LINKS
    for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")
]


class R1ProFKExtractorE1:
    """FK extractor producing 7D keypoints: [px, py, pz, qx, qy, qz, qw] per link."""

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
                        f"Joint '{jname}' ({label}) has nq={nq}, expected 1."
                    )
                idx_list.append(self.model.joints[jid].idx_q)

        self.torso_q = np.asarray(torso_q, dtype=np.float64)
        if self.torso_q.shape != (4,):
            raise ValueError(f"torso_q must have 4 entries, got {self.torso_q.shape}")
        self._q_base = pin.neutral(self.model)
        for idx_q, angle in zip(self._torso_idx_q, self.torso_q, strict=True):
            self._q_base[idx_q] = float(angle)

    def compute(self, left_arm: np.ndarray, right_arm: np.ndarray) -> np.ndarray:
        """Compute 16 keypoints as 7D (position + quaternion) in base_link frame.

        Returns: [16, 7] float32 — [px, py, pz, qx, qy, qz, qw] per keypoint.
        Quaternions are hemisphere-normalized (qw >= 0).
        """
        pin = self._pin
        q = self._q_base.copy()
        for idx_q, angle in zip(self._left_idx_q, left_arm, strict=True):
            q[idx_q] = float(angle)
        for idx_q, angle in zip(self._right_idx_q, right_arm, strict=True):
            q[idx_q] = float(angle)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        keypoints = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            oMf = self.data.oMf[fid]
            keypoints[i, :3] = oMf.translation
            quat = pin.Quaternion(oMf.rotation)
            raw_q = np.array([quat.x, quat.y, quat.z, quat.w],
                             dtype=np.float32)
            if raw_q[3] < 0:
                raw_q = -raw_q
            keypoints[i, 3:7] = raw_q
        return keypoints

    def compute_batch(self, left_arms: np.ndarray, right_arms: np.ndarray) -> np.ndarray:
        """[N, 7], [N, 7] -> [N, 16, 7]."""
        n = left_arms.shape[0]
        out = np.empty((n, NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i in range(n):
            out[i] = self.compute(left_arms[i], right_arms[i])
        return out


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
            raise KeyError(f"Required column '{col}' not found in {pq_path}.")
    left = np.stack(df["observation.state.left_arm"].values).astype(np.float64)
    right = np.stack(df["observation.state.right_arm"].values).astype(np.float64)
    for name, arr in [("left_arm", left), ("right_arm", right)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"NaN in observation.state.{name} in {pq_path}")
    return left, right


def _read_recorded_torso(df: pd.DataFrame, pq_path: Path) -> np.ndarray | None:
    col = "observation.state.torso"
    if col not in df.columns:
        return None
    torso = np.stack(df[col].to_numpy()).astype(np.float64)
    spread = float(np.abs(torso - torso[0]).max())
    if spread > 1e-6:
        raise ValueError(
            f"{col} varies within {pq_path} (max deviation {spread:.6f} rad)."
        )
    return torso[0]


def pass1_compute_bbox(parquet_files: list[Path], extractor: R1ProFKExtractorE1):
    """Pass 1: FK all frames, collect global position min/max (quaternion is self-normalized)."""
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    recorded_torso = None

    qw_min, qw_max = 1.0, 0.0
    quat_norm_err_max = 0.0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        left, right = _read_joint_angles(df, pq_path)
        if left is None:
            continue
        torso = _read_recorded_torso(df, pq_path)
        if torso is not None:
            if recorded_torso is not None and not np.allclose(recorded_torso, torso, atol=1e-6):
                raise ValueError(f"Torso pose differs between parquet files")
            recorded_torso = torso

        kpts = extractor.compute_batch(left, right)  # [N, 16, 7]
        pos = kpts[:, :, :3]
        quat = kpts[:, :, 3:7]

        frame_min = pos.reshape(-1, 3).min(axis=0)
        frame_max = pos.reshape(-1, 3).max(axis=0)
        global_min = np.minimum(global_min, frame_min)
        global_max = np.maximum(global_max, frame_max)

        qw_vals = quat[:, :, 3]
        qw_min = min(qw_min, float(qw_vals.min()))
        qw_max = max(qw_max, float(qw_vals.max()))
        quat_norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
        quat_norm_err_max = max(quat_norm_err_max, float(np.abs(quat_norms - 1.0).max()))

        total_frames += len(df)
        logger.info("  Pass 1: %s — %d frames (total: %d)", pq_path.name, len(df), total_frames)

    if recorded_torso is not None and not np.allclose(recorded_torso, extractor.torso_q, atol=1e-6):
        logger.warning(
            "Dataset records torso=%s but FK used torso_q=%s. See generate_r1pro_keypoints.py docstring.",
            np.array2string(recorded_torso, precision=4),
            np.array2string(extractor.torso_q, precision=4),
        )

    logger.info("Quaternion stats: qw_min=%.6f, qw_max=%.6f, norm_err_max=%.2e",
                qw_min, qw_max, quat_norm_err_max)
    if qw_min < 0:
        logger.error("FATAL: qw_min=%.6f < 0, hemisphere normalization failed!", qw_min)
        raise RuntimeError("Hemisphere normalization failed")
    if quat_norm_err_max > 0.01:
        logger.error("FATAL: quaternion norm error %.4f > 0.01, FK or Pinocchio issue!", quat_norm_err_max)
        raise RuntimeError("Quaternion norm check failed")

    return global_min.astype(np.float32), global_max.astype(np.float32), total_frames


def compute_r_pad(global_min: np.ndarray, global_max: np.ndarray, margin: float = BBOX_MARGIN) -> float:
    """Isotropic bounding radius with safety margin.

    R = max(|x_min|, x_max, |y_min|, y_max, |z_min|, z_max)
    R_pad = R × (1 + margin)
    """
    abs_extremes = np.maximum(np.abs(global_min), np.abs(global_max))
    R = float(abs_extremes.max())
    R_pad = R * (1.0 + margin)
    return R_pad


def pass2_write_keypoints(parquet_files: list[Path], extractor: R1ProFKExtractorE1,
                          r_pad: float, source_data_dir: Path, dest: Path):
    """Pass 2: FK → normalize → write observation.keypoint_3d [112] into parquet."""
    total_frames = 0
    oob_count = 0
    quat_err_count = 0

    for pq_path in parquet_files:
        rel = pq_path.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(dest_pq)
        left, right = _read_joint_angles(df, dest_pq)
        if left is None:
            continue

        kpts = extractor.compute_batch(left, right)  # [N, 16, 7]

        kpts[:, :, :3] /= r_pad

        pos_oob = (np.abs(kpts[:, :, :3]) > 1.01).any()
        if pos_oob:
            oob_count += 1
            logger.warning("  Position OOB in %s: max |pos| = %.4f",
                           dest_pq.name, np.abs(kpts[:, :, :3]).max())

        quat_norms = np.linalg.norm(kpts[:, :, 3:7].reshape(-1, 4), axis=1)
        quat_err = np.abs(quat_norms - 1.0).max()
        if quat_err > 0.01:
            quat_err_count += 1
            logger.warning("  Quaternion norm error in %s: max = %.6f", dest_pq.name, quat_err)

        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts]
        df.to_parquet(dest_pq)
        total_frames += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if oob_count:
        logger.warning("%d parquet files had OOB positions. Increase BBOX_MARGIN.", oob_count)
    if quat_err_count:
        logger.warning("%d parquet files had quaternion norm errors.", quat_err_count)
    return total_frames


def _copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing existing %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} exists. Use --force to overwrite.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Copying %s -> %s ...", source, dest)
    subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)


def _update_info_json(dest: Path) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * KEYPOINT_DIM],
        "names": KEYPOINT_FEATURE_NAMES,
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d [%d].", info_path, NUM_KEYPOINTS * KEYPOINT_DIM)


def _write_meta(dest: Path, r_pad: float, global_min: np.ndarray,
                global_max: np.ndarray, total_frames: int, torso_q: np.ndarray,
                urdf_path: str, bbox_margin: float = BBOX_MARGIN) -> None:
    meta = {
        "bbox_radius": r_pad,
        "bbox_margin": bbox_margin,
        "global_min_base_relative": global_min.tolist(),
        "global_max_base_relative": global_max.tolist(),
        "normalization": "base_link_origin_isotropic",
        "keypoint_dim": KEYPOINT_DIM,
        "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
        "rotation_representation": "quaternion_xyzw_hemisphere",
        "rotation_convention": "qw >= 0; negate if qw < 0",
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_links": KEYPOINT_LINKS,
        "total_frames": total_frames,
        "coordinate_system": (
            "base_link-relative, position divided by bbox_radius, "
            "quaternion hemisphere-normalized"
        ),
        "torso_q": torso_q.tolist(),
        "torso_q_physical": list(TORSO_Q_PHYSICAL),
        "torso_q_note": (
            "torso_q is what FK used. Inference MUST reuse this value. "
            "Do NOT substitute the live torso encoder reading. "
            "See generate_r1pro_keypoints.py docstring for the full explanation."
        ),
        "urdf": str(urdf_path),
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Source LeRobot dataset directory (read-only).")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination for the copy with 7D keypoints.")
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "assets" / "r1_pro_with_gripper.urdf"),
                        help="Path to R1 Pro URDF.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --dest if it exists.")
    parser.add_argument("--skip-copy", action="store_true",
                        help="Reuse existing --dest (e.g. from a partial run).")
    parser.add_argument("--torso-q", type=float, nargs=4, default=list(TORSO_Q_DEFAULT),
                        metavar=("J1", "J2", "J3", "J4"),
                        help="Torso joint angles (rad) baked into FK.")
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN,
                        help=f"Safety margin for R_pad (default: {BBOX_MARGIN}).")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    bbox_margin = args.bbox_margin

    if not args.skip_copy:
        _copy_dataset(source, dest, args.force)
    else:
        if not dest.exists():
            raise FileNotFoundError(f"--skip-copy but {dest} does not exist.")
        logger.info("--skip-copy: reusing %s", dest)

    logger.info("Loading URDF from %s", args.urdf)
    logger.info("FK torso_q = %s", args.torso_q)
    extractor = R1ProFKExtractorE1(args.urdf, torso_q=args.torso_q)

    source_parquets = _get_parquet_files(source / "data")
    logger.info("Found %d parquet files", len(source_parquets))

    logger.info("=== Pass 1: computing global bounding box ===")
    global_min, global_max, total_frames_p1 = pass1_compute_bbox(source_parquets, extractor)
    r_pad = compute_r_pad(global_min, global_max, margin=bbox_margin)

    logger.info("Global min (base-rel): %s", global_min)
    logger.info("Global max (base-rel): %s", global_max)
    logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, bbox_margin * 100)
    logger.info("Position range after normalization: [%.4f, %.4f]",
                -np.abs(global_min).max() / r_pad, np.abs(global_max).max() / r_pad)

    logger.info("=== Pass 2: writing 7D keypoints ===")
    total_frames_p2 = pass2_write_keypoints(
        source_parquets, extractor, r_pad, source / "data", dest
    )

    _update_info_json(dest)
    _write_meta(dest, r_pad, global_min, global_max, total_frames_p2,
                extractor.torso_q, args.urdf, bbox_margin=bbox_margin)

    logger.info("=== DONE ===")
    logger.info("  Frames: %d (Pass 1) / %d (Pass 2)", total_frames_p1, total_frames_p2)
    logger.info("  Output: %s", dest)
    logger.info("  keypoint_dim: %d (= %d keypoints × %d per point)",
                NUM_KEYPOINTS * KEYPOINT_DIM, NUM_KEYPOINTS, KEYPOINT_DIM)
    logger.info("  R_pad: %.6f m", r_pad)


if __name__ == "__main__":
    main()
