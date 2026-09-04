"""Offline FK-based 7D keypoint generation for Franka arm.

Two-pass pipeline — see dta_4dtrj_plan.md §5 for normalization design.

Output: observation.keypoint_3d [56] = 8 keypoints x 7D (px,py,pz,qx,qy,qz,qw).

Usage:
    python b/s/Frk/generate_franka_keypoints.py \
        --source /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
        --dest /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
        --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import sys
_UTIL_DIR = Path(__file__).resolve().parents[3] / "util_scripts"
if str(_UTIL_DIR) not in sys.path:
    sys.path.insert(0, str(_UTIL_DIR))
from generate_r1pro_keypoints_e1 import (
    compute_r_pad,
    _copy_dataset,
    _get_parquet_files,
)

DEFAULT_LINK_PREFIX = "fr3v2_1"
DEFAULT_BBOX_MARGIN = 0.15
DEFAULT_NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
STATE_COLUMN = "observation.state.arm"


def make_keypoint_links(prefix: str) -> list[str]:
    return [f"{prefix}_link{i}" for i in range(1, 8)] + [f"{prefix}_hand_tcp"]


def make_joint_names(prefix: str) -> list[str]:
    return [f"{prefix}_joint{i}" for i in range(1, 8)]


def make_feature_names(keypoint_links: list[str]) -> list[str]:
    return [
        f"{link}_{comp}"
        for link in keypoint_links
        for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")
    ]


class FrankaFKExtractor7D:
    def __init__(self, urdf_path: str, link_prefix: str = DEFAULT_LINK_PREFIX,
                 keypoint_links: list[str] | None = None):
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.keypoint_links = keypoint_links or make_keypoint_links(link_prefix)
        self.num_keypoints = len(self.keypoint_links)

        self.frame_ids = []
        for name in self.keypoint_links:
            fid = self.model.getFrameId(name)
            if fid >= self.model.nframes:
                raise ValueError(f"Frame '{name}' not found in URDF {urdf_path}")
            self.frame_ids.append(fid)

        joint_names = make_joint_names(link_prefix)
        self._arm_idx_q: list[int] = []
        for jname in joint_names:
            jid = self.model.getJointId(jname)
            if jid >= len(self.model.joints):
                raise ValueError(f"Joint '{jname}' not found in URDF {urdf_path}")
            nq = self.model.joints[jid].nq
            if nq != 1:
                raise ValueError(f"Joint '{jname}' has nq={nq}, expected 1 (revolute)")
            self._arm_idx_q.append(self.model.joints[jid].idx_q)

        self._q_base = pin.neutral(self.model)

    def compute(self, arm_joints: np.ndarray) -> np.ndarray:
        """[7] -> [num_keypoints, 7] (px,py,pz,qx,qy,qz,qw), hemisphere-normalized."""
        pin = self._pin
        q = self._q_base.copy()
        for idx_q, angle in zip(self._arm_idx_q, arm_joints, strict=True):
            q[idx_q] = float(angle)

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        keypoints = np.empty((self.num_keypoints, KEYPOINT_DIM), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            oMf = self.data.oMf[fid]
            keypoints[i, :3] = oMf.translation
            quat = pin.Quaternion(oMf.rotation)
            raw_q = np.array([quat.x, quat.y, quat.z, quat.w], dtype=np.float32)
            if raw_q[3] < 0:
                raw_q = -raw_q
            keypoints[i, 3:7] = raw_q
        return keypoints

    def compute_batch(self, arm_joints_batch: np.ndarray) -> np.ndarray:
        """[N, 7] -> [N, num_keypoints, 7]."""
        n = arm_joints_batch.shape[0]
        out = np.empty((n, self.num_keypoints, KEYPOINT_DIM), dtype=np.float32)
        for i in range(n):
            out[i] = self.compute(arm_joints_batch[i])
        return out


def _read_arm_joints(df: pd.DataFrame, pq_path: Path | None = None):
    if len(df) == 0:
        logger.warning("Skipping empty parquet (0 rows): %s", pq_path)
        return None
    if STATE_COLUMN not in df.columns:
        raise KeyError(f"Required column '{STATE_COLUMN}' not found in {pq_path}. "
                       f"Available: {list(df.columns)}")
    arm = np.stack(df[STATE_COLUMN].values).astype(np.float64)
    if np.any(np.isnan(arm)):
        raise ValueError(f"NaN in {STATE_COLUMN} in {pq_path}")
    return arm


def pass1_compute_bbox(parquet_files: list[Path], extractor: FrankaFKExtractor7D):
    global_min = np.full(3, np.inf, dtype=np.float64)
    global_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    qw_min, qw_max = 1.0, 0.0
    quat_norm_err_max = 0.0

    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        arm = _read_arm_joints(df, pq_path)
        if arm is None:
            continue

        kpts = extractor.compute_batch(arm)
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

    logger.info("Quaternion stats: qw_min=%.6f, qw_max=%.6f, norm_err_max=%.2e",
                qw_min, qw_max, quat_norm_err_max)
    if qw_min < 0:
        raise RuntimeError(f"Hemisphere normalization failed: qw_min={qw_min:.6f}")
    if quat_norm_err_max > 0.01:
        raise RuntimeError(f"Quaternion norm check failed: max error={quat_norm_err_max:.4f}")

    return global_min.astype(np.float32), global_max.astype(np.float32), total_frames


def pass2_write_keypoints(parquet_files: list[Path], extractor: FrankaFKExtractor7D,
                          r_pad: float, source_data_dir: Path, dest: Path):
    total_frames = 0
    oob_count = 0

    for pq_path in parquet_files:
        rel = pq_path.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(dest_pq)
        arm = _read_arm_joints(df, dest_pq)
        if arm is None:
            continue

        kpts = extractor.compute_batch(arm)
        kpts[:, :, :3] /= r_pad

        if (np.abs(kpts[:, :, :3]) > 1.01).any():
            oob_count += 1
            logger.warning("  Position OOB in %s: max |pos| = %.4f",
                           dest_pq.name, np.abs(kpts[:, :, :3]).max())

        df["observation.keypoint_3d"] = [row.reshape(-1) for row in kpts]
        df.to_parquet(dest_pq)
        total_frames += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if oob_count:
        logger.warning("%d parquet files had OOB positions. Increase --bbox-margin.", oob_count)
    return total_frames


def _update_info_json(dest: Path, num_keypoints: int, keypoint_links: list[str]) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    total_dim = num_keypoints * KEYPOINT_DIM
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [total_dim],
        "names": make_feature_names(keypoint_links),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d [%d].", info_path, total_dim)


def _write_meta(dest: Path, r_pad: float, global_min: np.ndarray,
                global_max: np.ndarray, total_frames: int,
                urdf_path: str, keypoint_links: list[str],
                bbox_margin: float) -> None:
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
        "num_keypoints": len(keypoint_links),
        "keypoint_links": keypoint_links,
        "total_frames": total_frames,
        "coordinate_system": (
            "base_link-relative, position divided by bbox_radius, "
            "quaternion hemisphere-normalized"
        ),
        "urdf": str(urdf_path),
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True,
                        help="Source LeRobot dataset (intermediate, no keypoints).")
    parser.add_argument("--dest", type=str, required=True,
                        help="Destination for the copy with 7D keypoints.")
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[3] / "b" / "d" / "Frk" / "fr3v2_1_franka_hand.urdf"),
                        help="Path to Franka URDF.")
    parser.add_argument("--link-prefix", type=str, default=DEFAULT_LINK_PREFIX,
                        help=f"URDF link/joint name prefix (default: {DEFAULT_LINK_PREFIX}).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite --dest if it exists.")
    parser.add_argument("--skip-copy", action="store_true",
                        help="Reuse existing --dest (e.g. from a partial run).")
    parser.add_argument("--bbox-margin", type=float, default=DEFAULT_BBOX_MARGIN,
                        help=f"Safety margin for R_pad (default: {DEFAULT_BBOX_MARGIN}).")
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

    logger.info("Loading URDF from %s (prefix=%s)", args.urdf, args.link_prefix)
    keypoint_links = make_keypoint_links(args.link_prefix)
    extractor = FrankaFKExtractor7D(args.urdf, link_prefix=args.link_prefix)
    logger.info("Keypoint links: %s", keypoint_links)

    source_parquets = _get_parquet_files(source / "data")
    logger.info("Found %d parquet files", len(source_parquets))

    logger.info("=== Pass 1: computing global bounding box ===")
    global_min, global_max, total_frames_p1 = pass1_compute_bbox(source_parquets, extractor)
    r_pad = compute_r_pad(global_min, global_max, margin=bbox_margin)

    logger.info("Global min (base-rel): %s", global_min)
    logger.info("Global max (base-rel): %s", global_max)
    logger.info("R_pad = %.6f m (margin=%.0f%%)", r_pad, bbox_margin * 100)

    logger.info("=== Pass 2: writing 7D keypoints ===")
    total_frames_p2 = pass2_write_keypoints(
        source_parquets, extractor, r_pad, source / "data", dest,
    )

    _update_info_json(dest, extractor.num_keypoints, keypoint_links)
    _write_meta(dest, r_pad, global_min, global_max, total_frames_p2,
                args.urdf, keypoint_links, bbox_margin)

    total_dim = extractor.num_keypoints * KEYPOINT_DIM
    logger.info("=== DONE ===")
    logger.info("  Frames: %d (Pass 1) / %d (Pass 2)", total_frames_p1, total_frames_p2)
    logger.info("  Output: %s", dest)
    logger.info("  keypoint_dim: %d (= %d keypoints x %d per point)",
                total_dim, extractor.num_keypoints, KEYPOINT_DIM)
    logger.info("  R_pad: %.6f m", r_pad)


if __name__ == "__main__":
    main()
