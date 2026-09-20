#!/usr/bin/env python3
"""Offline FK 7D keypoint generation for LIBERO LeRobot v3 datasets (v2).

Reads per-frame arm joints from `observation.state.joint_position` (7D) and
gripper finger positions from `observation.state[6:8]`, builds a 9D qpos,
runs MuJoCo FK on the robosuite Lift MJCF, writes:

    observation.keypoint_3d  shape [K * 7]  (default K=8, px,py,pz,qx,qy,qz,qw)

Uses the same FK engine as the eval-time StandaloneFK
(evaluation/LIBERO2/keypoint_utils.py) to guarantee train-eval consistency.

Usage:
    source /B/VENV/itnvla15rbt20/bin/activate
    export HF_HOME=/B/VENV/hf_home
    python b/s/libplus/generate_libero_kpt_v2.py \
        --source /path/to/libero_merged_lerobot_v3 \
        --dest /B/Dta/libero_merged_kpt_v2 \
        --xml evaluation/panda_robosuite_lift.xml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7  # 3 (pos) + 4 (quat xyzw)
ROBOT_NQPOS = 9
BBOX_MARGIN = 0.15

JOINT_COLUMN = "observation.state.joint_position"
STATE_COLUMN = "observation.state"

EEF_BODY_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")
KEYPOINT_ARM_BODY_NAMES = [
    "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
    "robot0_link5", "robot0_link6", "robot0_link7",
]

EVAL_DEFAULT_R_PAD = 1.8212722539901733


def resolve_eef_body_name(model) -> str:
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except (KeyError, ValueError):
            continue
    raise KeyError(f"No EEF body found (tried {EEF_BODY_CANDIDATES})")


def keypoint_body_names(model) -> list[str]:
    return [*KEYPOINT_ARM_BODY_NAMES, resolve_eef_body_name(model)]


def mujoco_quat_to_xyzw(xquat: np.ndarray) -> np.ndarray:
    """MuJoCo wxyz -> stored xyzw, hemisphere qw >= 0."""
    raw = np.array([xquat[1], xquat[2], xquat[3], xquat[0]], dtype=np.float32)
    if raw[3] < 0:
        raw = -raw
    return raw


class LiftMujocoFK:
    """MuJoCo FK using standalone Lift MJCF.

    Byte-compatible with evaluation/LIBERO2/keypoint_utils.py::StandaloneFK.
    """

    def __init__(self, xml_path: str | Path):
        import mujoco
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.body_names = keypoint_body_names(self.model)
        self.body_ids = {name: self.model.body(name).id for name in self.body_names}

    def compute(self, qpos: np.ndarray) -> np.ndarray:
        """qpos [9] -> keypoints [K, 7] in world frame (px,py,pz,qx,qy,qz,qw).

        Positions are in raw world coordinates (NOT normalized by R_pad).
        """
        n = min(len(qpos), ROBOT_NQPOS)
        self.data.qpos[:n] = qpos[:n]
        self._mujoco.mj_forward(self.model, self.data)

        kpts = np.empty((NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, name in enumerate(self.body_names):
            bid = self.body_ids[name]
            kpts[i, :3] = self.data.xpos[bid]
            kpts[i, 3:7] = mujoco_quat_to_xyzw(self.data.xquat[bid])
        return kpts

    def compute_batch(self, qpos_batch: np.ndarray) -> np.ndarray:
        out = np.empty((len(qpos_batch), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, qpos in enumerate(qpos_batch):
            out[i] = self.compute(qpos)
        return out


def compute_r_pad(global_min: np.ndarray, global_max: np.ndarray,
                  margin: float = BBOX_MARGIN) -> float:
    """Compute R_pad (isotropic normalization radius with margin).

    R_pad = max(|min|, |max|) * (1 + margin).  The margin is INCLUDED.
    Do NOT multiply by (1 + margin) again downstream.
    """
    extremes = np.maximum(np.abs(global_min), np.abs(global_max))
    return float(extremes.max() * (1.0 + margin))


def build_qpos(joint: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Build 9D qpos from joint angles and state vector.

    Same convention as evaluation/LIBERO2/keypoint_utils.py::_get_robot_qpos.
    """
    joint = np.asarray(joint, dtype=np.float64).reshape(-1)
    state = np.asarray(state, dtype=np.float64).reshape(-1)
    if joint.shape[0] != 7:
        raise ValueError(f"Expected 7 arm joints, got {joint.shape}")
    if state.shape[0] < 8:
        raise ValueError(f"Expected state >= 8 dims, got {state.shape}")
    qpos = np.empty(ROBOT_NQPOS, dtype=np.float64)
    qpos[:7] = joint
    qpos[7:9] = state[6:8]
    return qpos


def get_parquet_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return files


def read_frame_kinematics(df: pd.DataFrame, pq_path: Path | None = None) -> np.ndarray | None:
    if len(df) == 0:
        logger.warning("Skipping empty parquet: %s", pq_path)
        return None
    for col in (JOINT_COLUMN, STATE_COLUMN):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing in {pq_path}")
    qpos = np.stack([
        build_qpos(j, s)
        for j, s in zip(df[JOINT_COLUMN].values, df[STATE_COLUMN].values, strict=True)
    ])
    if np.any(np.isnan(qpos)):
        raise ValueError(f"NaN in qpos: {pq_path}")
    return qpos


def pass1_compute_bbox(parquets: list[Path], fk: LiftMujocoFK
                       ) -> tuple[np.ndarray, np.ndarray, int]:
    """Pass 1: scan all frames to find global 3D bounding box."""
    g_min = np.full(3, np.inf, dtype=np.float64)
    g_max = np.full(3, -np.inf, dtype=np.float64)
    total = 0
    quat_err_max = 0.0

    for pq in parquets:
        df = pd.read_parquet(pq, columns=[JOINT_COLUMN, STATE_COLUMN])
        qpos = read_frame_kinematics(df, pq)
        if qpos is None:
            continue
        kpts = fk.compute_batch(qpos)
        pos = kpts[:, :, :3].reshape(-1, 3)
        g_min = np.minimum(g_min, pos.min(axis=0))
        g_max = np.maximum(g_max, pos.max(axis=0))

        q_norms = np.linalg.norm(kpts[:, :, 3:7].reshape(-1, 4), axis=1)
        quat_err_max = max(quat_err_max, float(np.abs(q_norms - 1.0).max()))

        total += len(df)
        logger.info("  Pass 1: %s — %d frames (total: %d)", pq.name, len(df), total)

    if total == 0:
        raise RuntimeError("Pass 1 found zero frames.")
    if quat_err_max > 0.01:
        raise RuntimeError(f"Quaternion norm error {quat_err_max:.4f} > 0.01")
    logger.info("Pass 1 quat norm error max: %.6f", quat_err_max)
    return g_min.astype(np.float32), g_max.astype(np.float32), total


def pass2_write_keypoints(
    parquets: list[Path], fk: LiftMujocoFK, r_pad: float,
    source_data_dir: Path, dest: Path,
) -> int:
    """Pass 2: compute FK, normalize positions, write keypoint column."""
    total = 0
    oob_count = 0

    for pq in parquets:
        rel = pq.relative_to(source_data_dir)
        dest_pq = dest / "data" / rel
        df = pd.read_parquet(dest_pq)
        qpos = read_frame_kinematics(df, dest_pq)
        if qpos is None:
            continue

        kpts = fk.compute_batch(qpos)
        kpts[:, :, :3] /= r_pad

        if (np.abs(kpts[:, :, :3]) > 1.01).any():
            oob_count += 1
            logger.warning("  OOB in %s: max |pos| = %.4f",
                           dest_pq.name, np.abs(kpts[:, :, :3]).max())

        df["observation.keypoint_3d"] = [row.reshape(-1).astype(np.float32) for row in kpts]
        dest_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest_pq, index=False)
        total += len(df)
        logger.info("  Pass 2: %s — %d frames written", dest_pq.name, len(df))

    if oob_count:
        logger.warning("%d files had OOB positions", oob_count)
    return total


def feature_names(body_names: list[str]) -> list[str]:
    return [
        f"{name}_{comp}"
        for name in body_names
        for comp in ("px", "py", "pz", "qx", "qy", "qz", "qw")
    ]


def update_info_json(dest: Path, body_names: list[str]) -> None:
    info_path = dest / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["features"]["observation.keypoint_3d"] = {
        "dtype": "float32",
        "shape": [NUM_KEYPOINTS * KEYPOINT_DIM],
        "names": feature_names(body_names),
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated %s with observation.keypoint_3d [%d]",
                info_path, NUM_KEYPOINTS * KEYPOINT_DIM)


def write_keypoints_meta(
    dest: Path, r_pad: float, g_min: np.ndarray, g_max: np.ndarray,
    total_frames: int, xml_path: Path, margin: float, body_names: list[str],
    xml_md5: str,
) -> None:
    meta = {
        "bbox_radius": r_pad,
        "bbox_margin": margin,
        "global_min_world": g_min.tolist(),
        "global_max_world": g_max.tolist(),
        "normalization": "world_origin_isotropic_r_pad",
        "keypoint_dim": KEYPOINT_DIM,
        "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
        "rotation_representation": "quaternion_xyzw_hemisphere",
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_bodies": body_names,
        "robot_nqpos": ROBOT_NQPOS,
        "qpos_layout": "joint_position[0:7] + observation.state[6:8]",
        "total_frames": total_frames,
        "mjcf_path": str(xml_path.resolve()),
        "mjcf_md5": xml_md5,
        "coordinate_system": "MuJoCo world frame, divided by R_pad",
        "r_pad_includes_margin": True,
        "eval_default_r_pad": EVAL_DEFAULT_R_PAD,
    }
    meta_path = dest / "meta" / "keypoints_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote %s", meta_path)


def copy_dataset(source: Path, dest: Path, force: bool) -> None:
    if dest.exists():
        if force:
            logger.info("Removing %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} exists. --force to overwrite.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        logger.info("rsync %s -> %s", source, dest)
        subprocess.run(["rsync", "-a", f"{source}/", f"{dest}/"], check=True)
    else:
        shutil.copytree(source, dest, symlinks=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="Source LeRobot v3 dataset (must have joint_position column)")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Destination dataset with keypoints")
    parser.add_argument("--xml", type=Path,
                        default=Path("evaluation/panda_robosuite_lift.xml"),
                        help="Lift MJCF exported by export_lift_mjcf.py")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-copy", action="store_true")
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN)
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"Lift MJCF not found: {args.xml}\n"
            "Run: /B/VENV/libero_plus_client/bin/python b/s/libplus/export_lift_mjcf.py"
        )

    xml_md5 = hashlib.md5(args.xml.read_bytes()).hexdigest()
    logger.info("MJCF: %s (MD5: %s)", args.xml, xml_md5)

    source, dest = args.source.resolve(), args.dest.resolve()

    if not args.skip_copy:
        copy_dataset(source, dest, args.force)
    elif not dest.exists():
        raise FileNotFoundError(f"--skip-copy but {dest} missing")

    fk = LiftMujocoFK(args.xml)
    src_parquets = get_parquet_files(source / "data")
    logger.info("Found %d parquet files", len(src_parquets))

    logger.info("=== Pass 1: computing global bounding box ===")
    g_min, g_max, total_p1 = pass1_compute_bbox(src_parquets, fk)
    r_pad = compute_r_pad(g_min, g_max, margin=args.bbox_margin)
    logger.info("Global min: %s", g_min)
    logger.info("Global max: %s", g_max)
    logger.info("R_pad = %.10f (margin=%.0f%%)", r_pad, args.bbox_margin * 100)

    r_pad_diff = abs(r_pad - EVAL_DEFAULT_R_PAD)
    if r_pad_diff > 1e-4:
        logger.warning("R_pad differs from eval default by %.6f!", r_pad_diff)
        logger.warning("  Computed: %.10f", r_pad)
        logger.warning("  Eval default: %.10f", EVAL_DEFAULT_R_PAD)
        logger.warning("  Consider updating DEFAULT_R_PAD in keypoint_utils.py")
    else:
        logger.info("R_pad matches eval default (diff=%.2e)", r_pad_diff)

    logger.info("=== Pass 2: writing 7D keypoints ===")
    total_p2 = pass2_write_keypoints(src_parquets, fk, r_pad, source / "data", dest)

    update_info_json(dest, fk.body_names)
    write_keypoints_meta(
        dest, r_pad, g_min, g_max, total_p2,
        args.xml, args.bbox_margin, fk.body_names, xml_md5,
    )

    logger.info("=== DONE ===")
    logger.info("  Frames: %d (pass1) / %d (pass2)", total_p1, total_p2)
    logger.info("  Output: %s", dest)
    logger.info("  keypoint_3d dim: %d (= %d x %d)",
                NUM_KEYPOINTS * KEYPOINT_DIM, NUM_KEYPOINTS, KEYPOINT_DIM)


if __name__ == "__main__":
    main()
