"""Convert Franka put_cube_into_box HDF5 dataset to LeRobot format.

V3: ee_quat2 replaces raw ee_quat. Computed via FK from joint_positions,
hemisphere-normalized (qw >= 0), xyzw format.

Usage:
    source /B/VENV/itnvla15rbt20/bin/activate
    export HF_HOME=/B/VENV/hf_home
    python b/s/Frk3/convert_cubinbx_hdf5.py \
        --source /B/Dta/put_cube_into_box/put_cube_into_box_hdf5 \
        --dest /home/a26113/b/Dta/put_cube_into_box_lrb3 \
        --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf \
        --fps 30 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROBOT_TYPE = "franka_cubinbx"
DEFAULT_TARGET_FPS = 30
DEFAULT_IMAGE_SHAPE = (480, 640, 3)
CAMERA_NAMES = ["global", "wrist"]
CAMERA_HDF5_GROUPS = ["camera_global", "camera_wrist"]
FK_PREFIX = "fr3v2_1"


class FKQuatComputer:
    """Compute ee_quat2 from joint_positions via pinocchio FK."""

    def __init__(self, urdf_path: str):
        import pinocchio as pin
        self._pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.tcp_fid = self.model.getFrameId(f"{FK_PREFIX}_hand_tcp")
        joint_names = [f"{FK_PREFIX}_joint{i}" for i in range(1, 8)]
        self.arm_idx_q = [
            self.model.joints[self.model.getJointId(jn)].idx_q
            for jn in joint_names
        ]
        self.q_base = pin.neutral(self.model)

    def compute_batch(self, joint_positions: np.ndarray) -> np.ndarray:
        """(N, 7) -> (N, 4) xyzw hemisphere-normalized."""
        pin = self._pin
        N = len(joint_positions)
        result = np.empty((N, 4), dtype=np.float32)
        for i in range(N):
            q = self.q_base.copy()
            for idx_q, angle in zip(self.arm_idx_q, joint_positions[i]):
                q[idx_q] = float(angle)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.tcp_fid]
            fk_quat = pin.Quaternion(oMf.rotation)
            raw = np.array([fk_quat.x, fk_quat.y, fk_quat.z, fk_quat.w],
                           dtype=np.float32)
            if raw[3] < 0:
                raw = -raw
            result[i] = raw
        return result


def build_features(image_shape: tuple[int, ...] = DEFAULT_IMAGE_SHAPE,
                   use_videos: bool = True) -> dict:
    mode = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32", "shape": (15,),
            "names": {
                "state": [
                    "joint1", "joint2", "joint3", "joint4",
                    "joint5", "joint6", "joint7",
                    "gripper_width",
                    "ee_pos_x", "ee_pos_y", "ee_pos_z",
                    "ee_quat2_x", "ee_quat2_y", "ee_quat2_z", "ee_quat2_w",
                ],
            },
        },
        "action": {
            "dtype": "float32", "shape": (8,),
            "names": {
                "actions": [
                    "joint1", "joint2", "joint3", "joint4",
                    "joint5", "joint6", "joint7",
                    "gripper_cmd",
                ],
            },
        },
    }
    for cam_name in CAMERA_NAMES:
        features[f"observation.images.{cam_name}"] = {
            "dtype": mode, "shape": image_shape,
            "names": ["height", "width", "rgb"],
        }
    return features


def align_timestamps(state_ts: np.ndarray, camera_ts: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(state_ts, camera_ts)
    indices = np.clip(indices, 1, len(state_ts) - 1)
    left_diff = np.abs(state_ts[indices - 1] - camera_ts)
    right_diff = np.abs(state_ts[indices] - camera_ts)
    mask = left_diff <= right_diff
    indices[mask] -= 1
    return indices


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process_episode(
    hdf5_path: Path,
    dataset,
    task_str: str,
    fk_computer: FKQuatComputer,
    reference_camera: str = "camera_global",
) -> int:
    with h5py.File(hdf5_path, "r") as f:
        state_ts = f["robot_state/timestamps"][:]
        joint_pos = f["robot_state/joint_positions"][:]
        gripper_w = f["robot_state/gripper_width"][:]
        ee_pos = f["robot_state/ee_pos"][:]
        action_j = f["robot_state/action_joints"][:]
        action_g = f["robot_state/action_gripper"][:]

        ref_cam_ts = f[f"{reference_camera}/timestamps"][:]
        state_indices = align_timestamps(state_ts, ref_cam_ts)

        time_diffs = np.abs(state_ts[state_indices] - ref_cam_ts)
        max_diff_ms = time_diffs.max() * 1000
        if max_diff_ms > 20:
            logger.warning(
                "  Large state-camera alignment gap: max %.1f ms in %s",
                max_diff_ms, hdf5_path.name,
            )

        cam_images = {}
        for cam_group, cam_name in zip(CAMERA_HDF5_GROUPS, CAMERA_NAMES):
            if cam_group in f:
                cam_images[cam_name] = f[f"{cam_group}/color_image_jpeg"]
            else:
                logger.warning("Camera group '%s' not found in %s",
                               cam_group, hdf5_path.name)
                return 0

        n_cam_frames = min(
            len(ref_cam_ts),
            *(len(f[f"{cg}/timestamps"][:]) for cg in CAMERA_HDF5_GROUPS)
        )

        aligned_joints = joint_pos[state_indices[:n_cam_frames]]
        ee_quat2 = fk_computer.compute_batch(aligned_joints)

        n_added = 0
        for cam_idx in range(n_cam_frames):
            s_idx = state_indices[cam_idx]

            state_15d = np.concatenate([
                joint_pos[s_idx],                              # [0:7]
                np.clip(gripper_w[s_idx], 0.0, None),          # [7]
                ee_pos[s_idx],                                 # [8:11]
                ee_quat2[cam_idx],                             # [11:15]
            ]).astype(np.float32)

            action_8d = np.concatenate([
                action_j[s_idx],                               # [0:7]
                action_g[s_idx],                               # [7]
            ]).astype(np.float32)

            frame = {
                "task": task_str,
                "observation.state": state_15d,
                "action": action_8d,
            }
            for cam_name in CAMERA_NAMES:
                jpeg_data = cam_images[cam_name][cam_idx]
                rgb = decode_jpeg(bytes(jpeg_data))
                frame[f"observation.images.{cam_name}"] = rgb

            dataset.add_frame(frame)
            n_added += 1

        return n_added


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    parser.add_argument("--urdf", type=str, required=True,
                        help="URDF path for FK-based ee_quat2 computation.")
    parser.add_argument("--robot-type", type=str, default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-videos", action="store_true")
    parser.add_argument("--reference-camera", type=str, default="camera_global",
                        choices=["camera_global", "camera_wrist"])
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if dest.exists():
        if args.force:
            logger.info("Removing existing %s (--force)", dest)
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} exists. Use --force to overwrite.")

    logger.info("Initializing FK for ee_quat2 computation: %s", args.urdf)
    fk_computer = FKQuatComputer(args.urdf)

    meta_path = source / "meta.json"
    task_str = "put cube into box"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        task_str = meta.get("task", task_str)
        logger.info("Task: %s, Total episodes: %s", task_str, meta.get("total_episodes"))

    hdf5_files = sorted(source.glob("episode_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files in {source}")
    logger.info("Found %d HDF5 episode files", len(hdf5_files))

    with h5py.File(hdf5_files[0], "r") as f:
        sample_jpeg = bytes(f[f"{CAMERA_HDF5_GROUPS[0]}/color_image_jpeg"][0])
        sample_img = decode_jpeg(sample_jpeg)
        image_shape = sample_img.shape
    logger.info("Image shape: %s", image_shape)

    use_videos = not args.no_videos
    features = build_features(image_shape=image_shape, use_videos=use_videos)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dest.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id=dest.name,
        fps=args.fps,
        root=dest,
        robot_type=args.robot_type,
        features=features,
        use_videos=use_videos,
        tolerance_s=1 / args.fps,
    )

    total_frames = 0
    for ep_idx, hdf5_path in enumerate(hdf5_files):
        n_frames = process_episode(
            hdf5_path, dataset, task_str, fk_computer,
            reference_camera=args.reference_camera,
        )
        if n_frames > 0:
            dataset.save_episode()
            total_frames += n_frames
            logger.info(
                "Episode %d/%d (%s): %d frames (total: %d)",
                ep_idx + 1, len(hdf5_files), hdf5_path.name, n_frames, total_frames,
            )
        else:
            logger.warning("Episode %d (%s): skipped (0 frames)",
                           ep_idx, hdf5_path.name)

    logger.info("=== DONE ===")
    logger.info("  Total episodes: %d", len(hdf5_files))
    logger.info("  Total frames: %d", total_frames)
    logger.info("  FPS: %d", args.fps)
    logger.info("  Output: %s", dest)
    logger.info("  ee_quat2: FK-derived, xyzw, hemisphere-normalized")


if __name__ == "__main__":
    main()
