"""Fast HDF5-to-LeRobot converter for put_cube_into_box (V3).

Bypasses slow per-frame PNG writing by batch-processing episodes:
  1. Reads all HDF5 data, computes ee_quat2 via FK
  2. Writes parquet files directly
  3. Encodes videos using ffmpeg subprocess (JPEG→MP4, no PNG intermediate)

Usage:
    source /B/VENV/itnvla15rbt20/bin/activate
    export HF_HOME=/B/VENV/hf_home
    python b/s/Frk3/convert_cubinbx_fast.py \
        --source /B/Dta/put_cube_into_box/put_cube_into_box_hdf5 \
        --dest /home/a26113/b/Dta/put_cube_into_box_lrb3 \
        --urdf b/d/Frk2/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FK_PREFIX = "fr3v2_1"
FPS = 30
CAMERA_NAMES = ["global", "wrist"]
CAMERA_HDF5_GROUPS = ["camera_global", "camera_wrist"]


class FKQuatComputer:
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


def align_timestamps(state_ts, camera_ts):
    indices = np.searchsorted(state_ts, camera_ts)
    indices = np.clip(indices, 1, len(state_ts) - 1)
    left = np.abs(state_ts[indices - 1] - camera_ts)
    right = np.abs(state_ts[indices] - camera_ts)
    return np.where(left <= right, indices - 1, indices)


def decode_jpeg(jpeg_bytes):
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def encode_video_from_jpegs(jpeg_list, output_path, fps, width=640, height=480):
    """Encode video from list of JPEG byte arrays using ffmpeg pipe."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libsvtav1",
        "-pix_fmt", "yuv420p",
        "-crf", "30",
        "-preset", "6",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for jpeg_data in jpeg_list:
            rgb = decode_jpeg(bytes(jpeg_data))
            proc.stdin.write(rgb.tobytes())
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
    stderr = proc.stderr.read()
    rc = proc.wait()
    if rc != 0:
        logger.error("ffmpeg failed: %s", stderr.decode()[-500:])
        raise RuntimeError(f"ffmpeg exited with code {rc}")


def build_info_json(dest, n_episodes, total_frames, image_shape):
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_cubinbx",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "fps": FPS,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/{file_name:s}.parquet",
        "video_path": "videos/{video_key:s}/chunk-{episode_chunk:03d}/{file_name:s}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [15],
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
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "actions": [
                        "joint1", "joint2", "joint3", "joint4",
                        "joint5", "joint6", "joint7",
                        "gripper_cmd",
                    ],
                },
            },
            "observation.images.global": {
                "dtype": "video",
                "shape": list(image_shape),
                "names": ["height", "width", "rgb"],
                "video_info": {
                    "video.fps": FPS,
                    "video.height": image_shape[0],
                    "video.width": image_shape[1],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": list(image_shape),
                "names": ["height", "width", "rgb"],
                "video_info": {
                    "video.fps": FPS,
                    "video.height": image_shape[0],
                    "video.width": image_shape[1],
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
        },
    }
    info_path = dest / "meta" / "info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    return info


def write_episodes_jsonl(dest, episode_records):
    path = dest / "meta" / "episodes.jsonl"
    with open(path, "w") as f:
        for rec in episode_records:
            f.write(json.dumps(rec) + "\n")


def write_tasks_jsonl(dest, task_str):
    path = dest / "meta" / "tasks.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_str}) + "\n")


def process_all(source, dest, fk_computer, force=False):
    source = Path(source)
    dest = Path(dest)

    if dest.exists():
        if force:
            shutil.rmtree(dest)
        else:
            raise FileExistsError(f"{dest} exists")

    hdf5_files = sorted(source.glob("episode_*.hdf5"))
    n_episodes = len(hdf5_files)
    logger.info("Found %d HDF5 episodes", n_episodes)

    task_str = "put cube into box"
    meta_path = source / "meta.json"
    if meta_path.exists():
        task_str = json.loads(meta_path.read_text()).get("task", task_str)

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dest / "videos" / "observation.images.global" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dest / "videos" / "observation.images.wrist" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dest / "meta").mkdir(parents=True, exist_ok=True)

    total_frames = 0
    episode_records = []
    image_shape = None

    for ep_idx, hdf5_path in enumerate(hdf5_files):
        with h5py.File(hdf5_path, "r") as f:
            state_ts = f["robot_state/timestamps"][:]
            joint_pos = f["robot_state/joint_positions"][:]
            gripper_w = f["robot_state/gripper_width"][:]
            ee_pos = f["robot_state/ee_pos"][:]
            action_j = f["robot_state/action_joints"][:]
            action_g = f["robot_state/action_gripper"][:]

            ref_cam_ts = f["camera_global/timestamps"][:]
            n_cam_global = len(ref_cam_ts)
            n_cam_wrist = len(f["camera_wrist/timestamps"][:])
            n_cam = min(n_cam_global, n_cam_wrist)

            state_indices = align_timestamps(state_ts, ref_cam_ts[:n_cam])

            aligned_joints = joint_pos[state_indices]
            ee_quat2 = fk_computer.compute_batch(aligned_joints)

            rows_state = []
            rows_action = []
            for ci in range(n_cam):
                si = state_indices[ci]
                s15 = np.concatenate([
                    joint_pos[si],
                    np.clip(gripper_w[si], 0.0, None),
                    ee_pos[si],
                    ee_quat2[ci],
                ]).astype(np.float32)
                a8 = np.concatenate([
                    action_j[si],
                    action_g[si],
                ]).astype(np.float32)
                rows_state.append(s15)
                rows_action.append(a8)

            jpeg_global = [bytes(f["camera_global/color_image_jpeg"][i])
                           for i in range(n_cam)]
            jpeg_wrist = [bytes(f["camera_wrist/color_image_jpeg"][i])
                          for i in range(n_cam)]

            if image_shape is None:
                sample = decode_jpeg(jpeg_global[0])
                image_shape = sample.shape

        ep_len = n_cam
        df = pd.DataFrame({
            "observation.state": rows_state,
            "action": rows_action,
            "episode_index": [ep_idx] * ep_len,
            "frame_index": list(range(ep_len)),
            "timestamp": [i / FPS for i in range(ep_len)],
            "index": list(range(total_frames, total_frames + ep_len)),
            "task_index": [0] * ep_len,
        })

        pq_path = dest / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(pq_path)

        vid_global = dest / "videos" / "observation.images.global" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"
        vid_wrist = dest / "videos" / "observation.images.wrist" / "chunk-000" / f"episode_{ep_idx:06d}.mp4"

        encode_video_from_jpegs(jpeg_global, vid_global, FPS,
                                image_shape[1], image_shape[0])
        encode_video_from_jpegs(jpeg_wrist, vid_wrist, FPS,
                                image_shape[1], image_shape[0])

        episode_records.append({
            "episode_index": ep_idx,
            "tasks": [task_str],
            "length": ep_len,
        })

        total_frames += ep_len
        logger.info("Episode %d/%d (%s): %d frames, total=%d",
                     ep_idx + 1, n_episodes, hdf5_path.name, ep_len, total_frames)

    build_info_json(dest, n_episodes, total_frames, image_shape)
    write_episodes_jsonl(dest, episode_records)
    write_tasks_jsonl(dest, task_str)

    logger.info("=== DONE ===")
    logger.info("  Episodes: %d, Frames: %d", n_episodes, total_frames)
    logger.info("  Output: %s", dest)
    return total_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fk_computer = FKQuatComputer(args.urdf)
    process_all(args.source, args.dest, fk_computer, force=args.force)


if __name__ == "__main__":
    main()
