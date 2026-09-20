#!/usr/bin/env python3
"""Convert LIBERO RLDS TFRecord to LeRobot v3 with FK-computed 7D keypoints.

Reads RLDS episodes directly (TF-free), computes MuJoCo FK keypoints,
writes a complete LeRobot v3 dataset with:
  - observation.state [8] (EEF pose + gripper)
  - observation.state.joint_position [7] (arm joint angles)
  - action [7] (OSC_POSE delta + gripper)
  - observation.keypoint_3d [56] (8 keypoints x 7D pos+quat)
  - Images encoded as MP4 videos (agentview + wrist)

Two-pass pipeline:
  Pass 1: scan all episodes to compute global keypoint bounding box -> R_pad
  Pass 2: write parquet + video files with normalized keypoints

Usage:
    /B/VENV/itnvla15rbt20/bin/python b/s/libplus/rlds_to_lerobot_4d.py \
        --dest ~/b/Dta/opvla_libero_4d \
        --xml evaluation/panda_robosuite_lift.xml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
ROBOT_NQPOS = 9
BBOX_MARGIN = 0.15
EVAL_DEFAULT_R_PAD = 1.8212722539901733
FPS = 10  # LIBERO runs at 20Hz control, but RLDS decimated by 2x

EEF_BODY_CANDIDATES = ("gripper0_eef", "gripper0_right_eef")
KEYPOINT_ARM_BODY_NAMES = [
    "robot0_link1", "robot0_link2", "robot0_link3", "robot0_link4",
    "robot0_link5", "robot0_link6", "robot0_link7",
]

SUBSETS_ORDERED = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def resolve_eef_body_name(model) -> str:
    for name in EEF_BODY_CANDIDATES:
        try:
            model.body(name)
            return name
        except (KeyError, ValueError):
            continue
    raise KeyError(f"No EEF body found (tried {EEF_BODY_CANDIDATES})")


def mujoco_quat_to_xyzw(xquat: np.ndarray) -> np.ndarray:
    raw = np.array([xquat[1], xquat[2], xquat[3], xquat[0]], dtype=np.float32)
    if raw[3] < 0:
        raw = -raw
    return raw


class LiftMujocoFK:
    def __init__(self, xml_path: str | Path):
        import mujoco
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        eef = resolve_eef_body_name(self.model)
        self.body_names = [*KEYPOINT_ARM_BODY_NAMES, eef]
        self.body_ids = {n: self.model.body(n).id for n in self.body_names}

    def compute_batch(self, qpos_batch: np.ndarray) -> np.ndarray:
        out = np.empty((len(qpos_batch), NUM_KEYPOINTS, KEYPOINT_DIM), dtype=np.float32)
        for i, qpos in enumerate(qpos_batch):
            n = min(len(qpos), ROBOT_NQPOS)
            self.data.qpos[:n] = qpos[:n]
            self._mujoco.mj_forward(self.model, self.data)
            for j, name in enumerate(self.body_names):
                bid = self.body_ids[name]
                out[i, j, :3] = self.data.xpos[bid]
                out[i, j, 3:7] = mujoco_quat_to_xyzw(self.data.xquat[bid])
        return out


def compute_r_pad(g_min: np.ndarray, g_max: np.ndarray, margin: float = BBOX_MARGIN) -> float:
    extremes = np.maximum(np.abs(g_min), np.abs(g_max))
    return float(extremes.max() * (1.0 + margin))


def jpeg_bytes_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG")
    return img


def pass1_compute_bbox(rlds_reader, fk: LiftMujocoFK, subsets: list[str]
                       ) -> tuple[np.ndarray, np.ndarray, int, int]:
    g_min = np.full(3, np.inf, dtype=np.float64)
    g_max = np.full(3, -np.inf, dtype=np.float64)
    total_frames = 0
    total_episodes = 0

    for subset in subsets:
        for ep in rlds_reader.iter_episodes(subset):
            qpos9 = ep.qpos9()
            kpts = fk.compute_batch(qpos9)
            pos = kpts[:, :, :3].reshape(-1, 3)
            g_min = np.minimum(g_min, pos.min(axis=0))
            g_max = np.maximum(g_max, pos.max(axis=0))
            total_frames += len(qpos9)
            total_episodes += 1

        logger.info("Pass 1 [%s]: %d episodes so far, %d frames",
                    subset, total_episodes, total_frames)

    return g_min.astype(np.float32), g_max.astype(np.float32), total_frames, total_episodes


def pass2_write_dataset(
    rlds_reader, fk: LiftMujocoFK, r_pad: float, subsets: list[str],
    dest: Path, fps: int,
) -> tuple[int, int, list[dict], list[dict]]:
    data_dir = dest / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    tasks_meta = []
    task_map: dict[str, int] = {}

    global_frame_idx = 0
    global_ep_idx = 0

    for subset in subsets:
        for ep in rlds_reader.iter_episodes(subset):
            T = ep.num_steps
            qpos9 = ep.qpos9()
            kpts = fk.compute_batch(qpos9)
            kpts[:, :, :3] /= r_pad

            task_text = ep.instruction
            if task_text not in task_map:
                task_map[task_text] = len(task_map)
                tasks_meta.append({"task_index": task_map[task_text], "task": task_text})
            task_idx = task_map[task_text]

            state = ep.state           # [T, 8]
            joint = ep.joint_state     # [T, 7]
            action = ep.action         # [T, 7]

            rows = {
                "observation.state": [state[t].astype(np.float32) for t in range(T)],
                "observation.state.joint_position": [joint[t].astype(np.float32) for t in range(T)],
                "action": [action[t].astype(np.float32) for t in range(T)],
                "observation.keypoint_3d": [kpts[t].reshape(-1).astype(np.float32) for t in range(T)],
                "timestamp": np.arange(T, dtype=np.float32) / fps,
                "frame_index": np.arange(T, dtype=np.int64),
                "episode_index": np.full(T, global_ep_idx, dtype=np.int64),
                "index": np.arange(global_frame_idx, global_frame_idx + T, dtype=np.int64),
                "task_index": np.full(T, task_idx, dtype=np.int64),
            }
            df = pd.DataFrame(rows)
            pq_path = data_dir / f"episode_{global_ep_idx:06d}.parquet"
            df.to_parquet(pq_path, index=False)

            # Encode images to video
            for cam_key, cam_name in [
                ("observation.images.image", "image"),
                ("observation.images.image2", "wrist_image"),
            ]:
                vid_dir = dest / "videos" / "chunk-000" / cam_key
                vid_dir.mkdir(parents=True, exist_ok=True)
                vid_path = vid_dir / f"episode_{global_ep_idx:06d}.mp4"

                first_jpeg = ep.jpeg(cam_name, 0)
                first_img = jpeg_bytes_to_bgr(first_jpeg)
                h, w = first_img.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(vid_path), fourcc, fps, (w, h), True)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot open VideoWriter for {vid_path}")

                for t in range(T):
                    jpeg = ep.jpeg(cam_name, t)
                    frame = jpeg_bytes_to_bgr(jpeg)
                    writer.write(frame)
                writer.release()

            episodes_meta.append({
                "episode_index": global_ep_idx,
                "tasks": [task_text],
                "length": T,
            })

            global_frame_idx += T
            global_ep_idx += 1

            if global_ep_idx % 50 == 0:
                logger.info("Pass 2: %d episodes, %d frames written",
                            global_ep_idx, global_frame_idx)

        logger.info("Pass 2 [%s] done: %d episodes, %d frames total",
                    subset, global_ep_idx, global_frame_idx)

    return global_frame_idx, global_ep_idx, episodes_meta, tasks_meta


def write_meta(
    dest: Path, total_frames: int, total_episodes: int,
    episodes_meta: list[dict], tasks_meta: list[dict],
    r_pad: float, g_min: np.ndarray, g_max: np.ndarray,
    xml_path: Path, xml_md5: str, body_names: list[str],
    fps: int,
) -> None:
    meta_dir = dest / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "libero",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks_meta),
        "total_videos": total_episodes * 2,
        "total_chunks": 1,
        "chunks_size": 10000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 256, "video.width": 256,
                    "video.codec": "mp4v", "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False, "video.fps": fps,
                    "video.channels": 3, "has_audio": False,
                },
            },
            "observation.images.image2": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": 256, "video.width": 256,
                    "video.codec": "mp4v", "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False, "video.fps": fps,
                    "video.channels": 3, "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": ["eef_x", "eef_y", "eef_z",
                          "eef_ax", "eef_ay", "eef_az",
                          "gripper_finger1", "gripper_finger2"],
            },
            "observation.state.joint_position": {
                "dtype": "float32",
                "shape": [7],
                "names": [f"joint{i}" for i in range(1, 8)],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            },
            "observation.keypoint_3d": {
                "dtype": "float32",
                "shape": [NUM_KEYPOINTS * KEYPOINT_DIM],
                "names": [
                    f"{name}_{c}"
                    for name in body_names
                    for c in ("px", "py", "pz", "qx", "qy", "qz", "qw")
                ],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in episodes_meta:
            f.write(json.dumps(ep) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task in tasks_meta:
            f.write(json.dumps(task) + "\n")

    kpt_meta = {
        "bbox_radius": r_pad,
        "bbox_margin": BBOX_MARGIN,
        "global_min_world": g_min.tolist(),
        "global_max_world": g_max.tolist(),
        "normalization": "world_origin_isotropic_r_pad",
        "keypoint_dim": KEYPOINT_DIM,
        "keypoint_dim_layout": "px,py,pz,qx,qy,qz,qw",
        "rotation_representation": "quaternion_xyzw_hemisphere",
        "num_keypoints": NUM_KEYPOINTS,
        "keypoint_bodies": body_names,
        "robot_nqpos": ROBOT_NQPOS,
        "qpos_layout": "joint_state[0:7] + state[6:8]",
        "total_frames": total_frames,
        "mjcf_path": str(xml_path.resolve()),
        "mjcf_md5": xml_md5,
        "coordinate_system": "MuJoCo Lift world frame, divided by R_pad",
        "r_pad_includes_margin": True,
        "eval_default_r_pad": EVAL_DEFAULT_R_PAD,
    }
    with open(meta_dir / "keypoints_meta.json", "w") as f:
        json.dump(kpt_meta, f, indent=2)

    logger.info("Wrote meta to %s", meta_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, required=True,
                        help="Output LeRobot v3 dataset directory")
    parser.add_argument("--xml", type=Path,
                        default=Path("evaluation/panda_robosuite_lift.xml"))
    parser.add_argument("--rlds-root", type=Path,
                        default=Path("/B/Dta/opvla_libero"))
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--subsets", nargs="*", default=SUBSETS_ORDERED)
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN)
    args = parser.parse_args()

    if args.dest.exists() and not args.force:
        raise FileExistsError(f"{args.dest} exists. Pass --force to overwrite.")
    if args.dest.exists() and args.force:
        import shutil
        logger.info("Removing %s (--force)", args.dest)
        shutil.rmtree(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)

    if not args.xml.exists():
        raise FileNotFoundError(f"Lift MJCF not found: {args.xml}")

    xml_md5 = hashlib.md5(args.xml.read_bytes()).hexdigest()
    logger.info("MJCF: %s (MD5: %s)", args.xml, xml_md5)

    # Import RLDS reader (located at b/d/libplus/ds/asset/rlds_reader.py)
    # __file__ = b/s/libplus/rlds_to_lerobot_4d.py => .parent^4 = project root
    proj_root = Path(__file__).resolve().parents[3]
    rlds_path = proj_root / "b" / "d" / "libplus" / "ds" / "asset"
    sys.path.insert(0, str(rlds_path))
    import rlds_reader
    rlds_reader.DATA_ROOT = args.rlds_root

    fk = LiftMujocoFK(args.xml)

    t0 = time.time()
    logger.info("=== Pass 1: computing global bounding box ===")
    g_min, g_max, total_frames, total_episodes = pass1_compute_bbox(
        rlds_reader, fk, args.subsets)
    r_pad = compute_r_pad(g_min, g_max, margin=args.bbox_margin)
    logger.info("Global min: %s", g_min)
    logger.info("Global max: %s", g_max)
    logger.info("R_pad = %.10f (margin=%.0f%%)", r_pad, args.bbox_margin * 100)
    logger.info("Total: %d episodes, %d frames", total_episodes, total_frames)

    r_pad_diff = abs(r_pad - EVAL_DEFAULT_R_PAD)
    if r_pad_diff > 1e-4:
        logger.warning("R_pad differs from eval default by %.6f!", r_pad_diff)
    else:
        logger.info("R_pad matches eval default (diff=%.2e)", r_pad_diff)
    logger.info("Pass 1 took %.1fs", time.time() - t0)

    t1 = time.time()
    logger.info("=== Pass 2: writing LeRobot v3 dataset with keypoints ===")
    total_frames_p2, total_eps_p2, episodes_meta, tasks_meta = pass2_write_dataset(
        rlds_reader, fk, r_pad, args.subsets, args.dest, args.fps)

    write_meta(args.dest, total_frames_p2, total_eps_p2,
               episodes_meta, tasks_meta,
               r_pad, g_min, g_max, args.xml, xml_md5, fk.body_names, args.fps)
    logger.info("Pass 2 took %.1fs", time.time() - t1)

    logger.info("=== DONE ===")
    logger.info("  Output: %s", args.dest)
    logger.info("  Episodes: %d", total_eps_p2)
    logger.info("  Frames: %d (pass1) / %d (pass2)", total_frames, total_frames_p2)
    logger.info("  R_pad: %.10f", r_pad)
    logger.info("  Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
