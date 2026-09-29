#!/usr/bin/env python3
"""Write a LeRobot v3.0 Goal dataset whose keypoints live in the table frame.

Pass 1 scans every selected episode, runs GoalTableFK, and records the world
axis-aligned box. R_pad is computed once and written to meta/keypoints_meta.json
and meta/goal_train_eval_contract.json. Pass 2 writes frames through
LeRobotDataset.create so the on-disk layout is the v3.0 layout the trainer
already reads.

What gen2 changed versus gen1 (each item is a past incident, see 3d4d_gen2.markdown):

* Images are written in robosuite RAW orientation (img[::-1, ::-1] of the RLDS
  JPEG). RLDS is rot180 of raw; evaluation/LIBERO2 enforces raw.  (F1)
* robot_type is "panda" so panda.yaml maps image/image2 and the server accepts
  two images; stats key is "panda".                                (INVALID_IMAGE_COUNT)
* Every episode is validated before it is written: finite values, unit
  quaternions, |FK eef - observation.state[0:3]| below tolerance, joint angles
  inside the Panda limits. A failure aborts the run.               (coordinate drift)
* One contract file carries everything eval must agree with.        (train/eval drift)
* ``--no-video`` is a TEST-ONLY switch (stores PNG-in-parquet images). It is
  refused together with --confirm-full-run.

The 4,243-episode release stores each of 428 trajectories under several
renderings. This script keeps that multiplicity; dedup would change the published
training distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from contract import (  # noqa: E402
    BBOX_MARGIN,
    CHUNK_SIZE,
    CONTRACT_FILENAME,
    EPISODE_EEF_TOL_M,
    FPS,
    GOAL_BASE_XPOS,
    GOAL_MJCF,
    HISTORY_MAX_LEN,
    IMAGE_ORIENTATION,
    KEYPOINT_NAMES,
    KPT_4D_MODE,
    LIVE_EEF_TOL_M,
    RLDS_IMAGE_ORIENTATION,
    ROBOT_TYPE,
    STATS_KEY,
    feature_names,
)
from fk import GoalTableFK, compute_r_pad  # noqa: E402
from rlds_goal import DEFAULT_ROOT, iter_goal_episodes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("goal4d")

# Panda joint limits from the robosuite model (rad). Used only as a sanity gate.
_JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_JOINT_SLACK = 0.05


def _jpeg_to_rgb(blob: bytes, orientation: str = IMAGE_ORIENTATION) -> np.ndarray:
    """Decode one RLDS JPEG and put it in the requested orientation.

    RLDS frames are rot180 of the robosuite raw buffer. ``raw`` undoes that.
    """
    from PIL import Image

    arr = np.asarray(Image.open(BytesIO(blob)).convert("RGB"), dtype=np.uint8)
    if orientation == "raw":
        arr = arr[::-1, ::-1]
    elif orientation != "rlds":
        raise ValueError(f"orientation must be raw|rlds, got {orientation!r}")
    return np.ascontiguousarray(arr)


class EpisodeValidationError(RuntimeError):
    pass


def validate_episode(ep, poses: np.ndarray, tol_m: float = EPISODE_EEF_TOL_M) -> float:
    """Return max |FK eef - state[0:3]| in metres or raise EpisodeValidationError.

    ``poses`` is fk.batch_world(ep.qpos9()), shape [T, 8, 7].
    """
    where = f"{ep.perturbation}/{ep.task_name}"
    if ep.num_steps < 2:
        raise EpisodeValidationError(f"{where}: episode has {ep.num_steps} steps")
    for name, arr in (("state", ep.state), ("joint_state", ep.joint_state), ("action", ep.action), ("poses", poses)):
        if not np.all(np.isfinite(arr)):
            raise EpisodeValidationError(f"{where}: non-finite values in {name}")
    if len(ep.image_jpeg) != ep.num_steps or len(ep.wrist_jpeg) != ep.num_steps:
        raise EpisodeValidationError(f"{where}: image count != {ep.num_steps}")
    if np.any(ep.joint_state < _JOINT_LO - _JOINT_SLACK) or np.any(ep.joint_state > _JOINT_HI + _JOINT_SLACK):
        raise EpisodeValidationError(f"{where}: joint angle outside Panda limits")
    qn = np.linalg.norm(poses[:, :, 3:], axis=-1)
    if np.abs(qn - 1.0).max() > 1e-5:
        raise EpisodeValidationError(f"{where}: quaternion norm off by {np.abs(qn - 1.0).max():.2e}")
    err = float(np.linalg.norm(poses[:, 7, :3] - ep.state[:, :3], axis=-1).max())
    if err > tol_m:
        raise EpisodeValidationError(
            f"{where}: FK eef differs from observation.state[0:3] by {err:.4f} m (> {tol_m} m). "
            "The base frame is wrong for this episode."
        )
    return err


def pass1_bbox(root: Path, fk: GoalTableFK, max_episodes: int | None):
    gmin = np.full(3, np.inf, dtype=np.float64)
    gmax = np.full(3, -np.inf, dtype=np.float64)
    frames = 0
    episodes = 0
    worst_err = 0.0
    for ep in iter_goal_episodes(root, max_episodes=max_episodes):
        poses = fk.batch_world(ep.qpos9())
        worst_err = max(worst_err, validate_episode(ep, poses))
        pos = poses[:, :, :3].reshape(-1, 3)
        gmin = np.minimum(gmin, pos.min(axis=0))
        gmax = np.maximum(gmax, pos.max(axis=0))
        frames += ep.num_steps
        episodes += 1
        if episodes % 200 == 0:
            logger.info("pass1 episodes=%d frames=%d worst_eef_err=%.2e m", episodes, frames, worst_err)
    if episodes == 0:
        raise RuntimeError(f"no episodes under {root}")
    return gmin, gmax, frames, episodes, worst_err


def _features(with_video: bool) -> dict:
    image_dtype = "video" if with_video else "image"
    return {
        "observation.images.image": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": image_dtype,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az", "finger_l", "finger_r"],
        },
        "observation.state.joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"joint{i}" for i in range(1, 8)],
        },
        "observation.keypoint_3d": {
            "dtype": "float32",
            "shape": (56,),
            "names": feature_names(),
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
        },
    }


def pass2_write(
    root: Path,
    dest: Path,
    fk: GoalTableFK,
    r_pad: float,
    max_episodes: int | None,
    with_video: bool,
    repo_id: str,
    orientation: str = IMAGE_ORIENTATION,
) -> list[dict]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if dest.exists():
        raise FileExistsError(f"{dest} exists; pass --force from main before calling pass2")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=_features(with_video),
        root=dest,
        robot_type=ROBOT_TYPE,
        use_videos=with_video,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    sidecar = []
    for ep_index, ep in enumerate(iter_goal_episodes(root, max_episodes=max_episodes)):
        poses = fk.batch_world(ep.qpos9())
        validate_episode(ep, poses)
        kpts = fk.batch_keypoints(ep.qpos9(), r_pad)
        for t in range(ep.num_steps):
            frame = {
                "task": ep.instruction,
                "observation.state": ep.state[t].astype(np.float32),
                "observation.state.joint_position": ep.joint_state[t].astype(np.float32),
                "observation.keypoint_3d": kpts[t].reshape(-1),
                "action": ep.action[t].astype(np.float32),
                "observation.images.image": _jpeg_to_rgb(ep.image_jpeg[t], orientation),
                "observation.images.image2": _jpeg_to_rgb(ep.wrist_jpeg[t], orientation),
            }
            dataset.add_frame(frame)
        dataset.save_episode()
        sidecar.append(
            {
                "episode_index": ep_index,
                "perturbation": ep.perturbation,
                "task_name": ep.task_name,
                "instruction": ep.instruction,
                "length": ep.num_steps,
                "file_path": ep.file_path,
                "trajectory_hash": ep.trajectory_hash(),
            }
        )
        if (ep_index + 1) % 50 == 0:
            logger.info("pass2 wrote %d episodes", ep_index + 1)
    dataset.finalize()
    return sidecar


def write_contract(
    dest: Path,
    r_pad: float,
    gmin: np.ndarray,
    gmax: np.ndarray,
    frames: int,
    episodes: int,
    sidecar: list[dict],
    margin: float,
    worst_err: float,
    with_video: bool,
    orientation: str,
) -> None:
    meta = dest / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    kin_md5 = hashlib.md5((_HERE / "assets" / "panda_kin.xml").read_bytes()).hexdigest()
    goal_md5 = hashlib.md5(GOAL_MJCF.read_bytes()).hexdigest() if GOAL_MJCF.exists() else None
    payload = {
        "coordinate_system": "libero_goal_table_world",
        "base_xpos_m": GOAL_BASE_XPOS.tolist(),
        "lift_base_is_not_this_frame": [-0.56, 0.0, 0.912],
        "bbox_radius": r_pad,
        "bbox_margin": margin,
        "r_pad_includes_margin": True,
        "do_not_reuse_lift_r_pad": 1.8212722539901733,
        "global_min_world": gmin.tolist(),
        "global_max_world": gmax.tolist(),
        "normalization": "divide_xyz_by_r_pad_leave_quaternion",
        "quaternion": "xyzw_qw_nonnegative",
        "keypoint_bodies": KEYPOINT_NAMES,
        "state_orientation_body": "right_hand",
        "position_body": "gripper0_eef",
        "eef_vs_hand_yaw_deg": 90.0,
        "fps": FPS,
        "fingers_do_not_move_keypoints": True,
        "total_frames": frames,
        "total_episodes": episodes,
        "mjcf_md5": kin_md5,
        "goal_mjcf_md5": goal_md5,
        "history_is_derived": True,
        "history_offsets": "[-H..-1] strictly before current frame, packed at front",
    }
    (meta / "keypoints_meta.json").write_text(json.dumps(payload, indent=2))

    contract = {
        "schema": "goal_train_eval_contract/1",
        "robot_type": ROBOT_TYPE,
        "stats_key": STATS_KEY,
        "image_orientation": orientation,
        "rlds_image_orientation": RLDS_IMAGE_ORIENTATION,
        "cameras": {"observation.images.image": "agentview", "observation.images.image2": "wrist"},
        "resize_hw": [224, 224],
        "fps": FPS,
        "kpt_4d_mode": KPT_4D_MODE,
        "num_keypoints": len(KEYPOINT_NAMES),
        "keypoint_history_max_len": HISTORY_MAX_LEN,
        "chunk_size": CHUNK_SIZE,
        "r_pad": r_pad,
        "base_xpos_m": GOAL_BASE_XPOS.tolist(),
        "coordinate_system": "libero_goal_table_world",
        "eval_mjcf": "b/s/libplus2/gol/assets/panda_goal_table.xml",
        "eval_mjcf_md5": goal_md5,
        "kin_mjcf_md5": kin_md5,
        "live_eef_tol_m": LIVE_EEF_TOL_M,
        "episode_eef_err_max_m": worst_err,
        "has_video": bool(with_video),
        "wait_steps_committed_to_history": False,
        "history_includes_current_frame": False,
    }
    (meta / CONTRACT_FILENAME).write_text(json.dumps(contract, indent=2))
    with (meta / "goal_episodes.jsonl").open("w") as fh:
        for row in sidecar:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--rlds-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--no-video", action="store_true", help="TEST ONLY: store images in parquet")
    parser.add_argument("--confirm-full-run", action="store_true", help="assert this is the real training set")
    parser.add_argument("--image-orientation", choices=["raw", "rlds"], default=IMAGE_ORIENTATION)
    parser.add_argument("--bbox-margin", type=float, default=BBOX_MARGIN)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repo-id", default="libero_plus_goal_4d")
    args = parser.parse_args()

    if args.confirm_full_run and (args.no_video or args.max_episodes is not None):
        parser.error("--confirm-full-run refuses --no-video and --max-episodes")
    if args.confirm_full_run and args.image_orientation != IMAGE_ORIENTATION:
        parser.error(f"--confirm-full-run requires --image-orientation {IMAGE_ORIENTATION}")

    if args.dest.exists():
        if not args.force:
            raise FileExistsError(f"{args.dest} exists. Pass --force to replace it.")
        import shutil

        shutil.rmtree(args.dest)

    fk = GoalTableFK()
    gmin, gmax, frames, episodes, worst_err = pass1_bbox(args.rlds_root, fk, args.max_episodes)
    r_pad = compute_r_pad(gmin, gmax, args.bbox_margin)
    logger.info(
        "R_pad=%.10f  episodes=%d frames=%d  worst FK-vs-state eef err=%.2e m", r_pad, episodes, frames, worst_err
    )
    # pass2 scans the RLDS again; max_episodes must match pass1
    sidecar = pass2_write(
        args.rlds_root,
        args.dest,
        fk,
        r_pad,
        args.max_episodes,
        with_video=not args.no_video,
        repo_id=args.repo_id,
        orientation=args.image_orientation,
    )
    write_contract(
        args.dest,
        r_pad,
        gmin,
        gmax,
        frames,
        episodes,
        sidecar,
        args.bbox_margin,
        worst_err,
        with_video=not args.no_video,
        orientation=args.image_orientation,
    )
    logger.info("wrote %s", args.dest)


if __name__ == "__main__":
    main()
