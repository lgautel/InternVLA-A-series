"""Repack LIBERO Goal v1 (8D state) → v2 (14D state).

Reads v1 parquets, rewrites state layout, copies videos as independent files,
recomputes vector stats, writes v2 info.json and contract schema 2.

Does NOT call LeRobotDataset.create, does NOT re-encode video, does NOT use
hardlinks or symlinks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract_v2 import (
    ACTION_NAMES,
    CONTRACT_SCHEMA,
    COPY_ATOL,
    FINGER_ANTISYM_TOL_M,
    FINGER_NAMES,
    GRIPPER_INDEX,
    ROBOT_TYPE,
    STATE_DIM,
    STATE_NAMES,
    STATS_KEY,
    gripper_opening,
    pack_state_v2,
)

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QUANTILE_KEYS = ["q01", "q10", "q50", "q90", "q99"]

RECOMPUTE_FEATURES = {
    "observation.state": ("observation.state", 14),
    "observation.state.fingers": ("observation.state.fingers", 2),
    "observation.state.joint_position": ("observation.state.joint_position", 7),
    "action": ("action", 7),
    "observation.keypoint_3d": ("observation.keypoint_3d", 56),
}

COPY_FROM_V1 = [
    "observation.images.image", "observation.images.image2",
    "episode_index", "frame_index", "index", "task_index", "timestamp",
]

CROSS_VALIDATE_FEATURES = [
    "observation.state.joint_position", "action", "observation.keypoint_3d",
]


def load_v1_info(src: Path) -> dict:
    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"v1 info.json not found: {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    assert info["codebase_version"] == "v3.0", f"Expected v3.0, got {info['codebase_version']}"
    assert info["robot_type"] == "panda", f"Expected panda, got {info['robot_type']}"
    assert info["features"]["observation.state"]["shape"] == [8], \
        f"Expected state shape [8], got {info['features']['observation.state']['shape']}"
    assert info["features"]["action"]["shape"] == [7], \
        f"Expected action shape [7], got {info['features']['action']['shape']}"
    assert "observation.state.joint_position" in info["features"], "Missing joint_position feature"
    assert "observation.images.image" in info["features"], "Missing image feature"
    assert "observation.images.image2" in info["features"], "Missing image2 feature"
    return info


def rewrite_parquets(src: Path, dest: Path, max_episodes: int | None = None) -> tuple[int, int, int]:
    src_data = src / "data"
    dest_data = dest / "data"
    parquet_files = sorted(src_data.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {src_data}")

    total_frames = 0
    total_episodes = set()
    antisym_violations = 0

    for pf in parquet_files:
        rel = pf.relative_to(src_data)
        out_pf = dest_data / rel
        out_pf.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(pf)
        states_v1 = np.stack(table["observation.state"].to_pylist()).astype(np.float32)
        joints = np.stack(table["observation.state.joint_position"].to_pylist()).astype(np.float32)

        if max_episodes is not None:
            ep_col = table["episode_index"].to_pylist()
            mask = np.array([e < max_episodes for e in ep_col])
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            table = table.take(indices.tolist())
            states_v1 = states_v1[indices]
            joints = joints[indices]

        n = len(table)
        total_frames += n

        for ep in table["episode_index"].to_pylist():
            total_episodes.add(ep)

        eef6 = states_v1[:, 0:6]
        finger_l = states_v1[:, 6]
        finger_r = states_v1[:, 7]

        antisym = np.abs(finger_l + finger_r)
        antisym_violations += int(np.sum(antisym > FINGER_ANTISYM_TOL_M))

        state14 = pack_state_v2(joints, eef6, finger_l, finger_r)
        assert state14.shape == (n, STATE_DIM), f"Expected ({n}, {STATE_DIM}), got {state14.shape}"

        np.testing.assert_array_equal(state14[:, 0:7], joints,
                                      err_msg="state[0:7] != joint_position")
        np.testing.assert_array_equal(state14[:, 7:13], eef6,
                                      err_msg="state[7:13] != v1 state[0:6]")

        fingers2 = np.stack([finger_l, finger_r], axis=-1).astype(np.float32)

        state14_col = pa.FixedSizeListArray.from_arrays(
            pa.array(state14.ravel(), type=pa.float32()), STATE_DIM
        )
        fingers_col = pa.FixedSizeListArray.from_arrays(
            pa.array(fingers2.ravel(), type=pa.float32()), 2
        )

        out_columns = [
            ("observation.state", state14_col),
            ("observation.state.joint_position", table["observation.state.joint_position"]),
            ("observation.state.fingers", fingers_col),
            ("observation.keypoint_3d", table["observation.keypoint_3d"]),
            ("action", table["action"]),
        ]
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            out_columns.append((col, table[col]))

        out_table = pa.table(dict(out_columns))
        pq.write_table(out_table, out_pf)

    return total_frames, len(total_episodes), antisym_violations


def copy_videos(src: Path, dest: Path) -> int:
    src_videos = src / "videos"
    if not src_videos.exists():
        raise FileNotFoundError(f"No videos directory: {src_videos}")

    count = 0
    for root, _dirs, files in os.walk(src_videos):
        for fname in files:
            src_file = Path(root) / fname
            rel = src_file.relative_to(src_videos)
            dest_file = dest / "videos" / rel

            if src_file.is_symlink():
                raise RuntimeError(f"v1 video is a symlink, refusing: {src_file}")

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(src_file), str(dest_file))

            if os.path.samefile(str(src_file), str(dest_file)):
                raise RuntimeError(f"v2 video is same inode as v1 (hardlink?): {dest_file}")
            if dest_file.is_symlink():
                raise RuntimeError(f"v2 video is a symlink: {dest_file}")
            if src_file.stat().st_size != dest_file.stat().st_size:
                raise RuntimeError(
                    f"Size mismatch: {src_file} ({src_file.stat().st_size}) "
                    f"vs {dest_file} ({dest_file.stat().st_size})"
                )
            count += 1

    return count


def copy_meta_tables(src: Path, dest: Path) -> None:
    dest_meta = dest / "meta"
    dest_meta.mkdir(parents=True, exist_ok=True)

    src_episodes = src / "meta" / "episodes"
    dest_episodes = dest_meta / "episodes"
    if src_episodes.exists():
        shutil.copytree(str(src_episodes), str(dest_episodes))

    for fname in ["tasks.parquet", "goal_episodes.jsonl", "keypoints_meta.json"]:
        src_file = src / "meta" / fname
        if src_file.exists():
            shutil.copyfile(str(src_file), str(dest_meta / fname))
        else:
            print(f"  WARNING: {src_file} not found, skipping")


def compute_vector_stats(data: np.ndarray) -> dict:
    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains non-finite values (NaN or Inf)")
    N = data.shape[0]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    mean = np.clip(mean, mn, mx)
    std = np.where(mn == mx, 0.0, std)
    qs = np.quantile(data, QUANTILES, axis=0)

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": mn.tolist(),
        "max": mx.tolist(),
        "count": [int(N)],
    }
    for i, qk in enumerate(QUANTILE_KEYS):
        stats[qk] = qs[i].tolist()

    for d in range(data.shape[1]):
        assert mn[d] <= mean[d] <= mx[d], f"dim {d}: min <= mean <= max violated"
        assert std[d] >= 0, f"dim {d}: std < 0"
        assert mn[d] <= qs[0, d] <= qs[1, d] <= qs[2, d] <= qs[3, d] <= qs[4, d] <= mx[d], \
            f"dim {d}: quantile monotonicity violated"
    return stats


def recompute_vector_stats(dest: Path, v1_stats_path: Path, *, is_full_run: bool = True) -> None:
    parquet_files = sorted(dest.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files in {dest / 'data'}")

    feature_arrays: dict[str, list[np.ndarray]] = {k: [] for k in RECOMPUTE_FEATURES}
    for pf in parquet_files:
        table = pq.read_table(pf)
        for feat_key, (col_name, expected_dim) in RECOMPUTE_FEATURES.items():
            col_data = np.stack(table[col_name].to_pylist()).astype(np.float64)
            if col_data.ndim == 1:
                col_data = col_data.reshape(-1, 1)
            if col_data.shape[1] != expected_dim:
                raise ValueError(f"{feat_key} in {pf}: expected dim {expected_dim}, got {col_data.shape[1]}")
            feature_arrays[feat_key].append(col_data)

    v2_stats = {}
    for feat_key in RECOMPUTE_FEATURES:
        all_data = np.concatenate(feature_arrays[feat_key], axis=0)
        v2_stats[feat_key] = compute_vector_stats(all_data)

    with open(v1_stats_path) as f:
        v1_stats = json.load(f)
    for feat_key in COPY_FROM_V1:
        if feat_key not in v1_stats:
            raise KeyError(f"v1 stats missing key: {feat_key}")
        v2_stats[feat_key] = v1_stats[feat_key]

    for feat_key in CROSS_VALIDATE_FEATURES:
        if feat_key not in v1_stats:
            raise KeyError(f"v1 stats missing B-class key: {feat_key}")
        if is_full_run:
            v2_feat = v2_stats[feat_key]
            v1_feat = v1_stats[feat_key]
            for metric in ["mean", "min", "max"]:
                v1_arr = np.array(v1_feat[metric], dtype=np.float64)
                v2_arr = np.array(v2_feat[metric], dtype=np.float64)
                if not np.allclose(v1_arr, v2_arr, atol=1e-5, rtol=1e-4):
                    raise ValueError(
                        f"B-class cross-validation failed: {feat_key}.{metric} "
                        f"max_diff={np.max(np.abs(v1_arr - v2_arr)):.2e}"
                    )
        v2_stats[feat_key] = v1_stats[feat_key]
        print(f"  {feat_key}: using v1 stats (data is byte-identical)")
    if not is_full_run:
        print("  (smoke run: B-class cross-validation skipped, v1 stats copied directly)")

    stats_path = dest / "meta" / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(v2_stats, f, indent=2)
    print(f"  stats.json written with {len(v2_stats)} keys")


def write_info_v2(dest: Path, v1_info: dict, total_frames: int, total_episodes: int) -> None:
    v2_info = {
        "codebase_version": "v3.0",
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": v1_info["total_tasks"],
        "chunks_size": v1_info["chunks_size"],
        "fps": v1_info["fps"],
        "splits": v1_info["splits"],
        "data_path": v1_info["data_path"],
        "video_path": v1_info["video_path"],
    }

    data_size_mb = 0
    for pf in sorted(dest.glob("data/**/*.parquet")):
        data_size_mb += pf.stat().st_size / (1024 * 1024)
    v2_info["data_files_size_in_mb"] = round(data_size_mb)
    v2_info["video_files_size_in_mb"] = v1_info.get("video_files_size_in_mb", 0)

    features = {}
    for img_key in ["observation.images.image", "observation.images.image2"]:
        features[img_key] = v1_info["features"][img_key]

    features["observation.state"] = {
        "dtype": "float32",
        "shape": [STATE_DIM],
        "names": list(STATE_NAMES),
    }
    features["observation.state.joint_position"] = v1_info["features"]["observation.state.joint_position"]
    features["observation.state.fingers"] = {
        "dtype": "float32",
        "shape": [2],
        "names": list(FINGER_NAMES),
    }
    features["observation.keypoint_3d"] = v1_info["features"]["observation.keypoint_3d"]
    features["action"] = v1_info["features"]["action"]

    for scalar_key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        features[scalar_key] = v1_info["features"][scalar_key]

    v2_info["features"] = features

    info_path = dest / "meta" / "info.json"
    with open(info_path, "w") as f:
        json.dump(v2_info, f, indent=2)
    print(f"  info.json written: robot_type={ROBOT_TYPE}, state_dim={STATE_DIM}")


def write_contract_v2(dest: Path, v1_contract_path: Path) -> None:
    with open(v1_contract_path) as f:
        v1_contract = json.load(f)

    v2_contract = {
        "schema": CONTRACT_SCHEMA,
        "robot_type": ROBOT_TYPE,
        "stats_key": STATS_KEY,
        "parent_dataset": str(v1_contract_path.parent.parent),
        "state_layout": "joint7_eef6_gripper1",
        "state_dim": STATE_DIM,
        "state_names": list(STATE_NAMES),
        "gripper_opening": "finger_l - finger_r",
        "gripper_unit": "meter",
        "action_dim": 7,
        "action_semantics": "eef_delta6_plus_gripper_command",
        "dataset_action_mode": "abs",
        "prompt_control_mode": "end_effector",
        "retained_columns": ["observation.state.joint_position", "observation.state.fingers"],
        "videos_are_independent_copies": True,
        "image_orientation": v1_contract.get("image_orientation", "raw"),
        "cameras": v1_contract["cameras"],
        "resize_hw": v1_contract["resize_hw"],
        "fps": v1_contract["fps"],
        "kpt_4d_mode": v1_contract["kpt_4d_mode"],
        "num_keypoints": v1_contract["num_keypoints"],
        "keypoint_history_max_len": v1_contract["keypoint_history_max_len"],
        "chunk_size": v1_contract["chunk_size"],
        "r_pad": v1_contract["r_pad"],
        "base_xpos_m": v1_contract["base_xpos_m"],
        "coordinate_system": v1_contract.get("coordinate_system", "libero_goal_table_world"),
        "eval_mjcf": v1_contract["eval_mjcf"],
        "eval_mjcf_md5": v1_contract["eval_mjcf_md5"],
        "kin_mjcf_md5": v1_contract.get("kin_mjcf_md5"),
        "live_eef_tol_m": v1_contract["live_eef_tol_m"],
        "has_video": v1_contract.get("has_video", True),
        "wait_steps_committed_to_history": v1_contract.get("wait_steps_committed_to_history", False),
        "history_includes_current_frame": v1_contract.get("history_includes_current_frame", False),
    }

    contract_path = dest / "meta" / "goal_train_eval_contract.json"
    with open(contract_path, "w") as f:
        json.dump(v2_contract, f, indent=2)
    print(f"  contract written: schema={CONTRACT_SCHEMA}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Repack Goal v1 → v2 (14D state)")
    parser.add_argument("--src", type=Path, required=True, help="v1 dataset root")
    parser.add_argument("--dest", type=Path, required=True, help="v2 output root")
    parser.add_argument("--confirm-full-run", action="store_true",
                        help="Required for full 512K frame run")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Only repack first N episodes (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing dest (only if robot_type is v2)")
    args = parser.parse_args()

    if args.confirm_full_run and args.max_episodes is not None:
        print("ERROR: --confirm-full-run and --max-episodes are mutually exclusive", file=sys.stderr)
        return 1

    if args.max_episodes is not None and "smoke" not in str(args.dest):
        print("ERROR: --max-episodes requires 'smoke' in dest path", file=sys.stderr)
        return 1

    created_dest = False
    try:
        print(f"[1/8] Loading v1 info from {args.src}")
        v1_info = load_v1_info(args.src)
        expected_frames = v1_info["total_frames"]

        if args.max_episodes is None and not args.confirm_full_run:
            print("ERROR: full run requires --confirm-full-run", file=sys.stderr)
            return 1

        if args.dest.exists():
            if not args.force:
                print(f"ERROR: dest exists: {args.dest}. Use --force to overwrite.", file=sys.stderr)
                return 2
            dest_info_path = args.dest / "meta" / "info.json"
            if dest_info_path.exists():
                with open(dest_info_path) as f:
                    dest_info = json.load(f)
                if dest_info.get("robot_type") != ROBOT_TYPE:
                    print(
                        f"ERROR: dest robot_type is {dest_info.get('robot_type')!r}, "
                        f"not {ROBOT_TYPE!r}. Refusing --force to prevent deleting v1.",
                        file=sys.stderr,
                    )
                    return 3
            print(f"  Removing existing dest: {args.dest}")
            shutil.rmtree(args.dest)

        args.dest.mkdir(parents=True)
        created_dest = True

        print(f"[2/8] Rewriting parquets...")
        total_frames, total_episodes, antisym_count = rewrite_parquets(
            args.src, args.dest, max_episodes=args.max_episodes
        )
        print(f"  frames={total_frames}, episodes={total_episodes}, antisym_violations={antisym_count}")

        if args.max_episodes is None and total_frames != expected_frames:
            raise RuntimeError(
                f"Frame count mismatch: expected {expected_frames}, got {total_frames}"
            )

        print(f"[3/8] Copying videos (independent copies)...")
        video_count = copy_videos(args.src, args.dest)
        print(f"  {video_count} video files copied")

        print(f"[4/8] Copying meta tables...")
        copy_meta_tables(args.src, args.dest)

        print(f"[5/8] Recomputing vector stats...")
        v1_stats_path = args.src / "meta" / "stats.json"
        is_full = args.max_episodes is None
        recompute_vector_stats(args.dest, v1_stats_path, is_full_run=is_full)

        print(f"[6/8] Writing v2 info.json...")
        write_info_v2(args.dest, v1_info, total_frames, total_episodes)

        print(f"[7/8] Writing v2 contract...")
        v1_contract_path = args.src / "meta" / "goal_train_eval_contract.json"
        write_contract_v2(args.dest, v1_contract_path)

        print(f"[8/8] Verifying episode boundary consistency...")
        v2_parquets = sorted(args.dest.glob("data/**/*.parquet"))
        v2_episodes = set()
        v2_frames = 0
        for pf in v2_parquets:
            t = pq.read_table(pf, columns=["episode_index"])
            for ep in t["episode_index"].to_pylist():
                v2_episodes.add(ep)
            v2_frames += len(t)

        if v2_frames != total_frames:
            raise RuntimeError(f"Verification: frame count {v2_frames} != {total_frames}")
        if len(v2_episodes) != total_episodes:
            raise RuntimeError(f"Verification: episode count {len(v2_episodes)} != {total_episodes}")

        print(f"\nSUCCESS: v2 dataset at {args.dest}")
        print(f"  robot_type: {ROBOT_TYPE}")
        print(f"  state_dim: {STATE_DIM}")
        print(f"  frames: {total_frames}")
        print(f"  episodes: {total_episodes}")
        print(f"  videos: {video_count}")
        print(f"  antisym |qL+qR| > {FINGER_ANTISYM_TOL_M}m: {antisym_count} frames")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        if created_dest and args.dest.exists():
            print(f"  Cleaning up incomplete dest: {args.dest}", file=sys.stderr)
            shutil.rmtree(args.dest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
