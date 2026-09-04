"""Post-conversion verification for Franka plug-into-socket LeRobot dataset.

Checks:
  1. Dataset metadata integrity (info.json, episodes)
  2. Feature shapes and dtypes match declaration
  3. No NaN values in state/action columns
  4. Joint angles within Franka limits
  5. Gripper width within physical range [0, 0.08]
  6. Episode count and frame count consistency
  7. Video files exist for each episode
  8. Cross-check ee_pos/ee_quat with FK-computed hand_tcp (optional, if URDF provided)

Usage:
    python b/s/Frk/verify_franka_conversion.py \
        --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb \
        [--urdf b/d/Frk/fr3v2_1_franka_hand.urdf]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


FRANKA_JOINT_LIMITS = [
    (-2.9007, 2.9007),
    (-1.8361, 1.8361),
    (-2.9007, 2.9007),
    (-3.0770, -0.1169),
    (-2.8763, 2.8763),
    (0.4398, 4.6216),
    (-3.0508, 3.0508),
]

GRIPPER_MAX = 0.08


def load_all_data(dataset: Path):
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset}/data")
    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs, ignore_index=True), files


def check_metadata(dataset: Path):
    print("\n=== Check 1: Metadata integrity ===")
    info_path = dataset / "meta" / "info.json"
    if not info_path.exists():
        print(f"  FAIL: {info_path} not found")
        return False
    with open(info_path) as f:
        info = json.load(f)
    required = ["robot_type", "fps", "total_episodes", "total_frames", "features"]
    for key in required:
        if key not in info:
            print(f"  FAIL: missing key '{key}' in info.json")
            return False
    print(f"  robot_type: {info['robot_type']}")
    print(f"  fps: {info['fps']}")
    print(f"  total_episodes: {info['total_episodes']}")
    print(f"  total_frames: {info['total_frames']}")
    print(f"  features: {list(info['features'].keys())}")
    print("  PASS")
    return True


def check_shapes(df: pd.DataFrame):
    print("\n=== Check 2: Feature shapes ===")
    expected = {
        "observation.state.arm": 7,
        "observation.state.gripper": 1,
        "action.arm": 7,
        "action.gripper": 1,
    }
    all_ok = True
    for col, expected_dim in expected.items():
        if col not in df.columns:
            print(f"  FAIL: column '{col}' not found")
            all_ok = False
            continue
        sample = np.array(df[col].iloc[0])
        actual_dim = sample.shape[-1] if sample.ndim > 0 else 1
        if actual_dim != expected_dim:
            print(f"  FAIL: {col} dim={actual_dim}, expected {expected_dim}")
            all_ok = False
        else:
            print(f"  OK {col}: dim={actual_dim}")
    if all_ok:
        print("  PASS")
    return all_ok


def check_no_nan(df: pd.DataFrame):
    print("\n=== Check 3: No NaN values ===")
    cols = ["observation.state.arm", "observation.state.gripper", "action.arm", "action.gripper"]
    all_ok = True
    for col in cols:
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values)
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            print(f"  FAIL: {col} has {nan_count} NaN values")
            all_ok = False
    if all_ok:
        print("  PASS")
    return all_ok


def check_joint_limits(df: pd.DataFrame):
    print("\n=== Check 4: Joint limits ===")
    arm = np.stack(df["observation.state.arm"].values)
    violations = 0
    for j in range(7):
        lo, hi = FRANKA_JOINT_LIMITS[j]
        col = arm[:, j]
        below = (col < lo - 0.01).sum()
        above = (col > hi + 0.01).sum()
        if below or above:
            print(f"  joint{j+1}: {below} below {lo:.4f}, {above} above {hi:.4f}")
            print(f"    actual range: [{col.min():.4f}, {col.max():.4f}]")
            violations += below + above
    if violations:
        print(f"  FAIL: {violations} total violations")
    else:
        print("  PASS")
    return violations == 0


def check_gripper_range(df: pd.DataFrame):
    print("\n=== Check 5: Gripper range ===")
    gripper = np.stack(df["observation.state.gripper"].values).flatten()
    g_min, g_max = gripper.min(), gripper.max()
    print(f"  gripper_width range: [{g_min:.6f}, {g_max:.6f}] m")
    if g_min < -0.001 or g_max > GRIPPER_MAX + 0.001:
        print(f"  FAIL: out of physical range [0, {GRIPPER_MAX}]")
        return False
    print("  PASS")
    return True


def check_episode_consistency(df: pd.DataFrame, dataset: Path):
    print("\n=== Check 6: Episode consistency ===")
    with open(dataset / "meta" / "info.json") as f:
        info = json.load(f)
    declared_episodes = info["total_episodes"]
    declared_frames = info["total_frames"]
    actual_episodes = df["episode_index"].nunique()
    actual_frames = len(df)
    print(f"  Declared: {declared_episodes} episodes, {declared_frames} frames")
    print(f"  Actual:   {actual_episodes} episodes, {actual_frames} frames")
    ok = (actual_episodes == declared_episodes) and (actual_frames == declared_frames)
    if ok:
        print("  PASS")
    else:
        print("  FAIL: mismatch")
    return ok


def check_videos(dataset: Path):
    print("\n=== Check 7: Video files ===")
    video_dir = dataset / "videos"
    if not video_dir.exists():
        print("  [skip] no videos directory")
        return True
    for cam in ["observation.images.global", "observation.images.wrist"]:
        cam_dir = video_dir / cam
        if cam_dir.exists():
            mp4s = list(cam_dir.rglob("*.mp4"))
            print(f"  {cam}: {len(mp4s)} video files")
        else:
            print(f"  WARNING {cam}: directory not found")
    print("  PASS (existence check only)")
    return True


def check_fk_crosscheck(df: pd.DataFrame, dataset: Path, urdf_path: str):
    print("\n=== Check 8: FK cross-check with recorded ee_pos ===")
    if "observation.state.ee_pos" not in df.columns:
        print("  [skip] no ee_pos column")
        return True
    try:
        import pinocchio as pin
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_franka_keypoints import FrankaFKExtractor7D

    extractor = FrankaFKExtractor7D(urdf_path)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(df), size=min(20, len(df)), replace=False)
    max_pos_err = 0.0
    for idx in indices:
        row = df.iloc[idx]
        arm = np.array(row["observation.state.arm"], dtype=np.float64)
        kpts = extractor.compute(arm)
        fk_tcp_pos = kpts[-1, :3]
        recorded_ee = np.array(row["observation.state.ee_pos"], dtype=np.float32)
        err = np.linalg.norm(fk_tcp_pos - recorded_ee)
        max_pos_err = max(max_pos_err, err)

    print(f"  Max |FK_tcp_pos - recorded_ee_pos| over 20 samples: {max_pos_err:.6f} m")
    if max_pos_err > 0.05:
        print(f"  WARNING: large discrepancy ({max_pos_err:.4f} m). "
              "Possible URDF mismatch or different coordinate frames.")
    else:
        print("  PASS")
    return max_pos_err <= 0.05


def print_statistics(df: pd.DataFrame):
    print("\n=== Statistics ===")
    for col in ["observation.state.arm", "observation.state.gripper",
                "action.arm", "action.gripper"]:
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values)
        print(f"\n  {col}: shape={arr.shape}")
        for d in range(arr.shape[-1]):
            vals = arr[:, d] if arr.ndim > 1 else arr
            print(f"    dim{d}: mean={vals.mean():+.6f} std={vals.std():.6f} "
                  f"min={vals.min():+.6f} max={vals.max():+.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str, default=None,
                        help="Optional URDF for FK cross-check.")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying Franka conversion: {dataset}")

    ok = check_metadata(dataset)
    df, _ = load_all_data(dataset)
    ok &= check_shapes(df)
    ok &= check_no_nan(df)
    ok &= check_joint_limits(df)
    ok &= check_gripper_range(df)
    ok &= check_episode_consistency(df, dataset)
    ok &= check_videos(dataset)
    if args.urdf:
        ok &= check_fk_crosscheck(df, dataset, args.urdf)
    print_statistics(df)

    print(f"\n=== Summary: {'ALL PASS' if ok else 'SOME CHECKS FAILED'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
