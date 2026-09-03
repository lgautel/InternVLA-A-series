"""Read-only pre-flight checks on the R1 Pro `open0630_mj_clean` dataset.

Answers the four questions from `b/d/r1pro_migration_design.md` §10 that do NOT need the
URDF, so they can be settled before the FK pipeline exists:

1. Does the dataset actually look the way the design assumes (fps, frame count, key names
   and per-key widths)?  -- §10 item 6
2. Is the right arm actually used, or is door-opening a single-arm task?  -- §10 item 3
   A near-zero right-arm joint variance means half of the 16 keypoints carry no signal.
3. How much of each episode is the chassis-driving segment?  -- §10 item 1
   Keypoints are base-relative, so GeoPredict contributes nothing during that segment; a
   large share directly limits what the A/B experiment can detect.
4. Are the chassis action dims degenerate (near-zero std) and is the torso really all-zero?
   -- §10 items 4 and 5

The script only reads parquet columns and `meta/info.json`; it never writes to the dataset.

Usage:
    python util_scripts/precheck_r1pro_dataset.py --dataset ~/openpi-datasets/open0630_mj_clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Chassis is considered "moving" when any velocity component exceeds this magnitude.
# Matches the order of magnitude used by openpi0.5's diag_chassis_transition.py.
CHASSIS_MOVING_EPS = 0.01

STATE_KEYS = [
    "observation.state.left_arm",
    "observation.state.right_arm",
    "observation.state.left_gripper",
    "observation.state.right_gripper",
    "observation.state.chassis",
    "observation.state.torso",
]
ACTION_KEYS = [
    "action.left_arm",
    "action.right_arm",
    "action.left_gripper",
    "action.right_gripper",
    "action.chassis.velocities",
    "action.torso",
]


def stack(series: pd.Series) -> np.ndarray:
    """LeRobot stores vector features as per-row lists; make them a dense [N, D] array."""
    return np.stack(series.to_numpy()).astype(np.float64)


def load_columns(dataset: Path, columns: list[str]) -> pd.DataFrame:
    files = sorted(dataset.glob("data/chunk-*/file-*.parquet"))
    if not files:
        files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {dataset}/data")

    available = set(pd.read_parquet(files[0]).columns)
    missing = [c for c in columns if c not in available]
    wanted = [c for c in columns if c in available]
    if missing:
        print(f"[warn] columns absent from the dataset, skipped: {missing}")

    print(f"[info] reading {len(wanted)} columns from {len(files)} parquet files ...")
    frames = [pd.read_parquet(f, columns=wanted) for f in files]
    return pd.concat(frames, ignore_index=True)


def report_info(dataset: Path) -> None:
    info_path = dataset / "meta" / "info.json"
    print("\n=== 1. meta/info.json ===")
    if not info_path.exists():
        print(f"[warn] {info_path} not found")
        return
    info = json.loads(info_path.read_text())
    for key in ("robot_type", "fps", "total_episodes", "total_frames", "codebase_version"):
        print(f"  {key:18s} = {info.get(key)}")

    features = info.get("features", {})
    print("  features (state/action/image widths):")
    for name, spec in sorted(features.items()):
        if name.startswith(("observation.state", "action", "observation.images")):
            print(f"    {name:38s} shape={spec.get('shape')} dtype={spec.get('dtype')}")


def report_arm_usage(df: pd.DataFrame) -> None:
    print("\n=== 2. Is the right arm actually used? ===")
    for side in ("left", "right"):
        key = f"observation.state.{side}_arm"
        if key not in df.columns:
            print(f"  [skip] {key} absent")
            continue
        arr = stack(df[key])
        per_joint_std = arr.std(axis=0)
        print(f"  {side:5s} arm  per-joint std = {np.array2string(per_joint_std, precision=4)}")
        print(f"  {side:5s} arm  mean std = {per_joint_std.mean():.6f}, "
              f"range = {(arr.max(axis=0) - arr.min(axis=0)).max():.4f} rad")
    print("  --> if the right arm's std is ~0, drop to J=8 (left arm only); see risk 11")


def report_chassis_segments(df: pd.DataFrame) -> None:
    print("\n=== 3. Chassis segment share ===")
    key = "action.chassis.velocities"
    if key not in df.columns or "episode_index" not in df.columns:
        print(f"  [skip] need both {key} and episode_index")
        return

    vel = stack(df[key])
    moving = (np.abs(vel) > CHASSIS_MOVING_EPS).any(axis=1)
    episodes = df["episode_index"].to_numpy()

    total = len(moving)
    print(f"  frames total          = {total}")
    print(f"  chassis moving        = {moving.sum()} ({100 * moving.mean():.1f}%)")
    print(f"  chassis stopped       = {(~moving).sum()} ({100 * (~moving).mean():.1f}%)")

    lengths, moving_shares = [], []
    for ep in np.unique(episodes):
        sel = episodes == ep
        lengths.append(sel.sum())
        moving_shares.append(moving[sel].mean())
    lengths = np.asarray(lengths)
    moving_shares = np.asarray(moving_shares)

    print(f"  episodes              = {len(lengths)}")
    print(f"  episode length        = mean {lengths.mean():.0f}, "
          f"min {lengths.min()}, max {lengths.max()}")
    print(f"  per-episode moving %  = mean {100 * moving_shares.mean():.1f}, "
          f"p10 {100 * np.percentile(moving_shares, 10):.1f}, "
          f"p90 {100 * np.percentile(moving_shares, 90):.1f}")
    print("  --> this share is the fraction of training frames where GeoPredict adds nothing;")
    print("      the larger it is, the more §8.2 per-phase attribution matters (risk 4)")


def report_dim_stats(df: pd.DataFrame) -> None:
    print("\n=== 4. Per-dimension mean/std (norm_stats sanity) ===")
    for key in STATE_KEYS + ACTION_KEYS:
        if key not in df.columns:
            continue
        arr = stack(df[key])
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        flag = ""
        if np.all(std < 1e-8):
            flag = "  <-- ALL-CONSTANT (safe to drop)"
        elif np.any(std < 1e-3):
            flag = "  <-- has near-zero std dims, normalization will blow them up"
        print(f"  {key:34s} dim={arr.shape[1]}")
        print(f"      mean = {np.array2string(mean, precision=4, suppress_small=True)}")
        print(f"      std  = {np.array2string(std, precision=4, suppress_small=True)}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="path to the LeRobot dataset root (contains data/ and meta/)")
    args = parser.parse_args()

    dataset = args.dataset.expanduser()
    if not dataset.exists():
        raise SystemExit(f"dataset not found: {dataset}")

    report_info(dataset)
    df = load_columns(dataset, STATE_KEYS + ACTION_KEYS + ["episode_index"])
    report_arm_usage(df)
    report_chassis_segments(df)
    report_dim_stats(df)
    print("\ndone -- no files were modified.")


if __name__ == "__main__":
    main()
