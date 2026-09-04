"""Post-generation verification for Franka 7D keypoints.

Same 7-check framework as verify_e1_keypoints.py, adapted for Franka
single-arm (8 keypoints x 7D = 56-dim).

Checks:
  1. Shape: observation.keypoint_3d exists and has [56] per frame
  2. Position bounds: |pos| <= 1.01 after R_pad normalization
  3. Quaternion unit norm: |norm(q)-1| <= 0.001
  4. Hemisphere constraint: qw >= 0
  5. Temporal smoothness: frame-to-frame quaternion change < 0.5
  6. FK reproducibility: recompute random samples, compare with stored
  7. Per-dimension statistics

Usage:
    python b/s/Frk/verify_franka_keypoints.py \
        --dataset /home/luogang/hf_home/lerobot/plug_into_socket_lrb_4D \
        --urdf b/d/Frk/fr3v2_1_franka_hand.urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NUM_KEYPOINTS = 8
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM


def load_all_keypoints(dataset: Path):
    files = sorted(dataset.glob("data/**/*.parquet"))
    if not files:
        sys.exit(f"No parquet files under {dataset}/data")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        if "observation.keypoint_3d" not in df.columns:
            sys.exit(f"observation.keypoint_3d not found in {f}")
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    raw = np.stack(full["observation.keypoint_3d"].values)
    kpts = raw.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)
    return kpts, full


def check_shape(kpts):
    print(f"\n=== Check 1: Shape ===")
    print(f"  Total frames: {kpts.shape[0]}")
    print(f"  Shape per frame: [{kpts.shape[1]}, {kpts.shape[2]}] (expect [{NUM_KEYPOINTS}, {KEYPOINT_DIM}])")
    assert kpts.shape[1:] == (NUM_KEYPOINTS, KEYPOINT_DIM)
    print("  PASS")


def check_position_bounds(kpts):
    print(f"\n=== Check 2: Position bounds ===")
    pos = kpts[:, :, :3]
    pos_max = np.abs(pos).max()
    print(f"  max |position|: {pos_max:.6f} (threshold: 1.01)")
    ok = pos_max <= 1.01
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check_quaternion_norm(kpts):
    print(f"\n=== Check 3: Quaternion unit norm ===")
    quat = kpts[:, :, 3:7]
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    norm_err = np.abs(norms - 1.0)
    print(f"  norm error: mean={norm_err.mean():.2e}, max={norm_err.max():.2e}")
    ok = norm_err.max() <= 0.001
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check_hemisphere(kpts):
    print(f"\n=== Check 4: Hemisphere constraint (qw >= 0) ===")
    qw = kpts[:, :, 6]
    qw_min = qw.min()
    violations = (qw < -1e-7).sum()
    print(f"  qw min: {qw_min:.8f}, violations: {violations}")
    ok = violations == 0
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def check_temporal_smoothness(kpts, full_df):
    print(f"\n=== Check 5: Temporal smoothness ===")
    episodes = full_df["episode_index"].values
    quat = kpts[:, :, 3:7]
    max_jump = 0.0
    jump_count = 0
    total_transitions = 0
    for ep in np.unique(episodes):
        mask = episodes == ep
        ep_quat = quat[mask]
        if len(ep_quat) < 2:
            continue
        diffs = np.linalg.norm(ep_quat[1:] - ep_quat[:-1], axis=-1)
        frame_max = diffs.max(axis=1)
        max_jump = max(max_jump, float(frame_max.max()))
        jump_count += int((frame_max > 0.5).sum())
        total_transitions += len(frame_max)
    print(f"  Transitions: {total_transitions}, max jump: {max_jump:.6f}, jumps>0.5: {jump_count}")
    if jump_count > 0:
        print(f"  WARNING: {jump_count} large jumps")
    else:
        print("  PASS")
    return jump_count == 0


def check_fk_reproducibility(kpts, full_df, dataset, urdf_path):
    print(f"\n=== Check 6: FK reproducibility ===")
    try:
        import pinocchio as pin
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    meta_path = dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    r_pad = meta["bbox_radius"]

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from generate_franka_keypoints import FrankaFKExtractor7D

    extractor = FrankaFKExtractor7D(urdf_path)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(full_df), size=min(10, len(full_df)), replace=False)
    max_err = 0.0
    for idx in indices:
        row = full_df.iloc[idx]
        arm = np.array(row["observation.state.arm"], dtype=np.float64)
        recomputed = extractor.compute(arm)
        recomputed[:, :3] /= r_pad
        stored = kpts[idx]
        err = np.abs(recomputed - stored).max()
        max_err = max(max_err, err)

    print(f"  Max recomputation error (10 random frames): {max_err:.2e}")
    ok = max_err <= 1e-5
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def print_statistics(kpts):
    print(f"\n=== Check 7: Per-dimension statistics ===")
    flat = kpts.reshape(-1, KEYPOINT_DIM)
    labels = ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    print(f"  {'dim':>4s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print("  " + "-" * 50)
    for i, label in enumerate(labels):
        col = flat[:, i]
        print(f"  {label:>4s}  {col.mean():+10.6f}  {col.std():10.6f}  "
              f"{col.min():+10.6f}  {col.max():+10.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[3] / "b" / "d" / "Frk" / "fr3v2_1_franka_hand.urdf"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying Franka 7D keypoints in: {dataset}")

    kpts, full_df = load_all_keypoints(dataset)
    check_shape(kpts)
    ok = check_position_bounds(kpts)
    ok &= check_quaternion_norm(kpts)
    ok &= check_hemisphere(kpts)
    ok &= check_temporal_smoothness(kpts, full_df)
    ok &= check_fk_reproducibility(kpts, full_df, dataset, args.urdf)
    print_statistics(kpts)

    print(f"\n=== Summary: {'ALL PASS' if ok else 'SOME CHECKS FAILED'} ===")
    print(f"  Frames: {len(kpts)}, Keypoints: {NUM_KEYPOINTS} x {KEYPOINT_DIM}D = {TOTAL_DIM}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
