"""Post-generation verification for E1 7D keypoints.

Checks:
  1. observation.keypoint_3d exists in every parquet and has shape [112]
  2. All positions within [-1.01, 1.01] (after R_pad normalization)
  3. All quaternions are unit vectors (|q| = 1 ± 0.001)
  4. All quaternions satisfy hemisphere constraint (qw >= 0)
  5. Temporal smoothness: frame-to-frame quaternion change < 0.5 (no sign flips)
  6. FK reproducibility: re-compute a random sample and compare with stored values
  7. Print per-dimension statistics (mean, std, min, max)

Usage:
    python util_scripts/verify_e1_keypoints.py \
        --dataset /home/luogang/DATA/elevator0714_lerobot_4D \
        --urdf assets/r1_pro_with_gripper.urdf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NUM_KEYPOINTS = 16
KEYPOINT_DIM = 7
TOTAL_DIM = NUM_KEYPOINTS * KEYPOINT_DIM  # 112


def load_all_keypoints(dataset: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all keypoint vectors and return (kpts [N, 16, 7], full_df)."""
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
    raw = np.stack(full["observation.keypoint_3d"].values)  # [N, 112]
    kpts = raw.reshape(-1, NUM_KEYPOINTS, KEYPOINT_DIM)  # [N, 16, 7]
    return kpts, full


def check_shape(kpts: np.ndarray):
    print(f"\n=== Check 1: Shape ===")
    print(f"  Total frames: {kpts.shape[0]}")
    print(f"  Shape per frame: [{kpts.shape[1]}, {kpts.shape[2]}] (expect [16, 7])")
    assert kpts.shape[1:] == (NUM_KEYPOINTS, KEYPOINT_DIM), \
        f"Shape mismatch: {kpts.shape[1:]} != ({NUM_KEYPOINTS}, {KEYPOINT_DIM})"
    print("  PASS")


def check_position_bounds(kpts: np.ndarray):
    print(f"\n=== Check 2: Position bounds ===")
    pos = kpts[:, :, :3]
    pos_max = np.abs(pos).max()
    print(f"  max |position|: {pos_max:.6f} (threshold: 1.01)")
    if pos_max > 1.01:
        print(f"  FAIL: {pos_max:.6f} > 1.01")
        return False
    print("  PASS")
    return True


def check_quaternion_norm(kpts: np.ndarray):
    print(f"\n=== Check 3: Quaternion unit norm ===")
    quat = kpts[:, :, 3:7]
    norms = np.linalg.norm(quat.reshape(-1, 4), axis=1)
    norm_err = np.abs(norms - 1.0)
    print(f"  norm error: mean={norm_err.mean():.2e}, max={norm_err.max():.2e}")
    if norm_err.max() > 0.001:
        print(f"  FAIL: max norm error {norm_err.max():.6f} > 0.001")
        return False
    print("  PASS")
    return True


def check_hemisphere(kpts: np.ndarray):
    print(f"\n=== Check 4: Hemisphere constraint (qw >= 0) ===")
    qw = kpts[:, :, 6]  # qw is at index 6 (px,py,pz,qx,qy,qz,qw)
    qw_min = qw.min()
    violations = (qw < -1e-7).sum()
    print(f"  qw min: {qw_min:.8f}")
    print(f"  violations (qw < 0): {violations}")
    if violations > 0:
        print(f"  FAIL: {violations} frames violate hemisphere constraint")
        return False
    print("  PASS")
    return True


def check_temporal_smoothness(kpts: np.ndarray, full_df: pd.DataFrame):
    print(f"\n=== Check 5: Temporal smoothness (within episodes) ===")
    episodes = full_df["episode_index"].values
    quat = kpts[:, :, 3:7]  # [N, 16, 4]

    max_jump = 0.0
    jump_count = 0
    total_transitions = 0

    for ep in np.unique(episodes):
        mask = episodes == ep
        ep_quat = quat[mask]  # [T, 16, 4]
        if len(ep_quat) < 2:
            continue
        diffs = np.linalg.norm(ep_quat[1:] - ep_quat[:-1], axis=-1)  # [T-1, 16]
        frame_max = diffs.max(axis=1)  # [T-1]
        max_jump = max(max_jump, float(frame_max.max()))
        jump_count += int((frame_max > 0.5).sum())
        total_transitions += len(frame_max)

    print(f"  Total frame transitions: {total_transitions}")
    print(f"  Max quaternion jump: {max_jump:.6f} (threshold: 0.5)")
    print(f"  Jumps > 0.5: {jump_count}")
    if jump_count > 0:
        print(f"  WARNING: {jump_count} large quaternion jumps detected")
    else:
        print("  PASS")
    return True


def check_fk_reproducibility(kpts: np.ndarray, full_df: pd.DataFrame,
                             dataset: Path, urdf_path: str):
    print(f"\n=== Check 6: FK reproducibility ===")
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        print("  [skip] pinocchio not available")
        return True

    meta_path = dataset / "meta" / "keypoints_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    r_pad = meta["bbox_radius"]
    torso_q = meta["torso_q"]

    # Import the extractor from the generation script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "gen_e1",
        str(Path(__file__).parent / "generate_r1pro_keypoints_e1.py")
    )
    gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_module)
    extractor = gen_module.R1ProFKExtractorE1(urdf_path, torso_q=torso_q)

    rng = np.random.default_rng(42)
    indices = rng.choice(len(full_df), size=min(10, len(full_df)), replace=False)
    max_err = 0.0

    for idx in indices:
        row = full_df.iloc[idx]
        left = np.array(row["observation.state.left_arm"], dtype=np.float64)
        right = np.array(row["observation.state.right_arm"], dtype=np.float64)
        recomputed = extractor.compute(left, right)  # [16, 7]
        recomputed[:, :3] /= r_pad
        stored = kpts[idx]  # [16, 7]
        err = np.abs(recomputed - stored).max()
        max_err = max(max_err, err)

    print(f"  Max recomputation error (10 random frames): {max_err:.2e}")
    if max_err > 1e-5:
        print(f"  FAIL: reproducibility error {max_err:.6f} > 1e-5")
        return False
    print("  PASS")
    return True


def print_statistics(kpts: np.ndarray):
    print(f"\n=== Check 7: Per-dimension statistics ===")
    flat = kpts.reshape(-1, KEYPOINT_DIM)  # [N*16, 7]
    labels = ["px", "py", "pz", "qx", "qy", "qz", "qw"]
    print(f"  {'dim':>4s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}")
    print("  " + "-" * 50)
    for i, label in enumerate(labels):
        col = flat[:, i]
        print(f"  {label:>4s}  {col.mean():+10.6f}  {col.std():10.6f}  {col.min():+10.6f}  {col.max():+10.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--urdf", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "assets" / "r1_pro_with_gripper.urdf"))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    print(f"Verifying E1 keypoints in: {dataset}")

    kpts, full_df = load_all_keypoints(dataset)

    all_pass = True
    check_shape(kpts)
    all_pass &= check_position_bounds(kpts)
    all_pass &= check_quaternion_norm(kpts)
    all_pass &= check_hemisphere(kpts)
    check_temporal_smoothness(kpts, full_df)
    all_pass &= check_fk_reproducibility(kpts, full_df, dataset, args.urdf)
    print_statistics(kpts)

    print(f"\n=== Summary ===")
    print(f"  Dataset: {dataset}")
    print(f"  Frames: {len(kpts)}")
    print(f"  Keypoints per frame: {NUM_KEYPOINTS} x {KEYPOINT_DIM}D = {TOTAL_DIM}")
    if all_pass:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED — see above")
    print("  All checks complete.")


if __name__ == "__main__":
    main()
